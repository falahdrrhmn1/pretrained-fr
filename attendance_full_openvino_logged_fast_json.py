from pathlib import Path
import argparse
import json
import sys
import time
import threading
import queue
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import openvino as ov


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

# ----------------------------
# Stage 1: OpenVINO 1:1 FR
# ----------------------------

FR_MODEL_DIR = (
    ROOT
    / "models"
    / "commercial_test"
    / "openvino_fr"
)

FD_XML = (
    FR_MODEL_DIR
    / "face-detection-retail-0004"
    / "FP16"
    / "face-detection-retail-0004.xml"
)

LM_XML = (
    FR_MODEL_DIR
    / "landmarks-regression-retail-0009"
    / "FP16"
    / "landmarks-regression-retail-0009.xml"
)

REID_XML = (
    FR_MODEL_DIR
    / "face-reidentification-retail-0095"
    / "FP16"
    / "face-reidentification-retail-0095.xml"
)

FACE_DET_THRESHOLD = 0.60
BBOX_EXPAND_RATIO = 1.15

# Open Model Zoo face-recognition demo default threshold
# distance = (1 - cosine_similarity) * 0.5
FR_DISTANCE_THRESHOLD = 0.30

FR_SMOOTHING_WINDOW = 10
FR_MIN_SAMPLES = 5


# ----------------------------
# Stage 2: CVPR2024 Swin-V2
# OpenVINO inference-only IR
# ----------------------------

FAS_OV_DIR = (
    ROOT
    / "models"
    / "commercial_test"
    / "cvpr2024_fas"
    / "openvino"
)

FAS_OV_XML = (
    FAS_OV_DIR
    / "face_swin_v2_base_fp32.xml"
)

FAS_OV_BIN = (
    FAS_OV_DIR
    / "face_swin_v2_base_fp32.bin"
)

FAS_INPUT_SIZE = 224
FAS_LIVE_THRESHOLD = 0.50

# Keep the same two-sample temporal evidence as the previous version.
# Speed is improved by changing the runtime, not by reducing evidence.
FAS_SMOOTHING_WINDOW = 3
FAS_MIN_SAMPLES = 2

# Submit a fresh crop quickly when the worker becomes available.
FAS_SUBMIT_INTERVAL = 0.05
FAS_WARMUP_RUNS = 3

# <1 second is the performance target. This value does NOT force a
# decision; it is displayed for benchmarking. The hard timeout is 2 s.
TARGET_DECISION_SECONDS = 1.0
MAX_DECISION_SECONDS = 2.0

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


# ----------------------------
# Camera / UI
# ----------------------------

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_NAME = "ATTENDANCE - FULL OPENVINO + AUDIT LOG"

VIDEO_W = 800
VIDEO_H = 600

PANEL_W = 430
STATUS_H = 75

CANVAS_W = VIDEO_W + PANEL_W
CANVAS_H = VIDEO_H + STATUS_H

BUTTON_X1 = VIDEO_W + 45
BUTTON_Y1 = 475
BUTTON_X2 = CANVAS_W - 45
BUTTON_Y2 = 545


# ============================================================
# ARGUMENT
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Attendance candidate stack: OpenVINO 1:1 verification "
        "+ CVPR2024 Swin-V2 liveness."
    )
)

parser.add_argument(
    "--reference",
    required=True,
    help="Path foto reference, contoh: .\\reference_images\MY_FACE.jpg",
)

parser.add_argument(
    "--camera",
    type=int,
    default=CAMERA_INDEX,
)

parser.add_argument(
    "--employee-id",
    default=None,
    help=(
        "Claimed employee ID written to the audit log. "
        "Default: reference filename without extension."
    ),
)

parser.add_argument(
    "--location",
    default="UNKNOWN",
    help=(
        'Logical attendance location for the audit log, e.g. "Office A".'
    ),
)

parser.add_argument(
    "--fas-device",
    default="CPU",
    help=(
        "OpenVINO device for Swin-V2 liveness. "
        "Examples: CPU, GPU, AUTO. Default: CPU for stable low-latency sessions."
    ),
)

args = parser.parse_args()

REFERENCE_PATH = Path(args.reference)
EMPLOYEE_ID = args.employee_id or REFERENCE_PATH.stem
ATTENDANCE_LOCATION = args.location
FAS_DEVICE_NAME = args.fas_device.upper()

if not REFERENCE_PATH.exists():
    raise FileNotFoundError(
        f"Reference image tidak ditemukan: {REFERENCE_PATH}"
    )


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    FD_XML,
    LM_XML,
    REID_XML,
    FAS_OV_XML,
    FAS_OV_BIN,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"File/model belum ditemukan:\n{path}"
        )


# ============================================================
# LOCAL AUDIT LOG — JSON
# ============================================================

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_daily_log_path():
    """One valid JSON document per day."""
    return (
        LOG_DIR
        / f"attendance_{datetime.now().strftime('%Y-%m-%d')}.json"
    )


def security_flag_for(result):
    if result == "VERIFIED":
        return "NORMAL"
    if result == "SPOOF ATTACK":
        return "SPOOF_ATTEMPT"
    if result == "WRONG PERSON":
        return "IDENTITY_MISMATCH"
    if result == "WRONG PERSON + SPOOF":
        return "IDENTITY_AND_SPOOF"
    if result == "RETRY":
        return "TIMEOUT_RETRY"
    if result == "ABORTED_BY_USER":
        return "ABORTED_ATTEMPT"
    return "UNKNOWN"


def _optional_round(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def _load_daily_json(log_path):
    """Load the existing daily JSON file, or return a fresh document."""
    today = datetime.now().astimezone().date().isoformat()

    fresh = {
        "schema_version": 1,
        "date": today,
        "events": [],
    }

    if not log_path.exists() or log_path.stat().st_size == 0:
        return fresh

    try:
        with log_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError("Root JSON harus berupa object.")

        if not isinstance(data.get("events"), list):
            raise ValueError("Field 'events' harus berupa array.")

        return data

    except (json.JSONDecodeError, OSError, ValueError) as exc:
        # Do not silently destroy a damaged audit log.
        backup_path = log_path.with_name(
            f"{log_path.stem}_corrupt_"
            f"{datetime.now().strftime('%H%M%S')}"
            f"{log_path.suffix}"
        )

        try:
            log_path.replace(backup_path)
        except OSError:
            pass

        print(
            f"[AUDIT WARNING] Log JSON lama tidak dapat dibaca: {exc}"
        )
        print(
            f"[AUDIT WARNING] Membuat log baru. "
            f"Backup: {backup_path.name}"
        )

        return fresh


def _write_json_atomic(log_path, data):
    """Write through a temporary file, then atomically replace the JSON."""
    temp_path = log_path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
        handle.flush()

    temp_path.replace(log_path)


def append_audit_log(
    result,
    identity_state,
    liveness_state,
    decision_seconds,
    session_number,
):
    """Append one attendance/security event to the daily JSON document.

    Only metadata and model scores are stored.
    Camera frames and face images are NOT stored by this logger.
    """

    log_path = get_daily_log_path()

    event = {
        "timestamp_local": datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        ),
        "session_id": int(session_number),

        "employee": {
            "id": EMPLOYEE_ID,
            "reference_file": REFERENCE_PATH.name,
        },

        "context": {
            "location": ATTENDANCE_LOCATION,
            "camera_index": int(args.camera),
        },

        "decision": {
            "final_result": result,
            "security_flag": security_flag_for(result),
            "decision_time_sec": _optional_round(
                decision_seconds,
                3,
            ),
            "fr_ready_sec": _optional_round(
                fr_ready_time_sec,
                3,
            ),
            "fas_ready_sec": _optional_round(
                fas_ready_time_sec,
                3,
            ),
        },

        "face_verification": {
            "state": identity_state,
            "cosine_median": round(
                float(median_fr_similarity),
                6,
            ),
            "distance_median": round(
                float(median_fr_distance),
                6,
            ),
            "distance_threshold": round(
                float(FR_DISTANCE_THRESHOLD),
                6,
            ),
        },

        "liveness": {
            "state": liveness_state,
            "live_median": round(
                float(median_fas_live),
                6,
            ),
            "live_threshold": round(
                float(FAS_LIVE_THRESHOLD),
                6,
            ),
            "raw_state": raw_fas_state,
            "raw_live": round(
                float(raw_fas_live),
                6,
            ),
            "raw_spoof": round(
                float(raw_fas_spoof),
                6,
            ),
            "swin_inference_ms": round(
                float(raw_fas_infer_ms),
                3,
            ),
            "swin_inference_ms_avg": (
                round(
                    float(
                        np.mean(
                            list(fas_infer_ms_history)
                        )
                    ),
                    3,
                )
                if fas_infer_ms_history
                else None
            ),
            "device": FAS_DEVICE_NAME,
        },
    }

    document = _load_daily_json(log_path)
    document["events"].append(event)
    document["event_count"] = len(document["events"])
    document["last_updated"] = event["timestamp_local"]

    _write_json_atomic(
        log_path,
        document,
    )

    print(
        f"[AUDIT] {event['timestamp_local']} | "
        f"{EMPLOYEE_ID} | {result} | "
        f"FR={identity_state} | FAS={liveness_state} | "
        f"{event['decision']['decision_time_sec']}s "
        f"-> {log_path.name}"
    )


# ============================================================
# COLORS
# ============================================================

BLACK = (0, 0, 0)
DARK = (25, 25, 25)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)

GREEN = (0, 235, 0)
RED = (0, 0, 255)
YELLOW = (0, 230, 255)
CYAN = (255, 255, 0)


def draw_text(
    image,
    text,
    position,
    color=WHITE,
    scale=0.60,
    thickness=2,
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def state_color(state):
    if state in ("MATCH", "LIVE", "VERIFIED"):
        return GREEN

    if state in (
        "NOT MATCH",
        "FAKE",
        "SPOOF ATTACK",
        "WRONG PERSON",
        "WRONG PERSON + SPOOF",
    ):
        return RED

    return YELLOW


# ============================================================
# STAGE 1 — OPENVINO FACE VERIFICATION
# ============================================================

print("=" * 78)
print("LOADING STAGE 1 - OPENVINO 1:1 FACE VERIFICATION")
print("=" * 78)

ov_core = ov.Core()

fd_model = ov_core.read_model(str(FD_XML))
lm_model = ov_core.read_model(str(LM_XML))
reid_model = ov_core.read_model(str(REID_XML))

fd = ov_core.compile_model(fd_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
lm = ov_core.compile_model(lm_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
reid = ov_core.compile_model(reid_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})

fd_output = fd.output(0)
lm_output = lm.output(0)
reid_output = reid.output(0)

print("Face detector :", FD_XML.name)
print("Landmarks     :", LM_XML.name)
print("Face verifier :", REID_XML.name)
print("FR threshold  :", FR_DISTANCE_THRESHOLD)


REFERENCE_LANDMARKS = np.array(
    [
        (30.2946 / 96, 51.6963 / 112),
        (65.5318 / 96, 51.5014 / 112),
        (48.0252 / 96, 71.7366 / 112),
        (33.5493 / 96, 92.3655 / 112),
        (62.7299 / 96, 92.2041 / 112),
    ],
    dtype=np.float32,
)


def to_nchw_bgr(image, width, height):
    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    blob = resized.astype(np.float32)
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    return np.ascontiguousarray(
        blob,
        dtype=np.float32,
    )


def expand_bbox(
    bbox,
    frame_shape,
    ratio=BBOX_EXPAND_RATIO,
):
    h, w = frame_shape[:2]

    x1, y1, x2, y2 = map(float, bbox)

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw *= ratio
    bh *= ratio

    nx1 = max(
        0,
        int(round(cx - bw / 2.0)),
    )

    ny1 = max(
        0,
        int(round(cy - bh / 2.0)),
    )

    nx2 = min(
        w,
        int(round(cx + bw / 2.0)),
    )

    ny2 = min(
        h,
        int(round(cy + bh / 2.0)),
    )

    return nx1, ny1, nx2, ny2


def detect_faces(frame):
    h, w = frame.shape[:2]

    blob = to_nchw_bgr(
        frame,
        300,
        300,
    )

    result = fd(
        [blob]
    )[fd_output]

    detections = np.asarray(
        result
    ).reshape(-1, 7)

    boxes = []

    for det in detections:
        if det[0] < 0:
            continue

        confidence = float(det[2])

        if confidence < FACE_DET_THRESHOLD:
            continue

        x1 = int(
            np.clip(
                det[3] * w,
                0,
                w - 1,
            )
        )

        y1 = int(
            np.clip(
                det[4] * h,
                0,
                h - 1,
            )
        )

        x2 = int(
            np.clip(
                det[5] * w,
                x1 + 1,
                w,
            )
        )

        y2 = int(
            np.clip(
                det[6] * h,
                y1 + 1,
                h,
            )
        )

        boxes.append(
            (
                x1,
                y1,
                x2,
                y2,
                confidence,
            )
        )

    return boxes


def get_landmarks(
    frame,
    bbox,
):
    x1, y1, x2, y2 = expand_bbox(
        bbox,
        frame.shape,
    )

    face_crop = frame[
        y1:y2,
        x1:x2,
    ]

    if face_crop.size == 0:
        return None

    blob = to_nchw_bgr(
        face_crop,
        48,
        48,
    )

    output = lm(
        [blob]
    )[lm_output]

    points = np.asarray(
        output,
        dtype=np.float32,
    ).reshape(-1)

    if points.size < 10:
        return None

    landmarks = points[:10].reshape(
        5,
        2,
    )

    return (
        face_crop,
        landmarks,
        (x1, y1, x2, y2),
    )


def normalize_points(
    array,
    axis,
):
    arr = array.astype(
        np.float64
    ).copy()

    mean = arr.mean(
        axis=axis
    )

    arr -= mean

    std = arr.std()

    if std < 1e-12:
        raise ValueError(
            "Landmark geometry degenerate."
        )

    arr /= std

    return (
        arr,
        mean,
        std,
    )


def get_transform(
    src,
    dst,
):
    src_n, src_mean, src_std = normalize_points(
        src,
        axis=0,
    )

    dst_n, dst_mean, dst_std = normalize_points(
        dst,
        axis=0,
    )

    u, _, vt = np.linalg.svd(
        np.matmul(
            src_n.T,
            dst_n,
        )
    )

    r = np.matmul(
        u,
        vt,
    ).T

    transform = np.empty(
        (2, 3),
        dtype=np.float64,
    )

    transform[:, 0:2] = (
        r
        * (
            dst_std
            / src_std
        )
    )

    transform[:, 2] = (
        dst_mean.T
        - np.matmul(
            transform[:, 0:2],
            src_mean.T,
        )
    )

    return transform


def align_face(
    face_crop,
    landmarks,
):
    h, w = face_crop.shape[:2]

    scale = np.array(
        (w, h),
        dtype=np.float32,
    )

    desired = (
        REFERENCE_LANDMARKS
        * scale
    )

    actual = (
        landmarks
        * scale
    )

    transform = get_transform(
        desired,
        actual,
    )

    aligned = cv2.warpAffine(
        face_crop,
        transform,
        (w, h),
        flags=cv2.WARP_INVERSE_MAP,
    )

    return aligned


def get_embedding(
    frame,
    bbox,
):
    lm_result = get_landmarks(
        frame,
        bbox,
    )

    if lm_result is None:
        return None

    (
        face_crop,
        landmarks,
        expanded_bbox,
    ) = lm_result

    aligned = align_face(
        face_crop,
        landmarks,
    )

    blob = to_nchw_bgr(
        aligned,
        128,
        128,
    )

    output = reid(
        [blob]
    )[reid_output]

    embedding = np.asarray(
        output,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(
        embedding
    )

    if norm < 1e-12:
        return None

    embedding /= norm

    return embedding


def compare_embeddings(
    emb_a,
    emb_b,
):
    cosine_similarity = float(
        np.dot(
            emb_a,
            emb_b,
        )
    )

    cosine_similarity = float(
        np.clip(
            cosine_similarity,
            -1.0,
            1.0,
        )
    )

    distance = (
        1.0
        - cosine_similarity
    ) * 0.5

    return (
        cosine_similarity,
        distance,
    )


# ============================================================
# REFERENCE EMBEDDING
# ============================================================

print()
print("=" * 78)
print("BUILDING REFERENCE EMBEDDING")
print("=" * 78)

reference_image = cv2.imread(
    str(REFERENCE_PATH)
)

if reference_image is None:
    raise RuntimeError(
        "Reference image gagal dibaca."
    )

reference_faces = detect_faces(
    reference_image
)

if len(reference_faces) == 0:
    raise RuntimeError(
        "Tidak ada wajah pada reference image."
    )

if len(reference_faces) > 1:
    raise RuntimeError(
        "Reference image harus berisi tepat satu wajah."
    )

reference_bbox = reference_faces[0][:4]

reference_embedding = get_embedding(
    reference_image,
    reference_bbox,
)

if reference_embedding is None:
    raise RuntimeError(
        "Gagal membuat reference embedding."
    )

print("Reference :", REFERENCE_PATH)
print("Embedding :", reference_embedding.shape)
print("Reference face OK.")


# ============================================================
# STAGE 2 — CVPR2024 SWIN-V2 VIA OPENVINO
# ============================================================

print()
print("=" * 78)
print("LOADING STAGE 2 - CVPR2024 SWIN-V2 VIA OPENVINO")
print("=" * 78)
print("Available OpenVINO devices:", ov_core.available_devices)
print("Requested FAS device       :", FAS_DEVICE_NAME)

fas_ov_model = ov_core.read_model(str(FAS_OV_XML))

fas = ov_core.compile_model(
    fas_ov_model,
    FAS_DEVICE_NAME,
    {"PERFORMANCE_HINT": "LATENCY"},
)

fas_output = fas.output(0)

print("FAS IR model               :", FAS_OV_XML.name)
print("FAS target                 :", FAS_DEVICE_NAME)
print("Performance hint           : LATENCY")
print("Swin-V2 OpenVINO liveness loaded.")


# ============================================================
# FAS HELPERS
# ============================================================

def crop_for_liveness(
    frame,
    bbox,
    padding_ratio=0.08,
):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(float, bbox)

    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop.copy()


def preprocess_fas(face_crop):
    image = cv2.resize(
        face_crop,
        (FAS_INPUT_SIZE, FAS_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image, dtype=np.float32)


def softmax_2(logits):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.size < 2:
        raise RuntimeError(f"Unexpected FAS output shape: {logits.shape}")

    logits = logits[:2]
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)
    return probs


def run_fas(face_crop):
    blob = preprocess_fas(face_crop)
    output = fas([blob])[fas_output]
    probs = softmax_2(output)

    p_live = float(probs[0])
    p_spoof = float(probs[1])

    state = "LIVE" if p_live >= FAS_LIVE_THRESHOLD else "FAKE"

    return state, p_live, p_spoof


# ============================================================
# FAS WARM-UP
# ============================================================

# Pay one-time OpenVINO/device initialization cost BEFORE the camera session
# starts. This does not change the model, threshold, input size, or precision.
_warmup_crop = np.zeros(
    (FAS_INPUT_SIZE, FAS_INPUT_SIZE, 3),
    dtype=np.uint8,
)
_warmup_times = []

print(f"Warming up Swin-V2 on {FAS_DEVICE_NAME} ({FAS_WARMUP_RUNS} runs)...")
for _ in range(FAS_WARMUP_RUNS):
    _t0 = time.perf_counter()
    run_fas(_warmup_crop)
    _warmup_times.append((time.perf_counter() - _t0) * 1000.0)

print(
    "FAS warm-up times          :",
    ", ".join(f"{x:.1f} ms" for x in _warmup_times),
)
print(
    "FAS warm-up median         :",
    f"{float(np.median(_warmup_times)):.1f} ms",
)
del _warmup_crop


# ============================================================
# ASYNC LIVENESS WORKER
# ============================================================

fas_queue = queue.Queue(
    maxsize=1
)

fas_lock = threading.Lock()

fas_worker_state = {
    "session_id": -1,
    "sample_id": 0,
    "state": "WAITING",
    "p_live": 0.0,
    "p_spoof": 0.0,
    "infer_ms": 0.0,
    "busy": False,
}

fas_stop_event = threading.Event()


def fas_worker():
    while not fas_stop_event.is_set():

        try:
            item = fas_queue.get(
                timeout=0.1
            )
        except queue.Empty:
            continue

        if item is None:
            break

        (
            item_session_id,
            crop,
        ) = item

        with fas_lock:
            fas_worker_state[
                "busy"
            ] = True

        infer_start = time.perf_counter()

        try:
            (
                pred_state,
                p_live,
                p_spoof,
            ) = run_fas(
                crop
            )

            infer_ms = (
                time.perf_counter()
                - infer_start
            ) * 1000.0

            with fas_lock:
                fas_worker_state[
                    "session_id"
                ] = item_session_id

                fas_worker_state[
                    "sample_id"
                ] += 1

                fas_worker_state[
                    "state"
                ] = pred_state

                fas_worker_state[
                    "p_live"
                ] = p_live

                fas_worker_state[
                    "p_spoof"
                ] = p_spoof

                fas_worker_state[
                    "infer_ms"
                ] = infer_ms

        finally:
            with fas_lock:
                fas_worker_state[
                    "busy"
                ] = False

            fas_queue.task_done()


fas_thread = threading.Thread(
    target=fas_worker,
    daemon=True,
)

fas_thread.start()


def submit_latest_fas_crop(
    session_id,
    crop,
):
    """
    Queue size = 1.
    Kalau masih ada crop lama yang belum diproses,
    crop lama dibuang dan diganti frame terbaru.
    """

    try:
        while True:
            fas_queue.get_nowait()
            fas_queue.task_done()
    except queue.Empty:
        pass

    try:
        fas_queue.put_nowait(
            (
                session_id,
                crop,
            )
        )
    except queue.Full:
        pass


# ============================================================
# DECISION STATE
# ============================================================

fr_similarity_history = deque(
    maxlen=FR_SMOOTHING_WINDOW
)

fr_distance_history = deque(
    maxlen=FR_SMOOTHING_WINDOW
)

fas_live_history = deque(
    maxlen=FAS_SMOOTHING_WINDOW
)

fas_infer_ms_history = deque(
    maxlen=FAS_SMOOTHING_WINDOW
)

session_id = 0

last_consumed_fas_sample = 0
last_fas_submit_time = 0.0

raw_fr_similarity = 0.0
raw_fr_distance = 1.0

median_fr_similarity = 0.0
median_fr_distance = 1.0

fr_state = "CHECKING"

raw_fas_state = "WAITING"
raw_fas_live = 0.0
raw_fas_spoof = 0.0
raw_fas_infer_ms = 0.0

fas_state = "CHECKING"
median_fas_live = 0.0

final_state = None

# Timing for one attendance attempt.
# Starts only when exactly one face is detected.
session_started_at = None
decision_time_sec = None
session_elapsed_sec = 0.0
fr_ready_time_sec = None
fas_ready_time_sec = None

latest_bbox = None

# One audit row per finished session.
current_session_logged = False

retry_requested = False
fullscreen = False


def purge_fas_queue():
    try:
        while True:
            fas_queue.get_nowait()
            fas_queue.task_done()
    except queue.Empty:
        pass


def reset_session(log_user_abort=False):
    global session_id
    global last_consumed_fas_sample
    global last_fas_submit_time

    global raw_fr_similarity
    global raw_fr_distance
    global median_fr_similarity
    global median_fr_distance
    global fr_state

    global raw_fas_state
    global raw_fas_live
    global raw_fas_spoof
    global raw_fas_infer_ms
    global fas_state
    global median_fas_live

    global final_state
    global session_started_at
    global decision_time_sec
    global session_elapsed_sec
    global fr_ready_time_sec
    global fas_ready_time_sec
    global latest_bbox
    global current_session_logged

    # If the user explicitly aborts an in-progress attempt, record it.
    if (
        log_user_abort
        and final_state is None
        and session_started_at is not None
        and not current_session_logged
        and (len(fr_distance_history) > 0 or len(fas_live_history) > 0)
    ):
        append_audit_log(
            "ABORTED_BY_USER",
            fr_state,
            fas_state,
            session_elapsed_sec,
            session_id,
        )
        current_session_logged = True

    session_id += 1

    fr_similarity_history.clear()
    fr_distance_history.clear()
    fas_live_history.clear()
    fas_infer_ms_history.clear()

    purge_fas_queue()

    with fas_lock:
        last_consumed_fas_sample = (
            fas_worker_state[
                "sample_id"
            ]
        )

    last_fas_submit_time = 0.0

    raw_fr_similarity = 0.0
    raw_fr_distance = 1.0

    median_fr_similarity = 0.0
    median_fr_distance = 1.0

    fr_state = "CHECKING"

    raw_fas_state = "WAITING"
    raw_fas_live = 0.0
    raw_fas_spoof = 0.0
    raw_fas_infer_ms = 0.0

    fas_state = "CHECKING"
    median_fas_live = 0.0

    final_state = None

    session_started_at = None
    decision_time_sec = None
    session_elapsed_sec = 0.0
    fr_ready_time_sec = None
    fas_ready_time_sec = None

    latest_bbox = None
    current_session_logged = False


def get_final_state(
    identity_state,
    liveness_state,
):
    if (
        identity_state == "MATCH"
        and liveness_state == "LIVE"
    ):
        return "VERIFIED"

    if (
        identity_state == "MATCH"
        and liveness_state == "FAKE"
    ):
        return "SPOOF ATTACK"

    if (
        identity_state == "NOT MATCH"
        and liveness_state == "LIVE"
    ):
        return "WRONG PERSON"

    if (
        identity_state == "NOT MATCH"
        and liveness_state == "FAKE"
    ):
        return "WRONG PERSON + SPOOF"

    return None


# ============================================================
# RETRY BUTTON
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    global retry_requested

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if (
        BUTTON_X1 <= x <= BUTTON_X2
        and BUTTON_Y1 <= y <= BUTTON_Y2
    ):
        retry_requested = True


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    args.camera,
    cv2.CAP_DSHOW,
)

if not cap.isOpened():
    cap = cv2.VideoCapture(
        args.camera
    )

if not cap.isOpened():
    raise RuntimeError(
        "Webcam tidak dapat dibuka."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT,
)


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL,
)

cv2.resizeWindow(
    WINDOW_NAME,
    CANVAS_W,
    CANVAS_H,
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback,
)


fps_history = deque(
    maxlen=30
)

previous_time = time.perf_counter()


print()
print("=" * 78)
print("FULL ATTENDANCE CANDIDATE STARTED")
print("=" * 78)
print("Stage 1 : OpenVINO 1:1 Face Verification")
print("Stage 2 : CVPR2024 Swin-V2 Liveness (OpenVINO ASYNC)")
print("Low-latency mode: explicit device + FAS warm-up")
print("Decision timer: starts when exactly one face is detected")
print("Target decision time: <", TARGET_DECISION_SECONDS, "second")
print("Max decision time   :", MAX_DECISION_SECONDS, "seconds")
print("Audit log           :", LOG_DIR)
print()
print("R       : Retry / reset")
print("F       : Fullscreen")
print("Q / ESC : Exit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:

        if retry_requested:
            reset_session(log_user_abort=True)
            retry_requested = False

        ok, frame = cap.read()

        if not ok:
            break

        main_start = time.perf_counter()

        faces = detect_faces(
            frame
        )

        now = time.perf_counter()


        # ====================================================
        # EXACTLY ONE FACE
        # ====================================================

        if len(faces) == 1:

            bbox = faces[0][:4]

            latest_bbox = bbox

            # Start timer only when one valid face is actually present.
            if (
                final_state is None
                and session_started_at is None
            ):
                session_started_at = now

            if (
                final_state is None
                and session_started_at is not None
            ):
                session_elapsed_sec = (
                    now - session_started_at
                )

            # --------------------------------------------
            # Stage 1 runs in main loop: light & fast
            # --------------------------------------------

            fr_result = get_embedding(
                frame,
                bbox,
            )

            if fr_result is not None:

                (
                    raw_fr_similarity,
                    raw_fr_distance,
                ) = compare_embeddings(
                    reference_embedding,
                    fr_result,
                )

                if final_state is None:
                    fr_similarity_history.append(
                        raw_fr_similarity
                    )

                    fr_distance_history.append(
                        raw_fr_distance
                    )


                if fr_distance_history:

                    median_fr_similarity = float(
                        np.median(
                            list(
                                fr_similarity_history
                            )
                        )
                    )

                    median_fr_distance = float(
                        np.median(
                            list(
                                fr_distance_history
                            )
                        )
                    )

                    if (
                        len(fr_distance_history)
                        >= FR_MIN_SAMPLES
                    ):
                        if (
                            fr_ready_time_sec is None
                            and session_started_at is not None
                        ):
                            fr_ready_time_sec = (
                                time.perf_counter() - session_started_at
                            )

                        fr_state = (
                            "MATCH"
                            if (
                                median_fr_distance
                                <= FR_DISTANCE_THRESHOLD
                            )
                            else "NOT MATCH"
                        )

                    else:
                        fr_state = "CHECKING"


            # --------------------------------------------
            # Submit liveness crop to BACKGROUND worker
            # --------------------------------------------

            if final_state is None:

                if (
                    now - last_fas_submit_time
                    >= FAS_SUBMIT_INTERVAL
                ):
                    fas_crop = crop_for_liveness(
                        frame,
                        bbox,
                    )

                    if fas_crop is not None:
                        submit_latest_fas_crop(
                            session_id,
                            fas_crop,
                        )

                        last_fas_submit_time = now


            # --------------------------------------------
            # Consume NEW liveness result if available
            # --------------------------------------------

            with fas_lock:
                worker_snapshot = dict(
                    fas_worker_state
                )

            if (
                worker_snapshot[
                    "sample_id"
                ]
                != last_consumed_fas_sample
                and worker_snapshot[
                    "session_id"
                ] == session_id
            ):
                last_consumed_fas_sample = (
                    worker_snapshot[
                        "sample_id"
                    ]
                )

                raw_fas_state = (
                    worker_snapshot[
                        "state"
                    ]
                )

                raw_fas_live = float(
                    worker_snapshot[
                        "p_live"
                    ]
                )

                raw_fas_spoof = float(
                    worker_snapshot[
                        "p_spoof"
                    ]
                )

                raw_fas_infer_ms = float(
                    worker_snapshot[
                        "infer_ms"
                    ]
                )

                if final_state is None:
                    fas_live_history.append(
                        raw_fas_live
                    )
                    fas_infer_ms_history.append(
                        raw_fas_infer_ms
                    )


            # --------------------------------------------
            # Aggregate liveness
            # --------------------------------------------

            if fas_live_history:

                median_fas_live = float(
                    np.median(
                        list(
                            fas_live_history
                        )
                    )
                )

                if (
                    len(fas_live_history)
                    >= FAS_MIN_SAMPLES
                ):
                    if (
                        fas_ready_time_sec is None
                        and session_started_at is not None
                    ):
                        fas_ready_time_sec = (
                            time.perf_counter() - session_started_at
                        )

                    fas_state = (
                        "LIVE"
                        if (
                            median_fas_live
                            >= FAS_LIVE_THRESHOLD
                        )
                        else "FAKE"
                    )

                else:
                    fas_state = "CHECKING"

            else:
                fas_state = "CHECKING"


            # --------------------------------------------
            # Final decision
            # --------------------------------------------

            if final_state is None:

                candidate = get_final_state(
                    fr_state,
                    fas_state,
                )

                if candidate is not None:
                    final_state = candidate

                    if session_started_at is not None:
                        decision_time_sec = (
                            now - session_started_at
                        )

                    if not current_session_logged:
                        append_audit_log(
                            final_state,
                            fr_state,
                            fas_state,
                            decision_time_sec,
                            session_id,
                        )
                        current_session_logged = True

                # Hard upper bound for one attempt.
                # If usable evidence is not ready in time, return RETRY.
                if (
                    final_state is None
                    and session_started_at is not None
                    and session_elapsed_sec
                    >= MAX_DECISION_SECONDS
                ):
                    final_state = "RETRY"
                    decision_time_sec = session_elapsed_sec

                    if not current_session_logged:
                        append_audit_log(
                            final_state,
                            fr_state,
                            fas_state,
                            decision_time_sec,
                            session_id,
                        )
                        current_session_logged = True


        # ====================================================
        # MULTIPLE / NO FACE
        # ====================================================

        elif len(faces) > 1:

            latest_bbox = None

            if final_state is None:
                reset_session()

            fr_state = "MULTIPLE FACES"
            fas_state = "MULTIPLE FACES"

        else:

            latest_bbox = None

            if final_state is None:
                reset_session()

            fr_state = "NO FACE"
            fas_state = "NO FACE"


        # ====================================================
        # MAIN LOOP FPS
        # ====================================================

        main_end = time.perf_counter()

        main_processing_ms = (
            main_end - main_start
        ) * 1000.0

        dt = (
            main_end - previous_time
        )

        previous_time = main_end

        if dt > 0:
            fps_history.append(
                1.0 / dt
            )

        fps = (
            sum(fps_history)
            / len(fps_history)
            if fps_history
            else 0.0
        )


        # ====================================================
        # VIDEO AREA
        # ====================================================

        display_frame = cv2.resize(
            frame,
            (
                VIDEO_W,
                VIDEO_H,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        if latest_bbox is not None:

            sx = (
                VIDEO_W
                / CAMERA_WIDTH
            )

            sy = (
                VIDEO_H
                / CAMERA_HEIGHT
            )

            x1, y1, x2, y2 = map(
                int,
                latest_bbox,
            )

            dx1 = int(
                x1 * sx
            )

            dy1 = int(
                y1 * sy
            )

            dx2 = int(
                x2 * sx
            )

            dy2 = int(
                y2 * sy
            )

            if final_state is not None:
                box_color = state_color(
                    final_state
                )

            elif (
                fr_state == "MATCH"
                and fas_state == "LIVE"
            ):
                box_color = GREEN

            elif (
                fr_state in (
                    "NOT MATCH",
                )
                or fas_state == "FAKE"
            ):
                box_color = RED

            else:
                box_color = YELLOW


            cv2.rectangle(
                display_frame,
                (dx1, dy1),
                (dx2, dy2),
                box_color,
                4,
            )


        # ====================================================
        # CANVAS
        # ====================================================

        canvas = np.zeros(
            (
                CANVAS_H,
                CANVAS_W,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            0:VIDEO_H,
            0:VIDEO_W,
        ] = display_frame

        cv2.rectangle(
            canvas,
            (
                VIDEO_W,
                0,
            ),
            (
                CANVAS_W,
                VIDEO_H,
            ),
            DARK,
            -1,
        )


        panel_x = VIDEO_W + 25


        # ====================================================
        # STAGE 1 PANEL
        # ====================================================

        draw_text(
            canvas,
            "1. FACE VERIFICATION",
            (
                panel_x,
                45,
            ),
            WHITE,
            0.58,
        )

        draw_text(
            canvas,
            fr_state,
            (
                panel_x,
                85,
            ),
            state_color(
                fr_state
            ),
            0.82,
        )

        draw_text(
            canvas,
            (
                f"median distance: "
                f"{median_fr_distance:.4f}"
            ),
            (
                panel_x,
                120,
            ),
            GRAY,
            0.47,
        )

        draw_text(
            canvas,
            (
                f"cosine: "
                f"{median_fr_similarity:.4f}"
            ),
            (
                panel_x,
                148,
            ),
            CYAN,
            0.47,
        )

        draw_text(
            canvas,
            (
                f"threshold <= "
                f"{FR_DISTANCE_THRESHOLD:.2f}"
            ),
            (
                panel_x,
                176,
            ),
            GRAY,
            0.43,
        )


        # ====================================================
        # STAGE 2 PANEL
        # ====================================================

        draw_text(
            canvas,
            "2. LIVENESS",
            (
                panel_x,
                230,
            ),
            WHITE,
            0.58,
        )

        draw_text(
            canvas,
            fas_state,
            (
                panel_x,
                270,
            ),
            state_color(
                fas_state
            ),
            0.82,
        )

        draw_text(
            canvas,
            (
                f"median LIVE: "
                f"{median_fas_live:.3f}"
            ),
            (
                panel_x,
                305,
            ),
            GRAY,
            0.47,
        )

        draw_text(
            canvas,
            (
                f"raw: {raw_fas_state} "
                f"L={raw_fas_live:.3f} "
                f"S={raw_fas_spoof:.3f}"
            ),
            (
                panel_x,
                333,
            ),
            CYAN,
            0.42,
        )

        with fas_lock:
            is_fas_busy = bool(
                fas_worker_state[
                    "busy"
                ]
            )

        draw_text(
            canvas,
            (
                f"Swin OV: "
                f"{raw_fas_infer_ms:.0f} ms "
                f"({'BUSY' if is_fas_busy else 'IDLE'})"
            ),
            (
                panel_x,
                362,
            ),
            GRAY,
            0.43,
        )

        draw_text(
            canvas,
            (
                f"samples: "
                f"{len(fas_live_history)}/"
                f"{FAS_MIN_SAMPLES}+"
            ),
            (
                panel_x,
                390,
            ),
            GRAY,
            0.43,
        )


        # ====================================================
        # PERFORMANCE
        # ====================================================

        fr_ready_label = (
            f"{fr_ready_time_sec:.2f}s"
            if fr_ready_time_sec is not None
            else "--"
        )
        fas_ready_label = (
            f"{fas_ready_time_sec:.2f}s"
            if fas_ready_time_sec is not None
            else "--"
        )

        draw_text(
            canvas,
            (
                f"FPS {fps:.1f} | "
                f"FR ready {fr_ready_label} | "
                f"FAS ready {fas_ready_label}"
            ),
            (
                panel_x,
                430,
            ),
            WHITE,
            0.39,
        )

        if decision_time_sec is not None:
            timing_text = (
                f"Decision: {decision_time_sec:.2f} s "
                f"(target <{TARGET_DECISION_SECONDS:.1f}s)"
            )
        elif session_started_at is not None:
            timing_text = (
                f"Elapsed: {session_elapsed_sec:.2f} s "
                f"(target <{TARGET_DECISION_SECONDS:.1f}s)"
            )
        else:
            timing_text = (
                "Decision timer: waiting for face"
            )

        draw_text(
            canvas,
            timing_text,
            (
                panel_x,
                458,
            ),
            CYAN,
            0.43,
        )


        # ====================================================
        # RETRY BUTTON
        # ====================================================

        button_text = (
            "RETRY"
            if final_state is not None
            else "RESET"
        )

        cv2.rectangle(
            canvas,
            (
                BUTTON_X1,
                BUTTON_Y1,
            ),
            (
                BUTTON_X2,
                BUTTON_Y2,
            ),
            (
                80,
                80,
                180,
            ),
            -1,
        )

        cv2.rectangle(
            canvas,
            (
                BUTTON_X1,
                BUTTON_Y1,
            ),
            (
                BUTTON_X2,
                BUTTON_Y2,
            ),
            WHITE,
            2,
        )

        text_size = cv2.getTextSize(
            button_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            2,
        )[0]

        text_x = int(
            (
                BUTTON_X1
                + BUTTON_X2
                - text_size[0]
            )
            / 2
        )

        text_y = int(
            (
                BUTTON_Y1
                + BUTTON_Y2
                + text_size[1]
            )
            / 2
        )

        draw_text(
            canvas,
            button_text,
            (
                text_x,
                text_y,
            ),
            WHITE,
            0.85,
            2,
        )

        draw_text(
            canvas,
            "R = Retry/Reset",
            (
                panel_x,
                575,
            ),
            GRAY,
            0.40,
        )

        draw_text(
            canvas,
            "F = Fullscreen",
            (
                panel_x,
                597,
            ),
            GRAY,
            0.40,
        )

        draw_text(
            canvas,
            f"Log: {get_daily_log_path().name}",
            (
                panel_x,
                620,
            ),
            GRAY,
            0.36,
        )


        # ====================================================
        # FINAL BAR
        # ====================================================

        if final_state is None:

            if (
                fr_state
                in (
                    "NO FACE",
                    "MULTIPLE FACES",
                )
            ):
                display_final = "POSITION ONE FACE"
            else:
                display_final = "ANALYZING..."

            final_color = YELLOW

        else:
            display_final = final_state
            final_color = state_color(
                final_state
            )


        cv2.rectangle(
            canvas,
            (
                0,
                VIDEO_H,
            ),
            (
                CANVAS_W,
                CANVAS_H,
            ),
            final_color,
            -1,
        )

        if decision_time_sec is not None:
            final_text = (
                f"FINAL: {display_final}  "
                f"({decision_time_sec:.2f} s)"
            )
        else:
            final_text = (
                f"FINAL: {display_final}"
            )

        draw_text(
            canvas,
            final_text,
            (
                30,
                VIDEO_H + 50,
            ),
            WHITE,
            1.05,
            3,
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            canvas,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if (
            key == ord("q")
            or key == 27
        ):
            break

        if key == ord("r"):
            reset_session(log_user_abort=True)

        if key == ord("f"):

            fullscreen = not fullscreen

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                (
                    cv2.WINDOW_FULLSCREEN
                    if fullscreen
                    else cv2.WINDOW_NORMAL
                ),
            )

            if not fullscreen:
                cv2.resizeWindow(
                    WINDOW_NAME,
                    CANVAS_W,
                    CANVAS_H,
                )

finally:
    fas_stop_event.set()

    purge_fas_queue()

    try:
        fas_queue.put_nowait(
            None
        )
    except queue.Full:
        pass

    fas_thread.join(
        timeout=1.0
    )

    cap.release()
    cv2.destroyAllWindows()

