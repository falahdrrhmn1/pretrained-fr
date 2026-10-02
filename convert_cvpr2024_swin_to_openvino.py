from pathlib import Path
import argparse
import statistics
import sys
import time

import numpy as np
import openvino as ov
import torch


ROOT = Path(__file__).resolve().parent

FAS_REPO_DIR = (
    ROOT
    / "models"
    / "commercial_test"
    / "cvpr2024_fas"
)

FAS_WEIGHT_PATH = (
    FAS_REPO_DIR
    / "weights"
    / "face_swin_v2_base.pth"
)

FAS_SWIN_SOURCE = (
    FAS_REPO_DIR
    / "nets"
    / "swin_transformer_v2.py"
)

DEFAULT_OUTPUT_DIR = (
    FAS_REPO_DIR
    / "openvino"
)

INPUT_SHAPE = (1, 3, 224, 224)


parser = argparse.ArgumentParser(
    description=(
        "Convert the existing CVPR2024 face_swin_v2_base checkpoint "
        "to inference-only OpenVINO IR without quantization."
    )
)

parser.add_argument(
    "--device",
    default="CPU",
    help="Device used for OpenVINO benchmark: CPU, AUTO, GPU, etc.",
)

parser.add_argument(
    "--runs",
    type=int,
    default=8,
    help="Number of timed benchmark runs after warmup.",
)

parser.add_argument(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
)

args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_xml = output_dir / "face_swin_v2_base_fp32.xml"
output_bin = output_dir / "face_swin_v2_base_fp32.bin"

for path in [FAS_WEIGHT_PATH, FAS_SWIN_SOURCE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


print("=" * 78)
print("CVPR2024 SWIN-V2 -> OPENVINO IR")
print("=" * 78)
print("Checkpoint :", FAS_WEIGHT_PATH)
print("Output XML :", output_xml)
print("Precision  : FP32 weights (no quantization, no FP16 compression)")
print()


# The original source contains one hard-coded .cuda() call in the Swin code.
# Patch it exactly as in the working realtime prototype so CPU conversion works.
source_text = FAS_SWIN_SOURCE.read_text(encoding="utf-8")
old_text = "torch.tensor(1. / 0.01).cuda()"
new_text = "torch.tensor(1. / 0.01, device=x.device)"

if old_text in source_text:
    FAS_SWIN_SOURCE.write_text(
        source_text.replace(old_text, new_text),
        encoding="utf-8",
    )
    print("CPU compatibility patch applied to swin_transformer_v2.py")
else:
    print("CPU compatibility patch already present / not required")


sys.path.insert(0, str(FAS_REPO_DIR))
from nets.utils import get_model  # noqa: E402


print("Loading PyTorch architecture...")
base_model = get_model(
    "swin_v2_b",
    num_classes=2,
)

print("Loading checkpoint...")
checkpoint = torch.load(
    str(FAS_WEIGHT_PATH),
    map_location="cpu",
    weights_only=False,
)

state_dict = checkpoint["state_dict"]
clean_state_dict = {}

for key, value in state_dict.items():
    clean_key = key
    while clean_key.startswith("module."):
        clean_key = clean_key[len("module.") :]
    clean_state_dict[clean_key] = value

load_result = base_model.load_state_dict(
    clean_state_dict,
    strict=False,
)

print("Checkpoint arch :", checkpoint.get("arch"))
print("Missing keys    :", len(load_result.missing_keys))
print("Unexpected keys :", len(load_result.unexpected_keys))

if load_result.missing_keys:
    print("WARNING: missing keys:")
    for key in load_result.missing_keys[:20]:
        print("  -", key)

if load_result.unexpected_keys:
    print("WARNING: unexpected keys:")
    for key in load_result.unexpected_keys[:20]:
        print("  -", key)

base_model.eval()


class LogitsOnly(torch.nn.Module):
    """Normalize repository output to exactly one [N, 2] logits tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        return output


model = LogitsOnly(base_model).eval()
example = torch.randn(*INPUT_SHAPE, dtype=torch.float32)


@torch.inference_mode()
def pytorch_once(x):
    result = model(x)
    return result.detach().cpu().numpy()


print()
print("Checking original PyTorch output...")
pt_output = pytorch_once(example)
print("PyTorch output shape:", pt_output.shape)


print()
print("Converting with openvino.convert_model() ...")
convert_start = time.perf_counter()

ov_model = ov.convert_model(
    model,
    example_input=example,
)

convert_sec = time.perf_counter() - convert_start
print(f"Conversion completed in {convert_sec:.2f} s")

# IMPORTANT: save_model normally compresses weights to FP16 by default.
# We explicitly disable it here to preserve the original floating-point weights.
ov.save_model(
    ov_model,
    str(output_xml),
    compress_to_fp16=False,
)

if not output_xml.exists() or not output_bin.exists():
    raise RuntimeError("OpenVINO IR files were not created correctly.")

print("Saved:")
print("  ", output_xml)
print("  ", output_bin)
print(f"IR size: {(output_xml.stat().st_size + output_bin.stat().st_size) / 1024**2:.1f} MiB")


print()
print("Compiling OpenVINO model for benchmark...")
core = ov.Core()
print("Available devices:", core.available_devices)
print("Benchmark device :", args.device.upper())

compiled = core.compile_model(
    str(output_xml),
    args.device.upper(),
    {"PERFORMANCE_HINT": "LATENCY"},
)

ov_output_port = compiled.output(0)
example_np = example.numpy()

# Warm up OpenVINO.
for _ in range(3):
    compiled([example_np])

ov_result = np.asarray(
    compiled([example_np])[ov_output_port],
    dtype=np.float32,
)

max_abs_diff = float(np.max(np.abs(pt_output - ov_result)))
mean_abs_diff = float(np.mean(np.abs(pt_output - ov_result)))

print()
print("Numerical check on the same input:")
print(f"  max abs diff  : {max_abs_diff:.8f}")
print(f"  mean abs diff : {mean_abs_diff:.8f}")


# Benchmark PyTorch CPU using the same exact network.
try:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
except RuntimeError:
    pass

for _ in range(2):
    pytorch_once(example)

pt_times = []
for _ in range(max(1, args.runs)):
    start = time.perf_counter()
    pytorch_once(example)
    pt_times.append((time.perf_counter() - start) * 1000.0)

ov_times = []
for _ in range(max(1, args.runs)):
    start = time.perf_counter()
    compiled([example_np])
    ov_times.append((time.perf_counter() - start) * 1000.0)

pt_median = statistics.median(pt_times)
ov_median = statistics.median(ov_times)
speedup = pt_median / ov_median if ov_median > 0 else float("inf")

print()
print("=" * 78)
print("BENCHMARK")
print("=" * 78)
print(f"PyTorch median : {pt_median:.1f} ms")
print(f"OpenVINO median: {ov_median:.1f} ms")
print(f"Speed-up       : {speedup:.2f}x")

estimated_two_sample = (ov_median * 2.0) / 1000.0
print(f"Estimated two-sample FAS time: ~{estimated_two_sample:.2f} s")

if estimated_two_sample < 1.0:
    print("TARGET CHECK: PASS — two sequential liveness samples are under ~1 s.")
else:
    print(
        "TARGET CHECK: NOT YET <1 s — keep the same model/quality, but try "
        "--device AUTO or --device GPU if an OpenVINO-compatible accelerator is available."
    )

print()
print("Next run:")
print(
    r'.\.venv\Scripts\python.exe .\attendance_full_openvino_logged.py '
    r'--reference ".\FALAH.jpg" --employee-id 001 --location "Office A"'
)
