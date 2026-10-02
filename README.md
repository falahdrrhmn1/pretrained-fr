# OpenVINO Face Attendance

Prototype attendance menggunakan dua tahap verifikasi:

1. **Face Verification** menggunakan OpenVINO.
2. **Face Anti-Spoofing / Liveness Detection** menggunakan Swin-V2 yang dijalankan melalui OpenVINO.

Program melakukan **1:1 face verification**, yaitu membandingkan wajah dari webcam dengan satu foto reference yang diberikan melalui parameter `--reference`.

Program juga menyimpan hasil setiap proses attendance ke dalam file JSON harian.

## Requirements

Disarankan menggunakan:

- Windows 10 / Windows 11 64-bit
- Python 3.10–3.14 64-bit
- Git
- Git LFS
- Webcam

Mode yang direkomendasikan:

```text
--fas-device CPU
```

GPU tidak diperlukan untuk menjalankan versi standar.

---

## 1. Clone repository

Buka PowerShell.

Jalankan:

```powershell
git lfs install
```

Kemudian clone repository:

```powershell
git clone https://github.com/falahdrrhmn1/pretrained-fr.git
```

Masuk ke folder project:

```powershell
cd .\pretrained-fr
```

Pastikan file model Git LFS sudah didownload:

```powershell
git lfs pull
```

Cek file LFS:

```powershell
git lfs ls-files
```

---

## 2. Cek Python

Jalankan:

```powershell
python --version
```

atau:

```powershell
py --version
```

---

## 3. Buat virtual environment

Jalankan:

```powershell
python -m venv .venv
```

Jika perintah `python` tidak tersedia tetapi `py` tersedia:

```powershell
py -m venv .venv
```

Tidak perlu mengaktifkan virtual environment karena semua perintah di bawah menjalankan Python dari `.venv` secara langsung.

---

## 4. Upgrade pip

Jalankan:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

---

## 5. Install dependencies

Jalankan:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Tunggu sampai instalasi selesai.

---

## 6. Test dependencies

Jalankan:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; import numpy; import openvino; print('DEPENDENCIES OK')"
```

Jika benar, akan muncul:

```text
DEPENDENCIES OK
```

---

## 7. Cek OpenVINO device

Jalankan:

```powershell
.\.venv\Scripts\python.exe -c "import openvino as ov; print(ov.Core().available_devices)"
```

Contoh:

```text
['CPU']
```

atau:

```text
['CPU', 'GPU']
```

Untuk penggunaan standar, cukup menggunakan:

```text
CPU
```

---

## 8. Siapkan foto reference

Buat folder:

```powershell
New-Item -ItemType Directory -Force .\reference_images
```

Masukkan foto wajah pegawai ke folder tersebut.

Contoh:

```text
pretrained-fr/
│
├── attendance_full_openvino_logged_fast_json.py
├── requirements.txt
├── models/
│
└── reference_images/
    └── EMPLOYEE_001.jpg
```

Foto reference sebaiknya:

- hanya memiliki satu wajah;
- wajah terlihat jelas;
- tidak terlalu gelap;
- tidak blur;
- posisi wajah cukup frontal.

---

## 9. Jalankan program

Contoh untuk employee `001`:

```powershell
.\.venv\Scripts\python.exe `
    .\attendance_full_openvino_logged_fast_json.py `
    --reference ".\reference_images\EMPLOYEE_001.jpg" `
    --employee-id "001" `
    --location "Office A" `
    --fas-device CPU
```

Parameter:

```text
--reference
```

Path foto wajah yang dijadikan reference.

```text
--employee-id
```

ID pegawai yang sedang melakukan attendance.

Contoh:

```text
001
```

```text
--location
```

Lokasi attendance.

Contoh:

```text
Office A
```

```text
--fas-device
```

Device OpenVINO yang digunakan oleh model liveness.

Direkomendasikan:

```text
CPU
```

---

## 10. Expected output

Jika program berhasil dimulai, terminal kurang lebih akan menampilkan:

```text
LOADING STAGE 1 - OPENVINO 1:1 FACE VERIFICATION
```

kemudian:

```text
BUILDING REFERENCE EMBEDDING
```

kemudian:

```text
Reference face OK.
```

kemudian:

```text
LOADING STAGE 2 - CVPR2024 SWIN-V2 VIA OPENVINO
```

dan akhirnya:

```text
FULL ATTENDANCE CANDIDATE STARTED
```

Webcam kemudian akan terbuka.

---

## 11. Hasil keputusan

Program dapat memberikan beberapa hasil.

```text
VERIFIED
```

Face verification cocok dan wajah terdeteksi LIVE.

```text
WRONG PERSON
```

Wajah LIVE tetapi tidak cocok dengan reference.

```text
SPOOF ATTACK
```

Identity cocok tetapi liveness mendeteksi spoof/fake.

```text
WRONG PERSON + SPOOF
```

Identity tidak cocok dan liveness mendeteksi spoof/fake.

```text
RETRY
```

Program tidak mendapatkan evidence yang cukup sebelum timeout.

---

## 12. Keyboard controls

Saat jendela attendance aktif:

```text
R
```

Retry / reset attendance.

```text
F
```

Fullscreen.

```text
Q
```

Keluar dari aplikasi.

Tombol:

```text
ESC
```

juga dapat digunakan untuk keluar.

---

## 13. Attendance JSON log

Program secara otomatis membuat folder:

```text
logs/
```

Contoh:

```text
logs/
└── attendance_2026-10-02.json
```

Setiap proses attendance akan ditambahkan sebagai event ke file JSON pada tanggal tersebut.

Contoh informasi yang disimpan meliputi:

```text
timestamp
employee ID
location
final decision
face verification score
liveness score
decision time
OpenVINO device
```

Folder `logs` tidak perlu dibuat secara manual.

---

## 14. Menggunakan webcam lain

Default webcam:

```text
--camera 0
```

Jika laptop mempunyai beberapa kamera dan kamera utama tidak terbuka, coba:

```powershell
.\.venv\Scripts\python.exe `
    .\attendance_full_openvino_logged_fast_json.py `
    --reference ".\reference_images\EMPLOYEE_001.jpg" `
    --employee-id "001" `
    --location "Office A" `
    --camera 1 `
    --fas-device CPU
```

Coba:

```text
--camera 0
--camera 1
--camera 2
```

sesuai device yang tersedia.

---

## Troubleshooting

### `File/model belum ditemukan`

Pastikan Git LFS sudah aktif:

```powershell
git lfs install
git lfs pull
```

Kemudian cek:

```powershell
git lfs ls-files
```

Pastikan folder `models` tidak berubah struktur.

### `No module named cv2`

Jalankan:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `No module named openvino`

Jalankan:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `Reference image tidak ditemukan`

Periksa parameter:

```text
--reference
```

Pastikan path file benar.

### `Tidak ada wajah pada reference image`

Gunakan foto dengan wajah yang lebih jelas.

### `Reference image harus berisi tepat satu wajah`

Gunakan foto yang hanya memiliki satu wajah.

### `Webcam tidak dapat dibuka`

Tutup aplikasi lain yang sedang menggunakan webcam seperti Teams, Zoom, Discord, atau aplikasi Camera.

Kemudian coba camera index lain:

```text
--camera 1
```

### GPU bermasalah

Gunakan:

```text
--fas-device CPU
```

CPU merupakan mode default dan mode yang direkomendasikan untuk pengujian standar.


Siapkan foto reference Anda sendiri.

1. Buat folder:

   reference_images

2. Masukkan satu foto wajah Anda sendiri, misalnya:

   reference_images\MY_FACE.jpg

3. Foto reference harus:
   - berisi tepat satu wajah;
   - wajah terlihat jelas;
   - tidak blur;
   - pencahayaan cukup;
   - disarankan menghadap cukup frontal.

Folder reference_images dan file JPG/JPEG diabaikan oleh Git
dan tidak akan diupload ke repository.

---

## Menyiapkan Model Swin-V2 Liveness

Model OpenVINO Swin-V2 yang besar tidak disimpan di repository ini.

File berikut akan dibuat secara lokal:

    models\commercial_test\cvpr2024_fas\openvino\face_swin_v2_base_fp32.bin

Model berasal dari project resmi:

    Xianhua-He/cvpr2024-face-anti-spoofing-challenge

Model yang digunakan adalah:

    face_swin_v2_base.pth

### 1. Install dependency untuk setup model

Setelah virtual environment dan requirements utama selesai di-install, jalankan:

    .\.venv\Scripts\python.exe -m pip install torch torchvision timm gdown

Dependency ini diperlukan untuk mendownload checkpoint asli dan melakukan konversi ke OpenVINO.

### 2. Buat folder weights

Jalankan dari root folder project:

    New-Item `
        -ItemType Directory `
        -Force `
        ".\models\commercial_test\cvpr2024_fas\weights" |
    Out-Null

### 3. Download checkpoint resmi

Jalankan:

    .\.venv\Scripts\python.exe -m gdown `
        1E4UD8UK_KzjhpAvR6hYInlteOEaxDZbZ `
        -O ".\models\commercial_test\cvpr2024_fas\weights\face_swin_v2_base.pth"

Tunggu sampai download selesai.

Cek:

    Test-Path ".\models\commercial_test\cvpr2024_fas\weights\face_swin_v2_base.pth"

Output harus:

    True

### 4. Convert checkpoint ke OpenVINO

Jalankan:

    .\.venv\Scripts\python.exe `
        .\convert_cvpr2024_swin_to_openvino.py `
        --device CPU

Proses ini akan menghasilkan:

    models\commercial_test\cvpr2024_fas\openvino\
    ├── face_swin_v2_base_fp32.xml
    └── face_swin_v2_base_fp32.bin

File BIN berukuran sekitar 380 MB dan hanya disimpan di komputer lokal.

### 5. Verifikasi hasil conversion

Jalankan:

    Test-Path ".\models\commercial_test\cvpr2024_fas\openvino\face_swin_v2_base_fp32.xml"

dan:

    Test-Path ".\models\commercial_test\cvpr2024_fas\openvino\face_swin_v2_base_fp32.bin"

Keduanya harus menghasilkan:

    True

Cek ukuran file BIN:

    "{0:N2} MB" -f (
        (Get-Item ".\models\commercial_test\cvpr2024_fas\openvino\face_swin_v2_base_fp32.bin").Length / 1MB
    )

Ukuran normal sekitar:

    380 MB

Setelah model tersedia, lanjutkan ke bagian menjalankan aplikasi.

