<img width="500" height="500" alt="logo" src="https://github.com/user-attachments/assets/54c931f2-8c16-47e6-8a1e-8d301c43a5c2" />

# VideoTrack v1.1: Real-Time Face Recognition & Auto-Centering Tracker

**VideoTrack** is a high-speed, CPU-optimized face detection, recognition, and auto-centering camera tracking engine built with Python and OpenCV's Deep Neural Network (DNN) module.

It leverages **YuNet** for ultra-fast face detection and **SFace** for 128-dimensional vector feature extraction, paired with a thread-safe architecture and smooth Linear Interpolation (LERP) camera tracking.

### Demo of the auto-centering live feed:
<img title="" src="file:///C:/Users/Ruhaan/Pictures/Recording%202026-08-14%20200742.gif" alt="Recording 2026-08-14 200742.gif" width="354">

---

## ✨ Key Features

- **⚡ Dual-Model Neural Pipeline:**
  
  - **YuNet (`face_detection_yunet_2023mar.onnx`)**: High-speed face detection running on every frame.
  - **SFace (`face_recognition_sface_2021dec.onnx`)**: Deep feature vector extraction strided to every 15th frame for CPU efficiency.
- **🎯 Cinematic Auto-Centering (LERP):** Sub-pixel Linear Interpolation (alpha = 0.05) creates a smooth, floating tracking window that eliminates jitter while maintaining a cinematic glide.
  
- **⚡ Strided Inference Engine:** Heavy SFace recognition is throttled to every 15th frame under continuous tracking, maintaining high FPS without sacrificing real-time responsiveness.
  
- **🔄 Stale-ID Protection:** An IoU (Intersection over Union) continuity check detects when a different person steps into frame and forces an immediate re-recognition, so the previous person's name is never shown on a new face.
  
- **🔒 Thread-Safe Design:**
  
  - Explicit dual-mutex synchronization (`model_lock` + `database_lock`) prevents race conditions during model inference and database updates.
  - Background daemon threads handle the embedding, saving, and deletion work for face registration and ID deletion. The Name/ID input dialog runs on the main thread (required by Tkinter), so the video briefly pauses while a dialog is open.
- **💾 Persistent Embeddings:** Each registered face is saved as both an image and its 128-dimensional embedding (`.npy`), so profiles reload reliably after a restart.
  
- **📊 Live Performance Analytics:** Built-in rolling metrics dashboard (FPS, detection latency, recognition latency) togglable via hotkey with zero overhead when disabled.
  
- **🖥️ Dual-View Display:** Independently toggle between raw camera feed, auto-centered tracking crop, or both simultaneously.
  

---

## 🏗️ Architecture & Processing Pipeline

```
┌────────────────────────┐
│    Webcam Feed Input   │
│    (640x480 Hardware)  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  YuNet Face Detector   │
│   (Every Frame)        │
└───────────┬────────────┘
            │
    ┌───────┴───────────────────────┐
    ▼                               ▼
┌──────────────────┐    ┌──────────────────────────┐
│ Floating-Point   │    │ Strided SFace Recog.     │
│ LERP Tracking    │    │ (Every 15th Frame, or    │
│ (alpha = 0.05)   │    │  immediately on a face   │
│                  │    │  swap detected via IoU)  │
└───────────┬──────┘    └───────────┬──────────────┘
            │                       │
            └──────────┬────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│ Dual Video Display + Performance Dashboard       │
│  1. Auto-Centered Cropped Camera View            │
│  2. Raw Overview Camera Feed                     │
│  3. Live Stats Window (Togglable)                │
└──────────────────────────────────────────────────┘
```

---

## 📊 Measured Performance Benchmarks

*Tested on host environment (measured on v1.0; the detection and recognition path is unchanged in v1.1, but re-run the built-in dashboard on your own machine to confirm):*

- **CPU:** Intel Core i5-8365U
  
- **RAM:** 16 GB
  
- **OS:** Windows 11
  
- **Camera Input:** 640×480 Hardware Capture Resolution
  
  | Metric | Measured Result |
  | --- | --- |
  | **Average Frame Rate** | **32.0 FPS** |
  | **YuNet Detection Latency** | **~19.2 ms** |
  | **SFace Recognition Latency** | **~20.8 ms** |
  | **Recognition Interval** | Every 15th frame |
  | **CPU Usage Reduction** | ~55% lower vs. non-strided inference |
  

---

## 📁 Repository Structure

```text
VideoTrack/
├── face_tracker.py                      # Main application script
├── requirements.txt                     # Dependency manifest
├── .gitignore                           # Local face data exclude rules
├── face_detection_yunet_2023mar.onnx    # YuNet detector model weights
├── face_recognition_sface_2021dec.onnx  # SFace recognition model weights
└── known_faces/                         # Local directory for face profiles
    ├── .gitkeep
    ├── John_Doe_ID101.jpg               # Aligned face image
    ├── John_Doe_ID101.npy               # Stored 128-D embedding
    ├── Jane_Smith_ID102.jpg
    └── Jane_Smith_ID102.npy
```

> Add `known_faces/*.jpg`, `known_faces/*.jpeg`, `known_faces/*.png` and `known_faces/*.npy` to `.gitignore` so biometric data is never committed.

---

## 🛠️ Prerequisites & Installation

### 1. Requirements

- Python 3.8+
- OpenCV with DNN module support (`opencv-python`)
- NumPy
- Tkinter (usually bundled with Python)

### 2. Download dependencies

```bash
pip install opencv-python numpy
```

### 3. Required Pre-trained ONNX Models

Download and place these in your project root:

1. `face_detection_yunet_2023mar.onnx`
2. `face_recognition_sface_2021dec.onnx`

> Models courtesy of the [OpenCV DNN Face Model Zoo](https://github.com/opencv/opencv_zoo).

If a model file is missing, the program exits with a clear error message instead of crashing.

---

## 🚀 Execution

```bash
python face_tracker.py
```

### 🎮 Keyboard Controls

| Key | Action |
| --- | --- |
| **`1`** | Toggle Raw Camera Feed window |
| **`2`** | Toggle Auto-Centered Camera Feed window |
| **`b`** | Toggle bounding boxes and identity labels |
| **`s`** | Open/Close Performance Stats Dashboard |
| **`r`** | Register new face (opens Name/ID prompt) |
| **`d`** | Delete face by ID |
| **`q`** | Quit application safely |

### 🧾 Registration Rules

- **Name:** letters, digits, spaces and `-`. Other characters are removed. Capitalization is kept exactly as typed.
- **ID:** letters, digits and `-` only (no underscores or spaces). IDs must be unique; registering an existing ID is rejected.
- **Delete:** removes the profile's image and its stored embedding.
- **Older profiles:** images saved by v1.0 (without a `.npy` file) still load; if they cannot be re-detected, 112×112 aligned crops are used directly.

---

## 🧠 Key Engineering Decisions

- **Strided Recognition:** Running deep face recognition on every frame creates heavy CPU bottlenecks. By striding SFace inference to every 15th frame while keeping YuNet active on every frame, VideoTrack achieves a smooth 30+ FPS video pipeline with minimal latency.
  
- **Dual-Lock Concurrency:** `model_lock` isolates the OpenCV C++ inference calls between threads to prevent memory corruption, while `database_lock` ensures thread-safe reads/writes when appending new vector embeddings in real time.
  
- **Main-Thread Dialogs:** Tkinter is not thread-safe, so Name/ID dialogs run on the main thread. Only the model work and file I/O are offloaded to background threads.
  
- **Stored Embeddings:** Registration saves the embedding next to the image instead of re-detecting the face on startup. The saved image is a small aligned crop that the detector often fails to find again, which would silently drop profiles.
  
- **LERP Smoothing:** Direct target box centering creates noticeable camera jitter. Linear Interpolation (`smooth = smooth + alpha * (target - smooth)`) produces smooth, cinematic camera transitions.
  

---

### 🔐 Privacy & Local Data Handling

- **100% On-Device Processing:** No cloud APIs or network requests are made. All feature extraction, matching, and video rendering happen entirely locally.
  
- **What is stored:** For each registered person, `known_faces/` contains an aligned face **image** (`.jpg`) and the face **embedding** (`.npy`, a 128-dimensional vector). Both are biometric data.
  
- **Not encrypted:** Files are stored as plain, unencrypted files. Anyone with access to the folder can read or copy them.
  
- **Recommendations:** Enable full-disk encryption (BitLocker / FileVault), keep `known_faces/` out of cloud sync and backups, restrict folder permissions, and never commit it to version control.
  
- **Legal note:** If you register other people (students, employees, visitors), data-protection laws (e.g. GDPR, India's DPDP Act, BIPA) may require consent, a stated purpose, and limited retention.
  

---

## ⚙️ Configuration Parameters

These constants can be changed in `face_tracker.py`:

```python
MATCH_THRESHOLD = 0.36   # Cosine similarity threshold for identity matching
alpha = 0.05             # LERP smoothing factor (lower = smoother, higher = snappier)
CROP_SIZE = 450          # Max size (px) of the auto-centered crop (shrinks automatically on smaller frames)
IOU_THRESHOLD = 0.25     # Below this overlap, a new face is assumed and re-recognized immediately
```

---

## ⚠️ Known Limitations & Edge Cases

- **No liveness / anti-spoofing:** A photo or video of a registered person shown to the camera can be recognized. Do not rely on this as a security control without adding liveness detection.
- **Single face:** Only the largest detected face in the frame is tracked and recognized.
- **Dialog pause:** The video pauses while the registration or deletion dialog is open.
- **Unauthenticated deletion:** Anyone at the keyboard can delete a profile by ID.
- **Platform Compatibility:** Tested on Windows. Dialogs now run on the main thread, which should improve macOS/Linux behavior, but this has not been tested there.
- **Ghost Frame:** If the tracked face exits the frame, the auto-centered crop freezes on the last known LERP coordinates rather than clearing to black. (Cosmetic only; it does not affect the program.)
- **Hardware Resolution:** `cap.set()` requests 640×480 but actual resolution depends on camera driver capabilities.

---

## 📝 Changes in v1.1

- Registered faces now reload reliably after a restart (embeddings saved as `.npy`).
- Tkinter dialogs moved to the main thread.
- Names and IDs are sanitized; duplicate IDs are rejected; deletion removes all matching files and entries.
- Names keep the capitalization typed (no forced title-casing).
- Camera is checked before it is configured and always released on exit.
- Models load at startup with a clear error if missing, instead of at import.
- Auto-centered crop adapts to low-resolution cameras.

---

### 🤖 AI Collaboration Disclosure

This project was developed using AI assistance as a collaborative engineering and debugging tool. The software architecture, thread synchronization strategies, parameter tuning, performance measurements, and debugging were thoroughly reviewed, tested, and validated manually.

---

## 📄 License

Distributed under the MIT License.


