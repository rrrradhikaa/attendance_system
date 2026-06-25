# Multi-Modal Attendance System

An automated attendance tracking system that uses **simultaneous face and voice recognition** to identify students and log attendance — no manual input required.

---

## Features

- **Dual-modal authentication** - face and voice recognition run in parallel via separate threads; the higher-confidence result wins
- **ArcFace face recognition** - deep learning-based face embeddings with a trained SVM classifier
- **ECAPA-TDNN voice verification** - speaker embeddings with Adaptive Score Normalization (AS-norm) for robust matching
- **Fusion decision engine** - combines or arbitrates between modalities; boosts confidence when both agree on the same student
- **SQLite persistence** - stores student profiles, face encodings, voice samples, attendance records, and voice enrollment status
- **Automatic database initialization** - schema is created on first run with no manual setup
- **Attendance reporting** - query records by date or pull the full history as a DataFrame
- **Database backups** - timestamped `.db` snapshots with a single command
- **CLI menu** - register students, mark attendance, view reports, retrain classifier, and more

---

## Project Structure

```
.
├── integrated.py                   # Core system logic and CLI entry point
├── FaceRecognition.py       # AttendanceSystem, FaceCapture (face pipeline)
├── VoiceRecognition.py      # VoiceVerifier, VoiceRecorder (voice pipeline)
├── database/
│   └── attendance_system.db # SQLite database (auto-created)
├── models/                  # Trained classifier artifacts
├── logs/                    # System logs
├── temp/                    # Temporary audio files (auto-cleaned)
├── encryption_key/          # Key storage directory
└── backups/                 # Timestamped database backups
```

---

## Requirements

### Python

Python 3.8 or higher is required.

### Dependencies

```bash
pip install opencv-python numpy pandas librosa pyaudio deepface
```

> **Note:** `pyaudio` requires PortAudio. On Linux: `sudo apt install portaudio19-dev`. On macOS: `brew install portaudio`. On Windows, use a pre-built wheel from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### Hardware

- A webcam accessible via `cv2.VideoCapture(0)`
- A microphone for voice capture

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd attendance-system
pip install -r requirements.txt
```

### 2. Run the system

```bash
python integrated.py
```

The database and required directories are created automatically on first run.

### 3. Register a student

From the CLI menu, choose **Option 1**. You will be prompted for:

- Full name
- Student ID (must be unique)
- Roll number
- Class / section

The system will then:
1. Capture 10 face images via webcam and extract ArcFace embeddings
2. Optionally record voice samples for speaker enrollment
3. Train (or retrain) the face classifier

### 4. Mark attendance

Choose **Option 2**. Face and voice recognition run simultaneously for up to 10 seconds each. The system reports which modality produced the final decision and marks the record in the database (one entry per student per day).

---

## How Recognition Works

### Face Pipeline

1. Webcam frames are processed using Haar Cascade for face detection
2. Detected face ROIs are passed to `AttendanceSystem.extract_features()` (ArcFace via DeepFace)
3. Features are classified using a trained SVM with a probability threshold of `0.3`
4. The highest-confidence detection within the capture window is kept

### Voice Pipeline

1. 5 seconds of audio are recorded at 16 kHz mono via PyAudio
2. An ECAPA-TDNN embedding (192-dim) is extracted from the audio
3. The embedding is compared against all enrolled samples using cosine similarity
4. **AS-norm** is applied when ≥2 impostor scores are available:
   - Score normalized as `(score - cohort_mean) / cohort_std`
   - Mapped to `[0, 1]` and compared against a threshold of `0.55` (ECAPA) or `0.50` (MFCC fallback)

### Fusion Logic

| Scenario | Decision |
|----------|----------|
| Only face succeeds | Use face result |
| Only voice succeeds | Use voice result |
| Both succeed, different IDs | Use higher-confidence result |
| Both succeed, same ID | Use face confidence; label as `Face + Voice (Agree)` |
| Both fail | Attendance not marked |

---

## Database Schema

| Table | Description |
|---|---|
| `students` | Student profiles: name, ID, roll number, class, face encoding path |
| `attendance` | Daily attendance records with confidence and method used |
| `voice_samples` | Raw voice embeddings per student |
| `voice_enrollment` | Enrollment status and sample count per student |
| `voice_verifications` | Audit log of all voice verification attempts |

---

## CLI Menu Reference

| Option | Action |
|--------|--------|
| 1 | Register a new student (face + optional voice) |
| 2 | Mark attendance (simultaneous face + voice) |
| 3 | View today's attendance report |
| 4 | View all registered students |
| 5 | System statistics (student count, days tracked, etc.) |
| 6 | Retrain the face classifier |
| 7 | Back up the database |
| 8 | Exit |

---

## Configuration Notes

- **Face confidence threshold:** `0.3` (in `mark_attendance`)
- **Voice confidence threshold:** `0.55` for ECAPA, `0.50` for MFCC fallback
- **Recognition timeout:** 10 seconds per modality, 20 seconds total
- **Minimum face captures for registration:** 5 images (10 attempted)
- **Minimum face encodings to proceed:** 3 successful extractions

To adjust these, modify the constants directly in `app.py` and `VoiceRecognition.py`.

---

## Troubleshooting

**Camera not detected**
Ensure no other application is using the webcam. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras.

**`FaceRecognition` or `VoiceRecognition` import errors**
The system degrades gracefully — a warning is printed and the unavailable modality is skipped. Ensure all dependencies are installed and module files are in the same directory as `app.py`.

**Voice: "No voice detected"**
The amplitude threshold is `100` (raw int16). Check microphone permissions and ensure the mic is not muted at the OS level.

**Dimension mismatch warnings in voice verification**
Occurs when a student was enrolled with MFCC features but the current session uses ECAPA (or vice versa). Re-enroll the student to align embedding types.

**`Already marked for today`**
The attendance table enforces `UNIQUE(student_id, date)`. Each student can only be marked once per calendar day by design.

---

## License

MIT License. See `LICENSE` for details.
