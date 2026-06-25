import os
import logging
import sqlite3
import joblib
import time
import cv2 as cv
import numpy as np
import face_recognition
from collections import deque
from datetime import datetime
from cryptography.fernet import Fernet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# ===========================
# NEW: DeepFace imports for robust face recognition
# ===========================
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not installed. Please install with: pip install deepface")

# ---------------------------
# Setup directories & logging
# ---------------------------
os.makedirs("database", exist_ok=True)
os.makedirs("encryption_key", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_path = os.path.join(os.getcwd(), "logs", "attendance_system.log")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logging.info("✅ Attendance System Logging Initialized")

# ===========================
# CLASS: RobustFaceRecognizer
# ===========================
class RobustFaceRecognizer:
    """
    Robust face recognition using ArcFace (best accuracy)
    """
    def __init__(self, use_deepface=True):
        """
        Args:
            use_deepface: Whether to use DeepFace library (recommended)
        """
        self.model_name = "ArcFace"  # Fixed to ArcFace only
        self.use_deepface = use_deepface and DEEPFACE_AVAILABLE
        
        if self.use_deepface:
            logging.info(f"✅ RobustFaceRecognizer initialized with ArcFace model")
            self.model_config = {
                "ArcFace": {"target_size": (112, 112), "threshold": 0.30}
            }
        else:
            logging.warning("⚠️ DeepFace not available. ArcFace requires DeepFace. Please install with: pip install deepface")
            self.model_name = "traditional"
            self.model_config = {"traditional": {"target_size": None, "threshold": 0.35}}
    
    def extract_features(self, image_bgr):
        """
        Extract face embeddings from BGR image using ArcFace
        
        Args:
            image_bgr: numpy array in BGR format (from OpenCV)
            
        Returns:
            embedding: numpy array or None if no face detected
        """
        if image_bgr is None or image_bgr.size == 0:
            return None
        
        # Use DeepFace for robust feature extraction with ArcFace
        if self.use_deepface:
            try:
                # Convert to RGB for DeepFace
                img_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
                
                # Extract embedding using DeepFace with ArcFace
                embedding_objs = DeepFace.represent(
                    img_path=img_rgb,
                    model_name=self.model_name,
                    detector_backend="retinaface",  # Most accurate
                    align=True,  # Face alignment improves accuracy
                    enforce_detection=False  # Return None if no face
                )
                
                if embedding_objs and len(embedding_objs) > 0:
                    embedding = embedding_objs[0]["embedding"]
                    return np.array(embedding, dtype=np.float32)
                else:
                    return None
                    
            except Exception as e:
                # Fall back to traditional method if DeepFace fails
                if "Face could not be detected" not in str(e):
                    logging.warning(f"ArcFace extraction failed: {e}, falling back to traditional")
                return self._extract_traditional_features(image_bgr)
        
        # Traditional facial_recognition method (backward compatibility)
        else:
            return self._extract_traditional_features(image_bgr)
    
    def _extract_traditional_features(self, image_bgr):
        """Traditional feature extraction using facial_recognition library"""
        try:
            rgb_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            return face_encodings[0] if face_encodings else None
        except Exception as e:
            logging.error(f"Traditional feature extraction error: {e}")
            return None
    
    def verify_faces(self, img1_bgr, img2_bgr, threshold=None):
        """
        Direct face verification between two images using ArcFace
        
        Returns:
            tuple: (verified: bool, confidence: float)
        """
        if threshold is None:
            threshold = self.model_config.get(self.model_name, {}).get("threshold", 0.30)
        
        # Use DeepFace verification with ArcFace
        if self.use_deepface:
            try:
                img1_rgb = cv.cvtColor(img1_bgr, cv.COLOR_BGR2RGB)
                img2_rgb = cv.cvtColor(img2_bgr, cv.COLOR_BGR2RGB)
                
                result = DeepFace.verify(
                    img1_path=img1_rgb,
                    img2_path=img2_rgb,
                    model_name=self.model_name,
                    detector_backend="opencv",  # Faster for verification
                    align=True,
                    enforce_detection=False
                )
                
                # Convert distance to confidence (1 - normalized distance)
                distance = result["distance"]
                confidence = max(0, 1 - (distance / (threshold * 2)))
                return result["verified"], confidence
                
            except Exception as e:
                logging.warning(f"ArcFace verification failed: {e}")
        
        # Fallback to traditional comparison
        emb1 = self.extract_features(img1_bgr)
        emb2 = self.extract_features(img2_bgr)
        
        if emb1 is None or emb2 is None:
            return False, 0.0
        
        # Calculate distance and convert to confidence
        distance = np.linalg.norm(emb1 - emb2)
        confidence = max(0, 1 - (distance / (threshold * 2)))
        verified = distance < threshold
        
        return verified, confidence

# ===========================
# CLASS: MultiFrameVerifier
# ===========================
class MultiFrameVerifier:
    """
    Implements voting system across multiple frames for reliable attendance marking
    """
    def __init__(self, confidence_threshold=0.30, min_votes=3):
        self.confidence_threshold = confidence_threshold
        self.min_votes = min_votes
        self.voting_history = deque(maxlen=20)
        
    def verify(self, frame_predictions):
        """
        Verify identity using voting system across multiple frames
        
        Args:
            frame_predictions: List of tuples (name, confidence) for each frame
            
        Returns:
            tuple: (verified_name, avg_confidence, success)
        """
        if not frame_predictions:
            return None, 0.0, False
        
        # Confidence Filtering
        high_conf_predictions = [
            (name, conf) for name, conf in frame_predictions 
            if conf >= self.confidence_threshold
        ]
        
        if not high_conf_predictions:
            return None, 0.0, False
        
        # Vote Counting
        from collections import Counter
        name_votes = Counter()
        confidence_sum = {}
        
        for name, conf in high_conf_predictions:
            name_votes[name] += 1
            if name not in confidence_sum:
                confidence_sum[name] = 0
            confidence_sum[name] += conf
        
        if not name_votes:
            return None, 0.0, False
        
        # Find Winner
        best_name, votes = name_votes.most_common(1)[0]
        
        # Verification Decision
        if votes >= self.min_votes:
            avg_confidence = confidence_sum[best_name] / votes
            
            # Store result
            self.voting_history.append({
                'name': best_name,
                'confidence': avg_confidence,
                'votes': votes,
                'total_frames': len(frame_predictions)
            })
            
            return best_name, avg_confidence, True
        
        return None, 0.0, False

# ===========================
# CLASS: FaceDatabase
# ===========================
class FaceDatabase:
    def __init__(self):
        self.conn = sqlite3.connect("database/attendance_system.db", check_same_thread=False)
        self._init_db()
        self._init_encryption()

    def _init_db(self):
        cursor = self.conn.cursor()
        # Students table
        cursor.execute('''CREATE TABLE IF NOT EXISTS students(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT NOT NULL,
                       student_id TEXT UNIQUE NOT NULL,
                       roll_number TEXT,
                       class TEXT,
                       model_type TEXT DEFAULT 'ArcFace',
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # Face encodings table
        cursor.execute('''CREATE TABLE IF NOT EXISTS face_encodings(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       student_id INTEGER NOT NULL,
                       encrypted_data BLOB NOT NULL,
                       encoding_dim INTEGER,
                       FOREIGN KEY (student_id) REFERENCES students(id))''')

        # Attendance records table
        cursor.execute('''CREATE TABLE IF NOT EXISTS attendance(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       student_id INTEGER NOT NULL,
                       date DATE NOT NULL,
                       time TIME NOT NULL,
                       status TEXT DEFAULT 'Present',
                       confidence FLOAT,
                       model_used TEXT,
                       FOREIGN KEY (student_id) REFERENCES students(id),
                       UNIQUE(student_id, date))''')

        self.conn.commit()

    def _init_encryption(self):
        key_path = "encryption_key/encrypted.key"
        if not os.path.exists(key_path):
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)

        with open(key_path, 'rb') as f:
            self.cipher = Fernet(f.read())

    def add_student(self, name, student_id, roll_number, class_name, face_encoding_arrays, model_type="ArcFace"):
        """
        Add a student with multiple face encodings
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO students (name, student_id, roll_number, class, model_type) 
                           VALUES (?, ?, ?, ?, ?)''',
                          (name, student_id, roll_number, class_name, model_type))
            db_student_id = cursor.lastrowid

            for enc in face_encoding_arrays:
                if isinstance(enc, np.ndarray):
                    enc_bytes = enc.tobytes()
                    encoding_dim = enc.shape[0]
                else:
                    enc_bytes = np.array(enc).tobytes()
                    encoding_dim = len(enc)
                
                encrypted_data = self.cipher.encrypt(enc_bytes)
                cursor.execute('''INSERT INTO face_encodings (student_id, encrypted_data, encoding_dim) 
                               VALUES (?, ?, ?)''',
                               (db_student_id, encrypted_data, encoding_dim))

            self.conn.commit()
            logging.info(f"Added student {name} (ID: {student_id}, Roll: {roll_number}, Class: {class_name}) with model: {model_type}")
            return True
        except sqlite3.IntegrityError:
            logging.error(f"Student ID {student_id} already exists")
            return False
        except sqlite3.Error as e:
            logging.error(f"Database Error: {e}")
            self.conn.rollback()
            return False

    def get_all_encodings(self):
        """
        Returns list of tuples: (student_id_str, name, roll_number, class, numpy_array_encoding)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT s.student_id, s.name, s.roll_number, s.class, fe.encrypted_data, s.model_type
                            FROM students s
                            JOIN face_encodings fe ON s.id = fe.student_id
                            ORDER BY fe.id ASC''')

            rows = cursor.fetchall()
            results = []
            for row in rows:
                student_id_str = row[0]
                name = row[1]
                roll_number = row[2]
                class_name = row[3]
                enc_bytes = self.cipher.decrypt(row[4])
                model_type = row[5]
                
                # Handle ArcFace embeddings
                enc = np.frombuffer(enc_bytes, dtype=np.float32)
                results.append((student_id_str, name, roll_number, class_name, enc))
            return results
        except sqlite3.Error as e:
            logging.error(f"Database Error in get_all_encodings: {e}")
            return []

    def mark_attendance(self, student_id, confidence, model_used):
        """
        Mark attendance for a student
        """
        try:
            cursor = self.conn.cursor()
            today = datetime.now().date()
            current_time = datetime.now().time()
            
            # Check if attendance already marked for today
            cursor.execute('''SELECT id FROM attendance 
                           WHERE student_id = (SELECT id FROM students WHERE student_id = ?) 
                           AND date = ?''', (student_id, today))
            
            if cursor.fetchone():
                logging.info(f"Attendance already marked for student {student_id} on {today}")
                return False
            
            # Insert attendance record
            cursor.execute('''INSERT INTO attendance (student_id, date, time, status, confidence, model_used)
                           VALUES ((SELECT id FROM students WHERE student_id = ?), date('now'), time('now'), ?, ?, ?)''',
                          (student_id, 'Present', confidence, model_used))
            
            self.conn.commit()
            logging.info(f"Attendance marked for student {student_id} at {current_time} with confidence {confidence}")
            return True
        except sqlite3.Error as e:
            logging.error(f"Database Error in mark_attendance: {e}")
            return False

    def get_todays_attendance(self):
        """
        Get today's attendance records
        """
        try:
            cursor = self.conn.cursor()
            today = datetime.now().date()
            
            cursor.execute('''SELECT s.name, s.student_id, s.roll_number, s.class, 
                           a.time, a.confidence, a.model_used
                           FROM attendance a
                           JOIN students s ON a.student_id = s.id
                           WHERE a.date = ?
                           ORDER BY a.time DESC''', (today,))
            
            return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Database Error in get_todays_attendance: {e}")
            return []

    def save_classifier(self, clf, encoder, model_type="ArcFace"):
        try:
            joblib.dump(clf, "database/classifier.pkl")
            joblib.dump(encoder, "database/label_encoder.pkl")
            with open("database/model_info.txt", "w") as f:
                f.write(f"model_type={model_type}\n")
            logging.info(f"Saved classifier for model: {model_type}")
        except Exception as e:
            logging.error(f"Error saving classifier: {e}")

    def load_classifier(self):
        try:
            if os.path.exists("database/classifier.pkl") and os.path.exists("database/label_encoder.pkl"):
                clf = joblib.load("database/classifier.pkl")
                encoder = joblib.load("database/label_encoder.pkl")
                
                model_type = "ArcFace"
                if os.path.exists("database/model_info.txt"):
                    with open("database/model_info.txt", "r") as f:
                        for line in f:
                            if line.startswith("model_type="):
                                model_type = line.strip().split("=")[1]
                
                logging.info(f"Loaded classifier, model: {model_type}")
                return clf, encoder, model_type
        except Exception as e:
            logging.error(f"Error loading classifier: {e}")
        return None, None, "ArcFace"

# ===========================
# CLASS: FaceCapture
# ===========================
class FaceCapture:
    """
    Simple face capture for attendance system
    """
    def __init__(self):
        self.cap = None
    
    def capture_faces(self, num_faces=10, show_preview=True):
        """
        Capture multiple face images for registration
        """
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        face_images = []
        count = 0
        
        print(f"Look at the camera. Capturing {num_faces} face images...")
        print("Press 'q' to quit early")
        
        while count < num_faces:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Convert to RGB for face detection
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Detect faces using facial_recognition library
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                # Take the first face found
                top, right, bottom, left = face_locations[0]
                face_img = frame[top:bottom, left:right]
                
                # Basic quality check
                if face_img.shape[0] > 50 and face_img.shape[1] > 50:
                    face_images.append(face_img)
                    count += 1
                    
                    # Draw rectangle and counter
                    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv.putText(frame, f"Captured: {count}/{num_faces}", 
                              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if show_preview:
                cv.imshow("Face Capture - Press 'q' to quit", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv.destroyAllWindows()
        
        print(f"Captured {len(face_images)} face images")
        return face_images
    
    def capture_single_face_for_attendance(self, timeout=10):
        """
        Capture a single face for attendance marking
        """
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            return None
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Convert to RGB for face detection
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_img = frame[top:bottom, left:right]
                
                # Basic quality check
                if face_img.shape[0] > 50 and face_img.shape[1] > 50:
                    self.cap.release()
                    cv.destroyAllWindows()
                    return face_img
            
            # Show preview
            cv.imshow("Face Recognition - Press 'q' to quit", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv.destroyAllWindows()
        return None

# ===========================
# CLASS: AttendanceSystem
# ===========================
class AttendanceSystem:
    """
    Main attendance system with face recognition using ArcFace
    """
    def __init__(self, use_deepface=True):
        try:
            self.face_db = FaceDatabase()
            
            # Face recognizer - Fixed to use ArcFace only
            self.face_recognizer = RobustFaceRecognizer(
                use_deepface=use_deepface
            )
            
            # Multi-frame verifier
            self.verifier = MultiFrameVerifier(
                confidence_threshold=0.30,  # ArcFace threshold
                min_votes=3
            )
            
            # Face capture
            self.face_capture = FaceCapture()
            
            # Known face info
            self.known_encodings = []
            self.known_names = []
            self.known_student_ids = []
            self.name_to_student_id = {}
            
            # Model info
            self.current_model = "ArcFace"  # Fixed to ArcFace
            self.use_deepface = use_deepface
            
            self._load_known_faces()
            logging.info(f"Attendance System initialized with ArcFace model")
            
        except Exception as e:
            logging.error(f"Error initializing AttendanceSystem: {e}")
            raise
    
    def _load_known_faces(self):
        try:
            records = self.face_db.get_all_encodings()
            self.known_encodings = [enc for _, _, _, _, enc in records]
            self.known_names = [name for _, name, _, _, _ in records]
            self.known_student_ids = [student_id for student_id, _, _, _, _ in records]
            self.name_to_student_id = {name: student_id for student_id, name, _, _, _ in records}
            logging.info(f"Loaded {len(self.known_encodings)} face encodings for {len(set(self.known_names))} students")
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")
    
    def extract_features(self, image):
        """
        Extract face features using ArcFace
        """
        if image is None or image.size == 0:
            return None
        
        # Use ArcFace recognizer
        return self.face_recognizer.extract_features(image)
    
    def train_classifier(self):
        """
        Train an SVM classifier on the face encodings
        """
        try:
            records = self.face_db.get_all_encodings()
            if not records:
                print("No students in database to train classifier.")
                return

            X = []
            y = []
            for _, name, _, _, enc in records:
                try:
                    enc = np.array(enc).flatten()
                    X.append(enc)
                    y.append(name)
                except Exception as e:
                    logging.error(f"Error processing encoding for {name}: {e}")

            if not X:
                print("No encodings available for training.")
                return

            print(f"Training classifier with {len(X)} samples from {len(set(y))} students")

            if len(set(y)) < 2:
                print("At least two students are required to train the classifier.")
                return

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)

            # Use RBF kernel for ArcFace embeddings
            clf = SVC(kernel="rbf", probability=True, class_weight='balanced')

            clf.fit(X, y_encoded)

            self.face_db.save_classifier(clf, encoder, self.current_model)
            print(f"Classifier trained successfully with ArcFace.")
        except Exception as e:
            logging.error(f"Error training classifier: {e}")
            print("Classifier training failed.")
    
    def register_student(self):
        """
        Register a new student
        """
        print("\n=== Student Registration ===")
        name = input("Enter student name: ").strip()
        student_id = input("Enter student ID: ").strip()
        roll_number = input("Enter roll number: ").strip()
        class_name = input("Enter class: ").strip()

        if not all([name, student_id, roll_number, class_name]):
            print("All fields are required.")
            return

        # Capture face images
        print(f"\nLook at the camera. Capturing 10 face images using ArcFace...")
        face_images = self.face_capture.capture_faces(num_faces=10, show_preview=True)
        
        if len(face_images) < 5:
            print(f"Failed to capture enough valid images. Got {len(face_images)} images.")
            return

        # Extract encodings with ArcFace
        successful_encodings = []
        for img in face_images:
            features = self.extract_features(img)
            if features is not None:
                successful_encodings.append(np.array(features).flatten())
            else:
                print("Warning: Face not detected in one of the images.")

        if len(successful_encodings) < 3:
            print(f"Insufficient valid face samples (minimum 3 required, got {len(successful_encodings)}).")
            return

        # Add to database
        added = self.face_db.add_student(
            name=name,
            student_id=student_id,
            roll_number=roll_number,
            class_name=class_name,
            face_encoding_arrays=successful_encodings,
            model_type=self.current_model
        )
        
        if not added:
            print("Failed to add student (student ID might already exist).")
            return

        self._load_known_faces()
        self.train_classifier()
        print(f"\n✅ Student {name} registered successfully with ArcFace!")
    
    def mark_attendance_realtime(self):
        """
        Mark attendance using real-time face recognition with ArcFace
        """
        print("\n=== Mark Attendance ===")
        print("Using ArcFace for recognition...")
        print("Looking for faces...")
        
        # Capture multiple frames for reliable recognition
        frame_predictions = []
        captured_frames = []
        
        # Capture for 5 seconds
        start_time = time.time()
        cap = cv.VideoCapture(0)
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to RGB for face detection
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_img = frame[top:bottom, left:right]
                
                if face_img.shape[0] > 50 and face_img.shape[1] > 50:
                    captured_frames.append(face_img)
                    
                    # Draw rectangle on frame
                    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Display
            cv.imshow("Attendance - Press 'q' to quit", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv.destroyAllWindows()
        
        if len(captured_frames) < 3:
            print("Not enough face frames captured.")
            return
        
        print(f"Captured {len(captured_frames)} frames. Processing with ArcFace...")
        
        # Load classifier
        clf, encoder, model_type = self.face_db.load_classifier()
        if clf is None or encoder is None:
            print("Classifier not trained. Please register students first.")
            return
        
        # Process each frame with ArcFace
        for i, face_img in enumerate(captured_frames):
            features = self.extract_features(face_img)
            if features is None:
                continue
            
            try:
                features_arr = np.array(features).flatten().reshape(1, -1)
                probabilities = clf.predict_proba(features_arr)
                
                if probabilities.size > 0:
                    predicted_class = np.argmax(probabilities)
                    max_prob = float(np.max(probabilities))
                    name = encoder.inverse_transform([predicted_class])[0]
                    
                    frame_predictions.append((name, max_prob))
                    
            except Exception as e:
                logging.error(f"Frame prediction error: {e}")
        
        if not frame_predictions:
            print("No valid predictions made.")
            return
        
        # Multi-frame verification
        verified_name, avg_confidence, verification_success = self.verifier.verify(frame_predictions)
        
        if not verification_success:
            print("Attendance verification failed.")
            return
        
        print(f"\n✅ Verified: {verified_name}")
        print(f"Confidence: {avg_confidence:.3f}")
        
        # Get student ID
        student_id = self.name_to_student_id.get(verified_name)
        if student_id:
            # Mark attendance
            success = self.face_db.mark_attendance(
                student_id=student_id,
                confidence=avg_confidence,
                model_used=self.current_model
            )
            
            if success:
                print(f"Attendance marked for {verified_name} using ArcFace")
            else:
                print("Attendance already marked for today.")
        else:
            print("Student ID not found.")
    
    def view_todays_attendance(self):
        """
        View today's attendance records
        """
        print("\n=== Today's Attendance ===")
        records = self.face_db.get_todays_attendance()
        
        if not records:
            print("No attendance records for today.")
            return
        
        print(f"\n{'Name':<20} {'ID':<15} {'Roll No':<10} {'Class':<10} {'Time':<10} {'Confidence':<10} {'Model':<10}")
        print("=" * 95)
        
        for record in records:
            name, student_id, roll_number, class_name, time_str, confidence, model_used = record
            time_only = time_str.split('.')[0] if '.' in str(time_str) else str(time_str)
            print(f"{name:<20} {student_id:<15} {roll_number:<10} {class_name:<10} {time_only:<10} {confidence:.3f} {'ArcFace':<10}")

# ===========================
# Main Application
# ===========================
if __name__ == '__main__':
    try:
        print("\n" + "=" * 60)
        print("FACE RECOGNITION ATTENDANCE SYSTEM")
        print("=" * 60)
        print("Using ArcFace for face recognition")
        print("ArcFace provides highest accuracy for face recognition")
        print("-" * 60)
        
        if not DEEPFACE_AVAILABLE:
            print("Warning: DeepFace is not installed. ArcFace requires DeepFace.")
            print("Please install with: pip install deepface")
            use_deepface = False
        else:
            use_deepface = True
        
        # Initialize system with ArcFace
        system = AttendanceSystem(use_deepface=use_deepface)

        while True:
            print("\n" + "=" * 50)
            print("ATTENDANCE SYSTEM (ArcFace)")
            print("=" * 50)
            print("1. Register New Student")
            print("2. Mark Attendance (Real-time)")
            print("3. View Today's Attendance")
            print("4. Train Classifier")
            print("5. View System Status")
            print("6. Exit")
            print("-" * 50)

            choice = input("Select Option (1-6): ").strip()

            try:
                if choice == "1":
                    system.register_student()
                elif choice == "2":
                    system.mark_attendance_realtime()
                elif choice == "3":
                    system.view_todays_attendance()
                elif choice == "4":
                    system.train_classifier()
                elif choice == "5":
                    print("\n=== System Status ===")
                    print(f"Registered Students: {len(set(system.known_names))}")
                    print(f"Total Face Samples: {len(system.known_encodings)}")
                    print(f"Recognition Model: {system.current_model}")
                    print(f"Using DeepFace: {system.use_deepface}")
                    print(f"Verification Threshold: {system.verifier.confidence_threshold:.3f}")
                    print("=" * 20)
                elif choice == "6":
                    print("Attendance System shutting down...")
                    break
                else:
                    print("Invalid Choice.")
            except Exception as e:
                logging.error(f"Menu operation error: {e}")
                print(f"An error occurred: {e}")

        print("System shutdown complete.")
    except Exception as e:
        logging.error(f"System startup error: {e}")
        print("System failed to start. Check logs for details.")