# app.py - Core Attendance System Logic
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import sqlite3
import json

# Add current directory to path for imports
sys.path.append('.')

# Import modules
try:
    from FaceRecognition import AttendanceSystem, FaceCapture
    face_modules_available = True
except ImportError as e:
    print(f"Warning: Failed to import FaceRecognition module: {e}")
    face_modules_available = False

try:
    from VoiceRecognition import VoiceVerifier, VoiceRecorder
    voice_modules_available = True
except ImportError as e:
    print(f"Warning: Failed to import VoiceRecognition module: {e}")
    voice_modules_available = False

class CoreAttendanceSystem:
    def __init__(self):
        """Initialize core attendance system"""
        self.face_system = None
        self.voice_system = None
        self.initialize_systems()
        
        # Registration state
        self.registration_data = {}
        self.captured_images = []
        
        # Authentication state
        self.auth_result = {}
        
        # Check/create required directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create required directories if they don't exist"""
        required_dirs = ["database", "logs", "encryption_key", "temp", "models"]
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
        
        # Initialize database if needed
        if not os.path.exists("database/attendance_system.db"):
            print("First-time setup: Initializing database...")
            self.initialize_database()
    
    def initialize_database(self):
        """Initialize database schema"""
        try:
            conn = sqlite3.connect("database/attendance_system.db")
            cursor = conn.cursor()
            
            # Create students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    student_id TEXT UNIQUE NOT NULL,
                    roll_number TEXT,
                    class TEXT,
                    face_encoding_path TEXT,
                    model_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create attendance table
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

            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS voice_verifications(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    verification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence FLOAT,
                    success BOOLEAN,
                    method TEXT,
                    FOREIGN KEY (student_id) REFERENCES students(student_id))
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS voice_enrollment(
                    student_id TEXT PRIMARY KEY,
                    enrollment_status TEXT DEFAULT 'pending',
                    voice_samples_count INTEGER DEFAULT 0,
                    last_enrollment TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id))
            ''')
            
            conn.commit()
            conn.close()
            print("SUCCESS: Database initialized successfully!")
            
        except Exception as e:
            print(f"ERROR: Database initialization failed: {e}")
    
    def initialize_systems(self):
        """Initialize face and voice recognition systems"""
        try:
            if face_modules_available:
                self.face_system = AttendanceSystem(use_deepface=True)
                print("SUCCESS: Face recognition system initialized")
            else:
                print("ERROR: Face recognition system not available")
            
            if voice_modules_available:
                self.voice_system = VoiceVerifier()
                print("SUCCESS: Voice recognition system initialized")
            else:
                print("ERROR: Voice recognition system not available")
                
        except Exception as e:
            print(f"Error initializing systems: {e}")
    
    def register_student(self, name, student_id, roll_number, class_name):
        """Register a new student with face and voice data"""
        print(f"\n=== Starting Registration for {name} ===")
        
        self.registration_data = {
            'name': name,
            'student_id': student_id,
            'roll_number': roll_number,
            'class_name': class_name
        }
        
        # Step 1: Face capture
        face_success = self.register_face()
        if not face_success:
            print("ERROR: Face registration failed")
            return False
        
        # Step 2: Voice enrollment (optional)
        voice_success = self.register_voice()
        
        if face_success:
            print(f"\nSUCCESS: Registration completed successfully for {name}!")
            if voice_success:
                print("   - Face + Voice registration")
            else:
                print("   - Face-only registration")
            return True
        
        return False
    
    def register_face(self):
        """Register face data for student"""
        if not self.face_system or not face_modules_available:
            print("ERROR: Face system not available")
            return False
        
        print("\nStep 1: Face Registration")
        print("Please look directly at the camera...")
        
        try:
            # Capture face images
            face_capture = FaceCapture()
            captured_images = face_capture.capture_faces(num_faces=10, show_preview=False)
            
            if not captured_images or len(captured_images) < 5:
                print(f"ERROR: Only captured {len(captured_images) if captured_images else 0} images. Need at least 5.")
                return False
            
            print(f"SUCCESS: Captured {len(captured_images)} face images")
            
            # Extract face features
            successful_encodings = []
            for i, img in enumerate(captured_images):
                features = self.face_system.extract_features(img)
                if features is not None:
                    if not isinstance(features, np.ndarray):
                        features = np.array(features)
                    successful_encodings.append(features.flatten())
                print(f"  Extracted features: {i+1}/{len(captured_images)}")
            
            if len(successful_encodings) < 3:
                print(f"ERROR: Only extracted {len(successful_encodings)} features. Need at least 3.")
                return False
            
            print(f"SUCCESS: Extracted {len(successful_encodings)} face encodings")
            
            # Save to database
            added = self.face_system.face_db.add_student(
                name=self.registration_data['name'],
                student_id=self.registration_data['student_id'],
                roll_number=self.registration_data['roll_number'],
                class_name=self.registration_data['class_name'],
                face_encoding_arrays=successful_encodings,
                model_type="ArcFace"
            )
            
            if not added:
                print("ERROR: Failed to save to database. Student ID might already exist.")
                return False
            
            # Train classifier
            self.face_system.train_classifier()
            print("SUCCESS: Face classifier trained")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Face registration failed: {e}")
            return False
    
    def register_voice(self):
        """Register voice data for student (optional)"""
        if not self.voice_system or not voice_modules_available:
            print("WARNING: Voice system not available - skipping voice registration")
            return False
        
        print("\nStep 2: Voice Registration (Optional)")
        
        try:
            success = self.voice_system.enroll_student(self.registration_data['student_id'])
            if success:
                print("SUCCESS: Voice enrollment completed successfully!")
                return True
            else:
                print("WARNING: Voice enrollment failed or was skipped")
                return False
                
        except Exception as e:
            print(f"WARNING: Voice registration error: {e}")
            return False
    
    def mark_attendance(self):
        """Mark attendance using simultaneous face and voice recognition"""
        print("\n=== Starting Attendance Marking ===")
        
        if not self.face_system:
            print("ERROR: Face system not available")
            return False
        
        import threading
        import queue
        import time
        from datetime import datetime
        
        # Results queues for multi-threading
        face_result_queue = queue.Queue()
        voice_result_queue = queue.Queue()
        
        # Shared state
        stop_event = threading.Event()
        
        # Store final results
        face_result = {'success': False, 'name': None, 'student_id': None, 'confidence': 0, 'features': None, 'error': None}
        voice_result = {'success': False, 'student_id': None, 'confidence': 0, 'error': None}
        
        def capture_and_process_face():
            """Capture and process face in a separate thread (no GUI version)"""
            try:
                # Setup video capture without GUI
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if not cap.isOpened():
                    face_result_queue.put({'success': False, 'error': 'Camera not available'})
                    return
                
                # Load classifier once
                clf, encoder, model_type = self.face_system.face_db.load_classifier()
                if clf is None or encoder is None:
                    face_result_queue.put({'success': False, 'error': 'Classifier not trained'})
                    return
                
                best_face = {'confidence': 0, 'name': None, 'student_id': None, 'features': None}
                start_time = time.time()
                timeout = 10  # 10 seconds timeout
                
                print("\n[Face] Camera active - please look at the camera")
                
                # Simple text-based progress indicator
                dots = 0
                while time.time() - start_time < timeout and not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Detect faces (simplified, no GUI)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Process each face
                    for (x, y, w, h) in faces:
                        # Extract face ROI and process
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            features = self.face_system.extract_features(face_roi)
                            
                            if features is not None:
                                features_arr = np.array(features).flatten().reshape(1, -1)
                                probabilities = clf.predict_proba(features_arr)
                                
                                if probabilities.size > 0:
                                    predicted_class = np.argmax(probabilities)
                                    confidence = float(np.max(probabilities))
                                    name = encoder.inverse_transform([predicted_class])[0]
                                    student_id = self.face_system.name_to_student_id.get(name)
                                    
                                    # Update best match if confidence is higher
                                    if confidence > best_face['confidence'] and confidence > 0.3:
                                        best_face['confidence'] = confidence
                                        best_face['name'] = name
                                        best_face['student_id'] = student_id
                                        best_face['features'] = features
                                        print(f"\r[Face] Detected: {name} (Confidence: {confidence:.3f})", end='', flush=True)
                    
                    # Simple progress indicator
                    if len(faces) == 0:
                        dots = (dots + 1) % 4
                        print(f"\r[Face] Looking for face{'.' * dots}{' ' * (3-dots)}", end='', flush=True)
                    
                    time.sleep(0.1)  # Small delay to prevent CPU overload
                
                print()  # New line after progress indicator
                cap.release()
                
                # Put best result in queue
                if best_face['confidence'] > 0.3:
                    face_result_queue.put({
                        'success': True,
                        'name': best_face['name'],
                        'student_id': best_face['student_id'],
                        'confidence': best_face['confidence'],
                        'features': best_face['features']
                    })
                else:
                    face_result_queue.put({'success': False, 'error': 'No face recognized with sufficient confidence'})
                    
            except Exception as e:
                face_result_queue.put({'success': False, 'error': str(e)})
        
        def capture_and_process_voice():
            """Capture and process voice in a separate thread"""
            try:
                if not self.voice_system or not voice_modules_available:
                    voice_result_queue.put({'success': False, 'error': 'Voice system not available'})
                    return
                
                import pyaudio
                import wave
                import numpy as np
                
                print("\n[Voice] Listening for your name...")
                
                # Audio recording parameters
                CHUNK = 1024
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 16000
                RECORD_SECONDS = 5  # Record for 5 seconds
                
                # Initialize PyAudio
                p = pyaudio.PyAudio()
                
                # Open stream
                stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
                
                # Record audio with live level monitoring
                frames = []
                max_amplitude = 0
                
                print("[Voice] Recording... (say your name clearly)")
                
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    if stop_event.is_set():
                        break
                        
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Calculate audio level for visual feedback
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    amplitude = np.abs(audio_data).mean()
                    max_amplitude = max(max_amplitude, amplitude)
                    
                    # Simple audio level indicator
                    level = int(amplitude / 500)  # Adjust scaling as needed
                    if level > 0:
                        bar = '■' * min(level, 50)
                        print(f"\r[Voice] Level: {bar:<50}", end='', flush=True)
                
                print()  # New line after progress indicator
                
                # Stop and close stream
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                # Check if we actually got audio
                if max_amplitude < 100:  # Threshold for silence
                    voice_result_queue.put({'success': False, 'error': 'No voice detected'})
                    return
                
                print("[Voice] Processing...")
                
                # Save temporary audio file
                temp_audio_path = f"temp/voice_{int(time.time())}.wav"
                os.makedirs("temp", exist_ok=True)
                
                wf = wave.open(temp_audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # Verify voice against all enrolled students
                success, student_id, confidence = self.verify_voice_against_all(temp_audio_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
                
                if success:
                    voice_result_queue.put({
                        'success': True,
                        'student_id': student_id,
                        'confidence': confidence
                    })
                else:
                    voice_result_queue.put({'success': False, 'error': 'Voice verification failed'})
                    
            except Exception as e:
                voice_result_queue.put({'success': False, 'error': str(e)})
        
        # Start both threads
        face_thread = threading.Thread(target=capture_and_process_face)
        voice_thread = threading.Thread(target=capture_and_process_voice)
        
        face_thread.start()
        voice_thread.start()
        
        # Wait for results with timeout
        timeout = 20  # Total timeout seconds
        start_time = time.time()
        
        face_done = False
        voice_done = False
        
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            
            # Check face queue
            if not face_done and not face_result_queue.empty():
                face_result = face_result_queue.get()
                face_done = True
                if face_result['success']:
                    print(f"\n[Face] Recognized: {face_result['name']} (Confidence: {face_result['confidence']:.3f})")
                else:
                    print(f"\n[Face] {face_result.get('error', 'Failed')}")
            
            # Check voice queue
            if not voice_done and not voice_result_queue.empty():
                voice_result = voice_result_queue.get()
                voice_done = True
                if voice_result['success']:
                    print(f"[Voice] Verified student ID: {voice_result['student_id']} (Confidence: {voice_result['confidence']:.3f})")
                else:
                    print(f"[Voice] {voice_result.get('error', 'Failed')}")
            
            if face_done and voice_done:
                break
        
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish
        face_thread.join(timeout=2)
        voice_thread.join(timeout=2)
        
        # Fuse results
        print("\n=== Fusion Results ===")

        # Convert voice_result student_id to string if it's an integer
        if voice_result['success'] and voice_result['student_id'] is not None:
            voice_student_id = str(voice_result['student_id'])
        else:
            voice_student_id = None

        # Track the best result across both modalities
        best_result = {
            'success': False,
            'confidence': 0,
            'student_id': None,
            'name': None,
            'method': None,
            'face_confidence': 0,
            'voice_confidence': 0
        }

        # Check face result
        if face_result['success'] and face_result['confidence'] > 0.3:
            best_result['success'] = True
            best_result['confidence'] = face_result['confidence']
            best_result['student_id'] = face_result['student_id']
            best_result['name'] = face_result['name']
            best_result['method'] = "Face Only"
            best_result['face_confidence'] = face_result['confidence']
            best_result['voice_confidence'] = 0

        # Check voice result
        if voice_result['success'] and voice_result['confidence'] > 0.15:  # Lower threshold for voice
            # If face also succeeded, compare confidences
            if best_result['success']:
                # Compare confidences - use the higher one
                if voice_result['confidence'] > best_result['confidence']:
                    print(f"⚠ Voice ({voice_result['confidence']:.3f}) has higher confidence than Face ({best_result['confidence']:.3f}) - using voice")
                    best_result['confidence'] = voice_result['confidence']
                    best_result['student_id'] = voice_student_id
                    best_result['name'] = self.get_student_name(voice_student_id)
                    best_result['method'] = "Voice Only (Higher Confidence)"
                else:
                    print(f"✓ Face ({best_result['confidence']:.3f}) has higher confidence than Voice ({voice_result['confidence']:.3f}) - using face")
                    best_result['method'] = "Face Only (Higher Confidence)"
                
                # Store both confidences for comparison
                best_result['face_confidence'] = face_result['confidence'] if face_result['success'] else 0
                best_result['voice_confidence'] = voice_result['confidence']
            else:
                # Only voice succeeded
                best_result['success'] = True
                best_result['confidence'] = voice_result['confidence']
                best_result['student_id'] = voice_student_id
                best_result['name'] = self.get_student_name(voice_student_id)
                best_result['method'] = "Voice Only"
                best_result['face_confidence'] = 0
                best_result['voice_confidence'] = voice_result['confidence']

        # Also check if face and voice agree but voice has lower confidence
        if (face_result['success'] and voice_result['success'] and 
            face_result['student_id'] == voice_student_id and
            face_result['confidence'] > voice_result['confidence']):
            # They agree! Boost confidence slightly
            best_result['confidence'] = face_result['confidence']  # Use face confidence
            best_result['method'] = "Face + Voice (Agree)"
            best_result['face_confidence'] = face_result['confidence']
            best_result['voice_confidence'] = voice_result['confidence']
            print(f"✓ Face and voice both identify {best_result['name']} - using face confidence ({face_result['confidence']:.3f})")

        # Final decision
        if not best_result['success']:
            print("✗ Both face and voice recognition failed")
            return False

        # Display fusion results
        print(f"\nFusion Decision:")
        print(f"  - Selected: {best_result['method']}")
        print(f"  - Student: {best_result['name']} (ID: {best_result['student_id']})")
        print(f"  - Confidence: {best_result['confidence']:.3f}")
        if best_result['face_confidence'] > 0:
            print(f"  - Face confidence: {best_result['face_confidence']:.3f}")
        if best_result['voice_confidence'] > 0:
            print(f"  - Voice confidence: {best_result['voice_confidence']:.3f}")

        # Mark attendance
        print("\n=== Marking Attendance ===")

        success = self.face_system.face_db.mark_attendance(
            student_id=best_result['student_id'],
            confidence=best_result['confidence'],
            model_used=best_result['method']
        )

        if success:
            print(f"\n✓ ATTENDANCE MARKED SUCCESSFULLY!")
            print(f"  Student: {best_result['name']}")
            print(f"  Student ID: {best_result['student_id']}")
            print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"  Method: {best_result['method']}")
            print(f"  Confidence: {best_result['confidence']:.3f}")
            return True
        else:
            print("✗ Failed to mark attendance (already marked for today?)")
            return False

    def verify_voice_against_all(self, audio_path):
        """Verify voice against all enrolled students using ECAPA-TDNN"""
        try:
            conn = sqlite3.connect("database/attendance_system.db")
            cursor = conn.cursor()
            
            # Check if voice_samples table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='voice_samples'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return False, None, 0.0
            
            # Get all students with voice embeddings
            cursor.execute("""
                SELECT s.student_id, v.voice_embedding 
                FROM voice_samples v
                JOIN students s ON v.student_id = s.student_id
                WHERE v.voice_embedding IS NOT NULL
            """)
            
            samples = cursor.fetchall()
            conn.close()
            
            if not samples:
                print("[Voice] No enrolled voice samples found")
                return False, None, 0.0
            
            # Extract features from new audio
            if not hasattr(self.voice_system, 'feature_extractor'):
                print("[Voice] Feature extractor not available")
                return False, None, 0.0
                
            # Load audio file
            import librosa
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Extract embedding
            new_embedding = self.voice_system.feature_extractor.extract_embedding(audio_data, sr)
            
            if new_embedding is None:
                print("[Voice] Failed to extract features from audio")
                return False, None, 0.0
            
            # Determine embedding type based on shape
            embedding_dim = len(new_embedding)
            print(f"[Voice] Extracted embedding dimension: {embedding_dim}")
            
            all_similarities = []  # Will store all similarity scores
            
            for student_id, embedding_blob in samples:
                try:
                    # Load stored embedding
                    stored_features = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Handle different embedding dimensions
                    if embedding_dim != len(stored_features):
                        print(f"[Voice] Dimension mismatch: new={embedding_dim}, stored={len(stored_features)}")
                        
                        # If dimensions differ, use the smaller dimension for comparison
                        min_dim = min(embedding_dim, len(stored_features))
                        
                        # For ECAPA (192) vs MFCC (240), we can take the first 192 dimensions of MFCC
                        if embedding_dim == 192 and len(stored_features) == 240:
                            # New is ECAPA, stored is MFCC - use first 192 of MFCC
                            stored_features = stored_features[:192]
                            min_dim = 192
                        elif embedding_dim == 240 and len(stored_features) == 192:
                            # New is MFCC, stored is ECAPA - use ECAPA dimensions
                            new_embedding_trimmed = new_embedding[:192]
                            stored_features = stored_features
                            min_dim = 192
                        else:
                            # Generic case - use min dimension
                            min_dim = min(embedding_dim, len(stored_features))
                            new_embedding = new_embedding[:min_dim]
                            stored_features = stored_features[:min_dim]
                    else:
                        min_dim = embedding_dim
                    
                    if min_dim < 10:  # Too small to compare
                        continue
                    
                    # Calculate raw cosine similarity (store for AS-norm)
                    new_norm = new_embedding[:min_dim] / (np.linalg.norm(new_embedding[:min_dim]) + 1e-10)
                    stored_norm = stored_features[:min_dim] / (np.linalg.norm(stored_features[:min_dim]) + 1e-10)

                    similarity = np.dot(new_norm, stored_norm)
                    similarity = max(min(similarity, 1.0), -1.0)

                    # Store ALL similarities for later AS-norm calculation
                    # We'll store them in a list to process after collecting all scores
                    if 'all_similarities' not in locals():
                        all_similarities = []
                    all_similarities.append((student_id, similarity, embedding_dim, stored_features))
                            
                except Exception as e:
                    print(f"[Voice] Error comparing with sample: {e}")
                    continue
            
            # Process all similarities with AS-norm
            if all_similarities:
                # Sort by similarity to find best candidate
                all_similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Best candidate is first
                best_student_id = all_similarities[0][0]
                best_raw_score = all_similarities[0][1]
                best_embedding_dim = all_similarities[0][2]
                
                # Separate target vs impostor scores for AS-norm
                target_scores = []
                impostor_scores = []
                
                for student_id, score, emb_dim, _ in all_similarities:
                    if student_id == best_student_id:
                        target_scores.append(score)
                    else:
                        impostor_scores.append(score)
                
                # Apply AS-norm if we have enough impostor scores
                if len(impostor_scores) >= 2:  # Minimum cohort size
                    cohort_mean = np.mean(impostor_scores)
                    cohort_std = np.std(impostor_scores)
                    
                    if cohort_std > 0.05:
                        # AS-norm formula
                        as_norm_score = (best_raw_score - cohort_mean) / cohort_std
                        # Convert to [0,1] range for consistency
                        final_confidence = 0.5 * (as_norm_score / 3.0 + 1)
                        final_confidence = max(0.0, min(1.0, final_confidence))
                        
                        print(f"[Voice] AS-norm applied:")
                        print(f"  - Raw score: {best_raw_score:.3f}")
                        print(f"  - Cohort mean: {cohort_mean:.3f}")
                        print(f"  - Cohort std: {cohort_std:.3f}")
                        print(f"  - AS-norm score: {as_norm_score:.3f}")
                        print(f"  - Final confidence: {final_confidence:.3f}")
                    else:
                        final_confidence = best_raw_score
                        print(f"[Voice] Using direct score (cohort std=0): {final_confidence:.3f}")
                else:
                    final_confidence = best_raw_score
                    print(f"[Voice] Insufficient cohort samples ({len(impostor_scores)}). Using direct score: {final_confidence:.3f}")
                
                # Apply threshold (adjusted for AS-norm)
                if best_embedding_dim == 192:  # ECAPA
                    threshold = 0.55  # Higher threshold for AS-norm normalized scores
                else:  # MFCC
                    threshold = 0.50  # Adjusted threshold for MFCC with AS-norm
                
                if final_confidence > threshold:
                    print(f"[Voice] Match found: {best_student_id} with confidence {final_confidence:.3f}")
                    return True, best_student_id, final_confidence
                else:
                    print(f"[Voice] No match above threshold. Best confidence: {final_confidence:.3f}")
                    return False, None, final_confidence
            else:
                return False, None, 0.0
                
        except Exception as e:
            print(f"[Voice] Verification error: {e}")
            import traceback
            traceback.print_exc()
            return False, None, 0.0

    def get_student_name(self, student_id):
        """Get student name from student_id"""
        try:
            conn = sqlite3.connect("database/attendance_system.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM students WHERE student_id = ?", (student_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else "Unknown"
        except:
            return "Unknown"
    
    def get_attendance_report(self, date=None):
        """Get attendance report for a specific date or all records"""
        try:
            conn = sqlite3.connect("database/attendance_system.db")
            
            if date:
                query = """
                SELECT s.name, s.student_id, s.roll_number, s.class, 
                       a.time, a.confidence, a.model_used
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                WHERE a.date = ?
                ORDER BY a.time DESC
                """
                df = pd.read_sql_query(query, conn, params=(date,))
            else:
                query = """
                SELECT s.name, s.student_id, s.roll_number, s.class, 
                       a.date, a.time, a.confidence, a.model_used
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                ORDER BY a.date DESC, a.time DESC
                LIMIT 100
                """
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error getting attendance report: {e}")
            return pd.DataFrame()
    
    def get_registered_students(self):
        """Get list of all registered students"""
        try:
            conn = sqlite3.connect("database/attendance_system.db")
            query = """
            SELECT name, student_id, roll_number, class, created_at
            FROM students
            ORDER BY created_at DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error getting students: {e}")
            return pd.DataFrame()
    
    def retrain_classifier(self):
        """Retrain the face classifier"""
        if not self.face_system:
            print("ERROR: Face system not available")
            return False
        
        try:
            print("Retraining classifier...")
            self.face_system.train_classifier()
            print("SUCCESS: Classifier retrained successfully!")
            return True
        except Exception as e:
            print(f"ERROR: Error retraining classifier: {e}")
            return False
    
    def backup_database(self):
        """Create a backup of the database"""
        try:
            import shutil
            from datetime import datetime
            
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{backup_dir}/attendance_backup_{timestamp}.db"
            
            shutil.copy2("database/attendance_system.db", backup_file)
            print(f"SUCCESS: Database backed up to: {backup_file}")
            return True
            
        except Exception as e:
            print(f"ERROR: Backup failed: {e}")
            return False
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            conn = sqlite3.connect("database/attendance_system.db")
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM students")
            student_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM attendance")
            attendance_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
            days_count = cursor.fetchone()[0]
            
            # Check if voice enrollment table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='voice_enrollment'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM voice_enrollment WHERE enrollment_status = 'completed'")
                voice_enrolled = cursor.fetchone()[0]
            else:
                voice_enrolled = 0
            
            conn.close()
            
            stats = {
                'total_students': student_count,
                'total_attendance': attendance_count,
                'days_tracked': days_count,
                'voice_enrolled': voice_enrolled
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

def test_system():
    """Test the system with interactive prompts"""
    print("=== Multi-Modal Attendance System Test ===")
    
    # Initialize system
    system = CoreAttendanceSystem()
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Register New Student")
        print("2. Mark Attendance")
        print("3. View Today's Attendance Report")
        print("4. View All Registered Students")
        print("5. Get System Statistics")
        print("6. Retrain Classifier")
        print("7. Backup Database")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            # Register student
            print("\n--- Student Registration ---")
            name = input("Enter student name: ").strip()
            student_id = input("Enter student ID: ").strip()
            roll_number = input("Enter roll number: ").strip()
            class_name = input("Enter class/section: ").strip()
            
            if name and student_id:
                confirm = input(f"\nRegister {name} (ID: {student_id})? (y/n): ").lower()
                if confirm == 'y':
                    success = system.register_student(name, student_id, roll_number, class_name)
                    if success:
                        input("\nPress Enter to continue...")
                else:
                    print("Registration cancelled.")
            else:
                print("Name and Student ID are required.")
                
        elif choice == "2":
            # Mark attendance
            print("\n--- Mark Attendance ---")
            print("Make sure camera is ready...")
            input("Press Enter when ready to start face recognition...")
            
            success = system.mark_attendance()
            if success:
                print("\nAttendance marked successfully!")
            else:
                print("\nFailed to mark attendance.")
            input("\nPress Enter to continue...")
                
        elif choice == "3":
            # View today's report
            print("\n--- Today's Attendance Report ---")
            today = datetime.now().date()
            report = system.get_attendance_report(today)
            if not report.empty:
                print(f"\nFound {len(report)} attendance records for today:")
                print("="*80)
                print(report.to_string())
                print("="*80)
            else:
                print("No attendance records for today.")
            input("\nPress Enter to continue...")
                
        elif choice == "4":
            # View registered students
            print("\n--- Registered Students ---")
            students = system.get_registered_students()
            if not students.empty:
                print(f"\nFound {len(students)} registered students:")
                print("="*80)
                print(students.to_string())
                print("="*80)
            else:
                print("No students registered yet.")
            input("\nPress Enter to continue...")
                
        elif choice == "5":
            # Get system stats
            print("\n--- System Statistics ---")
            stats = system.get_system_stats()
            if stats:
                print("\nCurrent System Status:")
                print("-"*30)
                for key, value in stats.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                print("-"*30)
            input("\nPress Enter to continue...")
                
        elif choice == "6":
            # Retrain classifier
            print("\n--- Retrain Classifier ---")
            confirm = input("This will retrain the face recognition classifier. Continue? (y/n): ").lower()
            if confirm == 'y':
                success = system.retrain_classifier()
                if success:
                    print("Classifier retrained successfully!")
                else:
                    print("Failed to retrain classifier.")
            input("\nPress Enter to continue...")
                
        elif choice == "7":
            # Backup database
            print("\n--- Backup Database ---")
            success = system.backup_database()
            input("\nPress Enter to continue...")
                
        elif choice == "8":
            # Exit
            print("\nThank you for using the Attendance System. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1-8.")

def main():
    """Main function to run the system test"""
    try:
        test_system()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()