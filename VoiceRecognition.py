"""
Voice Recognition Module for Attendance System
Integrates with FaceRecognition.py for multimodal biometric authentication
"""

import os
import logging
import sqlite3
import json
import hashlib
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')

# ===========================
# Speech Recognition Imports
# ===========================
try:
    import speech_recognition as sr
    import sounddevice as sd
    import soundfile as sf
    import librosa

    from scipy.io import wavfile
    import noisereduce as nr
    
    # Voice verification libraries
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio
    import torchaudio.transforms as T
    
    # Try to import SpeechBrain with fallbacks
    try:
        from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
        SPEECHBRAIN_AVAILABLE = True
    except Exception as e:
        print(f"SpeechBrain import warning: {e}")
        SPEECHBRAIN_AVAILABLE = False
        
    VOICE_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Voice recognition libraries not fully installed: {e}")
    print("Please install with: pip install speechrecognition sounddevice soundfile librosa scipy noisereduce torch torchaudio speechbrain")
    VOICE_LIBS_AVAILABLE = False
    SPEECHBRAIN_AVAILABLE = False

# ===========================
# Setup logging
# ===========================
log_path = os.path.join(os.getcwd(), "logs", "voice_recognition.log")
os.makedirs("logs", exist_ok=True)

voice_logger = logging.getLogger("VoiceRecognition")
voice_logger.setLevel(logging.INFO)

# Remove existing handlers
voice_logger.handlers = []

# Add file handler
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
voice_logger.addHandler(file_handler)

# Add console handler with ASCII only
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
voice_logger.addHandler(console_handler)

# ===========================
# Robust Model Architecture Classes
# ===========================
class ECAPATDNNBranch(nn.Module):
    """
    Fixed ECAPA-TDNN Architecture with proper dimension handling
    """
    def __init__(self, input_dim=80, embedding_dim=192):
        super().__init__()
        
        # Frame-level layers
        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # Squeeze-and-Excitation blocks
        self.se1 = SELayer(512, 8)
        self.se2 = SELayer(512, 8)
        self.se3 = SELayer(512, 8)
        
        # Feature aggregation
        self.conv4 = nn.Conv1d(512*3, 1536, kernel_size=1)
        
        # Attention statistics pooling
        self.attention = nn.Sequential(
            nn.Conv1d(1536, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Final projection
        self.bn = nn.BatchNorm1d(4608)
        self.fc = nn.Linear(4608, embedding_dim)
        
    def forward(self, x):
        # x shape: [batch, freq, time]
        batch_size = x.size(0)
        
        # Ensure correct input dimensions
        if x.dim() == 3:
            if x.size(1) != 80 and x.size(2) == 80:
                x = x.transpose(1, 2)
        
        # Pad if time dimension is too small
        if x.size(-1) < 100:
            pad_size = 100 - x.size(-1)
            x = F.pad(x, (0, pad_size))
        
        # Frame-level processing with residual connections
        x1 = self.conv1(x)
        x1 = self.se1(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1)
        x2 = self.se2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv3(x2)
        x3 = self.se3(x3)
        x3 = F.relu(x3)
        
        # Ensure all branches have same time dimension
        min_time = min(x1.size(-1), x2.size(-1), x3.size(-1))
        x1 = x1[:, :, :min_time]
        x2 = x2[:, :, :min_time]
        x3 = x3[:, :, :min_time]
        
        # Concatenate features
        concat = torch.cat((x1, x2, x3), dim=1)
        x4 = self.conv4(concat)
        
        # Statistics pooling
        mean = torch.mean(x4, dim=2, keepdim=True)
        std = torch.std(x4, dim=2, keepdim=True)
        
        # Attention mechanism
        attention_weights = self.attention(x4)
        weighted = x4 * attention_weights
        attended_mean = torch.mean(weighted, dim=2, keepdim=True)
        
        # Concatenate statistics
        stats = torch.cat([
            attended_mean.repeat(1, 1, x4.size(-1)),
            mean.repeat(1, 1, x4.size(-1)),
            std.repeat(1, 1, x4.size(-1))
        ], dim=1)
        
        # Final pooling
        pooled = torch.mean(stats, dim=2)
        
        # Batch norm and projection
        out = self.bn(pooled)
        out = self.fc(out)
        
        return out


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer"""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ConvBranch(nn.Module):
    """
    Convolutional Neural Network for raw spectrogram processing
    """
    def __init__(self, in_channels=1, embedding_dim=192):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        embedding = self.fc(features)
        return embedding


class MultiHeadAttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple model branches
    """
    def __init__(self, num_branches, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_branches = num_branches
        self.feature_dim = feature_dim
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        # Branch importance weights
        self.branch_weights = nn.Parameter(torch.ones(num_branches))
        
    def forward(self, x):
        # x shape: [batch, num_branches, feature_dim]
        
        # Learn branch importance
        weights = F.softmax(self.branch_weights, dim=0)
        weighted_x = x * weights.unsqueeze(0).unsqueeze(-1)
        
        # Self-attention
        attended, _ = self.mha(weighted_x, weighted_x, weighted_x)
        attended = self.norm1(attended + weighted_x)
        
        # Feed-forward
        ffn_out = self.ffn(attended)
        fused = self.norm2(ffn_out + attended)
        
        # Flatten
        fused = fused.view(fused.size(0), -1)
        
        return fused



class VoiceFeatureExtractor:
    """
    Extract features from voice samples for speaker recognition
    """
    def __init__(self, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.voice_model = None

        print(f"[Voice] VOICE_LIBS_AVAILABLE: {VOICE_LIBS_AVAILABLE}")
        print(f"[Voice] SPEECHBRAIN_AVAILABLE: {SPEECHBRAIN_AVAILABLE}")
        
        # Initialize models
        self._initialize_models()

        if self.voice_model is not None:
            print("[Voice] ECAPA-TDNN is available and loaded")
        else:
            print("[Voice] ECAPA-TDNN is NOT available")
    
    def _initialize_models(self):
        """
        Initialize voice recognition models - USE ONLY PRE-TRAINED MODELS
        """
        # Use ONLY SpeechBrain ECAPA-TDNN (it's pre-trained)
        if VOICE_LIBS_AVAILABLE and SPEECHBRAIN_AVAILABLE:
            try:
                # Import here to avoid circular imports
                from speechbrain.pretrained import EncoderClassifier
                
                # Force CPU if CUDA issues, otherwise use CUDA
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                print(f"[Voice] Loading ECAPA-TDNN on {device}...")
                # Load pre-trained ECAPA-TDNN

                self.voice_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="models/ecapa_tdnn",
                    run_opts={"device": device}
                )

                # Test the model with a dummy input to ensure it works
                import torch
                dummy_input = torch.randn(1, 16000 * 2)  # 2 seconds of dummy audio
                with torch.no_grad():
                    _ = self.voice_model.encode_batch(dummy_input)
                voice_logger.info("Loaded pre-trained ECAPA-TDNN voice model")
                print("[VOICE] ECAPA-TDNN loaded successfully")               
                # Set robust_model to None - we won't use it
                self.robust_model = None
                
                
            except Exception as e:
                voice_logger.error(f"Failed to load ECAPA-TDNN: {e}")
                print(f"[VOICE] ECAPA-TDNN loading failed: {e}")
                self.voice_model = None
                self.robust_model = None
        else:
            print("[VOICE] SpeechBrain not available")
            self.voice_model = None
            self.robust_model = None
            
    
    def preprocess_audio(self, audio_data, original_sr=None):
        """
        Preprocess audio: resample, denoise, normalize
        """
        if not VOICE_LIBS_AVAILABLE:
            return None
        
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, np.ndarray):
                audio_np = audio_data.astype(np.float32)
            else:
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Convert to mono if stereo
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
            
            # Resample if needed
            if original_sr and original_sr != self.sample_rate:
                try:
                    audio_np = librosa.resample(
                        audio_np, 
                        orig_sr=original_sr, 
                        target_sr=self.sample_rate
                    )
                except Exception as e:
                    voice_logger.warning(f"Resampling error: {e}")
                    # If resampling fails, just use original with warning
                    self.sample_rate = original_sr
            
            # Normalize first to prevent noise reduction issues
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-10)
            
            # Denoise (only if audio is long enough)
            if len(audio_np) > 1000:
                try:
                    audio_np = nr.reduce_noise(y=audio_np, sr=self.sample_rate, prop_decrease=0.5)
                except Exception as e:
                    voice_logger.warning(f"Noise reduction error: {e}")
                    # Continue without noise reduction
            
            # Ensure proper length
            target_length = int(self.duration * self.sample_rate)
            if len(audio_np) > target_length:
                audio_np = audio_np[:target_length]
            elif len(audio_np) < target_length:
                padding = target_length - len(audio_np)
                audio_np = np.pad(audio_np, (0, padding), mode='constant')
            
            # Final normalization
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-10)

            return audio_np
        
        except Exception as e:
            voice_logger.error(f"Audio preprocessing error: {e}")
            return None
    
    def extract_embedding(self, audio_data, original_sr=None):
        """
        Extract speaker embedding using ECAPA-TDNN
        Returns 192-dimensional embedding vector
        """
        if not VOICE_LIBS_AVAILABLE:
            return None
        
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data, original_sr)
            if processed_audio is None or len(processed_audio) < 8000:
                voice_logger.warning("Audio too short for embedding extraction")
                return None
            
            # Use ECAPA-TDNN only
            if self.voice_model is not None:
                try:
                    embedding = self._extract_ecapa_embedding(processed_audio)
                    if embedding is not None:
                        voice_logger.info("Extracted ECAPA-TDNN embedding")
                        
                        return embedding
                    else:
                        voice_logger.warning("ECAPA extraction returned None")
                        return None
                except Exception as e:
                    voice_logger.warning(f"ECAPA extraction failed: {e}")
                    return None
            else:
                voice_logger.warning("ECAPA-TDNN model not available")
                return None
                
        except Exception as e:
            voice_logger.error(f"Embedding extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _extract_ecapa_embedding(self, audio_data):
        """
        Extract embedding using ECAPA-TDNN
        """
        try:
            # Convert to tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)  # Add batch dimension
            else:
                audio_tensor = audio_data
            
            # Ensure audio is long enough (at least 1 second)
            if audio_tensor.shape[-1] < 16000:
                # Pad if too short
                padding = 16000 - audio_tensor.shape[-1]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            
            # Move to same device as model
            device = next(self.voice_model.parameters()).device
            audio_tensor = audio_tensor.to(device)
            
            # Extract embedding
            with torch.no_grad():
                embeddings = self.voice_model.encode_batch(audio_tensor)
                embedding = embeddings.squeeze().cpu().numpy()
            
            # ECAPA-TDNN outputs 192-dimensional embeddings
            # Normalize
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # L2 normalize
            # norm = np.linalg.norm(embedding)
            # embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            voice_logger.info(f"Extracted ECAPA-TDNN embedding, shape: {embedding.shape}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            voice_logger.error(f"ECAPA embedding extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_spectral_features(self, audio_data, original_sr=None):
        """
        Extract additional spectral features
        """
        if not VOICE_LIBS_AVAILABLE:
            return None
        
        try:
            processed_audio = self.preprocess_audio(audio_data, original_sr)
            if processed_audio is None:
                return None
            
            features = []
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=processed_audio, sr=self.sample_rate
            )[0]
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=processed_audio, sr=self.sample_rate
            )[0]
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                processed_audio
            )[0]
            features.append(np.mean(zero_crossing_rate))
            features.append(np.std(zero_crossing_rate))
            
            # Root mean square energy
            rms = librosa.feature.rms(y=processed_audio)[0]
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            return np.array(features)
        
        except Exception as e:
            voice_logger.error(f"Spectral feature extraction error: {e}")
            return None

class VoiceDatabase:
    """
    Database for storing voice samples and embeddings
    """
    def __init__(self):
        self.conn = sqlite3.connect("database/attendance_system.db", check_same_thread=False, timeout=10)
        self._init_voice_tables()
    
    def _init_voice_tables(self):
        """
        Initialize voice-related database tables
        """
        cursor = self.conn.cursor()
        
        # Voice samples table
        cursor.execute('''CREATE TABLE IF NOT EXISTS voice_samples(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        voice_embedding BLOB,
                        embedding_type TEXT DEFAULT 'ECAPA',
                        spectral_features BLOB,
                        sample_file_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                        UNIQUE(student_id, sample_file_path))''')
        
        # Voice verification attempts
        cursor.execute('''CREATE TABLE IF NOT EXISTS voice_verifications(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       student_id TEXT NOT NULL,
                       verification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       confidence FLOAT,
                       success BOOLEAN,
                       method TEXT,
                       FOREIGN KEY (student_id) REFERENCES students(student_id))''')
        
        # Voice enrollment status
        cursor.execute('''CREATE TABLE IF NOT EXISTS voice_enrollment(
                       student_id TEXT PRIMARY KEY,
                       enrollment_status TEXT DEFAULT 'pending',
                       voice_samples_count INTEGER DEFAULT 0,
                       last_enrollment TIMESTAMP,
                       FOREIGN KEY (student_id) REFERENCES students(student_id))''')
        
        # Model performance tracking
        cursor.execute('''CREATE TABLE IF NOT EXISTS model_performance(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       model_type TEXT,
                       verification_id INTEGER,
                       confidence FLOAT,
                       true_positive BOOLEAN,
                       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       FOREIGN KEY (verification_id) REFERENCES voice_verifications(id))''')
        
        self.conn.commit()
        voice_logger.info("Voice database tables initialized")
    
    def add_voice_sample(self, student_id, embedding=None, embedding_type='ECAPA', 
                     spectral_features=None, sample_path=None):
        """
        Add a voice sample with embedding type information
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                cursor = self.conn.cursor()
                
                # Convert embeddings to bytes
                embedding_blob = None
                if embedding is not None:
                    embedding_blob = embedding.tobytes()
                
                # Insert voice sample with embedding type
                cursor.execute('''INSERT INTO voice_samples 
                            (student_id, voice_embedding, embedding_type, spectral_features, sample_file_path)
                            VALUES (?, ?, ?, ?, ?)''',
                            (student_id, embedding_blob, embedding_type, spectral_features, sample_path))
                
                # Update enrollment status
                cursor.execute('''INSERT OR REPLACE INTO voice_enrollment 
                            (student_id, voice_samples_count, last_enrollment, enrollment_status)
                            VALUES (?, 
                                    COALESCE((SELECT voice_samples_count + 1 FROM voice_enrollment WHERE student_id = ?), 1),
                                    CURRENT_TIMESTAMP,
                                    CASE WHEN COALESCE((SELECT voice_samples_count + 1 FROM voice_enrollment WHERE student_id = ?), 1) >= 3 
                                            THEN 'completed' ELSE 'pending' END)''',
                            (student_id, student_id, student_id))
                
                self.conn.commit()
                voice_logger.info(f"Added {embedding_type} voice sample for student {student_id}")
                return True
            
            except sqlite3.Error as e:
                if attempt == max_retries - 1:
                    voice_logger.error(f"Database error adding voice sample: {e}")
                    return False
                time.sleep(0.1)
    
    def get_voice_samples(self, student_id):
        """
        Retrieve all voice samples for a student
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT voice_embedding, spectral_features 
                           FROM voice_samples 
                           WHERE student_id = ?''', (student_id,))
            
            samples = cursor.fetchall()
            embeddings = []
            spectral_features_list = []
            
            for embedding_blob, spectral_blob in samples:
                if embedding_blob:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    embeddings.append(embedding)
                
                if spectral_blob:
                    spectral = np.frombuffer(spectral_blob, dtype=np.float32)
                    spectral_features_list.append(spectral)
            
            return embeddings, spectral_features_list
        
        except sqlite3.Error as e:
            voice_logger.error(f"Database error retrieving voice samples: {e}")
            return [], []
    
    def get_enrollment_status(self, student_id):
        """
        Get voice enrollment status for a student
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT enrollment_status, voice_samples_count 
                           FROM voice_enrollment 
                           WHERE student_id = ?''', (student_id,))
            
            result = cursor.fetchone()
            if result:
                return result[0], result[1]
            return 'not_enrolled', 0
        
        except sqlite3.Error as e:
            voice_logger.error(f"Database error getting enrollment status: {e}")
            return 'error', 0
    
    def log_verification(self, student_id, confidence, success, method="Robust Model"):
        """
        Log voice verification attempt
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO voice_verifications 
                           (student_id, confidence, success, method)
                           VALUES (?, ?, ?, ?)''',
                           (student_id, confidence, success, method))
            
            self.conn.commit()
            return True
        
        except sqlite3.Error as e:
            voice_logger.error(f"Database error logging verification: {e}")
            return False
    
    def get_all_enrolled_students(self):
        """
        Get all students with completed voice enrollment
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT student_id FROM voice_enrollment 
                           WHERE enrollment_status = 'completed' ''')
            
            return [row[0] for row in cursor.fetchall()]
        
        except sqlite3.Error as e:
            voice_logger.error(f"Database error getting enrolled students: {e}")
            return []

class VoiceRecorder:
    """
    Handles voice recording and playback
    """
    def __init__(self, sample_rate=16000, duration=5):
        self.sample_rate = sample_rate
        self.duration = duration
        self.recording = False
        
        if not VOICE_LIBS_AVAILABLE:
            voice_logger.warning("Voice recording libraries not available")
    
    def record_phrase(self, phrase="Please say your student ID clearly"):
        """
        Record a specific phrase from the user
        """
        if not VOICE_LIBS_AVAILABLE:
            print("Voice recording not available. Please install required libraries.")
            return None
        
        try:
            recognizer = sr.Recognizer()
            
            # List available microphones
            try:
                microphone_list = sr.Microphone.list_microphone_names()
                print(f"Available microphones: {microphone_list}")
                
                # Try to use default microphone
                microphone = sr.Microphone()
                
            except Exception as mic_error:
                print(f"Microphone error: {mic_error}")
                print("Please check your microphone connection and permissions.")
                return None
            
            print(f"\n[VOICE] Please say: '{phrase}'")
            print("[VOICE] Recording for 5 seconds...")
            
            with microphone as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[VOICE] Listening...")
                
                # Record audio
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                except sr.WaitTimeoutError:
                    print("[VOICE] No speech detected. Please try again.")
                    return None
                
                # Get audio data
                audio_data = audio.get_raw_data()
                
                # Convert to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                
                # Normalize
                if np.max(np.abs(audio_np)) > 0:
                    audio_np = audio_np / np.max(np.abs(audio_np))
                
                # Save for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"temp/voice_{timestamp}.wav"
                os.makedirs("temp", exist_ok=True)
                
                # Save as WAV
                wavfile.write(filename, 16000, (audio_np * 32767).astype(np.int16))
                
                print(f"[VOICE] Recording saved: {filename}")
                
                # Return audio data and sample rate
                return audio_np, 16000, filename
        
        except Exception as e:
            voice_logger.error(f"Recording error: {e}")
            print(f"[ERROR] Recording failed: {e}")
            return None
    
    def record_free_speech(self, duration=5):
        """
        Record free speech for voiceprint creation
        """
        if not VOICE_LIBS_AVAILABLE:
            print("Voice recording not available.")
            return None
        
        try:
            print(f"\n[VOICE] Please speak naturally for {duration} seconds...")
            print("[VOICE] You can say anything about yourself.")
            
            # Record using sounddevice
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            
            print("[VOICE] Recording...")
            sd.wait()
            
            # Flatten and normalize
            recording = recording.flatten()
            if np.max(np.abs(recording)) > 0:
                recording = recording / np.max(np.abs(recording))
            
            # Save recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temp/free_speech_{timestamp}.wav"
            
            sf.write(filename, recording, self.sample_rate)
            print(f"[VOICE] Free speech saved: {filename}")
            
            return recording, self.sample_rate, filename
        
        except Exception as e:
            voice_logger.error(f"Free speech recording error: {e}")
            return None

class VoiceVerifier:
    """
    Main voice verification class
    """
    def __init__(self):
        if not VOICE_LIBS_AVAILABLE:
            voice_logger.warning("Voice verification libraries not fully available")
        
        self.feature_extractor = VoiceFeatureExtractor()
        self.voice_db = VoiceDatabase()
        self.recorder = VoiceRecorder()
        
        # Thresholds for verification
        self.ecapa_threshold = 0.35  # Cosine similarity threshold for ECAPA
        self.robust_threshold = 0.65
        self.min_confidence = 0.3
        
        voice_logger.info("VoiceVerifier initialized")
    
    def enroll_student(self, student_id):
        """
        Enroll a student's voice
        """
        if not VOICE_LIBS_AVAILABLE:
            print("[ERROR] Voice enrollment requires speech recognition libraries.")
            return False
        
        # Check if student exists in face database
        face_db_path = "database/attendance_system.db"
        if not os.path.exists(face_db_path):
            print("[ERROR] Face database not found. Please register student first.")
            return False
        
        try:
            conn = sqlite3.connect(face_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM students WHERE student_id = ?", (student_id,))
            student_info = cursor.fetchone()
            conn.close()
            
            if not student_info:
                print(f"[ERROR] Student ID {student_id} not found in face database.")
                return False
            
            student_name = student_info[0]
            print(f"\n=== Voice Enrollment for {student_name} ({student_id}) ===")
            
            # Check existing enrollment
            status, count = self.voice_db.get_enrollment_status(student_id)
            if status == 'completed':
                print("[INFO] Student already has voice enrollment.")
                return True
            
            print(f"[INFO] Current voice samples: {count}/3 required")
            
            # Collect 3 voice samples
            samples_collected = 0
            for i in range(3 - count):
                print(f"\n[ENROLLMENT] Sample {count + i + 1}/3:")
                
                # Record phrase
                result = self.recorder.record_phrase("Present")
                
                if result is None:
                    print("[WARNING] Recording failed. Please try again.")
                    continue
                
                audio_data, sample_rate, filepath = result
                
                # Extract features using robust model
                print("[INFO] Extracting voice features using ECAPA-TDNN...")
                embedding = self.feature_extractor.extract_embedding(audio_data, sample_rate)
                spectral_features = self.feature_extractor.extract_spectral_features(audio_data, sample_rate)
                
                if embedding is None:
                    print("[WARNING] Could not extract voice features. Trying alternative method...")
                    # Try with free speech
                    result = self.recorder.record_free_speech(3)
                    if result:
                        audio_data, sample_rate, filepath = result
                        embedding = self.feature_extractor.extract_embedding(audio_data, sample_rate)
                        spectral_features = self.feature_extractor.extract_spectral_features(audio_data, sample_rate)
                
                if embedding is None:
                    print("[ERROR] Could not extract voice features. Please check microphone and try again.")
                    continue
                
                # Save to database
                success = self.voice_db.add_voice_sample(
                    student_id=student_id,
                    embedding=embedding,
                    spectral_features=spectral_features,
                    sample_path=filepath
                )
                
                if success:
                    samples_collected += 1
                    print(f"[SUCCESS] Voice sample {count + i + 1} recorded successfully.")
                else:
                    print("[ERROR] Failed to save voice sample.")
            
            # Final status
            status, final_count = self.voice_db.get_enrollment_status(student_id)
            if status == 'completed':
                print(f"\n[SUCCESS] Voice enrollment completed for {student_name}!")
                print(f"[INFO] Total samples: {final_count}")
                return True
            else:
                print(f"\n[WARNING] Voice enrollment incomplete. Samples: {final_count}/3")
                if final_count > 0:
                    print("[INFO] You can add more samples later.")
                return False
        
        except Exception as e:
            voice_logger.error(f"Enrollment error: {e}")
            print(f"[ERROR] Enrollment error: {e}")
            return False
    
    def verify_voice(self, student_id):
        """
        Enhanced voice verification with multiple feature types and weighted scoring
        """
        if not VOICE_LIBS_AVAILABLE:
            print("[ERROR] Voice verification not available.")
            return False, 0.0
        
        # Check enrollment
        status, sample_count = self.voice_db.get_enrollment_status(student_id)
        if status != 'completed':
            print(f"[ERROR] Student {student_id} not fully enrolled (samples: {sample_count}/3).")
            return False, 0.0
        
        print(f"\n=== Voice Verification for Student ID: {student_id} ===")
        print("[INFO] Please say your student ID clearly.")
        
        # Record verification sample
        result = self.recorder.record_phrase(f"My student ID is {student_id}")
        
        if result is None:
            print("[ERROR] Recording failed.")
            return False, 0.0
        
        audio_data, sample_rate, _ = result
        
        # Extract ECAPA-TDNN embedding
        print("[INFO] Extracting ECAPA-TDNN voice embedding...")
        test_embedding = self.feature_extractor.extract_embedding(audio_data, sample_rate)
        
        if test_embedding is None:
            print("[ERROR] Could not extract voice embedding.")
            return False, 0.0
        
        # Get enrolled samples
        enrolled_embeddings, _ = self.voice_db.get_voice_samples(student_id)
        
        if not enrolled_embeddings:
            print("[ERROR] No enrolled voice samples found.")
            return False, 0.0
        
        # Calculate cosine similarities
        similarities = []
        for enrolled in enrolled_embeddings:
            # Ensure same dimensions
            min_dim = min(len(test_embedding), len(enrolled))
            if min_dim > 0:
                # Cosine similarity
                test_norm = test_embedding[:min_dim] / np.linalg.norm(test_embedding[:min_dim])
                enrolled_norm = enrolled[:min_dim] / np.linalg.norm(enrolled[:min_dim])
                
                similarity = np.dot(test_norm, enrolled_norm)
                similarity = max(min(similarity, 1.0), -1.0)
                similarities.append(similarity)
        
        if not similarities:
            print("[ERROR] No similarity scores calculated.")
            return False, 0.0
        
        # Take best match
        max_similarity = max(similarities)
        
        # ECAPA-TDNN threshold (typically 0.6-0.7 for good accuracy)
        threshold = 0.35
        verification_passed = max_similarity >= threshold
        
        # Log verification
        self.voice_db.log_verification(
            student_id=student_id,
            confidence=float(max_similarity),
            success=verification_passed,
            method="ECAPA-TDNN"
        )
        
        # Display results
        print(f"\n{'='*50}")
        print(f"VERIFICATION RESULTS")
        print(f"{'='*50}")
        print(f"Similarity Score: {max_similarity:.3f}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Status: {'PASSED' if verification_passed else 'FAILED'}")
        
        return verification_passed, float(max_similarity)
    
    def check_audio_quality(self, audio_data, sample_rate):
        """
        Check if audio is good enough for enrollment
        """
        if len(audio_data) < sample_rate * 1.5:  # At least 1.5 seconds
            return False, "Audio too short"
        
        # Check signal-to-noise ratio (simple version)
        energy = np.mean(np.abs(audio_data))
        if energy < 0.01:  # Too quiet
            return False, "Audio too quiet"
        
        # Check for clipping
        if np.max(np.abs(audio_data)) > 0.95:
            return False, "Audio may be clipped"
        
        return True, "Good quality"
    
    def list_enrolled_students(self):
        """
        List all students with voice enrollment
        """
        enrolled_students = self.voice_db.get_all_enrolled_students()
        
        if not enrolled_students:
            print("[INFO] No students enrolled for voice verification.")
            return
        
        print("\n=== Students with Voice Enrollment ===")
        
        # Connect to face database for student details
        conn = sqlite3.connect("database/attendance_system.db")
        cursor = conn.cursor()
        
        for student_id in enrolled_students:
            cursor.execute('''SELECT name, roll_number, class 
                           FROM students WHERE student_id = ?''', (student_id,))
            student_info = cursor.fetchone()
            
            status, sample_count = self.voice_db.get_enrollment_status(student_id)
            
            if student_info:
                name, roll, class_name = student_info
                print(f"ID: {student_id:<10} Name: {name:<20} Roll: {roll:<10} Class: {class_name:<10} Samples: {sample_count}")
        
        conn.close()

# ===========================
# Main Application Interface
# ===========================
def main():
    """
    Main function for voice recognition system
    """
    print("\n" + "=" * 60)
    print("VOICE RECOGNITION SYSTEM")
    print("=" * 60)
    print("Features: ECAPA-TDNN for speaker verification")
    print("=" * 60)
    
    if not VOICE_LIBS_AVAILABLE:
        print("\n[WARNING] Some voice recognition libraries are not installed.")
        print("[INFO] Basic functionality may be limited.")
        print("[INFO] Install with: pip install speechrecognition sounddevice soundfile librosa scipy noisereduce torch torchaudio speechbrain")
    
    # Initialize voice verification system
    verifier = VoiceVerifier()
    
    while True:
        print("\n" + "=" * 50)
        print("VOICE RECOGNITION MENU")
        print("=" * 50)
        print("1. Enroll Student Voice")
        print("2. Verify Student Voice")
        print("3. List Enrolled Students")
        print("4. Check Enrollment Status")
        print("5. Exit")
        print("-" * 50)
        
        choice = input("Select Option (1-5): ").strip()
        
        try:
            if choice == "1":
                student_id = input("Enter Student ID to enroll: ").strip()
                if student_id:
                    verifier.enroll_student(student_id)
                else:
                    print("[ERROR] Student ID is required.")
            
            elif choice == "2":
                student_id = input("Enter Student ID to verify: ").strip()
                if student_id:
                    verifier.verify_voice(student_id)
                else:
                    print("[ERROR] Student ID is required.")
            
            elif choice == "3":
                verifier.list_enrolled_students()
            
            elif choice == "4":
                student_id = input("Enter Student ID to check status: ").strip()
                if student_id:
                    status, count = verifier.voice_db.get_enrollment_status(student_id)
                    print(f"\nEnrollment Status for {student_id}:")
                    print(f"Status: {status}")
                    print(f"Samples: {count}/3")
                else:
                    print("[ERROR] Student ID is required.")
            
            elif choice == "5":
                print("[INFO] Voice Recognition System shutting down...")
                break
            
            else:
                print("[ERROR] Invalid choice. Please select 1-5.")
        
        except KeyboardInterrupt:
            print("\n\n[INFO] Operation cancelled by user.")
        except Exception as e:
            voice_logger.error(f"Menu operation error: {e}")
            print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("database", exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] System interrupted by user.")
    except Exception as e:
        voice_logger.error(f"System error: {e}")
        print(f"[ERROR] System error: {e}")