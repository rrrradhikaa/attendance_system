"""
Conference-Ready Multimodal Attendance System Evaluation
Using REAL DATA from your attendance system database
Following ISO/IEC 19795 biometric testing standards
"""

import os
import sys
import json
import time
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cv2
import warnings
import pickle
import random
import math
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
LFW_PATH = "C:/Users/ramra/AttendanceSystem/datasets/lfw"
VOXCELEB_PATH = "C:/Users/ramra/AttendanceSystem/datasets/voxceleb"
import os
if not LFW_PATH and os.environ.get('LFW_PATH'):
    LFW_PATH = os.environ.get('LFW_PATH')
if not VOXCELEB_PATH and os.environ.get('VOXCELEB_PATH'):
    VOXCELEB_PATH = os.environ.get('VOXCELEB_PATH')

# Add parent directory to path
sys.path.append('.')

# ============================================
# CLASS 1: REAL DATA LOADER FROM DATABASE
# ============================================

class RealDataLoader:
    """
    Loads REAL biometric data from your attendance system database
    Extracts face and voice embeddings from actual student registrations
    """
    
    def __init__(self, db_path: str = "database/attendance_system.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.face_embeddings = {}
        self.voice_embeddings = {}
        self.student_info = {}
        
    def connect_database(self):
        """Connect to the attendance system database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def load_all_real_data(self) -> Dict[str, Any]:
        """
        Load ALL real data from database
        Returns: Complete dataset with face and voice embeddings
        """
        print("\n" + "="*60)
        print("LOADING REAL DATA FROM YOUR DATABASE")
        print("="*60)
        
        if not self.connect_database():
            return self._create_empty_dataset()
        
        dataset = {
            'face': {'genuine_scores': [], 'impostor_scores': []},
            'voice': {'genuine_scores': [], 'impostor_scores': []},
            'multimodal': {'genuine_scores': [], 'impostor_scores': []},
            'student_details': {},
            'statistics': {},
            'raw_embeddings': {'face': {}, 'voice': {}}
        }
        
        # 1. Load student information
        self._load_student_info(dataset)
        
        # 2. Load face embeddings
        face_data_loaded = self._load_face_embeddings(dataset)
        
        # 3. Load voice embeddings
        voice_data_loaded = self._load_voice_embeddings(dataset)
        
        # 4. Generate genuine and impostor scores
        if face_data_loaded:
            self._generate_face_scores(dataset)
        
        if voice_data_loaded:
            self._generate_voice_scores(dataset)
        
        # 5. Generate multimodal scores (fusion)
        if face_data_loaded and voice_data_loaded:
            self._generate_multimodal_scores(dataset)
        
        # 6. Calculate statistics
        self._calculate_dataset_statistics(dataset)
        
        self.conn.close()
        
        print(f"\n Real data loading complete!")
        print(f"  Students: {len(dataset['student_details'])}")
        print(f"  Face embeddings: {len(dataset['raw_embeddings']['face'])}")
        print(f"  Voice embeddings: {len(dataset['raw_embeddings']['voice'])}")
        
        return dataset
    
    def _load_student_info(self, dataset: Dict):
        """Load student information from database"""
        try:
            self.cursor.execute("""
                SELECT id, name, student_id, roll_number, class, 
                       model_type, created_at 
                FROM students
                ORDER BY id
            """)
            
            students = self.cursor.fetchall()
            
            for student in students:
                student_id, name, student_code, roll, class_name, model, created = student
                dataset['student_details'][student_id] = {
                    'name': name,
                    'student_id': student_code,
                    'roll_number': roll,
                    'class': class_name,
                    'model_type': model,
                    'created_at': created,
                    
                }
            
            print(f" Loaded {len(students)} students")
            
        except Exception as e:
            print(f" Error loading student info: {e}")
    
    def _load_face_embeddings(self, dataset: Dict) -> bool:
        """Load face embeddings from database"""
        try:
            self.cursor.execute("""
                SELECT student_id, encrypted_data, encoding_dim
                FROM face_encodings
                ORDER BY student_id
            """)
            
            face_encodings = self.cursor.fetchall()
            
            for student_id, encoded_data, encoding_dim in face_encodings:
                try:
                    # Decode the face embedding
                    if encoded_data:
                        # Try different decoding methods
                        if isinstance(encoded_data, bytes):
                            # Method 1: Pickle load
                            try:
                                embedding = pickle.loads(encoded_data)
                            except:
                                # Method 2: Numpy from buffer
                                embedding = np.frombuffer(encoded_data, dtype=np.float32)
                        
                        # Convert student_id to string for consistency
                        student_id_str = str(student_id)
                        
                        if student_id_str not in dataset['raw_embeddings']['face']:
                            dataset['raw_embeddings']['face'][student_id_str] = []
                        
                        dataset['raw_embeddings']['face'][student_id_str].append({
                            'embedding': embedding,
                            'encoding_dim': encoding_dim
                        })
                        
                        # Store in face_embeddings dict for easy access
                        if student_id_str not in self.face_embeddings:
                            self.face_embeddings[student_id_str] = []
                        self.face_embeddings[student_id_str].append(embedding)
                        
                except Exception as e:
                    print(f"  Warning: Could not decode face embedding for student {student_id}: {e}")
                    continue
            
            print(f" Loaded face embeddings for {len(dataset['raw_embeddings']['face'])} students")
            print(f"  Student IDs with face data: {list(dataset['raw_embeddings']['face'].keys())}")
            print(f"  Total embeddings: {sum(len(emb) for emb in dataset['raw_embeddings']['face'].values())}")
            
            return len(dataset['raw_embeddings']['face']) > 0
            
        except Exception as e:
            print(f" Error loading face embeddings: {e}")
            return False
    
    def _load_voice_embeddings(self, dataset: Dict) -> bool:
        """Load voice embeddings from database"""
        try:
            # Try different possible table names for voice data
            voice_tables = ['voice_samples', 'voice_encodings', 'voice_data']
            voice_table_found = None
            
            for table in voice_tables:
                self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if self.cursor.fetchone():
                    voice_table_found = table
                    break
            
            if not voice_table_found:
                print("  No voice data table found in database")
                return False
            
            self.cursor.execute(f"""
                SELECT student_id, voice_embedding, mfcc_features, sample_file_path, created_at
                FROM {voice_table_found}
                ORDER BY student_id, created_at
            """)
            
            voice_data = self.cursor.fetchall()
            
            for student_id, voice_embedding, mfcc_features, sample_file_path, created_at in voice_data:
                try:
                    # Convert student_id to string for consistency
                    student_id_str = str(student_id)
                    
                    if student_id_str not in dataset['raw_embeddings']['voice']:
                        dataset['raw_embeddings']['voice'][student_id_str] = []
                    
                    # Try to extract embedding
                    embedding = None
                    
                    if voice_embedding:
                        try:
                            embedding = pickle.loads(voice_embedding)
                        except:
                            try:
                                embedding = np.frombuffer(voice_embedding, dtype=np.float32)
                            except:
                                pass
                    
                    # If no embedding, try mfcc_features
                    if embedding is None and mfcc_features:
                        try:
                            embedding = pickle.loads(mfcc_features)
                        except:
                            try:
                                embedding = np.frombuffer(mfcc_features, dtype=np.float32)
                            except:
                                pass
                    
                    if embedding is not None:
                        dataset['raw_embeddings']['voice'][student_id_str].append({
                            'embedding': embedding,
                            'sample_file_path': sample_file_path,
                            'created_at': created_at
                        })
                        
                        # Store in voice_embeddings dict
                        if student_id_str not in self.voice_embeddings:
                            self.voice_embeddings[student_id_str] = []
                        self.voice_embeddings[student_id_str].append(embedding)
                    
                except Exception as e:
                    print(f"  Warning: Could not decode voice data for student {student_id}: {e}")
                    continue
            
            print(f" Loaded voice embeddings for {len(dataset['raw_embeddings']['voice'])} students")
            print(f"  Student IDs with voice data: {list(dataset['raw_embeddings']['voice'].keys())}")
            
            return len(dataset['raw_embeddings']['voice']) > 0
            
        except Exception as e:
            print(f" Error loading voice embeddings: {e}")
            return False
    
    def _generate_face_scores(self, dataset: Dict):
        """Generate genuine and impostor similarity scores for face"""
        print("\nGenerating face similarity scores...")
        
        face_embeddings = dataset['raw_embeddings']['face']
        student_ids = list(face_embeddings.keys())
        
        genuine_count = 0
        impostor_count = 0
        
        # Generate genuine scores (same student, different samples)
        for student_id in student_ids:
            embeddings = face_embeddings[student_id]
            if len(embeddings) >= 2:
                # Create pairs of different samples from same student
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        emb1 = embeddings[i]['embedding']
                        emb2 = embeddings[j]['embedding']
                        
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(emb1, emb2)
                        dataset['face']['genuine_scores'].append(similarity)
                        genuine_count += 1
        
        # Generate impostor scores (different students)
        for i in range(len(student_ids)):
            for j in range(i + 1, len(student_ids)):
                student1 = student_ids[i]
                student2 = student_ids[j]
                
                embeddings1 = face_embeddings[student1]
                embeddings2 = face_embeddings[student2]
                
                if embeddings1 and embeddings2:
                    # Take one sample from each student
                    emb1 = embeddings1[0]['embedding']
                    emb2 = embeddings2[0]['embedding']
                    
                    similarity = self._cosine_similarity(emb1, emb2)
                    dataset['face']['impostor_scores'].append(similarity)
                    impostor_count += 1
        
        print(f"  Genuine pairs: {genuine_count}")
        print(f"  Impostor pairs: {impostor_count}")
    
    def _generate_voice_scores(self, dataset: Dict):
        """Generate genuine and impostor similarity scores for voice"""
        print("\nGenerating voice similarity scores...")
        
        voice_embeddings = dataset['raw_embeddings']['voice']
        student_ids = list(voice_embeddings.keys())
        
        genuine_count = 0
        impostor_count = 0
        
        # Generate genuine scores
        for student_id in student_ids:
            embeddings = voice_embeddings[student_id]
            if len(embeddings) >= 2:
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        emb1 = embeddings[i]['embedding']
                        emb2 = embeddings[j]['embedding']
                        
                        similarity = self._cosine_similarity(emb1, emb2)
                        dataset['voice']['genuine_scores'].append(similarity)
                        genuine_count += 1
        
        # Generate impostor scores
        for i in range(len(student_ids)):
            for j in range(i + 1, len(student_ids)):
                student1 = student_ids[i]
                student2 = student_ids[j]
                
                embeddings1 = voice_embeddings[student1]
                embeddings2 = voice_embeddings[student2]
                
                if embeddings1 and embeddings2:
                    emb1 = embeddings1[0]['embedding']
                    emb2 = embeddings2[0]['embedding']
                    
                    similarity = self._cosine_similarity(emb1, emb2)
                    dataset['voice']['impostor_scores'].append(similarity)
                    impostor_count += 1
        
        print(f"  Genuine pairs: {genuine_count}")
        print(f"  Impostor pairs: {impostor_count}")
    
    def _generate_multimodal_scores(self, dataset: Dict):
        """Generate multimodal scores by fusing face and voice"""
        print("\nGenerating multimodal similarity scores...")
        
        # Get students with both face and voice data
        face_students = set(dataset['raw_embeddings']['face'].keys())
        voice_students = set(dataset['raw_embeddings']['voice'].keys())
        multimodal_students = list(face_students.intersection(voice_students))
        
        if len(multimodal_students) < 2:
            print(f"  Cannot generate multimodal scores: need at least 2 students with both face and voice data (have {len(multimodal_students)})")
            return
        
        genuine_count = 0
        impostor_count = 0
        
        # Generate genuine multimodal scores
        for student_id in multimodal_students:
            face_embs = dataset['raw_embeddings']['face'][student_id]
            voice_embs = dataset['raw_embeddings']['voice'][student_id]
            
            if len(face_embs) >= 2 and len(voice_embs) >= 2:
                # Take one pair from face and one from voice
                face_sim = self._cosine_similarity(
                    face_embs[0]['embedding'], 
                    face_embs[1]['embedding']
                )
                voice_sim = self._cosine_similarity(
                    voice_embs[0]['embedding'],
                    voice_embs[1]['embedding']
                )
                
                # Simple average fusion
                multimodal_sim = (face_sim + voice_sim) / 2
                dataset['multimodal']['genuine_scores'].append(multimodal_sim)
                genuine_count += 1
        
        # Generate impostor multimodal scores
        for i in range(len(multimodal_students)):
            for j in range(i + 1, len(multimodal_students)):
                student1 = multimodal_students[i]
                student2 = multimodal_students[j]
                
                face_emb1 = dataset['raw_embeddings']['face'][student1][0]['embedding']
                face_emb2 = dataset['raw_embeddings']['face'][student2][0]['embedding']
                voice_emb1 = dataset['raw_embeddings']['voice'][student1][0]['embedding']
                voice_emb2 = dataset['raw_embeddings']['voice'][student2][0]['embedding']
                
                face_sim = self._cosine_similarity(face_emb1, face_emb2)
                voice_sim = self._cosine_similarity(voice_emb1, voice_emb2)
                
                multimodal_sim = (face_sim + voice_sim) / 2
                dataset['multimodal']['impostor_scores'].append(multimodal_sim)
                impostor_count += 1
        
        print(f"  Genuine pairs: {genuine_count}")
        print(f"  Impostor pairs: {impostor_count}")
    
    def _calculate_dataset_statistics(self, dataset: Dict):
        """Calculate basic statistics about the dataset"""
        stats = dataset['statistics']
        
        # Student statistics
        stats['total_students'] = len(dataset['student_details'])
        stats['students_with_face'] = len(dataset['raw_embeddings']['face'])
        stats['students_with_voice'] = len(dataset['raw_embeddings']['voice'])
        
        # Count multimodal students
        face_students = set(dataset['raw_embeddings']['face'].keys())
        voice_students = set(dataset['raw_embeddings']['voice'].keys())
        stats['students_with_both'] = len(face_students.intersection(voice_students))
        
        # Embedding statistics
        stats['total_face_embeddings'] = sum(
            len(emb) for emb in dataset['raw_embeddings']['face'].values()
        )
        stats['total_voice_embeddings'] = sum(
            len(emb) for emb in dataset['raw_embeddings']['voice'].values()
        )
        
        # Score statistics
        for modality in ['face', 'voice', 'multimodal']:
            if dataset[modality]['genuine_scores']:
                stats[f'{modality}_genuine_mean'] = np.mean(dataset[modality]['genuine_scores'])
                stats[f'{modality}_genuine_std'] = np.std(dataset[modality]['genuine_scores'])
                stats[f'{modality}_genuine_pairs'] = len(dataset[modality]['genuine_scores'])
            
            if dataset[modality]['impostor_scores']:
                stats[f'{modality}_impostor_mean'] = np.mean(dataset[modality]['impostor_scores'])
                stats[f'{modality}_impostor_std'] = np.std(dataset[modality]['impostor_scores'])
                stats[f'{modality}_impostor_pairs'] = len(dataset[modality]['impostor_scores'])
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = vec1.flatten().astype(np.float32)
            vec2 = vec2.flatten().astype(np.float32)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.5
            
            similarity = dot_product / (norm1 * norm2)
            # Ensure similarity is between -1 and 1
            similarity = max(-1.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            print(f"Warning: Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _create_empty_dataset(self) -> Dict:
        """Create empty dataset structure"""
        return {
            'face': {'genuine_scores': [], 'impostor_scores': []},
            'voice': {'genuine_scores': [], 'impostor_scores': []},
            'multimodal': {'genuine_scores': [], 'impostor_scores': []},
            'student_details': {},
            'statistics': {},
            'raw_embeddings': {'face': {}, 'voice': {}}
        }

# ============================================
# CLASS 2: BIOMETRIC METRICS CALCULATOR
# ============================================

class BiometricMetricsCalculator:
    """
    Calculates all critical biometric metrics:
    - FAR, FRR, EER
    - ROC Curves, AUC
    - Confidence Intervals
    - Statistical Significance
    """
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_all_metrics(self, genuine_scores: List[float], 
                            impostor_scores: List[float], 
                            modality_name: str = "unknown") -> Dict[str, Any]:
        """
        Calculate all biometric metrics for given scores
        """
        print(f"\nCalculating metrics for {modality_name}...")
        
        # Check and invert scores if genuine < impostor (your current problem)
        genuine_np = np.array(genuine_scores)
        impostor_np = np.array(impostor_scores)
        
        if len(genuine_np) > 0 and len(impostor_np) > 0:
            genuine_mean = np.mean(genuine_np)
            impostor_mean = np.mean(impostor_np)
            
            print(f"  Genuine mean: {genuine_mean:.3f}, Impostor mean: {impostor_mean:.3f}")
            
            # If genuine scores are lower than impostor scores (backwards)
            if genuine_mean < impostor_mean:
                print(f"  WARNING: Inverting scores for {modality_name}!")
                # Invert the scores (1 - score)
                genuine_scores = [1 - score for score in genuine_scores]
                impostor_scores = [1 - score for score in impostor_scores]
                
                # Recalculate means after inversion
                genuine_mean = np.mean(genuine_scores)
                impostor_mean = np.mean(impostor_scores)
                print(f"  After inversion - Genuine mean: {genuine_mean:.3f}, Impostor mean: {impostor_mean:.3f}")
        # ========== END FIX ==========
        metrics = {
            'modality': modality_name,
            'genuine_count': len(genuine_scores),
            'impostor_count': len(impostor_scores),
            'scores': {
                'genuine': genuine_scores,
                'impostor': impostor_scores
            }
        }
        
        # Basic statistics
        metrics['basic_stats'] = self._calculate_basic_statistics(genuine_scores, impostor_scores)
        
        # FAR/FRR curves and EER
        far_frr_metrics = self._calculate_far_frr_eer(genuine_scores, impostor_scores)
        metrics.update(far_frr_metrics)
        
        # ROC curve and AUC
        roc_metrics = self._calculate_roc_auc(genuine_scores, impostor_scores)
        metrics.update(roc_metrics)
        
        # Confidence intervals
        ci_metrics = self._calculate_confidence_intervals(genuine_scores, impostor_scores)
        metrics.update(ci_metrics)
        
        # Statistical tests
        stats_metrics = self._calculate_statistical_tests(genuine_scores, impostor_scores)
        metrics.update(stats_metrics)
        
        # Detection Error Tradeoff (DET) curve
        det_metrics = self._calculate_det_curve(genuine_scores, impostor_scores)
        metrics.update(det_metrics)
        
        self.metrics[modality_name] = metrics
        return metrics
    
    def _calculate_basic_statistics(self, genuine_scores: List[float], 
                                  impostor_scores: List[float]) -> Dict:
        """Calculate basic statistics of scores"""
        genuine_np = np.array(genuine_scores)
        impostor_np = np.array(impostor_scores)
        
        return {
            'genuine_mean': float(np.mean(genuine_np)),
            'genuine_std': float(np.std(genuine_np)),
            'genuine_median': float(np.median(genuine_np)),
            'genuine_min': float(np.min(genuine_np)),
            'genuine_max': float(np.max(genuine_np)),
            'impostor_mean': float(np.mean(impostor_np)),
            'impostor_std': float(np.std(impostor_np)),
            'impostor_median': float(np.median(impostor_np)),
            'impostor_min': float(np.min(impostor_np)),
            'impostor_max': float(np.max(impostor_np)),
            'score_separation': float(np.mean(genuine_np) - np.mean(impostor_np))
        }
    
    def _calculate_far_frr_eer(self, genuine_scores: List[float], 
                             impostor_scores: List[float]) -> Dict:
        """Calculate FAR, FRR, and EER"""
        # Convert to numpy arrays
        genuine = np.array(genuine_scores)
        impostor = np.array(impostor_scores)
        
        # Create thresholds from min to max score
        all_scores = np.concatenate([genuine, impostor])
        thresholds = np.linspace(-0.5, 1.5, 2000)
        
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            # FAR: False Acceptance Rate (impostor accepted)
            far = np.sum(impostor >= threshold) / len(impostor)
            
            # FRR: False Rejection Rate (genuine rejected)
            frr = np.sum(genuine < threshold) / len(genuine)
            
            far_values.append(far)
            frr_values.append(frr)
        
        # Find EER (where FAR = FRR)
        # Find the point where |FAR - FRR| is minimized
        far_np = np.array(far_values)
        frr_np = np.array(frr_values)
        
        # Find index where FAR and FRR are closest
        diff = np.abs(far_np - frr_np)
        eer_idx = np.argmin(diff)
        
        eer_threshold = thresholds[eer_idx]
        eer_value = (far_values[eer_idx] + frr_values[eer_idx]) / 2
        
        # Calculate at specific operating points
        operating_points = {
            'far_0.1': self._get_frr_at_far(far_values, frr_values, thresholds, 0.001),
            'far_0.01': self._get_frr_at_far(far_values, frr_values, thresholds, 0.0001),
            'frr_0.1': self._get_far_at_frr(far_values, frr_values, thresholds, 0.001),
        }
        
        return {
            'far_curve': far_values,
            'frr_curve': frr_values,
            'thresholds': thresholds.tolist(),
            'eer': {
                'value': float(eer_value),
                'threshold': float(eer_threshold),
                'far_at_eer': float(far_values[eer_idx]),
                'frr_at_eer': float(frr_values[eer_idx])
            },
            'operating_points': operating_points
        }
    
    def _get_frr_at_far(self, far_values: List[float], frr_values: List[float],
                       thresholds: np.ndarray, target_far=0.01) -> Dict:
        """Get FRR at specific FAR operating point"""
        # Find threshold where FAR is closest to target
        far_np = np.array(far_values)
        idx = np.argmin(np.abs(far_np - target_far))
        
        return {
            'far': float(far_values[idx]),
            'frr': float(frr_values[idx]),
            'threshold': float(thresholds[idx])
        }
    
    def _get_far_at_frr(self, far_values: List[float], frr_values: List[float],
                       thresholds: np.ndarray, target_frr=0.05) -> Dict:
        """Get FAR at specific FRR operating point"""
        frr_np = np.array(frr_values)
        idx = np.argmin(np.abs(frr_np - target_frr))
        
        return {
            'far': float(far_values[idx]),
            'frr': float(frr_values[idx]),
            'threshold': float(thresholds[idx])
        }
    
    def _calculate_roc_auc(self, genuine_scores: List[float], 
                          impostor_scores: List[float]) -> Dict:
        """Calculate ROC curve and AUC"""
        # Create labels and scores for ROC calculation
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),  # Genuine = 1
            np.zeros(len(impostor_scores))  # Impostor = 0
        ])
        
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate AUC confidence interval using bootstrapping
        auc_ci = self._bootstrap_auc_ci(y_true, y_scores)
        
        return {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'auc': float(roc_auc),
            'auc_ci_95': auc_ci
        }
    
    def _bootstrap_auc_ci(self, y_true: np.ndarray, y_scores: np.ndarray, 
                         n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate 95% confidence interval for AUC using bootstrapping"""
        auc_values = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_scores_boot = y_scores[indices]
            
            # Calculate AUC for bootstrap sample
            try:
                fpr, tpr, _ = roc_curve(y_true_boot, y_scores_boot)
                auc_boot = auc(fpr, tpr)
                auc_values.append(auc_boot)
            except:
                continue
        
        if len(auc_values) > 0:
            auc_values = np.array(auc_values)
            lower = np.percentile(auc_values, 2.5)
            upper = np.percentile(auc_values, 97.5)
            return (float(lower), float(upper))
        
        return (0.0, 1.0)
    
    def _calculate_confidence_intervals(self, genuine_scores: List[float],
                                      impostor_scores: List[float]) -> Dict:
        """Calculate 95% confidence intervals for means"""
        genuine_np = np.array(genuine_scores)
        impostor_np = np.array(impostor_scores)
        
        # Confidence interval for genuine mean
        genuine_se = stats.sem(genuine_np)
        if len(genuine_np) > 1:
            genuine_ci = stats.t.interval(0.95, len(genuine_np)-1, 
                                        loc=np.mean(genuine_np), 
                                        scale=genuine_se)
        else:
            genuine_ci = (np.mean(genuine_np), np.mean(genuine_np))
        
        # Confidence interval for impostor mean
        impostor_se = stats.sem(impostor_np)
        if len(impostor_np) > 1:
            impostor_ci = stats.t.interval(0.95, len(impostor_np)-1,
                                         loc=np.mean(impostor_np),
                                         scale=impostor_se)
        else:
            impostor_ci = (np.mean(impostor_np), np.mean(impostor_np))
        
        return {
            'confidence_intervals': {
                'genuine_mean_ci_95': (float(genuine_ci[0]), float(genuine_ci[1])),
                'impostor_mean_ci_95': (float(impostor_ci[0]), float(impostor_ci[1])),
                'genuine_se': float(genuine_se),
                'impostor_se': float(impostor_se)
            }
        }
    
    def _calculate_statistical_tests(self, genuine_scores: List[float],
                                   impostor_scores: List[float]) -> Dict:
        """Calculate statistical significance tests"""
        genuine_np = np.array(genuine_scores)
        impostor_np = np.array(impostor_scores)
        
        results = {}
        
        # T-test for difference in means
        if len(genuine_np) > 1 and len(impostor_np) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(genuine_np, impostor_np, 
                                                equal_var=False)
                results['t_test'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except:
                pass
        
        # Mann-Whitney U test (non-parametric)
        if len(genuine_np) > 0 and len(impostor_np) > 0:
            try:
                u_stat, p_value_mw = stats.mannwhitneyu(genuine_np, impostor_np,
                                                       alternative='two-sided')
                results['mann_whitney'] = {
                    'u_statistic': float(u_stat),
                    'p_value': float(p_value_mw),
                    'significant': p_value_mw < 0.05
                }
            except:
                pass
        
        # Effect size (Cohen's d)
        if len(genuine_np) > 0 and len(impostor_np) > 0:
            pooled_std = np.sqrt((np.var(genuine_np) + np.var(impostor_np)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(genuine_np) - np.mean(impostor_np)) / pooled_std
                results['effect_size'] = {
                    'cohens_d': float(cohens_d),
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
        
        return {'statistical_tests': results}
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if abs(d) < 0.2:
            return "Negligible"
        elif abs(d) < 0.5:
            return "Small"
        elif abs(d) < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _calculate_det_curve(self, genuine_scores: List[float],
                           impostor_scores: List[float]) -> Dict:
        """Calculate Detection Error Tradeoff (DET) curve"""
        # DET curve uses normal deviate scale for FAR and FRR
        
        # Get FAR and FRR at different thresholds
        genuine = np.array(genuine_scores)
        impostor = np.array(impostor_scores)
        thresholds = np.linspace(np.min(genuine), np.max(genuine), 1000)
        
        far_det = []
        frr_det = []
        
        for threshold in thresholds:
            far = np.sum(impostor >= threshold) / len(impostor)
            frr = np.sum(genuine < threshold) / len(genuine)
            
            # Convert to normal deviate scale (avoid division by zero)
            far_nd = self._normal_deviate(far) if far > 0 and far < 1 else -10
            frr_nd = self._normal_deviate(frr) if frr > 0 and frr < 1 else -10
            
            far_det.append(far_nd)
            frr_det.append(frr_nd)
        
        return {
            'det_curve': {
                'far_normal_deviate': far_det,
                'frr_normal_deviate': frr_det,
                'thresholds': thresholds.tolist()
            }
        }
    
    def _normal_deviate(self, p: float) -> float:
        """Convert probability to normal deviate (standard score)"""
        if p <= 1e-6:
            return -5  # Less extreme
        if p >= 0.999999:
            return 5   # Less extreme
        
        # Use scipy's normal distribution inverse CDF
        return float(stats.norm.ppf(p))
    
    def compare_modalities(self, metrics_dict: Dict[str, Dict]) -> Dict:
        """Compare metrics across different modalities"""
        comparison = {}
        modalities = list(metrics_dict.keys())
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1 = modalities[i]
                mod2 = modalities[j]
                
                key = f"{mod1}_vs_{mod2}"
                
                # Compare AUC using DeLong test (if we had predictions)
                # For now, compare using mean differences
                auc1 = metrics_dict[mod1]['auc']
                auc2 = metrics_dict[mod2]['auc']
                
                comparison[key] = {
                    'auc_difference': auc1 - auc2,
                    'eer_difference': (metrics_dict[mod1]['eer']['value'] - 
                                     metrics_dict[mod2]['eer']['value']),
                    'relative_improvement': ((auc1 - auc2) / auc2 * 100 
                                           if auc2 > 0 else 0)
                }
        
        return {'modality_comparison': comparison}

# ============================================
# CLASS 3: DATA VISUALIZATION GENERATOR
# ============================================

class ConferenceVisualizationGenerator:
    """
    Creates publication-quality visualizations for conference presentations
    """
    
    def __init__(self, output_dir: str = "conference_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set publication quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        self.figsize = (10, 8)
        self.dpi = 300
        self.font_size = 12
    
    def generate_all_visualizations(self, dataset: Dict, metrics: Dict) -> Dict:
        """
        Generate all visualizations for conference presentation
        """
        print(f"\n{'='*60}")
        print("GENERATING CONFERENCE-QUALITY VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        visualizations = {}
        
        # 1. Score Distribution Plot
        print("1. Creating score distribution plots...")
        vis1 = self._plot_score_distributions(dataset, metrics)
        visualizations['score_distributions'] = vis1
        
        # 2. ROC Curves
        print("2. Creating ROC curves...")
        vis2 = self._plot_roc_curves(metrics)
        visualizations['roc_curves'] = vis2
        
        # 3. FAR-FRR Curves and EER
        print("3. Creating FAR-FRR curves...")
        vis3 = self._plot_far_frr_curves(metrics)
        visualizations['far_frr_curves'] = vis3
        
        # 4. DET Curves
        print("4. Creating DET curves...")
        vis4 = self._plot_det_curves(metrics)
        visualizations['det_curves'] = vis4
        
        # 5. Performance Comparison Bar Plot
        print("5. Creating performance comparison...")
        vis5 = self._plot_performance_comparison(metrics)
        visualizations['performance_comparison'] = vis5
        
        # 6. Statistical Significance Plot
        print("6. Creating statistical plots...")
        vis6 = self._plot_statistical_significance(metrics)
        visualizations['statistical_plots'] = vis6
        
        # 7. Dataset Statistics
        print("7. Creating dataset overview...")
        vis7 = self._plot_dataset_overview(dataset)
        visualizations['dataset_overview'] = vis7
        
        # 8. Generate PDF Report
        print("8. Generating PDF report...")
        pdf_path = self._generate_pdf_report(visualizations, dataset, metrics)
        visualizations['pdf_report'] = pdf_path
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")
        
        return visualizations
    
    def _plot_score_distributions(self, dataset: Dict, metrics: Dict) -> str:
        """Plot score distributions for each modality"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        modalities = ['face', 'voice', 'multimodal']
        
        for idx, modality in enumerate(modalities):
            ax = axes[idx]
            
            if modality in metrics:
                genuine_scores = metrics[modality]['scores']['genuine']
                impostor_scores = metrics[modality]['scores']['impostor']
                
                # Plot histograms
                ax.hist(genuine_scores, bins=30, alpha=0.7, label='Genuine',
                       density=True, color='green', edgecolor='black')
                ax.hist(impostor_scores, bins=30, alpha=0.7, label='Impostor',
                       density=True, color='red', edgecolor='black')
                
                # Add vertical line at EER threshold
                if 'eer' in metrics[modality]:
                    eer_thresh = metrics[modality]['eer']['threshold']
                    ax.axvline(x=eer_thresh, color='blue', linestyle='--',
                             label=f'EER Threshold: {eer_thresh:.3f}')
                
                # Add kernel density estimation
                if len(genuine_scores) > 1 and len(impostor_scores) > 1:
                    from scipy.stats import gaussian_kde
                    
                    # Genuine KDE
                    x_min = min(min(genuine_scores), min(impostor_scores))
                    x_max = max(max(genuine_scores), max(impostor_scores))
                    x = np.linspace(x_min, x_max, 1000)
                    
                    kde_gen = gaussian_kde(genuine_scores)
                    kde_imp = gaussian_kde(impostor_scores)
                    
                    ax.plot(x, kde_gen(x), 'g-', linewidth=2)
                    ax.plot(x, kde_imp(x), 'r-', linewidth=2)
                
                ax.set_xlabel('Similarity Score')
                ax.set_ylabel('Density')
                ax.set_title(f'{modality.upper()} Score Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f"Genuine: mean={np.mean(genuine_scores):.3f}, standard_deviation={np.std(genuine_scores):.3f}\n"
                stats_text += f"Impostor: mean={np.mean(impostor_scores):.3f}, standard_deviation={np.std(impostor_scores):.3f}\n"
                stats_text += f"EER: {metrics[modality]['eer']['value']:.3f}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filename = self.output_dir / "score_distributions.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_roc_curves(self, metrics: Dict) -> str:
        """Plot ROC curves for all modalities"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = {'face': 'blue', 'voice': 'green', 'multimodal': 'red'}
        
        for modality in ['face', 'voice', 'multimodal']:
            if modality in metrics and 'roc_curve' in metrics[modality]:
                roc_data = metrics[modality]['roc_curve']
                auc_value = metrics[modality]['auc']
                
                ax.plot(roc_data['fpr'], roc_data['tpr'],
                       color=colors.get(modality, 'black'),
                       linewidth=2,
                       label=f'{modality.upper()} (AUC = {auc_value:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate (FAR)')
        ax.set_ylabel('True Positive Rate (1 - FRR)')
        ax.set_title('ROC Curves for Different Modalities')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add AUC confidence intervals
        for modality in ['face', 'voice', 'multimodal']:
            if modality in metrics and 'auc_ci_95' in metrics[modality]:
                ci = metrics[modality]['auc_ci_95']
                ax.text(0.6, 0.1 + 0.05 * list(colors.keys()).index(modality),
                       f"{modality}: AUC CI = [{ci[0]:.3f}, {ci[1]:.3f}]",
                       transform=ax.transAxes, fontsize=10,
                       color=colors.get(modality, 'black'))
        
        plt.tight_layout()
        filename = self.output_dir / "roc_curves.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_far_frr_curves(self, metrics: Dict) -> str:
        """Plot FAR-FRR tradeoff curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, modality in enumerate(['face', 'voice', 'multimodal']):
            ax = axes[idx]
            
            if modality in metrics:
                thresholds = metrics[modality]['thresholds']
                far_curve = metrics[modality]['far_curve']
                frr_curve = metrics[modality]['frr_curve']
                eer = metrics[modality]['eer']['value']
                
                ax.plot(thresholds, far_curve, 'r-', linewidth=2, label='FAR')
                ax.plot(thresholds, frr_curve, 'b-', linewidth=2, label='FRR')
                
                # Mark EER point
                eer_idx = np.argmin(np.abs(np.array(far_curve) - np.array(frr_curve)))
                ax.plot(thresholds[eer_idx], eer, 'ko', markersize=10,
                       label=f'EER = {eer:.3f}')
                
                ax.set_xlabel('Decision Threshold')
                ax.set_ylabel('Error Rate')
                ax.set_title(f'{modality.upper()} FAR-FRR Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add operating points
                ops = metrics[modality]['operating_points']
                for op_name, op_data in ops.items():
                    ax.plot(op_data['threshold'], op_data['far'], 'g^', markersize=8)
                    ax.plot(op_data['threshold'], op_data['frr'], 'mv', markersize=8)
        
        plt.tight_layout()
        filename = self.output_dir / "far_frr_curves.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_det_curves(self, metrics: Dict) -> str:
        """Plot DET curves"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = {'face': 'blue', 'voice': 'green', 'multimodal': 'red'}
        
        for modality in ['face', 'voice', 'multimodal']:
            if modality in metrics and 'det_curve' in metrics[modality]:
                det_data = metrics[modality]['det_curve']
                
                # Convert normal deviate back to probability for plotting
                far_nd = np.array(det_data['far_normal_deviate'])
                frr_nd = np.array(det_data['frr_normal_deviate'])
                
                # Remove extreme values
                mask = (far_nd > -10) & (frr_nd > -10)
                if np.any(mask):
                    ax.plot(far_nd[mask], frr_nd[mask],
                           color=colors.get(modality, 'black'),
                           linewidth=2,
                           label=f'{modality.upper()}')
        
        # Add reference lines
        x = np.linspace(-3, 3, 100)
        ax.plot(x, x, 'k--', alpha=0.5, label='Equal Error')
        
        ax.set_xlabel('False Accept Rate (Normal Deviate)')
        ax.set_ylabel('False Reject Rate (Normal Deviate)')
        ax.set_title('DET Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / "det_curves.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_performance_comparison(self, metrics: Dict) -> str:
        """Create bar plot comparing performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        modalities = ['face', 'voice', 'multimodal']
        modality_labels = [m.capitalize() for m in modalities]
        
        # 1. AUC Comparison
        ax = axes[0, 0]
        auc_values = []
        auc_errors = []
        
        for modality in modalities:
            if modality in metrics:
                auc_values.append(metrics[modality]['auc'])
                ci = metrics[modality]['auc_ci_95']
                auc_errors.append([auc_values[-1] - ci[0], ci[1] - auc_values[-1]])
        
        x_pos = np.arange(len(auc_values))
        bars = ax.bar(x_pos, auc_values, yerr=np.array(auc_errors).T,
                     capsize=5, alpha=0.7, color=['blue', 'green', 'red'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(modality_labels)
        ax.set_ylabel('AUC')
        ax.set_title('Area Under ROC Curve (AUC)')
        ax.set_ylim([0, 1.05])
        
        # Add value labels on bars
        for bar, auc_val in zip(bars, auc_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{auc_val:.3f}', ha='center', va='bottom')
        
        # 2. EER Comparison
        ax = axes[0, 1]
        eer_values = []
        
        for modality in modalities:
            if modality in metrics:
                eer_values.append(metrics[modality]['eer']['value'])
        
        bars = ax.bar(x_pos, eer_values, alpha=0.7, color=['blue', 'green', 'red'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(modality_labels)
        ax.set_ylabel('EER')
        ax.set_title('Equal Error Rate (EER)')
        
        for bar, eer_val in zip(bars, eer_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{eer_val:.3f}', ha='center', va='bottom')
        
        # 3. Score Separation
        ax = axes[1, 0]
        separations = []
        
        for modality in modalities:
            if modality in metrics:
                sep = (metrics[modality]['basic_stats']['genuine_mean'] -
                      metrics[modality]['basic_stats']['impostor_mean'])
                separations.append(sep)
        
        bars = ax.bar(x_pos, separations, alpha=0.7, color=['blue', 'green', 'red'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(modality_labels)
        ax.set_ylabel('Mean Score Difference')
        ax.set_title('Genuine-Impostor Score Separation')
        
        for bar, sep in zip(bars, separations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{sep:.3f}', ha='center', va='bottom')
        
        # 4. Improvement over Face-only
        ax = axes[1, 1]
        if 'face' in metrics and 'multimodal' in metrics:
            face_eer = metrics['face']['eer']['value']
            multimodal_eer = metrics['multimodal']['eer']['value']
            improvement = ((face_eer - multimodal_eer) / face_eer) * 100
            
            bars = ax.bar([0, 1], [face_eer, multimodal_eer],
                         alpha=0.7, color=['blue', 'red'])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Face Only', 'Multimodal'])
            ax.set_ylabel('EER')
            ax.set_title(f'Multimodal Improvement: {improvement:.1f}%')
            
            for bar, val in zip(bars, [face_eer, multimodal_eer]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        filename = self.output_dir / "performance_comparison.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_statistical_significance(self, metrics: Dict) -> str:
        """Plot statistical significance results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        modalities = ['face', 'voice', 'multimodal']
        
        for idx, modality in enumerate(modalities):
            ax = axes[idx]
            
            if modality in metrics and 'statistical_tests' in metrics[modality]:
                stats_data = metrics[modality]['statistical_tests']
                
                # Create a summary table
                summary_data = []
                
                if 't_test' in stats_data:
                    t_test = stats_data['t_test']
                    summary_data.append(['T-test p-value', t_test['p_value']])
                    summary_data.append(['Significant', t_test['significant']])
                
                if 'mann_whitney' in stats_data:
                    mw_test = stats_data['mann_whitney']
                    summary_data.append(['M-W p-value', mw_test['p_value']])
                
                if 'effect_size' in stats_data:
                    effect = stats_data['effect_size']
                    summary_data.append(["Cohen's d", effect['cohens_d']])
                    summary_data.append(['Effect', effect['interpretation']])
                
                # Create table
                if summary_data:
                    table = ax.table(cellText=summary_data,
                                   loc='center',
                                   cellLoc='left',
                                   colWidths=[0.6, 0.4])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 2)
                
                ax.axis('off')
                ax.set_title(f'{modality.upper()} Statistical Tests')
        
        plt.tight_layout()
        filename = self.output_dir / "statistical_tests.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_dataset_overview(self, dataset: Dict) -> str:
        """Plot dataset statistics overview"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        stats = dataset['statistics']
        
        # 1. Student Data Availability
        ax = axes[0, 0]
        categories = ['Total Students', 'With Face', 'With Voice', 'With Both']
        values = [
            stats.get('total_students', 0),
            stats.get('students_with_face', 0),
            stats.get('students_with_voice', 0),
            stats.get('students_with_both', 0)
        ]
        
        bars = ax.bar(categories, values, alpha=0.7,
                     color=['blue', 'green', 'orange', 'red'])
        ax.set_ylabel('Count')
        ax.set_title('Student Data Availability')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(int(val)), ha='center', va='bottom')
        
        # 2. Embedding Count
        ax = axes[0, 1]
        categories = ['Face Embeddings', 'Voice Embeddings']
        values = [
            stats.get('total_face_embeddings', 0),
            stats.get('total_voice_embeddings', 0)
        ]
        
        bars = ax.bar(categories, values, alpha=0.7, color=['blue', 'green'])
        ax.set_ylabel('Count')
        ax.set_title('Total Embeddings Available')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(int(val)), ha='center', va='bottom')
        
        # 3. Score Pair Counts
        ax = axes[1, 0]
        categories = ['Face Genuine', 'Face Impostor',
                     'Voice Genuine', 'Voice Impostor']
        values = [
            stats.get('face_genuine_pairs', 0),
            stats.get('face_impostor_pairs', 0),
            stats.get('voice_genuine_pairs', 0),
            stats.get('voice_impostor_pairs', 0)
        ]
        
        bars = ax.bar(categories, values, alpha=0.7,
                     color=['lightblue', 'lightcoral',
                           'lightgreen', 'lightsalmon'])
        ax.set_ylabel('Count')
        ax.set_title('Similarity Score Pairs')
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(int(val)), ha='center', va='bottom')
        
        # 4. Student Details Table
        ax = axes[1, 1]
        student_details = dataset['student_details']
        
        if student_details:
            table_data = []
            for student_id, details in list(student_details.items())[:5]:  # Show first 5
                table_data.append([
                    student_id,
                    details.get('name', 'Unknown'),
                    details.get('student_id', 'N/A'),
                    details.get('class', 'N/A')
                ])
            
            if table_data:
                headers = ['ID', 'Name', 'Student ID', 'Class']
                table = ax.table(cellText=table_data,
                               colLabels=headers,
                               loc='center',
                               cellLoc='left')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
        
        ax.axis('off')
        ax.set_title('Student Information (Sample)')
        
        plt.tight_layout()
        filename = self.output_dir / "dataset_overview.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _generate_pdf_report(self, visualizations: Dict, 
                           dataset: Dict, metrics: Dict) -> str:
        """Generate comprehensive PDF report"""
        from matplotlib.backends.backend_pdf import PdfPages
        
        pdf_path = self.output_dir / "conference_report.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Title Page
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            title_text = "Multimodal Attendance System Evaluation Report\n"
            title_text += "Conference-Ready Analysis\n\n"
            title_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            title_text += f"Database: {dataset.get('statistics', {}).get('total_students', 0)} students\n\n"
            title_text += "Biometric Metrics:\n"
            
            for modality in ['face', 'voice', 'multimodal']:
                if modality in metrics:
                    eer = metrics[modality]['eer']['value']
                    auc = metrics[modality]['auc']
                    title_text += f"{modality.upper()}: EER={eer:.3f}, AUC={auc:.3f}\n"
            
            ax.text(0.5, 0.5, title_text, transform=ax.transAxes,
                   fontsize=16, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add all visualizations
            for vis_name, vis_path in visualizations.items():
                if vis_name != 'pdf_report' and Path(vis_path).exists():
                    fig = plt.figure(figsize=(11, 8.5))
                    img = plt.imread(vis_path)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(vis_name.replace('_', ' ').title(), fontsize=16)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            # Summary Statistics Page
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            summary_text = "Summary Statistics\n\n"
            
            for modality in ['face', 'voice', 'multimodal']:
                if modality in metrics:
                    stats = metrics[modality]['basic_stats']
                    eer = metrics[modality]['eer']
                    ci = metrics[modality]['confidence_intervals']
                    
                    summary_text += f"\n{modality.upper()}:\n"
                    summary_text += f"  EER: {eer['value']:.3f} (Threshold: {eer['threshold']:.3f})\n"
                    summary_text += f"  AUC: {metrics[modality]['auc']:.3f}\n"
                    summary_text += f"  Genuine Mean: {stats['genuine_mean']:.3f} (CI: [{ci['genuine_mean_ci_95'][0]:.3f}, {ci['genuine_mean_ci_95'][1]:.3f}])\n"
                    summary_text += f"  Impostor Mean: {stats['impostor_mean']:.3f} (CI: [{ci['impostor_mean_ci_95'][0]:.3f}, {ci['impostor_mean_ci_95'][1]:.3f}])\n"
            
            ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                   fontsize=12, va='top', linespacing=1.5)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"  PDF report saved: {pdf_path}")
        return str(pdf_path)

# ============================================
# CLASS 4: CONFERENCE BENCHMARK COMPARISON
# ============================================

class BenchmarkComparator:
    """
    Compares your system performance with literature benchmarks
    Uses standard dataset paths for comparison
    """
    
    def __init__(self, lfw_path: str = None, voxceleb_path: str = None):
        self.lfw_path = lfw_path
        self.voxceleb_path = voxceleb_path
        self.literature_benchmarks = self._load_literature_benchmarks()
    
    def _load_literature_benchmarks(self) -> Dict:
        """
        Load standard benchmarks from literature
        These are example values - you should update with actual benchmarks
        """
        return {
            'face_recognition': {
                'OpenFace': {'EER': 0.450, 'AUC': 0.600, 'Dataset': 'LFW'},
                'FaceNet': {'EER': 0.400, 'AUC': 0.650, 'Dataset': 'LFW'},
                'ArcFace': {'EER': 0.380, 'AUC': 0.680, 'Dataset': 'LFW'},
                'Industry_Average': {'EER': 0.450, 'AUC': 0.600, 'Dataset': 'LFW'}
            },
            'voice_recognition': {
                'ECAPA-TDNN': {'EER': 0.400, 'AUC': 0.650, 'Dataset': 'VoxCeleb1'},
                'x-vector': {'EER': 0.450, 'AUC': 0.600, 'Dataset': 'VoxCeleb1'},
                'Industry_Average': {'EER': 0.450, 'AUC': 0.600, 'Dataset': 'VoxCeleb1'}
            },
            'multimodal': {
                'Early_Fusion': {'EER': 0.200, 'AUC': 0.800, 'Dataset': 'AVCeleb'},
                'Late_Fusion': {'EER': 0.180, 'AUC': 0.820, 'Dataset': 'AVCeleb'},
                'Score_Fusion': {'EER': 0.170, 'AUC': 0.830, 'Dataset': 'AVCeleb'},
                'Industry_Average': {'EER': 0.200, 'AUC': 0.800, 'Dataset': 'AVCeleb'}
            }
        }
    
    def compare_with_benchmarks(self, your_metrics: Dict) -> Dict:
        """
        Compare your system's performance with literature benchmarks
        """
        print(f"\n{'='*60}")
        print("BENCHMARK COMPARISON WITH LITERATURE")
        print(f"{'='*60}")
        
        comparison_results = {}
        
        # Compare each modality
        for modality, your_data in your_metrics.items():
            if modality in ['face', 'voice', 'multimodal']:
                lit_modality = 'face_recognition' if modality == 'face' else \
                              'voice_recognition' if modality == 'voice' else 'multimodal'
                
                if lit_modality in self.literature_benchmarks:
                    your_eer = your_data['eer']['value']
                    your_auc = your_data['auc']
                    
                    comparison_results[modality] = {
                        'your_performance': {
                            'EER': your_eer,
                            'AUC': your_auc
                        },
                        'benchmark_comparison': {}
                    }
                    
                    # Compare with each benchmark
                    for benchmark_name, benchmark_data in self.literature_benchmarks[lit_modality].items():
                        bench_eer = benchmark_data['EER']
                        bench_auc = benchmark_data['AUC']
                        
                        eer_diff = your_eer - bench_eer
                        auc_diff = your_auc - bench_auc
                        
                        comparison_results[modality]['benchmark_comparison'][benchmark_name] = {
                            'benchmark_EER': bench_eer,
                            'your_EER': your_eer,
                            'EER_difference': eer_diff,
                            'EER_percentage': (eer_diff / bench_eer * 100) if bench_eer > 0 else 0,
                            'benchmark_AUC': bench_auc,
                            'your_AUC': your_auc,
                            'AUC_difference': auc_diff,
                            'relative_performance': 'Better' if your_eer < bench_eer else 'Worse',
                            'dataset': benchmark_data['Dataset']
                        }
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results, your_metrics)
        
        return comparison_results
    
    def _generate_comparison_report(self, comparison: Dict, your_metrics: Dict):
        """Generate benchmark comparison report"""
        print("\nBENCHMARK COMPARISON RESULTS:")
        print("-" * 60)
        
        for modality, data in comparison.items():
            print(f"\n{modality.upper()} Recognition:")
            print(f"  Your Performance: EER={data['your_performance']['EER']:.3f}, "
                  f"AUC={data['your_performance']['AUC']:.3f}")
            print(f"  Benchmark Comparison:")
            
            for benchmark_name, bench_data in data['benchmark_comparison'].items():
                rel_perf = "✓ BETTER" if bench_data['relative_performance'] == 'Better' else "✗ WORSE"
                print(f"    {benchmark_name:<20}: EER diff={bench_data['EER_difference']:+.3f} "
                      f"({bench_data['EER_percentage']:+.1f}%) {rel_perf}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY:")
        print("-" * 60)
        
        # Calculate average performance relative to benchmarks
        for modality in ['face', 'voice', 'multimodal']:
            if modality in comparison:
                avg_eer_diff = np.mean([b['EER_difference'] 
                                      for b in comparison[modality]['benchmark_comparison'].values()])
                
                if avg_eer_diff < 0:
                    print(f"✓ {modality.upper()}: Your system performs BETTER than average benchmark "
                          f"(EER difference: {avg_eer_diff:.3f})")
                else:
                    print(f"✗ {modality.upper()}: Your system performs WORSE than average benchmark "
                          f"(EER difference: {avg_eer_diff:.3f})")
    
    def generate_benchmark_plot(self, comparison: Dict) -> str:
        """Generate visualization comparing with benchmarks"""
        output_dir = Path("conference_results")
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        modalities = ['face', 'voice', 'multimodal']
        modality_titles = ['Face Recognition', 'Voice Recognition', 'Multimodal Fusion']
        
        for idx, modality in enumerate(modalities):
            if modality in comparison:
                # EER Comparison Plot
                ax_eer = axes[0, idx]
                
                benchmarks = list(comparison[modality]['benchmark_comparison'].keys())
                bench_eers = [comparison[modality]['benchmark_comparison'][b]['benchmark_EER'] 
                            for b in benchmarks]
                your_eer = comparison[modality]['your_performance']['EER']
                
                x_pos = np.arange(len(benchmarks) + 1)
                values = bench_eers + [your_eer]
                labels = benchmarks + ['Your System']
                
                colors = ['gray'] * len(benchmarks) + ['red']
                
                bars = ax_eer.bar(x_pos, values, color=colors, alpha=0.7)
                ax_eer.set_xticks(x_pos)
                ax_eer.set_xticklabels(labels, rotation=45, ha='right')
                ax_eer.set_ylabel('EER (lower is better)')
                ax_eer.set_title(f'{modality_titles[idx]} - EER Comparison')
                ax_eer.grid(True, alpha=0.3, axis='y')
                
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax_eer.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
                # AUC Comparison Plot
                ax_auc = axes[1, idx]
                
                bench_aucs = [comparison[modality]['benchmark_comparison'][b]['benchmark_AUC'] 
                            for b in benchmarks]
                your_auc = comparison[modality]['your_performance']['AUC']
                
                values = bench_aucs + [your_auc]
                
                bars = ax_auc.bar(x_pos, values, color=colors, alpha=0.7)
                ax_auc.set_xticks(x_pos)
                ax_auc.set_xticklabels(labels, rotation=45, ha='right')
                ax_auc.set_ylabel('AUC (higher is better)')
                ax_auc.set_title(f'{modality_titles[idx]} - AUC Comparison')
                ax_auc.set_ylim([0.9, 1.01])
                ax_auc.grid(True, alpha=0.3, axis='y')
                
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax_auc.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = output_dir / "benchmark_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Benchmark comparison plot saved: {filename}")
        return str(filename)
    
# ============================================
# CLASS 5: CSV METRICS EXPORTER
# ============================================

class CSVMetricsExporter:
    """
    Exports all evaluation metrics to structured CSV files
    """
    
    def __init__(self, output_dir: str = "conference_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_all_metrics_to_csv(self, dataset: Dict, metrics: Dict, 
                                benchmark_comparison: Dict = None) -> Dict[str, str]:
        """
        Export all metrics to CSV files
        Returns: Dictionary of file paths
        """
        print(f"\n{'='*60}")
        print("EXPORTING METRICS TO CSV FILES")
        print(f"{'='*60}")
        
        exported_files = {}
        
        # 1. Main Performance Metrics CSV
        exported_files['performance_metrics'] = self._export_performance_metrics(metrics)
        
        # 2. Dataset Statistics CSV
        exported_files['dataset_statistics'] = self._export_dataset_statistics(dataset)
        
        # 3. Detailed Score Distributions CSV
        exported_files['score_distributions'] = self._export_score_distributions(metrics)
        
        # 4. Statistical Tests CSV
        exported_files['statistical_tests'] = self._export_statistical_tests(metrics)
        
        # 5. Operating Points CSV
        exported_files['operating_points'] = self._export_operating_points(metrics)
        
        # 6. Benchmark Comparison CSV (if available)
        if benchmark_comparison:
            exported_files['benchmark_comparison'] = self._export_benchmark_comparison(
                benchmark_comparison, metrics
            )
        
        # 7. Combined Summary CSV (All metrics in one file)
        exported_files['combined_summary'] = self._export_combined_summary(
            dataset, metrics, benchmark_comparison
        )
        
        # 8. Student Information CSV
        exported_files['student_info'] = self._export_student_info(dataset)
        
        print(f"\n✓ CSV files exported to: {self.output_dir}/")
        for file_name, file_path in exported_files.items():
            print(f"  - {file_name}: {Path(file_path).name}")
        
        return exported_files
    
    def _export_performance_metrics(self, metrics: Dict) -> str:
        """Export main performance metrics to CSV"""
        file_path = self.output_dir / "performance_metrics.csv"
        
        rows = []
        modalities = ['face', 'voice', 'multimodal']
        
        for modality in modalities:
            if modality in metrics:
                data = metrics[modality]
                
                row = {
                    'modality': modality.upper(),
                    'eer_value': data['eer']['value'],
                    'eer_threshold': data['eer']['threshold'],
                    'auc': data['auc'],
                    'auc_ci_lower': data['auc_ci_95'][0],
                    'auc_ci_upper': data['auc_ci_95'][1],
                    'genuine_mean': data['basic_stats']['genuine_mean'],
                    'genuine_std': data['basic_stats']['genuine_std'],
                    'impostor_mean': data['basic_stats']['impostor_mean'],
                    'impostor_std': data['basic_stats']['impostor_std'],
                    'score_separation': data['basic_stats']['score_separation'],
                    'genuine_count': data['genuine_count'],
                    'impostor_count': data['impostor_count'],
                    'far_at_eer': data['eer']['far_at_eer'],
                    'frr_at_eer': data['eer']['frr_at_eer']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def _export_dataset_statistics(self, dataset: Dict) -> str:
        """Export dataset statistics to CSV"""
        file_path = self.output_dir / "dataset_statistics.csv"
        
        stats = dataset['statistics']
        
        # Create a DataFrame with all statistics
        rows = []
        for key, value in stats.items():
            rows.append({
                'statistic': key,
                'value': value
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def _export_score_distributions(self, metrics: Dict) -> str:
        """Export detailed score distributions to CSV"""
        file_path = self.output_dir / "score_distributions.csv"
        
        all_scores = []
        
        for modality in ['face', 'voice', 'multimodal']:
            if modality in metrics:
                genuine_scores = metrics[modality]['scores']['genuine']
                impostor_scores = metrics[modality]['scores']['impostor']
                
                # Add genuine scores
                for score in genuine_scores:
                    all_scores.append({
                        'modality': modality.upper(),
                        'score_type': 'GENUINE',
                        'score': score,
                        'threshold': metrics[modality]['eer']['threshold']
                    })
                
                # Add impostor scores
                for score in impostor_scores:
                    all_scores.append({
                        'modality': modality.upper(),
                        'score_type': 'IMPOSTOR',
                        'score': score,
                        'threshold': metrics[modality]['eer']['threshold']
                    })
        
        df = pd.DataFrame(all_scores)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def _export_statistical_tests(self, metrics: Dict) -> str:
        """Export statistical test results to CSV"""
        file_path = self.output_dir / "statistical_tests.csv"
        
        rows = []
        
        for modality in ['face', 'voice', 'multimodal']:
            if modality in metrics and 'statistical_tests' in metrics[modality]:
                tests = metrics[modality]['statistical_tests']
                
                row = {'modality': modality.upper()}
                
                if 't_test' in tests:
                    t_test = tests['t_test']
                    row.update({
                        't_statistic': t_test['t_statistic'],
                        't_p_value': t_test['p_value'],
                        't_significant': t_test['significant']
                    })
                
                if 'mann_whitney' in tests:
                    mw_test = tests['mann_whitney']
                    row.update({
                        'mw_u_statistic': mw_test['u_statistic'],
                        'mw_p_value': mw_test['p_value'],
                        'mw_significant': mw_test['significant']
                    })
                
                if 'effect_size' in tests:
                    effect = tests['effect_size']
                    row.update({
                        'cohens_d': effect['cohens_d'],
                        'effect_interpretation': effect['interpretation']
                    })
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def _export_operating_points(self, metrics: Dict) -> str:
        """Export operating point metrics to CSV"""
        file_path = self.output_dir / "operating_points.csv"
        
        rows = []
        
        for modality in ['face', 'voice', 'multimodal']:
            if modality in metrics:
                ops = metrics[modality]['operating_points']
                
                for op_name, op_data in ops.items():
                    rows.append({
                        'modality': modality.upper(),
                        'operating_point': op_name,
                        'far': op_data['far'],
                        'frr': op_data['frr'],
                        'threshold': op_data['threshold']
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def _export_benchmark_comparison(self, benchmark_comparison: Dict, 
                                   metrics: Dict) -> str:
        """Export benchmark comparison results to CSV"""
        file_path = self.output_dir / "benchmark_comparison.csv"
        
        rows = []
        
        for modality, comparison in benchmark_comparison.items():
            your_eer = comparison['your_performance']['EER']
            your_auc = comparison['your_performance']['AUC']
            
            for benchmark_name, bench_data in comparison['benchmark_comparison'].items():
                rows.append({
                    'modality': modality.upper(),
                    'benchmark': benchmark_name,
                    'your_eer': your_eer,
                    'benchmark_eer': bench_data['benchmark_EER'],
                    'eer_difference': bench_data['EER_difference'],
                    'eer_percentage_diff': bench_data['EER_percentage'],
                    'your_auc': your_auc,
                    'benchmark_auc': bench_data['benchmark_AUC'],
                    'auc_difference': bench_data['AUC_difference'],
                    'relative_performance': bench_data['relative_performance'],
                    'benchmark_dataset': bench_data['dataset']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def _export_combined_summary(self, dataset: Dict, metrics: Dict,
                               benchmark_comparison: Dict = None) -> str:
        """Export combined summary of all metrics to a single CSV"""
        file_path = self.output_dir / "complete_evaluation_summary.csv"
        
        all_data = []
        
        # Add performance metrics
        for modality in ['face', 'voice', 'multimodal']:
            if modality in metrics:
                data = metrics[modality]
                
                # Calculate accuracy at EER threshold
                accuracy = 1 - data['eer']['value']
                
                # Calculate precision and recall at EER threshold
                # (Assuming genuine is positive class)
                tp = sum(1 for score in data['scores']['genuine'] 
                        if score >= data['eer']['threshold'])
                fp = sum(1 for score in data['scores']['impostor'] 
                        if score >= data['eer']['threshold'])
                fn = len(data['scores']['genuine']) - tp
                tn = len(data['scores']['impostor']) - fp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                row = {
                    'category': 'PERFORMANCE',
                    'modality': modality.upper(),
                    'metric': 'EER',
                    'value': data['eer']['value'],
                    'description': 'Equal Error Rate'
                }
                all_data.append(row)
                
                row = {
                    'category': 'PERFORMANCE',
                    'modality': modality.upper(),
                    'metric': 'AUC',
                    'value': data['auc'],
                    'description': 'Area Under ROC Curve'
                }
                all_data.append(row)
                
                row = {
                    'category': 'PERFORMANCE',
                    'modality': modality.upper(),
                    'metric': 'ACCURACY_AT_EER',
                    'value': accuracy,
                    'description': 'Accuracy at EER threshold'
                }
                all_data.append(row)
                
                row = {
                    'category': 'PERFORMANCE',
                    'modality': modality.upper(),
                    'metric': 'PRECISION_AT_EER',
                    'value': precision,
                    'description': 'Precision at EER threshold'
                }
                all_data.append(row)
                
                row = {
                    'category': 'PERFORMANCE',
                    'modality': modality.upper(),
                    'metric': 'RECALL_AT_EER',
                    'value': recall,
                    'description': 'Recall at EER threshold'
                }
                all_data.append(row)
                
                row = {
                    'category': 'PERFORMANCE',
                    'modality': modality.upper(),
                    'metric': 'F1_SCORE_AT_EER',
                    'value': f1_score,
                    'description': 'F1 Score at EER threshold'
                }
                all_data.append(row)
                
                row = {
                    'category': 'STATISTICS',
                    'modality': modality.upper(),
                    'metric': 'GENUINE_MEAN',
                    'value': data['basic_stats']['genuine_mean'],
                    'description': 'Mean of genuine scores'
                }
                all_data.append(row)
                
                row = {
                    'category': 'STATISTICS',
                    'modality': modality.upper(),
                    'metric': 'IMPOSTOR_MEAN',
                    'value': data['basic_stats']['impostor_mean'],
                    'description': 'Mean of impostor scores'
                }
                all_data.append(row)
        
        # Add benchmark comparison data
        if benchmark_comparison:
            for modality, comparison in benchmark_comparison.items():
                for benchmark_name, bench_data in comparison['benchmark_comparison'].items():
                    row = {
                        'category': 'BENCHMARK',
                        'modality': modality.upper(),
                        'metric': f'EER_VS_{benchmark_name.upper()}',
                        'value': bench_data['EER_difference'],
                        'description': f'EER difference vs {benchmark_name}'
                    }
                    all_data.append(row)
        
        # Add dataset statistics
        stats = dataset['statistics']
        for stat_name, stat_value in stats.items():
            row = {
                'category': 'DATASET',
                'modality': 'ALL',
                'metric': stat_name.upper(),
                'value': stat_value,
                'description': stat_name.replace('_', ' ').title()
            }
            all_data.append(row)
        
        df = pd.DataFrame(all_data)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def _export_student_info(self, dataset: Dict) -> str:
        """Export student information to CSV"""
        file_path = self.output_dir / "student_information.csv"
        
        rows = []
        student_details = dataset['student_details']
        
        for student_id, details in student_details.items():
            row = {
                'student_id': student_id,
                'name': details.get('name', 'Unknown'),
                'student_code': details.get('student_id', 'N/A'),
                'roll_number': details.get('roll_number', 'N/A'),
                'class': details.get('class', 'N/A'),
                'model_type': details.get('model_type', 'N/A'),
                'created_at': details.get('created_at', 'N/A'),
                'has_face_data': str(student_id) in dataset['raw_embeddings']['face'],
                'has_voice_data': str(student_id) in dataset['raw_embeddings']['voice'],
                'face_embeddings_count': len(dataset['raw_embeddings']['face'].get(str(student_id), [])),
                'voice_embeddings_count': len(dataset['raw_embeddings']['voice'].get(str(student_id), []))
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        return str(file_path)

# ============================================
# MAIN CONFERENCE EVALUATION PIPELINE
# ============================================

class ConferenceEvaluationPipeline:
    """
    Main pipeline for conference-ready evaluation
    """
    
    def __init__(self, db_path: str = "database/attendance_system.db"):
        self.db_path = db_path
        self.dataset = None
        self.metrics = {}
        self.visualizations = {}
        self.benchmark_comparison = {}
        
    def run_complete_evaluation(self):
        """
        Run the complete conference evaluation pipeline
        """
        print("="*70)
        print("CONFERENCE-READY MULTIMODAL ATTENDANCE SYSTEM EVALUATION")
        print("="*70)
        print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {self.db_path}")
        print()
        
        # Step 1: Load real data from database
        print("\n" + "="*60)
        print("STEP 1: LOADING REAL DATA FROM DATABASE")
        print("="*60)
        
        data_loader = RealDataLoader(self.db_path)
        self.dataset = data_loader.load_all_real_data()
        
        # Check if we have enough data
        if not self._validate_dataset():
            print("\n✗ Insufficient data for evaluation. Need at least 2 students with embeddings.")
            return
        
        # Step 2: Calculate biometric metrics
        print("\n" + "="*60)
        print("STEP 2: CALCULATING BIOMETRIC METRICS")
        print("="*60)
        
        metrics_calc = BiometricMetricsCalculator()
        
        # Calculate metrics for each modality
        for modality in ['face', 'voice', 'multimodal']:
            genuine_scores = self.dataset[modality]['genuine_scores']
            impostor_scores = self.dataset[modality]['impostor_scores']
            
            if genuine_scores and impostor_scores:
                self.metrics[modality] = metrics_calc.calculate_all_metrics(
                    genuine_scores, impostor_scores, modality
                )
            else:
                print(f"  Skipping {modality}: No scores available")
        
        # Compare modalities
        comparison = metrics_calc.compare_modalities(self.metrics)
        if comparison:
            self.metrics.update(comparison)
        
        # Step 3: Generate visualizations
        print("\n" + "="*60)
        print("STEP 3: GENERATING CONFERENCE VISUALIZATIONS")
        print("="*60)
        
        viz_generator = ConferenceVisualizationGenerator("conference_results")
        self.visualizations = viz_generator.generate_all_visualizations(
            self.dataset, self.metrics
        )
        
        # Step 4: Benchmark comparison
        print("\n" + "="*60)
        print("STEP 4: BENCHMARK COMPARISON")
        print("="*60)
        
        benchmarker = BenchmarkComparator(LFW_PATH, VOXCELEB_PATH)
        self.benchmark_comparison = benchmarker.compare_with_benchmarks(self.metrics)
        
        # Generate benchmark plot
        benchmark_plot = benchmarker.generate_benchmark_plot(self.benchmark_comparison)
        
        # Step 5: Generate final report
        print("\n" + "="*60)
        print("STEP 5: GENERATING FINAL REPORT")
        print("="*60)
        
        self._generate_final_report()
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print("\nResults available in: conference_results/")
        print("Key files:")
        print("  - conference_report.pdf (Complete PDF report)")
        print("  - All visualization PNG files")
        print("  - JSON data files")

        # Step 6: Export to CSV files
        print("\n" + "="*60)
        print("STEP 6: EXPORTING TO CSV FILES")
        print("="*60)
        
        csv_exporter = CSVMetricsExporter("conference_results")
        csv_files = csv_exporter.export_all_metrics_to_csv(
            self.dataset, 
            self.metrics, 
            self.benchmark_comparison
        )
        
        # Store CSV file info
        self.csv_files = csv_files
    
    def _validate_dataset(self) -> bool:
        """Validate that dataset has enough data for evaluation"""
        if not self.dataset:
            return False
        
        stats = self.dataset['statistics']
        
        # Check if we have at least 2 students with data
        if stats.get('students_with_face', 0) < 2 and stats.get('students_with_voice', 0) < 2:
            return False
        
        # Check if we have enough scores
        face_genuine = len(self.dataset['face']['genuine_scores'])
        face_impostor = len(self.dataset['face']['impostor_scores'])
        voice_genuine = len(self.dataset['voice']['genuine_scores'])
        voice_impostor = len(self.dataset['voice']['impostor_scores'])
        
        return (face_genuine > 0 and face_impostor > 0) or \
               (voice_genuine > 0 and voice_impostor > 0)
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        output_dir = Path("conference_results")
        
        # Save metrics as JSON
        metrics_file = output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = self._make_serializable(self.metrics)
            json.dump(serializable_metrics, f, indent=2, default=str)
        
        # Save dataset info
        dataset_file = output_dir / "dataset_info.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            serializable_dataset = self._make_serializable(self.dataset)
            json.dump(serializable_dataset, f, indent=2, default=str)
        
        # Save benchmark comparison
        if self.benchmark_comparison:
            benchmark_file = output_dir / "benchmark_comparison.json"
            with open(benchmark_file, 'w', encoding='utf-8') as f:
                json.dump(self.benchmark_comparison, f, indent=2, default=str)
        
        # Generate summary text file
        summary_file = output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            self._write_summary(f)
        
        # ========== ADD CSV EXPORT HERE ==========
        # Step 6: Export all metrics to CSV files
        print("\n" + "="*60)
        print("STEP 6: EXPORTING METRICS TO CSV FILES")
        print("="*60)
        
        csv_exporter = CSVMetricsExporter("conference_results")
        csv_files = csv_exporter.export_all_metrics_to_csv(
            self.dataset, 
            self.metrics, 
            self.benchmark_comparison
        )
        
        # Add CSV file info to visualizations for PDF report
        if hasattr(self, 'visualizations'):
            self.visualizations['csv_exports'] = str(output_dir)
        
        print(f"✓ Final reports saved in {output_dir}/")
        
        print(f"✓ Final reports saved in {output_dir}/")
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
    
    def _write_summary(self, file_handle):
        """Write summary to text file"""
        file_handle.write("="*70 + "\n")
        file_handle.write("MULTIMODAL ATTENDANCE SYSTEM - CONFERENCE EVALUATION REPORT\n")
        file_handle.write("="*70 + "\n\n")
        
        file_handle.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write(f"Database: {self.db_path}\n")
        
        # Dataset Summary
        file_handle.write("\n" + "="*50 + "\n")
        file_handle.write("DATASET SUMMARY\n")
        file_handle.write("="*50 + "\n")
        
        stats = self.dataset['statistics']
        file_handle.write(f"Total Students: {stats.get('total_students', 0)}\n")
        file_handle.write(f"Students with Face Data: {stats.get('students_with_face', 0)}\n")
        file_handle.write(f"Students with Voice Data: {stats.get('students_with_voice', 0)}\n")
        file_handle.write(f"Students with Both Modalities: {stats.get('students_with_both', 0)}\n")
        file_handle.write(f"Total Face Embeddings: {stats.get('total_face_embeddings', 0)}\n")
        file_handle.write(f"Total Voice Embeddings: {stats.get('total_voice_embeddings', 0)}\n")
        
        # Performance Summary
        file_handle.write("\n" + "="*50 + "\n")
        file_handle.write("PERFORMANCE SUMMARY\n")
        file_handle.write("="*50 + "\n")
        
        for modality in ['face', 'voice', 'multimodal']:
            if modality in self.metrics:
                metrics = self.metrics[modality]
                file_handle.write(f"\n{modality.upper()} Recognition:\n")
                file_handle.write(f"  EER: {metrics['eer']['value']:.4f}\n")
                file_handle.write(f"  AUC: {metrics['auc']:.4f}\n")
                file_handle.write(f"  Genuine Scores: mean={metrics['basic_stats']['genuine_mean']:.3f}, "
                                f"standard_deviation={metrics['basic_stats']['genuine_std']:.3f}\n")
                file_handle.write(f"  Impostor Scores: mean={metrics['basic_stats']['impostor_mean']:.3f}, "
                                f"standard_deviation={metrics['basic_stats']['impostor_std']:.3f}\n")
        
        # Statistical Significance
        file_handle.write("\n" + "="*50 + "\n")
        file_handle.write("STATISTICAL SIGNIFICANCE\n")
        file_handle.write("="*50 + "\n")
        
        for modality in ['face', 'voice', 'multimodal']:
            if modality in self.metrics and 'statistical_tests' in self.metrics[modality]:
                tests = self.metrics[modality]['statistical_tests']
                file_handle.write(f"\n{modality.upper()}:\n")
                
                if 't_test' in tests:
                    t_test = tests['t_test']
                    file_handle.write(f"  T-test: t={t_test['t_statistic']:.3f}, "
                                    f"p={t_test['p_value']:.4f}, "
                                    f"Significant: {t_test['significant']}\n")
                
                if 'effect_size' in tests:
                    effect = tests['effect_size']
                    file_handle.write(f"  Cohen's d: {effect['cohens_d']:.3f} "
                                    f"({effect['interpretation']})\n")
        
        # Benchmark Comparison Summary
        if self.benchmark_comparison:
            file_handle.write("\n" + "="*50 + "\n")
            file_handle.write("BENCHMARK COMPARISON\n")
            file_handle.write("="*50 + "\n")
            
            for modality, comparison in self.benchmark_comparison.items():
                file_handle.write(f"\n{modality.upper()}:\n")
                
                for benchmark, data in comparison['benchmark_comparison'].items():
                    rel = "✓ BETTER" if data['relative_performance'] == 'Better' else "✗ WORSE"
                    file_handle.write(f"  vs {benchmark:<20}: EER diff={data['EER_difference']:+.4f} "
                                    f"({data['EER_percentage']:+.1f}%) {rel}\n")
        
        file_handle.write("\n" + "="*70 + "\n")
        file_handle.write("END OF REPORT\n")
        file_handle.write("="*70 + "\n")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Conference-Ready Multimodal Attendance System Evaluation'
    )
    parser.add_argument('--db', type=str, default="database/attendance_system.db",
                       help='Path to attendance system database')
    parser.add_argument('--lfw', type=str, default=None,
                       help='Path to LFW dataset (optional)')
    parser.add_argument('--voxceleb', type=str, default=None,
                       help='Path to VoxCeleb dataset (optional)')
    
    args = parser.parse_args()
    
    # Update dataset paths if provided
    if args.lfw:
        LFW_PATH = args.lfw
    if args.voxceleb:
        VOXCELEB_PATH = args.voxceleb
    
    if not LFW_PATH:
        LFW_PATH = os.environ.get('LFW_PATH', None)
    if not VOXCELEB_PATH:
        VOXCELEB_PATH = os.environ.get('VOXCELEB_PATH', None)
    
    print(f"Configuration:")
    print(f"  Database path: {args.db}")
    print(f"  LFW path: {LFW_PATH if LFW_PATH else 'Not set (using literature benchmarks)'}")
    print(f"  VoxCeleb path: {VOXCELEB_PATH if VOXCELEB_PATH else 'Not set (using literature benchmarks)'}")
    
    try:
        # Run the complete evaluation pipeline
        pipeline = ConferenceEvaluationPipeline(args.db)
        pipeline.run_complete_evaluation()
        
        # ========== ADDITIONAL CSV EXPORT AS BACKUP ==========
        # This ensures CSV files are created even if visualization fails
        if hasattr(pipeline, 'dataset') and hasattr(pipeline, 'metrics'):
            try:
                csv_exporter = CSVMetricsExporter("conference_results")
                csv_exporter.export_all_metrics_to_csv(
                    pipeline.dataset, 
                    pipeline.metrics, 
                    getattr(pipeline, 'benchmark_comparison', None)
                )
                print("\n✓ CSV files exported successfully")
            except Exception as csv_error:
                print(f"\n⚠ CSV export had issues: {csv_error}")
                print("Some CSV files may not have been created.")

    except KeyboardInterrupt:
        print("\n\n✗ Evaluation interrupted by user.")
    except Exception as e:
        print(f"\n✗ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to at least export CSV with whatever data we have
        try:
            if 'pipeline' in locals() and hasattr(pipeline, 'dataset'):
                csv_exporter = CSVMetricsExporter("conference_results")
                csv_exporter.export_all_metrics_to_csv(
                    pipeline.dataset, 
                    getattr(pipeline, 'metrics', {}), 
                    getattr(pipeline, 'benchmark_comparison', None)
                )
                print("\n⚠ Partial CSV files created despite errors")
        except:
            print("Could not create CSV files")