"""
SchedBuddy OCR + ML Prediction System
Combines PyTesseract OCR with ML-based course name prediction
"""

import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
from difflib import SequenceMatcher
from collections import defaultdict
import json
import pickle

# ====================
# 1. OCR PROCESSING MODULE
# ====================

class CORImageProcessor:
    """Handles image preprocessing and OCR extraction"""
    
    def __init__(self):
        # OCR Engine Mode, Page Segmentation Mode
        self.config = '--oem 3 --psm 6'  # LSTM OCR Engine, assume uniform text block
    
    def preprocess_image(self, image_path):
        """
        Enhance image quality before OCR
        Returns: preprocessed PIL Image
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Deskew if needed
        coords = np.column_stack(np.where(denoised > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:  # Only rotate if significant skew
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(
                denoised, M, (w, h), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_REPLICATE
            )
        
        # Convert back to PIL for Tesseract
        return Image.fromarray(denoised)
    
    def extract_text(self, image_path):
        """
        Perform OCR on preprocessed image
        Returns: raw text string
        """
        processed_img = self.preprocess_image(image_path)
        text = pytesseract.image_to_string(processed_img, config=self.config)
        return text
    
    def parse_cor_structure(self, raw_text):
        """
        Parse OCR text into structured course data
        Returns: list of course dictionaries
        """
        courses = []
        lines = raw_text.split('\n')
        
        # Common BU COR patterns (adjust based on actual format)
        course_code_pattern = r'[A-Z]{2,4}\s*\d{3,4}[A-Z]?'
        time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\s*-\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))'
        days_pattern = r'\b(M|T|W|Th|F|S|MTh|TF|MWF|MW|TTh)\b'
        
        current_course = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match course code
            code_match = re.search(course_code_pattern, line)
            if code_match:
                if current_course:
                    courses.append(current_course)
                current_course = {
                    'code': code_match.group(0),
                    'raw_line': line
                }
            
            # Extract time
            time_match = re.search(time_pattern, line)
            if time_match and current_course:
                current_course['time_start'] = time_match.group(1)
                current_course['time_end'] = time_match.group(2)
            
            # Extract days
            days_match = re.search(days_pattern, line)
            if days_match and current_course:
                current_course['days'] = days_match.group(1)
            
            # Extract room (typically ends with numbers)
            room_match = re.search(r'(?:Room|Rm\.?|Bldg\.?)\s*([A-Z0-9\-]+)', line, re.IGNORECASE)
            if room_match and current_course:
                current_course['room'] = room_match.group(1)
            
            # Course title (usually longest capitalized string)
            if current_course and 'title' not in current_course:
                # Remove known parts and keep remainder as title
                title_candidate = re.sub(course_code_pattern, '', line)
                title_candidate = re.sub(time_pattern, '', title_candidate)
                title_candidate = re.sub(days_pattern, '', title_candidate)
                title_candidate = title_candidate.strip()
                if len(title_candidate) > 5:  # Reasonable title length
                    current_course['title'] = title_candidate
        
        if current_course:
            courses.append(current_course)
        
        return courses


# ====================
# 2. ML PREDICTION MODULE
# ====================

class CourseNamePredictor:
    """
    ML model to predict correct course names from OCR output
    Uses historical corrections to improve accuracy
    """
    
    def __init__(self):
        self.course_database = defaultdict(lambda: {
            'correct_name': None,
            'variations': [],
            'frequency': 0
        })
        self.similarity_threshold = 0.75
    
    def train_from_correction(self, ocr_text, corrected_text, course_code):
        """
        Learn from user correction
        Args:
            ocr_text: What OCR extracted
            corrected_text: What user corrected it to
            course_code: Course code (e.g., "CS301")
        """
        # Store correction in database
        entry = self.course_database[course_code]
        
        if entry['correct_name'] is None:
            entry['correct_name'] = corrected_text
        elif entry['correct_name'] != corrected_text:
            # Handle conflicting corrections (use most frequent)
            entry['variations'].append(corrected_text)
        
        # Track OCR variation
        if ocr_text not in entry['variations']:
            entry['variations'].append(ocr_text)
        
        entry['frequency'] += 1
    
    def predict_course_name(self, ocr_text, course_code):
        """
        Predict correct course name from OCR output
        Returns: (predicted_name, confidence_score)
        """
        # Check if we have exact match in database
        if course_code in self.course_database:
            entry = self.course_database[course_code]
            
            # Direct match with known correct name
            if self._similarity(ocr_text, entry['correct_name']) > 0.95:
                return entry['correct_name'], 1.0
            
            # Check known variations
            best_match = None
            best_score = 0
            
            for variation in entry['variations']:
                score = self._similarity(ocr_text, variation)
                if score > best_score and score > self.similarity_threshold:
                    best_score = score
                    best_match = entry['correct_name']
            
            if best_match:
                return best_match, best_score
        
        # No match found - try fuzzy matching across all courses
        all_courses = []
        for code, entry in self.course_database.items():
            if entry['correct_name']:
                all_courses.append((code, entry['correct_name'], entry['frequency']))
        
        # Sort by frequency (popular courses more likely)
        all_courses.sort(key=lambda x: x[2], reverse=True)
        
        best_match = None
        best_score = 0
        
        for code, name, freq in all_courses[:50]:  # Check top 50 most frequent
            score = self._similarity(ocr_text, name)
            # Boost score slightly for frequent courses
            adjusted_score = score * (1 + (freq / 1000))
            
            if adjusted_score > best_score:
                best_score = score  # Return unadjusted score
                best_match = name
        
        if best_score > self.similarity_threshold:
            return best_match, best_score
        
        # No confident prediction - return OCR text
        return ocr_text, 0.0
    
    def _similarity(self, text1, text2):
        """Calculate similarity between two strings"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def save_model(self, filepath='course_predictor.pkl'):
        """Save trained model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.course_database), f)
    
    def load_model(self, filepath='course_predictor.pkl'):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                self.course_database = defaultdict(
                    lambda: {'correct_name': None, 'variations': [], 'frequency': 0},
                    pickle.load(f)
                )
        except FileNotFoundError:
            print("No existing model found. Starting fresh.")


# ====================
# 3. INTEGRATED PIPELINE
# ====================

class SchedBuddyAI:
    """Complete OCR + ML pipeline"""
    
    def __init__(self):
        self.ocr_processor = CORImageProcessor()
        self.predictor = CourseNamePredictor()
        self.predictor.load_model()  # Load existing knowledge
    
    def process_cor(self, image_path):
        """
        Complete pipeline: OCR extraction + ML prediction
        Returns: list of courses with predictions
        """
        # Step 1: OCR Extraction
        raw_text = self.ocr_processor.extract_text(image_path)
        courses = self.ocr_processor.parse_cor_structure(raw_text)
        
        # Step 2: ML Prediction for each course
        for course in courses:
            if 'title' in course:
                predicted_name, confidence = self.predictor.predict_course_name(
                    course['title'],
                    course.get('code', '')
                )
                
                course['predicted_title'] = predicted_name
                course['confidence'] = confidence
                course['needs_review'] = confidence < 0.85  # Flag low confidence
        
        return courses
    
    def record_user_correction(self, course_code, ocr_title, corrected_title):
        """User corrects a course name - learn from it"""
        self.predictor.train_from_correction(ocr_title, corrected_title, course_code)
        self.predictor.save_model()  # Persist learning


# ====================
# 4. USAGE EXAMPLE
# ====================

if __name__ == "__main__":
    # Initialize system
    ai_system = SchedBuddyAI()
    
    # Process a COR image
    courses = ai_system.process_cor("images/sample-cor.png")
    
    # Display results
    for course in courses:
        print(f"\nCourse Code: {course.get('code', 'N/A')}")
        print(f"Predicted Title: {course.get('predicted_title', 'N/A')}")
        print(f"Confidence: {course.get('confidence', 0):.2%}")
        print(f"Days: {course.get('days', 'N/A')}")
        print(f"Time: {course.get('time_start', '')} - {course.get('time_end', '')}")
        print(f"Room: {course.get('room', 'N/A')}")
        
        if course.get('needs_review'):
            print("⚠️ Low confidence - please review")
    
    # Simulate user correction
    ai_system.record_user_correction(
        course_code="CS301",
        ocr_title="Data Structares",  # OCR mistake
        corrected_title="Data Structures"
    )
    
    print("\n✅ Correction learned! Next time this will be predicted correctly.")