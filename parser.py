"""
Bicol University COR Parser - Fixed for Actual Tesseract Output
Works with the actual OCR output format from BU CORs
"""

import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Course:
    """Data class to hold course information"""
    code: str
    subject: str
    unit: str
    class_section: str
    days: str
    time: str
    room: str
    faculty: str
    credit: str = ""
    lec: str = ""
    lab: str = ""
    raw_line: str = ""

class BUCORParser:
    """Parser specifically designed for Bicol University COR format"""
    
    def __init__(self):
        # Tesseract configuration
        self.config = r'--oem 3 --psm 6'
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess the COR image for better OCR accuracy
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        """
        # Deskew if needed
        coords = np.column_stack(np.where(enhanced > 0))
        if coords.size > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:
                (h, w) = enhanced.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                enhanced = cv2.warpAffine(
                    enhanced, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        """
        return Image.fromarray(enhanced)
    
    def extract_raw_text(self, image_path: str) -> str:
        """
        Extract raw text from image using Tesseract
        """
        processed_img = self.preprocess_image(image_path)
        raw_text = pytesseract.image_to_string(processed_img, config=self.config)

        # Save raw OCR output for debugging
        try:
            with open('ocr_raw.txt', 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print("🔎 Wrote OCR raw text to: ocr_raw.txt")
        except Exception as e:
            print(f"⚠️ Could not write ocr_raw.txt: {e}")

        # Save processed image so we can inspect preprocessing
        try:
            processed_img.save('debug_processed.png')
            print("🖼️  Wrote processed image to: debug_processed.png")
        except Exception as e:
            print(f"⚠️ Could not save debug_processed.png: {e}")

        return raw_text
    
    def find_schedule_section(self, text: str) -> List[str]:
        """
        Find and extract lines that contain course information
        Returns list of lines between the header and "Totals::"
        """
        lines = text.split('\n')
        
        schedule_lines = []
        in_schedule = False
        
        keywords = ['code', 'subject', 'unit', 'class', 'days', 'time', 'room', 'faculty']

        for line in lines:
            line_lower = line.lower()
            # Start capturing after we see a header line with at least two schedule keywords
            kw_matches = sum(1 for kw in keywords if kw in line_lower)
            if kw_matches >= 2:
                in_schedule = True
                continue
            
            # Stop when we hit Totals
            if 'Totals::' in line or 'Total Units' in line:
                break
            
            # Capture schedule lines
            if in_schedule and line.strip():
                schedule_lines.append(line)
        
        # If we didn't find the explicit header-based schedule, try a fallback:
        if not schedule_lines:
            # Look for lines that look like course entries (code + units + class/day/time patterns)
            fallback_lines = []
            course_start_re = re.compile(r'^[A-Z]{2,4}\s+(?:\d{1,4}|Elective)')
            unit_re = re.compile(r'\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+')

            for line in lines:
                if course_start_re.search(line) and unit_re.search(line):
                    fallback_lines.append(line)

            if fallback_lines:
                print(f"🔁 Used fallback detection: found {len(fallback_lines)} course-like lines")
                return fallback_lines

        return schedule_lines

        # (Unreachable) keep for clarity
    
    def parse_course_line(self, line: str, next_lines: List[str] = None) -> Optional[Course]:
        """
        Parse a course line from the OCR output
        
        Format example:
        CS 114 Operating Systems 3.0 2.0 1.0 BSCS 3-A Sat 08:00 AM-10:00 AM CS-02-102 CANON, M
        
        Some courses span multiple lines (multi-day courses)
        """
        line = line.strip()
        
        # Skip empty lines and lines that are clearly not courses
        if not line or len(line) < 20:
            return None
        
        # Skip lines that don't start with a course code
        if not re.match(r'^[A-Z]{2,4}\s+\d{1,4}|^[A-Z]{2,4}\s+Elective', line):
            return None
        
        try:
            # Extract course code (e.g., "CS 114" or "CS Elective")
            code_match = re.match(r'^([A-Z]{2,4}\s+(?:\d{1,4}|Elective))', line)
            if not code_match:
                return None
            
            code = code_match.group(1).strip()
            remaining = line[len(code):].strip()
            
            # Extract units pattern (e.g., "3.0 2.0 1.0" or "3.0 3.0 0.0")
            unit_pattern = r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
            unit_match = re.search(unit_pattern, remaining)
            
            if not unit_match:
                return None
            
            credit = unit_match.group(1)
            lec = unit_match.group(2)
            lab = unit_match.group(3)
            
            # Subject is between code and units
            subject_end_pos = remaining.find(unit_match.group(0))
            subject = remaining[:subject_end_pos].strip()
            
            # Everything after units
            after_units = remaining[subject_end_pos + len(unit_match.group(0)):].strip()
            
            # Extract class section (e.g., "BSCS 3-A")
            class_pattern = r'(BSCS\s+\d+-[A-Z])'
            class_match = re.search(class_pattern, after_units)
            class_section = class_match.group(1) if class_match else ""
            
            if class_match:
                after_class = after_units[class_match.end():].strip()
            else:
                after_class = after_units
            
            # Extract days (Sat, Tu, TuTh, MW, F, etc.)
            days_pattern = r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun|M|T|W|Th|F|Sa|TuTh|MW|MWF|TTh|TF|MTh)\b'
            days_match = re.search(days_pattern, after_class)
            days = days_match.group(1) if days_match else ""
            
            if days_match:
                after_days = after_class[days_match.end():].strip()
            else:
                after_days = after_class
            
            # Extract time (e.g., "08:00 AM-10:00 AM" or "05:30 PM-08:30PM")
            time_pattern = r'(\d{1,2}:\d{2}\s*[AP]M\s*-\s*\d{1,2}:\d{2}\s*[AP]M)'
            time_match = re.search(time_pattern, after_days)
            time = time_match.group(1) if time_match else ""
            
            if time_match:
                after_time = after_days[time_match.end():].strip()
            else:
                after_time = after_days
            
            # Extract room (e.g., "CS-02-102", "CAL-01-401")
            room_pattern = r'([A-Z]{2,4}\s*-\s*\d{2}\s*-\s*\d{3})'
            room_match = re.search(room_pattern, after_time)
            room = room_match.group(1).replace(' ', '') if room_match else ""
            
            if room_match:
                after_room = after_time[room_match.end():].strip()
            else:
                after_room = after_time
            
            # Faculty is what remains (e.g., "CANON, M" or "Sotto, G")
            faculty_pattern = r'([A-Z][a-z]+,\s*[A-Z]\.?)'
            faculty_match = re.search(faculty_pattern, after_room)
            faculty = faculty_match.group(1) if faculty_match else after_room.strip()
            
            # Handle multi-line courses (check next_lines for additional times/rooms)
            if next_lines:
                for next_line in next_lines[:3]:  # Check up to 3 next lines
                    # If next line has time but no course code, it's a continuation
                    if re.search(time_pattern, next_line) and not re.match(r'^[A-Z]{2,4}\s+\d{1,4}', next_line):
                        extra_time = re.search(time_pattern, next_line)
                        extra_room = re.search(room_pattern, next_line)
                        if extra_time:
                            time += f" / {extra_time.group(1)}"
                        if extra_room:
                            room += f" / {extra_room.group(1).replace(' ', '')}"
            
            return Course(
                code=code,
                subject=subject,
                unit=f"{credit} {lec} {lab}",
                credit=credit,
                lec=lec,
                lab=lab,
                class_section=class_section,
                days=days,
                time=time,
                room=room,
                faculty=faculty,
                raw_line=line
            )
            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse line: {line[:60]}...")
            print(f"   Error: {e}")
            return None
    
    def extract_courses(self, image_path: str) -> List[Course]:
        """
        Main method to extract all courses from a COR image
        """
        print(f"📄 Processing: {image_path}")
        
        # Extract raw text
        print("🔍 Performing OCR...")
        raw_text = self.extract_raw_text(image_path)
        
        # Find schedule section
        print("📋 Extracting schedule lines...")
        schedule_lines = self.find_schedule_section(raw_text)
        
        if not schedule_lines:
            print("\n⚠️  Could not find schedule section in OCR output")
            print("Raw text preview:")
            print(raw_text[:500])
            return []
        
        print(f"   Found {len(schedule_lines)} potential course lines")
        
        # Parse courses
        print("📚 Parsing courses...")
        courses = []
        i = 0
        while i < len(schedule_lines):
            line = schedule_lines[i]
            
            # Get next few lines for multi-line course detection
            next_lines = schedule_lines[i+1:i+4] if i+1 < len(schedule_lines) else []
            
            course = self.parse_course_line(line, next_lines)
            if course:
                courses.append(course)
                print(f"   ✓ Parsed: {course.code} - {course.subject[:30]}...")
            
            i += 1
        
        return courses

def main():
    """Main execution function"""
    import sys
    import os
    
    print("\n" + "="*80)
    print("🎓 BICOL UNIVERSITY COR PARSER")
    print("="*80 + "\n")
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "images/sample-cor.png"
        print(f"ℹ️  No image provided. Using default: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        print("Usage: python bu_cor_parser.py <path_to_cor_image>")
        sys.exit(1)
    
    # Create parser
    parser = BUCORParser()
    
    try:
        courses = parser.extract_courses(image_path)
        
        if not courses:
            print("\n❌ No courses were extracted!")
            print("\nTroubleshooting tips:")
            print("1. Ensure the image is clear and readable")
            print("2. Check that Tesseract is installed: tesseract --version")
            print("3. Try running manually: tesseract your_image.png stdout")
            sys.exit(1)
        
        print(f"\n✅ Successfully extracted {len(courses)} course(s)!\n")
        print("="*80 + "\n")
        
        # Display results in a formatted table
        for i, course in enumerate(courses, 1):
            print(f"📚 COURSE {i}")
            print("-"*80)
            print(f"Code:         {course.code}")
            print(f"Subject:      {course.subject}")
            print(f"Units:        {course.unit} (Credit: {course.credit}, Lec: {course.lec}, Lab: {course.lab})")
            print(f"Class:        {course.class_section}")
            print(f"Days:         {course.days}")
            print(f"Time:         {course.time}")
            print(f"Room:         {course.room}")
            print(f"Faculty:      {course.faculty}")
            print()
        
        print("="*80)
        
        # Export to JSON
        import json
        output = []
        for course in courses:
            output.append({
                'code': course.code,
                'subject': course.subject,
                'unit': {
                    'credit': course.credit,
                    'lecture': course.lec,
                    'lab': course.lab
                },
                'class': course.class_section,
                'days': course.days,
                'time': course.time,
                'room': course.room,
                'faculty': course.faculty
            })
        
        output_file = 'extracted_schedule.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Schedule exported to: {output_file}")
        print(f"📊 Total courses: {len(courses)}")
        print(f"📖 Total units: {sum(float(c.credit) for c in courses)}")
        
    except Exception as e:
        print(f"\n❌ Error processing COR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()