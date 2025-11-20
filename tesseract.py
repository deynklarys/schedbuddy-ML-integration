import pytesseract
from PIL import Image

image_file = "images/sample-cor.png"
img = Image.open(image_file)

ocr_result = pytesseract.image_to_string(img)

print(ocr_result)