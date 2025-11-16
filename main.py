import cv2
from PIL import Image
import pytesseract

im_file = "data/cor.png"

im = Image.open(im_file)

# print(im)
# print(im.size)
# im.show()
# im.rotate(90).show()

im.save("data/save-image.png")

