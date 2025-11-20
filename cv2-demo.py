import pytesseract
import cv2

image = cv2.imread("images/sample-cor.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("images/gray.png", gray)

blur = cv2.GaussianBlur(gray, (7,7), 0) # modify 7,7
cv2.imwrite("images/blur.png", blur)

thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("images/thresh.png", thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
cv2.imwrite("images/kernel.png", kernel)

dilate = cv2.dilate(thresh, kernel, iterations=1)
cv2.imwrite("images/dilate.png", dilate)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnsts[0] if len(cnts) == 2 else cnts[1]