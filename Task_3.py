import cv2
import numpy as np

img = cv2.imread("Oleksiichuk.png")
print(img.shape)

imgResize = cv2.resize(img, (300, 200))
print(imgResize.shape)

imgCropped = img[0:300, 100:500]

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)