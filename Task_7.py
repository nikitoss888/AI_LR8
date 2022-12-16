import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

img = cv2.imread('coins_2.jpg')
filtered = cv2.pyrMeanShiftFiltering(img, 20, 40)

gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
_, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_filtered = []
for con in contours:
    area = cv2.contourArea(con)
    if area < 1000:
        contours_filtered.append(con)

cv2.drawContours(thresh_img, contours_filtered, -1, 255, -1)
dist = ndi.distance_transform_edt(thresh_img)
dist_copy = dist.copy()

local_max = peak_local_max(dist, indices=False, min_distance=20, labels=thresh_img)
markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]

watershed_img = watershed(-dist, markers, mask=thresh_img)
titles = ['Original image', 'Binary Image', 'Watershed']
images = [img, thresh_img, watershed_img]

fig = plt.gcf()
fig.set_size_inches(8, 6)

for i, img in enumerate(images):
    plt.subplot(3, 1, i + 1)
    plt.imshow(img, "jet" if i == 2 else "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
