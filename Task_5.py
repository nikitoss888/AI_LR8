import cv2
import numpy as np
import matplotlib.pyplot as plt


def main(full_img="messi_full.jpg", face_img="messi_face.jpg"):
    img = cv2.imread(full_img, 0)
    img2 = img.copy()
    template = cv2.imread(face_img, 0)

    w, h = template.shape[::-1]
    methods = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR",
                  "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap="gray")
        plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap="gray")
        plt.title("Detected Point"), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()


if __name__ == "__main__":
    main('Oleksiichuk.png', 'Oleksiichuk_face.png')