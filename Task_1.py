import cv2


def get_image():
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10,150)

    while True:
        success, img = cap.read()
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def load_image():
    img = cv2.imread("Oleksiichuk.png")
    cv2.imshow("Output", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # get_image()
    load_image()
