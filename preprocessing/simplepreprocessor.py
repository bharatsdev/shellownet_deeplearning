import cv2


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        print("[INFO] : Simple PreProcessor invoked...!")
        self.width = width
        self.height = height
        self.inter = inter

    def prepossess(self, image):
        print("[INFO] : Prepossess Resizing invoked...!")
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
