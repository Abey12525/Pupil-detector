import cv2
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np


class PupilD:
    def __init__(self, Img_path):
        self.Img_path = Img_path
        self.files = [f for f in listdir(self.Img_path) if isfile(join(self.Img_path, f))]

    def subplot(self):
        column = 5
        rows = 6
        fig = plt.figure(figsize=(100, 100))
        for i, f in enumerate(self.files):
            img = cv2.imread('{}{}'.format(self.Img_path, f))
            h, w, c = img.shape
            fig.add_subplot(rows, column, i + 1)
            plt.imshow(img[..., ::-1])
        plt.show()

    def detect(self, sadj=1.68, vadj=0.5, hadj=2.8, sharpE=10.29647, BlurK=41,
               threshold_low=28, threshold_high=255, area_low=55, area_high=24000):
        """Sharpening the image for better clarity"""
        kernel = np.array([[-1, -1, -1],
                           [-1, sharpE, -1],
                           [-1, -1, -1]])
        for f in self.files:
            img = cv2.imread('{}{}'.format(self.Img_path, f))
            img_cpy = img
            """sharpening"""
            img = cv2.filter2D(img, -1, kernel)
            """Bluring image to reduced noise"""
            img = cv2.GaussianBlur(img, (BlurK, BlurK), 1)
            """converting image into HSV so that we can adjust the hue , saturation , value/brightness"""
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
            (h, s, v) = cv2.split(img)
            s = s * sadj
            v = v * vadj
            h = h * hadj
            s = np.clip(s, 0, 255)
            v = np.clip(v, 0, 255)
            h = np.clip(h, 0, 255)
            "combining the modified HSV value into a single image"
            img = cv2.merge([h, s, v])
            """converting back to RGB"""
            img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)
            gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """converting img into gray scale and applying threshold to get the eye"""
            _, thresh = cv2.threshold(gry, threshold_low, threshold_high, cv2.THRESH_TOZERO)
            """ finding the pupil using contour"""
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if i == 3:
                        break
                if(area_low <= area <= area_high):
                    (cx,cy),radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(img_cpy,(int(cx), int(cy)), int(radius), (0,0,255), 8)
            cv2.imshow('g{}'.format(f), img_cpy)
            #cv2.imshow('{}'.format(f), thresh)
            k = cv2.waitKey()
            if k == 27:
                cv2.destroyAllWindows()
            if k != 27:
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    pclass = PupilD('./Images/')
    pclass.subplot()
    pclass.detect()
