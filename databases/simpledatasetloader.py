import os

import cv2
import numpy as np


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        for (idx, imagePath) in enumerate(imagePaths):
            img = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for processor in self.preprocessors:
                    img = processor.prepossess(img)
                data.append(img)
                labels.append(label)
            if verbose > 0 and idx > 0 and (idx + 1) % verbose == 0:
                print("[INFO] : Processed {}/{}".format((idx + 1), len(imagePath)))
        return np.array(data), np.array(labels)
