from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        print("[INFO] : Store the image dataformat ...!")
        self.dataFormat = dataFormat

    def prepossess(self, image):
        # Apply the Keras utility function that correctly rearrange the image dimensions of image
        print("[INFO] : ImageToArrayPreprocessor invoked... DataFormat:{}".format(self.dataFormat))
        return img_to_array(image, data_format=self.dataFormat)
