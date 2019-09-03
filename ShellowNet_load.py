import cv2

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from databases.simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
import glob
import numpy as np

print('[INFO] : load the model.....')
model = load_model('shallownet_weights.hdf5')

# initialize the class labels
classLabels = ["cat", "dog", "panda"]

print("[INFO] : Load all the images.....")
imagePaths = np.array(list(glob.iglob("databases/animals/*/*.*")))
# print(imagePaths)
print("[INFO] : Select image randomly.... ")
idxs = np.random.randint(0, len(imagePaths), size=(10,))
print(idxs)
imgPaths = imagePaths[idxs]

print('[INFO] : Process images.. !')
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
print('[INFO] : Load data from  disk and change pixel intensities range [0,1]')
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imgPaths, verbose=500)
data = data.astype(float) / 255.0

print('[INFO] : Predicting....!')
pred = model.predict(data, batch_size=32)
print(pred)
y_pred = pred.argmax(axis=1)
print(y_pred)
print('[INFO] : Display predict images...!')
for (i, imgPath) in enumerate(imgPaths):
    image = cv2.imread(imgPath)
    cv2.putText(image, 'Label {}'.format(classLabels[y_pred[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                2)
    cv2.imshow("Image :: ", image)
    cv2.waitKey(0)
