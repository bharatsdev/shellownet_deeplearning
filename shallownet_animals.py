from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import simplepreprocessor
from preprocessing import imagetoarraypreprocessor
from databases import simpledatasetloader
from nn.conv import shallownet
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

print("[INFO] : Load all the images.....")
imagePaths = list(glob.iglob("databases/animals/*/*.*"))
print(imagePaths)

print("[INFO] : Initialization for image processor ....!")
sp = simplepreprocessor.SimplePreprocessor(32, 32)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

print("[INFO] : Loading data from desk and scale the raw pixel intensities to the range [0,1] ....!")
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

print("[INFO] : Split dataset....!")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

print("[INFO] : Split dataset....!")
labelBi = LabelBinarizer()
trainY = labelBi.fit_transform(trainY)
testY = labelBi.fit_transform(testY)

print("[INFO] : Initialize the Shallow net..!")
model = shallownet.ShallowNet().build(width=32, height=32, depth=3, classes=3)
opt = SGD(lr=0.05)
model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])
print("[INFO] : training Self Shallow net....!")
History = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=2)

print('[[INFO] : Serializing network')
model.save('shallownet_weights.hdf5')

print("[INFO] : Evaluating Network....")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

plt.style.use('ggplot')
plt.title('ShellowNet')
plt.figure()
plt.plot(np.arange(0, 100), History.history['loss'], labels="Train_losses")
plt.plot(np.arange(0, 100), History.history['val_loss'], labels="Val_loss")
plt.plot(np.arange(0, 100), History.history['acc'], labels="Train_acc")
plt.plot(np.arange(0, 100), History.history['val_acc'], labels="Val_acc")
plt.xlabel('Epochs #')
plt.ylabel('Accuracy & Loss')
plt.legend()
plt.show()
