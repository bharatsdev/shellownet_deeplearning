from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv import shallownet
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] : Load datasets and rescal it [0,1]....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype(float) / 255.
testX = testX.astype(float) / 255.
print(trainY)

print("[INFO] : Convert the labels from integers to vectors ")
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Classes Array
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

sgd = SGD(lr=0.005)
model = shallownet.ShallowNet.build(width=32,
                                    height=32, depth=3, classes=len(labelNames))
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=40, verbose=1)


print('[[INFO] : Serializing network')
model.save('shallownet_weights.hdf5')

print("[INFO] : network Evaluation... ")
pred = model.predict(testX)
print(classification_report(testY.argmax(axis=1), pred.argmax(axis=1), target_names=labelNames))

print("[INFO] : Draw Plot... for loss and Accuracy ")
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label="Train_loss")
plt.plot(np.arange(0, 100), H.history['val_loss'], label="Val_loss")
plt.plot(np.arange(0, 100), H.history['acc'], label="Train_Acc")
plt.plot(np.arange(0, 100), H.history['val_acc'], label="Val_acc")
plt.ylabel('Loss/Accuracy')
plt.xlabel('Epochs #')
plt.xlabel('Training Accuracy and Loss')
plt.show()
plt.savefig('cifar10.png')
