# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background
import os
from cv2 import cv2
import pickle
import random
import argparse
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from pyimagesearch.angle_net import Angle_Net
from pyimagesearch.distance_net import Distance_Net 
from pyimagesearch.classify_net import Classify_Net
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")

EPOCHS = 15
INIT_LR = 1e-4     
BS = 250
BS_aug = BS
IMAGE_DIMS = (96, 96, 3)
data = []
labels = []
dataset_folder = "dataset_angle"
#dataset_folder='dataset_distance'
#dataset_folder='dataset_classify' 

if 'angle' in dataset_folder:
    sufix = 'angle'
elif 'distance' in dataset_folder:
    sufix='dist'
elif 'classify' in dataset_folder:
    sufix='classify'
print('[info]',sufix)
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(dataset_folder)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    try:
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    except:
        print('probably something is wrong with', imagePath)
    image = img_to_array(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.GaussianBlur(image,(13,13),0)
    data.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)
horizontal_flip = False if sufix == 'angle' else True
print('[info]:',horizontal_flip)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=2, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         fill_mode="nearest",horizontal_flip=horizontal_flip,)
                         #brightness_range=[0.6, 1.4])

# initialize the model
print("[INFO] compiling model...")
if sufix == 'angle':
    model = Angle_Net.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(lb.classes_))
elif sufix == 'classify':
    model = Classify_Net.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(lb.classes_))                   
else:
    model = Distance_Net.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                               depth=IMAGE_DIMS[2], classes=len(lb.classes_))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
if sufix=='classify':
    loss='binary_crossentropy'
else:
    loss='categorical_crossentropy'
model.compile(loss=loss, optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS_aug),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('model-'+sufix)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy %")
plt.legend(loc="upper left")
plt.savefig(r'plots\plot-acc-'+sufix+'.png')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.savefig(r'plots\plot-loss-'+sufix+'.png')


# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open('labelbin-'+sufix, "wb")
f.write(pickle.dumps(lb))
f.close()
