from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time
# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model("model")
lb = pickle.loads(open("labelbin", "rb").read())
L = [15, 30, 45, 60, 75, 90, 120]
history=[]

def get_prob(image):
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    rounded_prob = [round(p, 3) for p in model.predict(image)[0]]
    rounded_prob_new = rounded_prob[1:]
    rounded_prob_new.append(rounded_prob[0])
    p = np.array(rounded_prob_new)
    p[p < 0.1] = 0
    p = [a/sum(p) for a in p]
    predicted_distance = sum(np.multiply(L, p))
    predicted_distance = str(predicted_distance)

    return predicted_distance


cap = cv2.VideoCapture('testvid.mp4')
ret, frame = cap.read()
while(cap.isOpened()):
    predicted_distance = get_prob(frame)
    time.sleep(0.1)
    ret, frame = cap.read()
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predicted_distance,
                    (10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
