from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
from cv2 import cv2
import os
import time


print("[INFO] loading network...")
# !-- SETUP---!
sufix = 'angle'
#sufix = 'dist'
sufix = 'classify'
lpf_alpha=0.1
history_length=15 # store predicted values for lp filter
# !-- SETUP END---!

model = load_model("model-"+sufix)
lb = pickle.loads(open("labelbin-"+sufix, "rb").read())

# lengths/classes
classes_dist = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
classes_angle = [-33.75, -22.5, -11.25, 0, 11.25, 22.5, 33.75]
classes_ints = classes_angle if sufix == 'angle' else classes_dist
history = []


def lpf(history, alpha):
    y = []
    yk = history[0]
    for k in range(len(history)):
        yk += alpha * (history[k]-yk)
        y.append(yk)
    return y[-1]


def get_prob(image):
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    rounded_prob = [round(p, 3) for p in model.predict(image)[0]]
    if sufix=='classify':
        return rounded_prob
    # puts the classes in the right order from [2,3,1] to [1,2,3]
    classes_text = lb.classes_
    if sufix == 'angle':
        classes = []
        for Cls in classes_text:
            if not Cls == '0':
                Cls = '-' + \
                    Cls.split('m')[1] if 'm' in Cls else Cls.split('p')[1]
            classes.append(float(Cls))
    else:
        classes = [int(Cls) for Cls in classes_text]
    sorted_classes = [Cls for Cls in classes]
    sorted_classes.sort()
    ind = [sorted_classes.index(Cls) for Cls in classes]
    rounded_prob_sorted = []
    for i, item in enumerate(rounded_prob):
        rounded_prob_sorted.insert(ind[i], item)
    p = np.array(rounded_prob_sorted)
    # remove low probabileties and normalize
    small_probs = sorted(p)[0:(len(p)-2)]
    p = [Cls if Cls not in small_probs else 0 for Cls in p]
    p = [item/sum(p) for item in p]

    predicted_distance = sum(np.multiply(classes_ints, p))
    # Make the predicted distance depend on current and 4 last predictions
    if len(history) == history_length:
        history.pop(0)
    history.append(predicted_distance)
    predicted_distance = lpf(history, lpf_alpha )
    return predicted_distance


cap = cv2.VideoCapture(r'test_images_vids\testvid_'+sufix+'.mp4')
ret, frame = cap.read()
while(cap.isOpened()):
    predicted_distance = get_prob(frame)
    time.sleep(0.1)
    ret, frame = cap.read()
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(predicted_distance),
                    (10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
