from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
from cv2 import cv2
import os
import time
# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")

angle_model = load_model("model-angle")
angle_lb = pickle.loads(open("labelbin-angle", "rb").read())
distance_model = load_model("model-dist")
distance_lb = pickle.loads(open("labelbin-dist", "rb").read())
# lengths/classes
classes_dist = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
classes_angle = [-33.75, -22.5, -11.25, 0, 11.25, 22.5, 33.75]

history_dist = []
history_agl = []
nets = ['angle', 'distance']


def get_probs(image):
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	for net in nets:
		if net == 'angle':
			rounded_prob = [round(p, 3) for p in angle_model.predict(image)[0]]
			classes_text = angle_lb.classes_
		else:
			rounded_prob = [round(p, 3)
							for p in distance_model.predict(image)[0]]
			classes_text = distance_lb.classes_
        # puts the classes in the right order from [2,3,1] to [1,2,3]

		if net == 'angle':
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
		p[p < 0.1] = 0
		p = [item/sum(p) for item in p]
		if net == 'angle':
			predicted_agl = sum(np.multiply(classes_angle, p))
			if len(history_agl) == 5:
				history_agl.pop(0)
			history_agl.append(predicted_agl)
			predicted_agl_avrage = sum(history_agl)/5
		else:
			predicted_dist = sum(np.multiply(classes_dist, p))
			# Make the predicted distance depend on current and 4 last predictions
			if len(history_dist) == 5:
				history_dist.pop(0)
			history_dist.append(predicted_dist)
			predicted_dist_avrage = sum(history_dist)/5
	return int(predicted_agl_avrage), int(predicted_dist_avrage)


cap = cv2.VideoCapture(r'test_images_vids\testvid_angle.mp4')
ret, frame = cap.read()
while(cap.isOpened()):
    predicted_agl_avrage, predicted_dist_avrage = get_probs(frame)
    time.sleep(0.1)
    ret, frame = cap.read()
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Dist: '+str(predicted_dist_avrage),
                    (10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Angle: '+str(predicted_agl_avrage),
                    (400, 25), font, 0.7, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
