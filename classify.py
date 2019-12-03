from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
from cv2 import cv2
import os

# load the trained convolutional neural network and the label
# binarizer
L=[15,30,45,60,75,90,105,120,135,150]
print("[INFO] loading network...")
model = load_model("model-dist")
lb = pickle.loads(open("labelbin", "rb").read())
filelist = [f for f in os.listdir('test_images')]

for filename in filelist:
	image = cv2.imread('test_images\\'+filename)
	output = image.copy()
	
	# pre-process the image for classification
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# classify the input image
	print("[INFO] classifying image...")
	proba = model.predict(image)[0]
	rounded_prob=[round(p,3) for p in model.predict(image)[0]]
	rounded_prob_new=rounded_prob[1:]
	rounded_prob_new.append(rounded_prob[0])
	p=np.array(rounded_prob_new)
	p[p<0.2]=0
	p=[a/sum(p) for a in p]
	predicted_distance=sum(np.multiply(L,p))
	predicted_distance=str(predicted_distance)

	idx = np.argmax(proba)
	label = lb.classes_[idx]

	correct = "correct" if filename.rfind(label) != -1 else "incorrect"

	# build the label and draw the label on the image
	label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
	output = imutils.resize(output, width=600)
	cv2.putText(output, label, (10, 15),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 1)
	cv2.putText(output, str(rounded_prob_new), (10, 35),  cv2.FONT_HERSHEY_SIMPLEX,
		0.6, (0, 255, 0), 1)
	cv2.putText(output, filename.split('.')[0], (10, 50),  cv2.FONT_HERSHEY_SIMPLEX,
		0.6, (0, 255, 0), 1)
	cv2.putText(output, predicted_distance, (10, 70),  cv2.FONT_HERSHEY_SIMPLEX,
		0.6, (0, 255, 0), 1)
	# show the output image
	print("[INFO] {}".format(label))
	cv2.imshow("Output", output)
	cv2.waitKey(0)