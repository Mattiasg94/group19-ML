from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import os
import time
from cv2 import cv2
import matplotlib.pyplot as plt
from controller import P_controller
# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")

# !-- SETUP---!
lpf_alpha = 0.05
history_length = 50  # store predicted values for lp filter
prob_is_station_threshold = 0.8
vid_file=r'test_images_vids\testvid_angle.mp4'
# !-- SETUP END---!

angle_model = load_model("model-angle")
angle_lb = pickle.loads(open("labelbin-angle", "rb").read())
distance_model = load_model("model-dist")
distance_lb = pickle.loads(open("labelbin-dist", "rb").read())
classify_model = load_model("model-classify")
classify_lb = pickle.loads(open("labelbin-classify", "rb").read())

classes_dist = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
classes_angle = [-45,-33.75, -22.5, -11.25, 0, 11.25, 22.5, 33.75,45]

nets = ['angle', 'distance']
history_dist=[]
history_agl=[]
X=[]
Y=[]
angle_lst=[]
angle_raw_lst=[]
dist_lst=[]
dist_raw_lst=[]
prob_station_lst=[]
states_lst=np.array([[0],[0]])
dstates_lst=np.array([[0],[0]])
def lpf(history, alpha):
    y = []
    yk = history[0]
    for k in range(len(history)):
        yk += alpha * (history[k]-yk)
        y.append(yk)
    return y[-1]


def handle_outliners(classes_ints, p, history_type,Type):
    predicted_type = sum(np.multiply(classes_ints, p))
    if Type=='angle':
        angle_raw_lst.append(predicted_type)
    else:
        dist_raw_lst.append(predicted_type)
    if len(history_type) == history_length:
        history_type.pop(0)
    history_type.append(predicted_type)
    predicted_type = lpf(history_type, lpf_alpha)
    return predicted_type


def get_probs(image):
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prob_is_station = classify_model.predict(image)[0][0]
    if prob_is_station < prob_is_station_threshold:
        return None, None, round(prob_is_station, 2)
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
        # select only 2 biggest probs
        small_probs = sorted(p)[0:(len(p)-2)]
        p = [Cls if Cls not in small_probs else 0 for Cls in p]
        p = [item/sum(p) for item in p]
        if net == 'angle':
            predicted_agl = handle_outliners(classes_angle, p, history_agl,'angle')
        else:
            predicted_dist = handle_outliners(classes_dist, p, history_dist,'distance')

    return int(round(predicted_agl)), int(round(predicted_dist)), round(prob_is_station, 2)


def get_xy(predicted_agl, predicted_dist):
    x = np.sin(np.deg2rad(predicted_agl))*predicted_dist
    y =np.cos(np.deg2rad(predicted_agl))*predicted_dist
    return round(x), round(y)

T=[]
p=0
cap = cv2.VideoCapture(vid_file)
ret, frame = cap.read()
while(cap.isOpened()):
    #time.sleep(0.1)
    predicted_agl, predicted_dist, prob_is_station = get_probs(
        frame)
    if (predicted_agl or predicted_dist) == None:
        predicted_agl = predicted_dist =x = y='None'
    else:
        x, y = get_xy(predicted_agl, predicted_dist)
        X.append(x)
        Y.append(y)
        angle_lst.append(predicted_agl)
        dist_lst.append(predicted_dist)
        prob_station_lst.append(prob_is_station)
        states,dstates=P_controller(predicted_dist,predicted_agl,prob_is_station,states_lst)
        states_lst=np.append(states_lst,states,axis=1)
        dstates_lst=np.append(dstates_lst,dstates,axis=1)
        
        T.append(p)
        p=p+1
    ret, frame = cap.read()
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Dist: '+str(predicted_dist),
                    (10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Angle: '+str(predicted_agl),
                    (150, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'X: '+str(x)+' Y: '+str(y),
                    (300, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Prob is station: '+str(prob_is_station),
                    (550, 25), font, 0.7, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break     
    else:
        break
cap.release()
cv2.destroyAllWindows()
# plot x y
X=np.array(X)*-1
Y=np.array(Y)*-1
plt.style.use("ggplot")
plt.figure()
plt.plot(X,Y,label='path')
plt.title("Path")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left")
plt.savefig(r'plots\classify_plots\x_y.png')


# plot raw angle vs lpf angle
plt.style.use("ggplot")
plt.figure()
plt.plot(angle_raw_lst,label='raw')
plt.plot(angle_lst,label='filtered')
plt.plot(prob_station_lst,label='Prob station')
plt.title("Angles")
plt.xlabel("xx")
plt.ylabel("xx")
plt.legend(loc="upper left")
plt.savefig(r'plots\classify_plots\angles.png')

# plot raw distance vs lpf distance
plt.style.use("ggplot")
plt.figure()
plt.plot(dist_raw_lst,label='raw')
plt.plot(dist_lst,label='filtered')
plt.plot(prob_station_lst,label='Prob station')
plt.title("Distances")
plt.xlabel("xx")
plt.ylabel("xx")
plt.legend(loc="upper left")
plt.savefig(r'plots\classify_plots\distances.png')


# plot controller states
plt.style.use("ggplot")
plt.figure()
plt.plot(states_lst[0,:],label='d')
plt.plot(states_lst[1,:],label='theta')
plt.title("Controller states")
plt.xlabel("time t="+str(len(T)))
plt.ylabel("distance/angle")
plt.legend(loc="upper left")
plt.savefig(r'plots\classify_plots\states.png')


# plot controller states
plt.style.use("ggplot")
plt.figure()
plt.plot(dstates_lst[0,:],label='v_d')
plt.plot(dstates_lst[1,:],label='v_theta')
plt.title("Controller velecites")
plt.xlabel("time t="+str(len(T)))
plt.ylabel("v_distance/v_angle")
plt.legend(loc="upper left")
plt.savefig(r'plots\classify_plots\dstates.png')
