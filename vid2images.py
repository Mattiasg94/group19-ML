from cv2 import cv2
import os
# Remove files in folders before inserting new ones
remove_only = False  # only remove, do not add new


def distance_setup():
    folders = [r"dataset_distance\15\\", r"dataset_distance\30\\", r"dataset_distance\45\\",
               r"dataset_distance\60\\", r"dataset_distance\75\\", r"dataset_distance\90\\",
               r"dataset_distance\105\\", r"dataset_distance\120\\", r"dataset_distance\135\\", r"dataset_distance\150\\"]
    for folder in folders:
        filelist = [f for f in os.listdir(folder) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    classes = [[r'vids_distance\15.mp4'],
               [r'vids_distance\30.mp4'],
               [r'vids_distance\45.mp4'],
               [r'vids_distance\60.mp4'],
               [r'vids_distance\75.mp4'],
               [r'vids_distance\90.mp4'],
               [r'vids_distance\105.mp4'],
               [r'vids_distance\120.mp4'],
               [r'vids_distance\135.mp4'],
               [r'vids_distance\150.mp4']]
    return classes, folders


def angle_setup():
    folders = [r"dataset_angle\0\\",
               r"dataset_angle\m11.25\\", r"dataset_angle\m22.5\\", r"dataset_angle\m33.75\\",
               r"dataset_angle\p11.25\\", r"dataset_angle\p22.5\\", r"dataset_angle\p33.75\\",]
    for folder in folders:
        filelist = [f for f in os.listdir(folder) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    classes = [[r'vids_angle\0.mp4'],
               [r'vids_angle\m11.25.mp4'],
               [r'vids_angle\m22.5.mp4'],
               [r'vids_angle\m33.75.mp4'],
               [r'vids_angle\p11.25.mp4'],
               [r'vids_angle\p22.5.mp4'],
               [r'vids_angle\p33.75.mp4']]
    return classes, folders


# adds images to the folders.
classes, folders = distance_setup()
#classes, folders = angle_setup()

num_samples_per_class = 1
num_images_in_each=[]
if not remove_only:
    for i, class_vids in enumerate(classes):
        vid = class_vids[0]
        vidcap = cv2.VideoCapture(vid)
        ret, image = vidcap.read()
        count = 0
        directory = folders[i]
        vid_index = 0
        while ret:
            ret, image = vidcap.read()
            im_num = count
            if len(str(im_num)) == 1:
                num = '0000000'+str(im_num)
            elif len(str(im_num)) == 2:
                num = '000000'+str(im_num)
            elif len(str(im_num)) == 3:
                num = '00000'+str(im_num)
            try:
                image.shape # just to cehck if it exist            
                cv2.imwrite(directory+num+".png", image)
            except:
                print(vid,'only contains num frames:',count)
                break
            count += 1
            if count == num_samples_per_class:
                break
            if count >= (vid_index+1)*(num_samples_per_class/(len(class_vids))):
                vid_index += 1
                vid = class_vids[vid_index]
                vidcap = cv2.VideoCapture(vid)
        num_images_in_each.append(count)

# for num in num_images_in_each:
#     y=0
#     img = cv2.imread(file)
#     h=img.shape[0]
#     rand_int=np.random.uniform(0.6,0.95)
#     w=int(img.shape[1]*rand_int)

#     if i%2==0:    
#         x=img.shape[1]
#         crop_img = img[y:y+h, x-w:x]
#     else:
#         x=0
#         crop_img = img[y:y+h, x:x+w]


