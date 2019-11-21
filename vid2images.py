import cv2

import os
folders = [r"dataset_my_own\15\\", r"dataset_my_own\30\\", r"dataset_my_own\45\\",
           r"dataset_my_own\60\\", r"dataset_my_own\75\\", r"dataset_my_own\90\\", r"dataset_my_own\120\\"]
for folder in folders:
    filelist = [f for f in os.listdir(folder) if f.endswith(".png")]
    for f in filelist:
        os.remove(os.path.join(folder, f))

remove =True
videos = ['15.mp4', '30.mp4', '45.mp4',
          '60.mp4', '75.mp4', '90.mp4', '120.mp4']
num_samples = 400
if not remove:
    for i, vid in enumerate(videos):
        vidcap = cv2.VideoCapture(vid)
        ret, image = vidcap.read()
        count = 0
        directory = folders[i]
        while ret:
            im_num = count
            if len(str(im_num)) == 1:
                num = '0000000'+str(im_num)
            elif len(str(im_num)) == 2:
                num = '000000'+str(im_num)
            elif len(str(im_num)) == 3:
                num = '00000'+str(im_num)
            cv2.imwrite(directory+num+".png", image)
            ret, image = vidcap.read()
            count += 1
            if count == num_samples:
                break

# import os
# folders=[r'images_raw']
# for folder in folders:
#     filelist = [ f for f in os.listdir(folder) if f.endswith(".jpg") ]
#     for f in filelist:
#         os.remove(os.path.join(folder, f))


# im_num=0
# vidcap = cv2.VideoCapture('vid1.mp4')
# ret,image = vidcap.read()
# while ret:
#     image=cv2.flip(image,0)
#     if len(str(im_num))==1:
#         num='00'+str(im_num)
#     elif len(str(im_num))==2:
#         num='0'+str(im_num)
#     elif len(str(im_num))==3:
#         num=str(im_num)
#     cv2.imwrite(r"images_raw\img_"+str(num)+".jpg", image)
#     ret,image = vidcap.read()
#     im_num += 1
