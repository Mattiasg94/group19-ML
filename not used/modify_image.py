from cv2 import cv2
import numpy as np
from PIL import Image

'''
Add this code to vid2images
below is for if a background is to be croped into the image. Has some bugs. Mind the size of the background.
background = cv2.imread(r"walls\test_wall.jpg") 
background=cv2.resize(background,(image.shape[1],image.shape[0]))
image = modify_main(background, image)'''
def pix_is_head(size,i,j):
    
    if (size[0]/5)*2<i and i<(size[0]/5)*3:
        if size[1]/3<j and j<(size[1]/3)*2:
            return True
    return False
def white_background(img):
    # You may need to convert the color.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    pixels = im_pil.load()
    size=im_pil.size
    for i in range(im_pil.size[0]): 
        for j in range(im_pil.size[1]):
            if not pix_is_head(size,i,j): # all other but head 
                if pixels[i,j] >= (90,0,0): 
                    pixels[i,j] = (255, 255, 255) 
            else:
                if pixels[i,j] >= (120,0,0):
                    pixels[i,j] = (255, 255, 255)
    open_cv_image = np.array(im_pil) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

def nothing(x):
    pass

def crop_main(frame):
    # w, h, c = frame.shape
    # resize_coeff = 1
    # frame = cv2.resize(frame, (int(resize_coeff*h), int(resize_coeff*w)))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_blur = cv2.medianBlur(grey, 11)
    edges = cv2.Canny(grey_blur, 10, 10*2)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, 1)
    edges = cv2.erode(edges, kernel, 1)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_index, max_area = max(enumerate([cv2.contourArea(x) for x in contours]), key = lambda x: x[1])
    max_contour = contours[max_index]
    (x,y,w,h) = cv2.boundingRect(max_contour)
    crop_img = frame[y:y+h, x:x+w]
    return crop_img

#crop_main(cv2.imread(r'C:\Users\User\Downloads\cnn-keras\cnn-keras\croped_img.png'))
# cv2.namedWindow('CannyT')
# cv2.imshow('original',frame)
# cv2.imshow('CannyT',edges)
# cv2.imshow('cropped',crop_img)

# key = cv2.waitKey()
# cv2.destroyAllWindows()


def make_png(image):
    b_channel, g_channel, r_channel = cv2.split(image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)*255
    image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    trans_mask = image[:,:,1] == 255
    image[trans_mask] = [255, 255, 255, 0]
    return image

def overlay_transparent(background_img, img_to_overlay_t, x, y):
    #img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), (200,200))

    bg_img = background_img.copy()
    if background_img.shape[0]<img_to_overlay_t.shape[0]-300 or background_img.shape[1]<img_to_overlay_t.shape[1]-300:
        print("[ERROR]-Image dimentions are wrong")
        print("background",background_img.shape)
        print("overlay",img_to_overlay_t.shape)
    # Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))
    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a,5)
    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]
    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)
    # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img

def modify_main(background,img):
    img_only_station=white_background(img)
    img_cropt=crop_main(img_only_station)
    img=make_png(img_cropt)
    x=int(background.shape[0]/2 -img.shape[0]/2)
    y=background.shape[0]-img.shape[0]
    img_with_background=overlay_transparent(background, img, x, y)
    return img_with_background

# img=cv2.imread(r'croped_img.png')
# background = cv2.imread(r"walls\housewall_grey.jpg")
# img=main(background,img)
# cv2.imshow('image',img)
# cv2.waitKey()