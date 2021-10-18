import cv2
import os
import random
from scipy import ndarray
from skimage import img_as_ubyte
from skimage import exposure
import numpy as np
import glob

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def vertical_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[ ::-1,:]

def random_rescale(image_array: ndarray):
    return sk.transform.resize(image_array,(200,500))

def brightness(image_array: ndarray , gamma=1,gain=1):
    gamma=random.uniform(1-gamma,1+gamma)
    gamma_1=exposure.adjust_gamma(image_array,gamma,gain)

    return gamma_1

def change_contrast(image_array : ndarray):
    v_min, v_max = np.percentile(image_array, (2.0, 80.0))
    better_contrast = exposure.rescale_intensity(image_array, in_range=(v_min, v_max))
    return better_contrast



# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'resize': random_rescale,
    'vertical flip': vertical_flip,
    'horizontal_flip': horizontal_flip,
    'gamma corrected':brightness,

    'contrast adjust':change_contrast





}

folder_path = '/home/sound/Downloads/dataset'
num_files_desired = 20

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)

        # write image to the disk
        io.imsave(new_file_path, img_as_ubyte(transformed_image))

    num_generated_files += 1

# Performing canny edge detection for the dataset generated from the fucnctions above using the glob library

def Gausblur(image):
    blur = cv2.GaussianBlur(image,(7,7),1)

    return blur

def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

# Read in each image and convert to grayscale
images = [cv2.imread(file,0) for file in glob.glob('/home/sound/Downloads/dataset/*.*')]

# Iterate through each image, perform edge detection, and save image
number = 0
for image in images:
    canny = auto_canny(image)
    path = '/home/sound/Downloads/dataset/canny'
    cv2.imwrite(os.path.join(path, 'canny_{}.png'.format(number)), canny)
    number += 1

def laplace(image):


    return cv2.Laplacian(image,cv2.CV_64F)

images = [cv2.imread(file,0) for file in glob.glob('/home/sound/Downloads/dataset/*.*')]

# Iterate through each image, perform edge detection, and save image
number = 0
for image in images:
    laplacian = laplace(image)
    path2 = '/home/sound/Downloads/dataset/laplace'
    cv2.imwrite(os.path.join(path2, 'laplace_{}.png'.format(number)), laplacian)
    number += 1

def sobel_x_y(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # y
    res=np.hstack([sobelx, sobely])

    return res
images = [cv2.imread(file,0) for file in glob.glob('/home/sound/Downloads/dataset/*.*')]

# Iterate through each image, perform edge detection, and save image
number = 0
for image in images:
    sobel = sobel_x_y(image)
    path00 = '/home/sound/Downloads/dataset/laplace'
    cv2.imwrite(os.path.join(path00, 'sobel_{}.png'.format(number)), sobel)
    number += 1
#performing erode and dialte as it is useful for removing small white noises

def binary_thresh(image):

    retval,thresh=cv2.threshold(image,12,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    return thresh

images = [cv2.imread(file,0) for file in glob.glob('/home/sound/Downloads/dataset/*.*')]

# Iterate through each image, perform edge detection, and save image
number = 0
for image in images:
    binary = binary_thresh(image)
    path3 = '/home/sound/Downloads/dataset/binary'
    cv2.imwrite(os.path.join(path3, 'binary_{}.png'.format(number)), binary)
    number += 1

def erode(image):


    kernel = np.ones((3, 3), np.uint8)
    erode =cv2.erode(image, kernel, iterations=1)

    return erode

images = [cv2.imread(file,0) for file in glob.glob('/home/sound/Downloads/dataset/laplace/*.*')]

number = 0
for image in images:
     erosion = erode(image)
     path4 = '/home/sound/Downloads/dataset/laplace'
     cv2.imwrite(os.path.join(path4,'erode_{}.png'.format(number)), erosion)
     number += 1


def dilate(image):


    kernel = np.ones((5, 5), np.uint8)
    dilate1 =cv2.dilate(image, kernel, iterations=1)

    return dilate1

images = [cv2.imread(file,0) for file in glob.glob('/home/sound/Downloads/dataset/canny/*.*')]

number = 0
for image in images:
     dilated = dilate(image)
     path5 = '/home/sound/Downloads/dataset/dilate'
     cv2.imwrite(os.path.join(path5,'dilate_{}.png'.format(number)), dilated)
     number += 1

def deff(image):
    dilation=dilate(image)
    diff = cv2.absdiff(dilation, image)

    return diff
images = [cv2.imread(file,0) for file in glob.glob('/home/sound/Downloads/dataset/binary/*.*')]
number = 0
for image in images:
     d = deff(image)
     path6 = '/home/sound/Downloads/dataset/binary'
     cv2.imwrite(os.path.join(path6,'deff_{}.png'.format(number)), d)
     number += 1


 ##FIND UNSUAL STUFF


def getcontours(image,imgcontours):



    contours,hierachy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#to remove the unwanted contours
    for cnt in contours:
        area =cv2 .contourArea(cnt)
        #areaMin=cv2.getTrackbarPos("Area","Parameters")
        if area>300:
         cv2.drawContours(imgcontours,contours,-1,(255,255,255),7)
         peri = cv2.arcLength(cnt,True) #getting length of contour
         approx = cv2.approxPolyDP(cnt,0.01*peri,True) # approximating the type of shape
         print(len(approx)) # gets the number of points in the array
         x , y , w, h = cv2.boundingRect(approx)
         cv2.rectangle(imgcontours,(x ,y ),(x + w , y + h ),(0,255,0),5)

         cv2.putText(imgcontours, "Points:"+str(len(approx)),(x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,.5,(0,255,0),2)
         cv2.putText(imgcontours,"Area:" + str(int(area)),(x+w+20,y+45),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)


while True:
    #sample images taken
    image = cv2.imread('augmented_image_14.jpg')

    imgcontours = image.copy()

    imgblur= Gausblur(image)
    imggray=cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)
    threshold=binary_thresh(imggray)
    imgcanny=auto_canny(image,sigma=0.1)
    imdil=dilate(imgcanny)
    getcontours(imdil,imgcontours)
    cv2.imshow("output", np.hstack([image, imgcontours]))
    cv2.waitKey(0)




