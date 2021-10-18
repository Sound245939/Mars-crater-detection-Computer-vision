# AICV

The aim of this project is to be able to detect unsual objects on the surafce of Mars, as this is an ongoing research for various applications, such as to discover existence of past life. This project uses edge detection and feature detection to identify potentailly unsual objects that are just detected by the algorithm using various image analysis techniques, such as edge detection and feature detection.

By performing shape detection, the algorithm identifies regions of interest that need further 
investigation. It is able to detect regions such as craters, rockets, and sand dunes. After making 
observations of these regions of interest the user, who is the scientist will be able to extract this 
information to be able to study the images in much more detail and be able to determine what is 
unusual about this feature, if incase it is a mars microbe or not. 
This project is presented with the conceptual design of the working of the algorithm. 
With the use of edge detection and performing many computer vision manipulations on the 
images, the contours in the image are detected and the areas of interest within the image are 
labelled for further investigation by scientists.

For this particular software design the, the sequence of the compliments is what has been put to use. Since the 
overall idea is to be able to use the cv2.findCounturs function in OpenCv. The general working 
of findCountours makes use of the following methods in the following sequence:
#Blurring the image
#Converting image to greyscale
#Thresholding the image
#Performing edge detection
#Dilation of the image
#Getting contours of image

The use of data augmentation was put into this project. In order to begin with the project, the requirement of generating a large data set was a good 
idea, so that there would be more images for analysis without the need to search for over 40 
images on the internet.

Edge detection 
This is yet another image processing technique used to find the boundaries or edges of the 
image, by determining where the brightness of the image changes dramatically. As mentioned 
previously, the reason edge detection was adopted for this project was to be able to extract the 
structure of objects in an image in order to be able to indicate to the user viewing the results of 
the detected image to make further observations based of the the extracted areas of interest.
Since we were interested to find the number,size and relative location of objects/shapes in the 
image, edge detection allows us to focus on the parts of the image which are most useful, 
ignoring parts of the image that will not help us.

Further details of the implementation of the algorithm can be found in the pdf file.
