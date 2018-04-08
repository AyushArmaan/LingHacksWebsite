#!/usr/bin/env python3

#http://www.hackevolve.com/recognize-handwritten-digits-2/
import numpy as np
from array import *
import cv2
import imutils
import argparse
from skimage.filters import threshold_adaptive
from keras.models import load_model
from PIL import Image, ImageChops, ImageFilter
from matplotlib import pyplot as plt
import scipy.misc

#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image to recognize")
ap.add_argument("-m","--model",required=True,help="Path to saved classifier")
args = vars(ap.parse_args())


#read,resize and convert to grayscale
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

image = Image.open(args["image"]).convert('L')
image = trim(image)

image = image.resize((28,28))
# image = np.asarray(image)
# image = image.reshape(1,1,28,28)


scipy.misc.imsave('temp.png', image)


img_pred = cv2.imread ( 'temp.png' ,   0 )

# here also we inform the value for the depth = 1, number of rows and columns, which correspond 28x28 of the image.
img_pred = img_pred.reshape ( 1 , 1 , 28 , 28 )
img_pred = img_pred/255.0
model = load_model(args["model"])
pred = model.predict_classes ( img_pred )
pred_proba = model.predict_proba ( img_pred )
pred_proba = "% .2f %%" % (pred_proba [0] [pred] * 100.0)
print ( pred [ 0 ] , "with probability of" , pred_proba )


# image = np.asarray(image)
#
# plt.imshow(image)
# plt.show()


"""

width = float(image.size[0])
height = float(image.size[1])
new_Image = Image.new('L', (28,28), (255))

#resizing the image to be a square
dimmension = int(round((float(20.0/width*height)), 0))
if(dimmension == 0): #rare case
    dimmension = 1

if(width > height):
    image = image.resize((20, dimmension), Image.LANCZOS).filter(ImageFilter.SHARPEN)
    new_Image.paste(image, (4, int(round(((28 - dimmension / 2), 0)))))
else:
    image = image.resize((dimmension, 20), Image.LANCZOS).filter(ImageFilter.SHARPEN)
    new_Image.paste(image, (int(round((28 - dimmension / 2), 0)), 4))
"""



"""
image = image.convert('L')
pix_val = list(image.getdata())
pix_val = [pix_val]

#turning it into a 4d array 28*28
new_arr = [[0 for d in range(28)] for y in range(28)]
k=0
for i in range(28):
    for j in range(28):
        new_arr[i][j]=pix_val[0][k]
        k=k+1
"""

#image = np.array(pix_val)
#image = np.asarray(new_arr)

# image = image.reshape(1,1,28,28)
# model = load_model(args["model"])
# label = model.predict(image)
#
# print(str(label))


"""

image = np.array(image)
#image = imutils.resize(image,width=320)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Rectangular kernel with size 5x5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#apply blackhat and otsu thresholding
blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
_,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thresh = cv2.dilate(thresh,None)        #dilate thresholded image for better segmentation

#find external contours
(_, cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #instead of image: thresh
avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])      #contourArea for digit approximation

digits = []
boxes = []


for i,c in enumerate(cnts):
    if cv2.contourArea(c)<avgCntArea/10:
        continue
    mask = np.zeros(gray.shape,dtype="uint8")   #empty mask for each iteration

    (x,y,w,h) = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    cv2.drawContours(mask,[hull],-1,255,-1)     #draw hull on mask
    mask = cv2.bitwise_and(thresh,thresh,mask=mask) #segment digit from thresh

    digit = mask[y-8:y+h+8,x-8:x+w+8]       #just for better approximation
    digit = cv2.resize(digit,(28,28))
    boxes.append((x,y,w,h))
    digits.append(digit)

digits = np.array(digits)
model = load_model(args["model"])
#digits = digits.reshape(-1,784)    #for Multi-Layer-Perceptron
digits = digits.reshape(digits.shape[0],1,28,28)    #for Convolution Neural Networks
labels = model.predict_classes(digits)

#cv2.imshow("Original",image)
#cv2.imshow("Thresh",thresh)

#draw bounding boxes and print digits on them
for (x,y,w,h),label in sorted(zip(boxes,labels)):
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.putText(image,str(label),(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.imshow("Recognized",image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
"""
