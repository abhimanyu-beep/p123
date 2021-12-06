import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
if(not os.environ.get("pythonhttphverify")and getattr(ssl,"_create_unverified_context",none)):
    ssl._create_default_https_context=ssl._create_unverifiedcontext


X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)
X_train,X_test,y_test,y_train=train_test_split(X,y,random_state=9)
X_train_scaled=X_train/255
X_test_scales=X_test/255
cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        grey=cv2.Cvtcolor(frame,cv2.COLOR_BGR2GREY)
        height,width=grey.shape
        upperleft=(int(width/2-50),int(height/2-50))
        bottomright=(int(width/2+50),int(height/2+50)) 
        cv2.rectangle(grey,upperleft,bottomright,(0,255,0),2)
        roi=grey[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0],]
        pil=Image.fromarray(roi)
        Image_bw=Image.pil.convert("l")
        Image_bw=pil.convert()
        Image_bw_resize((28,28),Image.ANTIALIAS)
        pixelfilter=20
        minpixel=np.percential(Image_bw,minpixel)


""