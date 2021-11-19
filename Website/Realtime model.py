#!/usr/bin/env python
# coding: utf-8

# ## Realtime model

# In[1]:


import sys
import datetime
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import numpy as np
import os, random, cv2, shutil, platform, sys
from PIL import Image
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
x_train=pd.read_csv("x_train.csv")
y_train=pd.read_csv("y_train.csv")
x_test=pd.read_csv("x_test.csv")
y_test=pd.read_csv("y_test.csv")
col_list=['eccentricity', 'extent', 'moments_hu-0', 'moments_hu-1', 'moments_hu-2', 'euler_number', 'mean_intensity', 'std_intensity', '25th Percentile', '75th Percentile', 'iqr']


# In[6]:


def removeBackground(filepath, folderpath, filename):
    img = cv2.imread(filepath)#Get image link from the database
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36,39,20), (86,255,255))
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    mask_yellow = cv2.inRange(hsv, (21, 39, 64), (40, 255, 255))
    mask = cv2.bitwise_or(mask_green, mask_brown)
    mask = cv2.bitwise_or(mask, mask_yellow)
    res = cv2.bitwise_and(img,img, mask= mask)
    cv2.imwrite(filepath+"_removed.jpg", res)
    f_img= filepath + '_removed.jpg'
    img = Image.open(f_img)
    img = img.resize((6000,4000))
    img.save(f_img)
    return f_img


# In[7]:


def extractFeature(f_img):
    df = pd.DataFrame()
    image = rgb2gray(imread(f_img))
    binary = image < threshold_otsu(image)
    binary = closing(binary)
    label_img = label(binary)
    table = pd.DataFrame(regionprops_table(label_img, image,['convex_area', 'area', 'eccentricity','extent', 'inertia_tensor','major_axis_length', 'minor_axis_length','perimeter', 'solidity','orientation', 'moments_central','moments_hu', 'euler_number','equivalent_diameter','mean_intensity', 'bbox']))
    table['perimeter_area_ratio'] = table['perimeter']/table['area']
    real_images = []
    std = []
    mean = []
    percent25 = []
    percent75 = []
    for prop in regionprops(label_img): 
        min_row, min_col, max_row, max_col = prop.bbox
        img = image[min_row:max_row,min_col:max_col]
        real_images += [img]
        mean += [np.mean(img)]
        std += [np.std(img)]
        percent25 += [np.percentile(img, 25)] 
        percent75 += [np.percentile(img, 75)]
    table['mean_intensity'] = mean
    table['std_intensity'] = std
    table['25th Percentile'] = mean
    table['75th Percentile'] = std
    table['iqr'] = table['75th Percentile'] - table['25th Percentile']
    df = pd.concat([df, table.iloc[:1,:]], axis=0)
    df = df[col_list]
    return df


# In[8]:


from sklearn.preprocessing import StandardScaler
def normalize(df):
    frames = [df, x_train]
    result = pd.concat(frames)
    scaler = StandardScaler()
    df2 = scaler.fit_transform(result)
    df2 = pd.DataFrame(df2, columns = col_list)
    row_1=df.iloc[0]
    df3=pd.DataFrame(row_1)
    df3= df3.transpose()
    return df3


# In[9]:


def getresult(df3):
    seed = 14
    base_cls = DecisionTreeClassifier()
    num_trees = 101
    model = BaggingClassifier(base_estimator = base_cls,n_estimators = num_trees,random_state = seed)
    model.fit(x_train,y_train.values.ravel())
    pred1 = model.predict(df3)
    if(pred1[0]==0):
        print('Guava')
    elif(pred1[0]==1):
        print('jamun')
    elif(pred1[0]==2):
        print('Lemon')
    else:
        print('Mango')


# In[11]:

folderpath = sys.argv[1]
filename = sys.argv[2]
filepath = os.path.join(folderpath, filename)
f_img=removeBackground(filepath, folderpath, filename)
df=extractFeature(f_img)
df3= normalize(df)
getresult(df3)
