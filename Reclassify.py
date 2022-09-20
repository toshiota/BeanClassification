# .h5モデルでの新規画像結果出力　

#　無分類フォルダを読み込み
#　分類フォルダ生成
#　分類結果によって対象フォルダに保存

import tensorflow as tf
from tensorflow import keras

import cv2
import time
from keras.preprocessing.image import array_to_img, img_to_array, load_img
# Helper libraries
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


print(tf.__version__)

class_names = ['OMOTE_Bad', 'OMOTE_Good', 'URA_Bad', 'URA_Good', 'DOUBLE']

#モデル読み込み
new_model = keras.models.load_model('/Users/toshi/Pictures/Bean/20220831_1332model_output.h5', compile=False, custom_objects=None)

#モデル読み込み


#未分類（元画像）フォルダ指定
#images = glob.glob(os.path.join('/Volumes/UNTITLED/TEST/', "*.jpg"))

#os.chdir("/Users/Toshi/PycharmProjects/TEST0001/TrainData/1906161616TEST")
images = glob.glob(os.path.join('/Users/toshi/Pictures/Bean/good', "*.jpg"))
#images = glob.glob(os.path.join('/Volumes/KIOXIA/photo', "*.jpg"))
print(images)
for i in range(1,len(images)) :
#for i in range(1000):
    img_src = cv2.imread(images[i], cv2.IMREAD_COLOR)      #`保存用画像

    img = img_to_array(load_img(images[i], grayscale = False, target_size = (128,128)))    #処理用画像  , interpolation = 'nearest'
    X = []
    X.append(img)
    X = np.asarray(img) / 255.0
    X = X.reshape(1, 128, 128, 3)

    predictions_real = new_model.predict(X)        #学習データとの照合
    predicted_label = np.argmax(predictions_real)
    #最大値のラベル　[０：OMOTE_Bad　１；OMOTE_Good　２；URA_Bad　３；URA_Good']

    Score0=(predictions_real * 100)
    OO=str(round(Score0[0,0],2))
    ON=str(round(Score0[0,1],2))
    UO=str(round(Score0[0,2],2))
    UN=str(round(Score0[0,3],2))
    print(i, predicted_label)

    if predicted_label==0:
        image_path = '/0/' 
        cv2.imwrite( image_path + str(i)+"b.jpg", img_src))

    if predicted_label==1:
        image_path ='/1/'  
        cv2.imwrite( image_path + str(i)+"g.jpg", img_src)

    if predicted_label==2:
        image_path = '/2/' 
        cv2.imwrite( image_path + str(i)+"d.jpg", img_src)  

    if predicted_label==3:
        image_path = '/3/'
        cv2.imwrite( image_path + str(i)+".jpg", img_src)

    if predicted_label==4:
        image_path = '/4/' 
        cv2.imwrite( image_path + str(i)+"d.jpg", img_src)
        
print("Finish")

