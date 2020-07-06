import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

groups_folder_path = '../.image/stotal/'
categories = ["귀여운 고양이"]

num_classes = len(categories)

image_w = 100
image_h = 100

X = []
Y = []
num =0
for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir + filename)
            img = cv2.imread(image_dir + filename, cv2.IMREAD_GRAYSCALE)
            r_img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
            out_img = cv2.imwrite('../.image/stotal/stotal/cane'+str(num)+".jpg",r_img)
            num +=1
            print(r_img.shape)
            X.append(img / 255)
            Y.append(label)
            print(num)
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(type(X))
def dataset ():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    xy = (X_train, X_test, Y_train, Y_test)
    print(X_train.shape)
    print(X_train[0])
    print(X_train[1])
    print(Y[0])
    return (X_train, X_test, Y_train, Y_test)
    #np.save("./img_data.npy", xy)



