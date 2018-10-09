from os import listdir
import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from sklearn.externals import joblib
imageName = ["ERROR","Bien1","Bien2","Bien3","Bien4","Bien5","Bien6","Bien7","Bien8","Bien9","Bien10","Bien11","Bien12"]

SIZE = 50
CLASS_NUMBER = 11
np.set_printoptions(threshold='nan')
#Ham dem file de tao mang
def countFile():
    lenth =0
    for sign_type in range(CLASS_NUMBER):
        sign_list = listdir("dataset/{}".format(sign_type))
        for sign_file in sign_list:
            if '.png' in sign_file:
                lenth = lenth +1
    return lenth
#Ham doc file va tao database
def load_traffic_dataset():
    length = countFile()
    dataset = [[-1 for x in range(SIZE*SIZE)] for y in range(length)]
    r = 0
    labels= [[-1 for x in range(1)] for y in range(length)]
    for sign_type in range(CLASS_NUMBER):
        sign_list = listdir("dataset/{}".format(sign_type))
        for sign_file in sign_list:
            if '.png' in sign_file:
                path = "dataset/{}/{}".format(sign_type,sign_file)
                print(path)
                img = cv2.imread(path,0)
                img = cv2.resize(img, (SIZE, SIZE))
                c=0
                for row in img:
                    for pixel in row:
                        if pixel > 127:
                            pix =1
                        else:
                            pix =0
                        dataset[r][c]=pix
                        c=c+1
                labels[r][0] = sign_type                
                r = r +1
    return np.array(dataset), np.array(labels)
#Ham chuyen doi anh dau vao thanh mang gia tri
def deskew(img):
    x_test= []
    img = cv2.resize(img,(SIZE,SIZE))
    for row in img:
        for pixel in row:
            if (pixel > 127):
                x_test.append(1)
            else:
                x_test.append(0)
    return [x_test]
def trainning():
    #Random, hoan vi du lieu
    data, labels = load_traffic_dataset()
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(data))
    data, labels = data[shuffle], labels[shuffle]
    #Chia du lieu thanh 2 tap, tap train 90%, tap test 10%
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1, random_state=10)        
    y = np.ravel(labels_train)
    labels_train = np.array(y).astype(int)
    clf =SVC(gamma = 0.01)
    print("Trainning and export data...")
    clf.fit(data_train, labels_train)#Trainning du llieu
    joblib.dump(clf, 'dataset.pkl')#Xuat file luu tru
    return clf

#Ham doc file da trainning
def reloadData():
    clf = joblib.load('dataset.pkl')
    return clf
#Ham du doan ket qua dua tren tap train va dau vao
def getLabel(clf,image):
    testValue = deskew(image)
    index = clf.predict(testValue)[0]
    return index

