import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import pprint
import operator
import scipy.spatial.distance as dist

def read_training_data():
    trainingFile = open("trainIamageOutput1.txt", "r")
    lines = trainingFile.readlines()
    image_num = int(len(lines)/28)
    image_data = []
    for i in range(image_num):
        data = []
        for j in range(28*i,28*i+28):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [int(a) for a in line]
            data.append(elem)
        image_data.append(data)

    trainingLabel = open("traininglabels", "r")
    labels = trainingLabel.readlines()
    data_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        data_labels.append(int(label))

    return image_data, data_labels



def read_test_data():
    testFile = open("testIamageOutput1.txt", "r")
    lines = testFile.readlines()
    image_num = int(len(lines)/28)
    image_test = []
    for i in range(image_num):
        data = []
        for j in range(28 * i, 28 * i + 28):
            line = lines[j]
            line = line.rstrip('\n')
            elem = [int(a) for a in line]
            data.append(elem)
        image_test.append(data)

    testLabel = open("testlabels", "r")
    labels = testLabel.readlines()
    test_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        test_labels.append(int(label))

    return image_test, test_labels



def KNN_classifier(image_data,data_labels,image_test,test_labels,k):
#train
    [image_depth,image_rows, image_columns] = np.shape(image_data)
    for i in range(image_depth):
        temp = np.array(image_data[i])
        image_data[i] = temp.reshape(784,)

    [test_path,test_rows, test_columns] = np.shape(image_test)
    for i in range(test_path):
        temp = np.array(image_test[i])
        image_test[i] = temp.reshape(784,)

    num_features = len(image_data[0])
    num_class = 10
    num_datalabels = len(data_labels)
    num_testlabels = len(test_labels)
    prediction=[]

    for i in range(num_testlabels):
        nearestNeighbors={}
        for j in range(num_datalabels):
            nearestNeighbors[j] = dist.hamming(image_data[j],image_test[i])
        nearestNeighbors=sorted(nearestNeighbors.items(),key=operator.itemgetter(1))
        classvote={}
        neighbor = list(nearestNeighbors[:k])
        for i in range(len(neighbor)):
            response=data_labels[neighbor[i][0]]
            if response in classvote:
                classvote[response]+=1
            else:
                classvote[response]=1
        result=sorted(classvote.items(),key=operator.itemgetter(1),reverse=True)
        vote = result[0][0]
        prediction.append(vote)

    accuracyCount=0
    for i in range(0,1000):
        if int(prediction[i])==int(test_labels[i]):
            accuracyCount+=1
    print('Overall: '+ str(accuracyCount/1000.0))
    classificationList = []
    for i in range(0,10):
        classificationRate = 0.0
        classcount = 0
        for img in range(0,1000):
            if i == int(test_labels[img]):
                classcount+=1
                if i == int(prediction[img]):
                    classificationRate+=1
        print('Classification rate for digit {0} is: {1}'.format(i,round(classificationRate/classcount,4)))
        classificationList.append(classificationRate)

	# confusion matrix
    print('----Confusion Matrix----')
    confusionMatrix = []
    for i in range(0,10): #row is the real
        colList = []
        for j in range(0,10): #col is the predicted
            classcount = 0
            confusion = 0
            for img in range(0,1000):
                if i == test_labels[img]:
                    classcount += 1
                if prediction[img] == j and test_labels[img] == i:
                    confusion += 1
            colList.append(round(confusion/float(classcount),4))
        confusionMatrix.append(colList)


    for row in confusionMatrix:
        string=''
        for col in row:
            string+=str(round(col,4))+' '
        print(string)

def main():
    image_data, data_labels = read_training_data()
    image_test, test_labels = read_test_data()
    KNN_classifier(image_data,data_labels,image_test,test_labels,6)


if __name__== "__main__":
  main()
