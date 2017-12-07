import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import pprint
import random

class image:
    def __init__(self, _imagelist, _label):
        self.imagelist = _imagelist
        self.label = _label

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
    data_depth = len(labels)
    data_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        data_labels.append(int(label))

    return image_data, data_labels, data_depth



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
    test_depth = len(labels)
    test_labels = []
    for i in range(len(labels)):
        label = labels[i]
        label = label.rstrip('\n')
        test_labels.append(int(label))

    return image_test, test_labels, test_depth






def perceptrons_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth,epoch,isBia,isRandom,isShuffle):
    [image_depth,image_rows, image_columns] = np.shape(image_data)

    num_features = 2
    bias = 0
    weight = [[[0 for d in range(28)] for x in range(28)] for y in range(10)]
    images = []
    print("START")
    for d in range(10):
        for i in range(28):
            for j in range(28):
                if isRandom:
                    weight[d][i][j] = 0
                else:
                    weight[d][i][j] = random.uniform(0, 1)
    print("finish weight")
    for i in range(data_depth):
        images.append(image(image_data[i], data_labels[i]))

    if isShuffle:
        random.shuffle(images,random.random)
    if isBia:
        bias = 1
    else:
        bias = 0

    confusion = [[0 for x in range(10)] for y in range(10)]

    #training
    for step in range(1,epoch+1, 1):
        print("trianing")
        alpha = 5 / (5 + step)
        for i in range(len(images)):
            data = images[i].imagelist
            maxResult = -99999
            bestLabel = -1
            for d in range(10):
                result = 0
                for x in range(image_rows):
                    for y in range(image_columns):
                        result += weight[d][x][y]*data[x][y]+bias
                if result > maxResult:
                    maxResult = result
                    bestLabel = d

            trueLabel = images[i].label
            if bestLabel != trueLabel:
                for x in range(image_rows):
                    for y in range(image_columns):
                        weight[trueLabel][x][y] = weight[trueLabel][x][y]+alpha*data[x][y]
                        weight[bestLabel][x][y] = weight[bestLabel][x][y]-alpha*data[x][y]

    print("finish training")
    #testing
    totalAccuracy = 0
    numCurrDight = [0 for x in range(10)]
    for i in range(test_depth):
        test = image_test[i]
        trueLabel = test_labels[i]
        bestLabel = -1
        numCurrDight[trueLabel] += 1
        for d in range(10):
            result = bias
            maxResult = -99999
            for x in range(image_rows):
                for y in range(image_columns):
                    result += weight[d][x][y]*test[x][y]+bias
            if result > maxResult:
                maxResult = result
                bestLabel = d

        confusion[trueLabel][bestLabel] += 1
        if trueLabel == bestLabel:
            totalAccuracy += 1

    for i in range(10):
        for j in range(10):
            confusion[i][j] = confusion[i][j]/test_labels.count(i)

    print("the confusion matrix is ")
    for i in range(10):
        for j in range(10):
            print(confusion[i][j], end=" ")
        print()
    print("the total accuracy is ")
    totalAccuracy = totalAccuracy / test_depth
    print(totalAccuracy)



def main():
    image_data, data_labels,data_depth = read_training_data()
    image_test, test_labels,test_depth = read_test_data()
    result = perceptrons_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth,3,False,True,True)

if __name__== "__main__":
  main()
