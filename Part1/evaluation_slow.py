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

class digit:
    def __init__(self, _weight, _label):
        self.weight = _weight
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

    images = []
    digits = []
    for d in range(10):
        for i in range(28):
            for j in range(28):
                if isRandom:
                    weight = [[0 for x in range(28)] for y in range(28)]
                    weight[i][j] = 0
                    digits.append(digit(weight, d))
                else:
                    weight = [[0 for x in range(28)] for y in range(28)]
                    weight[i][j] = random.uniform(0, 1)
                    digits.append(digit(weight, d))

    for i in range(data_depth):
        images.append(image(image_data[i], data_labels[i]))

    if isBia:
        bias = 1
    else:
        bias = 0


    #training
    accuracy = []
    for step in range(1,epoch+1, 1):
        alpha = 5 / (5 + step)
        incorrect = 0
        if isShuffle:
            random.shuffle(images,random.random)
        for i in range(len(images)):
            data = images[i].imagelist
            maxResult = -99999
            bestLabel = -1
            for d in range(10):
                result = 0
                for x in range(image_rows):
                    for y in range(image_columns):
                        result += (digits[d].weight)[x][y]*data[x][y]+bias
                if result > maxResult:
                    maxResult = result
                    bestLabel = d

            trueLabel = images[i].label
            if bestLabel != trueLabel:
                incorrect += 1
                for x in range(image_rows):
                    for y in range(image_columns):
                        (digits[trueLabel].weight)[x][y] = (digits[trueLabel].weight)[x][y]+alpha*data[x][y]
                        (digits[bestLabel].weight)[x][y] = (digits[bestLabel].weight)[x][y]-alpha*data[x][y]
        currAcrc = round(((len(images)-incorrect)/len(images)), 4)
        print("step {0}, accuracy {1}".format(step, currAcrc))
        accuracy.append(currAcrc)

    confusion = [[0 for x in range(10)] for y in range(10)]

    #testing
    totalAccuracy = 0
    for i in range(test_depth):
        test = image_test[i]
        trueLabel = test_labels[i]
        bestLabel = -1
        maxResult = -99999
        for d in range(10):
            result = bias
            for x in range(image_rows):
                for y in range(image_columns):
                    result += (digits[d].weight)[x][y]*test[x][y]+bias
            if result > maxResult:
                maxResult = result
                bestLabel = d
        confusion[trueLabel][bestLabel] += 1
        if trueLabel == bestLabel:
            totalAccuracy += 1

    for i in range(10):
        for j in range(10):
            confusion[i][j] = round(confusion[i][j]/test_labels.count(i), 4)
    totalAccuracy = totalAccuracy / test_depth


    print("the confusion matrix is ")
    for i in range(10):
        for j in range(10):
            #print("HERE")
            print(confusion[i][j], end=" ")
        print()
    print("the total accuracy is ")
    print(totalAccuracy)

    return confusion,totalAccuracy





def main():
    image_data, data_labels,data_depth = read_training_data()
    image_test, test_labels,test_depth = read_test_data()
    perceptrons_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth,100,True,False,True)
    '''
    confusionList = [0 for i in range(1,101)]
    accuracyList = [0 for i in range(1,101)]
    for i in range(1,101):
        print(i)
        confusionList[i],accuracyList[i] = perceptrons_classifier(image_data,data_labels,data_depth,image_test,test_labels,test_depth,i,False,True,True)
    print("The maximum accuracy is {}".format(max(accuracyList)))
    idx = accuracyList.index(max(accuracyList))
    print("the confusion matrix with maximum accuracy is ")
    for i in range(10):
        for j in range(10):
            print(confusionList[idx][i][j], end=" ")
        x = np.linspace(1,101,1)
    '''
    plt.plot(x,accuracyList,'bo')
    plt.savefig('result.png')


if __name__== "__main__":
  main()
