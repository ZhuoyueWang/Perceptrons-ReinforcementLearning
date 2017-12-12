import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import pprint
import random

def read_training_data():
    trainingFile = open("trainIamageOutput2.txt", "r")
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
    testFile = open("testIamageOutput2.txt", "r")
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



def perceptrons_classifier(image_data,data_labels,image_test,test_labels,epoch,isBia,isRandom,isShuffle,alpha):
#train
    [image_depth,image_rows, image_columns] = np.shape(image_data)

    for i in range(image_depth):
        temp = np.array(image_data[i])
        image_data[i] = temp.reshape(784,)

    num_features = len(image_data[0])
    num_class = 10
    num_labels = len(data_labels)

    if isBia:
        num_features += 1
        image_data = np.insert(image_data,0,1,axis=1)

    if isRandom:
        weight =np.random.rand(epoch,num_class,num_features)
    else:
        weight =np.zeros((epoch,num_class,num_features))

    accuracy=[]
    idx=[]

    for ep in range(1,epoch):
        missList = np.ones(num_labels)
        weight[ep] = weight[ep-1]
        idxList = [i for i in range(num_labels)]
        if isShuffle:
            random.shuffle(idxList)
        for instance in idxList:
            result = []
            for cla in range(num_class):
                result.append(np.sum(np.multiply(weight[ep,cla],image_data[instance])))
            prediction=result.index(max(result))
            truth = data_labels[instance]
            if truth == prediction:
                missList[instance] = 0
            else:
                mult = np.multiply(image_data[instance],alpha/(alpha+ float(ep)))
                weight[ep,prediction]= np.subtract(weight[ep,prediction],mult)
                weight[ep,truth]=np.add(weight[ep,truth],mult)
        print('This is {0} epoch and its accuracy is: {1}'.format(ep,1-np.sum(missList)/num_labels))
        accuracy.append(1-np.sum(missList)/num_labels)
        idx.append(ep)

    plt.plot(idx,accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Training Curve')
    plt.axis([0,epoch,0,1])
    plt.show()
    plt.savefig('result.png')
    np.set_printoptions(threshold=np.nan)
    print(weight[0][0].shape)
    plt.imshow(np.reshape(weight[0][0],(28,28)),interpolation='nearest')
    plt.show()

#test
    [test_path,test_rows, test_columns] = np.shape(image_test)
    for i in range(test_path):
        temp = np.array(image_test[i])
        image_test[i] = temp.reshape(784,)
    if isBia:
        image_test=np.insert(image_test,0,1,axis=1)
    weight = weight[epoch-1]
    prediction=[]
    for instance in range(len(test_labels)):
        result = []
        for cla in range(num_class):
            result.append(np.sum(np.multiply(weight[cla],image_test[instance])))
        prediction.append(result.index(max(result)))

#data processing
    accuracyCount=0
    for i in range(0,1000):
        if prediction[i] == test_labels[i]:
            accuracyCount+=1
    print('Overall test accuracy: {}'.format(accuracyCount/1000))

    classificationList=[]
    for i in range(0,10):
        classificationRate = 0.0
        classcount = 0
        for img in range(0,1000):
            if i == test_labels[img]:
                classcount += 1
                if i == prediction[img]:
                    classificationRate += 1
        print('Classification rate for digit {0} is {1}'.format(i,round(classificationRate/classcount,5)))
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
            colList.append(round(confusion/float(classcount),5))
        confusionMatrix.append(colList)

    for row in confusionMatrix:
        string=''
        for col in row:
            string+=str(round(col,4))+' '
        print(string)





def main():
    image_data, data_labels = read_training_data()
    image_test, test_labels = read_test_data()
    perceptrons_classifier(image_data,data_labels,image_test,test_labels,100,False,True,True,100)


if __name__== "__main__":
  main()


'''
def perceptrons_classifier1(image_data,data_labels,data_depth,image_test,test_labels,test_depth,epoch,isBia,isRandom,isShuffle):
    [image_depth,image_rows, image_columns] = np.shape(image_data)

    num_features = 2
    bias = 0
    weight = [[[0 for d in range(28)] for x in range(28)] for y in range(10)]
    weight = np.array(weight)
    images = []
    for d in range(10):
        for i in range(28):
            for j in range(28):
                if isRandom:
                    weight[d][i][j] = 0
                else:
                    weight[d][i][j] = random.uniform(0, 1)
    for i in range(data_depth):
        temp = image(image_data[i], data_labels[i])
        images.append(temp)

    if isShuffle:
        random.shuffle(images,random.random)
    if isBia:
        bias = 1
    else:
        bias = 0

    confusion = [[0 for x in range(10)] for y in range(10)]
    confusion = np.array(confusion)

    #training
    for step in range(1,epoch+1, 1):
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
                    result += weight[d][x][y]*test[x][y]+bias
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

    return confusion,totalAccuracy

'''
