from math import log2
import operator
import pickle
import numpy as np

from sklearn.datasets import load_iris
from sklearn import tree


def creatDataset():  # Data set formation
    iris = load_iris()
    X, y = iris.data, iris.target
    dataset=[]
    for index,x in enumerate(X):
        x=list(x)
        x.append(y[index])
        dataset.append(x)
    labels=['SepalLength','SepalWidth','PetalLength','PetalWidth']
    return dataset, labels


def calcShannonEnt(dataset):  # Calculate information entropy
    num_of_Entries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = labelCounts[key] / num_of_Entries
        shannonEnt -= prob * log2(prob)
    return shannonEnt


def splitDataSet(dataset, axis, value):  
    subDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reduce_featVec = featVec[0:axis]
            reduce_featVec.extend(featVec[axis+1:])
            subDataSet.append(reduce_featVec)
    return subDataSet


# Select the best partition data set features (ID3 information gain method)
def chooseBestFeatureToSplit(dataset):
    num_of_feature = len(dataset[0])-1
    baseEntropy = calcShannonEnt(dataset)
    best_info_gain = 0.0
    bestLabel_index = -1
    for i in range(num_of_feature):
        feature_vec = [each[i] for each in dataset]
        uniqueVals = set(feature_vec)
        newEntroy = 0.0
        for value in uniqueVals:
            subdataset = splitDataSet(dataset, i, value)
            prob = len(subdataset) / float(len(dataset))
            newEntroy += prob * calcShannonEnt(subdataset)
        info_gain = baseEntropy - newEntroy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            bestLabel_index = i
    return bestLabel_index


# majority voting function
def majorityVote(classlist):
    classcount = {}
    for each in classlist:
        if each not in classcount.keys():
            classcount[each] = 0
        classcount[each] += 1
    sorted_classcount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classcount[0][0]

# tree creation
def creatTree(dataset, label):
    t_label = label[:]
    classlist = [each[-1] for each in dataset]
    # set the termination condition of the tree
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset) == 1:
        return majorityVote(classlist)
    best_feature_index = chooseBestFeatureToSplit(dataset)
    best_feature = t_label[best_feature_index]
    myTree = {best_feature: {}}
    del(t_label[best_feature_index])
    feature_vec = [each[best_feature_index] for each in dataset]
    uniqueValues = set(feature_vec)
    for value in uniqueValues:
        sublabel = t_label[:]
        myTree[best_feature][value] = creatTree(splitDataSet(dataset, best_feature_index, value), sublabel) #递归调用createTree来创建树
    return myTree

# discriminant function
def classify(mytree, feat_label, testvec):
    firstStr = list(mytree.keys())[0]
    secondDict = mytree[firstStr]
    feat_index = feat_label.index(firstStr)
    for key in secondDict.keys():
        if testvec[feat_index] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], feat_label, testvec)
            else:
                classLabel = secondDict[key]
    return classLabel 



if __name__ == "__main__":
    filepath = 'storaged_Tree'

    dataSet, Labels = creatDataset()
    np.random.shuffle(dataSet)
    train_dataSet=dataSet[:100]
    test_dataSet=dataSet[100:]
    mytree=creatTree(dataSet, Labels)
    # storeTree(mytree, filepath)
    # Tree = loadTree(filepath)
    # re = classify(mytree, Labels, [1, 1,1,1])
    # print(re)
    n=0
    for test in test_dataSet:
        re=classify(mytree, Labels,test)
        # print(re)
        # print(test[4])
        if (re==test[4]):
            n+=1
    print(n/50,'%')
