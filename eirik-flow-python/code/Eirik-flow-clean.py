import numpy as np
import matplotlib.pyplot as plt
import pickle

datapath = '/../Data/'
with open(datapath+'MNISTData.pkl', 'rb') as fp :
    data_dict = pickle.load(fp)

#constants
h = .1
n = 100
randOffset = 0.5
randRange  = 1.
picSize = len(data_dict['train_images'][0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#returns random weights and biaces 
def randWeights(shapeArr):
    wArr = []
    bVec = []
    for i in range(len(shapeArr)-1):
        wArr.append((np.random.rand(shapeArr[i+1], shapeArr[i]) - randOffset) * randRange) 
        bVec.append((np.random.rand(shapeArr[i+1]) - randOffset) * randRange)
    return wArr,bVec

#returns two lists of zero numpy arrays in the shape given by shapeArr
def zeroNabla(shapeArr):
    wArr = []
    bVec = []
    for i in range(len(shapeArr)-1):
        wArr.append(np.zeros((shapeArr[i+1], shapeArr[i])))
        bVec.append((np.zeros(shapeArr[i+1])))
    return wArr,bVec

#returns the guess given by the weights and biaces wArr and bVec
def guess(N,wArr,bVec,label):
    a = data_dict[label+'_images'][N]
    for i in range(len(wArr)):
        a = sigmoid(wArr[i] @ a + bVec[i])
    return a

#saves the weights and biaces
def save(wArr,bVec):
    pDict = {}
    pDict['w'] = wArr
    pDict['b'] = bVec
    with open(datapath+'pDict.pkl', 'wb') as fp :
        pickle.dump(pDict, fp)

#loads the weights and biaces
def load():
    with open(datapath+'pDict.pkl', 'rb') as fp :
        pDict = pickle.load(fp)
    return pDict['w'],pDict['b']

#tests given weights and biaces wArr and bVec on the 10 000 test_images and returns the proportion that where guessed correctly
def test(wArr,bVec):
    data = data_dict['test_images']
    labels = data_dict['test_labels']
    correct = 0
    for i in range(len(data)):
        c = np.argmax(labels[i])
        g = np.argmax(guess(i,wArr,bVec,'test'))
        if g == c:
            correct +=1
    return correct/len(data)

#trains the given weights and biaces wArr and bVec on the train_images data set. Passes through the dataset N times
def train(wArr,bVec,shapeArr,N):
    data = data_dict['train_images']
    cost = data_dict['train_labels']

    wNabla, bNabla = zeroNabla(shapeArr)

    a = []
    a.append(data[0])
    for i in range(len(wArr)):
        a.append(sigmoid(wArr[i] @ a[i] + bVec[i]))
    
    count = 0
    trained = False
    while not trained:
        print("count",count)
        for i in range(len(data)):
            a[0] = data[i]
            for j in range(0,len(wArr)):
                a[j+1] = sigmoid(wArr[j] @ a[j] + bVec[j])

            c = 2 * (a[len(wArr)] - cost[i])
            for j in range(len(wArr),0,-1):
                wNabla[j - 1]  += np.outer(a[j]*(1-a[j]) * c, a[j-1] )
                bNabla[j - 1]  += a[j] * (1 - a[j]) * c
                c =  wArr[j - 1].T @ ((a[j] * (1-a[j]))*c)

            if i % n == 0:
                for i in range(len(wArr)):
                    wArr[i] -= h *  wNabla[i]/n
                    bVec[i] -= h *  bNabla[i]/n
                wNabla, bNabla = zeroNabla(shapeArr)
        count += 1

        if count == n:
            trained = True

shapeArr = [picSize,16,16,10]
wArr, bVec = randWeights(shapeArr)
train(wArr,bVec,shapeArr,1)
save(wArr,bVec)
print(test(wArr,bVec))

