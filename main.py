#This code is machine specific but it is an excersize in my own understanding on Convolutional Neural Networks. The only thing that currently is not implemented is backpropagational weight and bias updates as my understanding of these topics is not yet 100% clear

import skimage
import sklearn
from sklearn import datasets
import numpy
import random
import matplotlib
import timeit
from skimage import data, io
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean

def findCost(outputVector, y):
    if y == 0:
        #outputVector[0] should be 0
        #outputVector[1] should be 1
        cost = outputVector[0] + (outputVector[1] - 1)
        cost = cost * cost
        return cost
    else:
        #outputVector[0] should be 1
        #outputVectore[1] should be 0
        cost = outputVector[1] + (outputVector[0] - 1)
        cost = cost * cost
        return cost

def sigmoid(x):
    return numpy.exp(-numpy.logaddexp(0, -x))

X = numpy.zeros((25000, 15, 15)) #Our dataset
y = numpy.zeros(25000) #Our targets

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)

def coinToss(): #Making binary decisions before learning
    x = random.random()
    if x > 0.5:
        return "H"
    elif x < 0.5:
        return "T"
    else:
        return coinToss()
    
def ReLU(img): #implements the ReLU layer (takes image and gets rid of negative values)
    for i in range (0, img.shape[0]):
        for j in range (0, img.shape[1]):
            img[i, j] = numpy.maximum(0, img[i, j])
    return img

def Pool(img): #pool layer with hyperparameter 2, takes regions of image and max-pools them
    poolSize = 2
    posYInImg = 0
    posXInImg = 0
    MaxY = int(numpy.floor(img.shape[0]/poolSize))
    MaxX = int(numpy.floor(img.shape[1]/poolSize))
    pooledImg = numpy.zeros((1, MaxY, MaxX))
    currentX = 0
    currentY = 0
    while currentX < MaxX and currentY < MaxY:
        currentMaxValue = 0
        for x in range(0, poolSize):
            for y in range(0, poolSize):
                if img[posYInImg + y, posXInImg + x] > currentMaxValue:
                    currentMaxValue = img[posYInImg + y, posXInImg + x]
        pooledImg[0, currentY, currentX] = currentMaxValue
        if currentY + 1 == MaxY:
            currentY = 0
            posYInImg = 0
            currentX = currentX + 1
            posXInImg = posXInImg + poolSize
        else:
            currentY = currentY+1
            posYInImg = posYInImg + poolSize
    return pooledImg[0, :, :]

def Convole(img, filterApply, filterDimensions, step): #Standard convolutional layer
    #Lets be sure img is grayscale
    img = skimage.color.rgb2gray(img)
    #Get some dimensions to help us set variables
    imgYDimension = img.shape[0]
    imgXDimension = img.shape[1]  
    outputImg = numpy.zeros((1, int(imgYDimension/step), int(imgXDimension/step)))  

    posYInImg = 0
    posXInImg = 0
    MaxX = numpy.floor(imgXDimension/step)
    MaxY = numpy.floor(imgYDimension/step)
    currentX = 0
    currentY = 0
    while currentX < MaxX  and currentY < MaxY:
        for x in range(0, filterDimensions):
            for y in range(0, filterDimensions):
                outputImg[0, currentY, currentX] += (img[posYInImg, posXInImg] * filterApply[y, x])
        if currentY + 1 == MaxY:
            currentY = 0
            posYInImg = 0
            currentX = currentX + 1
            posXInImg = posXInImg + step
        else:
            currentY = currentY+1
            posYInImg = posYInImg + step
    return outputImg[0, :, :]
    
#Setup
random.seed()

countOfCats = 0
countOfDogs = 0

#settings up  weight matrix and bias vectors randomly
arrFilter = numpy.zeros((1,3,3))
arrFilter[0, :, :] = numpy.array([[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]])
weightMatrix1 = numpy.random.rand(225, 10)
for weighty in range(0, 225):
    for weightx in range(0, 10):
        if coinToss() == "T":
            weightMatrix1[weighty, weightx] = weightMatrix1[weighty, weightx] * -1
weightMatrix2 = numpy.random.rand(10, 2)
for weighty in range(0, 10):
    for weightx in range(0, 2):
        if coinToss() == "T":
            weightMatrix2[weighty, weightx] = weightMatrix2[weighty, weightx] * -1
biasVector1 = numpy.random.rand(10)
for biasCount in range(0, 10):
    if coinToss() == "T":
        biasVector1[biasCount] = biasVector1[biasCount] * - 1
    biasVector1[biasCount] = biasVector1[biasCount] * 10
biasVector2 = numpy.random.rand(2)
for biasCount in range(0, 2):
    if coinToss() == "T":
        biasVector2[biasCount] = biasVector2[biasCount] * - 1
    biasVector2[biasCount] = biasVector2[biasCount] * 2
for i in range(0, 10): #randomise image order, only doing 10 images to save processing time when testing
    if coinToss() == "H" and countOfCats < 12500:
        imagePath = str(r"C:\Users\daniel.deburgo\Downloads\CatsAndDogs\train\cats\cat."+str(int(countOfCats))+".jpg")
        img = io.imread(imagePath) #Open our cat
        countOfCats = countOfCats + 1
        catDog = "Cat"
    else:
        imagePath = str(r"C:\Users\daniel.deburgo\Downloads\CatsAndDogs\train\dogs\dog."+str(int(countOfDogs))+".jpg")
        img = io.imread(imagePath) #Open our cat
        countOfDogs = countOfDogs + 1
        catDog = "Dog"
    img = resize(img, (1000, 1000))
    img = skimage.color.rgb2gray(img) #Convert our image to grayscale
    
    #io.imshow(img)
    #plt.show()
    
    #Begin convolution
    
    layerOneConvOut1 = ReLU(Convole(img, arrFilter[0, :, :], 3, 2))
    layerOnePoolOut1 = Pool(layerOneConvOut1)
    layerTwoConvOut1 = ReLU(Convole(layerOnePoolOut1, arrFilter[0, :, :], 3, 2))
    layerTwoPoolOut1 = Pool(layerTwoConvOut1)
    layerThreeConvOut1 = ReLU(Convole(layerTwoPoolOut1, arrFilter[0, :, :], 3, 2))
    layerThreePoolOut1 = Pool(layerThreeConvOut1)
    X[i, :, :] = layerThreePoolOut1
    if catDog == "Cat":
        y[i] = 1
    else:
        y[i] = 0
    print(str(int(i+1))+"/10")
    # io.imshow(layerThreePoolOut1)
    # plt.show()
   
    #GUESS WHAT THE IMAGE IS BASED ON GUESSINGMATRIX
    inputVector = layerThreePoolOut1.flatten() #225 neurons as numbers :)  
    hiddenVector = numpy.zeros(10)
    for hiddenCount in range(0, 10):
        for inputCount in range(0, 225):
            hiddenVector[hiddenCount] = hiddenVector[hiddenCount] + (inputVector[inputCount] * weightMatrix1[inputCount, hiddenCount])
        hiddenVector[hiddenCount] = hiddenVector[hiddenCount] + biasVector1[hiddenCount]
    hiddenVector = sigmoid(hiddenVector)
    outputVector = numpy.zeros(2)
    for outputCount in range(0, 2):
        for hiddenCount in range(0, 10):
            outputVector[outputCount] = outputVector[outputCount] + (hiddenVector[hiddenCount] * weightMatrix2[hiddenCount, outputCount])
        outputVector[outputCount] = outputVector[outputCount] + biasVector2[outputCount]
    outputVector = sigmoid(outputVector)
    print("OutputVector: "+str(outputVector))
    #IMPROVE THE GUESS
    #ITERATE IN EPOCHS
    


for i in range(1, 12500):
    imagePath = str(r"C:\Users\daniel.deburgo\Downloads\CatsAndDogs\test1\t"+str(int(i))+".jpg")
    img = io.imread(imagePath) #Open our image
    img = resize(img, (1000, 1000))
    img = skimage.color.rgb2gray(img) #Convert our image to grayscale
    
    #Begin convolution
    
    layerOneConvOut1 = ReLU(Convole(img, arrFilter[0, :, :], 3, 2))
    layerOnePoolOut1 = Pool(layerOneConvOut1)
    layerTwoConvOut1 = ReLU(Convole(layerOnePoolOut1, arrFilter[0, :, :], 3, 2))
    layerTwoPoolOut1 = Pool(layerTwoConvOut1)
    layerThreeConvOut1 = ReLU(Convole(layerTwoPoolOut1, arrFilter[0, :, :], 3, 2))
    layerThreePoolOut1 = Pool(layerThreeConvOut1)
    
    

    io.imshow(img)
    plt.show()
    
    
#Images leave layer 2 as (62, 62) matrices
#arrImg = numpy.zeros((noOfImages*2, 62, 62))
#fullyConnected(arrImg, [1, 1], 1)
