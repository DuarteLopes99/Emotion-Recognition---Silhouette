# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:33:05 2022

@author: DuarteLopes
"""
import tensorflow as tf 
import cv2
import os
import matplotlib.pyplot as plt 
import numpy as np 
import time
import random 
import pickle
from _faceDetection_ import faceDetection_MTCNN
from sklearn.utils import shuffle


with tf.device('/gpu:0'):
    
    def dataAugmentation (XData, YData, BATCH_SIZE, _type_):
       
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(#rescale=1./255,
                                                                        rotation_range=40,
                                                                        width_shift_range=0.1,
                                                                        height_shift_range=0.1,
                                                                        shear_range=0.1,
                                                                        zoom_range=0.4,
                                                                        horizontal_flip=True)
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255
                                                                      )
        test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255
                                                                     )
        if _type_ == "Train":
            dataSet_generator = train_datagen.flow(XData,
                                                   y = YData, 
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False)
            
            return dataSet_generator
        
        if _type_ == "Val":
            dataSet_generator = val_datagen.flow(XData,
                                             y = YData,
                                             batch_size = BATCH_SIZE,
                                             shuffle=False) 
        
            
            return dataSet_generator      
        
        if _type_ == "Test":
            dataSet_generator = test_datagen.flow(XData,
                                             batch_size = BATCH_SIZE,
                                             shuffle=False)                                       
            return dataSet_generator
        
        if _type_ == "Plot":
            plt_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                  width_shift_range=0.1,
                                                                  height_shift_range=0.1,
                                                                  shear_range=0.1,
                                                                  zoom_range=0.1,
                                                                  horizontal_flip=True)
            dataSet_generator = plt_datagen.flow(XData,
                                 batch_size = BATCH_SIZE,
                                 shuffle=False) 
            return dataSet_generator
    
    def create_trainingSet(Datadirectory,Classes):
        trainingData = []
        for category in Classes:
            path = os.path.join(Datadirectory,category) #Cria o path diretamente para a pasta de cada uma das emoções
            classes_num = Classes.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img)) #Cria o path para cada uma das imagens
                new_img = cv2.resize(img_array,(224,224))
                trainingData.append([new_img,classes_num])
        return trainingData
    
    def create_testingSet(Datadirectory,Classes):
        testingData = []
        for category in Classes:
            path = os.path.join(Datadirectory,category) #Cria o path diretamente para a pasta de cada uma das emoções
            classes_num = Classes.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img)) #Cria o path para cada uma das imagens
                new_img = cv2.resize(img_array,(224,224))
                testingData.append([new_img,classes_num])
        return testingData
    
    def create_dataSet(Datadirectory,Classes):
            Data = []
            for category in Classes:
                path = os.path.join(Datadirectory,category) #Cria o path diretamente para a pasta de cada uma das emoções
                classes_num = Classes.index(category)
                for img in os.listdir(path):
                    img_array = cv2.imread(os.path.join(path,img)) #Cria o path para cada uma das imagens
                    new_img = cv2.resize(img_array,(224,224))
                    Data.append([new_img,classes_num])
            return Data
    
    def dataDivider(DataSet, trainPercentage):
        nSamples = len(DataSet)
        random.shuffle (DataSet)
        lastTrainSample = int(np.ceil(trainPercentage * nSamples))
        trainingSet = DataSet [0 : lastTrainSample]
        testingSet = DataSet [lastTrainSample : nSamples]
        return trainingSet, testingSet
    
    def datDivider_mutipleList(DataSet, percentage):
        nSamples = len(DataSet[0])
        samples = DataSet[0]
        labels = DataSet[1]
        samples, labels = shuffle(samples, labels)
        
        lastTrainSample = int(np.ceil(percentage * nSamples))
        
        #Train
        trainSamples = samples [0 : lastTrainSample]
        trainLabels = labels [0 : lastTrainSample]
        trainSet = [trainSamples, trainLabels]
        
        #Test
        testSamples = samples [lastTrainSample : nSamples]
        testLabels = labels [lastTrainSample : nSamples]
        testSet = [testSamples, testLabels]
        
        return trainSet, testSet
        
    def dataSpliter(DataSet): 
        X = []
        Y = []
        
        for features, label in DataSet:
            X.append(features)
            Y.append(label)
                
        np.array(Y)
        
        return X, Y 
        
    def Number2Emotion (prediction,DataBase):
        if DataBase == "CK+":
            if prediction == 0:
                return "Anger"
            if prediction == 1:
                return "Comtempt"
            if prediction == 2:
                return "Disgust"
            if prediction == 3:
                return "Fear"
            if prediction == 4:
                return "Happiness"
            if prediction == 5:
                return "Sadness"
            if prediction == 6:
                return "Surprise"
            else:
                "Prediction Number does not match an Emotion"
        if DataBase == "FER":
            if prediction == 0:
                return "Anger"
            if prediction == 1:
                return "Disgust"
            if prediction == 2:
                return "Fear"
            if prediction == 3:
                return "Happiness"
            if prediction == 4:
                return "Neutral"
            if prediction == 5:
                return "Sadness"
            if prediction == 6:
                return "Surprise"
            else:
                "Prediction Number does not match an Emotion"
        if DataBase == "SFEW":
            if prediction == 0:
                return "Anger"
            if prediction == 1:
                return "Disgust"
            if prediction == 2:
                return "Fear"
            if prediction == 3:
                return "Happiness"
            if prediction == 4:
                return "Neutral"
            if prediction == 5:
                return "Sadness"
            if prediction == 6:
                return "Surprise"
            else:
                "Prediction Number does not match an Emotion"
        if DataBase == "JAFFE":
            if prediction == 0:
                return "Anger"
            if prediction == 1:
                return "Disgust"
            if prediction == 2:
                return "Fear"
            if prediction == 3:
                return "Happiness"
            if prediction == 4:
                return "Neutral"
            if prediction == 5:
                return "Sadness"
            if prediction == 6:
                return "Surprise"
            else:
                "Prediction Number does not match an Emotion"
            
    
    Datadirectory_train = r"C:\Users\Duarte Lopes\Desktop\Tese - AFER\BaseDados\FER2013\train"
    Datadirectory_test = r"C:\Users\Duarte Lopes\Desktop\Tese - AFER\BaseDados\FER2013\test"
    
    Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    #trainingSet = create_trainingSet(Datadirectory_train,Classes)
    #testingSet = create_trainingSet(Datadirectory_test,Classes)
    
    #random.shuffle (trainingSet)
    #random.shuffle (testingSet)
    
    
    def faceDetectionSet(dataset, _type_):
        if _type_ == "All Dataset":        
            nSamples = len(dataset)
            datasetFacesCut = []
            j = 0
            for i in range(nSamples):
                img_array = dataset[i][0]
                faces = faceDetection_MTCNN(img_array, 'Yes')
                if len(faces) == 0:
                    datasetFacesCut.append([dataset[i][0],dataset[i][1]])
                else :
                    new_img = faces[0]['FaceArray'][0]
                    datasetFacesCut.append([new_img,dataset[i][1]])
                    j=j+1
                print ("Faces detected and cut:",i)
            print("Total DataSet:", nSamples)
            print("Total DataSet of Detected Faces:", j)
            return datasetFacesCut
        if _type_ == "Only Samples":
            nSamples = len(dataset)
            datasetFacesCut = []
            j = 0
            for i in range(nSamples):
                img_array = dataset[i]
                faces = faceDetection_MTCNN(img_array, 'Yes')
                if len(faces) == 0:
                    datasetFacesCut.append(dataset[i])
                else :
                    new_img = faces[0]['FaceArray'][0]
                    datasetFacesCut.append(new_img)
                    j=j+1
                print ("Faces detected and cut:",i)
            print("Total DataSet:", nSamples)
            print("Total DataSet of Detected Faces:", j)
            return datasetFacesCut
    
    #trainingSetFacesCut = faceDetectionSet(trainingSet)
    #testingSetFacesCut = faceDetectionSet(testingSet)
    
    def oneChannel_threeChannel (img,h,w):
        if img.shape == (h,w): # if img is grayscale, expand
            #print ("convert 1-channel image to ", 3, " image.")
            new_img = np.zeros((h,w,3))
            for ch in range(3):
                for xx in range(h):
                    for yy in range(w):
                        new_img[xx,yy,ch] = img[xx,yy]
            img = new_img
        img = img.astype(np.uint8)
        return img
    
    def roundPrediction(labelsPredictions):
        labelsPreds = []
        for i in range(len(labelsPredictions)):
            predMax = np.max(labelsPredictions[i][:])
            predIndex = np.where(labelsPredictions[i][:] == predMax)
            pred = int(predIndex[0])
            labelsPreds.append(pred)
        labelsPreds = np.array(labelsPreds)
        return labelsPreds

# =============================================================================
#      # Store data (pickle format)
#     with open('testingSetFacesCut_FER.pickle', 'wb') as handle:
#         pickle.dump(testingSetFacesCut, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         
#     # Load data (deserialize)
#     with open('testingSetFacesCut_FER.pickle', 'rb') as handle:
#         testingSetFacesCut = pickle.load(handle)
# =============================================================================

