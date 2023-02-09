# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 13:00:56 2022

@author: DuarteLopes
"""
from DeepLearningFunctions import simple_CNN,attentionModel,modelLandmarks,modelTrasnferLearning,CNNModel
from ViT_Functions import create_vit_classifier
import tensorflow as tf 
from tensorflow.keras import optimizers
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import seaborn as sns
from sklearn.model_selection import KFold
from skimage import feature
import pickle
from _dataPreProcessing_ import roundPrediction, dataSpliter, faceDetectionSet, oneChannel_threeChannel, dataDivider, datDivider_mutipleList,dataAugmentation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

import visualkeras
from stn import spatial_transformer_network as transformer
#STN
from STN_functions import get_localization_network,get_affine_params,get_pixel_value,affine_grid_generator,bilinear_sampler,stn
###################### TESTING FILE ##########################################
with tf.device('/gpu:0'):
    
# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/Originais/CK+_TestingSet', 'rb') as handle:
#         TestSetCK = pickle.load(handle)
#     testX,testY = dataSpliter(TestSetCK)
# =============================================================================
    
# =============================================================================
#    #JAFFE 
#     with open('Train_Test_Sets_afterFaceDetect/Originais/JAFFE_TestingSet', 'rb') as handle:
#         TestSet= pickle.load(handle)
#     
#     testX, testY = TestSet 
# =============================================================================
# =============================================================================
#     #SFEW
#     with open('Train_Test_Sets_afterFaceDetect/Originais/SFEW_TestingSet', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet
# =============================================================================
 



# =============================================================================
# ################################## MASK ##############################################
#     with open('Train_Test_Sets_afterFaceDetect/Masked/CK+_Mask_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================

# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/Masked/JAFFE_Mask_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================
# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/Masked/SFEW_Mask_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================

# =============================================================================
# ################################## ONLY LANDMARK ##############################################
#     with open('Train_Test_Sets_afterFaceDetect/OnlyLandmark/CK+_OnlyLandmark_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================

# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/OnlyLandmark/JAFFE_OnlyLandmark_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================
# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/OnlyLandmark/SFEW_OnlyLandmark_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================

# =============================================================================
# ##################################### SILHOUETTE #############################
#     with open('Train_Test_Sets_afterFaceDetect/Sillhoutte/CK+_Silhoutte_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================
# =============================================================================
# 
#     with open('Train_Test_Sets_afterFaceDetect/Sillhoutte/JAFFE_Silhoutte_TestSet.pickle', 'rb') as handle:
#         TestSet = pickle.load(handle)
#     testX, testY = TestSet 
# =============================================================================
    with open('Train_Test_Sets_afterFaceDetect/Sillhoutte/SFEW_Silhoutte_TestSet.pickle', 'rb') as handle:
        TestSet = pickle.load(handle)
    testX, testY = TestSet 
 
    
 
    testX = np.array(testX)
    testY = np.array(testY)

    def labels (_type_):
        if _type_== "CK+":
            labels = ["Anger", "Comtempt", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
            return labels 
        if _type_== "SFEW/JAFFE/FER":
            labels = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
            return labels 
        
    def testingModel (testX, testY, weightsPath,checkpoint_filepath, model, nameModel, _type_, weights = False ):
        if _type_ == "CNN":
        ############## CNN ###################################################
            LEARNING_RATE = 0.0001
            DECAY = 1e-6
            W_DECAY = 0.0001
            OPTIMIZER = optimizers.Adam(learning_rate = LEARNING_RATE, decay = DECAY)
            BATCH_SIZE = 32
            LOSS = "sparse_categorical_crossentropy"
            
            if weights == True:
                model = model
                model.load_weights(weightsPath).expect_partial()
            else:
                name = nameModel
                model = load_model(name)
            
            model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = ["accuracy"])
            model.summary()
            
# =============================================================================
#             testData = dataAugmentation(testX, testY, BATCH_SIZE, "Test")
#             labelsPredictions = model.predict(testData) 
# =============================================================================
            
            labelsPredictions = model.predict(testX) 
            labelsPreds = roundPrediction(labelsPredictions)
            
        if _type_ == "ViT":
        ############## ViT ###################################################
            learning_rate = 0.001
            weight_decay = 0.0001
            image_size = 72  # We'll resize input images to this size
            #image_size = 224  # We'll resize input images to this size
            patch_size = 6  # Size of the patches to be extract from the input images
            num_patches = (image_size // patch_size) ** 2
            projection_dim = 64
            num_heads = 4
            transformer_units = [
                projection_dim * 2,
                projection_dim,
            ]  # Size of the transformer layers
            
            transformer_layers = 8
            mlp_head_units = [2048, 1024] 
        
            patch_size = 6
            input_shape = (224,224,3)
            
            ######################### Model ###########################################
            model = create_vit_classifier(input_shape,mlp_head_units,num_patches,projection_dim,transformer_layers,num_heads,transformer_units)
            model.load_weights(weightsPath).expect_partial()
            #model.load_weights(checkpoint_filepath)
            model.summary()
            #Compile
            
            LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, 
                                             weight_decay=weight_decay)
            
            
            model.compile(optimizer=optimizer,
                          loss = LOSS,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                         #tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
                         ])
            #_, accuracy, top_5_accuracy = model.evaluate(testX, testY)
            labelsPredictions = model.predict(testX)
            labelsPreds = roundPrediction(labelsPredictions)
        
        if _type_ == "LBP/HOG":
            #path = 'C://Users//Duarte Lopes//Desktop//Tese - AFER//Código///Models//'+nameModel
            model = pickle.load(open(nameModel,'rb'))
            labelsPredictions = model.predict(testX)
            labelsPreds = labelsPredictions
        ########################### Rounding Preds Values ########################
        
        print('Classification Report:\n', classification_report(testY, labelsPreds))
        cm = confusion_matrix(testY, labelsPreds)
        cmPercentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.rcParams.update({'font.size': 7})
        fig=plt.figure(1)
        ax = fig.add_subplot()
        cmHeatmap = sns.heatmap(cm, annot=True, cmap='Blues')
        #labels_ = labels ("CK+")
        labels_ = labels ("SFEW/JAFFE/FER")
        
        ax.set_xlabel("Predicted Label");ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(labels_);ax.yaxis.set_ticklabels(labels_)
        
        fig.tight_layout()
        #fig.savefig('TransVGG16_JAFFE_YES_DA_', dpi=200)
        
        fig=plt.figure(2)
        ax = fig.add_subplot()
        percentageMap = sns.heatmap(cmPercentage, annot=True, cmap='Blues')
        
        plt.rcParams.update({'font.size': 7})
        ax.set_xlabel("Predicted Label");ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(labels_);ax.yaxis.set_ticklabels(labels_)
        fig.tight_layout()
        fig.savefig('Landmarks_Results_CM//myCNN_SFEW_Silhouette_', dpi=200)
        
        report = model.evaluate(testX, testY)
        print("Accuracy Score:",accuracy_score(testY, labelsPreds))
        print("Accuracy Score:",report[1])
        print("Loss Score:",report[0])
        accTest = accuracy_score(testY, labelsPreds)
# =============================================================================
#         testData = dataAugmentation(testX, testY, BATCH_SIZE, "Val")
#         report = model.evaluate_generator(testData)
#         print("Accuracy Score:",accuracy_score(testY, labelsPreds))
#         print("Accuracy Score:",report[1])
#         print("Loss Score:",report[0])
#         accTest = accuracy_score(testY, labelsPreds)
# =============================================================================
        
        return model,accTest, labelsPreds, cmHeatmap
    
    #ViT
# =============================================================================
#     model = modelTrasnferLearning ("ResNet50")
#     nameModel = "C://Users//Duarte Lopes//Desktop//Tese - AFER//Código//Models//ResNet50//Original_CK+_ResNet50.h5"
#     weightsPath = 'C://Users//Duarte Lopes//Desktop//Tese - AFER//Código//CK+_ViT_Weights//CK+_ViT_All_NoAugm_weights'
#     checkpoint_filepath = "./tmp/checkpoint"
#     model,accTestViT, labelsPredsViT, cmHeatmapViT = testingModel (testX, testY, weightsPath,checkpoint_filepath, model, nameModel, 'ViT')
#     
# =============================================================================
    #CNN
    model = modelTrasnferLearning ("ResNet50")
    checkpoint_filepath = "./tmp/checkpoint"
    nameModel = "C://Users//Duarte Lopes//Desktop//Tese - AFER//Código//Models//Landmarks//Silhouette//Silhouette_SFEW_myCNN_Model.h5"
    weightsPath = 'C://Users//Duarte Lopes//Desktop//Tese - AFER//Código//Models//myCNN'
    model,accTest, labelsPreds, cmHeatmap = testingModel (testX, testY, weightsPath,checkpoint_filepath, model, nameModel, 'CNN')
    
    #LBP
# =============================================================================
#     nameModel = "C://Users//Duarte Lopes//Desktop//Tese - AFER//Código//Models//LBP//LBP_SFEW_SVM_Kfolds_10.sav"
#     model=modelTrasnferLearning ("ResNet50") #NAO INTERESSA
#     model,accTestJAFFE, labelsPredsJAFFE, cmHeatmapJAFFE = testingModel (testX, testY, weightsPath,checkpoint_filepath, model, nameModel, 'LBP/HOG',weights=True)
#     
# =============================================================================
