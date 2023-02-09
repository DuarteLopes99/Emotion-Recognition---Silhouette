# -*- coding: utf-8 -*-
"""
Created on Sun May 29 20:33:50 2022

@author: DuarteLopes
"""

########################### TRAINING  ################################
from DeepLearningFunctions import stupidAttempt,NewVgg,CNNModel_baseLine,mini_XCEPTION,simple_CNN,attentionModel,modelLandmarks,modelTrasnferLearning,CNNModel
import tensorflow as tf 
import tensorflow_addons as tfa
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import visualkeras
from stn import spatial_transformer_network as transformer
#STN
from STN_functions import get_localization_network,get_affine_params,get_pixel_value,affine_grid_generator,bilinear_sampler,stn
####################### SFEW Database ###################################
# =============================================================================
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# =============================================================================
with tf.device('/gpu:0'):
# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/Originais/CK+_TrainingSet', 'rb') as handle:
#         DataSet = pickle.load(handle)
#     X, Y = dataSpliter(DataSet)    
# =============================================================================





# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/OnlyLandmark/CK+_OnlyLandmark_TrainSet.pickle', 'rb') as handle:
#         DataSet = pickle.load(handle)
#     X, Y = DataSet
# 
# =============================================================================
    with open('Train_Test_Sets_afterFaceDetect/Masked/CK+_Mask_TrainSet.pickle', 'rb') as handle:
        DataSet = pickle.load(handle)
    X, Y = DataSet
     
# =============================================================================
#     with open('Train_Test_Sets_afterFaceDetect/Masked/SFEW_Mask_TrainSet.pickle', 'rb') as handle:
#         DataSet = pickle.load(handle)
#     X, Y = DataSet
# =============================================================================
    
    X = np.array(X)
    Y = np.array(Y)
    
    ######################## Model Fit #######################################
    cv_outer = KFold(n_splits = 3, shuffle=True)
    count = 0
    NameFileExcel = "EXCEL_Landmarks//Mask_CK+_TransVGG16_x2.xlsx"
    models = list()
    resultsList = list()
    
    model = modelTrasnferLearning ("VGG16")
    #model = CNNModel_baseLine(3)
    #model = NewVgg()
    
    model.summary()
    with pd.ExcelWriter(NameFileExcel) as writer:
        for train_ix, test_ix in cv_outer.split(X):
        ######################## Transfer Leanring MODEL #########################
        
            
            model = modelTrasnferLearning ("VGG16")
            #model = CNNModel_baseLine(3)
            #model = NewVgg()
            
            LEARNING_RATE = 0.0001
            DECAY = 1e-6
            W_DECAY = 0.0001
            #OPTIMIZER = tfa.optimizers.AdamW(learning_rate = LEARNING_RATE, 
                                             #weight_decay = W_DECAY)
            OPTIMIZER = optimizers.Adam(learning_rate = LEARNING_RATE, decay = DECAY)
            BATCH_SIZE = 32
            N_EPOCHS = 100
            LOSS = "sparse_categorical_crossentropy"
            #LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            
            #Total Loss = Sum
        # =============================================================================
        #     LOSS = {'out_1': 'binary_crossentropy',
        #                        'out_2': 'mse'}
        # =============================================================================
        # =============================================================================
        #     def custom_loss(y_true, y_pred):
        #         return K.mean(y_true - y_pred)**2
        # =============================================================================
            #LOSS = 'categorical_crossentropy'
            #LOSS = tf.keras.losses.CosineSimilarity()
            #LOSS = 'mae'
            #model.compile(loss = LOSS, optimizer = OPTIMIZER, 
                          #metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
            model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = ['accuracy'])
            
            #Data
            X_train, X_test = X[train_ix], X[test_ix]
            Y_train, Y_test = Y[train_ix], Y[test_ix]
        
    
            trainData = dataAugmentation(X_train, Y_train, BATCH_SIZE, "Train")
            valData = dataAugmentation(X_test,Y_test, BATCH_SIZE, "Val")
    # =============================================================================
    #         earlyStopping = EarlyStopping(monitor='val_loss',
    #                                       min_delta=0,
    #                                       patience=3,
    #                                       restore_best_weights=True)
    #         reduceLR = ReduceLROnPlateau(monitor='val_loss',
    #                                       factor=0.2,
    #                                       patience=3,
    #                                       restore_best_weights=True,
    #                                       min_delta=0.0001)    
    # =============================================================================
    
            #callBacksList = [earlyStopping,reduceLR]
            trainHistory = model.fit(x = X_train, 
                                     y = Y_train,
                                     #trainData,
                                     #validation_data = valData,
                                     epochs = N_EPOCHS,
                                     validation_data = (X_test, Y_test),
                                     #validation_data = (X_train, Y_train),
                                     #callbacks = callBacksList,
                                     verbose = 2)
            #Saving Metrics
            #Train
            accTrain = trainHistory.history['accuracy']
            lossTrain = trainHistory.history['loss']
            accTrainDF = pd.DataFrame (accTrain)
            lossTrainDF = pd.DataFrame (lossTrain)
            
            #Val
            accVal = trainHistory.history['val_accuracy']
            lossVal = trainHistory.history['val_loss']
            accValDF = pd.DataFrame (accVal)
            lossValDF = pd.DataFrame (lossVal)
    
    
    
            #Sending2Excel
            sheetNameAcc = 'Acc'+ str(count)
            sheetNameLoss = 'Loss'+ str(count)
            accTrainDF.to_excel(writer, sheet_name= sheetNameAcc)
            lossTrainDF.to_excel(writer, sheet_name= sheetNameLoss)
            sheetNameValAcc = 'Val_Acc'+ str(count)
            sheetNameValLoss = 'Val_Loss'+ str(count)
            accValDF.to_excel(writer, sheet_name= sheetNameValAcc)
            lossValDF.to_excel(writer, sheet_name= sheetNameValLoss)
               
    # =============================================================================
    #             #Genarator
    #             results = model.evaluate(valData)
    #             #results = model.evaluate(valX, valY)
    #             results = dict(zip(model.metrics_names,results))
    #             acc = results['accuracy']
    #             loss = results['loss']
    #             print('>ValAcc=%.3f, ValLoss=%.3f' % (acc, loss))
    # =============================================================================
            
            #Normal
            preds = model.predict(X_test)
            preds = roundPrediction(preds)
            corrected = np.sum(Y_test == preds)
            acc = (corrected/np.shape(Y_test)[0])
            print('>HomeMade=%.3f, ValAcc=%.3f, ValLoss=%.3f' % (acc,accVal[-1], lossVal[-1]))
            
            #DataGenerator
    
            
            models.append(model)
            #resultsList.append(results)
# =============================================================================
#             nameModel = "Mask_SFEW_myVGG_Model"+str(count)
#             model = models[count]
#             model.save('C://Users//Duarte Lopes//Desktop//Tese - AFER//Código///Models//Landmarks//'+nameModel+'.h5')
#             #model.save_weights('C://Users//Duarte Lopes//Desktop//Tese - AFER//Código///Models//Landmarks//'+nameModel+'_weights')
#             print('Model Saved')
#             count = count +1 
# =============================================================================
# =============================================================================
# nameModel = "Masked_JAFFE_myCNN_Model_"
# model = models[2]
# model.save('C://Users//Duarte Lopes//Desktop//Tese - AFER//Código///Models//Landmarks//'+nameModel+'.h5')
# model.save_weights('C://Users//Duarte Lopes//Desktop//Tese - AFER//Código///Models//Landmarks//'+nameModel+'_weights')
# print('Model Saved')
# =============================================================================
    
    #preds = model.predict(valData)
    #preds = roundPrediction(preds)
    # =============================================================================
    #     
    #     print('Classification Report:\n', classification_report(t, labelsPreds))
    #     cm = confusion_matrix(testY, labelsPreds)
    #     cmPercentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     cmHeatmap = sns.heatmap(cm, annot=True, cmap='Blues')
    #     plt.figure(1)
    # =============================================================================
# =============================================================================
#     from PIL import ImageFont
#     from collections import defaultdict
#     from tensorflow.python.keras.layers import InputLayer, BatchNormalization,Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
#     
#     color_map = defaultdict(dict)
#     color_map[InputLayer]['fill'] = 'purple'
#     color_map[Conv2D]['fill'] = 'blue'
#     #color_map[BatchNormalization]['fill'] = ''
#     color_map[Dropout]['fill'] = 'red'
#     color_map[MaxPooling2D]['fill'] = 'green'
#     color_map[Dense]['fill'] = 'orange'
#     color_map[Flatten]['fill'] = 'black'
#     
#     font_type_1 = ImageFont.truetype("bahnschrift.ttf",25)
#     visualkeras.layered_view(model,legend=True, font=font_type_1, color_map=color_map,to_file='output.png')).show()
# =============================================================================
