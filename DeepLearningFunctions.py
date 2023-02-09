# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 01:34:30 2022

@author: DuarteLopes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM,Attention,Reshape,Lambda,Add, GlobalAveragePooling2D, AveragePooling2D, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model, Model
from attention import Attention
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from STN_functions import get_localization_network,get_affine_params,get_pixel_value,affine_grid_generator,bilinear_sampler,stn


##################### DEEP LEANRING FUNCTIONS ###############################

def modelTrasnferLearning (_type_):
    if (_type_ == "MobileNetV2"):
        baseModel = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))
        
        baseInput = baseModel.layers[0].input
        baseOutput = baseModel.layers[-2].output #Penultima layer
        
        Output1 = layers.Dense(128)(baseOutput)
        Output2 = layers.Activation('relu')(Output1)
        Output3 = layers.Dense(64)(Output2)
        Output4 = layers.Activation('relu')(Output3)
        finalOutput = layers.Dense(7, activation = 'softmax')(Output4)
        
        finalModel = keras.Model(inputs = baseInput, outputs = finalOutput)
        return finalModel
    if _type_ == "ResNet152" :
        input_tensor = Input(shape=(224,224,3))
        baseResNet = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_tensor=input_tensor, classes=2)
        baseResNet = Model(baseResNet.input,baseResNet.layers[-1].output)
        baseResNet = baseResNet(input_tensor)
        
        pool = layers.AveragePooling2D(pool_size=(7, 7))(baseResNet)
        flatten = layers.Flatten(name="flatten")(pool)
        reluLayer = layers.Dense(256,activation='relu')(flatten)
        dropOut = layers.Dropout(0.5)(reluLayer)
        finalOutput = layers.Dense(7, activation = 'softmax')(dropOut)
        
        ResNet = Model(input_tensor, finalOutput, name="ResNet152")
        
        return ResNet
    
    if (_type_ == "ResNet50"):
        baseModel = tf.keras.applications.ResNet50(include_top= False,weights="imagenet",input_shape=(224, 224, 3))
        inputs = keras.Input(shape=(224,224,3))
        baseModel.trainable = False
        
        
        baseInput = baseModel.input
        x = baseModel.output
        #x = baseModel(inputs, training = False)
        
        #kernel_regularizer=l2(0.01)
        globalpoling = layers.GlobalAveragePooling2D()(x)
        dense1 = layers.Dense(512, activation='relu')(globalpoling)
        dropout1 = layers.Dropout(0.4)(dense1)
        dense2 = layers.Dense(256, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.2)(dense2)
        finalOutput = layers.Dense(7, activation = 'softmax')(dropout2)#, activity_regularizer=regularizers.L2(0.01)
        
        finalModel = keras.Model(inputs = baseInput , outputs = finalOutput)
        return finalModel
    if (_type_ == "VGG16"):
        baseModel = tf.keras.applications.VGG16(include_top= False,weights="imagenet", input_shape=(224, 224, 3))
        
        inputs = keras.Input(shape=(224,224,3))
        baseModel.trainable = False
        
        
        baseInput = baseModel.input
        x = baseModel(inputs, training = False)#
        x = baseModel.output
        
        globalpoling = layers.GlobalAveragePooling2D()(x)
        dense1 = layers.Dense(512, activation='relu')(globalpoling)
        dropout1 = layers.Dropout(0.4)(dense1)

        finalOutput = layers.Dense(7, activation = 'softmax', activity_regularizer=regularizers.L2(0.01))(dropout1)
        
        
        finalModel = keras.Model(inputs = baseInput, outputs = finalOutput)
        return finalModel
        
    if (_type_ == "VGG16_IncludeTop"):
        baseModel = tf.keras.applications.VGG16(include_top= True,weights="imagenet", input_shape=(224, 224, 3))
        
        inputs = keras.Input(shape=(224,224,3))
        baseModel.trainable = False
        
        
        baseInput = baseModel.input
        x = baseModel(inputs, training = False)#
        x = baseModel.output
        
        #globalpoling = layers.GlobalAveragePooling2D()(x)
        #dense1 = layers.Dense(512, activation='relu')(globalpoling)
        flat1 = layers.Flatten()(x)
        dropout1 = layers.Dropout(0.4)(flat1)

        finalOutput = layers.Dense(7, activation = 'softmax', activity_regularizer=regularizers.L2(0.01))(dropout1)
        
        
        finalModel = keras.Model(inputs = baseInput, outputs = finalOutput)
        return finalModel
        
def modelLandmarks():
    input_tensor = Input(shape=(224,224,3))
    
    conv1 = layers.Conv2D(16, 2, activation='relu',padding="same")(input_tensor)
    conv2 = layers.Conv2D(16, 2, activation='relu',padding="same")(conv1)
    maxpooling1 = MaxPooling2D (pool_size = (2,2))(conv2)
    conv3 = layers.Conv2D(32, 2, activation='relu',padding="same")(maxpooling1)
    conv4 = layers.Conv2D(32, 2, activation='relu',padding="same")(conv3)
    maxpooling2 = MaxPooling2D (pool_size = (2,2))(conv4)
    conv5 = layers.Conv2D(64, 2, activation='relu',padding="same")(maxpooling2)
    conv6 = layers.Conv2D(64, 2, activation='relu',padding="same")(conv5)
    maxpooling3 = MaxPooling2D (pool_size = (2,2))(conv6)
    conv7 = layers.Conv2D(128, 2, activation='relu',padding="same")(maxpooling3)
    conv8 = layers.Conv2D(128, 2, activation='relu',padding="same")(conv7)
    tranpose1 = layers.Conv2DTranspose(64,2,(2, 2))(conv8)
    concatenate1 = layers.Concatenate(axis=0)([tranpose1, conv6])
    conv9 = layers.Conv2D(64, 2, activation='relu',padding="same")(concatenate1)
    conv10 = layers.Conv2D(64, 2, activation='relu',padding="same")(conv9)
    tranpose2 = layers.Conv2DTranspose(32,2,(2, 2))(conv10)
    concatenate2 = layers.Concatenate(axis=0)([tranpose2, conv4])
    conv11 = layers.Conv2D(32, 2, activation='relu',padding="same")(concatenate2)
    conv12 = layers.Conv2D(32, 2, activation='relu',padding="same")(conv11)
    tranpose3 = layers.Conv2DTranspose(16,2,(2, 2))(conv12)
    concatenate3 = layers.Concatenate(axis=0)([tranpose3, conv2])
    conv13 = layers.Conv2D(16, 2, activation='relu',padding="same")(concatenate3)
    conv14 = layers.Conv2D(1, 2, activation='relu',padding="same")(conv13)
    

    #Classification 
    conv15 = layers.Conv2D(16, 2, activation='relu',padding="same")(input_tensor)
    conv16 = layers.Conv2D(16, 2, activation='relu',padding="same")(conv15)
    maxpooling4 = MaxPooling2D (pool_size = (2,2))(conv16)
    conv17 = layers.Conv2D(32, 2, activation='relu',padding="same")(maxpooling4)
    conv18 = layers.Conv2D(32, 2, activation='relu',padding="same")(conv17)
    maxpooling5 = MaxPooling2D (pool_size = (2,2))(conv18)
    conv19 = layers.Conv2D(64, 2, activation='relu',padding="same")(maxpooling5)
    conv20 = layers.Conv2D(64, 2, activation='relu',padding="same")(conv19)
    maxpooling6 = MaxPooling2D (pool_size = (2,2))(conv20)
    #multiplied1 = tf.keras.layers.Multiply()([reshape, maxpooling6])
    
    dense1 =  layers.Dense(512, activation='relu')(maxpooling6)
    dropout = layers.Dropout(0.5)(dense1)
    dense2 =  layers.Dense(512, activation='relu')(dropout)
    dropout2 = layers.Dropout(0.5)(dense2)
    finalOutput = layers.Dense(7, activation='softmax')(dropout2)
    
    
    finalModel = keras.Model(inputs = input_tensor, outputs = finalOutput)
    
    return finalModel

def CNNModel():
    input_tensor = Input(shape=(224,224,3))
    
    conv15 = layers.Conv2D(16, 3, activation='relu',padding="same")(input_tensor)
    conv16 = layers.Conv2D(16, 3, activation='relu',padding="same")(conv15)
    batch1 = layers.BatchNormalization()(conv16)
    maxpooling4 = MaxPooling2D (pool_size = (2,2))(batch1)
    conv17 = layers.Conv2D(32, 3, activation='relu',padding="same")(maxpooling4)
    conv18 = layers.Conv2D(32, 3, activation='relu',padding="same")(conv17)
    batch2 = layers.BatchNormalization()(conv18)
    maxpooling5 = MaxPooling2D (pool_size = (2,2))(batch2)
    conv19 = layers.Conv2D(64, 3, activation='relu',padding="same")(maxpooling5)
    conv20 = layers.Conv2D(64, 3, activation='relu',padding="same")(conv19)
    batch3 = layers.BatchNormalization()(conv20)
    maxpooling6 = MaxPooling2D (pool_size = (2,2))(batch3)
    conv21 = layers.Conv2D(128, 3, activation='relu',padding="same")(maxpooling6)
    conv22 = layers.Conv2D(128, 3, activation='relu',padding="same")(conv21)
    batch4 = layers.BatchNormalization()(conv22)
    maxpooling7 = MaxPooling2D (pool_size = (2,2))(batch4)
    conv23 = layers.Conv2D(256, 3, activation='relu',padding="same")(maxpooling7)
    conv24 = layers.Conv2D(256, 3, activation='relu',padding="same")(conv23)
    batch5 = layers.BatchNormalization()(conv24)
    maxpooling8 = MaxPooling2D (pool_size = (2,2))(batch5)
    conv25 = layers.Conv2D(512, 3, activation='relu',padding="same")(maxpooling8)
    conv26 = layers.Conv2D(512, 3, activation='relu',padding="same")(conv25)
    batch6 = layers.BatchNormalization()(conv26)
    maxpooling9 = MaxPooling2D (pool_size = (2,2))(batch6)
    
    flat1 = layers.Flatten()(maxpooling9)
    dense1 = layers.Dense(512, activation='relu',kernel_regularizer=l2(0.0001))(flat1)
    dropOut1 = layers.Dropout(0.4)(dense1)
    finalOutput = layers.Dense(7, activation='softmax')(dropOut1)
    
    finalModel = keras.Model(inputs = input_tensor, outputs = finalOutput)
    
    return finalModel

def CNNModel_baseLine(kernel):
    input_tensor = Input(shape=(224,224,3))
    
    conv17 = layers.Conv2D(32, kernel, activation='relu',padding="same")(input_tensor)
    conv18 = layers.Conv2D(32, kernel, activation='relu',padding="same")(conv17)
    batch2 = layers.BatchNormalization()(conv18)
    maxpooling5 = MaxPooling2D (pool_size = (2,2))(batch2)
    conv19 = layers.Conv2D(64, kernel, activation='relu',padding="same")(maxpooling5)
    batch3 = layers.BatchNormalization()(conv19)
    maxpooling6 = MaxPooling2D (pool_size = (2,2))(batch3)
    conv22 = layers.Conv2D(128, kernel, activation='relu',padding="same")(maxpooling6)
    batch4 = layers.BatchNormalization()(conv22)
    maxpooling4 = MaxPooling2D (pool_size = (2,2))(batch4)
    conv20 = layers.Conv2D(256, kernel, activation='relu',padding="same")(maxpooling4)
    batch7 = layers.BatchNormalization()(conv20)
    maxpooling7 = MaxPooling2D (pool_size = (2,2))(batch7)
    
    flat = layers.Flatten()(maxpooling7)
    dense1 = layers.Dense(1024, activation='relu')(flat)
    dropOut3 = layers.Dropout(0.5)(dense1)
    finalOutput = layers.Dense(7, activation='softmax')(dropOut3)
    
    finalModel = keras.Model(inputs = input_tensor, outputs = finalOutput)
    
    return finalModel

def simple_CNN(input_shape, num_classes):

    model = Sequential()
    model.add(layers.Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(.5))

    model.add(layers.Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(.5))

    model.add(layers.Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(.5))

    model.add(layers.Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Dropout(.5))

    model.add(layers.Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(
        filters=num_classes, kernel_size=(3, 3), padding='same'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Activation('softmax', name='predictions'))
    return model

def stupidAttempt():    
    input_tensor = Input(shape=(224,224,3))
    
    conv17 = layers.Conv2D(32, 3, activation='relu',padding="same")(input_tensor)
    batch2 = layers.BatchNormalization()(conv17)
    maxpooling5 = MaxPooling2D (pool_size = (2,2))(batch2)
    
    conv18 = layers.Conv2D(128, 3, activation='relu',padding="same")(maxpooling5)
    batch3 = layers.BatchNormalization()(conv18)
    maxpooling9 = MaxPooling2D (pool_size = (2,2))(batch3)
    
    
    flat1 = layers.Flatten()(maxpooling9)
# =============================================================================
#     dense1 = layers.Dense(512, activation='relu',kernel_regularizer=l2(0.0001))(flat1)
#     dropOut1 = layers.Dropout(0.5)(dense1)
# =============================================================================
    
    finalOutput = layers.Dense(7, activation='softmax')(flat1)
    
    finalModel = keras.Model(inputs = input_tensor, outputs = finalOutput)
    return finalModel
def NewVgg():
    input_tensor = Input(shape=(224,224,3))
    
    conv17 = layers.Conv2D(32, 3, activation='relu',padding="same")(input_tensor)
    conv18 = layers.Conv2D(32, 3, activation='relu',padding="same")(conv17)
    batch2 = layers.BatchNormalization()(conv18)
    maxpooling5 = MaxPooling2D (pool_size = (2,2))(batch2)
    
    conv15 = layers.Conv2D(64, 3, activation='relu',padding="same")(maxpooling5)
    conv16 = layers.Conv2D(64, 3, activation='relu',padding="same")(conv15)
    batch1 = layers.BatchNormalization()(conv16)
    maxpooling4 = MaxPooling2D (pool_size = (2,2))(batch1)
    
    conv17 = layers.Conv2D(128, 3, activation='relu',padding="same")(maxpooling4)
    conv18 = layers.Conv2D(128, 3, activation='relu',padding="same")(conv17)
    batch3 = layers.BatchNormalization()(conv18)
    maxpooling9 = MaxPooling2D (pool_size = (2,2))(batch3)
    
    #conv195 = layers.Conv2D(256, 3, activation='relu',padding="same")(maxpooling9)
    #conv201 = layers.Conv2D(256, 3, activation='relu',padding="same")(conv195)
    conv192 = layers.Conv2D(256, 3, activation='relu',padding="same")(maxpooling9)
    conv20 = layers.Conv2D(256, 3, activation='relu',padding="same")(conv192)

    batch4 = layers.BatchNormalization()(conv20)
    maxpooling6 = MaxPooling2D (pool_size = (2,2))(batch4)
    
    conv2240 = layers.Conv2D(512, 3, activation='relu',padding="same")(maxpooling6)
    conv2245 = layers.Conv2D(512, 3, activation='relu',padding="same")(conv2240)
    conv222 = layers.Conv2D(512, 3, activation='relu',padding="same")(conv2245)
    conv22 = layers.Conv2D(512, 3, activation='relu',padding="same")(conv222)
    batch4 = layers.BatchNormalization()(conv22)
    maxpooling7 = MaxPooling2D (pool_size = (2,2))(batch4)
    
    
    flat1 = layers.Flatten()(maxpooling7)
    dense1 = layers.Dense(2048, activation='relu',kernel_regularizer=l2(0.0001))(flat1)
    dropOut1 = layers.Dropout(0.3)(dense1)
    #dense1 = layers.Dense(1024, activation='relu',kernel_regularizer=l2(0.0001))(flat1)
    #dropOut1 = layers.Dropout(0.4)(dense1)
    
    #dense2 = layers.Dense(512, activation='relu',kernel_regularizer=l2(0.0001))(dropOut1)
    #dropOut2 = layers.Dropout(0.4)(dense2)
    
# =============================================================================
#     dense3 = layers.Dense(256, activation='relu',kernel_regularizer=l2(0.0001))(dropOut2)
#     dropOut3 = layers.Dropout(0.4)(dense3)
# =============================================================================
    finalOutput = layers.Dense(7, activation='softmax')(dropOut1)
    
    finalModel = keras.Model(inputs = input_tensor, outputs = finalOutput)
    
    return finalModel
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # module 1
    residual = layers.Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = layers.Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = layers.Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x =layers.SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = layers.Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = layers.Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model
    
    
    
def attentionModel(IMG_SIZE, use_stn=True):
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    if use_stn:
        x = stn(inputs)
    else:
        x = inputs
    
    conv1 = layers.Conv2D(10, (5, 5), activation="relu", kernel_initializer="he_normal")(x)
    maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(20, (5, 5), activation="relu", kernel_initializer="he_normal")(maxpool1)
    spatialDropout1 = layers.SpatialDropout2D(0.5)(conv2)
    maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))(spatialDropout1)
    x = tf.reshape(maxpool2, (-1, 320))
    dense1 = layers.Dense(50, activation="relu", kernel_initializer="he_normal")(x)
    dropout1 = layers.Dropout(0.5)(dense1)
    outputs = layers.Dense(10, activation="softmax")(dropout1)
    
    finalModel = keras.Model(inputs = inputs, outputs = outputs)
    return finalModel

    
    
# =============================================================================
# def attentionModel():
#     input_dim = 224
#     units = 64
#     input_tensor = Input(shape=(shape))
#     x = LSTM(units, input_shape=(None, input_dim), return_sequences=True)(input_tensor)
#     x = Attention(units=32)(x)
#     x = Dense(1)(x)
#     model = Model(input_tensor, x)
#     return model
# =============================================================================
from sklearn.neural_network import MLPClassifier

#def MLP()