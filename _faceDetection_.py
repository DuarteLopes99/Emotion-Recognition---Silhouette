# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:48:08 2021

@author: DuarteLopes
"""
import sys, os
sys.path.append('C://Users//Duarte Lopes//Desktop//Tese - AFER//Código')

import tensorflow as tf

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import cv2
from retinaface import RetinaFace
# MTCNN - Multi-Task Cascaded Convolutional Neural Network
from mtcnn_cv2 import MTCNN #Modelo de detecao cara Joint Faces ...
import pickle
#with tf.device('/gpu:0'):
print("Num of GPUs:",len(tf.config.experimental.list_physical_devices('GPU')))
# =============================================================================
# with open('Train_Test_Sets_afterFaceDetect/Originais/SFEW_TrainingSet', 'rb') as handle:
#     TestSetCK = pickle.load(handle)
#     img = TestSetCK[0][72]
#     img1 = TestSetCK[0][72]
# =============================================================================
#filename = 'ImagensExemplo/sharon_stone1.jpg'
#filename = 'ImagensExemplo/FEUPcaffe.jpg' 
#filename = 'ImagensExemplo/seleção.jfif'
#filename1 = 'ImagensExemplo/Seleçãofestejos.jpg'


# =============================================================================
# pixels = plt.imread(filename)   
# plt.imshow(pixels)
# 
# #Utilizar o detetor de Faces MTCNN
# detector = MTCNN()
# 
# #Detetar posição da face e o respetivo score
# results = detector.detect_faces(pixels)
# score = results[0]['confidence']
# #print(int(score*100))
# 
# # Extrair as posições da localização da face
# x1, y1, width, height = results[0]['box']
# x2, y2 = x1 + width, y1 + height
# # Array só da Face
# face = pixels[y1:y2, x1:x2]
# 
# #Resize - Para ser o mesmo tamanho da entrada do modelos 224 x 224
# image = Image.fromarray(face)
# image = image.resize((224, 224))
# face = np.asarray(image)
# plt.imshow(face)
# 
# #Detetar pontos especificos como olhos e boca 
# #Depende se queremos aplicar os pontos na face ja recortada ou na original
# resultsFace = detector.detect_faces(face)
# fiducialPoints = resultsFace[0]['keypoints']
# 
# #Desenhar pontos na imagem em si 
# ax = plt.gca()
# for key, value in fiducialPoints.items():
#     dot = Circle(value, radius = 2, color='red')
#     ax.add_patch(dot)
# 
# plt.show()
# =============================================================================


#Formato Função MTCNN
def faceDetection_MTCNN(array, _newImage_, required_size=(224, 224)):
    pixels = array
    
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    
    #Iniciar variaveis
    allFaces = []
    allBoxesOnePic = []
    faceScore = {"FaceArray":[], "Score":[]}
    boxPositions = {"Box":[], "Score":[]}
    
    if _newImage_ == 'Yes':
        if len(results) == 1:             
            # Extrair as posições da localização da face
            x, y, width, height = results[0]['box']
            x2, y2 = x + width, y + height
            
            #Score of the detection
            score = results[0]['confidence']*100
            score = "{:.2f}".format(score)
        
            # Array só da Face
            justFace = pixels[y:y2, x:x2]
            image = Image.fromarray(justFace)
            image = image.resize((224, 224))
            face = np.asarray(image)
            
            #Creating a dict with score and face
            faceScore["FaceArray"].append(face)
            faceScore["Score"].append(score)
                
            allFaces.append(faceScore)
                
            #Recriando o dict evito que as variaveis fiquem sempre na mesma linha do array
            faceScore = {"FaceArray":[], "Score":[]}
            return allFaces
        else:
            for i in range(0, len(results)):             
                # Extrair as posições da localização da face
                x, y, width, height = results[i]['box']
                x2, y2 = x + width, y + height
                
                #Score of the detection
                score = results[i]['confidence']*100
                score = "{:.2f}".format(score)
            
                # Array só da Face
                justFace = pixels[y:y2, x:x2]
                
                image = Image.fromarray(justFace)
                image = image.resize((224, 224))
                face = np.asarray(image)
                
                #Creating a dict with score and face
                faceScore["FaceArray"].append(face)
                faceScore["Score"].append(score)
                
                allFaces.append(faceScore)
                
                #Recriando o dict evito que as variaveis fiquem sempre na mesma linha do array
                faceScore = {"FaceArray":[], "Score":[]}
            return allFaces
        
    if _newImage_ == 'No':
        #Desenhar boxes nas faces 
        for i in range(0, len(results)): 
            # Extrair as posições da localização da face
            x, y, width, height = results[i]['box']
            x2, y2 = x + width, y + height
            faces_boxes = cv2.rectangle(pixels, (x, y), (x2, y2), (8,255,8), 2)
                       
            #Score de cada box
            score = results[i]['confidence']*100
            score = "{:.2f}".format(score)
            
            cv2.putText(pixels,score,(x, y), cv2.FONT_HERSHEY_PLAIN, 1, (8,255,8), 1)
            
            boxPositions["Box"].append([x,x2,y,y2])  
            boxPositions["Score"].append(score)
            
            allBoxesOnePic.append(boxPositions)
            
            #Recriando o dict evito que as variaveis fiquem sempre na mesma linha do array
            boxPositions = {"Box":[], "Score":[]}
        return faces_boxes, allBoxesOnePic
# =============================================================================
# 
# def retinaFace(filename, _newImage_, required_size=(224, 224)):
#     resp = RetinaFace.detect_faces(filename)
#     
#     #Iniciar variaveis
#     allBoxesOnePic = []
#     boxPositions = {"Box":[], "Score":[]}
#     if _newImage_ == 'No':
#         #Desenhar boxes nas faces 
#         for i in range(0, len(resp)): 
#             # Extrair as posições da localização da face
#             face = 'face_'+str(i+1)
#             valuesFace = resp[face]
#             boxes = valuesFace['facial_area']
#             a,b,c,d = boxes
#             #x2, y2 = x + width, y + height
#             faces_boxes = cv2.rectangle(pixels, (c, d), (a, b), (255,0,0), 3)
#                        
#             #Score de cada box
#             #score = resp[i]["score"]*100
#             #score = "{:.2f}".format(score)
#             
#             #cv2.putText(pixels,score,(x, y), cv2.FONT_HERSHEY_PLAIN, 2, (8,255,8), 2)
#             
#             boxPositions["Box"].append([a,b,c,d])  
#             #boxPositions["Score"].append(score)
#             
#             allBoxesOnePic.append(boxPositions)
#             
#             #Recriando o dict evito que as variaveis fiquem sempre na mesma linha do array
#             boxPositions = {"Box":[], "Score":[]}
#         return faces_boxes, allBoxesOnePic
# 
# =============================================================================
def faceDetection_ViolaJones(array,type_haarcascade,_newImage_, required_size=(224, 224)):
    pixels = array
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + type_haarcascade)     
    faces = detector.detectMultiScale(pixels)
       
    allFaces = []
    if _newImage_ == 'Yes':
        if len(faces) == 1:
            for value in faces:                
                # Extrair as posições da localização da face
                x, y, width, height = value
                x2, y2 = x + width, y + height
            
                # Array só da Face
                justFace = pixels[y:y2, x:x2]
                image = Image.fromarray(justFace)
                image = image.resize((224, 224))
                face = np.asarray(image)
                return face
            
        else:
            for value in faces:                
                # Extrair as posições da localização da face
                x, y, width, height = value
                x2, y2 = x + width, y + height
            
                # Array só da Face
                justFace = pixels[y:y2, x:x2]
                image = Image.fromarray(justFace)
                image = image.resize((224, 224))
                face = np.asarray(image)
                
                allFaces.append(face)
            return allFaces
        
    if _newImage_ == 'No':
        #Desenhar boxes nas faces 
        numBox = 0
        for value in faces:
            # Extrair as posições da localização da face
            x, y, width, height = value
            x2, y2 = x + width, y + height
            faces_boxes = cv2.rectangle(pixels, (x, y), (x2, y2), (77,77,255), 3)
            numBox=numBox+1
        
        return faces_boxes,numBox
      
                        # TESTING VIOLA JONES FUNCTION #
# =============================================================================
# faces_VV = faceDetection_ViolaJones(filename,'haarcascade_frontalface_default.xml', 'Yes')
# _face_ = faceDetection_ViolaJones(filename,'haarcascade_frontalface_default.xml', 'No')
# 
# plt.imshow(_face_)
# for i in range(0, len(faces_VV)):   
#     plt.figure(i)
#     plt.imshow(faces_VV[i])
# =============================================================================

                        # TESTING MCTNN FUNCTION #
# =============================================================================
# pixels = plt.imread(filename)
# faces = faceDetection_MTCNN(pixels, 'Yes')
# new_img = faces[0]['FaceArray'][0]
# plt.figure(2)
# plt.tight_layout()
# plt.imshow(new_img)
# plt.savefig('Seleção1.png', dpi=200)
# =============================================================================
# =============================================================================
# pixels1 = plt.imread(filename1)
# pixels = plt.imread(filename1)
# 
# allBoxesOnePic1, boxesPositions1 = faceDetection_MTCNN(pixels1, 'No')
# 
# Numface_,numBox = faceDetection_ViolaJones(pixels ,'haarcascade_frontalface_default.xml', 'No')
# retinaFace, retinaBoxes = retinaFace("imagem.jpg", 'No')
# 
# plt.figure(1)
# plt.subplot(1,3,1)
# title = "Num. Faces Detected:" + str(numBox)
# plt.title(title,fontsize = '10')
# plt.imshow(retinaFace)
# plt.axis('off')
# 
# plt.subplot(1,3,2)
# title = "Num. Faces Detected:" + str(len(boxesPositions1))
# plt.title(title,fontsize = '10')
# plt.imshow(allBoxesOnePic1)
# plt.axis('off')
# 
# plt.subplot(1,3,3)
# title = "Num. Faces Detected:" + str(len(retinaBoxes))
# plt.title(title,fontsize = '10')
# plt.imshow(retinaFace)
# plt.axis('off')
# # =============================================================================
# # for i in range(0, len(faces)):
# #     score = str(faces[i]['Score'])
# #     title = 'Face ' + str(i) + '; Score:' + score 
# #     plt.figure()
# #     plt.title(title)
# #     plt.imshow(faces[i]['FaceArray'][0]) #Necessario o 0 para excolher o array em si !!
# # =============================================================================
# plt.tight_layout()
# plt.axis('off')
# plt.savefig('SeleçãoBothMTCNNVSViolaVSRetina.png', dpi=200)
# =============================================================================



########## FaceCutting ################################################
file = 'ImagensExemplo/CK_ex.png'
pixels = cv2.imread(file)
pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
#pixels = testX[3]
faces1, boxesPositions1= faceDetection_MTCNN(pixels, 'No')

#plt.figure(1)
#plt.tight_layout()
#plt.axis('off')
#plt.imshow(faces1)
#plt.savefig('ExMTCNN_Cutt.jpg', dpi=200)

########## FaceDetecting ################################################
#pixels1 = testX[3]
pixels1 = cv2.imread(file)
pixels1 = cv2.cvtColor(pixels1, cv2.COLOR_BGR2RGB)

faces = faceDetection_MTCNN(pixels1, 'Yes')
new_img = faces[0]['FaceArray'][0]
#plt.figure(2)
#plt.tight_layout()
#plt.axis('off')
#.imshow(new_img)
#plt.savefig('ExMTCNN_Detect.jpg', dpi=200)

# =============================================================================
# plt.figure(3)
# plt.subplot(1,2,1)
# title = "Face Detected"
# plt.title(title,fontsize = '10')
# plt.imshow(faces1)
# plt.axis('off')
# plt.subplot(1,2,2)
# title = "Sample After Face Cut"
# plt.title(title,fontsize = '10')
# plt.imshow(new_img)
# plt.axis('off')
# plt.savefig('detect_Cut_SFEW.jpg', dpi=200)
# =============================================================================
plt.figure(3)
title = "Face Detected"
plt.imshow(faces1)
plt.axis('off')
plt.savefig('detect_CK.jpg', dpi=200)

plt.figure(4)
title = "Sample After Face Cut"
plt.imshow(new_img)
plt.axis('off')
plt.savefig('Cut_CK.jpg', dpi=200)
