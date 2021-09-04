import numpy as np 
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame

pygame.mixer.init()
model = load_model('trained_keras_model')

with open('mean_image_rgb.pickle', 'rb') as f:
    mean = pickle.load(f, encoding='latin1')

labels = pd.read_csv('labels.csv')
probability_minimum = 0.5
threshold = 0.2
path_to_cfg = 'cfg/my_yolov4_test.cfg'
path_to_weights = 'my_yolov4_train_8000.weights'

network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layers_all = network.getLayerNames()
layers_names_output = [layers_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


webcam = cv2.VideoCapture(0)
height, width = None, None
plt.rcParams['figure.figsize'] = (3, 3)
frame_no, processing_time = 0, 0

while True:
    ret, frame = webcam.read()
    if not ret: break
    if height is None and width is None:
        height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)

    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    frame_no += 1
    processing_time += end - start

    print('Czas przetwarzania klatki nr {0}: {1:.5f} s'.format(frame_no, end - start))

    bounding_boxes = []
    confidences = []
    class_numbers = []
    
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([width, height, width, height])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
            
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            c_ts = frame[y_min:y_min+int(box_height), x_min:x_min+int(box_width), :]
            
            if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                pass
            else:
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False) 
                blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)

                scores = model.predict(blob_ts)
                prediction = np.argmax(scores)
                print('{0} ({1:.2%})'.format(labels['SignName'][prediction], confidences[i]))

                pygame.mixer.music.load('sounds/{num}.mp3'.format(num = prediction))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() == True:
                    continue

webcam.release()