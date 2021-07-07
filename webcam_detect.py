import numpy as np 
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('trained_keras_model')

with open('dataset-preprocessed/mean_image_rgb.pickle', 'rb') as f:
    mean = pickle.load(f, encoding='latin1')
print(mean['mean_image_rgb'].shape)  # (3, 32, 32)

labels = pd.read_csv('./darknet_for_colab/labels.csv')



probability_minimum = 0.5 #Prawdopodobieństwa >20% będą odrzucane
threshold = 0.2 # Filtrowanie złych bounding boxes przez non-maximum suppression
path_to_cfg = 'darknet_for_colab/cfg/my_yolov4_test.cfg'
path_to_weights = 'my_yolov4_train_8000.weights'

network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layers_all = network.getLayerNames()
# Getting only detection YOLO v3 layers that are 82, 94 and 106
layers_names_output = [layers_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8') # Generowanie koloru RGB bounding box dla każdej klasy




webcam = cv2.VideoCapture(0)
writer = None # Writer that will be used to write processed frames
height, width = None, None # Variables for spatial dimensions of the frames
plt.rcParams['figure.figsize'] = (3, 3)
f, t = 0, 0

while True:
    ret, frame = video.read()
    if not ret: break
       
    if height is None and width is None:
        height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    network.setInput(blob) # Forward pass with blob through output layers
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    f += 1
    t += end - start

    print('Czas przetwarzania klatki nr {0}: {1:.5f} s'.format(f, end - start))

    bounding_boxes = []
    confidences = []
    class_numbers = []

    
    for result in output_from_network: # Going through all output layers after feed forward pass
        for detected_objects in result: # Going through all detections from current output layer
            scores = detected_objects[5:] # Getting 80 classes' probabilities for current detected object
            class_current = np.argmax(scores) # Getting index of the class with the maximum value of probability
            confidence_current = scores[class_current] # Getting value of probability for defined class

            if confidence_current > probability_minimum: # Eliminating weak predictions by minimum probability
                box_current = detected_objects[0:4] * np.array([width, height, width, height]) # Scaling bounding box coordinates to the initial frame size

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
            
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    if len(results) > 0: # Checking if there is any detected object been left
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            c_ts = frame[y_min:y_min+int(box_height), x_min:x_min+int(box_width), :] # Cut fragment with Traffic Sign
            
            if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                pass
            else:
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False) # Getting preprocessed blob with Traffic Sign of needed shape
                blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)
                # plt.imshow(blob_ts[0, :, :, :])
                # plt.show()

                scores = model.predict(blob_ts) # Klasyfikowanie znaku to jednej z 43 klas
                prediction = np.argmax(scores)
                print('{0} ({1:.2%})'.format(labels['SignName'][prediction], confidences[i]))
    
                playsound('sounds/{num}.mp3'.format(num = prediction))
                
                colour_box_current = colours[class_numbers[i]].tolist() # Colour for current bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2) # Drawing bounding box on the original current frame
                text_box_current = '{}: {:.4f}'.format(labels['SignName'][prediction], confidences[i]) # Preparing text with label and confidence for current bounding box
                cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2) # Putting text with label and confidence on the original image

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('result.mp4', fourcc, 25, (frame.shape[1], frame.shape[0]), True)

video.release()
writer.release()
