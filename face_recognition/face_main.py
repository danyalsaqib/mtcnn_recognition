import cv2
import numpy as np
import pandas as pd
import json
import os
from mtcnn_ort import MTCNN
from utils import *
import onnxruntime

# Initializing the detector
detector = MTCNN()

# Initializing the recognizer
model_path = "models/arcface.onnx"
session = onnxruntime.InferenceSession(model_path, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
src[:, 0] += 8.0

tform = trans.SimilarityTransform()


"""
def face_infer(recieved_image_path):
    # images_path is the list containing 
    images_path = json.load(recieved_image_path)
    for image_path in images_path[]:
"""

def infer_image(img_path):

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces_raw(image)
    #detection_results_show(img_path, faces)

    for i in range(len(faces[0])):
        # Size of Image is less than 100 pixel,scrap it
        image = cv2.imread(img_path)
        if ((int(faces[0][i][3]) - int(faces[0][i][1])) > 100 or (int(faces[0][i][2]) - int(faces[0][i][0])) > 100):
            crop_image = image[int(faces[0][i][1]): int(faces[0][i][3]), int(faces[0][i][0]):int(faces[0][i][2])]
            landmarks = [int(faces[1][0][i]), int(faces[1][1][i]), int(faces[1][2][i]), int(faces[1][3][i]), int(faces[1][4][i]),
                        int(faces[1][5][i]), int(faces[1][6][i]), int(faces[1][7][i]), int(faces[1][8][i]), int(faces[1][9][i])]
            if find_roll(landmarks) > - 20 and  find_roll(landmarks) < 20 and find_yaw(landmarks) > -50 and find_yaw(landmarks) < 50 and find_pitch(landmarks) < 2 and find_pitch(landmarks) > 0.5:
                #cropped_results_show(crop_image, landmarks)
                # Recognition
                facial5points = np.reshape(landmarks, (2, 5)).T
                tform.estimate(facial5points, src)
                M = tform.params[0:2, :]
                img = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
                #cropped_results(img)
                #print(img.shape)
                blob = cv2.dnn.blobFromImage(img, 1, (112, 112), (0, 0, 0))
                blob -= 127.5
                blob /= 128
                result = session.run([output_name], {input_name: blob})
                dictionary = {'names': "bajwa sb", 'embeddings': result[0][0].tolist()}
                
                print("Embedding successfully obtained. Shape: ", len(result[0][0]))
                return result[0][0]

                #with open('dictionary.json', 'w') as handle:
                #    json.dump(dictionary, handle)

                #distance = findCosineDistance(result[0][0], bajwa)
                #print(distance)


            else:
                print("Invalid Pose")
                return -1
        else:
            print("Image size is small")
            return -1

def findCosineDistance(source_representation, test_representation):
    #print("Source Representation: ", source_representation.shape)
    #print("Test Representation: ", test_representation.shape)
    a = np.matmul(np.transpose(source_representation), test_representation)
    print("a: ", a.shape)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

if __name__ == '__main__':
    detector = MTCNN()
    test_pic = "Aamir Hussain Liaquat_img_1.jpg"
    infer_image(test_pic)
    
