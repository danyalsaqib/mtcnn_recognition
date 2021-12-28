import cv2
import numpy as np
import json
import os
from mtcnn_ort import MTCNN
from utils import *
import onnxruntime

# Initializing the detector
detector = MTCNN()

# Initializing the recognizer
os.chdir("face_recognition")
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
distance_threshold = 0.5

file_batch_dict = {
    "frame_1" : "images/image_1.jpeg",
    "frame_2" : "images/image_2.jpeg",
    "frame_3" : "images/image_3.jpeg",
    "frame_4" : "images/image_4.jpeg",
    "frame_5" : "images/image_5.jpeg"
}

def face_infer_batch(batch_dict):
    print("Length : ", len(batch_dict))
    results = {}

    for image_path in batch_dict:
        results[image_path] = infer_image(batch_dict[image_path])
    print(results)
    return results


def infer_image(img_path):

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces_raw(image)
    celeb = []
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

                f = open('dict_arcface.json')
                data = json.load(f)
                #print("Length of dict : ", len(data["embeddings"]))
                distance = 1000
                best_distance = 1
                best_index = -1
                for j in range(len(data["embeddings"])):
                    #print("Embedding : ", data["embeddings"][j])
                    distance = findCosineDistance(result[0][0], data["embeddings"][j])
                    if (distance < distance_threshold) and (distance < best_distance):
                        best_index = j
                        best_distance = distance
                        print("Best Distance : ", best_distance)

                if (best_index != -1):
                    celeb += [data["names"][best_index]]
                    
 

                f.close()

            else:
                print("Invalid Pose")
        else:
            print("Image size is small")
    print(celeb)
    return celeb

if __name__ == '__main__':
    detector = MTCNN()
    test_pic = "images/image_1.jpeg"
    face_infer_batch(file_batch_dict)
