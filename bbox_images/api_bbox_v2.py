import cv2
import numpy as np
import os
from deepface import DeepFace
import re
import json
import base64
import io

from tensorflow.keras.preprocessing import image
import tensorflow as tf
from pytube import YouTube

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 10GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=10024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

embed_model=tf.keras.models.load_model("embedfinalall.h5")

def base64_to_image(base64_string):
    image_binary = base64.b64decode(base64_string)
    image_array = np.frombuffer(image_binary, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def findCosineDistance(source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
def preprocess(img):
    img=cv2.resize(img,(60,60))
    img=img/255.
    img=np.expand_dims(img,axis=0)
    return img

def get_embd(data_infer):
      img_embd=embed_model(data_infer)
      return img_embd
    
def check_input(imgs_lst):
    distance_set = set()
    for i in range(len(imgs_lst)):
        img1 = preprocess(imgs_lst[i])
        embed1 = get_embd(img1)
        for j in range(i + 1, len(imgs_lst)):
            img2 = preprocess(imgs_lst[j])
            embed2 = get_embd(img2)
            result = findCosineDistance(embed1[0], embed2[0])
            distance_set.add(result)
    if len(distance_set) == 1:
        status = True
    else:
        status = False
    return status

def extract_faces(imgs_lst,status,detector='retinaface'):
    output_dict = dict()
    bboxed_faces = []
    if not os.path.exists('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces'):
        os.makedirs('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces',exist_ok=True)
    for i,img_arr in enumerate(imgs_lst):
        faces = DeepFace.extract_faces(img_arr,enforce_detection=False,align=True,detector_backend = detector)
        len_x = img_arr.shape[0]
        len_y = img_arr.shape[1]
        if len(faces) == 1:
            box = faces[0]['facial_area']
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            if status == True:
                cv2.rectangle(img_arr, (x, y),(x + w, y + h),(255, 0, 0), 2)
            else:
                cv2.rectangle(img_arr, (x, y),(x + w, y + h),(0, 255, 0), 2)
            img_rgb = cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces/face_{i}.jpg",img_rgb)
    bbox_face_paths = ['/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces/'+i for i in os.listdir('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces')]
    for i in bbox_face_paths :
        bboxed_faces.append('http://10.10.0.212:8888'+ i.split('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets')[-1])
    output_dict['bboxed_faces'] = bboxed_faces
    return output_dict


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    data : list

@app.post("/get_face_bbox/")
async def process_video(data: ImageInput):
    imgs_lst = []
    base64_lst = data.data 
    print('#'*100)
    for i in base64_lst:
        print(i)
        print('#'*100)
        base64_str = i['base64String']
        img = base64_to_image(base64_str)
        imgs_lst.append(img)
    if os.path.exists('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces'):
        os.system('rm -r -f /home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces')
    status = check_input(imgs_lst)
    output_dict = extract_faces(imgs_lst,status)
    return output_dict

if __name__ == "__main__":
    uvicorn.run("api_bbox_v2:app",host="0.0.0.0",port=8199, log_level="info")
    
