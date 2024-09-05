import numpy as np
import os
import cv2
import mediapipe as mp
import os

from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5,model_selection=0)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/get_img_db/")
async def get_img_db():
    face_dir = ['/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/img_db/'+ i for i in os.listdir("/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/img_db")]
    output_dict = dict()
    db_faces_lst = []
    for i in face_dir:
        confidence_dict = dict()
        temp_dict = dict()
        try:
            for j in [i+'/'+ face for face in os.listdir(i)]:
                    face_results = face_detection.process(cv2.imread(j))
                    confidence_dict[j] = face_results.detections[0].score[0]
            sorted_confidence_dict = dict(sorted(confidence_dict.items(), key=lambda item: item[1], reverse=True))
            face_img = list(sorted_confidence_dict.keys())[0]
            face_name = (i.split('img_db/')[-1]).replace('_',' ').strip()
            temp_dict['face']='http://10.10.0.212:8888'+face_img.split('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets')[-1]
            temp_dict['face_name'] = face_name
            db_faces_lst.append(temp_dict)
        except Exception as e: 
            print(e)
    output_dict['db'] = db_faces_lst
    return output_dict

if __name__ == "__main__":
    uvicorn.run("img_db_api_v1:app",host="0.0.0.0",port=8222, log_level="info")
