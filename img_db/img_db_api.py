import numpy as np
import os

from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
    
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
    face_dirs = ['/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/img_db/'+ i for i in os.listdir("/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/img_db")]
    output_dict = dict()
    db_faces_lst = []
    for i in face_dirs:
        temp_dict = dict()
        faces = [i  +'/' + face for face in os.listdir(i) if face.endswith('.jpg')]
        face_idx = np.random.randint(0,len(faces))
        face_img = faces[face_idx]
        face_name = (i.split('img_db/')[-1]).replace('_',' ')
        temp_dict['face']='http://10.10.0.212:8888'+face_img.split('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets')[-1]
        temp_dict['face_name'] = face_name
        db_faces_lst.append(temp_dict)
    output_dict['db'] = db_faces_lst
    return output_dict

if __name__ == "__main__":
    uvicorn.run("img_db_api:app",host="0.0.0.0",port=8222, log_level="info")
