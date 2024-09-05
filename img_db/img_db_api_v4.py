import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
import cv2
import re
from deepface import DeepFace

from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import dotenv_values

env_vars = dotenv_values(".env")
temp_dir = env_vars["TEMP_DIR"]
assets_dir = env_vars["ASSETS_DIR"]
img_db_dir = env_vars["IMG_dB_dIR"]


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sort_names(list_image_dicts):
    unknown_names = [
        name_dict["face_name"]
        for name_dict in list_image_dicts
        if "unknown" in name_dict["face_name"].lower()
    ]
    known_names = [
        name_dict["face_name"]
        for name_dict in list_image_dicts
        if ("unknown" not in name_dict["face_name"].lower())
    ]

    known_names.sort()

    unknown_names.sort()
    known_names.extend(unknown_names)
    sort_list = []
    for name in known_names:
        for face_dict in list_image_dicts:
            if face_dict["face_name"] == name:
                sort_list.append(face_dict)
                break
    return sort_list


def custom_sort_by_number(lst):
    def extract_number_from_face_name(d):
        match = re.search(r"\d+", d["face_name"])
        return int(match.group()) if match else 0

    return sorted(lst, key=extract_number_from_face_name)


@app.post("/get_img_db/")
async def get_img_db():
    face_dir = [img_db_dir + i for i in os.listdir(img_db_dir)]
    output_dict = dict()
    db_faces_lst = []
    for i in face_dir:
        confidence_dict = dict()
        temp_dict = dict()
        back_track_count = 0
        for j in [i + "/" + face for face in os.listdir(i)]:
            img = cv2.imread(j)
            print("img_path is ", j)
            face_results = DeepFace.extract_faces(
                img, enforce_detection=False, align=True, detector_backend="retinaface"
            )
            print("here----")
            confidence = face_results[0]["confidence"]
            print("confidence is ", confidence)
            confidence_dict[j] = confidence
            img = cv2.imread(j)

            # face_results = face_detection.process(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            # if confidence:

            sorted_confidence_dict = dict(
                sorted(confidence_dict.items(), key=lambda item: item[1], reverse=True)
            )
            face_img = list(sorted_confidence_dict.keys())[0]
            face_name = (i.split("img_db/")[-1]).replace("_", " ").strip()
            temp_dict["face"] = (
                "http://10.10.0.212:8888/" + face_img.split(assets_dir)[-1]
            )
            temp_dict["face_name"] = face_name
            """
            else:
                if back_track_count==0:
                    all_faces = [i+'/'+ face for face in os.listdir(i)]
                    anchor_idx = np.random.randint(0,len(all_faces))
                    face_img = all_faces[anchor_idx]
                    face_name = (i.split('img_db/')[-1]).replace('_',' ').strip()
                    temp_dict['face']='http://10.10.0.212:8888'+face_img.split('/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets')[-1]
                    temp_dict['face_name'] = face_name
                #back_track_count += 1
                """
        face_paths = [i + "/" + x for x in os.listdir(i) if x.endswith(".jpg")]
        temp_dict["all_faces"] = [
            "http://10.10.0.212:8888/" + y.split(assets_dir)[-1] for y in face_paths
        ]
        if "face_name" not in list(temp_dict.keys()):
            print("i is-------", i)
            temp_dict["face_name"] = (i.split("img_db/")[-1]).replace("_", " ").strip()
        db_faces_lst.append(temp_dict)
    my_list = sort_names(db_faces_lst)
    sorted_list = custom_sort_by_number(my_list)
    output_dict["db"] = sorted_list
    return output_dict


if __name__ == "__main__":
    uvicorn.run("img_db_api_v4:app", host="0.0.0.0", port=8222, log_level="info")
