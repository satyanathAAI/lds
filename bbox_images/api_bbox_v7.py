import cv2
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from deepface import DeepFace
import re
import json
import base64
import io

from tensorflow.keras.preprocessing import image
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 10GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=10024)]
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
from pytube import YouTube

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware
import torch
from dotenv import dotenv_values

env_vars = dotenv_values(".env")
temp_dir = env_vars["TEMP_DIR"]
assets_dir = env_vars["ASSETS_DIR"]
# temp_dir='/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces'
flag_gpu_available = torch.cuda.is_available()
if flag_gpu_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# embed_model = tf.keras.models.load_model(os.getcwd(), "bbox_images", "embedfinalall.h5")
# mp_face = mp.solutions.face_detection
# face_detection = mp_face.FaceDetection(min_detection_confidence=0.5,model_selection=0)


def check_input_deepface(imgs_lst):

    distance_set = set()
    check_vals = []
    for i in range(len(imgs_lst)):
        img1 = get_face(imgs_lst[i])

        count = 0
        for j in range(0, len(imgs_lst)):
            if j == i:
                continue
            img2 = get_face(imgs_lst[j])
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                detector_backend="skip",
                model_name="ArcFace",
            )
            print(result)

            if result["distance"] <= 0.7:
                count += 1
            distance_set.add(result["distance"])
        if count / len(imgs_lst) >= 0.3:
            check_vals.append(True)
        else:
            check_vals.append(False)
    if len(distance_set) == 1:
        status = False
    else:
        status = True
    return status, check_vals


def get_anchor(imgs, check_vals):
    confidence_dict = dict()
    temp_dict = dict()
    max_val = -10000
    anchor = ""
    anchor_idx = 0
    for idx, img in enumerate(imgs):
        if not check_vals[idx]:
            continue
        try:
            face_results = DeepFace.extract_faces(
                img, enforce_detection=True, align=True, detector_backend="retinaface"
            )
            confidence = face_results[0]["confidence"]
            if confidence > max_val:
                max_val = confidence
                anchor = img
                anchor_idx = idx

        except:
            pass
    return anchor, anchor_idx


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


def extract_faces(
    imgs_lst,
    status,
    boxes,
    check_vals,
    faces_dir,
    images_dir,
    bbox_face_paths,
    normal_imgs,
    detector="retinaface",
):
    output_dict = dict()
    bboxed_faces = []
    print("check vals are :", check_vals)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    for i, img_arr in enumerate(imgs_lst):
        img_arr_copy = img_arr.copy()

        x, y, w, h = boxes[i]
        if status and check_vals[i]:
            face = img_arr[y : y + h, x : x + w]
            cv2.imwrite(f"{faces_dir}/face_{i}.jpg", face)
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        else:
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        # img_rgb_copy = cv2.cvtColor(img_arr_copy, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{images_dir}/face_{i}.jpg", img_arr)
        cv2.imwrite(f"{normal_imgs}/face_{i}.jpg", img_arr_copy)

    bbox_face_paths_new = [f"{images_dir}/{i}" for i in os.listdir(images_dir)]
    for i in bbox_face_paths_new:
        bboxed_faces.append("http://10.10.0.212:8888/" + i.split(assets_dir)[-1])

    output_dict["bboxed_faces"] = bboxed_faces
    return output_dict


def get_face(image):
    faces = DeepFace.extract_faces(
        image, enforce_detection=False, align=True, detector_backend="retinaface"
    )
    box = faces[0]["facial_area"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    xmax = x + w
    ymax = y + h
    face = image[y:ymax, x:xmax]
    return face


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
    data: list


@app.post("/get_face_bbox/")
async def process_video(data: ImageInput):
    imgs_lst = []
    multiple_faces_lst = []
    box_coords = []
    no_faces_images = []
    base64_lst = data.data
    print("#" * 100)
    for i in base64_lst:
        # print(i)
        print("#" * 100)
        base64_str = i["base64String"]
        img = base64_to_image(base64_str)
        
        faces = DeepFace.extract_faces(
            img, enforce_detection=False, align=True, detector_backend="retinaface"
        )
        if len(faces) == 1:
            box = faces[0]["facial_area"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            imgs_lst.append(img)
            box_coords.append([x, y, w, h])
        elif len(faces) > 1:
            multiple_faces_lst.append(img)
        else:
            no_faces_images.append(img)
    print("temp dir is ", temp_dir)
    if os.path.exists(temp_dir):
        os.system(f"rm -r -f {temp_dir}")
        time.sleep(2)
    images_dir = f"{temp_dir}/images"
    faces_dir = f"{temp_dir}/faces"
    normal_imgs = f"{temp_dir}/normal_images"
    anchor_dir = f"{temp_dir}/anchor"
    os.makedirs(images_dir)
    os.makedirs(faces_dir)
    os.makedirs(normal_imgs)
    os.makedirs(anchor_dir)
    bbox_face_paths = []
    status, check_vals = check_input_deepface(imgs_lst)
    try:
        anchor, anchor_idx = get_anchor(imgs_lst, check_vals)
        print("anchor is ", anchor)
        anchor_face = get_face(anchor)
        anchor_face_copy = anchor_face.copy()
        anchor_face_path = os.path.join(anchor_dir, "anchor_face.jpg")
        cv2.imwrite(anchor_face_path, anchor_face_copy)
    except:
        return {"error": "unable to get anchor face from given images"}

    for index, img_multi in enumerate(multiple_faces_lst):
        faces = DeepFace.extract_faces(
            img_multi,
            enforce_detection=False,
            align=True,
            detector_backend="retinaface",
        )
        min_val = -10000
        min_index = 0
        boxes = []
        img_multi_copy = img_multi.copy()
        faces_copy = []
        for idx, face in enumerate(faces):
            box = face["facial_area"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            boxes.append([x, y, w, h])
            face_refer = img_multi[y : y + h, x : x + h]
            face_refer_copy = face_refer.copy()
            faces_copy.append(face_refer_copy)
            result = DeepFace.verify(
                img1_path=anchor_face,
                img2_path=face_refer,
                detector_backend="skip",
                model_name="ArcFace",
            )
            # result=verify(anchor_face,face_refer)

            if result["distance"] < 0.6:
                min_val = result
                min_index = idx
        for id, box_ in enumerate(boxes):
            if id == min_index:
                x, y, w, h = box_
                cv2.imwrite(
                    os.path.join(faces_dir, f"multi_face_{index}.jpg"), faces_copy[id]
                )

                cv2.rectangle(img_multi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                x, y, w, h = box_
                cv2.rectangle(img_multi, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # img_rgb_multi = cv2.cvtColor(img_multi, cv2.COLOR_BGR2RGB)
        # img_rgb_copy = cv2.cvtColor(img_multi_copy, cv2.COLOR_BGR2RGB)
        write_path = os.path.join(images_dir, f"multi_face_{index}.jpg")

        cv2.imwrite(write_path, img_multi)
        cv2.imwrite(f"{normal_imgs}/multi_face_{index}.jpg", img_multi_copy)
        bbox_face_paths.append(
            "http://10.10.0.212:8888/" + write_path.split(assets_dir)[-1]
        )

    for index, no_face_image in enumerate(no_faces_images):
        write_path = os.path.join(images_dir, f"no_face_{index}.jpg")
        cv2.imwrite(write_path, no_face_image)
        bbox_face_paths.append(
            "http://10.10.0.212:8888/" + write_path.split(assets_dir)[-1]
        )

    output_dict = extract_faces(
        imgs_lst,
        status,
        box_coords,
        check_vals,
        faces_dir,
        images_dir,
        bbox_face_paths,
        normal_imgs,
    )
    anchor_path = "http://10.10.0.212:8888/" + anchor_face_path.split(assets_dir)[-1]
    output_dict["anchor_path"] = anchor_path

    return output_dict


if __name__ == "__main__":
    uvicorn.run("api_bbox_v7:app", host="0.0.0.0", port=8199, log_level="info")
