import cv2
import numpy as np
import os
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import boto3


verify_face_url="http://52.22.41.2:8099/match_faces/"
# temp_dir='/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/bbox_faces'

aws_region = 'us-east-2'  # Replace with your AWS region
aws_access_key_id =   # Replace with your AWS access key ID
aws_secret_access_key = 
s3= boto3.client(
    's3',
    region_name="us-east-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

import boto3

# AWS Credentials
aws_region = 'us-east-2'  # Replace with your AWS region
aws_access_key_id = ''  # Replace with your AWS access key ID
aws_secret_access_key = '  # Replace with your AWS secret access key

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

def delete_s3_folder(bucket_name, s3_folder):
    try:
        # List objects in the specified S3 folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
        
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_file_path = obj['Key']
                print(f"Deleting s3://{bucket_name}/{s3_file_path}...")
                s3_client.delete_object(Bucket=bucket_name, Key=s3_file_path)
                print(f"Successfully deleted {s3_file_path}.")
        else:
            print(f"No objects found in S3 folder {s3_folder}.")
        
        # Check if there are more objects to delete (pagination)
        while response.get('IsTruncated'):
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder, ContinuationToken=response.get('NextContinuationToken'))
            for obj in response['Contents']:
                s3_file_path = obj['Key']
                print(f"Deleting s3://{bucket_name}/{s3_file_path}...")
                s3_client.delete_object(Bucket=bucket_name, Key=s3_file_path)
                print(f"Successfully deleted {s3_file_path}.")
                
    except Exception as e:
        print(f"Failed to delete objects. Error: {e}")




# embed_model = tf.keras.models.load_model(os.getcwd(), "bbox_images", "embedfinalall.h5")
# mp_face = mp.solutions.face_detection
# face_detection = mp_face.FaceDetection(min_detection_confidence=0.5,model_selection=0)


def check_input_deepface(imgs_lst):

    distance_set = set()
    check_vals = []
    for i in range(len(imgs_lst)):
        img1 = get_face(imgs_lst[i])
        filename=f"face{i}.jpg"
        cv2.imwrite(filename,img1)
        
        s3.upload_file(
                filename, "lds-test-public", f"temp_bbox/{filename}"
            )
        s3_url_face=f"https://lds-test-public.s3.amazonaws.com/temp_bbox/{filename}"
        


        count = 0
        for j in range(0, len(imgs_lst)):
            if j == i:
                continue
            img2 = get_face(imgs_lst[j])
            filename=f"temp_Face{j}.jpg"
            cv2.imwrite(filename,img2)
            
            s3.upload_file(
                    filename, "lds-test-public", f"temp_bbox/{filename}"
                )
            s3_url_temp=f"https://lds-test-public.s3.amazonaws.com/temp_bbox/{filename}"
            input_to_verify={ "face1": s3_url_face,
                                "face2":s3_url_temp}
            
            response = requests.post(verify_face_url, json=input_to_verify)
            result=response.json()
           
        

            if result["distance"] <= 0.7:
                count += 1
            distance_set.add(result["distance"])
        if count / len(imgs_lst) >= 0.3:
            check_vals.append(True)
        else:
            check_vals.append(False)
    print(distance_set)
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
            request_body = {
            "imgurl": img,
            "videoname": "empty",
            "count": 0
        }
            response = requests.post(url_extract_face, json=request_body)
            response_json=response.json()
            face_results=response_json["result"]
                
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
 
    for i, img_arr in enumerate(imgs_lst):
        img_arr_copy = img_arr.copy()

        x, y, w, h = boxes[i]
        if status and check_vals[i]:
            face = img_arr[y : y + h, x : x + w]
            # cv2.imwrite(f"{faces_dir}/face_{i}.jpg", face)
            cv2.imwrite(f"face_{i}.jpg",face)
            s3.upload_file(
                    f"face_{i}.jpg", "lds-test-public", f"{faces_dir}/face_{i}.jpg"
                )
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        else:
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        # img_rgb_copy = cv2.cvtColor(img_arr_copy, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"{images_dir}/face_{i}.jpg", img_arr)
        cv2.imwrite(f"face_{i}.jpg", img_arr)
        s3.upload_file(
                   f"face_{i}.jpg", "lds-test-public", f"{images_dir}/face_{i}.jpg"
                )
        url_faces=f"https://lds-test-public.s3.amazonaws.com/{images_dir}/face_{i}.jpg"
        bbox_face_paths.append(url_faces)
        # cv2.imwrite(f"{normal_imgs}/face_{i}.jpg", img_arr_copy)
        cv2.imwrite("face_{i}.jpg",img_arr_copy)
        s3.upload_file(
                   f"face_{i}.jpg", "lds-test-public", f"{normal_imgs}/face_{i}.jpg"
                )

    # bbox_face_paths_new = [f"{images_dir}/{i}" for i in os.listdir(images_dir)]
    # for i in bbox_face_paths_new:
    #     bboxed_faces.append("http://10.10.0.212:8888/" + i.split(assets_dir)[-1])

    output_dict["bboxed_faces"] = bbox_face_paths
    return output_dict


def get_face(image):
    response = requests.get(image, stream=True, timeout=60)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    request_body = {
        "imgurl": image,
        "videoname": "empty",
        "count": 0
    }
    response = requests.post(url_extract_face, json=request_body)
    response_json=response.json()
    result=response_json["result"]
    box = result[0]["facial_area"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    xmax = x + w
    ymax = y + h
    face = frame[y:ymax, x:xmax]

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

url_extract_face="http://52.22.41.2:8099/extract_face/"

@app.post("/get_face_bbox/")
async def process_video(data: ImageInput):
    imgs_lst = []
    img_list_images=[]
    multiple_faces_lst = []
    box_coords = []
    no_faces_images = []
    base64_lst = data.data
    print("#" * 100)
    delete_s3_folder("lds-test-public", "temp_bbox")
    for j,i in enumerate(base64_lst):
        # print(i)
        print("#" * 100)
        base64_str = i["base64String"]
        img = base64_to_image(base64_str)
        filename=f"img_{j}.jpg"
        cv2.imwrite(filename,img)
        s3.upload_file(
                filename, "lds-test-public", f"temp_bbox/{filename}"
            )
        s3_url=f"https://lds-test-public.s3.amazonaws.com/temp_bbox/{filename}"
        request_body = {
            "imgurl": s3_url,
            "videoname": "empty",
            "count": 0
        }
        print("filename",s3_url)
        response = requests.post(url_extract_face, json=request_body)
        response_json=response.json()
        print(response_json)
        result=response_json["result"]

        
        if len(result) == 1:
            box = result[0]["facial_area"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            imgs_lst.append(s3_url)
            img_list_images.append(img)
            box_coords.append([x, y, w, h])
        elif len(result) > 1:
            multiple_faces_lst.append(s3_url)
        else:
            no_faces_images.append(s3_url)
    

    images_dir = f"temp_bbox/images"
    faces_dir = f"temp_bbox/faces"
    normal_imgs = f"temp_bbox/normal_images"
    anchor_dir = f"temp_bbox/anchor"
    bbox_face_paths = []
    status, check_vals = check_input_deepface(imgs_lst)
    try:
        anchor, anchor_idx = get_anchor(imgs_lst, check_vals)
        print("anchor is ", anchor)
        anchor_face = get_face(anchor)
        anchor_face_copy = anchor_face.copy()
        cv2.imwrite("anchor_face.jpg", anchor_face_copy)
        anchor_face_path = f"{anchor_dir}/anchor_face.jpg"
        s3.upload_file(
                    "anchor_face.jpg", "lds-test-public", anchor_face_path
                )
        anchor_url=f"https://lds-test-public.s3.amazonaws.com/{anchor_face_path}"
        
    except:
        return {"error": "unable to get anchor face from given images"}

    for index, img_multi in enumerate(multiple_faces_lst):
        response = requests.get(img_multi, stream=True, timeout=60)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        request_body = {
            "imgurl": img_multi,
            "videoname": "empty",
            "count": 0
        }
        response = requests.post(url_extract_face, json=request_body)
        response_json=response.json()
        faces=response_json["result"]
        min_val = -10000
        min_index = 0
        boxes = []
        frame_copy = frame.copy()
        faces_copy = []
        for idx, face in enumerate(faces):
            box = face["facial_area"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            boxes.append([x, y, w, h])
            face_refer = frame[y : y + h, x : x + h]
            face_refer_copy = face_refer.copy()
            faces_copy.append(face_refer_copy)
            cv2.imwrite("face_refer_copy.jpg",face_refer_copy)
            s3.upload_file(
                    "face_refer_copy.jpg", "lds-test-public", "temp_box/face_refer_copy.jpg"
                )
            face_refer_url=f"https://lds-test-public.s3.amazonaws.com/temp_box/face_refer_copy.jpg"
            input_to_verify={ "face1": anchor_url,
                            "face2":face_refer_url}
            
            response = requests.post(verify_face_url, json=input_to_verify)
            result=response.json()
            # result=verify(anchor_face,face_refer)

            if result["distance"] < 0.6:
                min_val = result
                min_index = idx
        for id, box_ in enumerate(boxes):
            if id == min_index:
                x, y, w, h = box_
                
                
                cv2.imwrite(
                    os.path.join(f"multi_face_{index}.jpg"), faces_copy[id]
                )
                s3.upload_file(
                    f"multi_face_{index}.jpg", "lds-test-public", f"{faces_dir}/multi_face_{index}.jpg"
                )

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                x, y, w, h = box_
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # img_rgb_multi = cv2.cvtColor(img_multi, cv2.COLOR_BGR2RGB)
        # img_rgb_copy = cv2.cvtColor(img_multi_copy, cv2.COLOR_BGR2RGB)
        #write_path = os.path.join(images_dir, f"multi_face_{index}.jpg")
        write_path = f"multi_face_{index}.jpg"
        cv2.imwrite(write_path, frame)
        s3.upload_file(
                    f"multi_face_{index}.jpg", "lds-test-public", f"{images_dir}/multi_face_{index}.jpg"
                )
        url_multiface=f"https://lds-test-public.s3.amazonaws.com/{images_dir}/multi_face_{index}.jpg"
        # cv2.imwrite(f"{normal_imgs}/multi_face_{index}.jpg", frame_copy)
        cv2.imwrite("multi_face_{index}.jpg",frame_copy)
        s3.upload_file(
                    f"multi_face_{index}.jpg", "lds-test-public", f"{normal_imgs}/multi_face_{index}.jpg"
                )
        bbox_face_paths.append(
            url_multiface
        )

    for index, no_face_image in enumerate(no_faces_images):
        write_path = os.path.join(f"no_face_{index}.jpg")
        cv2.imwrite(write_path, no_face_image)
        s3.upload_file(
                    f"no_face_{index}.jpg", "lds-test-public", f"{images_dir}/no_face_{index}.jpg"
                )
        url_noface=f"https://lds-test-public.s3.amazonaws.com/{images_dir}/no_face_{index}.jpg"

        bbox_face_paths.append(
            url_noface
        )

    output_dict = extract_faces(
        img_list_images,
        status,
        box_coords,
        check_vals,
        faces_dir,
        images_dir,
        bbox_face_paths,
        normal_imgs,
    )
    
    output_dict["anchor_path"] = anchor_url

    return output_dict


if __name__ == "__main__":
    uvicorn.run("api_bbox_aws:app", host="0.0.0.0", port=8199, log_level="info")
