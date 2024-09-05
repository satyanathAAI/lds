
import os


import re

import boto3
from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import requests




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
aws_region = 'us-east-2'  # Replace with your AWS region
aws_access_key_id = ''  # Replace with your AWS access key ID
aws_secret_access_key = ''
s3= boto3.client(
    's3',
    region_name="us-east-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
url_extract_face="http://52.22.41.2:8099/extract_face/"
@app.post("/get_img_db/")
async def get_img_db():
    bucket_name = 'lds-test-public'
    prefix = 'img_db/'

    # List objects with the specified prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')


    # Print the folder names
    if 'CommonPrefixes' in response:
        output_dict = dict()
        db_faces_lst = []
        for prefix in response['CommonPrefixes']:
            confidence_dict = dict()
            temp_dict = dict()
            back_track_count = 0
            print(prefix)
            response_images = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix["Prefix"], Delimiter='/')
            prefix=prefix["Prefix"]
            print("response images",response_images)
            face_paths=[]
            for j in response_images['Contents']:
                j=j["Key"]
                img_url=f"https://lds-test-public.s3.amazonaws.com/{j}"
                request_body = {
                    "imgurl": img_url,
                    "videoname": "empty",
                    "count": 0
                }
                try:
                    response = requests.post(url_extract_face, json=request_body)
                    response_json=response.json()
                    face_results=response_json["result"]
                    
                    print("here----")
                    confidence = face_results[0]["confidence"]
                except:
                    confidence=0.0
                print("confidence is ", confidence)
                confidence_dict[j] = confidence
               

                # face_results = face_detection.process(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
                # if confidence:

                sorted_confidence_dict = dict(
                    sorted(confidence_dict.items(), key=lambda item: item[1], reverse=True)
                )
                face_img = list(sorted_confidence_dict.keys())[0]
                face_name = (prefix.split("img_db/")[-1]).replace("_", " ").strip()
                temp_dict["face"] = (
                    f"https://lds-test-public.s3.amazonaws.com/{face_img}"
                )
                temp_dict["face_name"] = face_name
                face_paths.append(img_url)
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
         
            temp_dict["all_faces"] = face_paths
            if "face_name" not in list(temp_dict.keys()):
                print("prefix is-------", )
                temp_dict["face_name"] = (prefix.split("img_db/")[-1]).replace("_", " ").strip()
            db_faces_lst.append(temp_dict)
    my_list = sort_names(db_faces_lst)
    sorted_list = custom_sort_by_number(my_list)
    output_dict["db"] = sorted_list
    return output_dict


if __name__ == "__main__":
    uvicorn.run("img_db_aws:app", host="0.0.0.0", port=8222, log_level="info")
