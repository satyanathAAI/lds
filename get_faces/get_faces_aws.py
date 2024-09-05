import cv2
import requests
import numpy as np
import os
from api_utils.mongo_utils import insert_to_mongo
from api_utils.s3_utils import check_folder_exists
import re
import json
import time
import boto3

collection_name = "athelete_detection"
verify_face_url = "http://52.22.41.2:8099/match_faces/"
aws_region = "us-east-2"  # Replace with your AWS region
aws_access_key_id = ""  # Replace with your AWS access key ID
aws_secret_access_key = ""
s3 = boto3.client(
    "s3",
    region_name="us-east-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)


# import yaml
# config_file_path = '/home2/asgtestdrive2023/Projects/MAM/AI-Team/search/prod/config.yaml'
# # Setting the GPU to the device :- cuda:2
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def preprocess_video_names(name):
    name = name.strip()
    name = re.sub(r"[^\w\s.]", "", name)
    name = name.strip()
    clean_name = name.replace(" ", "_")
    return clean_name


def download_youtube_video(url, output_path="local_videos"):
    """url : string, youtube video URL.
    output_path : string, path where you want the video to be downloaded at.

    Returns the path where the video is present"""
    try:
        yt = YouTube(url)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        video = yt.streams.get_highest_resolution()
        print(f"Downloading: {video.title}...")
        filename = video.title
        clean_filename = preprocess_video_names(filename)
        video.download(output_path, filename=f"{clean_filename}.mp4")
        print("Download complete!")
        clean_filename = f"{clean_filename}.mp4"
        return clean_filename
    except Exception as e:
        print(f"Error: {str(e)}")


url_extract_face = "http://52.22.41.2:8099/extract_face/"
url_blur_classifier = "http://52.22.41.2:8099/blur_classifier/"


# This function extracts the faces from the given video
def extract_faces(video_file, n_fps=3):
    """video_file : string, name of the video.
    n_fps : int, at what rate you want to extract the frames from the video.

    Returns the path at which all the extracted faces are being stored along with the video filename.
    """
    print("Reading the Video...")
    start = time.time()
    videoname = video_file.split("/")[-1]
    faces_path = f"Face_detect/{videoname}/faces/"
    status = check_folder_exists(bucket_name="lds-test-public", folder_name=faces_path)
    
    if not status:

        print("video file is", video_file)
        cap = cv2.VideoCapture(video_file)

        global fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        print(f"Total frames: {total_frames}")
        print(f"Frames per second (fps): {fps}")
        print(f"Duration (seconds): {duration}")

        interval = int(60)
        frames_lst = list(range(0, total_frames, interval))
        count = 1
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count in frames_lst:
                try:
                    cv2.imwrite("frame.jpg", frame)
                    content_type = "image/jpeg"
                    s3.upload_file(
                        "frame.jpg",
                        "lds-test-public",
                        "frame.jpg",
                        ExtraArgs={"ContentType": content_type},
                    )
                    s3_url = f"https://lds-test-public.s3.amazonaws.com/frame.jpg"
                    request_body = {
                        "imgurl": s3_url,
                        "videoname": videoname,
                        "count": count,
                    }
                    response = requests.post(url_extract_face, json=request_body)
                    response_json = response.json()
                    result = response_json["facepaths"]
                    frames.extend(result)

                except Exception as e:
                    print(e)
                    pass
            count += 1
    else:
        pass

    end = time.time()
    print("Time took :", end - start)
    return frames, video_file


def filter_clear_faces(frames, video_file):
    start = time.time()
    videoname = video_file.split("/")[-1]
    status = check_folder_exists(
        bucket_name="lds-test-public",
        folder_name=f"Face_detect/{videoname}/no_blur_faces/",
    )
    staus_2 = check_folder_exists(
        bucket_name="lds-test-public",
        folder_name=f"Face_detect/{videoname}/blur_faces/",
    )
    if not (status) and not (staus_2):
        if len(frames) != 0:
            clear_faces = []
            for i in frames:
                frame_num = i.split("Frame_")[-1].split(".")[0]
                imagename = f"Frame_{frame_num}.jpg"

                request_body = {
                    "videoname": video_file.split("/")[-1],
                    "image_name": imagename,
                    "faceurl": i,
                }
                print("request body", request_body)
                response = requests.post(url=url_blur_classifier, json=request_body)
                response_json = response.json()
                print("response_json", response_json)
                s3_url = response_json["s3url"]
                if "no_blur_faces" in s3_url:
                    clear_faces.append(s3_url)
        else:
            clear_faces = []
    else:
        videoname = video_file.split("/")[-1]
        response = s3.list_objects_v2(
            Bucket="lds-test-public", Prefix=f"{videoname}/no_blur_faces/"
        )
        clear_faces = []
        if "Contents" in response:
            for obj in response["Contents"]:
                s3_file_path = obj["Key"]
                url = f"https://lds-test-public.s3.amazonaws.com/{s3_file_path}"
                clear_faces.append(url)

    end = time.time()
    print("Time took :", end - start)
    return clear_faces


def copy_object(source_bucket, source_key, destination_bucket, destination_key):
    copy_source = {"Bucket": source_bucket, "Key": source_key}
    s3.copy_object(
        CopySource=copy_source, Bucket=destination_bucket, Key=destination_key
    )


def extract_unqiue_faces(clear_faces, video_file):
    videoname = video_file.split("/")[-1]
    status = check_folder_exists(
        bucket_name="lds-test-public",
        folder_name=f"Face_detect/{videoname}/unique_faces/",
    )
    if not status:

        if len(clear_faces) != 0:
            faces_to_remove = []
            for i in range(len(clear_faces)):
                if i not in faces_to_remove:
                    for j in range(i + 1, len(clear_faces)):
                        print(clear_faces[i], clear_faces[j])
                        input_to_verify = {
                            "face1": clear_faces[i],
                            "face2": clear_faces[j],
                        }

                        response = requests.post(verify_face_url, json=input_to_verify)
                        result = response.json()

                        # result = findCosineDistance(embed1[0], embed2[0])
                        if result["distance"] <= 0.6:
                            faces_to_remove.append(j)
            for index in sorted(set(faces_to_remove), reverse=True):
                clear_faces.pop(index)

            for j in clear_faces:
                frame_num = j.split("Frame_")[-1].split(".jpg")[0]
                timeline = round((int(frame_num) / fps), 2)
                s3_file_path = j.split("https://lds-test-public.s3.amazonaws.com/")[-1]
                copy_object(
                    source_bucket="lds-test-public",
                    source_key=s3_file_path,
                    destination_bucket="lds-test-public",
                    destination_key=f"{videoname}/unique_faces/Frame_{frame_num}_time_{timeline}.jpg",
                )
            response = s3.list_objects_v2(
                Bucket="lds-test-public", Prefix=f"{videoname}/unique_faces/"
            )
            clear_faces = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    s3_file_path = obj["Key"]
                    url = f"https://lds-test-public.s3.amazonaws.com/{s3_file_path}"
                    clear_faces.append(url)

        else:
            clear_faces = []
    else:
        response = s3.list_objects_v2(
            Bucket="lds-test-public", Prefix=f"{videoname}/unique_faces/"
        )
        clear_faces = []
        if "Contents" in response:
            for obj in response["Contents"]:
                s3_file_path = obj["Key"]
                url = f"https://lds-test-public.s3.amazonaws.com/{s3_file_path}"
                clear_faces.append(url)

    return clear_faces, video_file


verify_url = f"http://localhost:8014/verify"


def name_faces(clear_faces, video_file):
    if not len(clear_faces) == 0:
        # with open('new_embed.json','r') as file:
        #     data = json.load(file)
        # keys_dict=list(data.keys())
        output_dict = {}
        dict_elements = []
        for i in clear_faces:
            temp_dict = dict()
            input_to_verify = {"face_path": i}
            # input_to_verify={"face_path": i.split(assets_path)[-1]}
            result = requests.post(verify_url, json=input_to_verify)
            result_json = result.json()
            identity = result_json["identity"]

            timeline = i.split("_time_")[-1].split(".jpg")[0]
            temp_dict["img"] = i
            temp_dict["name"] = identity
            temp_dict["timeline"] = float(timeline)
            dict_elements.append(temp_dict)
        output_dict["faces"] = dict_elements
    else:
        output_dict = {}
        output_dict["faces"] = []

    insert_to_mongo(output_dict, collection_name)

    return output_dict


def main_logic(video_file):
    print("Pipeline initiated.")

    frames, video_file = extract_faces(video_file)
    clear_faces = filter_clear_faces(frames, video_file)
    unique_faces, video_file = extract_unqiue_faces(clear_faces, video_file)
    output_dict = name_faces(unique_faces, video_file)
    del output_dict["_id"]
    print(output_dict)
    return output_dict
