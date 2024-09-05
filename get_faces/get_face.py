import cv2
import requests
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from deepface import DeepFace
import re
import json
import time
import tensorflow as tf

from tensorflow.keras.preprocessing import image
import tensorflow as tf
from pytube import YouTube
from dotenv import dotenv_values

# Load the .env file
env_vars = dotenv_values(".env")
http_server_url = env_vars["HTTPSERVERURL"]
assets_path = env_vars["ASSETS_DIR"]


# import yaml
# config_file_path = '/home2/asgtestdrive2023/Projects/MAM/AI-Team/search/prod/config.yaml'
# # Setting the GPU to the device :- cuda:2
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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


model = tf.keras.models.load_model(
    os.path.join(os.getcwd(), "get_faces", "Final_classification_model.keras")
)


def preprocess_video_names(name):
    name = name.strip()
    name = re.sub(r"[^\w\s.]", "", name)
    name = name.strip()
    clean_name = name.replace(" ", "_")
    return clean_name


local_videos = env_vars["LOCAL_VID_DIR"]


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


# This function extracts the faces from the given video
def extract_faces(video_file, n_fps=3):
    """video_file : string, name of the video.
    n_fps : int, at what rate you want to extract the frames from the video.

    Returns the path at which all the extracted faces are being stored along with the video filename.
    """
    print("Reading the Video...")
    start = time.time()
    if not os.path.exists(f"{assets_path}/Face_detect/{video_file}/faces/"):
        os.makedirs(f"{assets_path}/Face_detect/{video_file}/faces/", exist_ok=True)
        print(local_videos)
        print("video file is", video_file)
        cap = cv2.VideoCapture(local_videos + "/" + video_file)

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
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count in frames_lst:
                try:
                    cv2.imwrite("frame.jpg", frame)
                    faces = DeepFace.extract_faces(
                        "frame.jpg",
                        detector_backend="retinaface",
                        enforce_detection=True,
                    )
                    image_area = frame.shape[0] * frame.shape[1]
                    for face in faces:
                        bbox = face["facial_area"]
                        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                        face_area = int(w * h)
                        tx = (x, y)
                        br = (int(x + w), int(y + h))
                        face_img = frame[tx[1] : br[1], tx[0] : br[0]]
                        cv2.imwrite(
                            f"{assets_path}/Face_detect/{video_file}/faces/Frame_{count}.jpg",
                            face_img,
                        )
                except Exception as e:
                    print(e)
                    pass
            count += 1
    else:
        pass
    out_dir = f"{assets_path}/Face_detect/{video_file}/faces/"
    end = time.time()
    print("Time took :", end - start)
    return out_dir, video_file


def preprocess_image(image_path):
    """image_path : string, path of the image for which you want to pre-process.

    Returns the image arrays which are ready to be fed into the model."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_to_model = np.expand_dims(img_array, axis=0)
    img_to_model = img_to_model / 255.0
    return img_array, img_to_model



@tf.function(jit_compile=False)
def predict_function(img_model):
    pred = model(img_model)
    print(pred)
    return pred


def filter_clear_faces(faces_dir, video_file, model):
    start = time.time()
    if not os.path.exists(
        f"{assets_path}/Face_detect/{video_file}/no_blur_faces/"
    ) and not os.path.exists(f"{assets_path}/Face_detect/{video_file}/blur_faces/"):
        os.makedirs(
            f"{assets_path}/Face_detect/{video_file}/no_blur_faces/", exist_ok=True
        )
        os.makedirs(
            f"{assets_path}/Face_detect/{video_file}/blur_faces/", exist_ok=True
        )
        imgs = [faces_dir + i for i in os.listdir(faces_dir)]
        if len(imgs) != 0:
            clear_faces = []
            for i in imgs:
                img, img_to_model = preprocess_image(i)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pred = predict_function(img_to_model)
                frame_num = i.split("Frame_")[-1].split(".")[0]
                if pred > 0.5:
                    clear_faces.append(i)
                    cv2.imwrite(
                        f"{assets_path}/Face_detect/{video_file}/no_blur_faces/Frame_{frame_num}.jpg",
                        img_rgb,
                    )
                else:
                    cv2.imwrite(
                        f"{assets_path}/Face_detect/{video_file}/blur_faces/Frame_{frame_num}.jpg",
                        img_rgb,
                    )
            clear_faces = [
                f"{assets_path}/Face_detect/{video_file}/no_blur_faces/" + i
                for i in os.listdir(
                    f"{assets_path}/Face_detect/{video_file}/no_blur_faces/"
                )
            ]
        else:
            clear_faces = []
    else:
        clear_faces = [
            f"{assets_path}/Face_detect/{video_file}/no_blur_faces/" + i
            for i in os.listdir(
                f"{assets_path}/Face_detect/{video_file}/no_blur_faces/"
            )
        ]
    end = time.time()
    print("Time took :", end - start)
    return clear_faces


embed_model = tf.keras.models.load_model(
    os.path.join(os.getcwd(), "get_faces", "embedfinalall.h5")
)


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (60, 60))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def get_embd(data_infer):
    img_embd = embed_model(data_infer)
    return img_embd


def extract_unqiue_faces(clear_faces, video_file):
    if not os.path.exists(f"{assets_path}/Face_detect/{video_file}/unique_faces/"):
        os.makedirs(
            f"{assets_path}/Face_detect/{video_file}/unique_faces/", exist_ok=True
        )
        if len(clear_faces) != 0:
            faces_to_remove = []
            for i in range(len(clear_faces)):
                if i not in faces_to_remove:
                    img1 = preprocess(clear_faces[i])

                    embed1 = get_embd(img1)
                    for j in range(i + 1, len(clear_faces)):
                        img2 = preprocess(clear_faces[j])
                        embed2 = get_embd(img2)
                        print(i, j)
                        result = DeepFace.verify(
                            clear_faces[i],
                            clear_faces[j],
                            detector_backend="skip",
                            model_name="ArcFace",
                        )
                        # result = findCosineDistance(embed1[0], embed2[0])
                        if result["distance"] <= 0.6:
                            faces_to_remove.append(j)
            for index in sorted(set(faces_to_remove), reverse=True):
                clear_faces.pop(index)

            for j in clear_faces:
                img = cv2.imread(j)
                frame_num = j.split("Frame_")[-1].split(".jpg")[0]
                timeline = round((int(frame_num) / fps), 2)
                cv2.imwrite(
                    f"{assets_path}/Face_detect/{video_file}/unique_faces/Frame_{frame_num}_time_{timeline}.jpg",
                    img,
                )
            clear_faces = [
                f"{assets_path}/Face_detect/{video_file}/unique_faces/" + i
                for i in os.listdir(
                    f"{assets_path}/Face_detect/{video_file}/unique_faces/"
                )
            ]

        else:
            clear_faces = []
    else:
        clear_faces = [
            f"{assets_path}/Face_detect/{video_file}/unique_faces/" + i
            for i in os.listdir(f"{assets_path}/Face_detect/{video_file}/unique_faces/")
        ]
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
            new_i = http_server_url + i.split(assets_path)[-1]
            timeline = new_i.split("_time_")[-1].split(".jpg")[0]
            temp_dict["img"] = new_i
            temp_dict["name"] = identity
            temp_dict["timeline"] = float(timeline)
            dict_elements.append(temp_dict)
        output_dict["faces"] = dict_elements
    else:
        output_dict = {}
        output_dict["faces"] = []
    with open(f"{assets_path}/Face_detect/{video_file}/api_response.json", "w") as file:
        json.dump(output_dict, file)
    return output_dict


def main_logic(video_file):
    print("Pipeline initiated.")
    if video_file.startswith("https:"):
        video_file = download_youtube_video(video_file)
        faces_dir, video_file = extract_faces(video_file)
        clear_faces = filter_clear_faces(faces_dir, video_file, model)
        unique_faces, video_file = extract_unqiue_faces(clear_faces, video_file)
        print(unique_faces)
        output_dict = name_faces(unique_faces, video_file)
    else:
        faces_dir, video_file = extract_faces(video_file)
        clear_faces = filter_clear_faces(faces_dir, video_file, model)
        unique_faces, video_file = extract_unqiue_faces(clear_faces, video_file)
        output_dict = name_faces(unique_faces, video_file)
    return output_dict
