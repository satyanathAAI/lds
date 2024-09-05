import json
import jsonpickle
from typing import List,Dict
import random
import string
import numpy as np
import cv2
import re
import pytesseract
from pytesseract import Output
import os
from post_processers.post_process_base import PostProcessor
import time

def generate_random_str()-> str:
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res


def remove_black(image,boxes)-> bool:

    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([70,70,70], dtype = "uint16")
    black_mask = cv2.inRange(image, lower_black, upper_black)
    for box in boxes:
        black_mask[box[0]:box[1],box[2]:box[3]]=255.
    ratio=np.sum(sum(black_mask==255.))/(black_mask.shape[0]*black_mask.shape[1])
    if ratio>0.98:
        return True
    else:
        return False

def remove_special_characters(input_string: str)-> str:
    # Define a regular expression pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'  # This pattern matches anything that is not alphanumeric or whitespace

    # Use re.sub() to replace the matched characters with an empty string
    clean_string = re.sub(pattern, '', input_string)
    if "not applicable" in clean_string:
        clean_string="none"

    return clean_string

def remove_white(image,boxes)-> bool:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,168])
    upper_white = np.array([172,111,255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    for box in boxes:
        mask[box[0]:box[1],box[2]:box[3]]=255.
    ratio=np.sum(sum(mask==255.))/(mask.shape[0]*mask.shape[1])
    if ratio>0.98:
        return True
    else:
        return False

def check_random(img):
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    c_boxes=[]
    for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            box_ratio=(d['width'][i]*d['height'][i])/(img.shape[0]*img.shape[1])
            if (d["conf"][i] != -1) and (box_ratio<0.8):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                c_boxes.append([y,y+h,x,x+w])
    skip_flag_black=remove_black(img,c_boxes)
    skip_flag_white=remove_white(img,c_boxes)
    final_value=(skip_flag_black or skip_flag_white)
    return final_value

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def mainpulate_background_vals(val:str)->str:
    val=val.replace(","," ")
    if ("dark" in val) and ("sport" in val):
        val=val.replace("dark","none")
    value=val.split(":")[-1]
    if "not possible" in value:
        value="none"
    return value

def extract_backgrounds(values):
    split_values=[mainpulate_background_vals(val) for val in values]
    final_str=[split.strip() for split in split_values]
    return final_str

def create_dir(path):
    print("path is :",path)
    if not os.path.exists(path):
        os.makedirs(path)
        print("path created",path)

def extract_json(response_openai:str):
    response_openai=response_openai.replace("json","")
    response_openai=response_openai.replace("```","")
    json_result_openai=jsonpickle.loads(response_openai)
    print("json api out",json_result_openai)
    return json_result_openai

def post_process_model_output(out_background:List,model_params:List,processor:PostProcessor)-> Dict:
        counts,background_dir,fps,frames=model_params
        start_time=time.time()
    # background_results=LLavaPostProcessor.postprocess(out_background,counts,background_dir,fps,frames)
        background_results=processor.postprocess(out_background,counts,background_dir,fps,frames)
        print("background",background_results)
        
        print("time taken for post processing of background is :",time.time()-start_time)
        
        return{"background":background_results}