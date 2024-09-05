from post_processers.post_process_base import PostProcessor
import cv2
from api_utils.utils import extract_backgrounds,remove_special_characters
from dotenv import dotenv_values
env_vars = dotenv_values(".env")
store_path=env_vars["ASSETS_DIR"]

class LLavaPostProcessor(PostProcessor):
    
    @staticmethod
    def postprocess(text_input, count_track, background_dir, fps, frames):
        background_results=[]
        for i,out_background in enumerate(text_input):
            count=count_track[i]
            background_dict={}
            data_background=out_background.split("\n")#text_reader_background.readlines()
            final_value=""
            if len(data_background)==1:
                original_values=data_background[0].split(",")
                final_value=extract_backgrounds(original_values)
            else:
                final_value=extract_backgrounds(data_background)        
            if len(final_value):
                skip_flag=0
                for value in final_value:
                    if len(value.split(" "))>3:
                        skip_flag=1
                if skip_flag:
                    continue       
                background_dict["info"]={"location":"# "+remove_special_characters(final_value[0]),"weather":"# "+remove_special_characters(final_value[1]),"sport":"# "+remove_special_characters(final_value[2])}
            else:
                background_dict["info"]={"location": "none" ,"weather": "none","sport": "none"}
            cv2.imwrite(f"{background_dir}/frame{count}.jpg",frames[i])
            backgroundpath=f"{background_dir}/frame{count}.jpg"
            background_dict["img"]=env_vars["HTTP_SERVER_URL"]+backgroundpath.split(store_path)[-1]
            background_dict["timeline"]=count/fps
            background_results.append(background_dict)
        return background_results
        