from post_processers.post_process_base import PostProcessor
from api_utils.utils import extract_json
from dotenv import dotenv_values
import cv2

env_vars = dotenv_values(".env")


class OpenAIPostProcessor(PostProcessor):
    @staticmethod
    def postprocess(text_input, count_track, background_dir, fps, frames):
        background_results = []

        for i, out_background in enumerate(text_input):
            count = count_track[i]
            background_dict = extract_json(out_background)
            print("background dict is", background_dict)
            cv2.imwrite(f"{background_dir}/frame{count}.jpg", frames[i])
            backgroundpath = f"{background_dir}/frame{count}.jpg"
            background_dict["img"] = (
                env_vars["HTTP_SERVER_URL"] + backgroundpath.split(store_path)[-1]
            )
            background_dict["timeline"] = count / fps
            background_results.append(background_dict)
        return background_results
