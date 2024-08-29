from flask import Flask, request
from flask_cors import CORS
import os
import json
import base64

from constants import Augument_open_api_key, search_rag_api, global_output_fodler_path
from utils import ( get_summary_from_gpt4turbo,
                    prompt2,
                    create_folder_save_files_in_main_output_folder,
                    send_to_rag
                   )

empty_json = {
    "ip_details": "", 
    "summary": "", 
    "tags": []
}

app = Flask(__name__)
CORS(app)


@app.route("/image_upload", methods=["POST"])
def main():
    data = request.get_json()
    encoded_file_data = data["encoded_data"]
    filename = data["file_name"]

    # Decode the Base64 data
    file = base64.b64decode(encoded_file_data)

    global empty_json

    os.makedirs("images", exist_ok=True)
    os.makedirs("images_json", exist_ok=True)

    # Get the file name and extension
    file_name_without_extension, extension = os.path.splitext(filename)
    
    # server to host images to display
    server_file_path = f"http://10.10.0.212:1122/images/{filename}"
    
    # store in local
    img_saved_path = os.path.join("images", filename)
    img_json_saved_path = os.path.join("images_json", f"{file_name_without_extension}.json")
    
    # store in Main folder
    main_folder_img_summary_saved_path = f"{global_output_fodler_path}/{file_name_without_extension}/{file_name_without_extension}_summary.txt"
    main_folder_img_json_saved_path = f"{global_output_fodler_path}/{file_name_without_extension}/{file_name_without_extension}_json.json"

    # We currently support PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif).
    if extension in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        if os.path.exists(main_folder_img_summary_saved_path) and os.path.exists(main_folder_img_json_saved_path): 
            print("files already existed")
            with open(main_folder_img_json_saved_path, "r") as f:
                json_data = json.load(f)
            print("fetched json file from Main OUTPUT folder successfully")
            # send to rag
            send_to_rag(main_folder_img_summary_saved_path, file_name_without_extension, search_rag_api)
            return json_data
        else:
            print("its a new image, getting tags and summary from open ai started")
            # save image in folder
            with open(img_saved_path, "wb") as f:
                f.write(file)
            try:
                answer = get_summary_from_gpt4turbo(img_saved_path, Augument_open_api_key, prompt2)
                if answer is None:
                    print("error while getting tags and sumary form open ai")
                    empty_json["path"] = server_file_path
                    return empty_json
                else:
                    answer = answer[7:-3]
                    # Parse the JSON string into a Python dictionary
                    json_data = json.loads(answer)
                    json_data["path"] = server_file_path
                    # Saving Json
                    with open(img_json_saved_path, "w") as f:
                        json.dump(json_data, f, indent=4)

                    # send files to main folder json and summary
                    create_folder_save_files_in_main_output_folder(file_name_without_extension, global_output_fodler_path, json_data, search_rag_api)
                    print("completed the process")
                    return json_data    
            except Exception as e:
                print(e)
                print("error while getting tags and sumary form open ai")
                empty_json["path"] = server_file_path
                return empty_json
    else:
        return "We currently support PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif)."

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=1111)
