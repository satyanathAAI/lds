from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import os
import json
import base64
import requests
import time
import jsonpickle
import boto3
from pymongo import MongoClient
from pymongo.ssl_support import CLIENT_CERTIFICATE_KEY_FILE, CA_FILE


from urllib.parse import urlparse
s3 = boto3.client('s3')
def extract_bucket_and_key(s3_url):
    # Parse the S3 URL
    parsed_url = urlparse(s3_url)
    # Extract the object key from the path
    object_key = parsed_url.path.lstrip('/')
    
    return object_key


def check_cache(self, filename) :
        db = self.mongoclient[self.dbname]
        collection = db[self.collection]
        search_query = {"filename": filename}

        # Define the projection to return only the value of the filename field
        projection = {"_id": 0}

        # Find the document with the specified filename and project the result
        result = collection.find_one(search_query, projection)
        if result is not None:

            return result, True
        else:
            return None, False

def insert_to_mongo(self, record,filename):
    db = self.mongoclient[self.dbname]
    collection = db[self.collection]
    record_background = jsonpickle.dumps(record)
    record_background["filename"]=filename
    record_to_insert = record_background
    try:
        collection.insert_one(record_to_insert)
        print("record inserted")
    except Exception as e:
        print(f"Error inserting to mongo collection {self.collection}")


# environment : # Need pip install openai====0.28
# from utils import ( get_transcript_from_audio_using_openai, 
#                     create_directories,prompt_get, 
#                     get_text_summary_and_tags,
#                     create_folder_save_files_in_main_output_folder,
#                     send_to_Rag )

from utils import ( get_transcript_from_audio_using_openai, get_text_summary_and_tags_bedrock,
                    create_directories,prompt_get, 
                    get_text_summary_and_tags,
                    create_folder_save_files_in_main_output_folder,
                    send_to_Rag )


from folder_config import ( directories_to_create,
                            Audio_folder,
                            Audio_transcripts_folder,
                            Audio_json_folder )

from constants import ( Augument_open_api_key, 
                        global_output_fodler_path)

# print('imports is done')

# Variable to store transcript from audio.
audio_transcript = ''
empty_json = {
    "ip_details": "",
    "summary": "",
    "tags": []
}

# global_output_fodler_path = "/home2/asgtestdrive2023/Project/AI-Team/assetiq-church/outputs/OUTPUT/Audio"
# search_rag_api = 'http://10.10.0.212:7175/file_upload_to_search_rag_KB'

app = Flask(__name__)
CORS(app)
from io import BytesIO
from pydub import AudioSegment
@app.route("/audio_upload", methods=["POST"])
def upload_file():
    # global audio_transcript
    
    data = request.get_json()
    # encoded_file_data = data['encoded_data']
    # filename = data['file_name']

    s3_url= data['s3_url']
# The URL of the audio file

# Download the audio file
    response = requests.get(s3_url)
    response.raise_for_status()

# Load the audio file into pydub's AudioSegment
    audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
    filename = extract_bucket_and_key(s3_url)
    # Save the audio file to the local file system
    output_file_path = f"/home/ubuntu/lds/lds_audio/Audio_cache/Output/{filename}"
    audio.export(output_file_path, format="mp3")
    
    # Create a directories for saving videos and audio and trascript files
    create_directories(directories_to_create)

    # Get the file name and extension
    file_name_without_extension, extension = os.path.splitext(filename)

    # store in local
    # audio_saved_path = os.path.join(Audio_folder, filename)
    audio_saved_path = f"/home/ubuntu/lds/lds_audio/Audio_cache/Output/{filename}"
    audio_json_saved_path = os.path.join(Audio_json_folder, f'{file_name_without_extension}.json')
    audio_transcript_saved_path = os.path.join(Audio_transcripts_folder, f'{file_name_without_extension}.txt')
    s3.upload_file(f'/home/ubuntu/lds/lds_audio/Audio_cache/Output/{filename}','testlds' , f'Output_audio/{filename}')
    server_file_path  = f'https://testlds.s3.amazonaws.com/Output_audio/{filename}'
    # server to host images to display 
    
    # store in Main folder
    main_folder_audio_transcript_saved_path = f"{global_output_fodler_path}/{file_name_without_extension}/{file_name_without_extension}_transcript.txt"
    main_folder_audio_json_saved_path = f"{global_output_fodler_path}/{file_name_without_extension}/{file_name_without_extension}_json.json"
    main_folder_audio_summary_saved_path = f"{global_output_fodler_path}/{file_name_without_extension}/{file_name_without_extension}_summary.txt"
    
    if extension in ['.mp3','.wav']:
        
        #search in document db with filename and return the data
        # if os.path.exists(main_folder_audio_transcript_saved_path) and os.path.exists(main_folder_audio_json_saved_path) and os.path.exists(main_folder_audio_summary_saved_path):
            
        #     print('files already existed')
        #     with open(main_folder_audio_json_saved_path, "r") as f:
        #         json_data = json.load(f)
        #     print('fetched json file from folder successfully')
        #     # send to rag
        #     # send_to_Rag(main_folder_audio_summary_saved_path, file_name_without_extension, 'summary' , search_rag_api)
        #     # send_to_Rag(main_folder_audio_transcript_saved_path, file_name_without_extension, 'transcript' , search_rag_api)
        #     return json_data s     
        json_data,search=check_cache(filename)
        if search is True:
            json_data['filename'] = filename
            return json_data
        else:
            try:
                # Get Transcript from Audio using Open Ai
                audio_transcript = get_transcript_from_audio_using_openai(audio_saved_path, Augument_open_api_key)
                if audio_transcript is None:
                    print('audio_transcript not given by open ai')
                    empty_json['path']=server_file_path
                    return empty_json   
                else:  
                    prompt1 = prompt_get(audio_transcript)
                    answer = get_text_summary_and_tags(Augument_open_api_key,prompt1)
                    # answer = get_text_summary_and_tags_bedrock(prompt1)
                    if answer is not None:
                        # save trancript in local
                        with open(audio_transcript_saved_path, "w") as f:
                            f.write(audio_transcript)
                        # get json
                        answer = answer[7:-3]
                        json_data = json.loads(answer)
                        json_data['path'] = server_file_path
                        
                        # save json
                        # insert n document db audio collection with filename
                        # with open(audio_json_saved_path, "w") as f:
                        #     json.dump(json_data, f, indent=4)
                        insert_to_mongo(json_data,filename)   
                        # send files to main/global folder json , transcript and summary
                        # create_folder_save_files_in_main_output_folder(file_name_without_extension, 
                        #                                                global_output_fodler_path, 
                        #                                                json_data, 
                        #                                                audio_transcript, 
                        #                                                search_rag_api)
                        print('Whole process complete')
                        return json_data
                    else:
                        print('error while getting tags and sumary form open ai')
                        empty_json['path']=server_file_path
                        return empty_json
            except Exception as e:
                print(e)   
                print('error')
                empty_json['path']=server_file_path
                return empty_json
    else:
       return 'We currently support MP3 (.mp3), Wav file(.wav).'

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=1112,debug=False)


