# from moviepy.editor import VideoFileClip
import os
import openai
import requests
import base64
import json
import time
import boto3


# functions --------------------------------------------------------
def create_directories(list_of_directories):
    for dir in list_of_directories:
        os.makedirs(dir, exist_ok=True)

# ----------------------------------------------------------------------------
def get_transcript_from_audio_using_openai(audio_file_path, open_api_key):
    try:
        # Set up your OpenAI API key
        openai.api_key = open_api_key
        # Open the audio file
        audio_file = open(audio_file_path, "rb")
        # Create the transcription
        transcription = openai.Audio.transcribe(
            model="whisper-1", file=audio_file, response_format="text"
        )
        return transcription
    except Exception as e:
        print(e)
        return None
        
# --------------------------------------------------------------------------
def prompt_get(text):
    prompt = f"""
    Please provide a concise summary and at least 20 relevant tags for the following text. The output should be in JSON format with keys 'summary' and 'tags'. The 'tags' should be an array of relevant keywords or phrases that capture the main themes and topics of the text.
 
    Text: \"{text}\"
    """
    return prompt
    
# ---------------------------------------------------------------------------
def get_text_summary_and_tags(api_key, prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
 
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2048,
        "temperature": 0
    }
 
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return None
    
def  get_text_summary_and_tags_bedrock(prompt):
    # Initialize the Bedrock client
   
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    # Set the model ID, e.g., Llama 3 8b Instruct.
    model_id = "meta.llama2-70b-chat-v1"

    # Define the prompt for the model.

    # Embed the prompt in Llama 3's instruction format.


    # Format the request payload using the model's native structure.
    native_request = {
        "prompt": prompt ,
        "max_gen_len": 512,
        "temperature": 0.5,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["generation"]
    return response_text
    
        
# --------------------------------------------------------
def send_to_Rag(filepath, file_name_without_extension, typee_of_Text, search_rag_api):
    try:
        with open(filepath, 'rb') as file:
            # Prepare the files dictionary for the request
            files = {
                'file': (f'{file_name_without_extension}_{typee_of_Text}.txt', file)
            }
    
            try:
                response = requests.post(search_rag_api, files=files)
                if response.status_code == 200:
                    print("File uploaded successfully to RAG.")
                else:
                    print(f"Failed to upload file. Status code: {response.status_code}")
                response.close()
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
        
# ---------------------------------------------------------------
def create_folder_save_files_in_main_output_folder(file_name_without_extension, output_folder, getted_json, audio_transcript, search_rag_api):

    audio_summary_transcript_folder = f"{output_folder}/{file_name_without_extension}"
    os.makedirs(audio_summary_transcript_folder, exist_ok=True)

    # save summary and transcript in .txt
    path_to_save_summary = f"{audio_summary_transcript_folder}/{file_name_without_extension}_summary.txt"
    if not os.path.exists(path_to_save_summary):
        with open(path_to_save_summary, "w") as f:
            f.write(getted_json["summary"])
    else:
        print('this file named summary already available in main output folder to this file.')

    path_to_save_transcript = f"{audio_summary_transcript_folder}/{file_name_without_extension}_transcript.txt"
    if not os.path.exists(path_to_save_transcript):
        with open(path_to_save_transcript, "w") as f:
            f.write(audio_transcript)
    else:
        print('this file named transcript already available in main output folder to this file.')

    path_to_save_json = f"{audio_summary_transcript_folder}/{file_name_without_extension}_json.json"
    if not os.path.exists(path_to_save_json):
        with open(path_to_save_json, "w") as f:
            json.dump(getted_json, f, indent=4)
    else:
        print('this file named json already available in main output folder.')

    # send_to_Rag(path_to_save_summary, file_name_without_extension, 'summary' , search_rag_api)
    # # time.sleep(15)
    # send_to_Rag(path_to_save_transcript, file_name_without_extension, 'transcript' , search_rag_api)

#  ---------------------------------------------------------------------------------------------------
    
# def update_to_rag_summary(file_name_without_extension, summary_nd_tags_json, search_rag_api):
#     try:
#         base64_encoded_summary = base64.b64encode(summary_nd_tags_json['summary'].encode("utf-8")).decode("utf-8")
#         payload = {
#             "file_name" : f'{file_name_without_extension}_summary.txt',
#             "encoded_data" : base64_encoded_summary
#         }
#         response = requests.post(search_rag_api, json = payload)
#         # Check if the request was successful
#         if response.status_code == 200:
#             print('Data sent to RAG API successfully')
#         else:
#             print(f'Failed to send data to RAG API. Status code: {response.status_code}')
#         response.close()
#     except Exception as e:
#         print(e)

# def update_to_rag_transcript(file_name_without_extension, transcript, search_rag_api):
#     try:
#         base64_encoded_summary = base64.b64encode(transcript.encode("utf-8")).decode("utf-8")
#         payload = {
#             "file_name" : f'{file_name_without_extension}_transcript.txt',
#             "encoded_data" : base64_encoded_summary
#         }
#         response = requests.post(search_rag_api, json = payload)
#         # Check if the request was successful
#         if response.status_code == 200:
#             print('Data sent to RAG API successfully')
#         else:
#             print(f'Failed to send data to RAG API. Status code: {response.status_code}')
#         response.close()
#     except Exception as e:
#         print(e)