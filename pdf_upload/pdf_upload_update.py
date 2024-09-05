# Import basic python packages
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    send_from_directory,
    send_file,
    jsonify,
    flash,
    render_template,
    g,
)
from api_utils.mongo_utils import insert_to_mongo,check_cache
import datetime, time


import json
from flask_cors import CORS
import base64

import requests
import re
import boto3
from lds_audio.Audio_cache.utils import get_text_summary_and_tags_bedrock
# Initialize the Textract client
textract = boto3.client('textract',region_name="us-east-1")


def extract_json(text):
    summary_match = re.search(r'"summary": "(.*?)"', text, re.DOTALL)
    summary = summary_match.group(1) if summary_match else None

    # Extract the tags using regex
    tags_match = re.search(r'"tags": \[(.*?)\]', text, re.DOTALL)
    tags_string = tags_match.group(1) if tags_match else None

    # Convert the tags string into a list
    tags = [tag.strip().strip('"') for tag in tags_string.split(',')] if tags_string else []

    # Create the output dictionary
    output = {
        "summary": summary,
        "tags": tags
    }
    return output

def start_document_text_detection(bucket_name, document_name):
    # Start the asynchronous text detection job
    response = textract.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': document_name
            }
        }
    )
    return response['JobId']

def get_document_text_detection(job_id):
    # Retrieve the results of the text detection job
    # max_tries = 10
    # next_token = None
    max_tries = 60  # Increased number of retries
    wait_time = 10  # Increased wait time between retries (in seconds)
    next_token = None
    full_text = ""

    while max_tries > 0:
        if next_token:
            response = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
        else:
            
            response = textract.get_document_text_detection(JobId=job_id)
           

        # Check if the job is completed
        status = response['JobStatus']
        print(status)
        if status in ['SUCCEEDED', 'FAILED']:
            if status == 'SUCCEEDED':
                # Process and accumulate text from the response
                for block in response['Blocks']:
                    if block['BlockType'] == 'LINE':
                        full_text += block['Text'] + "\n"
                print("\n\n",full_text)
                return full_text

                # Check if there are more pages of results
                # next_token = response.get('NextToken', None)
                # if not next_token:
                #     return full_text
            else:
                raise Exception(f"Job {job_id} failed with status: {status}")

        # Wait before retrying
        time.sleep(wait_time)
       





app = Flask(__name__)
CORS(app)



@app.route("/file_upload", methods=["POST"])
def file_path():
    # logging.info("Entered into file_upload function from app_integration.py")

    #for encoded_data
    data = request.get_json()
    # encoded_file_data = data['encoded_data']
    filename = data['file_name']
    fileurl=f"https://lds-test-public.s3.amazonaws.com/{filename}"
    bucket_name="lds-test-public"
    result,status=check_cache(videoname=filename,collection_name="pdf")
    print(result,status)
    if status:
        return result
    
    job_id = start_document_text_detection(bucket_name,filename)
    full_text = get_document_text_detection(job_id)
    prompt = f"""Please provide a concise summary and at least 20 relevant tags for the following text. The output should be in JSON format with keys 'summary' and 'tags'. The 'tags' should be an array of relevant keywords or phrases that capture the main themes and topics of the text.
                        Text: \"{full_text[:188000]}\"
                        """
    response=get_text_summary_and_tags_bedrock(prompt=prompt)
    response_json=extract_json(response)
    response_json["file"]=fileurl
    response_json["filename"]=filename
    insert_to_mongo(response_json,"pdf")
    del response_json["filename"]
    try:
        del response_json["_id"]
    except:
        pass
    return response_json



    

    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7173, debug=False)
