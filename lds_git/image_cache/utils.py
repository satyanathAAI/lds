import base64
import requests
import os
import base64
import json
import boto3

sagemaker_runtime = boto3.client("runtime.sagemaker", region_name="us-east-1")

# Your SageMaker endpoint name (extracted from ARN)
# endpoint_name = "huggingface-pytorch-inference-2024-08-19-11-02-01-018"
endpoint_name='huggingface-pytorch-inference-2024-08-22-04-39-53-690'

prompt = """Please analyze the given image and provide the following details:
1. Summary: A brief description of the image, including any notable elements or themes, especially if it is religious in nature.
2. Intellectual Property (IP): If there is any intellectual property information (such as copyright notices, watermarks, or trademarks) imprinted on the image, extract and provide that information. If there is no IP information, state "no".
3. Tags: Generate a minimum of 5 relevant tags that capture the theme and key elements of the image.
Output the response in JSON format with the following keys: 'Summary', 'IP', 'Tags'."""

prompt1 = """Please analyze the given image and provide the following details:
1. Summary: A brief description of the image, including any notable elements or themes, especially if it is religious in nature.
2. Intellectual Property (IP): If there is any intellectual property information (such as copyright notices, watermarks, or trademarks) imprinted on the image, extract and provide that information. If there is no IP information, state "no".
3. Tags: Generate a minimum of 5 relevant tags that capture the theme and key elements of the image.
Output the response in JSON format with the following keys: 'summary', 'ip_details', 'tags'."""

prompt2 = """Please analyze the given image and provide the following details:
1. Summary: A brief description of the image, including any notable elements or themes, especially if it is religious in nature.
2. Intellectual Property (IP): If there is any intellectual property information (such as copyright notices, watermarks, or trademarks) imprinted on the image, extract and provide that information. If there is no IP information, state "no".
3. Tags: Generate a minimum of 5 relevant tags that capture the image type, theme, and key elements of the image.
Output the response in JSON format with the following keys: 'summary', 'ip_details', 'tags'."""


# Function to encode the image
def encode_image(image_path):
    # print(image_path)
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(e)
        print(
            f"error at encode IMAGE - No Summary getting for this image -------> {image_path}"
        )
        return None


import re
import ast


def get_summary_from_sagemaker(s3_url, api_key, prompt):
    # Getting the base64 string

    # Prepare input data for the model
    # Modify according to the input format expected by your model

    data = {
        "image": s3_url,
        "question": prompt,
        # "max_new_tokens" : 1024,
        # "temperature" : 0.2,
        # "stop_str" : "###"
    }
    # Convert the input data to JSON format
    payload = json.dumps(data)

    try:
        # Invoke the SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",  # Adjust content type if necessary
            Body=payload,
        )

        # Read and decode the response
        decoded_response = response["Body"].read().decode("utf-8")

        out_result = decoded_response.replace("\n", "").replace("\\_", "_")
        out_result = re.sub(r"\\_", "_", out_result)

        # Convert the cleaned string to a JSON object
        out_result = json.loads(out_result)
        out_result = ast.literal_eval(out_result)

        print("json_object...", type(out_result))

        print("Response from SageMaker endpoint:")
        return out_result
    except Exception as e:
        print(f"An error occurred: {e}")


def get_summary_from_gpt4turbo(image_path, api_key, prompt):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    if base64_image is not None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0,
        }

        try:
            # response = requests.post(
            #     "https://api.openai.com/v1/chat/completions",
            #     headers=headers,
            #     json=payload,
            # )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            return None
    else:
        return None


def send_to_rag(img_summary_saved_path, file_name_without_extension, search_rag_api):
    try:
        with open(img_summary_saved_path, "rb") as file:
            # Prepare the files dictionary for the request
            files = {"file": (f"{file_name_without_extension}_summary.txt", file)}

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


def create_folder_save_files_in_main_output_folder(
    file_name_without_extension, output_folder, getted_json, search_rag_api
):

    image_summary_folder = f"{output_folder}/{file_name_without_extension}"
    os.makedirs(image_summary_folder, exist_ok=True)

    # save summary in .txt
    img_summary_saved_path = (
        f"{image_summary_folder}/{file_name_without_extension}_summary.txt"
    )
    img_json_saved_path = (
        f"{image_summary_folder}/{file_name_without_extension}_json.json"
    )

    if not os.path.exists(img_summary_saved_path):
        with open(img_summary_saved_path, "w") as f:
            f.write(getted_json["summary"])
    else:
        print("img_summary already available in main output folder to this file.")

    if not os.path.exists(img_json_saved_path):
        with open(img_json_saved_path, "w") as f:
            json.dump(getted_json, f, indent=4)
    else:
        print("img_json already available in main output folder to this file.")

    send_to_rag(img_summary_saved_path, file_name_without_extension, search_rag_api)


# ------------------------------------------------------------------------------------
# send summary to rag
# with open(img_summary_saved_path, 'rb') as file:
#     # Prepare the files dictionary for the request
#     files = {
#         'file': (f'{file_name_without_extension}_summary.txt', file)
#     }

#     try:
#         response = requests.post(search_rag_api, files=files)
#         if response.status_code == 200:
#             print("File uploaded successfully to RAG.")
#         else:
#             print(f"Failed to upload file. Status code: {response.status_code}")
#         response.close()
#     except Exception as e:
#         print(e)
# --------------------------------------------------------------------------------------
