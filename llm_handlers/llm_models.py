from llm_handlers.llm_infer_base import LlmHandler
from typing import Dict, List
import numpy as np

import base64


import concurrent.futures
from dotenv import dotenv_values
from api_utils.utils import generate_random_str
import jsonlines
import time
import os
import boto3
import json
import ast
import re

env_vars = dotenv_values(".env")

llava_sagemaker_endpoint = env_vars["SAGEMAKER_LLAVA_ENDPOINT"]


class LlavaHandler(LlmHandler):

    def __init__(self, url: str, modelname: str) -> None:
        self.url = url
        self.modelname = modelname

    def get_client(self):
        return httpclient.InferenceServerClient(url=self.url)

    def get_llm_params(self) -> Dict:
        return {"url": self.url, "modelname": self.modelname}

    def preproces_input(
        self, formated_prompt_list: List[str], filename_list: List[str]
    ):
        text_obj = np.array(formated_prompt_list, dtype="object").reshape((-1, 1))
        path_object = np.array(filename_list, dtype="object").reshape((-1, 1))
        input_text = httpclient.InferInput(
            "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        )
        input_path = httpclient.InferInput(
            "path", path_object.shape, np_to_triton_dtype(path_object.dtype)
        )
        input_path.set_data_from_numpy(path_object)
        input_text.set_data_from_numpy(text_obj)
        output_text = httpclient.InferRequestedOutput("generated_text")
        return input_text, input_path, output_text

    def format_prompt(self, prompt_list: List[str]) -> List[str]:
        return prompt_list

    def call_llm(self, prompt_list: List[str], filename_list: List[str]) -> List[str]:
        modelparams = self.get_llm_params()
        client = self.get_client()
        input_text, input_path, output_text = self.preproces_input(
            formated_prompt_list=prompt_list, filename_list=filename_list
        )
        query_response = client.infer(
            model_name=modelparams["modelname"],
            inputs=[input_text, input_path],
            outputs=[output_text],
        )
        out_response = query_response.as_numpy("generated_text")
        return [out.decode("utf-8") for out in out_response]


class OpenAIModel(LlmHandler):
    url = "/v1/chat/completions"

    def __init__(self, modelname: str, api_key: str):
        self.modelname = modelname
        self.api_key = api_key

    def get_client(self):
        return OpenAI(api_key=self.api_key)

    def get_llm_params(self):
        return {"modelname": self.modelname, "apikey": self.api_key}

    def format_prompt(self, prompt_list):
        pass

    @staticmethod
    def encode_image(image_path: str) -> List:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def convert_to_base64(file_paths: List[str]):
        base64_images = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(OpenAIModel.encode_image, file_paths)
            base64_images = list(results)

        return base64_images

    def batch_preprocess_input(
        self, formated_prompt_list: List[str], filename_list: List[str]
    ) -> List:
        base64_imgs = OpenAIModel.convert_to_base64(file_paths=filename_list)
        list_messages = [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds in json format. Help me getting background and loction in the image!",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{formated_prompt_list[idx]}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base_str}"},
                        },
                    ],
                },
            ]
            for idx, base_str in enumerate(base64_imgs)
        ]

        return list_messages

    def create_jsonlfile(self, list_messages: List) -> str:
        requests_batch = [
            {
                "custom_id": generate_random_str(),
                "method": "POST",
                "url": OpenAIModel.url,
                "body": {
                    "model": self.modelname,
                    "messages": message,
                    "temperature": 0.0,
                },
            }
            for message in list_messages
        ]
        write_path = os.path.join(jsonl_dir, f"{generate_random_str()}.jsonl")
        print("len of request batch", len(requests_batch))
        with jsonlines.open(write_path, mode="w") as writer:
            writer.write_all(requests_batch)
        return write_path

    def upload_n_create_batch(self, jsonl_path: str):
        batch_input_file = self.client.files.create(
            file=open(jsonl_path, "rb"), purpose="batch"
        )
        batch_input_file_id = batch_input_file.id

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "nightly eval job"},
        )
        return batch_object

    def retreive_results(self, output_file_id):

        content = self.client.files.content(output_file_id)
        return content

    def preproces_input(
        self, formated_prompt_list: List[str], filename_list: List[str]
    ) -> List:
        baseimagelist = OpenAIModel.encode_image(filename_list[0])
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in json format. Help me getting background and loction in the image!",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{formated_prompt_list[0]}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{baseimagelist}"},
                    },
                ],
            },
        ]

        return messages

    def check_status_batch(self, batch_object) -> bool:
        print(self.client.batches.retrieve(batch_id=batch_object.id))
        if batch_object.output_file_id is not None:
            return True
        else:
            return False

    def call_llm(self, prompt_list, filename_list):
        client = self.get_client()
        messages = self.preproces_input(
            formated_prompt_list=prompt_list, filename_list=filename_list
        )
        modelparams = self.get_llm_params()
        modelname = modelparams["modelname"]
        response = client.chat.completions.create(
            model=modelname,
            messages=messages,
            temperature=0.0,
        )
        return [response.choices[0].message.content]

    def batch_call_llm(self, prompt_list, filename_list):
        start_time = time.time()
        list_messages = self.batch_preprocess_input(
            formated_prompt_list=prompt_list, filename_list=filename_list
        )
        print("time taken for batch prepocesing is ", time.time() - start_time)
        self.client = self.get_client()
        jsonpath = self.create_jsonlfile(list_messages=list_messages)
        batch_object = self.upload_n_create_batch(jsonl_path=jsonpath)
        print("batch object is ", batch_object)
        for counter in range(10):
            print(f"checking status for {counter} time")

            status = self.check_status_batch(batch_object)
            if status:
                break
            time.sleep(1)
        output = self.retreive_results(output_file_id=batch_object.output_file_id)
        print("output is ", output)
        return output


class OpenAIModelText(LlmHandler):
    def __init__(self, modelname: str, apikey: str) -> None:
        self.modelname = modelname
        self.api_key = apikey

    def get_llm_params(self):
        pass

    def get_client(
        self,
    ):
        return OpenAI(api_key=self.api_key)

    def format_prompt(self, prompt: str, transcript: str) -> str:
        formatted_prompt = f"{prompt}\n Transcript : {transcript}"
        return formatted_prompt

    def preproces_input(self, formated_prompt: str):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in json format. Help me getting tags from a transcript of christian community video!",
            },
            {"role": "user", "content": f"{formated_prompt}"},
        ]

        return messages

    def call_llm(self, prompt: str, transcript: str):
        client = self.get_client()
        fromatted_prompt = self.format_prompt(prompt=prompt, transcript=transcript)

        messages = self.preproces_input(formated_prompt=fromatted_prompt)
        response = client.chat.completions.create(
            model=self.modelname,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content


class OpenAIModelFrameSummarizer(OpenAIModel):
    def preproces_input(
        self, formated_prompt_list: List[str], filename_list: List[str]
    ) -> List:
        baseimagelist = super().encode_image(filename_list[0])
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in json format. Help me getting summary of Image!",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{formated_prompt_list[0]}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{baseimagelist}"},
                    },
                ],
            },
        ]

        return messages


class OpeanAIModelTextSummarizer(OpenAIModelText):

    def format_prompt(self, prompt: str, transcript: str, summary: str) -> str:
        if not len(transcript):
            formatted_prompt = f"{prompt}\n Video Content : {summary}"
        else:
            formatted_prompt = f"{prompt}\n Transcript of Video: {transcript}\n Video Content Generated From Processing Sequence of Frames from Video: {summary}"

        return formatted_prompt

    def preproces_input(self, formated_prompt: str):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Help me in summarizing the entire Video Content !",
            },
            {"role": "user", "content": f"{formated_prompt}"},
        ]

        return messages

    def call_llm(self, prompt: str, transcript: str, summary: str):
        client = self.get_client()
        fromatted_prompt = self.format_prompt(
            prompt=prompt, transcript=transcript, summary=summary
        )

        messages = self.preproces_input(formated_prompt=fromatted_prompt)
        response = client.chat.completions.create(
            model=self.modelname,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content


class LlavaSagemaker(LlmHandler):
    def __init__(self) -> None:
        super().__init__()

    def get_client(self):
        return boto3.client("runtime.sagemaker", region_name="us-east-1")

    def get_llm_params(self) -> Dict:
        pass

    def preproces_input(
        self, formated_prompt_list: List[str], filename_list: List[str]
    ):
        data = {
            "image": filename_list[0],
            "question": formated_prompt_list[0],
            # "max_new_tokens" : 1024,
            # "temperature" : 0.2,
            # "stop_str" : "###"
        }

        # Convert the input data to JSON format

        return json.dumps(data)

    def format_prompt(self, prompt_list: List[str]) -> List[str]:
        return prompt_list

    def call_llm(self, prompt_list: List[str], filename_list: List[str]) -> List[str]:

        client = self.get_client()

        payload = self.preproces_input(
            formated_prompt_list=prompt_list, filename_list=filename_list
        )

        response = client.invoke_endpoint(
            EndpointName=llava_sagemaker_endpoint,
            ContentType="application/json",  # Adjust content type if necessary
            Body=payload,
        )

        # Read and decode the response
        decoded_response = response["Body"].read().decode("utf-8")

        out_result = decoded_response.replace("\n", "").replace("\\_", "_")
        out_result = re.sub(r"\\_", "_", out_result)

        # Convert the cleaned string to a JSON object
        print("out_result is :", out_result)
        out_result = json.loads(out_result)
        out_result = ast.literal_eval(out_result)

        print("json_object...", type(out_result))

        print("Response from SageMaker endpoint:")
        return out_result
