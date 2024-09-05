import json
import time
from typing import Dict
import os
import av
import time
import cv2
from api_utils.utils import *
from dotenv import dotenv_values
from llm_handlers.llm_models import LlavaSagemaker
from pipelines.aws_pipeline import AwsPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

# from post_processers.llava_postprocessor import LLavaPostProcessor
from post_processers.openai_postprocessor import OpenAIPostProcessor

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoInput(BaseModel):
    videourl: str


# Load all the necessary environment variables from  .env file
CLUSTER_ENDPOINT = (
    "docdb-2024-08-21-06-19-45.cluster-c5yo62eims1o.us-east-1.docdb.amazonaws.com"
)
PORT = 27017
USERNAME = "ldsadmin"
PASSWORD = "lds2connect"
DATABASE_NAME = "ldstest"
SSL_CA_CERT = "/home/ubuntu/lds/global-bundle.pem"
uri = f"mongodb://{USERNAME}:{PASSWORD}@{CLUSTER_ENDPOINT}:{PORT}/{DATABASE_NAME}?ssl=true"

# Connect to the DocumentDB cluster
client = MongoClient(uri, tls=True, tlsCAFile=SSL_CA_CERT)


llava_sagemaker_model = LlavaSagemaker()

pipeline = AwsPipeline(
    s3dir="BackgroundTagging",
    mongoclient=client,
    collection="BackgroundTagging",
    dbname="ldstest",
)
prompt = "Given an image, please analyze it and provide me with concise background details. If the background is blurry, consider it as 'Unknown'. If the background is blurry and a person is present, consider the background as 'Person'. Format the information as a JSON object with 'background' as the key and its one or two-word details as the value. Please provide the information in the following format:\n\n{\n 'background': 'Background details'\n}\n\nAnalyzing the image, provide the necessary information and format it accordingly, ensuring that the 'background' field includes relevant details such as road, street, church, or any other notable background elements, expressed in one or two words. If the background is blurry, use 'Unknown', and if the background is blurry and a person is present, use 'Person' as the background value."


@app.post("/infer/")
async def process_video(video_data: VideoInput):
    video_file = video_data.videourl
    output_dict = pipeline.start(
        model=llava_sagemaker_model, videourl=video_file, prompt=prompt, batch_size="1"
    )
    return output_dict


if __name__ == "__main__":
    uvicorn.run("api_background_aws:app", host="0.0.0.0", port=8006, log_level="info")
