from search.search_main_aws import SearchHelper_Aws
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HelperInput(BaseModel):
    videourl: str


csv_dir = "/home/ubuntu/lds/search_csv_dir"


search_helper = SearchHelper_Aws(
    background_collection="BackgroundTagging",
    summarization_collection="VideoSummarization",
    face_collection="athelete_detection",
    transcipt_collection="transcript_collection",
    csv_dir="/home/ubuntu/lds/search_csv_dir",
)
# Load all the necessary environment variables from  .env file


@app.post("/club_responses/")
async def process_video(club_data: HelperInput):
    videoname = club_data.videourl
    response = search_helper.club_response(videoname=videoname)

    return {"out": response}


if __name__ == "__main__":
    uvicorn.run("search_helper_api:app", host="0.0.0.0", port=9022, log_level="info")
