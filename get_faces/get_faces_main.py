from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from get_faces.get_faces_aws import main_logic

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


@app.post("/get_faces/")
async def process_video(video_data: VideoInput):
    video_file = video_data.videourl
    output_dict = main_logic(video_file)
    return output_dict


if __name__ == "__main__":
    uvicorn.run("get_faces_main:app", host="0.0.0.0", port=8099, log_level="info")
