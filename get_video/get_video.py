from fastapi import FastAPI
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


class InputApi(BaseModel):
    videourl: str


@app.post("/get_video")
async def some(video_data: InputApi):
    video_url = video_data.videourl
    return {"video": video_url}


if __name__ == "__main__":
    uvicorn.run("get_video:app", host="0.0.0.0", port=6996, log_level="info")
