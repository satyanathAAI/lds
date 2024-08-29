
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import requests

app = FastAPI()
import boto3
s3_client = boto3.client('s3', region_name='us-east-1')
bucket_name='lds-test-public'


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can specify specific origins if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.). You can restrict as needed.
    allow_headers=["*"],  # Allows all headers. You can restrict as needed.
)
@app.post("/upload-video/")
async def upload_video(request: Request):
    # Read the raw JSON data
    data = await request.json()
    
    # Extract the 'videourl' parameter
    videourl = data["Body"]
    filename= data["Key"]
    # Print the 'videourl'
    print(f"Received video URL: {videourl}")
    response = requests.get(videourl)
    video_data = BytesIO(response.content)
    s3_client.upload_fileobj(video_data, bucket_name, f'Inputvideo/{filename}')
    return JSONResponse(content={"message": "Video URL received", "videourl": f'https://lds-test-public.s3.amazonaws.com/Inputvideo/{filename}'})

# To run the application, save this code to a file (e.g., `main.py`) and use the command:
# uvicorn main:app --reload





if __name__ == "__main__":
    import uvicorn
    uvicorn.run("s3_returnUrl:app",host="0.0.0.0", port=1234, reload=True)