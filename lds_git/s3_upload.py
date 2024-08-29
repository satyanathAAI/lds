from fastapi import FastAPI, UploadFile, HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import urlparse

app = FastAPI()
s3_client = boto3.client('s3', region_name='us-east-1')
bucket_name = 'kendraassetiq'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can specify specific origins if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.). You can restrict as needed.
    allow_headers=["*"],  # Allows all headers. You can restrict as needed.
)

@app.post("/upload/")
async def upload_file(file: UploadFile):
    try:
        s3_client.upload_fileobj(file.file, bucket_name, f'test/{file.filename}')
        return JSONResponse(content={"filename": file.filename, "status": "uploaded"})
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="AWS credentials not available")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("s3_upload:app", host="0.0.0.0", port=8000, reload=True)














