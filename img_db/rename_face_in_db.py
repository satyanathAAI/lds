from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware

import os
import pandas as pd

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Rename_Input(BaseModel):
    old_name: str
    new_name: str
    
@app.post("/rename_face/")
async def rename_face(query: Rename_Input):
    old_name = query.old_name
    new_name = query.new_name
    df = pd.read_csv('/home2/asgtestdrive2023/Projects/MAM/AI-Team/search/prod/search_final_for_sure.csv')
    idx_lst = df[df['athletes'].str.contains(old_name)].index.tolist()
    for i in idx_lst:
        df['athletes'].iloc[i] = df['athletes'].iloc[i].replace(old_name,new_name)
    df = df.astype({'athletes':'string'})
    df.to_csv('/home2/asgtestdrive2023/Projects/MAM/AI-Team/search/prod/search_final_for_sure.csv',index=False)
    old_name = old_name.replace(' ','_')
    new_name = new_name.replace(' ','_')
    output_dict = dict()
    face_dirs = [i for i in os.listdir("/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/img_db") if not i.startswith('.')]
    if not new_name in face_dirs:
        os.rename(f"/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/img_db/{old_name}",f"/home2/asgtestdrive2023/Projects/MAM/appeng/UIV6/src/assets/img_db/{new_name}")
        output_dict['status'] = "Renamed Successfully"
    else:
        output_dict['status'] = "Given name already exsists!"
    return output_dict

if __name__ == "__main__":
    uvicorn.run("rename_face_in_db:app",host="0.0.0.0",port=8181, log_level="info")
