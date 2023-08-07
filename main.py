from fastapi import FastAPI, File, UploadFile
import shutil
import os
from io import BytesIO

import numpy as np
from deepface import DeepFace
from PIL import Image
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64


app = FastAPI()

# Mount a directory containing static files (e.g., images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2Templates
templates = Jinja2Templates(directory="templates")

# HTML template for displaying images 

@app.get("/")
async def index():
    
        
    return "Server Running"




@app.post("/upload/")
async def upload_images(request: Request, image1: UploadFile = File(...), image2: UploadFile = File(...)):
    # Save uploaded images to the 'static' directory
    image1_path = os.path.join("static", image1.filename)
    image2_path = os.path.join("static", image2.filename)

    with open(image1_path, "wb") as f1, open(image2_path, "wb") as f2:
        shutil.copyfileobj(image1.file, f1)
        shutil.copyfileobj(image2.file, f2)

    
    result = DeepFace.verify(image1_path,image2_path,
                         enforce_detection=True,
                         model_name='Dlib',
                         detector_backend='dlib',
                         distance_metric='euclidean')
        
    similarity = result["verified"]
 

    cord_img1 = result['facial_areas']['img1']
    cord_img2 = result['facial_areas']['img2']

    x1 = cord_img1['x']
    y1 = cord_img1['y']
    w1 = cord_img1['w']
    h1 = cord_img1['h']


    x2 = cord_img2['x']
    y2 = cord_img2['y']
    w2 = cord_img2['w']
    h2 = cord_img2['h']
    
    RESIZE_SHAPE = (200,200)



    cv2_img1,cv2_img2 = Image.open(image1_path),Image.open(image2_path)

    print(cord_img2,cord_img1)

    cv2_img1 = cv2_img1.crop((x1, y1, x1 + w1, y1 + h1))
    cv2_img2 = cv2_img2.crop((x2, y2, x2 + w2, y2 + h2))

    cv2_img1 = cv2_img1.resize(RESIZE_SHAPE)
    cv2_img2 = cv2_img2.resize(RESIZE_SHAPE)

    cropped_img1_path = './static/op1.jpg'
    cropped_img2_path = './static/op2.jpg'

    cv2_img1.save(cropped_img1_path)
    cv2_img2.save(cropped_img2_path)




    # Convert images to base64 strings for display
    with open(cropped_img1_path, "rb") as f1, open(cropped_img2_path, "rb") as f2:
        image1_base64 = base64.b64encode(f1.read()).decode("utf-8")
        image2_base64 = base64.b64encode(f2.read()).decode("utf-8")

    for image_path in os.listdir('./static'):

        if image_path.endswith(".txt"):
            continue
        del_path = './static/' + image_path

        if os.path.exists(del_path):
            os.remove(del_path)


    return templates.TemplateResponse(
        "image_template.html",
        {
          "request": request,
          "image1": f"data:image/jpeg;base64,{image1_base64}", 
          "image2": f"data:image/jpeg;base64,{image2_base64}",
          'result':str(similarity)
          }
    )

    
@app.get('/upload_images/')
async def read_item(request:Request):
    
    return templates.TemplateResponse(
            "form_template.html",
            {"request": request}
        )