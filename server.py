import base64

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from starlette.responses import JSONResponse

from identify import Identify
from identify_blip import IdentifyBlip
from identify_gemini import IdentifyGemini

app = FastAPI()
load_dotenv()
identify = Identify()
# identify_blip = IdentifyBlip()
identify_gemini = IdentifyGemini()

class IdentifyRequest(BaseModel):
    image_url: str

@app.post("/identify")
def run(req: IdentifyRequest):
    name = identify.get_name(base64.b64decode(req.image_url))
    urls = {
        "plastic bottle": [
            {
                "icon": 1,
                "url": "https://lifewithoutplastic.com/vacuum-insulated-stainless-steel-bottle-680-ml-23-oz/",
                "desc": "Vacuum Insulated Stainless Steel Bottle"
            },
            {
                "icon": 6,
                "url": "https://lifewithoutplastic.com/double-wall-reusable-glass-travel-mug-400-ml-13-5-oz/",
                "desc": "Double Wall Reusable Glass Travel Mug"
            },
        ],
        "plastic food container": [
            {
                "icon": 5,
                "url": "https://lifewithoutplastic.com/vacuum-insulated-stainless-steel-bottle-680-ml-23-oz/",
                "desc": "Stainless Steel Rectangular Food Storage Container"
            },
            {
                "icon": 4,
                "url": "https://lifewithoutplastic.com/round-stainless-steel-airtight-take-out-container-with-dividers/",
                "desc": "Round Stainless Steel Airtight Take-Out Container"
            },
        ],
        "plastic phone case": [
            {
                "icon": 3,
                "url": "https://pelacase.com/products/seashell-aquarius-water-eco-friendly-iphone-16-pro-case",
                "desc": "Seashell Aquarius Water iPhone 16 Pro Case"
            },
            {
                "icon": 2,
                "url": "https://pelacase.com/products/black-beagle-buds-eco-friendly-samsung-galaxy-s25-plus-case",
                "desc": "Black Beagle Buds Samsung Galaxy S25+(Plus) Case"
    },
        ],
        "other": []
    }

    return {
        "name": name,
        "urls": urls[name]
    }

# @app.post("/identify-v2")
# def run_v2(req: IdentifyRequest):
#     result = identify_blip.run(base64.b64decode(req.image_url))
#     return result

@app.post("/identify-gemini")
def run_gemini(req: IdentifyRequest):
    result = identify_gemini.run(req.image_url)
    return result

@app.post("/test")
async def test(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File is not an image"}, status_code=400)

    image_data = await file.read()

    try:
        return {
            "name": identify.get_name(image_data)
        }
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)


@app.post("/test-gemini")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File is not an image"}, status_code=400)

    image_data = await file.read()

    try:
        return identify_gemini.run(image_data)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)


@app.get("/")
def read_root():
    return "Microplastics API"
