from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from torch_utils import transform_image, get_prediction

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get('/predict')
async def predict(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction":''})

@app.post('/predict')
async def predict(request: Request, response_class=HTMLResponse, image: UploadFile=File(...)):

    if image is None or image == "":
            return templates.TemplateResponse("index.html", {"request": request, "prediction": 'no file'})

    if not allowed_file(image.filename):
        return templates.TemplateResponse("index.html", {"request": request, "prediction": 'format not supported'})

    try:
        img_bytes = await image.read()
        tensor = transform_image(img_bytes)
        prediction = get_prediction(tensor)
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction.item()}) 
    except:
        return {'error': 'error during prediction'}

@app.get('/')
async def root():
    return {'message': 'Hello World'}
