"""Optional FastAPI inference service for CivicLens trained models."""
from pathlib import Path
from io import BytesIO
import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv('CIVICLENS_MODEL_DIR', str(BASE / 'models')))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN=[.485,.456,.406]; STD=[.229,.224,.225]; SIZE=224
app=FastAPI(title='CivicLens Image Classification API', version='1.0.0', description='Transfer-learning inference for pothole and garbage image classifiers.')
origins=os.getenv('CIVICLENS_ALLOWED_ORIGINS','*').split(',')
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=['GET','POST'], allow_headers=['*'])
models_loaded={}; model_errors={}

def load_task(task):
    path=MODEL_DIR/f'{task}_mobilenet_v3_small.pth'
    if not path.exists(): return None
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=models.mobilenet_v3_small(weights=None)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,2)
    model.load_state_dict(checkpoint['state_dict']); model.eval().to(DEVICE)
    return (model,checkpoint)

def get_task(task):
    if task not in models_loaded and task not in model_errors:
        try: models_loaded[task]=load_task(task)
        except Exception as e: model_errors[task]=str(e)
    return models_loaded.get(task)

@app.get('/')
def root(): return {'service':'CivicLens ML API','status':'ready','models':{t:(get_task(t) is not None) for t in ['pothole','garbage']},'note':'Model status is false until trained weights are placed in ml/models/.'}
@app.get('/health')
def health(): return {'status':'ok','device':DEVICE,'models':{t:(get_task(t) is not None) for t in ['pothole','garbage']},'model_errors':model_errors}
@app.post('/predict')
async def predict(file: UploadFile=File(...)):
    if file.content_type and not file.content_type.startswith('image/'): raise HTTPException(400,'Upload an image file.')
    raw=await file.read()
    if len(raw)>12*1024*1024: raise HTTPException(413,'Image must be smaller than 12 MB.')
    try: image=Image.open(BytesIO(raw)).convert('RGB')
    except (UnidentifiedImageError, OSError): raise HTTPException(400,'Could not read the uploaded image.')
    tf=transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    x=tf(image).unsqueeze(0).to(DEVICE)
    scores={}; missing=[]
    with torch.no_grad():
        for task in ['pothole','garbage']:
            entry=get_task(task)
            if not entry: missing.append(task); continue
            model,ckpt=entry; probs=model(x).softmax(1)[0].cpu().tolist()
            positive_idx=ckpt['class_to_idx'][ckpt['positive_class']]
            scores[task]=round(float(probs[positive_idx]),4)
    if not scores:
        raise HTTPException(503,'No trained model weights are installed. Train models using ml/train.py first.')
    ranked=sorted(scores.items(),key=lambda kv:kv[1],reverse=True)
    top_task,top_score=ranked[0]
    # Conservative demo threshold; validate/calibrate on held-out data before real deployment.
    suggestion=None
    if top_score>=0.65:
        suggestion='Pothole' if top_task=='pothole' else 'Garbage dump'
    return {'scores':scores,'suggestion':suggestion,'threshold':0.65,'models_missing':missing,'message':'Model output is a suggestion, not a verified diagnosis. Confirm the issue manually.' if suggestion else 'No confident supported category. Please choose the category manually.'}
