from typing import Annotated
from PIL import Image
import io

import torch
from torchvision import transforms as T
from torch.nn import functional as F

from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

import logging


log = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# categories
with open('cifar10_classes.txt') as f:
    catgs = [i.strip() for i in f.readlines()]


# Load model checkpoint
ckpt_path = './cifar_model_script.pt'
log.info(f"Instantiating scripted model <{ckpt_path}>")
model = torch.jit.load(ckpt_path)
model = model.eval()


predict_transform = T.Compose(
                        [
                            T.Resize((32, 32)), 
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )


@app.post("/infer")
async def infer(image: Annotated[bytes, File()]):
    img: Image.Image = Image.open(io.BytesIO(image))
    image = predict_transform(img)
    image = torch.unsqueeze(image, 0)
    #print(image.shape)
    #image = torch.tensor(image[None, None, ...], dtype=torch.float32)
    with torch.no_grad():
        logits = model.forward(image)
    
    preds = F.softmax(logits, dim=1).squeeze(0).tolist()

    out = torch.topk(torch.tensor(preds), len(catgs))
    topk_prob  = out[0].tolist()
    topk_label = out[1].tolist()
    
    confidence_map = {catgs[topk_label[i]]: topk_prob[i] for i in range(len(catgs))}
    print(confidence_map)
    return confidence_map



@app.get("/health")
async def health():
    return {"message": "ok"}