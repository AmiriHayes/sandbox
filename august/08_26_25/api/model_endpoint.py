from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN(10+26+26)
model.load_state_dict(torch.load("api/model_weights.pth", map_location="cpu"))
model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return{"message": "api working"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    pixels = data.get("input_image")

    if pixels is None: return {"error_0": "Expected pixels but did not recieve data"}
    if len(pixels) != 784: return {"error_1": "Exoected 784 values for 28x28 image"}

    x = torch.tensor(pixels, dtype=torch.float32).view(1, 28, 28)
    x = (x - 0.5) / 0.5
    x = torch.flip(x.transpose(1, 2), dims=[2])
    x = x.unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())
    
    emnist_labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]

    return {
        "prediction": emnist_labels[pred_idx],
        "confidence": confidence
    }