import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

app = FastAPI()

# Configure CORS
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model class (same as the model you trained)
class GenderAgeCNN(nn.Module):
    def __init__(self):
        super(GenderAgeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2_gender = nn.Linear(256, 2)
        self.fc2_age = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        gender_output = self.fc2_gender(x)
        age_output = self.fc2_age(x)
        return gender_output, age_output

# Load the saved model
model = GenderAgeCNN()
model.load_state_dict(torch.load('gender_age_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        gender_output, age_output = model(image)
        _, gender_pred = torch.max(gender_output, 1)
        age_pred = age_output.item()

    gender_label = "male" if gender_pred.item() == 0 else "female"

    return {"gender": gender_label, "age": round(age_pred)}

