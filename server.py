from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Define FastAPI app
app = FastAPI()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.BatchNorm1d(model.fc.in_features),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(512),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 2)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence for reliable prediction

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        confidence = confidence.item()
        prediction_label = "Fake" if prediction.item() == 0 else "Real"

        if confidence < CONFIDENCE_THRESHOLD:
            prediction_label = "Uncertain"

        return {
            "prediction": prediction_label,
            "confidence": confidence
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def main():
    content = """
    <html>
        <head>
            <title>Fake Image Detector</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
                body {
                    font-family: 'Poppins', sans-serif;
                    margin: 0;
                    padding: 0;
                    background: url('https://source.unsplash.com/1600x900/?technology,abstract') no-repeat center center fixed;
                    background-size: cover;
                }
                .container {
                    width: 40%;
                    margin: 100px auto;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
                    text-align: center;
                }
                h2 {
                    font-size: 24px;
                    color: #333;
                    margin-bottom: 15px;
                }
                form {
                    margin-top: 20px;
                }
                input[type="file"] {
                    padding: 10px;
                    border: 2px solid #ccc;
                    border-radius: 5px;
                    width: 85%;
                    display: block;
                    margin: auto;
                }
                input[type="submit"] {
                    margin-top: 15px;
                    background-color: #007BFF;
                    color: white;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    width: 90%;
                    font-size: 16px;
                    transition: background 0.3s ease;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Upload an Image for Verification</h2>
                <form action="/predict/" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept="image/*">
                    <input type="submit" value="Verify Image">
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


