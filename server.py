from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import uvicorn
from torchvision.models import resnet18, ResNet18_Weights

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        label = "Fake" if prediction == 0 else "Real"
        confidence = f"{probabilities[0][prediction]:.2f}"
        color = "#ff4d4d" if prediction == 0 else "#4CAF50"

        result_html = f"""
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; margin: 50px; }}
                .container {{
                    max-width: 500px; margin: auto; padding: 20px; border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2); background: #fff;
                }}
                h1 {{ color: {color}; }}
                .result {{
                    font-size: 24px; font-weight: bold; padding: 20px;
                    border-radius: 8px; background: {color}; color: white;
                }}
                .back-btn {{
                    margin-top: 20px; text-decoration: none; background: #007BFF; color: white;
                    padding: 10px 20px; border-radius: 5px; display: inline-block;
                }}
                .back-btn:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Image Classification Result</h1>
                <p class="result">Prediction: {label} ({confidence} confidence)</p>
                <a class="back-btn" href="/">Go Back</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=result_html)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def main():
    content = """
    <html>
    <head>
        <title>Upload an Image</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
            .container {
                max-width: 400px; margin: auto; padding: 20px;
                border-radius: 10px; background: #f9f9f9;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            h1 { color: #333; }
            form {
                margin-top: 20px; padding: 20px; background: #fff;
                border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            input[type="file"] { margin-bottom: 15px; }
            input[type="submit"] {
                background-color: #007BFF; color: white;
                border: none; padding: 10px 20px; border-radius: 5px;
                cursor: pointer; font-size: 16px;
            }
            input[type="submit"]:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload an Image</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <br>
                <input type="submit" value="Classify Image">
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content)


