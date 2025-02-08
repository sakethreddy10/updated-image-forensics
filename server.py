from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Define FastAPI app
app = FastAPI()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.6),  # Match dropout with Model.py
    torch.nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformations (Must match Model.py preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Confidence threshold (Increased to 70%)
CONFIDENCE_THRESHOLD = 0.70

# Function to preprocess image
def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Prediction API with Confidence Threshold
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image and convert to RGB
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get image details
        image_name = file.filename
        image_format = image.format
        image_size = image.size  # (width, height)
        
        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        confidence = confidence.item()
        prediction_label = "Fake" if prediction.item() == 0 else "Real"

        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            prediction_label = "Uncertain"

        # Generate UI response
        result = f"""
        <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center; margin: 50px; background-color: #f4f4f4; }}
                    .container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); display: inline-block; }}
                    h1 {{ color: #333; }}
                    .result {{ font-size: 22px; font-weight: bold; padding: 10px; border-radius: 5px; }}
                    .real {{ color: #008000; background-color: #e0ffe0; }}
                    .fake {{ color: #ff0000; background-color: #ffe0e0; }}
                    .uncertain {{ color: #ff8c00; background-color: #fff4e0; }}
                    .details {{ font-size: 18px; margin-top: 15px; }}
                    a {{ text-decoration: none; color: #007BFF; font-size: 16px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Prediction Result</h1>
                    <p class="details"><b>Image Name:</b> {image_name}</p>
                    <p class="details"><b>Format:</b> {image_format}</p>
                    <p class="details"><b>Size:</b> {image_size[0]}x{image_size[1]} pixels</p>
                    <p class="result {prediction_label.lower()}">Prediction: {prediction_label} ({confidence:.2f} Confidence)</p>
                    <a href="/">Go back</a>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Improved HTML Upload Page with Better UI
@app.get("/")
def main():
   content ="""
    <html>
        <head>
            <title>Image Verification</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
                body {{
                    font-family: 'Poppins', sans-serif;
                    margin: 0;
                    padding: 0;
                    background: url('https://source.unsplash.com/1600x900/?technology,abstract') no-repeat center center fixed;
                    background-size: cover;
                }}
                .container {{
                    width: 40%;
                    margin: 100px auto;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
                    text-align: center;
                }}
                h2 {{
                    font-size: 24px;
                    color: #333;
                    margin-bottom: 15px;
                }}
                form {{
                    margin-top: 20px;
                }}
                input[type="file"] {{
                    padding: 10px;
                    border: 2px solid #ccc;
                    border-radius: 5px;
                    width: 85%;
                    display: block;
                    margin: auto;
                }}
                input[type="submit"] {{
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
                }}
                input[type="submit"]:hover {{
                    background-color: #0056b3;
                }}
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

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

