from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Define your FastAPI app
app = FastAPI()

# Load your PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess the image
def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        result = f"Fake ({probabilities[0][0]:.2f})" if prediction == 0 else f"Real ({probabilities[0][1]:.2f})"

        # Return the prediction as HTML response
        result = f"""
        <html>
            <body>
                <h1>Prediction Result</h1>
                <p><b>Prediction:</b> {result}</p>
                <a href="/">Go back</a>
            </body>
        </html>
        """
        return HTMLResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Define the HTML form endpoint
@app.get("/")
def main():
   content = """
    <html>
        <head>
            <title>Image Upload</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 50px; text-align: center; }
                form { border: 1px solid #f1f1f1; padding: 20px; border-radius: 10px; display: inline-block; }
                input[type="file"] { margin-bottom: 20px; }
                input[type="submit"] { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                input[type="submit"]:hover { background-color: #45a049; }
            </style>
        </head>
        <body>
            <h1>Upload an Image for Classification</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Upload and Classify">
            </form>
        </body>
    </html>
    """
   return HTMLResponse(content=content)

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


