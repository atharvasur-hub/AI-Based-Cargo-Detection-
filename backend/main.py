from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import base64
import uvicorn

# Initialize the Web Server
app = FastAPI(title="CIIBS Security API")

# Allow the frontend website to talk to this backend safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOUR trained AI Model (Must be in the same folder!)
model = YOLO("best.pt")

# Create the listening post for the frontend
@app.post("/predict")
async def analyze_xray(file: UploadFile = File(...)):
    # Read the image uploaded by the user
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Run the image through the AI model
    results = model(image)
    result = results[0]
    
    # Extract the detailed stats for the dashboard
    detections = []
    for box in result.boxes:
        detection = {
            "class_name": result.names[int(box.cls)], 
            "confidence": float(box.conf),            
            "box_coordinates": box.xyxy[0].tolist()   
        }
        detections.append(detection)

    # Convert the annotated image into a web-friendly format (Base64)
    annotated_image_array = result.plot()
    annotated_image_pil = Image.fromarray(annotated_image_array[..., ::-1]) 
    
    buffered = io.BytesIO()
    annotated_image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Send the final package back to the frontend website
    return {
        "status": "success",
        "detections": detections,
        "annotated_image_base64": img_str
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)