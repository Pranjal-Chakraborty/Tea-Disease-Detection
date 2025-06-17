# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException  # Used to create a web API and handle file uploads
from fastapi.responses import JSONResponse  # Used to return JSON responses
from fastapi.middleware.cors import CORSMiddleware  # Handles Cross-Origin Resource Sharing (CORS) for the API
from fastapi.encoders import jsonable_encoder  # Converts data into a JSON-friendly format
import uvicorn  # Runs the FastAPI application
import numpy as np  # Handles numerical operations and arrays
import cv2  # Provides image processing functionalities
from ultralytics import YOLO  # Used to load and work with the YOLO model for object detection
import base64  # Encodes and decodes data, used here for converting images to Base64
import logging  # Used to log messages and errors
from PIL import Image, UnidentifiedImageError  # PIL is used for opening and handling images

# Initialize the FastAPI application
app = FastAPI(title="Tea Leaf Disease Severity Analyzer")  # Sets up a web application with the title "Tea Leaf Disease Severity Analyzer"

# Add middleware for handling CORS, which allows requests from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,  # Allow credentials (e.g., cookies) in requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all HTTP headers
)

# Set up logging to display messages in the console
logging.basicConfig(level=logging.INFO)

# Load the segmentation model (YOLOv8) by providing the path to the trained model
segmentation_model_path = "D:/User/Downloads/YoloResults/content/runs/segment/train/weights/best.pt"
segmentation_model = YOLO(segmentation_model_path)

# Define an API endpoint for analyzing leaf images
@app.post("/analyze")
async def analyze_leaf(file: UploadFile = File(...)):  # Accepts an image file from the user
    try:
        # Check if the uploaded file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # Read the uploaded image file into memory
        contents = await file.read()
        # Convert the image file into a format that OpenCV can process (numpy array)
        nparr = np.frombuffer(contents, np.uint8)
        cv2_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decodes the image into a format usable by OpenCV

        # Use the segmentation model to make predictions on the uploaded image
        results = segmentation_model.predict(cv2_image)  # Runs the YOLOv8 model on the image
        result = results[0]  # Get the first result from the predictions

        # Initialize a dictionary to store segmentation data
        segmentation_data = {
            "boundary_pixels": 0,  # Number of pixels that form the boundary of the leaf
            "disease_pixels": 0,  # Number of pixels affected by disease
            "total_leaf_area": 0,  # Total number of pixels representing the leaf
            "severity": 0,  # Severity percentage of the disease
            "classification": "None",  # Health classification of the leaf
            "original_image": None,  # Base64-encoded original image
            "diseased_area_image": None,  # Base64-encoded image showing diseased areas
            "annotated_image": None,  # Base64-encoded annotated image
        }

        # If masks are present in the model's prediction, process them
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()  # Convert masks to a numpy array
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Get class IDs for the detected objects
            names = result.names  # Get class names from the model

            # Create a blank mask to store the final results
            final_mask = np.zeros_like(masks[0], dtype=np.uint8)

            # Process each mask based on its class (boundary or diseases)
            for i, mask in enumerate(masks):
                mask = mask > 0.5  # Convert probabilities to binary masks
                class_id = class_ids[i]  # Get the class ID of the current mask
                class_name = names[class_id]  # Get the class name (e.g., "boundary" or "diseases")

                if class_name == "boundary":  # Boundary mask
                    final_mask[mask] = 1
                elif class_name == "diseases":  # Diseased area mask
                    final_mask[mask] = 2

            # Calculate pixel counts for boundaries and diseased areas
            boundary_pixels = int(np.sum(final_mask == 1))
            disease_pixels = int(np.sum(final_mask == 2))
            total_leaf_area = boundary_pixels + disease_pixels  # Total area is the sum of boundary and disease pixels
            severity = int((disease_pixels / total_leaf_area) * 100) if total_leaf_area > 0 else 0  # Calculate severity percentage

            # Classify the leaf based on the severity
            segmentation_classification = (
                "Healthy" if severity <= 5 else
                "Mild" if severity <= 20 else
                "Moderate" if severity <= 50 else
                "Severe" if severity <= 100 else
                "Healthy"
            )

            # Update the segmentation data with calculated values
            segmentation_data.update({
                "boundary_pixels": boundary_pixels,
                "disease_pixels": disease_pixels,
                "total_leaf_area": total_leaf_area,
                "severity": severity,
                "classification": segmentation_classification,
            })

            # Generate an annotated image from the model's prediction
            annotated = result.plot()

            # Create an image showing only the diseased areas
            diseased_mask = final_mask == 2
            diseased_mask_resized = cv2.resize(
                diseased_mask.astype(np.uint8),
                (cv2_image.shape[1], cv2_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            diseased_image = np.zeros_like(cv2_image)
            for c in range(3):  # Apply the mask to all color channels
                diseased_image[:, :, c] = cv2_image[:, :, c] * diseased_mask_resized

            # Convert images to Base64 for returning in the API response
            _, buffer_original = cv2.imencode('.jpg', cv2_image)
            _, buffer_diseased = cv2.imencode('.jpg', diseased_image)
            _, buffer_annotated = cv2.imencode('.jpg', annotated)

            segmentation_data["original_image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer_original).decode('utf-8')}"
            segmentation_data["diseased_area_image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer_diseased).decode('utf-8')}"
            segmentation_data["annotated_image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer_annotated).decode('utf-8')}"

        # Return the segmentation results as a JSON response
        return JSONResponse(content=jsonable_encoder(segmentation_data), status_code=200)

    except UnidentifiedImageError:
        # Handle cases where the image could not be opened or is corrupted
        raise HTTPException(status_code=400, detail="Unable to read the image. It might be corrupted or unsupported.")

    except Exception as e:
        # Log unexpected errors and return a generic error response
        logging.exception("Unexpected error during analysis.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Run the FastAPI application if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=53125)
