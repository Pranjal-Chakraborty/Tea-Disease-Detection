
# 🌿 Tea Leaf Disease Classification API

**Tea Leaf Disease Classification** is a FastAPI-based application that uses a deep learning model to analyze tea leaf images and classify them into healthy or diseased categories. This tool aims to assist farmers, researchers, and agricultural professionals in identifying leaf diseases efficiently and effectively.

---

## 🚀 Features

- **Accurate Disease Detection**: Utilizes a pre-trained ConvNeXt model fine-tuned for tea leaf classification.
- **User-Friendly API**: Simple to upload an image and get detailed classification results.
- **Cross-Origin Resource Sharing (CORS)**: Allows integration with web or mobile applications seamlessly.
- **Efficient Processing**: Leverages GPU (if available) for fast predictions.
- **Error Handling**: Provides clear error messages for invalid inputs or corrupted files.

---

## 🛠️ How It Works

1. **User Uploads an Image**: The API accepts an image file (e.g., `.jpg`, `.png`) through the `/analyze` endpoint.
2. **Image Preprocessing**:
   - Resized to 224x224 pixels.
   - Normalized to match the model's training configuration.
3. **Prediction**:
   - The ConvNeXt model processes the image and generates probabilities for all possible classes.
   - The class with the highest probability is returned as the predicted result.
4. **Response**:
   - A JSON response includes the predicted class and the probability scores for all classes.

---

## 🛠️ Installation

### Prerequisites

Ensure the following are installed on your machine:
- Python 3.8+
- `pip` (Python package manager)
- GPU support (optional, for faster inference)

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/tea-leaf-disease-classification.git
   cd tea-leaf-disease-classification
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the trained model and configuration files:
   - Place the `.pth` model file in the specified directory (`D:/User/Downloads/convnext_leaf_model_2` by default).
   - Ensure the configuration file (`config.json`) is in the same directory.

4. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 53125
   ```

5. Test the API:
   Open your browser or use a tool like [Postman](https://www.postman.com/) to send a `POST` request to:
   ```
   http://localhost:53125/analyze
   ```

---

## 📖 API Documentation

### Endpoints

#### **`POST /analyze`**

- **Description**: Upload an image to classify it as "Healthy" or "Diseased".
- **Request**:
  - Header: `Content-Type: multipart/form-data`
  - Body: Upload an image file.
- **Response**:
  ```json
  {
    "predicted_class": "Healthy",
    "probabilities": {
      "Healthy": 0.85,
      "Diseased": 0.15
    }
  }
  ```

- **Errors**:
  - `400`: Invalid file type or corrupted image.
  - `500`: Internal server error.

---

## 🧠 Model Details

- **Architecture**: ConvNeXt (Small variant).
- **Trained On**: A dataset of annotated tea leaf images.
- **Classes**: 
  - `Healthy`
  - `Diseased`
- **Preprocessing**:
  - Image size: 224x224.
  - Normalization: Mean = (0.5, 0.5, 0.5), Std = (0.5, 0.5, 0.5).

---

## 🧪 Testing

You can test the application by uploading sample tea leaf images. To ensure optimal performance:
- Use clear images with good lighting.
- Avoid blurry or distorted files.

---

## 📂 Project Structure

```
tea-leaf-disease-classification/
├── main.py                   # The main application script
├── requirements.txt          # Required Python libraries
├── README.md                 # Project documentation
├── convnext_leaf_model_2/    # Directory for model and config files
│   ├── convnext_leaf_model_2.pth
│   └── config.json
```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure that your code adheres to the project's coding style and includes appropriate tests.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💡 Acknowledgments

- **[PyTorch](https://pytorch.org/)** for the deep learning framework.
- **[TIMM](https://github.com/huggingface/timm)** for providing the ConvNeXt architecture.
- **FastAPI** for the rapid and efficient web API development.

---
