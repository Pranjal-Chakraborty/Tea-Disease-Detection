
# 🌱 Tea Leaf Segmentation and Severity Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-✔️-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-✔️-orange)
![License](https://img.shields.io/github/license/yourusername/tea-leaf-segmentation)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

> **Segment and analyze tea leaf diseases with precision using YOLOv8 and advanced severity calculation techniques!**

---

## 🖋️ About

The **Tea Leaf Segmentation and Severity Analyzer** is a cutting-edge project designed to:
- **Segment tea leaf regions** using YOLOv8 for precise boundary and disease detection.
- **Quantify disease severity** by calculating the proportion of diseased areas.
- **Classify severity levels** into distinct categories such as "Healthy," "Mild," "Moderate," or "Severe."

The project integrates state-of-the-art machine learning and API technologies to deliver seamless and accurate analysis.

---

## 🚀 Features

- **Segmentation with YOLOv8**: Detects leaf boundaries and diseased regions with high precision.
- **Severity Calculation**: Computes the percentage of diseased pixels to total leaf area.
- **Classification**: Categorizes leaves based on severity levels for actionable insights.
- **Interactive API**: Enables real-time analysis and integration with other applications.
- **Base64-Encoded Image Outputs**: Provides annotated, original, and diseased area images.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or above
- Required libraries (see `requirements.txt`)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tea-leaf-segmentation.git
   cd tea-leaf-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:
   ```bash
   uvicorn app_severity:app --host 0.0.0.0 --port 8000
   ```

4. Access the API at `http://127.0.0.1:8000`.

---

## 📂 File Structure

```plaintext
📦 Tea_Leaf_Segmentation
├── app_severity.py            # API logic for segmentation and severity calculation
├── requirements.txt           # Python dependencies
├── Tea_Disease_Severity_Model_training-val-test.ipynb  # Segmentation model training and testing
├── severity_analysis.ipynb    # Detailed severity analysis notebook
└── README.md                  # Project documentation
```

---

## 🖼️ API Endpoints

### `/analyze`
- **Method**: POST
- **Input**: Image file (JPG/PNG)
- **Response**:
  - `boundary_pixels`: Pixels forming the leaf boundary.
  - `disease_pixels`: Pixels affected by disease.
  - `total_leaf_area`: Total pixels representing the leaf.
  - `severity`: Severity percentage.
  - `classification`: Health classification (Healthy/Mild/Moderate/Severe).
  - `annotated_image`: Base64-encoded annotated image.
  - `diseased_area_image`: Base64-encoded diseased region image.
  - `original_image`: Base64-encoded original image.

---

## 🧪 How It Works

1. Upload an image of a tea leaf via the `/analyze` endpoint.
2. The system processes the image using **YOLOv8** to segment boundaries and detect diseases.
3. It calculates disease severity and classifies the health of the leaf.
4. Outputs include annotated images, severity metrics, and classification labels.

---

## 🔗 Related Documents

1. [Severity Analysis Notebook](severity_analysis.ipynb): Detailed calculations for determining severity levels.
2. [Segmentation Model Training Notebook](Tea_Disease_Severity_Model_training-val-test.ipynb): YOLOv8 segmentation model training and validation details.

---

## 🎨 Preview

![Preview](https://via.placeholder.com/800x400?text=Segmented+Annotated+Image+Preview)

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request with your feature or bug fix.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
