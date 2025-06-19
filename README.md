# 🌿 Tea Plant Disease Detection System

<div align="center">

![Tea Plant Disease Detection](https://img.shields.io/badge/Tea%20Disease-Detection-green?style=for-the-badge&logo=leaf)
![Machine Learning](https://img.shields.io/badge/ML-Computer%20Vision-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**An AI-Powered Solution for Precision Agriculture**

*Revolutionizing tea cultivation through cutting-edge machine learning, computer vision, and natural language processing*

[🚀 Quick Start](#-quick-start) • [📊 Demo](#-demo) • [🛠️ Installation](#️-installation) • [📚 Documentation](#-documentation) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Project Overview

<details>
<summary><b>🎯 Problem Statement</b></summary>

Tea cultivation faces significant challenges:
- **20-30% crop losses** due to undetected diseases
- **Manual inspection** is labor-intensive and error-prone
- **Delayed treatment** leads to widespread contamination
- **Lack of expertise** in remote farming areas

</details>

<details>
<summary><b>💡 Our Solution</b></summary>

An end-to-end automated system that:
- ✅ **Detects** diseases with 85% accuracy using ConvNeXt
- ✅ **Analyzes** severity through pixel-level segmentation
- ✅ **Recommends** treatments via RAG-based AI
- ✅ **Scales** for farms of all sizes

</details>

---

## 🚀 Key Features

<table>
<tr>
<td align="center" width="33%">

### 🔍 Disease Detection
**ConvNeXt Model**
- 98% accuracy
- 7 disease types
- Real-time analysis

</td>
<td align="center" width="33%">

### 🎯 Severity Analysis
**YOLOv8 Segmentation**
- Pixel-level precision
- Automated scoring
- Visual heat maps

</td>
<td align="center" width="33%">

### 💊 Smart Recommendations
**RAG-Powered AI**
- Context-aware advice
- Personalized treatments
- Sustainable solutions

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Start<br><span style="color:gray">Input Image</span>] --> B[Image Preprocessing<br><span style="color:gray"></span>]
    B --> C[Disease Classification System<br><span style="color:gray">(ConvNeXt)</span>]
    C --> D{Is the leaf Healthy?<br><span style="color:gray"></span>}
    D -->|Yes| E[Recommendation System<br><span style="color:gray">(RAG)</span><br>Display that the leaf is healthy. Leaf boundary and yield increase on yield]
    D -->|No| F[Disease Severity Analysis<br><span style="color:gray"></span>]
    E --> I
    F --> G[Segmentation System<br><span style="color:gray">(YOLOv8)</span>]
    G --> H[Recommendation System<br><span style="color:gray">(RAG)</span><br>Display the name of disease, affected area, it's severity and treatment recommendation]
    H --> I[Stop<br><span style="color:gray"></span>]

    %% Styling to match the original diagram's color scheme with deep black font
    style A fill:#ef9a9a,stroke:#d32f2f,stroke-width:2px,color:#000000
    style B fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000000
    style C fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000000
    style D fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000000
    style E fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000000
    style F fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000000
    style G fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000000
    style H fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000000
    style I fill:#ef9a9a,stroke:#d32f2f,stroke-width:2px,color:#000000
```

---

## 📊 Disease Categories

<details>
<summary><b>🔬 Supported Diseases (Click to expand)</b></summary>

| Disease | Symptoms | Severity Impact |
|---------|----------|-----------------|
| 🟤 **Anthracnose** | Dark, sunken lesions | High |
| 🟫 **Brown Blight** | Brown patches with yellow halos | Very High |
| 🟢 **Algal Leaf Spot** | Green-blue circular spots | Medium |
| 👁️ **Bird Eye Spot** | Small circular lesions | Medium |
| 🌫️ **Gray Blight** | Gray-brown patches | High |
| 🔴 **Red Leaf Spot** | Reddish-brown spots | Medium |
| ⚪ **White Spot** | White circular lesions | Medium |

</details>

---

## 🛠️ Technology Stack

<div align="center">

| Component | Technology | Purpose |
|-----------|------------|---------|
| 🧠 **AI Models** | ConvNeXt, YOLOv8 | Disease detection & segmentation |
| 🤖 **NLP** | RAG + Google Gemini | Treatment recommendations |
| 🗄️ **Database** | DataStax Astra | Scalable data storage |
| 🌐 **Backend** | Flask, Python | API & server logic |
| 🎨 **Frontend** | HTML/CSS/JavaScript | User interface |
| ☁️ **Deployment** | Docker, Cloud-ready | Scalable deployment |

</div>

---

## 📈 Performance Metrics

<details>
<summary><b>🎯 Model Performance</b></summary>

### Classification Results
```
ConvNeXt Model:
├── Accuracy: 85.2%
├── Precision: 84.7%
├── Recall: 83.9%
└── F1-Score: 84.3%

Comparison with ResNet:
├── ConvNeXt: 85.2% ✅
└── ResNet-26d: 82.1% ❌
```

### Segmentation Results
```
YOLOv8 Model:
├── mAP@0.50: 0.87
├── Boundary Overlap: 88-94%
├── Pixel Accuracy: 91.3%
└── IoU Score: 0.82

Comparison with SegFormer:
├── YOLOv8: 0.87 mAP ✅
└── SegFormer-B2: 0.75 mAP ❌
```

</details>

---

## 🚀 Quick Start

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?logo=flask)

### Installation

<details>
<summary><b>📦 Step-by-step installation</b></summary>

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection.git
   cd Tea-Disease-Detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export ASTRA_DB_TOKEN=your_astra_token
   export GEMINI_API_KEY=your_gemini_key
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   ```
   http://localhost:5000
   ```

</details>

---

## 💻 Usage

### Web Interface

1. **Upload Image** 📸
   - Drag & drop or click to select tea leaf image
   - Supports JPG, PNG formats

2. **AI Analysis** 🤖
   - Automatic disease detection
   - Severity assessment
   - Segmentation visualization

3. **Get Recommendations** 💡
   - Personalized treatment plans
   - Chemical & organic solutions
   - Prevention strategies

### API Usage

<details>
<summary><b>🔌 REST API Endpoints</b></summary>

```python
# Disease Detection
POST /api/detect
Content-Type: multipart/form-data
Body: image file

Response:
{
  "disease": "Brown Blight",
  "confidence": 0.89,
  "severity": "Moderate",
  "severity_score": 35.2,
  "recommendations": [...],
  "segmentation_mask": "base64_image"
}
```

</details>

---

## 📊 Demo

<div align="center">

### 🎥 System Workflow

```
📱 Upload Image → 🔍 AI Analysis → 📊 Results → 💊 Treatment Plan
```

### Sample Results

| Input | Detection | Severity | Recommendation |
|-------|-----------|----------|----------------|
| 🍃 Leaf Image | Brown Blight (89% confidence) | Moderate (35%) | Copper fungicide treatment |

</div>

---

## 🌍 Impact & Benefits

<div align="center">

| 💰 Economic | 🌱 Environmental | 📈 Technical |
|-------------|------------------|---------------|
| 20-30% loss reduction | Reduced pesticide use | 85% accuracy |
| Increased farmer income | Sustainable practices | Real-time processing |
| Cost-effective solution | Precision application | Scalable architecture |

</div>

---

## 🤝 Team Contributors

<div align="center">

<table>
<tr>
<td align="center">
<img src="https://github.com/Pranjal-Chakraborty.png" width="100px;" alt="Pranjal"/><br>
<b>Pranjal Chakraborty</b><br>
<sub>ML Specialist</sub>
</td>
<td align="center">
<img src="https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection/blob/main/assets/team/arya.jpg" width="100px;" alt="Arya"/><br>
<b>Arya Chatterjee</b><br>
<sub>ML Specialist</sub>
</td>
<td align="center">
<img src="https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection/blob/main/assets/team/yash.jpg" width="100px;" alt="Yash"/><br>
<b>Yash Thakur</b><br>
<sub>Frontend and UI/UX</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection/blob/main/assets/team/devasmita.jpg" width="100px;" alt="Devasmita"/><br>
<b>Devasmita Kundu</b><br>
<sub>ML Specialist</sub>
</td>
<td align="center">
<img src="https://avatars.githubusercontent.com/u/101434286?s=400&u=7920a4039f13437111e8c24df96bbc5b11ddaf8d&v=4" width="100px;" alt="Samrat"/><br>
<b>Samrat Ghosh</b><br>
<sub>Frontend Development</sub>
</td>
<td align="center">
<img src="https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection/blob/main/assets/team/jaya.jpg" width="100px;" alt="Jaya"/><br>
<b>Jaya Sinha Mahapatra</b><br>
<sub>Research & UI/UX</sub>
</td>
</tr>
</table>

**Supervisor:** Dr. Subhrapratim Nath, Head of CSE, MSIT

</div>

---

## 🔮 Future Roadmap

<details>
<summary><b>🚁 Phase 1: Drone Integration</b></summary>

- Aerial imaging capabilities
- Large-scale plantation monitoring
- Automated flight path optimization

</details>

<details>
<summary><b>🌤️ Phase 2: Multimodal Analysis</b></summary>

- Weather data integration
- Soil condition analysis
- Environmental factor correlation

</details>

<details>
<summary><b>🔍 Phase 3: Explainable AI</b></summary>

- Model interpretability
- Decision transparency
- Farmer education tools

</details>

<details>
<summary><b>📱 Phase 4: Mobile Application</b></summary>

- iOS/Android apps
- Offline capabilities
- Real-time notifications

</details>

---

## 📚 Research & References

<details>
<summary><b>📖 Key Publications</b></summary>

1. **ConvNeXt Architecture**
   - Liu et al. (2022). "A ConvNet for the 2020s"
   - arXiv:2201.03545

2. **YOLOv8 Segmentation**
   - Reis et al. (2024). "Real-Time Flying Object Detection"
   - CVPRW 2024

3. **RAG Implementation**
   - Li et al. (2025). "Knowledge-Aware Refinement for RAG"
   - arXiv:2505.xxxxx

4. **Dataset Source**
   - Gibson Kimutai & Anna Förster
   - "Tea Sickness Dataset"

</details>

---

## 📄 License

<div align="center">

![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

</div>

---

## 🔗 Quick Links

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection)
[![Documentation](https://img.shields.io/badge/Docs-Read%20More-blue?logo=gitbook)](./docs)
[![Issues](https://img.shields.io/badge/Issues-Report%20Bug-red?logo=github)](https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection/issues)
[![Discussions](https://img.shields.io/badge/Discussions-Join%20Chat-green?logo=github)](https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection/discussions)

</div>

---

### ⭐ Show Your Support

**If this project helped you, please consider giving it a star!**

[![GitHub stars](https://img.shields.io/github/stars/Pranjal-Chakraborty/Tea-Disease-Detection?style=social)](https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection)
[![GitHub forks](https://img.shields.io/github/forks/Pranjal-Chakraborty/Tea-Disease-Detection?style=social)](https://github.com/Pranjal-Chakraborty/Tea-Disease-Detection)

**Made with ❤️**

</div>
