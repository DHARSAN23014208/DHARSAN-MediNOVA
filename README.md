# 🩺 CureSight - AI-Powered Medical Diagnosis

An intelligent health screening application that uses machine learning to provide preliminary assessments for 7 different diseases. Built with Flask and TensorFlow, featuring a modern glassmorphism UI design.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/License-Educational-yellow.svg)

## ✨ Features

- **7 Disease Predictions** - Heart disease, diabetes, kidney disease, liver disease, breast cancer, malaria, and pneumonia
- **AI-Powered Analysis** - Uses trained machine learning models with 95%+ accuracy
- **Modern UI** - Beautiful dark theme with glassmorphism effects and smooth animations
- **Detailed Insights** - Provides risk factors, recommendations, and health tips for each diagnosis
- **Image Analysis** - CNN-based detection for malaria (blood smears) and pneumonia (X-rays)
- **Responsive Design** - Works on desktop and mobile devices

## 🔬 Supported Diagnoses

| Disease | Model Type | Accuracy | Input Type |
|---------|-----------|----------|------------|
| Heart Disease | Random Forest | ~85% | Clinical Parameters (13 features) |
| Diabetes | XGBoost | ~92% | Health Metrics (8 features) |
| Kidney Disease | Random Forest | ~98% | Lab Results (24 features) |
| Liver Disease | Random Forest | ~75% | Enzyme Levels (10 features) |
| Breast Cancer | Random Forest | ~95% | Cell Measurements (5 features) |
| Malaria | CNN | ~95% | Blood Smear Images |
| Pneumonia | CNN | ~90% | Chest X-Ray Images |

## 🛠️ Technologies Used

### Backend
- **Flask** - Python web framework
- **TensorFlow/Keras** - Deep learning for image classification
- **Scikit-learn** - Traditional ML algorithms
- **XGBoost & LightGBM** - Gradient boosting models
- **Pandas & NumPy** - Data processing

### Frontend
- **HTML5 & CSS3** - Modern markup and styling
- **Vanilla JavaScript** - Interactive features
- **Google Fonts (Inter)** - Typography

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- ~2GB disk space (for models and dependencies)

### Installation

1. **Clone or download the project**
   ```bash
   cd mini-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

## 📁 Project Structure

```
mini-project/
├── app.py                 # Flask application (main entry point)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── models/                # Pre-trained ML models (.pkl, .h5)
│   ├── heart_disease_model.pkl
│   ├── diabetes_model.pkl
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   ├── breast_cancer_model.pkl
│   ├── malaria_cnn_model_fast.h5
│   └── pneumonia_model_new.h5
├── templates/             # HTML templates
│   ├── home.html
│   ├── features.html
│   ├── heart.html
│   ├── diabetes.html
│   ├── kidney.html
│   ├── liver.html
│   ├── breast_cancer.html
│   ├── malaria.html
│   └── pneumonia.html
├── static/                # CSS and static assets
│   └── style.css
├── data/                  # Training datasets
│   ├── cell_images/       # Malaria cell images
│   └── chest_xray/        # Pneumonia X-ray images
├── scripts/               # Model training scripts
│   └── train_malaria_fast.py
└── work_files/            # Jupyter notebooks (original research)
```

## 🎯 Usage Examples

### Form-Based Diagnosis
1. Navigate to the Features page
2. Select a disease (e.g., Heart Disease)
3. Fill in the clinical parameters
4. Click "Analyze Results"
5. View the prediction with insights and recommendations

### Image-Based Diagnosis
1. Go to Malaria or Pneumonia detection
2. Upload a blood smear image or chest X-ray
3. Click "Analyze Image"
4. View the AI prediction with detailed analysis



## 📝 License

This project is for educational use only. The datasets used are from public sources with their respective licenses.

---

<p align="center">
  Made with ❤️ by Sanjay Sivaramakrishnan M and Team
</p>
