# 🍅 Tomato Leaf Disease Detection

A deep learning-based tomato leaf disease classification system built using **PyTorch, EfficientNet-B0, and Streamlit**.

The system accepts a tomato leaf image and predicts one of **11 trained classes**, along with prediction confidence and Top-3 predictions. A confidence-based rejection mechanism is also included for images that are not confidently recognized as one of the trained tomato-leaf classes.

---

## 📌 Overview

The project uses **EfficientNet-B0** with transfer learning/fine-tuning for tomato leaf disease classification.

### Workflow

```text
Input Image
    ↓
Image Preprocessing
    ↓
Resize to 224 × 224
    ↓
Normalization
    ↓
EfficientNet-B0
    ↓
Classification Layer
    ↓
Softmax Probabilities
    ↓
Prediction + Confidence
    ↓
Accept Prediction / Unknown Image
```

---

## ✨ Features

- 🍅 Tomato leaf disease classification
- 🧠 EfficientNet-B0 deep learning model
- 🔄 Transfer learning / fine-tuning
- 🖼️ Streamlit image upload interface
- 📊 Prediction confidence
- 🏆 Top-3 predictions
- ⚠️ Low-confidence image rejection
- 📋 Model information display
- 💻 CPU/GPU support depending on PyTorch configuration

---

## 📂 Dataset

The model was trained on a tomato leaf disease dataset containing **11 classes**.

### Image preprocessing

Input images are:

1. Converted to RGB
2. Resized to **224 × 224**
3. Converted to PyTorch tensors
4. Normalized using ImageNet normalization

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## 🌱 Disease Classes

The model supports the following 11 classes:

| # | Class |
|---:|---|
| 1 | Bacterial Spot |
| 2 | Early Blight |
| 3 | Late Blight |
| 4 | Leaf Mold |
| 5 | Septoria Leaf Spot |
| 6 | Two-Spotted Spider Mites |
| 7 | Target Spot |
| 8 | Tomato Yellow Leaf Curl Virus |
| 9 | Tomato Mosaic Virus |
| 10 | Healthy |
| 11 | Powdery Mildew |

---

## 🧠 Model Architecture

### EfficientNet-B0

The project uses **EfficientNet-B0**, a convolutional neural network architecture that provides a good balance between accuracy and computational efficiency.

The model was adapted for the 11-class tomato disease classification task.

### Model configuration

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Framework | PyTorch |
| Input Size | 224 × 224 |
| Input Channels | 3 RGB |
| Number of Classes | 11 |
| Loss Function | Cross Entropy Loss |
| Optimizer | Adam |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Best Epoch | 9 |
| Classification | Multi-class |

---

## 🔄 Transfer Learning

A pretrained EfficientNet-B0 architecture was adapted for this dataset.

The original classification layer was replaced with a custom classification layer containing **11 output neurons**.

```text
Pretrained EfficientNet-B0
          ↓
Feature Extraction
          ↓
Custom Classification Layer
          ↓
11 Tomato Leaf Classes
```

---

## 📊 Model Performance

The trained model achieved:

| Metric | Result |
|---|---:|
| Validation Accuracy | **99.04%** |
| Test Accuracy | **98.79%** |
| Best Epoch | **9** |

These results represent performance on the evaluation data. Real-world performance can differ because of differences in lighting, camera quality, background, leaf orientation, disease severity, and image quality.

---

## 🎯 Prediction Process

For an uploaded image:

```text
Upload Image
     ↓
RGB Conversion
     ↓
Resize → 224 × 224
     ↓
Normalize
     ↓
EfficientNet-B0
     ↓
Softmax
     ↓
11 Class Probabilities
     ↓
Highest Probability
```

The model produces probabilities for all 11 classes.

Example:

```text
1. Bacterial Spot       99.59%
2. Septoria Leaf Spot    0.18%
3. Leaf Mold              0.16%
```

---

## ⚠️ Unknown Image Detection

A normal multi-class classifier will always choose one of its trained classes, even for an unrelated image.

For example, a rose leaf is not one of the 11 trained classes. Without rejection logic, the model could still produce:

```text
Healthy — 31.39%
```

To reduce misleading predictions, the Streamlit application uses a **confidence-based rejection mechanism**.

### Current threshold

```text
60%
```

### Decision logic

```text
              Input Image
                   ↓
            EfficientNet-B0
                   ↓
          Highest Confidence
                   ↓
        ┌──────────┴──────────┐
        │                     │
     >= 60%                  < 60%
        │                     │
        ↓                     ↓
Accept Prediction      Unknown / Reject
```

If the highest confidence is below 60%, the application displays:

```text
⚠️ Unknown / Unrecognized Image
```

The image may be:

- Not a leaf
- Not a tomato leaf
- A disease not included in the training dataset
- Too different from the training images
- Poor quality or unclear

### Important

This is a **confidence-based rejection mechanism**, not a separately trained "Not Tomato" classifier. Therefore, it should not be considered a definitive detector of every non-tomato image.

---

## 🖥️ Streamlit Application

The trained model is deployed using **Streamlit**.

The application provides:

- Image upload
- Uploaded image preview
- Disease prediction
- Confidence score
- Top-3 predictions
- Unknown-image warning
- Model information
- Trained class information

### Run the application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
tomato-disease-detection/
│
├── app.py
│
├── models/
│   └── tomato_disease_efficientnet_b0_final.pth
│
├── test_model.py
│
├── requirements.txt
│
└── README.md
```

### File descriptions

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `models/` | Stores trained model checkpoint |
| `.pth` file | Trained EfficientNet-B0 weights and metadata |
| `test_model.py` | Model loading/testing |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Computer vision and model utilities |
| EfficientNet-B0 | Disease classification |
| Pillow | Image processing |
| NumPy | Numerical operations |
| Streamlit | Web application |
| CUDA | GPU acceleration when available |

---

## ⚙️ Installation

### 1. Create a virtual environment

#### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The `requirements.txt` file should contain:

```text
torch
torchvision
streamlit
pillow
numpy
```

---

## ▶️ Running the Application

From the project root:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🧪 Testing

The model was tested using known tomato disease images as well as external unseen images.

Example:

```text
True Class: Bacterial Spot
Predicted: Bacterial Spot
Confidence: 99.59%
```

Another example:

```text
True Class: Late Blight
Predicted: Late Blight
Confidence: 99.98%
```

External images were also tested to examine generalization.

---

## 🌹 Out-of-Distribution Testing

A rose leaf was tested even though rose leaves are not part of the 11 trained classes.

The model produced:

```text
1. Healthy               31.39%
2. Leaf Mold             25.42%
3. Septoria Leaf Spot    15.72%
```

Because the highest confidence was only **31.39%**, which is below the configured **60% threshold**, the Streamlit application rejects the prediction as:

```text
Unknown / Unrecognized Image
```

This demonstrates why a closed-set classifier needs an additional rejection mechanism when it is exposed to images outside its training distribution.

---

## 💾 Model Checkpoint

The saved `.pth` checkpoint contains:

```text
model_name
num_classes
classes
class_to_idx
idx_to_class
image_size
state_dict
best_epoch
validation_accuracy
test_accuracy
```

This allows the trained model to be loaded later without retraining.

---

## 🔬 Explainable AI

A planned extension is **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM can highlight the regions of the leaf that contributed most to the model's prediction.

```text
Input Leaf Image
       ↓
EfficientNet-B0
       ↓
Disease Prediction
       ↓
Grad-CAM
       ↓
Heatmap
       ↓
Disease-Relevant Regions
```

This can improve model interpretability.

---

## 🔮 Future Improvements

- Better Out-of-Distribution (OOD) detection
- Dedicated tomato-vs-non-tomato classifier
- Grad-CAM visualization
- Disease severity estimation
- Larger and more diverse datasets
- Better confidence calibration
- Mobile application
- Cloud deployment
- Real-time camera detection
- Additional tomato diseases

---

## ⚠️ Limitations

1. The model is trained on only 11 classes.
2. A standard softmax classifier can still be overconfident on unknown images.
3. The 60% threshold is an initial confidence threshold and should be calibrated using dedicated in-distribution and out-of-distribution validation data.
4. Real-world performance can differ from test-set performance.
5. Images with unusual lighting, backgrounds, or poor quality may reduce accuracy.
6. The system does not replace professional agricultural diagnosis.

---

## 📌 Technical Summary

```text
Model:
EfficientNet-B0

Framework:
PyTorch

Approach:
Transfer Learning / Fine-tuning

Task:
11-Class Image Classification

Input:
224 × 224 RGB Image

Loss:
Cross Entropy Loss

Optimizer:
Adam

Scheduler:
ReduceLROnPlateau

Validation Accuracy:
99.04%

Test Accuracy:
98.79%

Deployment:
Streamlit

Unknown Threshold:
60%
```

---

## 🏁 Conclusion

This project demonstrates a deep learning approach for automated tomato leaf disease classification using **EfficientNet-B0 and PyTorch**.

The model achieved **99.04% validation accuracy** and **98.79% test accuracy** on the evaluation data.

The trained model was integrated into a **Streamlit application** that allows users to upload leaf images and receive:

- Disease predictions
- Confidence scores
- Top-3 predictions
- Unknown-image warnings

The project provides a foundation for future improvements such as **Grad-CAM explainability, improved OOD detection, disease severity estimation, and real-world deployment**.

---

## ⚠️ Disclaimer

This project is intended for **educational, research, and screening purposes**.

Predictions should not be considered a definitive agricultural diagnosis. Professional agricultural or plant pathology expertise should be consulted for confirmation and treatment decisions.

---

## ⭐ Project Highlights

```text
🍅 Tomato Leaf Disease Detection
🧠 EfficientNet-B0
🔥 PyTorch
📊 99.04% Validation Accuracy
🎯 98.79% Test Accuracy
🌱 11 Classes
🖥️ Streamlit
⚠️ Unknown Image Rejection
🔬 External Image Testing
🚀 Grad-CAM as Future Work
```
#   C r o p _ D i s e a s e _ D e t e c t o r  
 