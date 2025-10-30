# 🍽️ Count-Your-Calories

<img width="1897" height="977" alt="image" src="https://github.com/user-attachments/assets/c509e1de-98a4-4ca9-8e5d-f786ec8d6485" />
<img width="1893" height="974" alt="image" src="https://github.com/user-attachments/assets/a8b742be-5bc5-4005-bd26-fb2a818c43ec" />


An AI-powered food detection and calorie estimation application using YOLOv8 and computer vision. Upload a photo of your meal to automatically detect food items and estimate total calories.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- 🔍 **Automatic Food Detection** - Detects multiple food items in a single image
- 📊 **Calorie Estimation** - Estimates calories for each detected food item
- 🎨 **Visual Annotations** - Displays bounding boxes with calorie information
- 🔊 **Voice Output** - Text-to-speech summary of detected items
- 📚 **Calorie Database** - Browse 150+ food items with calorie information
- 🎯 **Confidence Scoring** - Shows detection confidence for each item
- 📱 **Responsive Design** - Works on desktop and mobile devices

## 🖼️ Screenshots

### Upload Interface
Beautiful drag-and-drop interface for uploading food images.

### Detection Results
Annotated images with bounding boxes, calorie counts, and confidence scores.

### Calorie Database
Searchable database of 150+ food items with nutritional information.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended)
- Optional: CUDA-capable GPU for faster inference

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Count-Your-Calories.git
cd Count-Your-Calories
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv8 model**
```bash
# The model will be automatically downloaded on first run
# Or manually download:
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### Running the Application

1. **Start the Flask server**
```bash
python app.py
```

2. **Open your browser**
```
http://localhost:5000
```

3. **Upload a food image and analyze!**

## 📁 Project Structure

```
Count-Your-Calories/
├── app.py                  # Flask backend server
├── index.html              # Frontend interface
├── styles.css              # Styling
├── script.js               # Frontend logic
├── requirements.txt        # Python dependencies
├── model_manager.py        # Model management utility
├── train_food_model.py     # Model training script
├── README.md               # Documentation
├── uploads/                # Uploaded images
├── outputs/                # Processed images
├── models/                 # YOLO models
└── datasets/               # Training datasets
```

## 🛠️ Configuration

### Model Selection

The application uses YOLOv8 Medium by default. You can change the model in `app.py`:

```python
# Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
primary_model = YOLO('yolov8m.pt')
```

**Model Comparison:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6MB | ⚡⚡⚡ | ⭐⭐ | Quick testing |
| YOLOv8s | 22MB | ⚡⚡ | ⭐⭐⭐ | Balanced |
| YOLOv8m | 52MB | ⚡ | ⭐⭐⭐⭐ | Recommended ✓ |
| YOLOv8l | 87MB | 🐌 | ⭐⭐⭐⭐⭐ | High accuracy |
| YOLOv8x | 136MB | 🐌🐌 | ⭐⭐⭐⭐⭐ | Maximum accuracy |

### Calorie Database

Update calorie values in `app.py`:

```python
CALORIE_DATABASE = {
    'pizza': 266,
    'burger': 295,
    'sushi': 143,
    # Add more items...
}
```

## 📊 API Endpoints

### POST `/api/detect`
Detect food items in an image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
```json
{
  "detected_items": [
    {
      "name": "Pizza",
      "confidence": 92.5,
      "calories": 266,
      "bbox": [x1, y1, x2, y2],
      "area_ratio": 15.2
    }
  ],
  "total_calories": 532,
  "num_items": 2,
  "annotated_image": "data:image/jpeg;base64,...",
  "avg_confidence": 89.3
}
```

### GET `/api/calorie-database`
Retrieve the calorie database.

**Response:**
```json
{
  "pizza": 266,
  "burger": 295,
  "sushi": 143
}
```

### GET `/api/model-info`
Get information about the loaded model.

**Response:**
```json
{
  "model": "YOLOv8 Medium",
  "framework": "Ultralytics",
  "classes": 80,
  "food_database_size": 150
}
```

## 🎓 Training Custom Models

### Using Model Manager

```bash
python model_manager.py
```

Features:
- List available models
- Download different YOLO versions
- Test models on sample images
- Compare model performance
- Get recommendations

### Training on Custom Dataset

```bash
python train_food_model.py
```

**Dataset Requirements:**
- Minimum 100-200 images per class
- YOLO format annotations (class x y w h)
- 80/20 train/validation split

**Recommended Datasets:**
- [Roboflow Food Datasets](https://universe.roboflow.com/) (pre-annotated)
- UEC Food-256 (256 food categories)
- Food-11 (11 categories with annotations)

### Training Example

```python
from train_food_model import FoodModelTrainer

trainer = FoodModelTrainer(base_model='yolov8m.pt')
trainer.setup_dataset_structure()

# Train model
results = trainer.train_model(
    data_yaml='datasets/food/food_dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640
)
```

## 🔧 Advanced Features

### GPU Acceleration

Enable GPU support for faster inference:

```python
# In app.py, update the detect_food_items function:
results = primary_model(
    img,
    device='cuda',  # Change from 'cpu' to 'cuda'
    ...
)
```

### Batch Processing

Process multiple images:

```python
import glob
from pathlib import Path

images = glob.glob('path/to/images/*.jpg')
for img_path in images:
    result, error = detect_food_items(img_path)
    if result:
        print(f"{img_path}: {result['total_calories']} kcal")
```

### Export Model

Export to different formats:

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# ONNX (faster inference)
model.export(format='onnx')

# TensorRT (4x faster on GPU)
model.export(format='engine')

# CoreML (iOS deployment)
model.export(format='coreml')
```

## 🐛 Troubleshooting

### Model Download Issues

```bash
# Manually download model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### Port Already in Use

```python
# Change port in app.py
app.run(debug=True, port=5001)  # Change from 5000
```

### Low Detection Accuracy

1. Use a larger model (yolov8l or yolov8x)
2. Adjust confidence threshold in `app.py`:
   ```python
   results = primary_model(img, conf=0.20)  # Lower from 0.25
   ```
3. Train a custom model on food-specific dataset

### Memory Issues

1. Use smaller model (yolov8n or yolov8s)
2. Reduce batch size during training
3. Resize images before processing

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- 🎯 Improve calorie estimation accuracy
- 🍕 Expand food database
- 🌍 Add multi-language support
- 📱 Create mobile app version
- 🎨 Enhance UI/UX design
- 🧪 Add unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library
- Food calorie data from USDA FoodData Central


## 🗺️ Roadmap

- [ ] Add nutrition facts (protein, carbs, fats)
- [ ] Implement meal history tracking
- [ ] Add barcode scanning for packaged foods
- [ ] Create meal planning recommendations
- [ ] Add dietary restriction filters (vegan, gluten-free, etc.)
- [ ] Implement user accounts and profiles
- [ ] Add social sharing features
- [ ] Mobile app (iOS/Android)
- [ ] Integration with fitness trackers

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Flask Documentation](https://flask.palletsprojects.com/en/3.0.x/)
- [Computer Vision Basics](https://opencv.org/courses/)
- [Food Nutrition APIs](https://developer.nutritionix.com/)

---
