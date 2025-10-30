"""
Advanced Food Detection Model Training Script
Fine-tune YOLOv8 on Food-101 or custom food datasets
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import os
import shutil
from PIL import Image
import requests
from tqdm import tqdm
import zipfile

class FoodModelTrainer:
    def __init__(self, base_model='yolov8m.pt'):
        """
        Initialize the trainer
        
        Args:
            base_model: Base YOLO model to fine-tune (n/s/m/l/x)
        """
        self.base_model = base_model
        self.model = None
        self.dataset_path = Path('datasets/food')
        
    def setup_dataset_structure(self):
        """Create proper YOLOv8 dataset structure"""
        print("Setting up dataset structure...")
        
        # Create directories
        dirs = [
            self.dataset_path / 'images' / 'train',
            self.dataset_path / 'images' / 'val',
            self.dataset_path / 'labels' / 'train',
            self.dataset_path / 'labels' / 'val'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✓ Dataset structure created")
    
    def download_food101(self):
        """Download and prepare Food-101 dataset"""
        print("Downloading Food-101 dataset...")
        print("Note: This is a 5GB download and may take some time...")
        
        dataset_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        dataset_file = "food-101.tar.gz"
        
        # Download
        if not Path(dataset_file).exists():
            response = requests.get(dataset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dataset_file, 'wb') as file, tqdm(
                desc=dataset_file,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            
            print("✓ Download complete")
        
        # Extract
        print("Extracting dataset...")
        shutil.unpack_archive(dataset_file, '.')
        print("✓ Extraction complete")
    
    def create_yolo_dataset_yaml(self, classes, train_path, val_path):
        """Create YAML configuration for YOLOv8"""
        
        yaml_content = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = self.dataset_path / 'food_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✓ Created dataset YAML at {yaml_path}")
        return yaml_path
    
    def train_model(self, 
                   data_yaml,
                   epochs=100,
                   imgsz=640,
                   batch=16,
                   project='food_models',
                   name='yolov8_food',
                   patience=20):
        """
        Train the food detection model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project folder name
            name: Experiment name
            patience: Early stopping patience
        """
        print("\n" + "="*60)
        print("Starting Model Training")
        print("="*60)
        
        # Load base model
        self.model = YOLO(self.base_model)
        
        # Training parameters
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            patience=patience,
            save=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=4,
            # Optimization parameters
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        return results
    
    def validate_model(self, data_yaml):
        """Validate the trained model"""
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return None
        
        print("\nValidating model...")
        results = self.model.val(data=str(data_yaml))
        
        print("\nValidation Results:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def export_model(self, format='onnx'):
        """Export model to different formats"""
        if self.model is None:
            print("No model loaded.")
            return
        
        print(f"\nExporting model to {format}...")
        self.model.export(format=format)
        print(f"✓ Model exported to {format}")


def quick_train_example():
    """Quick training example with sample dataset"""
    
    # Example food classes (subset for demonstration)
    food_classes = [
        'pizza', 'burger', 'sushi', 'pasta', 'salad',
        'cake', 'ice_cream', 'sandwich', 'hot_dog', 'fries'
    ]
    
    trainer = FoodModelTrainer(base_model='yolov8m.pt')
    
    # Setup dataset
    trainer.setup_dataset_structure()
    
    # Create dataset YAML
    yaml_path = trainer.create_yolo_dataset_yaml(
        classes=food_classes,
        train_path='images/train',
        val_path='images/val'
    )
    
    print("\n" + "="*60)
    print("Dataset Setup Complete")
    print("="*60)
    print("\nNext Steps:")
    print("1. Add your training images to: datasets/food/images/train/")
    print("2. Add your validation images to: datasets/food/images/val/")
    print("3. Add corresponding labels (YOLO format) to labels folders")
    print("4. Run: trainer.train_model(yaml_path)")
    print("\nYOLO Label Format (per line in .txt file):")
    print("class_id center_x center_y width height (all normalized 0-1)")
    print("Example: 0 0.5 0.5 0.3 0.4")
    print("="*60)
    
    return trainer, yaml_path


def train_from_roboflow():
    """
    Train using dataset from Roboflow
    Roboflow provides ready-to-use food datasets in YOLO format
    """
    
    print("="*60)
    print("Training with Roboflow Dataset")
    print("="*60)
    
    # Example: Using Roboflow's food dataset
    # You need to get your API key from roboflow.com
    
    print("\nSteps to use Roboflow dataset:")
    print("1. Go to: https://universe.roboflow.com/")
    print("2. Search for 'food detection' datasets")
    print("3. Select a dataset (e.g., 'Food Detection', 'Fast Food')")
    print("4. Click 'Download' and select 'YOLOv8' format")
    print("5. Copy the code snippet provided by Roboflow")
    print("\nExample Roboflow code:")
    print("""
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("workspace-id").project("project-id")
    dataset = project.version(1).download("yolov8")
    """)
    
    # After downloading, train with:
    # trainer = FoodModelTrainer('yolov8m.pt')
    # trainer.train_model(
    #     data_yaml='path/to/downloaded/data.yaml',
    #     epochs=100
    # )


def convert_food101_to_yolo():
    """
    Convert Food-101 dataset to YOLO format
    Note: Food-101 is classification dataset, needs object detection annotations
    """
    
    print("="*60)
    print("Food-101 to YOLO Conversion")
    print("="*60)
    
    print("\nImportant Notes:")
    print("- Food-101 is an IMAGE CLASSIFICATION dataset")
    print("- It does NOT have bounding box annotations")
    print("- For object detection, you need:")
    print("  1. Bounding box coordinates for each food item")
    print("  2. Use a pre-annotated food detection dataset instead")
    print("\nRecommended Datasets with Object Detection:")
    print("- UEC Food-256 (has bounding boxes)")
    print("- Food-11 (with annotations)")
    print("- Roboflow food datasets (pre-annotated)")
    print("- Create custom annotations using tools like:")
    print("  * LabelImg (https://github.com/heartexlabs/labelImg)")
    print("  * CVAT (https://www.cvat.ai/)")
    print("  * Roboflow (https://roboflow.com/)")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         Food Detection Model Training Script              ║
    ║                  YOLOv8 Fine-tuning                       ║
    ╚════════════════════════════════════════════════════════════╝
    
    Choose an option:
    
    1. Quick Start (Setup dataset structure)
    2. Train from Roboflow dataset
    3. Download Food-101 (classification only)
    4. Training best practices
    5. Exit
    """)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        trainer, yaml_path = quick_train_example()
        
        # Uncomment to start training after adding data:
        # trainer.train_model(yaml_path, epochs=50, batch=16)
        
    elif choice == '2':
        train_from_roboflow()
        
    elif choice == '3':
        trainer = FoodModelTrainer()
        trainer.download_food101()
        convert_food101_to_yolo()
        
    elif choice == '4':
        print("""
        ╔════════════════════════════════════════════════════════════╗
        ║              Training Best Practices                      ║
        ╚════════════════════════════════════════════════════════════╝
        
        1. DATASET PREPARATION:
           - Minimum 100-200 images per class
           - 1500-2000 total images for good results
           - 80/20 train/val split
           - Diverse lighting, angles, backgrounds
           
        2. MODEL SELECTION:
           - YOLOv8n: Fast, lightweight (3.2M params)
           - YOLOv8s: Balanced (11.2M params)
           - YOLOv8m: Better accuracy (25.9M params) ✓ Recommended
           - YOLOv8l: High accuracy (43.7M params)
           - YOLOv8x: Best accuracy (68.2M params)
           
        3. TRAINING PARAMETERS:
           - Epochs: 50-100 (use early stopping)
           - Batch size: 16 (adjust based on GPU memory)
           - Image size: 640x640
           - Learning rate: 0.001 (AdamW optimizer)
           - Patience: 20 epochs
           
        4. DATA AUGMENTATION:
           - Horizontal flips (50%)
           - HSV color jitter
           - Random scaling (±50%)
           - Mosaic augmentation
           - MixUp (optional)
           
        5. EVALUATION METRICS:
           - mAP@50: Should be > 0.7 for good model
           - mAP@50-95: Should be > 0.5
           - Precision & Recall: Balanced is important
           
        6. IMPROVING RESULTS:
           - Add more diverse training data
           - Use pre-trained food detection model
           - Increase image resolution (768 or 1024)
           - Train for more epochs
           - Adjust augmentation parameters
           
        7. DEPLOYMENT:
           - Export to ONNX for faster inference
           - Use TensorRT for production (4x faster)
           - Quantize model for mobile (INT8)
        """)
    
    elif choice == '5':
        print("Exiting...")
    
    else:
        print("Invalid choice!")