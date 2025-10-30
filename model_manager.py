"""
Food Detection Model Manager
Easy switching between different YOLO models
"""

import os
from pathlib import Path
from ultralytics import YOLO
import requests
from tqdm import tqdm
import json

class ModelManager:
    def __init__(self):
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / 'models_config.json'
        self.models = self.load_config()
        
    def load_config(self):
        """Load models configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self):
        """Save models configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def download_model(self, url, model_name, description=""):
        """Download a model from URL"""
        model_path = self.models_dir / f"{model_name}.pt"
        
        if model_path.exists():
            print(f"✓ Model already exists: {model_name}")
            return str(model_path)
        
        print(f"Downloading {model_name}...")
        
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                desc=model_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            # Save to config
            self.models[model_name] = {
                'path': str(model_path),
                'url': url,
                'description': description
            }
            self.save_config()
            
            print(f"✓ Downloaded: {model_name}")
            return str(model_path)
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            if model_path.exists():
                model_path.unlink()
            return None
    
    def list_models(self):
        """List all available models"""
        print("\n" + "="*70)
        print("Available Models")
        print("="*70)
        
        # Official YOLOv8 models
        official_models = {
            'yolov8n': 'YOLOv8 Nano - Fastest (3.2M params)',
            'yolov8s': 'YOLOv8 Small - Balanced (11.2M params)',
            'yolov8m': 'YOLOv8 Medium - Good accuracy (25.9M params) ⭐',
            'yolov8l': 'YOLOv8 Large - Better accuracy (43.7M params)',
            'yolov8x': 'YOLOv8 X-Large - Best accuracy (68.2M params)',
        }
        
        print("\n📦 Official YOLO Models:")
        for i, (name, desc) in enumerate(official_models.items(), 1):
            status = "✓" if Path(f"{name}.pt").exists() else " "
            print(f"  {status} {i}. {name}.pt - {desc}")
        
        # Custom models
        if self.models:
            print("\n🎯 Custom Food Detection Models:")
            for i, (name, info) in enumerate(self.models.items(), 1):
                status = "✓" if Path(info['path']).exists() else "✗"
                print(f"  {status} {len(official_models)+i}. {name}")
                print(f"      {info.get('description', 'No description')}")
        
        print("="*70 + "\n")
    
    def download_official_model(self, model_size='m'):
        """Download official YOLOv8 model"""
        model_name = f'yolov8{model_size}'
        print(f"Downloading {model_name}.pt...")
        
        try:
            model = YOLO(f'{model_name}.pt')
            print(f"✓ Model ready: {model_name}.pt")
            return f'{model_name}.pt'
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    def test_model(self, model_path, test_image='test_food.jpg'):
        """Test a model on a sample image"""
        if not Path(model_path).exists():
            print(f"✗ Model not found: {model_path}")
            return
        
        if not Path(test_image).exists():
            print(f"✗ Test image not found: {test_image}")
            print("Please provide a test image")
            return
        
        print(f"\nTesting model: {model_path}")
        print("="*60)
        
        try:
            model = YOLO(model_path)
            results = model(test_image, verbose=False)
            
            print(f"✓ Model loaded successfully")
            print(f"Classes: {len(model.names)}")
            
            for result in results:
                boxes = result.boxes
                print(f"\nDetections: {len(boxes)}")
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    print(f"  - {name}: {conf*100:.1f}%")
            
            print("="*60)
            
        except Exception as e:
            print(f"✗ Testing failed: {e}")
    
    def compare_models(self, test_image='test_food.jpg'):
        """Compare multiple models on the same image"""
        if not Path(test_image).exists():
            print("Please provide a test image")
            return
        
        models_to_test = []
        
        # Check official models
        for size in ['n', 's', 'm', 'l', 'x']:
            model_path = f'yolov8{size}.pt'
            if Path(model_path).exists():
                models_to_test.append((f'YOLOv8{size.upper()}', model_path))
        
        # Check custom models
        for name, info in self.models.items():
            if Path(info['path']).exists():
                models_to_test.append((name, info['path']))
        
        if not models_to_test:
            print("No models found to compare")
            return
        
        print("\n" + "="*70)
        print("Model Comparison")
        print("="*70)
        
        results_comparison = []
        
        for model_name, model_path in models_to_test:
            print(f"\nTesting: {model_name}")
            try:
                model = YOLO(model_path)
                import time
                
                start = time.time()
                results = model(test_image, verbose=False)
                inference_time = (time.time() - start) * 1000
                
                num_detections = len(results[0].boxes)
                avg_confidence = 0
                if num_detections > 0:
                    confidences = [float(box.conf[0]) for box in results[0].boxes]
                    avg_confidence = sum(confidences) / len(confidences)
                
                results_comparison.append({
                    'model': model_name,
                    'detections': num_detections,
                    'avg_conf': avg_confidence,
                    'time_ms': inference_time
                })
                
                print(f"  Detections: {num_detections}")
                print(f"  Avg Confidence: {avg_confidence*100:.1f}%")
                print(f"  Inference Time: {inference_time:.1f}ms")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"{'Model':<20} {'Detections':<12} {'Avg Conf':<12} {'Time (ms)':<10}")
        print("-"*70)
        
        for result in sorted(results_comparison, key=lambda x: x['avg_conf'], reverse=True):
            print(f"{result['model']:<20} {result['detections']:<12} "
                  f"{result['avg_conf']*100:>6.1f}%      {result['time_ms']:>6.1f}")
        
        print("="*70 + "\n")
    
    def recommend_model(self):
        """Recommend best model based on requirements"""
        print("\n" + "="*70)
        print("Model Recommendations")
        print("="*70)
        
        recommendations = [
            {
                'use_case': '🚀 Quick Testing / Development',
                'model': 'yolov8n.pt',
                'reason': 'Fastest inference, good for prototyping'
            },
            {
                'use_case': '⚖️ Balanced Performance',
                'model': 'yolov8m.pt',
                'reason': 'Best balance of speed and accuracy (Current choice ✓)'
            },
            {
                'use_case': '🎯 Maximum Accuracy',
                'model': 'yolov8x.pt',
                'reason': 'Best detection quality, slower inference'
            },
            {
                'use_case': '🍕 Food-Specific Detection',
                'model': 'Custom trained model',
                'reason': 'Fine-tuned on food dataset, 70-85% mAP'
            },
            {
                'use_case': '📱 Mobile / Edge Devices',
                'model': 'yolov8n.pt (quantized)',
                'reason': 'Smallest size, fastest on CPU'
            },
            {
                'use_case': '🖥️ Server / GPU Deployment',
                'model': 'yolov8l.pt or yolov8x.pt',
                'reason': 'High accuracy with GPU acceleration'
            }
        ]
        
        for rec in recommendations:
            print(f"\n{rec['use_case']}")
            print(f"  → Model: {rec['model']}")
            print(f"  → Reason: {rec['reason']}")
        
        print("\n" + "="*70 + "\n")


def main():
    manager = ModelManager()
    
    while True:
        print("""
╔════════════════════════════════════════════════════════════╗
║            Food Detection Model Manager                   ║
╚════════════════════════════════════════════════════════════╝

1. List available models
2. Download official YOLO model
3. Download custom food model (from URL)
4. Test a model
5. Compare models
6. Get recommendations
7. Exit

""")
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            manager.list_models()
            
        elif choice == '2':
            print("\nAvailable sizes: n, s, m, l, x")
            size = input("Enter model size (default: m): ").strip() or 'm'
            manager.download_official_model(size)
            
        elif choice == '3':
            url = input("Enter model URL: ").strip()
            name = input("Enter model name: ").strip()
            desc = input("Enter description (optional): ").strip()
            manager.download_model(url, name, desc)
            
        elif choice == '4':
            model_path = input("Enter model path (e.g., yolov8m.pt): ").strip()
            test_image = input("Enter test image path (default: test_food.jpg): ").strip()
            manager.test_model(model_path, test_image or 'test_food.jpg')
            
        elif choice == '5':
            test_image = input("Enter test image path (default: test_food.jpg): ").strip()
            manager.compare_models(test_image or 'test_food.jpg')
            
        elif choice == '6':
            manager.recommend_model()
            
        elif choice == '7':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice!")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()