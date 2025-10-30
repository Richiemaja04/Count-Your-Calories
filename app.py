from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import base64
import os
from pathlib import Path
import json
import torch
import requests
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_FOLDER = 'models'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
Path(MODEL_FOLDER).mkdir(exist_ok=True)

# Initialize models
primary_model = None
backup_model = None

# Food-101 class names (subset of most common foods)
FOOD101_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

def download_food_model():
    """Download a pre-trained food detection model"""
    print("Attempting to download food-specific model...")
    
    # Try multiple sources for food detection models
    model_sources = [
        {
            'name': 'YOLOv8 Food-101',
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'file': 'yolov8n-food.pt'
        }
    ]
    
    # For now, we'll use the standard YOLOv8 and fine-tune detection
    # In production, you would download a food-specific model
    try:
        # Use YOLOv8 medium for better accuracy
        model = YOLO('yolov8m.pt')
        print("✓ Loaded YOLOv8 Medium model")
        return model
    except Exception as e:
        print(f"✗ Error loading medium model: {e}")
        try:
            model = YOLO('yolov8n.pt')
            print("✓ Loaded YOLOv8 Nano model (backup)")
            return model
        except Exception as e2:
            print(f"✗ Error loading nano model: {e2}")
            return None

# Load models
print("=" * 60)
print("Loading AI Models...")
print("=" * 60)

primary_model = download_food_model()

# Enhanced calorie database with more accurate values
CALORIE_DATABASE = {
    # Fruits (per 100g or average serving)
    'apple': 52, 'apple_pie': 237, 'banana': 89, 'orange': 47, 
    'strawberry': 32, 'watermelon': 30, 'grapes': 69, 'mango': 60,
    'pineapple': 50, 'peach': 39, 'pear': 57, 'cherry': 50,
    
    # Main dishes (per serving)
    'pizza': 266, 'hamburger': 295, 'hot_dog': 290, 'sandwich': 250,
    'spaghetti': 221, 'spaghetti_bolognese': 195, 'spaghetti_carbonara': 385,
    'pasta': 200, 'rice': 130, 'fried_rice': 228, 'noodles': 138,
    'lasagna': 135, 'ravioli': 175, 'gnocchi': 130, 'risotto': 143,
    'burrito': 206, 'breakfast_burrito': 326, 'taco': 226, 'tacos': 226,
    'quesadilla': 180, 'chicken_quesadilla': 190, 'nachos': 346,
    
    # Asian cuisine
    'sushi': 143, 'sashimi': 127, 'ramen': 436, 'pho': 215,
    'pad_thai': 355, 'bibimbap': 490, 'dumplings': 41, 'gyoza': 41,
    'spring_rolls': 140, 'samosa': 252, 'curry': 190, 'chicken_curry': 190,
    'miso_soup': 40, 'hot_and_sour_soup': 91, 'takoyaki': 85,
    'peking_duck': 337, 'edamame': 122,
    
    # Proteins (per serving)
    'chicken': 165, 'chicken_wings': 203, 'beef': 250, 'steak': 271,
    'pork_chop': 231, 'filet_mignon': 227, 'prime_rib': 346,
    'fish': 206, 'grilled_salmon': 206, 'tuna_tartare': 108,
    'egg': 78, 'eggs': 78, 'omelette': 154, 'eggs_benedict': 450,
    'deviled_eggs': 62, 'tofu': 76, 'beans': 127,
    
    # Seafood
    'fish_and_chips': 585, 'fried_calamari': 175, 'shrimp': 99,
    'shrimp_and_grits': 420, 'crab_cakes': 160, 'oysters': 41,
    'mussels': 172, 'scallops': 111, 'lobster_bisque': 194,
    'lobster_roll_sandwich': 436, 'clam_chowder': 119,
    
    # Vegetables & Salads (per serving)
    'broccoli': 55, 'carrot': 41, 'tomato': 18, 'lettuce': 5,
    'potato': 77, 'french_fries': 312, 'corn': 86, 'peas': 81,
    'salad': 50, 'caesar_salad': 170, 'greek_salad': 106,
    'caprese_salad': 182, 'beet_salad': 90, 'seaweed_salad': 67,
    'coleslaw': 152, 'onion_rings': 411, 'garlic_bread': 186,
    
    # Snacks & Desserts (per serving)
    'cake': 257, 'chocolate_cake': 352, 'carrot_cake': 408,
    'red_velvet_cake': 478, 'cheesecake': 321, 'cup_cakes': 305,
    'cookie': 142, 'macarons': 97, 'donut': 195, 'donuts': 195,
    'ice_cream': 207, 'frozen_yogurt': 159, 'chocolate_mousse': 170,
    'tiramisu': 240, 'panna_cotta': 221, 'creme_brulee': 223,
    'bread_pudding': 212, 'churros': 237, 'beignets': 320,
    'cannoli': 204, 'baklava': 334, 'strawberry_shortcake': 341,
    'chips': 152, 'popcorn': 375, 'chocolate': 235, 'candy': 150,
    'waffles': 291, 'pancakes': 227, 'french_toast': 282,
    
    # Sandwiches & Wraps
    'club_sandwich': 590, 'grilled_cheese_sandwich': 291,
    'pulled_pork_sandwich': 506, 'cheese_plate': 380,
    
    # Appetizers
    'bruschetta': 106, 'falafel': 333, 'hummus': 166,
    'guacamole': 160, 'escargots': 47, 'foie_gras': 462,
    'ceviche': 140, 'beef_carpaccio': 135, 'beef_tartare': 197,
    
    # Beverages (per serving)
    'coffee': 2, 'tea': 2, 'juice': 112, 'soda': 150, 'coke': 140,
    'milk': 103, 'water': 0, 'wine': 85, 'beer': 153,
    
    # Bread & Bakery (per serving)
    'bread': 79, 'croissant': 406, 'muffin': 174, 'bagel': 289,
    
    # Other popular items
    'soup': 86, 'french_onion_soup': 117, 'cheese': 402,
    'macaroni_and_cheese': 166, 'yogurt': 59, 'cereal': 379,
    'poutine': 510, 'paella': 200, 'huevos_rancheros': 460,
    
    # Common COCO classes mapping
    'bowl': 100, 'cup': 50, 'bottle': 50, 'wine_glass': 50,
    'banana': 89, 'apple': 52, 'orange': 47, 'broccoli': 55,
    'carrot': 41, 'hot_dog': 290, 'pizza': 266, 'donut': 195,
    'cake': 257, 'sandwich': 250,
}

def get_calorie_estimate(food_name, confidence=0.0):
    """Get calorie estimate for detected food item with confidence adjustment"""
    food_name_lower = food_name.lower().replace('_', ' ').replace('-', ' ')
    
    # Direct match
    if food_name_lower in CALORIE_DATABASE:
        return CALORIE_DATABASE[food_name_lower]
    
    # Try without spaces
    food_key = food_name_lower.replace(' ', '_')
    if food_key in CALORIE_DATABASE:
        return CALORIE_DATABASE[food_key]
    
    # Partial match
    for key in CALORIE_DATABASE:
        key_normalized = key.replace('_', ' ')
        food_normalized = food_name_lower.replace('_', ' ')
        if key_normalized in food_normalized or food_normalized in key_normalized:
            return CALORIE_DATABASE[key]
    
    # Default estimate based on food type inference
    if any(word in food_name_lower for word in ['cake', 'dessert', 'ice', 'chocolate']):
        return 250  # Desserts average
    elif any(word in food_name_lower for word in ['salad', 'vegetable', 'green']):
        return 80   # Vegetables average
    elif any(word in food_name_lower for word in ['meat', 'beef', 'pork', 'chicken']):
        return 200  # Protein average
    elif any(word in food_name_lower for word in ['pasta', 'rice', 'noodle', 'bread']):
        return 180  # Carbs average
    
    return 150  # General default

def post_process_detections(results, img_shape):
    """Enhanced post-processing with NMS and filtering"""
    detected_items = []
    
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
            
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Get class name
            class_name = primary_model.names[cls]
            
            # Get calorie estimate
            calories = get_calorie_estimate(class_name, conf)
            
            # Skip non-food items
            if calories == 0:
                continue
            
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate box area (for portion estimation)
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            img_area = img_shape[0] * img_shape[1]
            area_ratio = box_area / img_area
            
            # Adjust calories based on visible portion size
            if area_ratio < 0.05:  # Small portion
                calories = int(calories * 0.7)
            elif area_ratio > 0.3:  # Large portion
                calories = int(calories * 1.3)
            
            detected_items.append({
                'name': class_name.replace('_', ' ').title(),
                'confidence': round(conf * 100, 2),
                'calories': calories,
                'bbox': [x1, y1, x2, y2],
                'area_ratio': round(area_ratio * 100, 2)
            })
    
    return detected_items

def detect_food_items(image_path):
    """Enhanced food detection with better model and post-processing"""
    if primary_model is None:
        return None, "Model not loaded"
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not read image"
    
    img_shape = img.shape
    
    # Run inference with optimized parameters
    results = primary_model(
        img,
        conf=0.25,        # Lower threshold for more detections
        iou=0.45,         # NMS threshold
        max_det=50,       # Maximum detections
        device='cpu',     # Use 'cuda' if GPU available
        verbose=False
    )
    
    # Post-process detections
    detected_items = post_process_detections(results, img_shape)
    
    if not detected_items:
        return None, "No food items detected in the image"
    
    total_calories = sum(item['calories'] for item in detected_items)
    
    # Draw enhanced annotations
    for item in detected_items:
        x1, y1, x2, y2 = item['bbox']
        
        # Color coding based on calorie content
        if item['calories'] < 100:
            color = (0, 255, 0)      # Green - Low calorie
        elif item['calories'] < 250:
            color = (0, 165, 255)    # Orange - Medium
        else:
            color = (0, 0, 255)      # Red - High calorie
        
        # Draw bounding box with thickness based on confidence
        thickness = 2 if item['confidence'] > 70 else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Create multi-line label
        label_lines = [
            f"{item['name']}",
            f"{item['calories']} kcal",
            f"{item['confidence']:.1f}%"
        ]
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 18
        
        # Draw label background
        max_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] 
                        for line in label_lines])
        label_height = line_height * len(label_lines) + 10
        
        cv2.rectangle(
            img, 
            (x1, y1 - label_height), 
            (x1 + max_width + 10, y1), 
            color, 
            -1
        )
        
        # Draw label text
        for i, line in enumerate(label_lines):
            y_offset = y1 - label_height + 15 + (i * line_height)
            cv2.putText(
                img, line, (x1 + 5, y_offset),
                font, font_scale, (255, 255, 255), font_thickness
            )
    
    # Add enhanced summary panel
    panel_height = 120
    panel_width = 450
    cv2.rectangle(img, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
    
    # Summary text
    summary_lines = [
        f"Total Items: {len(detected_items)}",
        f"Total Calories: {total_calories} kcal",
        f"Model: YOLOv8 Medium",
        f"Avg Confidence: {np.mean([item['confidence'] for item in detected_items]):.1f}%"
    ]
    
    for i, line in enumerate(summary_lines):
        y_pos = 35 + (i * 22)
        cv2.putText(
            img, line, (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    
    # Save annotated image
    output_path = os.path.join(OUTPUT_FOLDER, 'annotated_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    
    return {
        'detected_items': detected_items,
        'total_calories': total_calories,
        'num_items': len(detected_items),
        'output_image': output_path,
        'avg_confidence': round(np.mean([item['confidence'] for item in detected_items]), 2)
    }, None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded file
    filename = 'uploaded_' + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Detect food items
    result, error = detect_food_items(filepath)
    
    if error:
        return jsonify({'error': error}), 500
    
    # Read annotated image and convert to base64
    with open(result['output_image'], 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    result['annotated_image'] = f"data:image/jpeg;base64,{img_data}"
    del result['output_image']
    
    return jsonify(result)

@app.route('/api/calorie-database', methods=['GET'])
def get_calorie_db():
    # Filter out zero-calorie (non-food) items
    food_items = {k: v for k, v in CALORIE_DATABASE.items() if v > 0}
    return jsonify(food_items)

@app.route('/api/update-item', methods=['POST'])
def update_item():
    data = request.json
    item_name = data.get('name', '').lower()
    new_calories = data.get('calories', 0)
    
    if item_name and new_calories > 0:
        CALORIE_DATABASE[item_name] = new_calories
        return jsonify({'message': 'Item updated successfully'})
    
    return jsonify({'error': 'Invalid data'}), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model': 'YOLOv8 Medium',
        'framework': 'Ultralytics',
        'classes': len(primary_model.names) if primary_model else 0,
        'food_database_size': len([k for k, v in CALORIE_DATABASE.items() if v > 0])
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🍽️  Food Detection & Calorie Estimation Server")
    print("=" * 60)
    if primary_model:
        print("✓ Model loaded successfully!")
        print(f"✓ Tracking {len([k for k, v in CALORIE_DATABASE.items() if v > 0])} food items")
    else:
        print("✗ Model failed to load!")
    print("=" * 60)
    print("🌐 Server running on http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=5000, threaded=True)