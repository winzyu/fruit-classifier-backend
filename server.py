# server.py
from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import io
import time
import os
import warnings
from tqdm import tqdm
from flask_cors import CORS

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.transform = None
        
    def initialize(self):
        """Initialize all models and transforms"""
        print("\n🚀 Starting server initialization...")
        
        # Initialize transform
        print("\n📐 Setting up image transformation pipeline...")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("✅ Transform pipeline ready")
        
        # Get list of model files
        models_dir = 'models'
        model_files = [f for f in os.listdir(models_dir) 
                      if f.endswith('_classifier_resnet18.pth')]
        
        print(f"\n🔍 Found {len(model_files)} models to load")
        
        # Load each model
        for model_file in tqdm(model_files, desc="Loading models"):
            fruit_name = model_file.split('_classifier_resnet18.pth')[0]
            model_path = os.path.join(models_dir, model_file)
            
            try:
                start_time = time.time()
                model = self.load_model(model_path)
                load_time = time.time() - start_time
                
                self.models[fruit_name] = {
                    'model': model,
                    'load_time': load_time
                }
                print(f"✅ Loaded {fruit_name} model in {load_time:.2f} seconds")
                
            except Exception as e:
                print(f"❌ Failed to load {fruit_name} model: {str(e)}")
        
        print("\n📊 Model loading summary:")
        for fruit_name, info in self.models.items():
            print(f"  • {fruit_name}: Loaded in {info['load_time']:.2f} seconds")
        
        print(f"\n✨ Initialization complete! {len(self.models)} models ready for inference")
    
    def load_model(self, model_path):
        """Load a single PyTorch model"""
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    
    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())
    
    def process_image(self, image_bytes):
        """Process image bytes into tensor"""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def predict(self, fruit_name, image_tensor):
        """Make prediction using specified model"""
        if fruit_name not in self.models:
            raise ValueError(f"No model available for {fruit_name}")
            
        with torch.no_grad():
            outputs = self.models[fruit_name]['model'](image_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return 'Fresh' if predicted.item() == 0 else 'Rotten'

# Global model manager instance
model_manager = ModelManager()

@app.route('/models', methods=['GET'])
def get_models():
    """Return list of available models"""
    return jsonify({
        'models': model_manager.get_available_models()
    })

@app.route('/predict/<fruit_name>', methods=['POST'])
def predict(fruit_name):
    """Make prediction for specified fruit"""
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    if fruit_name not in model_manager.models:
        return jsonify({'error': f'No model available for {fruit_name}'}), 404
    
    try:
        # Process image
        image_processing_start = time.time()
        image_bytes = request.files['file'].read()
        image_tensor = model_manager.process_image(image_bytes)
        image_processing_time = time.time() - image_processing_start
        
        # Make prediction
        prediction_start = time.time()
        result = model_manager.predict(fruit_name, image_tensor)
        prediction_time = time.time() - prediction_start
        
        total_time = time.time() - start_time
        
        return jsonify({
            'fruit': fruit_name,
            'prediction': result,
            'timing': {
                'image_processing': f"{image_processing_time:.3f}s",
                'prediction': f"{prediction_time:.3f}s",
                'total': f"{total_time:.3f}s"
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize models before starting server
    model_manager.initialize()
    # Run the server
    print("\n🌐 Starting server on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
