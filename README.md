# FrescAI Backend - Fruit Freshness Classifier

Backend service for the FrescAI application that serves deep learning models for fruit freshness classification.

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/winzyu/AISC_fruits.git
cd fruit-classifier-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up models:
```bash
mkdir models
# Add your trained *_classifier_resnet18.pth models to the models directory
```

5. Start the server:
```bash
python server.py
```

The server will be available at `http://localhost:5000`

## 📚 API Documentation

### GET /models
Returns list of available classification models.

Response:
```json
{
    "models": ["apple", "banana", "orange"]
}
```

### POST /predict/<fruit_name>
Classifies an image for a specific fruit type.

- **URL Parameter**: fruit_name (string) - Name of the fruit model to use
- **Body**: multipart/form-data with 'file' containing the image
- **Supported formats**: PNG, JPG, JPEG

Response:
```json
{
    "fruit": "apple",
    "prediction": "Fresh",
    "timing": {
        "image_processing": "0.123s",
        "prediction": "0.456s",
        "total": "0.579s"
    }
}
```

## 🛠️ Technologies Used

- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **Pillow**: Image processing
- **NumPy**: Numerical computations
- **Flask-CORS**: CORS support

## 📁 Project Structure

```
AISC_fruits/
├── models/                # Trained PyTorch models
├── server.py             # Flask application
└── requirements.txt      # Python dependencies
```

## 🔗 Related Repositories

- [fruit-classifier-frontend](https://github.com/your-username/fruit-classifier-frontend): React frontend for this service

## 🤝 Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details
