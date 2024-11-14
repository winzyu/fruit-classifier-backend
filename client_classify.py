# client.py
import requests
import time
import sys
import os
from tqdm import tqdm

class FruitClassifierClient:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.available_models = self.get_available_models()
    
    def get_available_models(self):
        """Get list of available models from server"""
        try:
            response = requests.get(f"{self.server_url}/models")
            if response.status_code == 200:
                return response.json()['models']
            else:
                print(f"Error getting models: {response.json()['error']}")
                return []
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to server. Make sure the server is running.")
            sys.exit(1)
    
    def flip_strawberry_prediction(self, prediction):
        """Flip the prediction for strawberry model"""
        return 'Rotten' if prediction == 'Fresh' else 'Fresh'
    
    def predict(self, image_path, fruit_name):
        """Send image to server for prediction"""
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file '{image_path}' does not exist")
            return
        
        print(f"\n🔄 Processing {os.path.basename(image_path)}...")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                
                with tqdm(total=1, desc="Sending request", unit="request") as pbar:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.server_url}/predict/{fruit_name}",
                        files=files
                    )
                    pbar.update(1)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Special case for strawberry model
                    if fruit_name.lower() == 'strawberry':
                        original_prediction = result['prediction']
                        result['prediction'] = self.flip_strawberry_prediction(original_prediction)
                        # print("\n🔄 Note: Flipped prediction for strawberry model")
                        # print(f"  • Original prediction: {original_prediction}")
                    
                    print("\n📊 Results:")
                    print(f"🍎 Fruit: {result['fruit']}")
                    print(f"📋 Prediction: {result['prediction']}")
                    print("\n⏱️ Timing:")
                    print(f"  • Image processing: {result['timing']['image_processing']}")
                    print(f"  • Prediction: {result['timing']['prediction']}")
                    print(f"  • Total server time: {result['timing']['total']}")
                    print(f"  • Total request time: {time.time() - start_time:.3f}s")
                else:
                    print(f"❌ Error: {response.json()['error']}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to server. Make sure the server is running.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    # Set up argument parsing
    if len(sys.argv) < 2:
        print("\n📋 Usage:")
        print("  python client.py <image_path> [fruit_name]")
        print("  python client.py --list-fruits")
        print("\n📝 Examples:")
        print("  python client.py test_images/apple.jpg")
        print("  python client.py test_images/apple.jpg apple")
        print("  python client.py --list-fruits")
        sys.exit(1)
    
    client = FruitClassifierClient()
    
    # Handle --list-fruits flag
    if sys.argv[1] == '--list-fruits':
        print("\n🍎 Available fruit/vegetable classifiers:")
        for i, model in enumerate(client.available_models, 1):
            print(f"  {i}. {model}")
        return
    
    image_path = sys.argv[1]
    
    # Handle fruit name argument or prompt for selection
    if len(sys.argv) > 2:
        fruit_name = sys.argv[2]
        if fruit_name not in client.available_models:
            print(f"❌ Error: No classifier found for {fruit_name}")
            print("\n🍎 Available classifiers:")
            for model in client.available_models:
                print(f"  • {model}")
            return
    else:
        print("\n🍎 Available fruit/vegetable classifiers:")
        for i, model in enumerate(client.available_models, 1):
            print(f"  {i}. {model}")
        
        while True:
            try:
                choice = int(input("\n📝 Select a fruit/vegetable (enter number): ")) - 1
                if 0 <= choice < len(client.available_models):
                    fruit_name = client.available_models[choice]
                    break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    # Make prediction
    client.predict(image_path, fruit_name)

if __name__ == "__main__":
    main()
