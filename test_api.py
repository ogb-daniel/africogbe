import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health")
        print(f"Health check: {health_response.json()}")
    except requests.exceptions.ConnectionError:
        print("Server not running. Start with: python main.py")
        return
    
    # Test prediction endpoint
    test_data = {"Age": 1, "WorkingMemory_Score": 80}  # 3 years old

    try:
        response = requests.post(
            f"{base_url}/predict/working-memory",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction for {test_data['Age']} months: {result['predicted']:.2f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_api()