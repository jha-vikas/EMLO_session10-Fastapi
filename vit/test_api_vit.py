import requests
import logging
import time


BASE_URL = "http://127.0.0.1:8080"
url = f"{BASE_URL}/infer"

logging.basicConfig(filename='vit/vit_log.txt', level=logging.INFO)

def api_call(image_loc:str):
    payload = {}
    files=[
    ('image',('image_test.jpg',open(image_loc,'rb'),'image/jpeg'))
    ]
    headers = {}

    max_retries = 3
    retry_delay = 10


    for i in range(max_retries):
        try:
            start_time = time.time()
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            end_time = time.time()
            response_time = end_time - start_time
            response.raise_for_status()  # Raise an exception for non-2xx responses
            return response, response_time
        except requests.exceptions.RequestException:
            time.sleep(retry_delay)
            continue
    raise ConnectionError("API call failed after multiple retries.")


def test_api_calls(num_calls=100, image_loc='./vit/image_test.jpg'):
    total_response_time = 0

    with open('./vit/cifar10_classes.txt', "r") as f:
            catgs = [s.strip() for s in f.readlines()]


    for i in range(num_calls):
        response, response_time = api_call(image_loc)
        total_response_time += response_time

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        data = response.json()
        assert isinstance(data, dict)

        # Add more assertions as needed based on the expected response format
        assert all(class_name in data for class_name in catgs)

        # Log the response time for each API call
        logging.info(f"API Call {i + 1}: Response Time: {response_time:.3f} seconds")

    average_response_time = total_response_time / num_calls
    logging.info(f"Average Response Time after {num_calls} API calls: {average_response_time:.3f} seconds")
    

def test_health_api():
    url = f"{BASE_URL}/health"
    response = requests.get(url)

    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


if __name__ == "__main__":
     test_api_calls(num_calls=100)
