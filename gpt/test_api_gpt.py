import requests
import logging
import time

BASE_URL = "http://127.0.0.1:8000"
url = f"{BASE_URL}/infer"

logging.basicConfig(filename='gpt/gpt_log.txt', level=logging.INFO)

def api_call(input_text:str):
    params = {'input_txt': input_text}

    max_retries = 3
    retry_delay = 10

    for i in range(max_retries):
        try:
            start_time = time.time()
            response = requests.request("POST", url, params=params)
            end_time = time.time()
            response_time = end_time - start_time
            response.raise_for_status()  # Raise an exception for non-2xx responses
            return response, response_time
        except requests.exceptions.RequestException:
            time.sleep(retry_delay)
            continue
    raise ConnectionError("API call failed after multiple retries.")


def test_api_calls(num_calls=100, input_text='hello'):
    total_response_time = 0

    for i in range(num_calls):
        response, response_time = api_call(input_text)
        total_response_time += response_time

        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, str)

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