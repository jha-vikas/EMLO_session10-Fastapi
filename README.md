# EMLOv3 | Assignment 10

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)


## FastAPI Deployment with Docker
- GPT & ViT jit models are deployed using FastAPI in containers.
- To run ViT:
`
docker compose -f docker-compose.yml up --build demo_vit
`
- To run GPT:
`
docker compose -f docker-compose.yml up --build demo_gpt
`

## Testing both deployment
- Test 100 calls to each model.
- ViT:
`
python ./vit/test_api_vit.py
`
- GPT:
`
python ./gpt/test_api_gpt.py
`

Average time taken based on 100 calls:
ViT: 0.023 seconds
GPT: 0.873 seconds


## Author

- Vikas Jha