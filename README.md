# NVML REST API

A FastAPI-based REST API for monitoring NVIDIA GPUs using the NVIDIA Management Library (NVML) via the nvidia-ml-py package.

## Features

- Get information about all available NVIDIA GPUs
- Monitor GPU memory usage
- Check GPU utilization
- Query device capabilities
- Mock mode for development without NVIDIA GPUs

## Installation

```bash
pip install -r requirements.txt
```

## Docker

You can also run the API using Docker:

### Build the Docker image

```bash
docker build -t nvml-rest-api .
```

### Run the container

```bash
docker run -d --gpus all -p 8000:8000 --name nvml-api nvml-rest-api
```

The `--gpus all` flag passes through all GPUs to the container. You can also specify individual GPUs if needed.

### View logs

```bash
docker logs -f nvml-api
```

### Stop the container

```bash
docker stop nvml-api
```

## Usage

Start the server:

```bash
uvicorn nvml_rest_api.main:app --reload
```

Access the API documentation at http://localhost:8000/docs

### Mock Mode

The API automatically falls back to mock mode when:
- NVIDIA GPUs are not available
- NVML library is not found
- NVIDIA drivers are not installed

In mock mode, the API provides simulated GPU data for testing and development.

## API Endpoints

- `GET /api/v1/gpus`: List all available GPUs
- `GET /api/v1/gpus/{device_id}`: Get detailed information about a specific GPU
- `GET /api/v1/gpus/{device_id}/memory`: Get memory information for a specific GPU
- `GET /api/v1/gpus/{device_id}/utilization`: Get utilization metrics for a specific GPU
- `GET /api/v1/status`: Get system status information including mock mode

Example Output:
```js
// example_url: "http://127.0.0.1:8000/api/v1/gpus"
response = {
    "count": 1,
    "gpus": [
        {
            "id": 0,
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": "GPU-1f1ad567-3b5c-9e6d-ee7c-8f4d4b1c790e",
            "memory": {
                "total": 25769803776,
                "free": 18024562688,
                "used": 7745241088
            },
            "utilization": { "gpu": 18, "memory": 23 },
            "power_usage": 31.99,
            "power_limit": 370.0,
            "temperature": 48,
            "fan_speed": 0,
            "performance_state": "P8",
            "compute_mode": "Default",
            "persistence_mode": false
        }
    ]
}
```
