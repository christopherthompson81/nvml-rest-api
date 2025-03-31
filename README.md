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
            "name": "Unknown",
            "uuid": "Unknown",
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