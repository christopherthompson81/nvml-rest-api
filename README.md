# NVML REST API

A FastAPI-based REST API for monitoring NVIDIA GPUs using the NVIDIA Management Library (NVML) via the nvidia-ml-py package.

## Features

- Get information about all available NVIDIA GPUs
- Monitor GPU memory usage
- Check GPU utilization
- Query device capabilities

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

## API Endpoints

- `GET /api/v1/gpus`: List all available GPUs
- `GET /api/v1/gpus/{device_id}`: Get detailed information about a specific GPU
- `GET /api/v1/gpus/{device_id}/memory`: Get memory information for a specific GPU
- `GET /api/v1/gpus/{device_id}/utilization`: Get utilization metrics for a specific GPU
