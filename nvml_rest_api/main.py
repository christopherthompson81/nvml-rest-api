"""Main FastAPI application for the NVML REST API."""

import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from nvml_rest_api.models import GPUInfo, GPUList, MemoryInfo, UtilizationInfo
from nvml_rest_api.nvml_service import NVMLService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NVML REST API",
    description="REST API for monitoring NVIDIA GPUs using NVML",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the NVML service
def get_nvml_service():
    service = NVMLService()
    try:
        yield service
    finally:
        # Service will clean up in its __del__ method
        pass


@app.get("/api/v1/gpus", response_model=GPUList, tags=["GPUs"])
def get_all_gpus(nvml_service: NVMLService = Depends(get_nvml_service)):
    """
    Get information about all available GPUs.
    
    Returns a list of all NVIDIA GPUs in the system with basic information.
    """
    gpus = nvml_service.get_all_gpus()
    return GPUList(count=len(gpus), gpus=gpus)


@app.get("/api/v1/gpus/{device_id}", response_model=GPUInfo, tags=["GPUs"])
def get_gpu_info(device_id: int, nvml_service: NVMLService = Depends(get_nvml_service)):
    """
    Get detailed information about a specific GPU.
    
    Parameters:
    - device_id: The index of the GPU (0-based)
    
    Returns detailed information about the specified GPU.
    """
    if device_id < 0 or device_id >= nvml_service.get_device_count():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GPU with ID {device_id} not found"
        )
    
    gpu_info = nvml_service.get_gpu_info(device_id)
    if not gpu_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to get information for GPU with ID {device_id}"
        )
    
    return gpu_info


@app.get("/api/v1/gpus/{device_id}/memory", response_model=MemoryInfo, tags=["GPUs"])
def get_gpu_memory(device_id: int, nvml_service: NVMLService = Depends(get_nvml_service)):
    """
    Get memory information for a specific GPU.
    
    Parameters:
    - device_id: The index of the GPU (0-based)
    
    Returns memory information (total, free, used) for the specified GPU.
    """
    if device_id < 0 or device_id >= nvml_service.get_device_count():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GPU with ID {device_id} not found"
        )
    
    handle = nvml_service.get_device_handle(device_id)
    if not handle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to get handle for GPU with ID {device_id}"
        )
    
    return nvml_service.get_memory_info(handle)


@app.get("/api/v1/gpus/{device_id}/utilization", response_model=UtilizationInfo, tags=["GPUs"])
def get_gpu_utilization(device_id: int, nvml_service: NVMLService = Depends(get_nvml_service)):
    """
    Get utilization metrics for a specific GPU.
    
    Parameters:
    - device_id: The index of the GPU (0-based)
    
    Returns utilization information (GPU and memory utilization percentages) for the specified GPU.
    """
    if device_id < 0 or device_id >= nvml_service.get_device_count():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GPU with ID {device_id} not found"
        )
    
    handle = nvml_service.get_device_handle(device_id)
    if not handle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to get handle for GPU with ID {device_id}"
        )
    
    util_info = nvml_service.get_utilization_info(handle)
    if not util_info:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get utilization information for GPU with ID {device_id}"
        )
    
    return util_info


@app.get("/", tags=["Health"])
def health_check(nvml_service: NVMLService = Depends(get_nvml_service)):
    """
    Health check endpoint.
    
    Returns a simple message to confirm the API is running.
    """
    gpu_count = nvml_service.get_device_count()
    status = "ok" if nvml_service.initialized else "limited"
    message = f"NVML REST API is running with {gpu_count} GPUs detected"
    if not nvml_service.initialized:
        message += " (NVML initialization failed, running in limited mode)"
    
    return {"status": status, "message": message, "gpu_count": gpu_count}
