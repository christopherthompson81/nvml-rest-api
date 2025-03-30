"""Data models for the NVML REST API."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class MemoryInfo(BaseModel):
    """GPU memory information."""
    total: int
    free: int
    used: int


class UtilizationInfo(BaseModel):
    """GPU utilization information."""
    gpu: int
    memory: int


class GPUInfo(BaseModel):
    """Basic GPU information."""
    id: int
    name: str
    uuid: str
    memory: MemoryInfo
    utilization: Optional[UtilizationInfo] = None
    power_usage: Optional[float] = None
    power_limit: Optional[float] = None
    temperature: Optional[int] = None
    fan_speed: Optional[int] = None
    performance_state: Optional[str] = None
    compute_mode: Optional[str] = None
    persistence_mode: Optional[bool] = None


class GPUList(BaseModel):
    """List of GPUs with count."""
    count: int
    gpus: List[GPUInfo]


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
