"""Service for interacting with NVIDIA GPUs via NVML."""

import logging
from typing import Dict, List, Optional

import pynvml
from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlInit, nvmlShutdown

from nvml_rest_api.models import GPUInfo, MemoryInfo, UtilizationInfo

logger = logging.getLogger(__name__)


class NVMLService:
    """Service for interacting with NVIDIA GPUs via NVML."""

    def __init__(self):
        """Initialize the NVML service."""
        self.initialized = False
        try:
            nvmlInit()
            self.initialized = True
            logger.info("NVML initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            logger.warning("This could be because no NVIDIA GPUs are available or the NVIDIA driver is not installed.")
            logger.warning("The API will run but will report 0 GPUs.")

    def __del__(self):
        """Clean up NVML on destruction."""
        if self.initialized:
            try:
                nvmlShutdown()
                logger.info("NVML shutdown successfully")
            except Exception as e:
                logger.error(f"Failed to shutdown NVML: {e}")

    def get_device_count(self) -> int:
        """Get the number of NVIDIA GPUs in the system."""
        if not self.initialized:
            return 0
        try:
            return nvmlDeviceGetCount()
        except Exception as e:
            logger.error(f"Failed to get device count: {e}")
            return 0

    def get_device_handle(self, device_id: int):
        """Get the handle for a specific GPU."""
        if not self.initialized:
            return None
        try:
            return nvmlDeviceGetHandleByIndex(device_id)
        except Exception as e:
            logger.error(f"Failed to get device handle for device {device_id}: {e}")
            return None

    def get_device_name(self, handle) -> str:
        """Get the name of a GPU."""
        try:
            return pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get device name: {e}")
            return "Unknown"

    def get_device_uuid(self, handle) -> str:
        """Get the UUID of a GPU."""
        try:
            return pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get device UUID: {e}")
            return "Unknown"

    def get_memory_info(self, handle) -> MemoryInfo:
        """Get memory information for a GPU."""
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return MemoryInfo(
                total=mem_info.total,
                free=mem_info.free,
                used=mem_info.used
            )
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return MemoryInfo(total=0, free=0, used=0)

    def get_utilization_info(self, handle) -> Optional[UtilizationInfo]:
        """Get utilization information for a GPU."""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return UtilizationInfo(
                gpu=util.gpu,
                memory=util.memory
            )
        except Exception as e:
            logger.error(f"Failed to get utilization info: {e}")
            return None

    def get_power_usage(self, handle) -> Optional[float]:
        """Get current power usage of a GPU in milliwatts."""
        try:
            return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception as e:
            logger.error(f"Failed to get power usage: {e}")
            return None

    def get_power_limit(self, handle) -> Optional[float]:
        """Get power management limit of a GPU in milliwatts."""
        try:
            return pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception as e:
            logger.error(f"Failed to get power limit: {e}")
            return None

    def get_temperature(self, handle) -> Optional[int]:
        """Get temperature of a GPU in Celsius."""
        try:
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception as e:
            logger.error(f"Failed to get temperature: {e}")
            return None

    def get_fan_speed(self, handle) -> Optional[int]:
        """Get fan speed percentage of a GPU."""
        try:
            return pynvml.nvmlDeviceGetFanSpeed(handle)
        except Exception as e:
            logger.error(f"Failed to get fan speed: {e}")
            return None

    def get_performance_state(self, handle) -> Optional[str]:
        """Get performance state of a GPU."""
        try:
            state = pynvml.nvmlDeviceGetPerformanceState(handle)
            return f"P{state}"
        except Exception as e:
            logger.error(f"Failed to get performance state: {e}")
            return None

    def get_compute_mode(self, handle) -> Optional[str]:
        """Get compute mode of a GPU."""
        try:
            mode = pynvml.nvmlDeviceGetComputeMode(handle)
            modes = {
                pynvml.NVML_COMPUTEMODE_DEFAULT: "Default",
                pynvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: "Exclusive Thread",
                pynvml.NVML_COMPUTEMODE_PROHIBITED: "Prohibited",
                pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: "Exclusive Process"
            }
            return modes.get(mode, "Unknown")
        except Exception as e:
            logger.error(f"Failed to get compute mode: {e}")
            return None

    def get_persistence_mode(self, handle) -> Optional[bool]:
        """Get persistence mode of a GPU."""
        try:
            mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
            return mode == pynvml.NVML_FEATURE_ENABLED
        except Exception as e:
            logger.error(f"Failed to get persistence mode: {e}")
            return None

    def get_gpu_info(self, device_id: int) -> Optional[GPUInfo]:
        """Get comprehensive information about a GPU."""
        handle = self.get_device_handle(device_id)
        if not handle:
            return None

        memory = self.get_memory_info(handle)
        
        return GPUInfo(
            id=device_id,
            name=self.get_device_name(handle),
            uuid=self.get_device_uuid(handle),
            memory=memory,
            utilization=self.get_utilization_info(handle),
            power_usage=self.get_power_usage(handle),
            power_limit=self.get_power_limit(handle),
            temperature=self.get_temperature(handle),
            fan_speed=self.get_fan_speed(handle),
            performance_state=self.get_performance_state(handle),
            compute_mode=self.get_compute_mode(handle),
            persistence_mode=self.get_persistence_mode(handle)
        )

    def get_all_gpus(self) -> List[GPUInfo]:
        """Get information about all available GPUs."""
        gpu_count = self.get_device_count()
        return [self.get_gpu_info(i) for i in range(gpu_count) if self.get_gpu_info(i) is not None]
