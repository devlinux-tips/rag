"""
Device detection for embedding computation.
Provides testable device detection abstraction layer.
"""

import logging

from .embeddings import DeviceInfo


class TorchDeviceDetector:
    """
    Production device detector using PyTorch device detection.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_best_device(self, preferred_device: str = "auto") -> DeviceInfo:
        """Detect best available device."""
        if preferred_device != "auto":
            # User specified a device, try to use it
            if self.is_device_available(preferred_device):
                return self._get_device_info(preferred_device)
            else:
                self.logger.warning(f"Preferred device {preferred_device} not available, auto-detecting")

        # Auto-detection logic
        # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU

        if self._is_mps_available():
            return self._get_device_info("mps")
        elif self._is_cuda_available():
            return self._get_device_info("cuda")
        else:
            return self._get_device_info("cpu")

    def is_device_available(self, device: str) -> bool:
        """Check if device is available."""
        if device == "mps":
            return self._is_mps_available()
        elif device == "cuda":
            return self._is_cuda_available()
        elif device == "cpu":
            return True  # CPU is always available
        else:
            return False

    def _is_mps_available(self) -> bool:
        """Check if MPS (Apple Silicon) is available."""
        try:
            import torch

            return (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
            )
        except Exception:
            return False

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import warnings

            import torch

            # Suppress CUDA initialization warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
                return torch.cuda.is_available()
        except Exception:
            return False

    def _get_device_info(self, device: str) -> DeviceInfo:
        """Get detailed information about a device."""
        if device == "mps":
            return self._get_mps_info()
        elif device == "cuda":
            return self._get_cuda_info()
        elif device == "cpu":
            return self._get_cpu_info()
        else:
            raise ValueError(f"Unknown device: {device}")

    def _get_mps_info(self) -> DeviceInfo:
        """Get MPS device information."""
        device_name = "Apple Silicon MPS"
        properties = {}

        try:
            import platform

            machine = platform.machine()
            if machine == "arm64":
                device_name = f"Apple Silicon {machine}"
                properties["machine"] = machine

                # Try to detect Apple Silicon generation
                processor = platform.processor()
                if "M4" in processor:
                    device_name = "Apple Silicon M4"
                elif "M3" in processor:
                    device_name = "Apple Silicon M3"
                elif "M2" in processor:
                    device_name = "Apple Silicon M2"
                elif "M1" in processor:
                    device_name = "Apple Silicon M1"

                properties["processor"] = processor
        except Exception as e:
            self.logger.debug(f"Could not detect Apple Silicon details: {e}")

        return DeviceInfo(
            device_type="mps",
            device_name=device_name,
            available_memory=None,  # MPS doesn't expose memory info easily
            device_properties=properties,
        )

    def _get_cuda_info(self) -> DeviceInfo:
        """Get CUDA device information."""
        try:
            import torch

            device_count = torch.cuda.device_count()
            if device_count == 0:
                raise RuntimeError("CUDA reported as available but no devices found")

            # Use first CUDA device
            device_id = 0
            properties = torch.cuda.get_device_properties(device_id)

            # Get memory info
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            available_memory_bytes = total_memory

            try:
                # Try to get actual free memory
                torch.cuda.empty_cache()
                free_memory, total_memory = torch.cuda.mem_get_info(device_id)
                available_memory_bytes = free_memory
            except Exception:
                pass

            available_memory_mb = int(available_memory_bytes / (1024 * 1024))

            device_properties = {
                "name": properties.name,
                "major": properties.major,
                "minor": properties.minor,
                "total_memory": int(total_memory / (1024 * 1024)),  # MB
                "multiprocessor_count": properties.multi_processor_count,
            }

            return DeviceInfo(
                device_type="cuda",
                device_name=properties.name,
                available_memory=available_memory_mb,
                device_properties=device_properties,
            )

        except Exception as e:
            self.logger.error(f"Failed to get CUDA device info: {e}")
            return DeviceInfo(
                device_type="cuda", device_name="CUDA Device", available_memory=None, device_properties={}
            )

    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information."""
        import platform

        import psutil

        try:
            # Get CPU information
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            memory_info = psutil.virtual_memory()

            properties = {
                "cores": cpu_count,
                "logical_cores": cpu_count_logical,
                "total_memory": int(memory_info.total / (1024 * 1024)),  # MB
                "architecture": platform.machine(),
                "processor": platform.processor(),
            }

            device_name = f"CPU ({cpu_count} cores, {cpu_count_logical} threads)"
            available_memory_mb = int(memory_info.available / (1024 * 1024))

        except Exception as e:
            self.logger.debug(f"Could not get detailed CPU info: {e}")
            properties = {}
            device_name = "CPU"
            available_memory_mb = None

        return DeviceInfo(
            device_type="cpu",
            device_name=device_name,
            available_memory=available_memory_mb,
            device_properties=properties,
        )


class MockDeviceDetector:
    """
    Mock device detector for testing.
    Allows complete control over device detection for unit tests.
    """

    def __init__(self):
        self.available_devices = {"cpu"}
        self.device_infos = {
            "cpu": DeviceInfo(
                device_type="cpu", device_name="Mock CPU", available_memory=8192, device_properties={"cores": 4}
            )
        }
        self.best_device = "cpu"
        self.call_log = []
        self.should_raise = None

    def set_available_devices(self, devices):
        """Set list of available devices."""
        self.available_devices = set(devices)

    def set_device_info(self, device: str, info: DeviceInfo):
        """Set device info for specific device."""
        self.device_infos[device] = info

    def set_best_device(self, device: str):
        """Set the device that should be returned as best."""
        self.best_device = device

    def set_exception(self, exception: Exception):
        """Set exception to raise on next detection."""
        self.should_raise = exception

    def get_calls(self):
        """Get log of all method calls made."""
        return self.call_log.copy()

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    def detect_best_device(self, preferred_device: str = "auto") -> DeviceInfo:
        """Mock device detection."""
        self.call_log.append({"method": "detect_best_device", "preferred_device": preferred_device})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        if preferred_device != "auto" and preferred_device in self.device_infos:
            return self.device_infos[preferred_device]

        return self.device_infos[self.best_device]

    def is_device_available(self, device: str) -> bool:
        """Mock device availability check."""
        self.call_log.append({"method": "is_device_available", "device": device})

        return device in self.available_devices
