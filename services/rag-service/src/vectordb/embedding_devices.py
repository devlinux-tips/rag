"""
Device detection for embedding computation.
Provides testable device detection abstraction layer.
"""

import logging

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_decision_point,
    log_error_context,
    log_performance_metric,
)
from .embeddings import DeviceInfo


class TorchDeviceDetector:
    """
    Production device detector using PyTorch device detection.
    """

    def __init__(self):
        get_system_logger()
        log_component_start("device_detector", "init", detector_type="TorchDeviceDetector")
        self.logger = logging.getLogger(__name__)
        log_component_end("device_detector", "init", "TorchDeviceDetector initialized")

    def detect_best_device(self, preferred_device: str = "auto") -> DeviceInfo:
        """Detect best available device."""
        logger = get_system_logger()
        log_component_start("device_detector", "detect_best_device", preferred=preferred_device)

        try:
            if preferred_device != "auto":
                logger.debug("device_detector", "detect_best_device", f"Checking preferred device: {preferred_device}")
                if self.is_device_available(preferred_device):
                    log_decision_point(
                        "device_detector", "detect_best_device", f"preferred={preferred_device}", "using_preferred"
                    )
                    device_info = self._get_device_info(preferred_device)
                    log_component_end(
                        "device_detector", "detect_best_device", f"Using preferred device: {preferred_device}"
                    )
                    return device_info
                else:
                    logger.warning(
                        "device_detector",
                        "detect_best_device",
                        f"Preferred device {preferred_device} not available, auto-detecting",
                    )
                    log_decision_point(
                        "device_detector", "detect_best_device", f"preferred={preferred_device}", "fallback_to_auto"
                    )

            logger.trace("device_detector", "detect_best_device", "Starting auto-detection: MPS > CUDA > CPU")

            if self._is_mps_available():
                log_decision_point("device_detector", "detect_best_device", "auto_detection", "selected_mps")
                device_info = self._get_device_info("mps")
                log_component_end(
                    "device_detector", "detect_best_device", f"Auto-detected: MPS - {device_info.device_name}"
                )
                return device_info
            elif self._is_cuda_available():
                log_decision_point("device_detector", "detect_best_device", "auto_detection", "selected_cuda")
                device_info = self._get_device_info("cuda")
                log_component_end(
                    "device_detector", "detect_best_device", f"Auto-detected: CUDA - {device_info.device_name}"
                )
                return device_info
            else:
                log_decision_point("device_detector", "detect_best_device", "auto_detection", "fallback_cpu")
                device_info = self._get_device_info("cpu")
                log_component_end(
                    "device_detector", "detect_best_device", f"Fallback to: CPU - {device_info.device_name}"
                )
                return device_info

        except Exception as e:
            log_error_context("device_detector", "detect_best_device", e, {"preferred_device": preferred_device})
            raise

    def is_device_available(self, device: str) -> bool:
        """Check if device is available."""
        logger = get_system_logger()
        logger.trace("device_detector", "is_device_available", f"Checking availability: {device}")

        if device == "mps":
            available = self._is_mps_available()
            logger.debug("device_detector", "is_device_available", f"MPS available: {available}")
            return available
        elif device == "cuda":
            available = self._is_cuda_available()
            logger.debug("device_detector", "is_device_available", f"CUDA available: {available}")
            return available
        elif device == "cpu":
            logger.trace("device_detector", "is_device_available", "CPU always available: True")
            return True
        else:
            logger.debug("device_detector", "is_device_available", f"Unknown device {device}: False")
            return False

    def _is_mps_available(self) -> bool:
        """Check if MPS (Apple Silicon) is available."""
        logger = get_system_logger()
        logger.trace("device_detector", "_is_mps_available", "Checking MPS availability")

        try:
            import torch

            has_mps_backend = hasattr(torch.backends, "mps")
            is_available = torch.backends.mps.is_available() if has_mps_backend else False
            is_built = torch.backends.mps.is_built() if has_mps_backend else False

            result = has_mps_backend and is_available and is_built
            logger.debug(
                "device_detector",
                "_is_mps_available",
                f"MPS check: backend={has_mps_backend}, available={is_available}, built={is_built}, result={result}",
            )
            return result

        except Exception as e:
            logger.debug("device_detector", "_is_mps_available", f"MPS check failed: {e}")
            return False

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        logger = get_system_logger()
        logger.trace("device_detector", "_is_cuda_available", "Checking CUDA availability")

        try:
            import warnings

            import torch

            logger.trace("device_detector", "_is_cuda_available", "Suppressing CUDA initialization warnings")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
                result = torch.cuda.is_available()

            logger.debug("device_detector", "_is_cuda_available", f"CUDA available: {result}")
            return result

        except Exception as e:
            logger.debug("device_detector", "_is_cuda_available", f"CUDA check failed: {e}")
            return False

    def _get_device_info(self, device: str) -> DeviceInfo:
        """Get detailed information about a device."""
        get_system_logger()
        log_component_start("device_detector", "_get_device_info", device=device)

        try:
            if device == "mps":
                device_info = self._get_mps_info()
            elif device == "cuda":
                device_info = self._get_cuda_info()
            elif device == "cpu":
                device_info = self._get_cpu_info()
            else:
                raise ValueError(f"Unknown device: {device}")

            log_performance_metric(
                "device_detector", "_get_device_info", "available_memory_mb", device_info.available_memory or 0
            )
            log_component_end(
                "device_detector",
                "_get_device_info",
                f"Device info: {device_info.device_name} ({device_info.available_memory or 'unknown'} MB)",
            )
            return device_info

        except Exception as e:
            log_error_context("device_detector", "_get_device_info", e, {"device": device})
            raise

    def _get_mps_info(self) -> DeviceInfo:
        """Get MPS device information."""
        logger = get_system_logger()
        log_component_start("device_detector", "_get_mps_info")

        device_name = "Apple Silicon MPS"
        properties = {}

        try:
            import platform

            machine = platform.machine()
            logger.trace("device_detector", "_get_mps_info", f"Platform machine: {machine}")

            if machine == "arm64":
                device_name = f"Apple Silicon {machine}"
                properties["machine"] = machine
                logger.debug("device_detector", "_get_mps_info", f"Detected ARM64 architecture: {machine}")

                processor = platform.processor()
                logger.trace("device_detector", "_get_mps_info", f"Platform processor: {processor}")

                if "M4" in processor:
                    device_name = "Apple Silicon M4"
                    log_decision_point("device_detector", "_get_mps_info", "chip_detection", "apple_m4")
                elif "M3" in processor:
                    device_name = "Apple Silicon M3"
                    log_decision_point("device_detector", "_get_mps_info", "chip_detection", "apple_m3")
                elif "M2" in processor:
                    device_name = "Apple Silicon M2"
                    log_decision_point("device_detector", "_get_mps_info", "chip_detection", "apple_m2")
                elif "M1" in processor:
                    device_name = "Apple Silicon M1"
                    log_decision_point("device_detector", "_get_mps_info", "chip_detection", "apple_m1")
                else:
                    log_decision_point("device_detector", "_get_mps_info", "chip_detection", "unknown_apple_silicon")

                properties["processor"] = processor
        except Exception as e:
            logger.debug("device_detector", "_get_mps_info", f"Could not detect Apple Silicon details: {e}")

        return DeviceInfo(
            device_type="mps",
            device_name=device_name,
            available_memory=None,  # MPS doesn't expose memory info easily
            device_properties=properties,
        )

    def _get_cuda_info(self) -> DeviceInfo:
        """Get CUDA device information."""
        logger = get_system_logger()
        log_component_start("device_detector", "_get_cuda_info")

        try:
            import torch

            device_count = torch.cuda.device_count()
            logger.debug("device_detector", "_get_cuda_info", f"CUDA device count: {device_count}")

            if device_count == 0:
                raise RuntimeError("CUDA reported as available but no devices found")

            device_id = 0
            properties = torch.cuda.get_device_properties(device_id)
            logger.debug("device_detector", "_get_cuda_info", f"Using CUDA device {device_id}: {properties.name}")

            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            available_memory_bytes = total_memory
            logger.trace("device_detector", "_get_cuda_info", f"Total CUDA memory: {total_memory // (1024 * 1024)} MB")

            try:
                torch.cuda.empty_cache()
                free_memory, total_memory = torch.cuda.mem_get_info(device_id)
                available_memory_bytes = free_memory
                logger.debug(
                    "device_detector", "_get_cuda_info", f"Free CUDA memory: {free_memory // (1024 * 1024)} MB"
                )
            except Exception as e:
                logger.trace("device_detector", "_get_cuda_info", f"Could not get precise memory info: {e}")

            available_memory_mb = int(available_memory_bytes / (1024 * 1024))
            log_performance_metric("device_detector", "_get_cuda_info", "available_memory_mb", available_memory_mb)
            log_performance_metric(
                "device_detector", "_get_cuda_info", "multiprocessor_count", properties.multi_processor_count
            )

            device_properties = {
                "name": properties.name,
                "major": properties.major,
                "minor": properties.minor,
                "total_memory": int(total_memory / (1024 * 1024)),
                "multiprocessor_count": properties.multi_processor_count,
            }

            device_info = DeviceInfo(
                device_type="cuda",
                device_name=properties.name,
                available_memory=available_memory_mb,
                device_properties=device_properties,
            )

            log_component_end(
                "device_detector", "_get_cuda_info", f"CUDA info: {properties.name} ({available_memory_mb} MB)"
            )
            return device_info

        except Exception as e:
            logger = get_system_logger()
            log_error_context("device_detector", "_get_cuda_info", e, {})
            return DeviceInfo(
                device_type="cuda", device_name="CUDA Device", available_memory=None, device_properties={}
            )

    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information."""
        logger = get_system_logger()
        log_component_start("device_detector", "_get_cpu_info")

        import platform

        import psutil

        try:
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            memory_info = psutil.virtual_memory()

            logger.debug(
                "device_detector", "_get_cpu_info", f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical"
            )
            logger.trace(
                "device_detector",
                "_get_cpu_info",
                f"Memory: {memory_info.total // (1024 * 1024)} MB total, {memory_info.available // (1024 * 1024)} MB available",
            )

            architecture = platform.machine()
            processor = platform.processor()
            logger.trace("device_detector", "_get_cpu_info", f"Architecture: {architecture}, Processor: {processor}")

            properties = {
                "cores": cpu_count,
                "logical_cores": cpu_count_logical,
                "total_memory": int(memory_info.total / (1024 * 1024)),
                "architecture": architecture,
                "processor": processor,
            }

            device_name = f"CPU ({cpu_count} cores, {cpu_count_logical} threads)"
            available_memory_mb = int(memory_info.available / (1024 * 1024))

            log_performance_metric(
                "device_detector", "_get_cpu_info", "physical_cores", float(cpu_count) if cpu_count is not None else 0.0
            )
            log_performance_metric(
                "device_detector",
                "_get_cpu_info",
                "logical_cores",
                float(cpu_count_logical) if cpu_count_logical is not None else 0.0,
            )
            log_performance_metric("device_detector", "_get_cpu_info", "available_memory_mb", available_memory_mb)

        except Exception as e:
            logger = get_system_logger()
            logger.debug("device_detector", "_get_cpu_info", f"Could not get detailed CPU info: {e}")
            properties = {}
            device_name = "CPU"
            available_memory_mb = None

        device_info = DeviceInfo(
            device_type="cpu",
            device_name=device_name,
            available_memory=available_memory_mb,
            device_properties=properties,
        )

        log_component_end(
            "device_detector", "_get_cpu_info", f"CPU info: {device_name} ({available_memory_mb or 'unknown'} MB)"
        )
        return device_info
