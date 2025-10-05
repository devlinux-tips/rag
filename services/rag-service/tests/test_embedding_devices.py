"""
Comprehensive tests for vectordb/embedding_devices.py
Tests device detection, device information, and mock implementations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from src.vectordb.embedding_devices import (
    TorchDeviceDetector,
)
from tests.conftest import (
    MockDeviceDetector,
)
from src.vectordb.embeddings import DeviceInfo


class TestTorchDeviceDetector:
    """Test TorchDeviceDetector class."""

    def setup_method(self):
        """Set up test instance."""
        self.detector = TorchDeviceDetector()

    def test_initialization(self):
        """Test detector initialization."""
        assert hasattr(self.detector, 'logger')

    @patch.object(TorchDeviceDetector, 'is_device_available')
    @patch.object(TorchDeviceDetector, '_get_device_info')
    def test_detect_best_device_preferred_available(self, mock_get_info, mock_is_available):
        """Test detecting best device when preferred device is available."""
        mock_is_available.return_value = True
        expected_info = DeviceInfo("cuda", "Test GPU", 4096, {})
        mock_get_info.return_value = expected_info

        result = self.detector.detect_best_device("cuda")

        mock_is_available.assert_called_once_with("cuda")
        mock_get_info.assert_called_once_with("cuda")
        assert result == expected_info

    @patch.object(TorchDeviceDetector, 'is_device_available')
    @patch.object(TorchDeviceDetector, '_is_mps_available')
    @patch.object(TorchDeviceDetector, '_get_device_info')
    def test_detect_best_device_preferred_unavailable(self, mock_get_info, mock_mps, mock_is_available):
        """Test detecting best device when preferred device is unavailable."""
        mock_is_available.return_value = False
        mock_mps.return_value = True
        expected_info = DeviceInfo("mps", "Apple Silicon", None, {})
        mock_get_info.return_value = expected_info

        result = self.detector.detect_best_device("cuda")

        mock_is_available.assert_called_once_with("cuda")
        mock_mps.assert_called_once()
        mock_get_info.assert_called_once_with("mps")
        assert result == expected_info

    @patch.object(TorchDeviceDetector, '_is_mps_available')
    @patch.object(TorchDeviceDetector, '_get_device_info')
    def test_detect_best_device_auto_mps(self, mock_get_info, mock_mps):
        """Test auto-detection prefers MPS when available."""
        mock_mps.return_value = True
        expected_info = DeviceInfo("mps", "Apple Silicon", None, {})
        mock_get_info.return_value = expected_info

        result = self.detector.detect_best_device("auto")

        mock_mps.assert_called_once()
        mock_get_info.assert_called_once_with("mps")
        assert result == expected_info

    @patch.object(TorchDeviceDetector, '_is_mps_available')
    @patch.object(TorchDeviceDetector, '_is_cuda_available')
    @patch.object(TorchDeviceDetector, '_get_device_info')
    def test_detect_best_device_auto_cuda(self, mock_get_info, mock_cuda, mock_mps):
        """Test auto-detection uses CUDA when MPS unavailable."""
        mock_mps.return_value = False
        mock_cuda.return_value = True
        expected_info = DeviceInfo("cuda", "NVIDIA GPU", 8192, {})
        mock_get_info.return_value = expected_info

        result = self.detector.detect_best_device("auto")

        mock_mps.assert_called_once()
        mock_cuda.assert_called_once()
        mock_get_info.assert_called_once_with("cuda")
        assert result == expected_info

    @patch.object(TorchDeviceDetector, '_is_mps_available')
    @patch.object(TorchDeviceDetector, '_is_cuda_available')
    @patch.object(TorchDeviceDetector, '_get_device_info')
    def test_detect_best_device_auto_cpu_fallback(self, mock_get_info, mock_cuda, mock_mps):
        """Test auto-detection falls back to CPU when GPU unavailable."""
        mock_mps.return_value = False
        mock_cuda.return_value = False
        expected_info = DeviceInfo("cpu", "CPU (4 cores)", 16384, {})
        mock_get_info.return_value = expected_info

        result = self.detector.detect_best_device("auto")

        mock_mps.assert_called_once()
        mock_cuda.assert_called_once()
        mock_get_info.assert_called_once_with("cpu")
        assert result == expected_info

    @patch.object(TorchDeviceDetector, '_is_mps_available')
    def test_is_device_available_mps(self, mock_mps):
        """Test device availability check for MPS."""
        mock_mps.return_value = True
        assert self.detector.is_device_available("mps") is True

        mock_mps.return_value = False
        assert self.detector.is_device_available("mps") is False

    @patch.object(TorchDeviceDetector, '_is_cuda_available')
    def test_is_device_available_cuda(self, mock_cuda):
        """Test device availability check for CUDA."""
        mock_cuda.return_value = True
        assert self.detector.is_device_available("cuda") is True

        mock_cuda.return_value = False
        assert self.detector.is_device_available("cuda") is False

    def test_is_device_available_cpu(self):
        """Test device availability check for CPU."""
        assert self.detector.is_device_available("cpu") is True

    def test_is_device_available_unknown(self):
        """Test device availability check for unknown device."""
        assert self.detector.is_device_available("unknown") is False

    @patch('torch.backends.mps.is_available')
    @patch('torch.backends.mps.is_built')
    def test_is_mps_available_success(self, mock_is_built, mock_is_available):
        """Test MPS availability check when available."""
        mock_is_available.return_value = True
        mock_is_built.return_value = True

        with patch('torch.backends') as mock_backends:
            mock_backends.mps = Mock()
            mock_backends.mps.is_available = mock_is_available
            mock_backends.mps.is_built = mock_is_built

            result = self.detector._is_mps_available()
            assert result is True

    def test_is_mps_available_no_torch(self):
        """Test MPS availability check when torch not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'torch'")):
            result = self.detector._is_mps_available()
            assert result is False

    @patch('torch.backends')
    def test_is_mps_available_no_mps_backend(self, mock_backends):
        """Test MPS availability check when MPS backend not available."""
        # Remove mps attribute to simulate older PyTorch
        del mock_backends.mps

        result = self.detector._is_mps_available()
        assert result is False

    @patch('torch.cuda.is_available')
    def test_is_cuda_available_success(self, mock_cuda_available):
        """Test CUDA availability check when available."""
        mock_cuda_available.return_value = True

        result = self.detector._is_cuda_available()
        assert result is True

    @patch('torch.cuda.is_available')
    def test_is_cuda_available_failure(self, mock_cuda_available):
        """Test CUDA availability check when not available."""
        mock_cuda_available.return_value = False

        result = self.detector._is_cuda_available()
        assert result is False

    def test_is_cuda_available_no_torch(self):
        """Test CUDA availability check when torch not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'torch'")):
            result = self.detector._is_cuda_available()
            assert result is False

    @patch('torch.cuda.is_available')
    def test_is_cuda_available_with_exception(self, mock_cuda_available):
        """Test CUDA availability check with exception."""
        mock_cuda_available.side_effect = RuntimeError("CUDA error")

        result = self.detector._is_cuda_available()
        assert result is False

    @patch.object(TorchDeviceDetector, '_get_mps_info')
    def test_get_device_info_mps(self, mock_get_mps):
        """Test getting device info for MPS."""
        expected_info = DeviceInfo("mps", "Apple Silicon", None, {})
        mock_get_mps.return_value = expected_info

        result = self.detector._get_device_info("mps")
        assert result == expected_info
        mock_get_mps.assert_called_once()

    @patch.object(TorchDeviceDetector, '_get_cuda_info')
    def test_get_device_info_cuda(self, mock_get_cuda):
        """Test getting device info for CUDA."""
        expected_info = DeviceInfo("cuda", "NVIDIA GPU", 8192, {})
        mock_get_cuda.return_value = expected_info

        result = self.detector._get_device_info("cuda")
        assert result == expected_info
        mock_get_cuda.assert_called_once()

    @patch.object(TorchDeviceDetector, '_get_cpu_info')
    def test_get_device_info_cpu(self, mock_get_cpu):
        """Test getting device info for CPU."""
        expected_info = DeviceInfo("cpu", "CPU", 16384, {})
        mock_get_cpu.return_value = expected_info

        result = self.detector._get_device_info("cpu")
        assert result == expected_info
        mock_get_cpu.assert_called_once()

    def test_get_device_info_unknown(self):
        """Test getting device info for unknown device."""
        with pytest.raises(ValueError, match="Unknown device: unknown"):
            self.detector._get_device_info("unknown")

    @patch('platform.machine')
    @patch('platform.processor')
    def test_get_mps_info_apple_silicon_m4(self, mock_processor, mock_machine):
        """Test getting MPS info for Apple Silicon M4."""
        mock_machine.return_value = "arm64"
        mock_processor.return_value = "M4 Pro"

        result = self.detector._get_mps_info()

        assert result.device_type == "mps"
        assert result.device_name == "Apple Silicon M4"
        assert result.available_memory is None
        assert result.device_properties["machine"] == "arm64"
        assert result.device_properties["processor"] == "M4 Pro"

    @patch('platform.machine')
    @patch('platform.processor')
    def test_get_mps_info_apple_silicon_m1(self, mock_processor, mock_machine):
        """Test getting MPS info for Apple Silicon M1."""
        mock_machine.return_value = "arm64"
        mock_processor.return_value = "Apple M1"

        result = self.detector._get_mps_info()

        assert result.device_name == "Apple Silicon M1"

    @patch('platform.machine')
    def test_get_mps_info_non_arm64(self, mock_machine):
        """Test getting MPS info for non-ARM64 architecture."""
        mock_machine.return_value = "x86_64"

        result = self.detector._get_mps_info()

        assert result.device_type == "mps"
        assert result.device_name == "Apple Silicon MPS"

    @patch('platform.machine')
    def test_get_mps_info_with_exception(self, mock_machine):
        """Test getting MPS info when platform detection fails."""
        mock_machine.side_effect = Exception("Platform error")

        result = self.detector._get_mps_info()

        assert result.device_type == "mps"
        assert result.device_name == "Apple Silicon MPS"
        assert result.device_properties == {}

    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.mem_get_info')
    @patch('torch.cuda.empty_cache')
    def test_get_cuda_info_success(self, mock_empty_cache, mock_mem_info, mock_get_props, mock_device_count):
        """Test getting CUDA info successfully."""
        mock_device_count.return_value = 1

        # Mock device properties
        mock_props = Mock()
        mock_props.name = "NVIDIA GeForce RTX 4090"
        mock_props.major = 8
        mock_props.minor = 9
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
        mock_props.multi_processor_count = 128
        mock_get_props.return_value = mock_props

        # Mock memory info
        free_memory = 20 * 1024 * 1024 * 1024  # 20GB free
        total_memory = 24 * 1024 * 1024 * 1024  # 24GB total
        mock_mem_info.return_value = (free_memory, total_memory)

        result = self.detector._get_cuda_info()

        assert result.device_type == "cuda"
        assert result.device_name == "NVIDIA GeForce RTX 4090"
        assert result.available_memory == 20 * 1024  # MB
        assert result.device_properties["name"] == "NVIDIA GeForce RTX 4090"
        assert result.device_properties["major"] == 8
        assert result.device_properties["minor"] == 9
        assert result.device_properties["total_memory"] == 24 * 1024  # MB
        assert result.device_properties["multiprocessor_count"] == 128

    @patch('torch.cuda.device_count')
    def test_get_cuda_info_no_devices(self, mock_device_count):
        """Test getting CUDA info when no devices available."""
        mock_device_count.return_value = 0

        result = self.detector._get_cuda_info()

        # Should return fallback DeviceInfo when no devices found
        assert result.device_type == "cuda"
        assert result.device_name == "CUDA Device"
        assert result.available_memory is None
        assert result.device_properties == {}

    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    def test_get_cuda_info_memory_error(self, mock_get_props, mock_device_count):
        """Test getting CUDA info when memory info fails."""
        mock_device_count.return_value = 1

        mock_props = Mock()
        mock_props.name = "Test GPU"
        mock_props.major = 7
        mock_props.minor = 5
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_props.multi_processor_count = 80
        mock_get_props.return_value = mock_props

        # Mock mem_get_info to raise exception
        with patch('torch.cuda.mem_get_info', side_effect=Exception("Memory error")):
            result = self.detector._get_cuda_info()

        assert result.device_type == "cuda"
        assert result.device_name == "Test GPU"
        assert result.available_memory == 8 * 1024  # Falls back to total memory

    @patch('torch.cuda.device_count')
    def test_get_cuda_info_general_exception(self, mock_device_count):
        """Test getting CUDA info with general exception."""
        mock_device_count.side_effect = Exception("CUDA error")

        result = self.detector._get_cuda_info()

        assert result.device_type == "cuda"
        assert result.device_name == "CUDA Device"
        assert result.available_memory is None
        assert result.device_properties == {}

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('platform.machine')
    @patch('platform.processor')
    def test_get_cpu_info_success(self, mock_processor, mock_machine, mock_memory, mock_cpu_count):
        """Test getting CPU info successfully."""
        mock_cpu_count.side_effect = [8, 16]  # physical, logical cores

        mock_memory_info = Mock()
        mock_memory_info.total = 32 * 1024 * 1024 * 1024  # 32GB
        mock_memory_info.available = 16 * 1024 * 1024 * 1024  # 16GB available
        mock_memory.return_value = mock_memory_info

        mock_machine.return_value = "x86_64"
        mock_processor.return_value = "Intel(R) Core(TM) i9-9900K"

        result = self.detector._get_cpu_info()

        assert result.device_type == "cpu"
        assert result.device_name == "CPU (8 cores, 16 threads)"
        assert result.available_memory == 16 * 1024  # MB
        assert result.device_properties["cores"] == 8
        assert result.device_properties["logical_cores"] == 16
        assert result.device_properties["total_memory"] == 32 * 1024  # MB
        assert result.device_properties["architecture"] == "x86_64"
        assert result.device_properties["processor"] == "Intel(R) Core(TM) i9-9900K"

    @patch('psutil.cpu_count')
    def test_get_cpu_info_with_exception(self, mock_cpu_count):
        """Test getting CPU info when psutil fails."""
        mock_cpu_count.side_effect = Exception("psutil error")

        result = self.detector._get_cpu_info()

        assert result.device_type == "cpu"
        assert result.device_name == "CPU"
        assert result.available_memory is None
        assert result.device_properties == {}


class TestMockDeviceDetector:
    """Test MockDeviceDetector class."""

    def setup_method(self):
        """Set up test instance."""
        self.detector = MockDeviceDetector()

    def test_initialization(self):
        """Test mock detector initialization."""
        assert self.detector.available_devices == {"cpu"}
        assert "cpu" in self.detector.device_infos
        assert self.detector.best_device == "cpu"
        assert self.detector.call_log == []
        assert self.detector.should_raise is None

        # Check default CPU info
        cpu_info = self.detector.device_infos["cpu"]
        assert cpu_info.device_type == "cpu"
        assert cpu_info.device_name == "Mock CPU"
        assert cpu_info.available_memory == 8192
        assert cpu_info.device_properties == {"cores": 4}

    def test_set_available_devices(self):
        """Test setting available devices."""
        devices = ["cpu", "cuda", "mps"]
        self.detector.set_available_devices(devices)

        assert self.detector.available_devices == set(devices)

    def test_set_device_info(self):
        """Test setting device info."""
        cuda_info = DeviceInfo("cuda", "Mock GPU", 16384, {"memory": "16GB"})
        self.detector.set_device_info("cuda", cuda_info)

        assert self.detector.device_infos["cuda"] == cuda_info

    def test_set_best_device(self):
        """Test setting best device."""
        self.detector.set_best_device("cuda")
        assert self.detector.best_device == "cuda"

    def test_set_exception(self):
        """Test setting exception to raise."""
        exception = RuntimeError("Test error")
        self.detector.set_exception(exception)
        assert self.detector.should_raise == exception

    def test_get_calls(self):
        """Test getting call log."""
        self.detector.call_log.append({"method": "test", "param": "value"})

        calls = self.detector.get_calls()
        assert calls == [{"method": "test", "param": "value"}]

        # Verify it returns a copy
        calls.append({"method": "new"})
        assert len(self.detector.call_log) == 1

    def test_clear_calls(self):
        """Test clearing call log."""
        self.detector.call_log.append({"method": "test"})
        assert len(self.detector.call_log) == 1

        self.detector.clear_calls()
        assert self.detector.call_log == []

    def test_detect_best_device_auto(self):
        """Test detecting best device with auto."""
        result = self.detector.detect_best_device("auto")

        assert result == self.detector.device_infos["cpu"]
        assert len(self.detector.call_log) == 1
        call = self.detector.call_log[0]
        assert call["method"] == "detect_best_device"
        assert call["preferred_device"] == "auto"

    def test_detect_best_device_preferred_available(self):
        """Test detecting best device with preferred device available."""
        cuda_info = DeviceInfo("cuda", "Mock GPU", 8192, {})
        self.detector.set_device_info("cuda", cuda_info)

        result = self.detector.detect_best_device("cuda")

        assert result == cuda_info

    def test_detect_best_device_preferred_unavailable(self):
        """Test detecting best device with preferred device unavailable."""
        # Don't add mps to device_infos, so it's unavailable
        result = self.detector.detect_best_device("mps")

        # Should fall back to best_device (cpu)
        assert result == self.detector.device_infos["cpu"]

    def test_detect_best_device_with_exception(self):
        """Test detecting best device raises set exception."""
        exception = ValueError("Mock error")
        self.detector.set_exception(exception)

        with pytest.raises(ValueError, match="Mock error"):
            self.detector.detect_best_device("auto")

        # Exception should be cleared after raising
        assert self.detector.should_raise is None

    def test_is_device_available_true(self):
        """Test device availability check returns True for available devices."""
        self.detector.set_available_devices(["cpu", "cuda"])

        assert self.detector.is_device_available("cpu") is True
        assert self.detector.is_device_available("cuda") is True

        # Check call logging
        assert len(self.detector.call_log) == 2
        assert self.detector.call_log[0]["method"] == "is_device_available"
        assert self.detector.call_log[0]["device"] == "cpu"

    def test_is_device_available_false(self):
        """Test device availability check returns False for unavailable devices."""
        self.detector.set_available_devices(["cpu"])

        assert self.detector.is_device_available("mps") is False

    def test_integration_workflow(self):
        """Test complete mock detector workflow."""
        # Setup multiple devices
        self.detector.set_available_devices(["cpu", "cuda", "mps"])

        cuda_info = DeviceInfo("cuda", "Mock NVIDIA", 24576, {"sm": "8.6"})
        mps_info = DeviceInfo("mps", "Mock Apple Silicon", None, {"generation": "M1"})

        self.detector.set_device_info("cuda", cuda_info)
        self.detector.set_device_info("mps", mps_info)
        self.detector.set_best_device("cuda")

        # Test availability checks
        assert self.detector.is_device_available("cpu") is True
        assert self.detector.is_device_available("cuda") is True
        assert self.detector.is_device_available("mps") is True
        assert self.detector.is_device_available("unknown") is False

        # Test auto detection
        auto_result = self.detector.detect_best_device("auto")
        assert auto_result == cuda_info

        # Test specific device detection
        specific_result = self.detector.detect_best_device("mps")
        assert specific_result == mps_info

        # Verify call logging
        calls = self.detector.get_calls()
        assert len(calls) == 6  # 4 availability checks + 2 detections

    def test_call_logging_comprehensive(self):
        """Test comprehensive call logging."""
        self.detector.set_available_devices(["cpu", "cuda"])
        cuda_info = DeviceInfo("cuda", "Test GPU", 8192, {})
        self.detector.set_device_info("cuda", cuda_info)

        # Make various calls
        self.detector.is_device_available("cpu")
        self.detector.is_device_available("cuda")
        self.detector.detect_best_device("auto")
        self.detector.detect_best_device("cuda")
        self.detector.is_device_available("mps")

        calls = self.detector.get_calls()
        assert len(calls) == 5

        # Verify call details
        assert calls[0] == {"method": "is_device_available", "device": "cpu"}
        assert calls[1] == {"method": "is_device_available", "device": "cuda"}
        assert calls[2] == {"method": "detect_best_device", "preferred_device": "auto"}
        assert calls[3] == {"method": "detect_best_device", "preferred_device": "cuda"}
        assert calls[4] == {"method": "is_device_available", "device": "mps"}

        # Clear and verify
        self.detector.clear_calls()
        assert self.detector.call_log == []


class TestIntegration:
    """Test integration scenarios with device detection."""

    def test_device_detection_priority_flow(self):
        """Test device detection priority flow with mock."""
        detector = MockDeviceDetector()

        # Setup all device types
        detector.set_available_devices(["cpu", "cuda", "mps"])

        cuda_info = DeviceInfo("cuda", "RTX 4090", 24576, {"compute": "8.9"})
        mps_info = DeviceInfo("mps", "M2 Pro", None, {"cores": 12})

        detector.set_device_info("cuda", cuda_info)
        detector.set_device_info("mps", mps_info)

        # Test priority: MPS > CUDA > CPU
        detector.set_best_device("mps")
        result = detector.detect_best_device("auto")
        assert result.device_type == "mps"

        # Test CUDA preference when specifically requested
        result = detector.detect_best_device("cuda")
        assert result.device_type == "cuda"
        assert result.device_name == "RTX 4090"

        # Test CPU fallback
        result = detector.detect_best_device("cpu")
        assert result.device_type == "cpu"

    def test_error_scenarios(self):
        """Test various error scenarios."""
        detector = MockDeviceDetector()

        # Test detection with exception
        error = RuntimeError("Device initialization failed")
        detector.set_exception(error)

        with pytest.raises(RuntimeError, match="Device initialization failed"):
            detector.detect_best_device("auto")

        # Test unavailable preferred device
        detector.set_available_devices(["cpu"])
        result = detector.detect_best_device("cuda")  # Not available

        # Should fall back to best available
        assert result.device_type == "cpu"

    def test_memory_information_handling(self):
        """Test handling of different memory information scenarios."""
        detector = MockDeviceDetector()

        # Device with specific memory
        cuda_info = DeviceInfo("cuda", "GPU with memory", 16384, {"memory_type": "GDDR6"})
        detector.set_device_info("cuda", cuda_info)

        result = detector.detect_best_device("cuda")
        assert result.available_memory == 16384

        # Device without memory info (like MPS)
        mps_info = DeviceInfo("mps", "Apple Silicon", None, {"unified_memory": True})
        detector.set_device_info("mps", mps_info)

        result = detector.detect_best_device("mps")
        assert result.available_memory is None
        assert result.device_properties["unified_memory"] is True

    def test_device_properties_comprehensive(self):
        """Test comprehensive device properties handling."""
        detector = MockDeviceDetector()

        # Complex device properties
        complex_properties = {
            "compute_capability": "8.6",
            "multiprocessor_count": 84,
            "memory_bandwidth": "1008 GB/s",
            "base_clock": "1695 MHz",
            "boost_clock": "1860 MHz",
            "memory_type": "GDDR6X"
        }

        gpu_info = DeviceInfo(
            device_type="cuda",
            device_name="NVIDIA GeForce RTX 3080 Ti",
            available_memory=12288,
            device_properties=complex_properties
        )

        detector.set_device_info("cuda", gpu_info)

        result = detector.detect_best_device("cuda")

        assert result.device_name == "NVIDIA GeForce RTX 3080 Ti"
        assert result.available_memory == 12288
        assert result.device_properties["compute_capability"] == "8.6"
        assert result.device_properties["multiprocessor_count"] == 84
        assert result.device_properties["memory_bandwidth"] == "1008 GB/s"

    @patch.object(TorchDeviceDetector, '_is_mps_available')
    @patch.object(TorchDeviceDetector, '_is_cuda_available')
    @patch.object(TorchDeviceDetector, '_get_device_info')
    def test_real_detector_auto_selection(self, mock_get_info, mock_cuda, mock_mps):
        """Test real detector auto-selection logic."""
        detector = TorchDeviceDetector()

        # Test scenario: MPS available, CUDA not available
        mock_mps.return_value = True
        mock_cuda.return_value = False
        mps_info = DeviceInfo("mps", "Apple M2", None, {})
        mock_get_info.return_value = mps_info

        result = detector.detect_best_device("auto")

        assert result == mps_info
        mock_mps.assert_called_once()
        mock_get_info.assert_called_once_with("mps")

    def test_device_comparison_scenarios(self):
        """Test scenarios for comparing different devices."""
        detector = MockDeviceDetector()

        # Setup multiple devices for comparison
        devices_info = {
            "cpu": DeviceInfo("cpu", "16-core CPU", 32768, {"cores": 16}),
            "cuda": DeviceInfo("cuda", "RTX 4090", 24576, {"memory_bus": "384-bit"}),
            "mps": DeviceInfo("mps", "M2 Ultra", None, {"unified_memory": "128GB"})
        }

        for device, info in devices_info.items():
            detector.set_device_info(device, info)

        detector.set_available_devices(devices_info.keys())

        # Test each device selection
        for device_type in devices_info:
            result = detector.detect_best_device(device_type)
            expected = devices_info[device_type]

            assert result.device_type == expected.device_type
            assert result.device_name == expected.device_name
            assert result.available_memory == expected.available_memory
            assert result.device_properties == expected.device_properties