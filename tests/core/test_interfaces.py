"""
Tests for core interfaces and data structures.

Tests ProcessorType enum and ProcessorResult data structure.
Updated for new simplified architecture.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.core.base_processor import ProcessorType, ProcessorResult


class TestProcessorType:
    """Test ProcessorType enum."""

    def test_processor_type_enum(self):
        """Test ProcessorType enum has expected values."""
        assert ProcessorType.DATA_FETCHER.value == "data_fetcher"
        assert ProcessorType.FEATURE_BUILDER.value == "feature_builder"
        assert ProcessorType.FACTOR_AUGMENTER.value == "factor_augmenter"
        assert ProcessorType.BACKTEST_RUNNER.value == "backtest_runner"

    def test_processor_type_string_representation(self):
        """Test ProcessorType string representation."""
        assert str(ProcessorType.DATA_FETCHER) == "ProcessorType.DATA_FETCHER"
        assert ProcessorType.DATA_FETCHER.name == "DATA_FETCHER"
    
    def test_processor_type_equality(self):
        """Test ProcessorType equality comparison."""
        assert ProcessorType.DATA_FETCHER == ProcessorType.DATA_FETCHER
        assert ProcessorType.DATA_FETCHER != ProcessorType.FEATURE_BUILDER
    
    def test_processor_type_in_collection(self):
        """Test ProcessorType can be used in collections."""
        types = [ProcessorType.DATA_FETCHER, ProcessorType.FEATURE_BUILDER]
        assert ProcessorType.DATA_FETCHER in types
        assert ProcessorType.FACTOR_AUGMENTER not in types


class TestProcessorResult:
    """Test ProcessorResult data structure."""

    def test_processor_result_creation(self):
        """Test ProcessorResult creation."""
        result = ProcessorResult(
            success=True,
            data={"test": "data"},
            metadata={"key": "value"}
        )

        assert result.success is True
        assert result.data == {"test": "data"}
        assert result.metadata == {"key": "value"}
        assert result.error is None

    def test_processor_result_with_error(self):
        """Test ProcessorResult with error."""
        error = ValueError("Test error")
        result = ProcessorResult(
            success=False,
            data=None,
            metadata={"error_context": "test"},
            error=error
        )

        assert result.success is False
        assert result.data is None
        assert result.metadata == {"error_context": "test"}
        assert result.error == error

    def test_processor_result_defaults(self):
        """Test ProcessorResult with minimal parameters."""
        result = ProcessorResult(
            success=True,
            data={"result": "success"},
            metadata={}
        )

        assert result.success is True
        assert result.data == {"result": "success"}
        assert result.metadata == {}
        assert result.error is None
    
    def test_processor_result_empty_data(self):
        """Test ProcessorResult with empty data."""
        result = ProcessorResult(
            success=True,
            data={},
            metadata={}
        )
        
        assert result.success is True
        assert result.data == {}
        assert result.metadata == {}
        assert result.error is None
    
    def test_processor_result_none_data(self):
        """Test ProcessorResult with None data."""
        result = ProcessorResult(
            success=False,
            data=None,
            metadata={},
            error=RuntimeError("Processing failed")
        )
        
        assert result.success is False
        assert result.data is None
        assert result.metadata == {}
        assert isinstance(result.error, RuntimeError)
    
    def test_processor_result_complex_data(self):
        """Test ProcessorResult with complex data structures."""
        complex_data = {
            "results": [1, 2, 3],
            "nested": {"key": "value"},
            "timestamp": datetime.now()
        }
        
        result = ProcessorResult(
            success=True,
            data=complex_data,
            metadata={"processing_time": 1.5, "version": "1.0"}
        )
        
        assert result.success is True
        assert result.data == complex_data
        assert result.metadata["processing_time"] == 1.5
        assert result.metadata["version"] == "1.0"
        assert result.error is None


class TestProcessorResultEdgeCases:
    """Test edge cases for ProcessorResult."""
    
    def test_processor_result_large_data(self):
        """Test ProcessorResult with large data."""
        large_data = {"data": list(range(1000))}
        
        result = ProcessorResult(
            success=True,
            data=large_data,
            metadata={}
        )
        
        assert result.success is True
        assert len(result.data["data"]) == 1000
        assert result.data["data"][0] == 0
        assert result.data["data"][-1] == 999
    
    def test_processor_result_nested_error(self):
        """Test ProcessorResult with nested exception."""
        try:
            raise ValueError("Inner error")
        except ValueError as inner:
            try:
                raise RuntimeError("Outer error") from inner
            except RuntimeError as outer:
                result = ProcessorResult(
                    success=False,
                    data=None,
                    metadata={},
                    error=outer
                )
        
        assert result.success is False
        assert result.data is None
        assert isinstance(result.error, RuntimeError)
        assert isinstance(result.error.__cause__, ValueError)
    
    def test_processor_result_metadata_types(self):
        """Test ProcessorResult with various metadata types."""
        metadata = {
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None
        }
        
        result = ProcessorResult(
            success=True,
            data={"test": "data"},
            metadata=metadata
        )
        
        assert result.metadata["string"] == "value"
        assert result.metadata["integer"] == 42
        assert result.metadata["float"] == 3.14
        assert result.metadata["boolean"] is True
        assert result.metadata["list"] == [1, 2, 3]
        assert result.metadata["dict"] == {"nested": "value"}
        assert result.metadata["none"] is None


class TestDataStructureIntegration:
    """Test integration between data structures."""
    
    def test_processor_result_with_processor_type_metadata(self):
        """Test ProcessorResult containing ProcessorType in metadata."""
        result = ProcessorResult(
            success=True,
            data={"processed": "data"},
            metadata={
                "processor_type": ProcessorType.DATA_FETCHER.value,
                "processor_name": "TestProcessor"
            }
        )
        
        assert result.metadata["processor_type"] == "data_fetcher"
        assert result.metadata["processor_name"] == "TestProcessor"
    
    def test_multiple_processor_results(self):
        """Test working with multiple ProcessorResult instances."""
        results = [
            ProcessorResult(success=True, data={"step": 1}, metadata={}),
            ProcessorResult(success=True, data={"step": 2}, metadata={}),
            ProcessorResult(success=False, data=None, metadata={}, error=ValueError("Step 3 failed"))
        ]
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(successful_results) == 2
        assert len(failed_results) == 1
        assert successful_results[0].data["step"] == 1
        assert successful_results[1].data["step"] == 2
        assert isinstance(failed_results[0].error, ValueError) 