"""
Checkpoint Management Module
Similar to C++ serialization and RAII patterns
"""

from typing import Dict, Any, Optional, List
import json
import time
from pathlib import Path
from .pipeline_state import PipelineState


class CheckpointData:
    """
    Data class for checkpoint information
    Similar to C++ POD (Plain Old Data) structures
    """
    
    def __init__(self, step_name: str, timestamp: float, state_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self.step_name = step_name
        self.timestamp = timestamp
        self.state_data = state_data
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'step_name': self.step_name,
            'timestamp': self.timestamp,
            'state_data': self.state_data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Deserialize from dictionary"""
        return cls(
            step_name=data['step_name'],
            timestamp=data['timestamp'],
            state_data=data['state_data'],
            metadata=data.get('metadata', {})
        )


class CheckpointManager:
    """
    Manages pipeline checkpoints
    Similar to C++ resource management with RAII
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # In-memory checkpoint storage (like C++ smart pointers)
        self.current_checkpoint: Optional[CheckpointData] = None
        self.checkpoint_history: List[CheckpointData] = []
        
        # Auto-save configuration
        self.auto_save_enabled = True
        self.max_history_size = 10
    
    def create_checkpoint(self, step_name: str, state: PipelineState, metadata: Optional[Dict[str, Any]] = None) -> CheckpointData:
        """
        Create a new checkpoint
        Similar to C++ move semantics for efficient data transfer
        """
        checkpoint = CheckpointData(
            step_name=step_name,
            timestamp=time.time(),
            state_data=state.export_state(),
            metadata=metadata or {}
        )
        
        # Set as current checkpoint
        self.current_checkpoint = checkpoint
        
        # Add to history
        self.checkpoint_history.append(checkpoint)
        
        # Maintain history size limit
        if len(self.checkpoint_history) > self.max_history_size:
            self.checkpoint_history.pop(0)
        
        # Auto-save if enabled
        if self.auto_save_enabled:
            self._save_checkpoint(checkpoint)
        
        return checkpoint
    
    def restore_checkpoint(self, checkpoint: CheckpointData, state: PipelineState) -> bool:
        """
        Restore state from checkpoint
        Similar to C++ copy constructor with validation
        """
        try:
            state.import_state(checkpoint.state_data)
            self.current_checkpoint = checkpoint
            return True
        except Exception as e:
            print(f"Failed to restore checkpoint: {e}")
            return False
    
    def get_current_checkpoint(self) -> Optional[CheckpointData]:
        """Get the current checkpoint"""
        return self.current_checkpoint
    
    def clear_current_checkpoint(self):
        """Clear the current checkpoint"""
        self.current_checkpoint = None
    
    def list_checkpoints(self) -> List[CheckpointData]:
        """List all checkpoints in history"""
        return self.checkpoint_history.copy()
    
    def find_checkpoint_by_step(self, step_name: str) -> Optional[CheckpointData]:
        """Find the latest checkpoint for a specific step"""
        for checkpoint in reversed(self.checkpoint_history):
            if checkpoint.step_name == step_name:
                return checkpoint
        return None
    
    def _save_checkpoint(self, checkpoint: CheckpointData):
        """
        Save checkpoint to disk
        Similar to C++ serialization with error handling
        """
        try:
            filename = f"checkpoint_{checkpoint.timestamp}_{checkpoint.step_name}.json"
            filepath = self.checkpoint_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        Load checkpoint from disk
        Similar to C++ deserialization with validation
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob(f"*{checkpoint_id}*.json"))
            if not checkpoint_files:
                return None
            
            # Load the most recent file if multiple matches
            filepath = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return CheckpointData.from_dict(data)
                
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """
        Clean up old checkpoint files
        Similar to C++ RAII automatic cleanup
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Remove old files beyond keep_count
            for filepath in checkpoint_files[keep_count:]:
                filepath.unlink()
                
        except Exception as e:
            print(f"Failed to cleanup checkpoints: {e}")
    
    def export_checkpoint_summary(self) -> Dict[str, Any]:
        """Export summary of all checkpoints"""
        return {
            'current_checkpoint': self.current_checkpoint.to_dict() if self.current_checkpoint else None,
            'checkpoint_count': len(self.checkpoint_history),
            'latest_checkpoints': [
                {
                    'step_name': cp.step_name,
                    'timestamp': cp.timestamp,
                    'formatted_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cp.timestamp))
                }
                for cp in self.checkpoint_history[-5:]  # Last 5 checkpoints
            ]
        }


class AutoCheckpointer:
    """
    Automatic checkpoint creation at predefined points
    Similar to C++ RAII guard classes
    """
    
    def __init__(self, manager: CheckpointManager, state: PipelineState):
        self.manager = manager
        self.state = state
        self.checkpoint_points = {
            'demand_analysis_complete',
            'data_fetch_complete', 
            'code_generation_complete',
            'code_evaluation_complete',
            'pipeline_complete'
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        if exc_type is None:
            # Success - create completion checkpoint
            self.manager.create_checkpoint(
                step_name="auto_complete",
                state=self.state,
                metadata={'auto_created': True, 'success': True}
            )
        else:
            # Exception - create error checkpoint
            self.manager.create_checkpoint(
                step_name="auto_error",
                state=self.state,
                metadata={'auto_created': True, 'success': False, 'error': str(exc_val)}
            )
    
    def checkpoint_if_needed(self, step_name: str, force: bool = False):
        """Create checkpoint if this is a designated checkpoint point"""
        if force or step_name in self.checkpoint_points:
            self.manager.create_checkpoint(
                step_name=step_name,
                state=self.state,
                metadata={'auto_created': True}
            ) 