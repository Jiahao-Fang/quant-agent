"""
Prompt Manager for handling AI prompt templates across all processors.

Manages prompt templates organized by processor type and provides
a unified interface for template retrieval and formatting.
"""

import os
import importlib.util
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    name: str
    content: str
    processor_type: str
    description: Optional[str] = None
    variables: Optional[List[str]] = None


class PromptManager:
    """
    Manages prompt templates for all processors.
    
    Organizes prompts by processor type and provides methods for:
    - Loading templates from files
    - Formatting templates with variables
    - Template discovery and validation
    """
    
    def __init__(self, prompt_lib_path: Optional[str] = None):
        """
        Initialize PromptManager.
        
        Args:
            prompt_lib_path: Path to prompt library directory. 
                           Defaults to src/prompt_lib relative to this file.
        """
        if prompt_lib_path is None:
            # Default to src/prompt_lib relative to this file
            current_dir = Path(__file__).parent
            prompt_lib_path = current_dir
        
        self.prompt_lib_path = Path(prompt_lib_path)
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Load all templates on initialization
        self._load_all_templates()
    
    def _load_all_templates(self) -> None:
        """Load all prompt templates from the prompt library."""
        processor_dirs = [
            'data_fetcher',
            'feature_builder', 
            'factor_augmenter',
            'backtest_runner',
            'common'
        ]
        
        for processor_type in processor_dirs:
            processor_path = self.prompt_lib_path / processor_type
            if processor_path.exists():
                self._load_processor_templates(processor_type, processor_path)
            else:
                self.logger.warning(f"Processor prompt directory not found: {processor_path}")
    
    def _load_processor_templates(self, processor_type: str, processor_path: Path) -> None:
        """Load templates for a specific processor type."""
        if processor_type not in self.templates:
            self.templates[processor_type] = {}
        
        # Load Python files containing template constants
        for py_file in processor_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                self._load_templates_from_file(processor_type, py_file)
            except Exception as e:
                self.logger.error(f"Failed to load templates from {py_file}: {e}")
    
    def _load_templates_from_file(self, processor_type: str, file_path: Path) -> None:
        """Load template constants from a Python file."""
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            self.logger.warning(f"Could not load spec for {file_path}")
            return
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for template constants (variables ending with _TEMPLATE)
        for attr_name in dir(module):
            if attr_name.endswith('_TEMPLATE') and not attr_name.startswith('_'):
                template_content = getattr(module, attr_name)
                if isinstance(template_content, str):
                    template_name = attr_name.lower().replace('_template', '')
                    
                    # Extract variables from template (simple {variable} detection)
                    variables = self._extract_template_variables(template_content)
                    
                    template = PromptTemplate(
                        name=template_name,
                        content=template_content,
                        processor_type=processor_type,
                        description=f"Template from {file_path.name}",
                        variables=variables
                    )
                    
                    self.templates[processor_type][template_name] = template
                    self.logger.debug(f"Loaded template: {processor_type}.{template_name}")
    
    

    def _extract_template_variables(self, template_content: str) -> List[str]:
        import string
        formatter = string.Formatter()
        variables = set()
        for literal_text, field_name, format_spec, conversion in formatter.parse(template_content):
            if field_name is not None:
                variables.add(field_name)
        return list(variables)

    def get_template(self, processor_type: str, template_name: str) -> Optional[PromptTemplate]:
        """
        Get a specific template.
        
        Args:
            processor_type: Type of processor (data_fetcher, feature_builder, etc.)
            template_name: Name of the template
            
        Returns:
            PromptTemplate if found, None otherwise
        """
        processor_templates = self.templates.get(processor_type, {})
        return processor_templates.get(template_name)
    
    def format_template(
        self, 
        processor_type: str, 
        template_name: str, 
        **kwargs
    ) -> Optional[str]:
        """
        Format a template with provided variables.
        
        Args:
            processor_type: Type of processor
            template_name: Name of the template
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted template string, or None if template not found
        """
        template = self.get_template(processor_type, template_name)
        if template is None:
            self.logger.error(f"Template not found: {processor_type}.{template_name}")
            return None
        
        try:
            return template.content.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing template variable {e} for {processor_type}.{template_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error formatting template {processor_type}.{template_name}: {e}")
            return None
    
    def list_templates(self, processor_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available templates.
        
        Args:
            processor_type: If specified, only list templates for this processor
            
        Returns:
            Dictionary mapping processor types to lists of template names
        """
        if processor_type:
            if processor_type in self.templates:
                return {processor_type: list(self.templates[processor_type].keys())}
            else:
                return {}
        
        return {
            proc_type: list(templates.keys()) 
            for proc_type, templates in self.templates.items()
        }
    
    def get_template_info(self, processor_type: str, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a template.
        
        Args:
            processor_type: Type of processor
            template_name: Name of the template
            
        Returns:
            Dictionary with template information
        """
        template = self.get_template(processor_type, template_name)
        if template is None:
            return None
        
        return {
            'name': template.name,
            'processor_type': template.processor_type,
            'description': template.description,
            'variables': template.variables,
            'content_length': len(template.content),
            'content_preview': template.content[:200] + "..." if len(template.content) > 200 else template.content
        }
    
    def validate_template(self, processor_type: str, template_name: str, **kwargs) -> Dict[str, Any]:
        """
        Validate that all required variables are provided for a template.
        
        Args:
            processor_type: Type of processor
            template_name: Name of the template
            **kwargs: Variables to check
            
        Returns:
            Dictionary with validation results
        """
        template = self.get_template(processor_type, template_name)
        if template is None:
            return {
                'valid': False,
                'error': f"Template not found: {processor_type}.{template_name}"
            }
        
        if template.variables is None:
            return {'valid': True, 'missing_variables': [], 'extra_variables': []}
        
        required_vars = set(template.variables)
        provided_vars = set(kwargs.keys())
        
        missing_vars = required_vars - provided_vars
        extra_vars = provided_vars - required_vars
        
        return {
            'valid': len(missing_vars) == 0,
            'missing_variables': list(missing_vars),
            'extra_variables': list(extra_vars),
            'required_variables': list(required_vars)
        }
    
    def reload_templates(self) -> None:
        """Reload all templates from disk."""
        self.templates.clear()
        self._load_all_templates()
        self.logger.info("Reloaded all prompt templates")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about loaded templates."""
        total_templates = sum(len(templates) for templates in self.templates.values())
        
        processor_counts = {
            proc_type: len(templates) 
            for proc_type, templates in self.templates.items()
        }
        
        return {
            'total_templates': total_templates,
            'processor_types': list(self.templates.keys()),
            'templates_per_processor': processor_counts,
            'prompt_lib_path': str(self.prompt_lib_path)
        }


# Global instance for easy access
_global_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global PromptManager instance."""
    global _global_prompt_manager
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager()
    return _global_prompt_manager


def set_prompt_manager(manager: PromptManager) -> None:
    """Set the global PromptManager instance."""
    global _global_prompt_manager
    _global_prompt_manager = manager 