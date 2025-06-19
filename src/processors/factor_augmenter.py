"""
FactorAugmenter Processor - AI-powered factor enhancement and optimization.
"""

from typing import Dict, Any, List
import logging
import json
import datetime

from ..core.base_processor import BaseProcessor, ProcessorType, ProcessorState
from ..core.decorators import evaluable, debuggable, observable
from ..prompt_lib import get_prompt_manager
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@observable(observers=["ui", "logger"])
@evaluable(max_retries=2)
@debuggable(max_retries=1)
class FactorAugmenter(BaseProcessor):
    """
    AI-driven factor augmenter that enhances features with transformations and combinations.
    
    Capabilities:
    - Observable: UI monitoring and logging
    - Evaluable: Factor quality assessment and validation
    - Debuggable: Factor computation error analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FactorAugmenter.
        
        Required config:
        - model_name: Name of the OpenAI model to use (default: 'gpt-4')
        
        Optional config:
        - enhancement_methods: List of enhancement techniques to use
        """
        super().__init__(config)
        
        # Extract dependencies from config
        self.model_name = config.get('model_name', 'gpt-4')
        self.prompt_manager = config.get('prompt_manager', get_prompt_manager())
        self.enhancement_methods = config.get('enhancement_methods', ['transform', 'combine', 'normalize'])
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.model_name)
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def get_processor_type(self) -> ProcessorType:
        """Return processor type."""
        return ProcessorType.FACTOR_AUGMENTER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        """
        Core factor augmentation logic: enhance and combine features.
        
        Args:
            state: Input contains raw_features and enhancement_spec
            
        Returns:
            Updated state with enhanced factors
        """
        try:
            # Extract input data
            input_data = state['input_data']
            if not isinstance(input_data, dict):
                raise ValueError("Input must be a dictionary")
            
            raw_features = input_data.get('raw_features', {})
            enhancement_spec = input_data.get('enhancement_spec', {})
            
            if not raw_features:
                raise ValueError("Input must contain 'raw_features'")
            
            self.logger.info(f"Augmenting {len(raw_features)} raw features...")
            
            # Step 1: Generate enhancement strategies
            enhancement_strategies = self._generate_enhancement_strategies(raw_features, enhancement_spec)
            self.logger.info(f"Generated {len(enhancement_strategies)} enhancement strategies")
            
            # Step 2: Apply enhancements
            enhanced_factors = self._apply_enhancements(raw_features, enhancement_strategies)
            self.logger.info(f"Successfully created {len(enhanced_factors)} enhanced factors")
            
            # Update state with results
            state['output_data'] = {
                'enhanced_factors': enhanced_factors,
                'enhancement_strategies': enhancement_strategies,
                'raw_features_count': len(raw_features),
                'enhanced_factors_count': len(enhanced_factors)
            }
            state['status'] = 'success'
            
            return state
            
        except Exception as e:
            self.logger.error(f"Factor augmentation failed: {e}")
            state['error'] = e
            state['status'] = 'error'
            raise
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        """
        Evaluate enhanced factors for quality and improvement.
        
        Args:
            state: State with enhanced factors
            
        Returns:
            Updated state with evaluation results
        """
        try:
            output_data = state.get('output_data')
            if not output_data:
                state['eval_passed'] = False
                state['eval_reason'] = "No output data to evaluate"
                return state
            
            enhanced_factors = output_data.get('enhanced_factors', {})
            raw_features_count = output_data.get('raw_features_count', 0)
            
            # Check 1: Enhanced factors exist
            if not enhanced_factors:
                state['eval_passed'] = False
                state['eval_reason'] = "No enhanced factors generated"
                return state
            
            # Check 2: Improvement in factor count
            min_improvement_ratio = self.config.get('min_improvement_ratio', 1.5)
            improvement_ratio = len(enhanced_factors) / max(raw_features_count, 1)
            
            if improvement_ratio < min_improvement_ratio:
                state['eval_passed'] = False
                state['eval_reason'] = f"Insufficient enhancement: {improvement_ratio:.2f}x < {min_improvement_ratio}x required"
                return state
            
            # Check 3: Factor quality metrics
            quality_issues = []
            for factor_name, factor_data in enhanced_factors.items():
                if not self._validate_factor_quality(factor_name, factor_data):
                    quality_issues.append(factor_name)
            
            max_quality_issues = self.config.get('max_quality_issues', 0)
            if len(quality_issues) > max_quality_issues:
                state['eval_passed'] = False
                state['eval_reason'] = f"Too many quality issues: {len(quality_issues)} factors failed validation"
                return state
            
            # Check 4: Diversity of enhancement methods
            strategies = output_data.get('enhancement_strategies', [])
            unique_methods = set(strategy.get('method', 'unknown') for strategy in strategies)
            min_methods = self.config.get('min_enhancement_methods', 2)
            
            if len(unique_methods) < min_methods:
                state['eval_passed'] = False
                state['eval_reason'] = f"Insufficient enhancement diversity: {len(unique_methods)} < {min_methods} methods"
                return state
            
            # All checks passed
            state['eval_passed'] = True
            state['eval_reason'] = (
                f"Factor enhancement passed: {len(enhanced_factors)} factors "
                f"({improvement_ratio:.2f}x improvement) using {len(unique_methods)} methods"
            )
            
            self.logger.info(f"Factor evaluation passed: {state['eval_reason']}")
            return state
            
        except Exception as e:
            self.logger.error(f"Factor evaluation failed: {e}")
            state['eval_passed'] = False
            state['eval_reason'] = f"Evaluation error: {str(e)}"
            return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        """
        Debug factor augmentation errors and determine retry strategy.
        
        Args:
            state: State with error information
            
        Returns:
            Updated state with debug analysis and retry decision
        """
        error = state.get('error')
        if not error:
            state['should_retry'] = False
            return state
        
        error_str = str(error).lower()
        
        # Analyze error type and determine retry strategy
        if 'computation' in error_str or 'calculation' in error_str:
            # Factor computation errors - worth retrying with simpler methods
            state['should_retry'] = True
            state['debug_reason'] = "Factor computation error - will retry with simpler enhancement methods"
            self.logger.info("Debug: Computation error - scheduling retry with simpler methods")
            
        elif 'dimension' in error_str or 'shape' in error_str:
            # Dimensional mismatch errors - need to align feature dimensions
            state['should_retry'] = True
            state['debug_reason'] = "Dimension error - will retry with dimension alignment"
            self.logger.info("Debug: Dimension error - scheduling retry with alignment")
            
        elif 'correlation' in error_str or 'singular' in error_str:
            # Mathematical issues - may need different enhancement approach
            state['should_retry'] = True
            state['debug_reason'] = "Mathematical error - will retry with alternative methods"
            self.logger.info("Debug: Mathematical error - scheduling retry with alternatives")
            
        elif 'memory' in error_str or 'resource' in error_str:
            # Resource errors - unlikely to resolve with retry
            state['should_retry'] = False
            state['debug_reason'] = "Resource limitation - manual optimization required"
            self.logger.warning("Debug: Resource error - manual intervention needed")
            
        elif 'data' in error_str or 'missing' in error_str:
            # Data issues - may resolve with data validation
            state['should_retry'] = False
            state['debug_reason'] = "Data quality issue - check input features"
            self.logger.warning("Debug: Data issue - check input quality")
            
        else:
            # Unknown error - try once more
            state['should_retry'] = True
            state['debug_reason'] = f"Unknown error: {error_str[:100]} - will retry with conservative approach"
            self.logger.warning(f"Debug: Unknown error - {error_str[:100]}")
        
        return state
    
    def _generate_enhancement_strategies(self, raw_features: Dict[str, Any], enhancement_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate enhancement strategies using AI.
        
        Args:
            raw_features: Original features to enhance
            enhancement_spec: Enhancement requirements and preferences
            
        Returns:
            List of enhancement strategies
        """
        # Prepare feature summary for AI
        feature_summary = self._summarize_features(raw_features)
        
        # Get prompt template
        prompt_content = self.prompt_manager.format_template(
            processor_type='factor_augmenter',
            template_name='factor_enhancement_strategy',
            raw_features=json.dumps(feature_summary, indent=2),
            enhancement_spec=json.dumps(enhancement_spec, indent=2),
            available_methods=self.enhancement_methods
        )
        
        if prompt_content is None:
            # Fallback prompt if template not found
            prompt_content = f"""Generate enhancement strategies for the following features:

Raw Features: {json.dumps(feature_summary, indent=2)}
Enhancement Spec: {json.dumps(enhancement_spec, indent=2)}
Available Methods: {self.enhancement_methods}

Return a list of enhancement strategies in JSON format."""
            self.logger.warning("Using fallback prompt - factor_enhancement_strategy template not found")
        
        # Call LLM
        response = self.llm.invoke(prompt_content)
        response_content = response.content
        
        # Parse enhancement strategies from response
        strategies = self._parse_enhancement_strategies(response_content)
        if not strategies:
            raise ValueError("No valid enhancement strategies found in AI response")
        
        return strategies
    
    def _apply_enhancements(self, raw_features: Dict[str, Any], strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply enhancement strategies to create new factors.
        
        Args:
            raw_features: Original features
            strategies: Enhancement strategies to apply
            
        Returns:
            Dictionary of enhanced factors
        """
        enhanced_factors = {}
        
        # Keep original features
        for feature_name, feature_data in raw_features.items():
            enhanced_factors[f"original_{feature_name}"] = feature_data
        
        # Apply each enhancement strategy
        for i, strategy in enumerate(strategies):
            try:
                method = strategy.get('method', 'transform')
                params = strategy.get('parameters', {})
                target_features = strategy.get('target_features', [])
                
                # Apply enhancement based on method
                if method == 'transform':
                    new_factors = self._apply_transformation(raw_features, target_features, params)
                elif method == 'combine':
                    new_factors = self._apply_combination(raw_features, target_features, params)
                elif method == 'normalize':
                    new_factors = self._apply_normalization(raw_features, target_features, params)
                else:
                    self.logger.warning(f"Unknown enhancement method: {method}")
                    continue
                
                # Add new factors with strategy prefix
                for factor_name, factor_data in new_factors.items():
                    enhanced_name = f"{method}_{i}_{factor_name}"
                    enhanced_factors[enhanced_name] = factor_data
                    
            except Exception as e:
                self.logger.warning(f"Failed to apply strategy {i}: {e}")
                continue
        
        return enhanced_factors
    
    def _apply_transformation(self, features: Dict[str, Any], target_features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation enhancement method."""
        transform_type = params.get('type', 'log')
        result = {}
        
        for feature_name in target_features:
            if feature_name in features:
                feature_data = features[feature_name]
                values = feature_data.get('values', [])
                
                if transform_type == 'log':
                    # Log transformation (mock)
                    transformed_values = [abs(v) + 0.001 for v in values]  # Mock log transform
                elif transform_type == 'square':
                    # Square transformation
                    transformed_values = [v * v for v in values]
                elif transform_type == 'sqrt':
                    # Square root transformation
                    transformed_values = [abs(v) ** 0.5 for v in values]
                else:
                    # Default identity transform
                    transformed_values = values
                
                result[f"{transform_type}_{feature_name}"] = {
                    'values': transformed_values,
                    'method': 'transformation',
                    'transform_type': transform_type,
                    'source_feature': feature_name,
                    'created_at': datetime.datetime.now().isoformat()
                }
        
        return result
    
    def _apply_combination(self, features: Dict[str, Any], target_features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply combination enhancement method."""
        combination_type = params.get('type', 'add')
        result = {}
        
        # Simple pairwise combinations
        for i, feature1 in enumerate(target_features):
            for feature2 in target_features[i+1:]:
                if feature1 in features and feature2 in features:
                    values1 = features[feature1].get('values', [])
                    values2 = features[feature2].get('values', [])
                    
                    # Ensure same length
                    min_len = min(len(values1), len(values2))
                    values1 = values1[:min_len]
                    values2 = values2[:min_len]
                    
                    if combination_type == 'add':
                        combined_values = [v1 + v2 for v1, v2 in zip(values1, values2)]
                    elif combination_type == 'multiply':
                        combined_values = [v1 * v2 for v1, v2 in zip(values1, values2)]
                    elif combination_type == 'ratio':
                        combined_values = [v1 / (v2 + 1e-8) for v1, v2 in zip(values1, values2)]
                    else:
                        # Default to addition
                        combined_values = [v1 + v2 for v1, v2 in zip(values1, values2)]
                    
                    result[f"{combination_type}_{feature1}_{feature2}"] = {
                        'values': combined_values,
                        'method': 'combination',
                        'combination_type': combination_type,
                        'source_features': [feature1, feature2],
                        'created_at': datetime.datetime.now().isoformat()
                    }
        
        return result
    
    def _apply_normalization(self, features: Dict[str, Any], target_features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization enhancement method."""
        normalization_type = params.get('type', 'zscore')
        result = {}
        
        for feature_name in target_features:
            if feature_name in features:
                feature_data = features[feature_name]
                values = feature_data.get('values', [])
                
                if normalization_type == 'zscore':
                    # Z-score normalization
                    mean_val = sum(values) / len(values) if values else 0
                    variance = sum((v - mean_val) ** 2 for v in values) / len(values) if values else 1
                    std_val = variance ** 0.5
                    normalized_values = [(v - mean_val) / (std_val + 1e-8) for v in values]
                    
                elif normalization_type == 'minmax':
                    # Min-max normalization
                    min_val = min(values) if values else 0
                    max_val = max(values) if values else 1
                    range_val = max_val - min_val
                    normalized_values = [(v - min_val) / (range_val + 1e-8) for v in values]
                    
                else:
                    # Default to z-score
                    normalized_values = values
                
                result[f"{normalization_type}_norm_{feature_name}"] = {
                    'values': normalized_values,
                    'method': 'normalization',
                    'normalization_type': normalization_type,
                    'source_feature': feature_name,
                    'created_at': datetime.datetime.now().isoformat()
                }
        
        return result
    
    def _summarize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize features for AI processing."""
        summary = {
            'feature_count': len(features),
            'feature_names': list(features.keys()),
            'feature_details': {}
        }
        
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, dict):
                values = feature_data.get('values', [])
                if values:
                    summary['feature_details'][feature_name] = {
                        'length': len(values),
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
        
        return summary
    
    def _parse_enhancement_strategies(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse enhancement strategies from AI response."""
        try:
            # Try to extract JSON from response
            if "```json" in response_content:
                start = response_content.find("```json") + 7
                end = response_content.find("```", start)
                json_content = response_content[start:end].strip()
            else:
                json_content = response_content.strip()
            
            strategies = json.loads(json_content)
            if isinstance(strategies, list):
                return strategies
            
        except (json.JSONDecodeError, KeyError):
            self.logger.warning("Failed to parse AI response as JSON")
        
        # Fallback: create default strategies
        return [
            {
                'method': 'transform',
                'parameters': {'type': 'log'},
                'target_features': ['all']
            },
            {
                'method': 'combine',
                'parameters': {'type': 'add'},
                'target_features': ['all']
            },
            {
                'method': 'normalize',
                'parameters': {'type': 'zscore'},
                'target_features': ['all']
            }
        ]
    
    def _validate_factor_quality(self, factor_name: str, factor_data: Dict[str, Any]) -> bool:
        """Validate enhanced factor quality."""
        if not isinstance(factor_data, dict):
            return False
        
        values = factor_data.get('values', [])
        if not values or not isinstance(values, list):
            return False
        
        try:
            # Check for reasonable numeric values
            numeric_values = [float(v) for v in values if v is not None]
            if not numeric_values:
                return False
            
            # Check for extreme values, NaN, or all identical values
            if any(abs(v) > 1e6 for v in numeric_values):
                return False
            
            if len(set(numeric_values)) == 1:  # All values identical
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False 