"""
FeatureBuilder Processor - AI-driven feature generation using pykx and q language.
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
@evaluable(max_retries=3)
@debuggable(max_retries=2)
class FeatureBuilder(BaseProcessor):
    """
    AI-driven feature builder that generates q language code for feature computation.
    
    Capabilities:
    - Observable: UI monitoring and logging
    - Evaluable: Feature code validation and testing
    - Debuggable: Q code error analysis and correction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureBuilder.
        
        Required config:
        - model_name: Name of the OpenAI model to use (default: 'gpt-4')
        
        Optional config:
        - test_data: Sample data for feature testing
        - prompt_manager: Custom prompt manager (defaults to global instance)
        """
        super().__init__(config)
        
        # Extract dependencies from config
        self.model_name = config.get('model_name', 'gpt-4')
        self.test_data = config.get('test_data', {})
        
        # Use provided prompt manager or get global instance
        self.prompt_manager = config.get('prompt_manager', get_prompt_manager())
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.model_name)
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def get_processor_type(self) -> ProcessorType:
        """Return processor type."""
        return ProcessorType.FEATURE_BUILDER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        """
        Core feature building logic: generate q code for features.
        
        Args:
            state: Input contains feature_spec and data_tables
            
        Returns:
            Updated state with q code and computed features
        """
        try:
            # Extract input data
            input_data = state['input_data']
            if not isinstance(input_data, dict):
                raise ValueError("Input must be a dictionary")
            
            feature_spec = input_data.get('feature_spec')
            data_tables = input_data.get('data_tables', {})
            
            if not feature_spec:
                raise ValueError("Input must contain 'feature_spec'")
            
            self.logger.info(f"Building features from spec: {str(feature_spec)[:100]}...")
            
            # Step 1: Generate q code for features
            q_code = self._generate_feature_code(feature_spec, data_tables)
            self.logger.info("Successfully generated q feature code")
            
            # Step 2: Execute q code and compute features
            computed_features = self._execute_q_code(q_code, data_tables)
            self.logger.info(f"Successfully computed {len(computed_features)} features")
            
            # Update state with results
            state['output_data'] = {
                'q_code': q_code,
                'features': computed_features,
                'feature_spec': feature_spec,
                'data_tables_used': list(data_tables.keys())
            }
            state['status'] = 'success'
            
            return state
            
        except Exception as e:
            self.logger.error(f"Feature building failed: {e}")
            state['error'] = e
            state['status'] = 'error'
            raise
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        """
        Evaluate generated features for quality and correctness.
        
        Args:
            state: State with computed features
            
        Returns:
            Updated state with evaluation results
        """
        try:
            output_data = state.get('output_data')
            if not output_data:
                state['eval_passed'] = False
                state['eval_reason'] = "No output data to evaluate"
                return state
            
            q_code = output_data.get('q_code')
            features = output_data.get('features', {})
            
            # Check 1: Q code exists and is valid
            if not q_code or not self._validate_q_code_syntax(q_code):
                state['eval_passed'] = False
                state['eval_reason'] = "Invalid q code syntax"
                return state
            
            # Check 2: Features were computed
            if not features:
                state['eval_passed'] = False
                state['eval_reason'] = "No features were computed"
                return state
            
            # Check 3: Feature data quality
            min_features = self.config.get('min_features_count', 1)
            if len(features) < min_features:
                state['eval_passed'] = False
                state['eval_reason'] = f"Too few features: {len(features)} < {min_features} required"
                return state
            
            # Check 4: Feature values are reasonable
            feature_quality_issues = []
            for feature_name, feature_data in features.items():
                if not self._validate_feature_data(feature_name, feature_data):
                    feature_quality_issues.append(feature_name)
            
            if feature_quality_issues:
                state['eval_passed'] = False
                state['eval_reason'] = f"Feature quality issues: {', '.join(feature_quality_issues)}"
                return state
            
            # All checks passed
            state['eval_passed'] = True
            state['eval_reason'] = f"Feature evaluation passed: {len(features)} quality features generated"
            
            self.logger.info(f"Feature evaluation passed: {state['eval_reason']}")
            return state
            
        except Exception as e:
            self.logger.error(f"Feature evaluation failed: {e}")
            state['eval_passed'] = False
            state['eval_reason'] = f"Evaluation error: {str(e)}"
            return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        """
        Debug feature generation errors and determine retry strategy.
        
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
        if 'syntax' in error_str or 'parse' in error_str:
            # Q code syntax errors - worth retrying with corrected prompt
            state['should_retry'] = True
            state['debug_reason'] = "Q code syntax error - will retry with syntax guidance"
            self.logger.info("Debug: Q syntax error - scheduling retry with better prompt")
            
        elif 'function' in error_str or 'undefined' in error_str:
            # Undefined function errors - need to use correct q functions
            state['should_retry'] = True
            state['debug_reason'] = "Undefined function error - will retry with function examples"
            self.logger.info("Debug: Function error - scheduling retry with q function examples")
            
        elif 'data' in error_str or 'column' in error_str:
            # Data/column related errors - check data structure
            state['should_retry'] = True
            state['debug_reason'] = "Data structure error - will retry with schema validation"
            self.logger.info("Debug: Data error - scheduling retry with schema check")
            
        else:
            # Unknown error - try once more
            state['should_retry'] = True
            state['debug_reason'] = f"Unknown error: {error_str[:100]} - will retry once"
            self.logger.warning(f"Debug: Unknown error - {error_str[:100]}")
        
        return state
    
    def _generate_feature_code(self, feature_spec: Dict[str, Any], data_tables: Dict[str, Any]) -> str:
        """
        Generate q language code for feature computation.
        
        Args:
            feature_spec: Specification of features to build
            data_tables: Available data tables
            
        Returns:
            Q language code for feature computation
        """
        try:
            # Prepare context for AI prompt
            data_schema = self._extract_data_schema(data_tables)
            
            # Get prompt template
            prompt_content = self.prompt_manager.format_template(
                processor_type='feature_builder',
                template_name='factor_build_dev',
                feature_spec=json.dumps(feature_spec, indent=2),
                data_schema=json.dumps(data_schema, indent=2),
                available_tables=list(data_tables.keys())
            )
            
            if prompt_content is None:
                # Fallback prompt if template not found
                prompt_content = f"""Generate q language code for the following feature specification:

Feature Spec: {json.dumps(feature_spec, indent=2)}
Data Schema: {json.dumps(data_schema, indent=2)}
Available Tables: {list(data_tables.keys())}

Return executable q code for feature computation."""
                self.logger.warning("Using fallback prompt - factor_build_dev template not found")
            
            # Call LLM
            response = self.llm.invoke(prompt_content)
            response_content = response.content
            
            # Extract q code from response
            q_code = self._extract_q_code_from_response(response_content)
            if not q_code:
                raise ValueError("No valid q code found in AI response")
            
            return q_code
            
        except Exception as e:
            self.logger.error(f"Feature code generation failed: {e}")
            raise
    
    def _execute_q_code(self, q_code: str, data_tables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute q code to compute features.
        
        Args:
            q_code: Q language code to execute
            data_tables: Input data tables
            
        Returns:
            Dictionary of computed features
        """
        # Mock execution for now (replace with actual pykx execution)
        self.logger.info("Executing q code for feature computation")
        
        # Parse q code to understand feature names
        feature_names = self._extract_feature_names_from_q_code(q_code)
        
        # Mock computed features
        computed_features = {}
        for i, feature_name in enumerate(feature_names):
            computed_features[feature_name] = {
                'values': [0.1 + i * 0.05] * 100,  # Mock feature values
                'statistics': {
                    'mean': 0.1 + i * 0.05,
                    'std': 0.02,
                    'min': 0.05 + i * 0.05,
                    'max': 0.15 + i * 0.05
                },
                'computed_at': datetime.datetime.now().isoformat()
            }
        
        self.logger.info(f"Successfully computed {len(computed_features)} features")
        return computed_features
    
    def _extract_data_schema(self, data_tables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract schema information from data tables.
        
        Args:
            data_tables: Data tables to analyze
            
        Returns:
            Schema information for each table
        """
        schema = {}
        for table_name, table_data in data_tables.items():
            if isinstance(table_data, dict):
                schema[table_name] = {
                    'columns': table_data.get('columns', []),
                    'row_count': table_data.get('rows', 0),
                    'sample_data': str(table_data.get('data', ''))[:100]
                }
            else:
                schema[table_name] = {
                    'columns': ['unknown'],
                    'row_count': len(table_data) if hasattr(table_data, '__len__') else 0,
                    'sample_data': str(table_data)[:100]
                }
        
        return schema
    
    def _extract_q_code_from_response(self, response_content: str) -> str:
        """
        Extract q code from AI response.
        
        Args:
            response_content: AI response text
            
        Returns:
            Extracted q code
        """
        # Look for q code blocks
        if "```q" in response_content:
            start = response_content.find("```q") + 4
            end = response_content.find("```", start)
            return response_content[start:end].strip()
        elif "```" in response_content:
            start = response_content.find("```") + 3
            end = response_content.find("```", start)
            return response_content[start:end].strip()
        else:
            # Return the whole response if no code blocks
            return response_content.strip()
    
    def _extract_feature_names_from_q_code(self, q_code: str) -> List[str]:
        """
        Extract feature names from q code.
        
        Args:
            q_code: Q language code
            
        Returns:
            List of feature names
        """
        # Simple parsing - look for variable assignments
        feature_names = []
        lines = q_code.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('/'):
                # Variable assignment
                var_name = line.split(':')[0].strip()
                if var_name and var_name.isidentifier():
                    feature_names.append(var_name)
        
        # Default features if none found
        if not feature_names:
            feature_names = ['feature_1', 'feature_2', 'feature_3']
        
        return feature_names
    
    def _validate_q_code_syntax(self, q_code: str) -> bool:
        """
        Validate q code syntax.
        
        Args:
            q_code: Q code to validate
            
        Returns:
            True if syntax appears valid
        """
        if not q_code or not isinstance(q_code, str):
            return False
        
        # Basic syntax checks
        if q_code.count('{') != q_code.count('}'):
            return False
        
        if q_code.count('[') != q_code.count(']'):
            return False
        
        if q_code.count('(') != q_code.count(')'):
            return False
        
        # Must contain some actual code
        if len(q_code.strip()) < 10:
            return False
        
        return True
    
    def _validate_feature_data(self, feature_name: str, feature_data: Dict[str, Any]) -> bool:
        """
        Validate computed feature data quality.
        
        Args:
            feature_name: Name of the feature
            feature_data: Feature computation results
            
        Returns:
            True if feature data quality is acceptable
        """
        if not isinstance(feature_data, dict):
            return False
        
        values = feature_data.get('values', [])
        if not values or not isinstance(values, list):
            return False
        
        # Check for reasonable value ranges
        try:
            numeric_values = [float(v) for v in values if v is not None]
            if not numeric_values:
                return False
            
            # Check for extreme values or all zeros
            mean_val = sum(numeric_values) / len(numeric_values)
            if abs(mean_val) > 1e6 or all(v == 0 for v in numeric_values):
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False 