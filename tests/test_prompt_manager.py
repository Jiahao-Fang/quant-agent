#!/usr/bin/env python3
"""
Test script for the new PromptManager system.
"""

import sys
import os
sys.path.append('src')

from src.prompt_lib import get_prompt_manager, PromptManager

def test_prompt_manager():
    """Test the PromptManager functionality."""
    
    print("=== Testing PromptManager ===")
    
    # Get global prompt manager
    pm = get_prompt_manager()
    
    # Test status
    status = pm.get_status()
    print(f"\nPrompt Manager Status:")
    print(f"- Total templates: {status['total_templates']}")
    print(f"- Processor types: {status['processor_types']}")
    print(f"- Templates per processor: {status['templates_per_processor']}")
    
    # List all templates
    templates = pm.list_templates()
    print(f"\nAvailable Templates:")
    for proc_type, template_names in templates.items():
        print(f"  {proc_type}: {template_names}")
    
    # Test template retrieval
    print(f"\n=== Testing Template Retrieval ===")
    
    # Test data_fetcher templates
    if 'data_fetcher' in templates:
        for template_name in templates['data_fetcher']:
            template = pm.get_template('data_fetcher', template_name)
            if template:
                print(f"✓ Found data_fetcher.{template_name} ({len(template.content)} chars)")
                
                # Test template info
                info = pm.get_template_info('data_fetcher', template_name)
                if info:
                    print(f"  Variables: {info['variables']}")
            else:
                print(f"✗ Failed to get data_fetcher.{template_name}")
    
    # Test feature_builder templates
    if 'feature_builder' in templates:
        for template_name in templates['feature_builder']:
            template = pm.get_template('feature_builder', template_name)
            if template:
                print(f"✓ Found feature_builder.{template_name} ({len(template.content)} chars)")
            else:
                print(f"✗ Failed to get feature_builder.{template_name}")
    
    # Test template formatting
    print(f"\n=== Testing Template Formatting ===")
    
    # Test data_fetcher template formatting
    if 'data_fetcher' in templates and 'data_fetcher_lead' in templates['data_fetcher']:
        formatted = pm.format_template(
            processor_type='data_fetcher',
            template_name='data_fetcher_lead',
            feature_description="Test feature description for BTCUSDT momentum"
        )
        
        if formatted:
            print(f"✓ Successfully formatted data_fetcher_lead template ({len(formatted)} chars)")
            print(f"  Preview: {formatted[:200]}...")
        else:
            print(f"✗ Failed to format data_fetcher_lead template")
    
    # Test validation
    print(f"\n=== Testing Template Validation ===")
    
    if 'data_fetcher' in templates and 'data_fetcher_lead' in templates['data_fetcher']:
        validation = pm.validate_template(
            processor_type='data_fetcher',
            template_name='data_fetcher_lead',
            feature_description="Test description"
        )
        
        print(f"Validation result: {validation}")
    
    # Test error cases
    print(f"\n=== Testing Error Cases ===")
    
    # Non-existent processor
    template = pm.get_template('non_existent', 'template')
    print(f"Non-existent processor: {template is None}")
    
    # Non-existent template
    template = pm.get_template('data_fetcher', 'non_existent_template')
    print(f"Non-existent template: {template is None}")
    
    # Invalid formatting
    formatted = pm.format_template(
        processor_type='data_fetcher',
        template_name='data_fetcher_lead'
        # Missing required variable
    )
    print(f"Invalid formatting: {formatted is None}")

def test_processor_integration():
    """Test processor integration with PromptManager."""
    
    print(f"\n=== Testing Processor Integration ===")
    
    try:
        from processors.data_fetcher import DataFetcher
        
        # Mock AI integration
        class MockAI:
            def generate_response(self, prompt):
                class MockResponse:
                    content = '{"test": "response"}'
                return MockResponse()
        
        # Create DataFetcher with PromptManager
        config = {
            'ai_integration': MockAI()
        }
        
        fetcher = DataFetcher(config)
        print(f"✓ DataFetcher created successfully with PromptManager")
        print(f"  Processor type: {fetcher.get_processor_type()}")
        
    except Exception as e:
        print(f"✗ DataFetcher integration failed: {e}")
    
    try:
        from processors.feature_builder import FeatureBuilder
        
        # Mock AI integration
        class MockAI:
            def generate_response(self, prompt):
                class MockResponse:
                    content = 'feature1: 1+1\nfeature2: 2+2'
                return MockResponse()
        
        config = {
            'ai_integration': MockAI()
        }
        
        builder = FeatureBuilder(config)
        print(f"✓ FeatureBuilder created successfully with PromptManager")
        print(f"  Processor type: {builder.get_processor_type()}")
        
    except Exception as e:
        print(f"✗ FeatureBuilder integration failed: {e}")

if __name__ == '__main__':
    test_prompt_manager()
    test_processor_integration()
    print(f"\n=== Test Complete ===") 