"""
Data build debug stage 1 template for code debugging assistance.
This template transforms factor computation code into a debugging version.
"""

DATA_BUILD_DEBUG_1_TEMPLATE = """You are a code debugging assistant. Your task is to transform the given factor computation code into a debugging version that can capture intermediate results and validate each step.

**Original Code:**
```python
{original_code}
```

**Your Task:**
Transform this code to include comprehensive debugging and validation at each step. Follow these requirements:

1. **Add intermediate variable capture**: For every major computation step, capture the result with detailed information
2. **Type and shape validation**: Check data types, shapes, and basic statistics
3. **Null value detection**: Check for unexpected null values
4. **Data range validation**: Verify reasonable value ranges
5. **Logic validation**: Add assertions for expected behavior

**Debugging Template Structure:**

```python
import pykx as kx
from typing import Dict
import traceback

def compute_factor_debug(data_dict: Dict[str, kx.Table]) -> Dict:
    debug_info = {{
        "steps": [],
        "variables": {{}},
        "errors": [],
        "success": False,
        "final_result": None
    }}
    
    try:
        # Step 1: Data extraction
        debug_info["steps"].append("Step 1: Data extraction")
        # [Original data extraction code here]
        
        # Capture and validate
        debug_info["variables"]["step1_data_keys"] = list(data_dict.keys())
        debug_info["variables"]["step1_table_shapes"] = {{k: v.shape if hasattr(v, 'shape') else len(v) for k, v in extracted_tables.items()}}
        debug_info["variables"]["step1_table_types"] = {{k: type(v).__name__ for k, v in extracted_tables.items()}}
        
        # Continue for each step...
        
        # Final step: Set success and result
        debug_info["success"] = len(debug_info["errors"]) == 0
        # Only store first 5 rows of result to avoid memory issues
        debug_info["final_result"] = result.head(5) if hasattr(result, 'head') else result
        
    except Exception as e:
        debug_info["errors"].append(f"Error: {{str(e)}}")
        debug_info["errors"].append(f"Traceback: {{traceback.format_exc()}}")
    
    return debug_info
```

**For each computation step, add:**

1. **Step description**: Clear description of what this step does
2. **Variable capture**: Store intermediate results with descriptive names
3. **Data validation**: 
   - Check if result is not None/empty
   - Verify data types (kx.Table, numpy array, etc.)
   - Check shapes/lengths match expectations
   - Validate value ranges (no inf, extreme outliers)
   - Count null values
4. **Logic assertions**: Add assertions for expected relationships

**Example transformation:**

Original:
```python
ma5 = trades.apply(lambda x: kx.q.mavg(5, x))['price']
```

Debug version:
```python
# Step X: Calculate 5-period moving average
debug_info["steps"].append("Step X: Calculate 5-period moving average")
ma5 = trades.apply(lambda x: kx.q.mavg(5, x))['price']

# Capture and validate
debug_info["variables"][f"step{{X}}_ma5_type"] = type(ma5).__name__
debug_info["variables"][f"step{{X}}_ma5_length"] = len(ma5) if hasattr(ma5, '__len__') else 'scalar'
debug_info["variables"][f"step{{X}}_ma5_null_count"] = ma5.isna().sum() if hasattr(ma5, 'isna') else 0
debug_info["variables"][f"step{{X}}_ma5_min"] = float(ma5.min()) if hasattr(ma5, 'min') else ma5
debug_info["variables"][f"step{{X}}_ma5_max"] = float(ma5.max()) if hasattr(ma5, 'max') else ma5
debug_info["variables"][f"step{{X}}_ma5_sample"] = ma5.head(3).tolist() if hasattr(ma5, 'head') else str(ma5)[:100]

# Validate expectations
if len(ma5) != len(trades):
    debug_info["errors"].append(f"Step {{X}}: MA5 length {{len(ma5)}} != trades length {{len(trades)}}")
if ma5.isna().sum() > len(trades) * 0.5:  # More than 50% null
    debug_info["errors"].append(f"Step {{X}}: Too many null values in MA5: {{ma5.isna().sum()}}")
```

**Transform the entire original code following this pattern. Ensure every major variable is captured and validated.**

**Important Notes:**
- For the `final_result`, only store the first 5 rows to avoid memory issues: `debug_info["final_result"] = result.head(5) if hasattr(result, 'head') else result`
- Always set `debug_info["success"] = True` at the end if no errors occurred
- Use safe attribute access with `hasattr()` checks to avoid AttributeError

Your response will be used as a parameter for a function, please return only Python code."""
