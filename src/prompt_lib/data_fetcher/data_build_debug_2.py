"""
Data build debug stage 2 template for code debugging and fixing.
This template analyzes debugging output and fixes code issues.
"""

DATA_BUILD_DEBUG_2_TEMPLATE = """You are an expert code debugger specializing in quantitative finance and PyKx. You need to analyze debugging output and fix code issues.

**Original Code:**
```python
{original_code}
```

**Debug Execution Results:**
```
Error Messages: {error_messages}
Debug Variables: {debug_variables}
Steps Completed: {completed_steps}
```

**Debugging Context:**
- Factor Description: {factor_description}
- Data Description: {data_description}
- Current Retry: {current_retry} / {max_retries}

**Your Analysis Task:**

1. **Error Classification**: Determine if this is:
   - Syntax/Runtime Error (code bug)
   - Data Structure Error (wrong data access/manipulation)
   - Logic Error (incorrect financial calculation)
   - Data Quality Issue (missing/corrupted data)

2. **Root Cause Analysis**: Based on the debug variables and error messages:
   - Which step failed or produced unexpected results?
   - What was the expected vs actual output?
   - Are the data types and shapes correct?
   - Are there missing data issues?

3. **Fix Strategy**: 
   - For Code Bugs: Fix syntax, method calls, variable references
   - For Data Issues: Add proper data validation and handling
   - For Logic Errors: Correct the financial calculation logic
   - For Look-ahead Bias: Ensure only historical data is used

**Common PyKX Issues to Check:**
- Incorrect use of kx.q functions vs pandas methods
- Wrong data access patterns for data_dict
- Missing null value handling
- Incorrect time window operations
- Wrong axis specifications in apply functions

**Fix Guidelines:**
1. Keep the same overall structure and logic intent
2. Add proper error handling for data edge cases
3. Ensure no look-ahead bias in calculations
4. Use efficient PyKx operations
5. Maintain the original function signature

**Output Format:**
Return the complete fixed Python code that addresses the identified issues. Include:
- Fixed computation logic
- Proper error handling
- Clear variable names
- Comments explaining the fixes

Your response will be used as a parameter for a function, please return only Python code."""