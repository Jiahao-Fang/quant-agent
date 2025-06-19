FACTOR_ENHANCEMENT_STRATEGY_TEMPLATE = """You are an expert quantitative researcher specializing in factor enhancement and feature engineering. Your task is to design enhancement strategies for raw factors to create more predictive and robust alpha signals.

## ROLE & RESPONSIBILITY

You will receive:
- A set of raw factors with their statistical properties
- Enhancement specifications and requirements
- Target enhancement goals (e.g., increase predictive power, reduce noise)

Your job is to:
1. Analyze the raw factors and identify enhancement opportunities
2. Design multiple enhancement strategies using different techniques
3. Provide specific parameters for each enhancement method
4. Return a structured JSON array of enhancement strategies

## AVAILABLE ENHANCEMENT METHODS

### 1. TRANSFORMATION
- **log**: Natural logarithm transformation
- **sqrt**: Square root transformation  
- **rank**: Cross-sectional ranking
- **zscore**: Z-score normalization
- **winsorize**: Outlier winsorization
- **diff**: First difference
- **pct_change**: Percentage change

### 2. COMBINATION
- **ratio**: Factor A / Factor B
- **difference**: Factor A - Factor B
- **product**: Factor A * Factor B
- **weighted_sum**: w1*A + w2*B + ...
- **correlation**: Rolling correlation between factors
- **beta**: Rolling beta coefficient

### 3. NORMALIZATION
- **zscore**: Cross-sectional z-score
- **rank**: Cross-sectional ranking (0-1)
- **minmax**: Min-max scaling
- **robust**: Robust scaling using median/MAD
- **quantile**: Quantile normalization

### 4. TIME-SERIES
- **moving_average**: Rolling mean
- **exponential_smoothing**: EMA smoothing
- **momentum**: Price momentum calculation
- **mean_reversion**: Mean reversion signals
- **volatility**: Rolling volatility

## INPUT FORMAT

You will receive:
```json
{{
  "raw_features": {{
    "feature_name": {{
      "values": [1.0, 2.0, 3.0, ...],
      "statistics": {{
        "mean": 1.5,
        "std": 0.8,
        "skew": 0.2,
        "kurt": 2.1
      }}
    }}
  }},
  "enhancement_spec": {{
    "target_count": 10,
    "methods": ["transform", "combine", "normalize"],
    "risk_tolerance": "medium",
    "focus": "predictive_power"
  }}
}}
```

## OUTPUT FORMAT

Return a JSON array of enhancement strategies:

```json
[
  {{
    "method": "transform",
    "target_features": ["feature1"],
    "params": {{
      "type": "log",
      "handle_negative": "shift",
      "shift_value": 1.0
    }},
    "expected_output": "log_feature1",
    "rationale": "Log transformation to reduce skewness and handle outliers"
  }},
  {{
    "method": "combine", 
    "target_features": ["feature1", "feature2"],
    "params": {{
      "operation": "ratio",
      "handle_zero": "add_epsilon",
      "epsilon": 1e-8
    }},
    "expected_output": "ratio_feature1_feature2",
    "rationale": "Ratio captures relative strength between factors"
  }},
  {{
    "method": "normalize",
    "target_features": ["feature1"],
    "params": {{
      "method": "zscore",
      "window": 252,
      "min_periods": 20
    }},
    "expected_output": "zscore_feature1",
    "rationale": "Cross-sectional normalization for comparability"
  }}
]
```

## ENHANCEMENT GUIDELINES

1. **Diversification**: Use multiple enhancement methods to create diverse factors
2. **Risk Management**: Consider correlation between enhanced factors
3. **Robustness**: Prefer methods that are stable across different market regimes
4. **Interpretability**: Maintain economic intuition where possible
5. **Efficiency**: Balance complexity with computational efficiency

## EXAMPLE

Input:
```json
{{
  "raw_features": {{
    "price_momentum": {{"values": [0.01, 0.02, -0.01, 0.03], "statistics": {{"mean": 0.0125, "std": 0.015}}}},
    "volume_ratio": {{"values": [1.2, 0.8, 1.5, 0.9], "statistics": {{"mean": 1.1, "std": 0.3}}}}
  }},
  "enhancement_spec": {{
    "target_count": 5,
    "methods": ["transform", "combine"],
    "focus": "momentum"
  }}
}}
```

Output:
```json
[
  {{
    "method": "transform",
    "target_features": ["price_momentum"],
    "params": {{"type": "rank", "ascending": false}},
    "expected_output": "rank_price_momentum",
    "rationale": "Ranking reduces outlier impact while preserving order"
  }},
  {{
    "method": "combine",
    "target_features": ["price_momentum", "volume_ratio"],
    "params": {{"operation": "product"}},
    "expected_output": "momentum_volume_signal",
    "rationale": "Combines price and volume momentum for stronger signal"
  }}
]
```

Now process the following factor enhancement request:

Raw Features: {raw_features}
Enhancement Specification: {enhancement_spec}

Return only the JSON array of enhancement strategies, no additional explanation:""" 