# AI Quant Agent - Intelligent Quantitative Factor Research Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestrated-blue.svg)](https://langchain-ai.github.io/langgraph/)
[![KDB+](https://img.shields.io/badge/KDB+-Integrated-green.svg)](https://kx.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An AI-driven quantitative factor research platform that leverages LangGraph orchestration and KDB+ database integration for intelligent market data analysis and factor development.

## 🚀 Features

### Core Capabilities
- **🤖 AI-Driven Processing**: GPT-4 powered intelligent decision making throughout the pipeline
- **🔄 LangGraph Orchestration**: Dynamic workflow orchestration with state management and checkpointing
- **📊 KDB+ Integration**: Native integration with KDB+ database for high-performance time-series data
- **🎯 Specialized Processors**: Four core processors with distinct responsibilities:
  - **DataFetcher**: AI-optimized KDB+ query generation and data retrieval
  - **FeatureBuilder**: AI code generation using pykx and q language
  - **FactorAugmenter**: AI-powered factor enhancement and optimization
  - **BacktestRunner**: AI strategy design and intelligent metric selection

### Advanced Features
- **🛠️ Human Intervention**: Pause, resume, and inject modifications at any stage
- **📈 Real-time Monitoring**: Interactive UI with live pipeline status tracking
- **🔍 Debug & Evaluation**: Built-in debugging and result evaluation capabilities
- **💾 Checkpoint System**: Save and restore processing state at any point
- **🎨 Modern UI**: Beautiful Streamlit-based interface with real-time updates

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DataFetcher   │───▶│  FeatureBuilder │───▶│ FactorAugmenter │───▶│ BacktestRunner  │
│                 │    │                 │    │                 │    │                 │
│ • AI Query Gen  │    │ • AI Code Gen   │    │ • Factor Opt    │    │ • Strategy Gen  │
│ • KDB+ Fetch    │    │ • pykx/q Code   │    │ • Enhancement   │    │ • Performance   │
│ • Data Quality  │    │ • Validation    │    │ • Evaluation    │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────────────────┐
                    │         LangGraph Core              │
                    │                                     │
                    │ • State Management                  │
                    │ • Workflow Orchestration            │
                    │ • Checkpointing                     │
                    │ • Human Intervention                │
                    └─────────────────────────────────────┘
```

### Directory Structure
```
quant-agent/
├── src/
│   ├── core/                    # Core pipeline infrastructure
│   │   ├── base_processor.py    # Base processor with LangGraph integration
│   │   ├── decorators/          # Capability-based decorators
│   │   ├── pipeline_coordinator.py
│   │   └── workflow_nodes/      # LangGraph workflow nodes
│   ├── processors/              # Specialized processors
│   │   ├── data_fetcher.py      # AI-driven data retrieval
│   │   ├── feature_builder.py   # AI code generation
│   │   ├── factor_augmenter.py  # Factor enhancement
│   │   └── backtest_runner.py   # Strategy backtesting
│   ├── ui/                      # Interactive user interface
│   │   ├── components/          # Reusable UI components
│   │   ├── core/                # UI core infrastructure
│   │   └── factor_pipeline_ui.py
│   ├── prompt_lib/              # AI prompt templates
│   │   ├── data_fetcher/        # Data fetching prompts
│   │   ├── feature_builder/     # Code generation prompts
│   │   ├── factor_augmenter/    # Factor enhancement prompts
│   │   └── backtest_runner/     # Strategy design prompts
│   └── common/                  # Shared utilities
├── tests/                       # Comprehensive test suite
├── docs/                        # Project documentation
└── requirements.txt             # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- KDB+ database (for data access)
- OpenAI API key (for AI capabilities)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/quant-agent.git
cd quant-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and KDB+ configuration
```

### Environment Configuration
Create a `.env` file with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
KDB_DB_PATH=D:/kdbdb
MODEL_NAME=gpt-4
```

## 🚀 Usage

### Command Line Interface
```bash
# Run the main pipeline
python -m src.main --feature-description "Create a momentum factor for BTCUSDT"

# Run with specific configuration
python -m src.main --config config.yaml --feature-description "Build volatility factor"
```

### Interactive UI
```bash
# Start the Streamlit interface
cd src/ui
streamlit run factor_pipeline_ui.py

# Or use the provided runner
python run_ui.py
```

### Example Factor Requests

#### Momentum Factor
```
Create a momentum factor based on BTCUSDT 5-minute price changes, 
smoothed with a 20-period rolling average
```

#### Volatility Factor
```
Build a volatility factor using ETHUSDT hourly returns 
with 14-period standard deviation
```

#### Mean Reversion Factor
```
Generate a mean reversion factor for BTCUSDT using 
price deviation from 30-period moving average
```

## 🔧 Development

### Architecture Principles
- **LangGraph-First**: All workflows use LangGraph for orchestration
- **Capability-Based**: Processors use decorators to declare capabilities
- **AI-Driven**: GPT-4 integration for intelligent decision making
- **KDB+ Native**: Direct integration with KDB+ database
- **Human-Centric**: Support for human intervention and monitoring

### Processor Capabilities
Each processor can be decorated with capabilities:

```python
@observable(observers=["ui", "logger"])
@evaluable(max_retries=2)
@debuggable(max_retries=1)
@interruptible(save_point_id="data_fetch")
class DataFetcher(BaseProcessor):
    # Implementation...
```

### Adding New Processors
1. Inherit from `BaseProcessor`
2. Implement required abstract methods
3. Add capability decorators as needed
4. Register in `ProcessorFactory`

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/processors/
pytest tests/core/
```

## 📊 Performance & Scalability

### KDB+ Integration
- Native pykx integration for high-performance data access
- Optimized query generation using AI
- Efficient time-series data handling

### LangGraph Benefits
- Parallel processing where applicable
- State management and checkpointing
- Dynamic workflow orchestration
- Human intervention support

### Caching & Optimization
- Query result caching
- Expensive operation memoization
- Parallel data processing

## 🔍 Monitoring & Debugging

### Real-time Monitoring
- Interactive UI with live status updates
- Pipeline stage progress tracking
- Error reporting and debugging information

### Logging
- Structured logging throughout the system
- Comprehensive error handling
- Audit trail for research reproducibility

### Debugging Features
- Code evolution tracking
- Step-by-step execution monitoring
- Variable inspection and state analysis

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Follow the coding standards:
   - Use Python 3.12+ typing features
   - Follow LangGraph patterns
   - Maintain test coverage above 85%
   - Update documentation for all changes

### Code Quality
- **Formatting**: Black for code formatting
- **Linting**: Ruff for linting and import sorting
- **Type Checking**: MyPy for static type checking
- **Testing**: Comprehensive test suite with pytest

### Documentation
- Update API documentation for changes
- Maintain architecture documentation
- Include usage examples and tutorials

## 📚 Documentation

## 🏆 Key Technologies

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: AI integration and prompt management
- **KDB+/pykx**: High-performance time-series database
- **Streamlit**: Interactive web interface
- **OpenAI GPT-4**: AI-driven decision making
- **Python 3.12+**: Modern Python features and typing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangGraph team for the excellent workflow orchestration framework
- KX Systems for the powerful KDB+ database
- OpenAI for providing the GPT-4 API
- The open-source community for various supporting libraries

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the UI guide in `src/ui/README.md`

---

**Built with ❤️ for quantitative research and AI-driven factor development** 