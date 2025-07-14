# Contributing to EchoSphere AI-vCPU

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Poetry or pip for dependency management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BigBossBoolingB/EchoShereAIvCPU.git
cd EchoShereAIvCPU
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting and style checking
- **MyPy**: Type checking
- **isort**: Import sorting

### Running Quality Checks

```bash
# Format code with Black
black .

# Check linting with Flake8
flake8 .

# Type checking with MyPy
mypy echosphere/

# Sort imports
isort .
```

### Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=echosphere

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Architecture Overview

EchoSphere follows a hybrid Actor-Blackboard model:

- **WorkspaceActor**: Central coordination using Global Workspace Theory
- **Knowledge Source Actors (KSAs)**: Specialized processing modules
- **Vector Symbolic Architecture**: Neuro-symbolic representation
- **Neo4j Integration**: Hybrid vector-graph memory storage

### Making Changes

1. Create a feature branch
2. Make your changes
3. Run quality checks
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

### Project Structure

```
echosphere/
├── cognitive/          # GWT implementation and actors
├── memory/            # VSA and graph storage
├── execution/         # Task execution engine
├── learning/          # RL and adaptive components
├── monitoring/        # Observability and metrics
├── verification/      # Formal verification tools
└── utils/            # Configuration and utilities
```
