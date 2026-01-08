# Kaggle Tools

## About

A modern Python toolkit containing utilities, scripts, and code snippets commonly used in Kaggle challenges and machine learning tasks. This library provides reusable components for data science competitions, deep learning workflows, and general ML experimentation.

## Features

- **Metrics**: Custom Keras metrics and evaluation tools
- **Classification**: Decision tree visualization and building utilities
- **Image Processing**: Image augmentation and preprocessing pipelines
- **Deep Learning**: Model management and training utilities
- **Diagnostics**: Performance monitoring and statistical analysis
- **Scaling**: Distributed computing with Dask
- **Submission**: Helper tools for competition submissions

## Installation

### Basic Installation

Install the package using pip:

```bash
pip install kaggle-tools
```

### Development Installation

For development work, clone the repository and install with development dependencies:

```bash
git clone https://github.com/yassineAlouini/kaggle-tools.git
cd kaggle-tools
pip install -e ".[dev]"
```

### Optional Dependencies

Install additional dependencies for specific features:

```bash
# For image processing
pip install kaggle-tools[image]

# For documentation
pip install kaggle-tools[docs]

# All optional dependencies
pip install kaggle-tools[dev,image,docs]
```

### Virtual Environment (Recommended)

Using a virtual environment is recommended to avoid dependency conflicts:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install kaggle-tools

# Using conda
conda create -n kaggle-tools python=3.10
conda activate kaggle-tools
pip install kaggle-tools
```

## Requirements

- Python ≥ 3.8
- Core dependencies: keras, pandas, tensorflow, xgboost, hyperopt

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

This project uses `black` for code formatting and `ruff` for linting:

```bash
# Format code
black kaggle_tools/

# Lint code
ruff kaggle_tools/

# Type checking
mypy kaggle_tools/
```

## Project Structure

```
kaggle-tools/
├── kaggle_tools/
│   ├── classification/      # Tree-based model utilities
│   ├── deep_learning/       # Deep learning model management
│   ├── diagnostic/          # Performance and statistical diagnostics
│   ├── features_engineering/# Feature engineering tools
│   ├── image_processing/    # Image augmentation and processing
│   ├── metrics/             # Custom metrics for model evaluation
│   ├── scale/               # Distributed computing tools
│   ├── submission/          # Competition submission helpers
│   └── tests/               # Unit tests
├── docs/                    # Documentation
├── pyproject.toml           # Project configuration and dependencies
└── README.md               # This file
```

## Contributing

Contributions are welcome! Feel free to:

- Submit bug reports and feature requests via [GitHub Issues](https://github.com/yassineAlouini/kaggle-tools/issues)
- Submit pull requests with improvements
- Share tools and utilities you find useful

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and formatting (`pytest`, `black .`, `ruff .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

The MIT License (MIT)

Copyright (c) 2018-2026 Yassine Alouini

See [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.
