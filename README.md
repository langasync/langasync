# langasync

Async API integration library for multiple providers.

## Setup

Activate the virtual environment:
```bash
source venv/bin/activate
```

Install in editable mode:
```bash
pip install -e .
```

Install with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
langasync/
├── langasync/
│   ├── core/              # Core abstractions and base classes
│   │   ├── base.py        # BaseProvider class
│   │   ├── exceptions.py  # Custom exceptions
│   │   └── types.py       # Shared type definitions
│   └── providers/         # Provider implementations
├── tests/
│   ├── test_core/
│   └── test_providers/
└── pyproject.toml
```

## Running Tests

```bash
pytest
```
