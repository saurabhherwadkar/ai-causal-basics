# AI Causal Basics

A structured causal inference learning framework that demonstrates foundational concepts in causal reasoning using probabilistic graphical models. Built with **pgmpy**, **Pyro**, and **DoWhy**, this project provides well-organized, production-quality code for learning and experimenting with causal AI techniques.

## Key Learning Objectives

- Understand the difference between correlation and causation through hands-on probabilistic modeling
- Build and parameterize Bayesian Networks using directed acyclic graphs (DAGs) to encode causal assumptions
- Apply parameter learning techniques including Maximum Likelihood Estimation, Bayesian estimation, and Expectation-Maximization for latent variables
- Perform exact inference using Variable Elimination to answer conditional probability queries on causal models
- Use Pyro probabilistic programming to implement generative causal models and run importance sampling for posterior inference
- Simulate and visualize causal structures with DoWhy, including instruments, confounders, effect modifiers, and front-door paths
- Reason about interventions (do-calculus) versus observations and understand why causal graphs matter for decision-making
- Gain practical experience with the transportation survey dataset to see how demographic variables causally influence outcomes
- Learn production-quality software patterns for causal AI projects including configuration management, structured logging, and test-driven development
- Develop intuition for when and how to apply causal inference methods versus standard machine learning approaches

## About the Project

This project walks through core causal inference concepts step by step:

1. **Probability Primer** - Discrete probability distributions, normalization, and factor representations
2. **Bayesian Networks** - DAG construction, parameter learning (MLE, Bayesian, EM), and conditional probability queries
3. **Sampling-Based Inference** - Pyro probabilistic programming with importance sampling for posterior computation
4. **Causal Visualization** - DoWhy-generated DAGs with instruments, confounders, effect modifiers, and front-door variables

The transportation survey dataset (from [altdeep/causalML](https://github.com/altdeep/causalML)) is used throughout to demonstrate how Age, Sex, Education, Occupation, Residence, and Transportation mode are causally related.

## Project Structure

```
ai-causal-basics/
├── config/
│   ├── config.yaml              # Main configuration (log levels, URLs, model params)
│   └── config.local.yaml        # Local overrides (git-ignored, env-specific)
├── src/
│   ├── __init__.py              # Root package initializer
│   ├── run_probability_primer.py # Entry point: probability demos
│   ├── run_bayesian_network.py  # Entry point: full BN pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── discrete_distribution.py  # Probability distribution wrapper
│   │   ├── bayesian_network.py       # BN model, DAG, latent variable EM
│   │   └── pyro_causal_model.py      # Pyro generative model + importance sampling
│   ├── inference/
│   │   ├── __init__.py
│   │   └── variable_elimination.py   # Exact inference queries
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── dag_visualizer.py         # DoWhy simulation + NetworkX rendering
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py          # YAML config with local overrides
│       ├── data_loader.py            # Secure data download with caching
│       └── logger.py                 # Rotating file + console logging
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared test fixtures
│   ├── test_config_loader.py
│   ├── test_logger.py
│   ├── test_discrete_distribution.py
│   ├── test_bayesian_network.py
│   ├── test_variable_elimination.py
│   ├── test_pyro_causal_model.py
│   └── test_dag_visualizer.py
├── logs/                             # Log output directory (git-ignored)
├── output/figures/                   # Generated visualizations (git-ignored)
├── data/cache/                       # Cached datasets (git-ignored)
├── .gitignore
├── Pipfile                           # Dependency management
├── Pipfile.lock                      # Locked dependency versions
├── pyproject.toml                    # Linting, formatting, test config
├── LICENSE
└── README.md
```

## Dependencies

### Runtime Dependencies

| Package       | Version  | Purpose                                      |
|---------------|----------|----------------------------------------------|
| pgmpy         | >= 0.1.25 | Bayesian network construction and inference |
| pyro-ppl      | >= 1.9.1  | Probabilistic programming (importance sampling) |
| torch         | >= 2.3.0  | Tensor computation backend for Pyro         |
| dowhy         | >= 0.11.1 | Causal inference and DAG simulation         |
| pandas        | >= 2.2.0  | Data manipulation and CSV loading           |
| numpy         | >= 1.26.0 | Numerical computing                         |
| networkx      | >= 3.3    | Graph data structures and algorithms        |
| matplotlib    | >= 3.9.0  | Plotting and visualization                  |
| graphviz      | >= 0.20.3 | Graph rendering (Pyro model visualization)  |
| pyyaml        | >= 6.0.1  | YAML configuration file parsing             |

### Development Dependencies

| Package       | Version  | Purpose                                      |
|---------------|----------|----------------------------------------------|
| pytest        | >= 8.2.0  | Testing framework                           |
| pytest-cov    | >= 5.0.0  | Code coverage reporting                     |
| pytest-mock   | >= 3.14.0 | Mocking utilities for tests                 |
| ruff          | >= 0.4.0  | Fast Python linter                          |
| black         | >= 24.4.0 | Code formatter                              |
| mypy          | >= 1.10.0 | Static type checker                         |
| bandit        | >= 1.7.8  | Security vulnerability scanner              |
| safety        | >= 3.2.0  | Dependency vulnerability checker            |

## Deployment and Setup

### Prerequisites

- Python 3.11 or higher
- pipenv (install via `pip install pipenv`)
- Graphviz system package (for Pyro model rendering)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/ai-causal-basics.git
   cd ai-causal-basics
   ```

2. **Install dependencies:**

   ```bash
   pipenv install --dev
   ```

3. **Activate the virtual environment:**

   ```bash
   pipenv shell
   ```

4. **Verify installation:**

   ```bash
   pipenv run test
   ```

### Environment-Specific Configuration

Create a `config/config.local.yaml` file to override settings for your environment:

```yaml
# Example: override log level and disable plot display
logging:
  level: "DEBUG"

visualization:
  show_plots: false
```

This file is git-ignored and will not be committed.

## Usage

### Running the Probability Primer

```bash
python -m src.run_probability_primer
```

Demonstrates discrete probability distributions, phi values, and normalization.

### Running the Bayesian Network Pipeline

```bash
python -m src.run_bayesian_network
```

Runs the full pipeline: DAG definition, MLE/Bayesian parameter learning, EM for latent variables, Variable Elimination inference, Pyro importance sampling, and DoWhy visualization.

## Development

### Running Tests

```bash
# Run all tests with coverage
pipenv run test

# Run specific test module
pipenv run pytest tests/test_bayesian_network.py -v

# Run tests matching a pattern
pipenv run pytest -k "test_create" -v
```

### Linting and Formatting

```bash
# Check linting issues
pipenv run lint

# Auto-format code
pipenv run format

# Run type checking
pipenv run typecheck
```

### Security Scanning

```bash
# Scan source code for security issues
pipenv run security

# Check dependencies for known vulnerabilities
pipenv run safety check
```

## Configuration Reference

All configurable parameters are in `config/config.yaml`. Key settings:

| Setting                          | Default          | Description                           |
|----------------------------------|------------------|---------------------------------------|
| `logging.level`                  | INFO             | Log verbosity (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `logging.file_enabled`           | true             | Enable file logging                   |
| `data.cache_enabled`             | true             | Cache downloaded datasets locally     |
| `data.request_timeout_seconds`   | 30               | HTTP download timeout                 |
| `model.pyro.num_samples`         | 5000             | Importance sampling sample count      |
| `model.bayesian_network.pseudo_counts` | 1          | Dirichlet prior strength              |
| `visualization.show_plots`       | true             | Display plots interactively           |
| `security.ssl_verify`            | true             | Verify SSL certificates               |

## Architecture Highlights

- **Single Responsibility** - Each class has one well-defined purpose
- **Configuration-Driven** - All parameters externalized to YAML
- **Secure by Default** - URL whitelisting, SSL verification, input validation
- **Observable** - Structured logging with configurable levels and rotation
- **Testable** - Dependency injection, factory patterns, comprehensive test suite
- **Memory-Efficient** - Local data caching, file rotation, lazy initialization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
