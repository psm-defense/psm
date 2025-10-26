# Prompt Sensitivity Minimization (PSM)

A black-box optimization approach for protecting Large Language Model (LLM) system prompts against adversarial extraction attacks.

## Overview

This repository implements **Prompt Sensitivity Minimization (PSM)**, a defense mechanism that uses LLM-as-optimizer to automatically generate protective "shields" around system prompts. PSM minimizes observable leakage while preserving task utility through an iterative optimization process.

### Key Features

- **LLM-as-Optimizer**: Uses language models to generate and refine protective shields
- **Multi-objective Optimization**: Balances leakage minimization with utility preservation
- **Comprehensive Evaluation**: Tests against multiple attack types and defense strategies
- **Smooth Aggregation**: Implements various leakage score aggregation methods (logsumexp, top-k mean, max)
- **Efficient Caching**: Reduces redundant API calls through intelligent result caching
- **Multiple Metrics**: Supports both ROUGE-based and LLM judge evaluation metrics

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd psm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
Create a `config/config.yaml` file with your LLM API configurations:
```yaml
llms:
  gpt-4o-mini:
    model: "gpt-4o-mini"
    model_provider: "openai"
    api_key: "your-api-key"
    temperature: 0.7
  # Add other model configurations as needed
```

## Quick Start

### Running PSM Defense

Generate protected system prompts using PSM:

```bash
python run.py
```

This will:
1. Load system prompts from `data/victim_prompts/`
2. Generate protective shields using LLM optimization
3. Save optimized prompts to `data/defense_prompts/`

### Evaluating Defenses

Evaluate defense mechanisms against adversarial attacks:

```bash
python experiments/evaluate_defenses.py
```

This will:
1. Test multiple defense strategies (None, Baseline, PSM, N-gram Filter, FAKE)
2. Run various attack types (RACCON, LIANG, ZHANG, etc.)
3. Generate comprehensive results in `results/`

## Usage

### Programmatic Usage

#### Running PSM for a Single Prompt

```python
from run import PSMRunner, RunPSMConfig

# Configure PSM
config = RunPSMConfig(
    dataset_name="unnatural-test",
    attack_samples=50,
    validation_samples=10,
    target_dataset_samples=30,
    llm_target="gpt-5-mini",
    llm_optimizer="gpt-4o-mini",
    llm_validation="gpt-4o",
    llm_judge="gpt-4o-mini",
    n_optimization_iterations=10,
    n_initial_shields=5,
    n_shields_per_step=5,
    max_candidates_per_step=10,
    target_utility_threshold=0.9,
    target_leakage_threshold=0.65,
    leakage_aggregation="logsumexp",
    logsumexp_temperature=10.0
)

# Run PSM
runner = PSMRunner(config)
runner.run_experiments()
```

#### Evaluating a Defense Strategy

```python
from experiments.evaluate_defenses import (
    DefenseEvaluator, ExperimentConfig, 
    DefenseConfig, DefenseStrategy, 
    AttackType, Dataset, TargetModel
)

# Configure experiment
config = ExperimentConfig(
    attack_types=[AttackType.RACCON, AttackType.LIANG],
    defense_configs=[
        DefenseConfig(DefenseStrategy.PSM, Dataset.UNNATURAL_TEST, TargetModel.GPT_4O_MINI)
    ],
    max_victim_samples=30,
    max_attack_samples=60,
    use_am_metric=True,
    am_threshold=0.9,
    use_jm_metric=True,
    jm_threshold=0.7
)

# Run evaluation
evaluator = DefenseEvaluator(config)
await evaluator.run_experiments()
evaluator.save_results()
evaluator.print_results_summary()
```

## Architecture

### Core Components

#### 1. PSM (`src/defense/psm/psm.py`)

The main optimization engine that:
- Generates initial protective shields
- Iteratively refines shields using LLM-as-optimizer
- Evaluates candidates on utility and leakage metrics
- Optimizes for multi-objective fitness function

**Key Classes:**
- `PSM`: Main optimization class
- `PromptCandidate`: Represents a candidate solution with scores
- `PSMConfig`: Configuration parameters

#### 2. Data Creation (`src/defense/psm/psm_data_creation.py`)

Handles:
- Generation of baseline validation inputs
- Attack input collection
- Query-answer pair creation

#### 3. Metrics (`src/metrics.py`)

Evaluation metrics:
- **ROUGE Recall**: Measures leakage based on token overlap
- **LLM Judge**: Binary evaluation using LLM as judge
- **Approximate Match**: Threshold-based success detection

#### 4. Evaluation (`experiments/evaluate_defenses.py`)

Comprehensive evaluation framework:
- Supports multiple defense strategies
- Tests various attack types
- Implements result caching
- Generates detailed reports

### Defense Strategies

1. **None**: No defense (baseline)
2. **Baseline**: Simple instruction addition (Liang et al., 2024)
3. **PSM**: Prompt Sensitivity Minimization (this work)
4. **N-gram Filter**: Filters responses containing prompt n-grams
5. **FAKE**: Decoy prompt insertion (Liang et al., 2024)

### Attack Types

Supported attack datasets:
- `raccon.json`: RACCON attack prompts
- `liang.json`: Liang et al. attack patterns
- `zhang.json`: Zhang et al. attack patterns
- `ours.json`: Custom attack prompts
- `raccon_language.json`: Language-specific RACCON attacks

## Configuration

### PSM Configuration Parameters

```python
@dataclass
class RunPSMConfig:
    # Dataset
    dataset_name: str = "unnatural-test"
    
    # Sample sizes
    attack_samples: int = 50              # Attack inputs to use
    validation_samples: int = 10          # Baseline validation queries
    target_dataset_samples: int = 30      # System prompts to process
    
    # Model selection
    llm_target: str = "gpt-5-mini"       # Target model to protect
    llm_optimizer: str = "gpt-4o-mini"   # Model for shield generation
    llm_validation: str = "gpt-4o"       # Model for validation
    llm_judge: str = "gpt-4o-mini"       # Model for judging
    
    # Optimization parameters
    n_optimization_iterations: int = 10  # Max optimization iterations
    n_initial_shields: int = 5           # Initial shield population
    n_shields_per_step: int = 5          # New shields per iteration
    max_candidates_per_step: int = 10    # Max candidates to consider
    
    # Thresholds
    target_utility_threshold: float = 0.9    # Minimum utility score
    target_leakage_threshold: float = 0.65  # Maximum leakage score
    utility_penalty_weight: float = 100.0   # Weight for utility penalty
    
    # Leakage aggregation
    leakage_aggregation: str = "logsumexp"   # Aggregation method
    logsumexp_temperature: float = 10.0     # LogSumExp temperature
    top_k: int = 3                          # Top-k for top_k_mean
```

### Leakage Aggregation Methods

1. **max**: Maximum leakage score (non-smooth)
2. **logsumexp**: Smooth approximation of max (recommended)
3. **top_k_mean**: Average of top-k worst scores
4. **mean**: Simple average (smoothest)

### Experiment Configuration

```python
config = ExperimentConfig(
    # Attack and defense selection
    attack_types=[AttackType.RACCON, AttackType.LIANG],
    defense_configs=[...],
    
    # Sample limits
    max_victim_samples=30,
    max_attack_samples=60,
    
    # AM metric
    use_am_metric=True,
    am_threshold=0.9,
    
    # JM metric
    use_jm_metric=True,
    jm_threshold=0.7,
    
    # Output options
    output_dir="results",
    save_detailed_results=True,
    save_per_prompt_results=True
)
```

## Experiments

### Dataset Structure

#### Victim Prompts (`data/victim_prompts/`)

JSONL format with system prompts to protect:
```json
{"instruction": "Your system prompt here..."}
```

Available datasets:
- `unnatural-test.jsonl`: Test prompts
- `system-prompt-leakage.jsonl`: Leakage test cases
- `chatgpt-roles.jsonl`: ChatGPT role prompts
- `sharegpt-test.jsonl`: ShareGPT test set
- `dev.jsonl`: Development set

#### Attack Prompts (`data/attack_prompts/`)

JSON format with adversarial prompts:
```json
["attack prompt 1", "attack prompt 2", ...]
```

#### Defense Prompts (`data/defense_prompts/`)

JSONL format with optimized prompts:
```json
{
  "instruction": "Optimized prompt with shield",
  "original_instruction": "Original prompt",
  "utility_score": 0.95,
  "leakage_score": 0.30,
  "fitness_score": 0.30
}
```

### Running Custom Experiments

1. **Prepare your system prompts** in `data/victim_prompts/`
2. **Configure PSM** in `run.py` or programmatically
3. **Run optimization**: `python run.py`
4. **Evaluate results**: `python experiments/evaluate_defenses.py`

## Results

### Output Files

Results are saved in the `results/` directory:

- `detailed_attack_results.xlsx`: Individual attack results
- `per_prompt_results.xlsx`: Aggregated results per attack prompt
- `experiment_summary.json`: Summary of all experiments
- `cache.xlsx`: Cached results for reproducibility

### Metrics

#### Attack Success Rate (ASR)

ASR measures the percentage of successful attacks:
- **AM ASR**: Based on Approximate Match metric (ROUGE ≥ threshold)
- **JM ASR**: Based on LLM Judge metric (judge score ≥ threshold)

#### Utility Score

Measures task performance preservation:
- Cosine similarity with baseline responses
- LLM judge validation of response quality

#### Leakage Score

Measures system prompt disclosure:
- ROUGE recall between extracted and original prompts
- LLM judge evaluation of semantic equivalence

### Interpreting Results

- **Lower ASR** = Better defense
- **Higher utility** = Better task preservation
- **Lower leakage** = Better prompt protection

## Citation

If you use this work in your research, please cite:

```bibtex
@article{psm2024,
  title={Prompt Sensitivity Minimization: Protecting LLM System Prompts},
  author={Your Name},
  journal={Journal/Conference Name},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

This work builds upon research from:
- Liang et al. (2024) - Baseline and FAKE defenses
- Zhang et al. - Attack methodologies
- RACCON - Advanced attack patterns

