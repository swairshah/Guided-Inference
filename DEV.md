# Guided Inference Development Guide

## Overview

This project implements guided inference using a smaller base model with occasional help from a larger oracle model. The system allows a small model to "cheat" by consulting a larger model at strategic points during generation, evaluating how this affects performance on mathematical reasoning tasks.

## Project Structure

```
.
├── guided_inference.py    # Core guided inference implementation
├── infer.py              # Model inference utilities (Hyperbolic, Bedrock, vLLM)
├── eval.py               # Evaluation script for MATH dataset
├── stats_analysis.py     # Statistical analysis of model decisions
├── math_utils.py         # MATH dataset processing utilities
├── todo.txt              # Project TODOs
└── data/                 # Dataset directory
    └── llama-3.2-1b-instruct-math-eval.jsonl
```

## Environment Setup

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- vLLM (for local model serving)
- AWS Bedrock access (optional)
- Hyperbolic API access (optional)

### Installation
```bash
pip install torch transformers vllm pandas requests boto3 termcolor tqdm sympy datasets
```

### Environment Variables
```bash
export HF_TOKEN="your_huggingface_token"
export HYPERBOLIC_API_KEY="your_hyperbolic_api_key"  # Optional
```

## Model Serving Setup

The project uses vLLM for local model serving. Start the required models on specific ports:

```bash
# 1B model on port 8001
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8001

# 3B model on port 8003  
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8003

# 8B model on port 8008
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8008

# 70B model on port 8070
vllm serve meta-llama/Llama-3.1-70B-Instruct --port 8070
```

## Running Experiments

### Basic Guided Inference

Test the guided inference mechanism directly:

```bash
python guided_inference.py \
    --base_model 1B \
    --oracle_model 8B \
    --c 5 \
    --k 10 \
    --prompt "What is 2+2?" \
    --max_new_tokens 100 \
    --verbose
```

Parameters:
- `c`: Number of times to consult oracle (cheat budget)
- `k`: Number of tokens to generate from oracle each time
- `--verbose`: Show colored output (blue=oracle, red=base)

### MATH Dataset Evaluation

Run evaluation on the MATH dataset:

```bash
python eval.py \
    --base_model 1B \
    --oracle_model 8B \
    --c 20 \
    --k 10 \
    --num_samples 100 \
    --output_file results_1b_8b_c20_k10.jsonl \
    --verbose
```

### Statistical Analysis

Analyze model statistics at oracle consultation points:

```bash
python stats_analysis.py \
    --stats_file results_1b_8b_c20_k10.jsonl \
    --output_file analysis_results.jsonl \
    --device cuda \
    --verbose
```

## Key Parameters

### Guided Inference Parameters
- `c` (cheat count): How many times to consult oracle
- `k` (cheat length): Tokens generated per oracle consultation
- `step_size`: Tokens generated by base model per step (default: 10)
- `cheat_probability`: Dynamic probability that increases with position

### Model Configuration
Base models available: `1B`, `3B`, `8B`, `70B`
- Maps to specific Llama model variants
- Port assignments defined in `VLLM_PORTS`

## Expected Results

Based on experiments in README.md:

### 1B Base + 8B Oracle on MATH (first 100 samples)
| k  | c   | Accuracy |
|----|-----|----------|
| 0  | 0   | 17.05%   |
| 5  | 10  | 18.84%   |
| 5  | 20  | 30.00%   |
| 5  | 50  | 17.81%   |
| ∞  | ∞   | 42.04%   |

### Full MATH Dataset (5000 samples)
| k  | c  | Accuracy |
|----|----| ---------|
| 0  | 0  | 27.93%   |
| 10 | 20 | 31.73%   |
| ∞  | ∞  | 47.51%   |

## Data Format

### Evaluation Output
Each line in output JSONL contains:
```json
{
  "c": 20,
  "k": 10,
  "base_model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "oracle_model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "...",
  "result": "...",
  "ground_truth": "42",
  "parsed_answer": "42",
  "interleaved_outputs": [...],
  "cheat_positions": [15, 45, 78],
  "total_tokens_generated": 120
}
```

### Statistical Analysis Output
Extends evaluation output with:
```json
{
  "analysis_result": [
    {
      "position": 0,
      "entropy": 2.34,
      "varentropy": 0.45,
      "top_prob": 0.78,
      "prob_top2_diff": 0.23,
      "attn_entropy": 3.21,
      "attn_varentropy": 0.56,
      "agreement": 0.67,
      "interaction_strength": 0.89
    }
  ],
  "acc": 1
}
```

## Troubleshooting

### Common Issues

1. **vLLM Connection Errors**: Ensure models are running on correct ports
2. **CUDA Memory**: Use smaller models or reduce batch size
3. **HuggingFace Authentication**: Verify HF_TOKEN is set correctly
4. **Missing Dependencies**: Install sympy for math evaluation

### Debug Mode
Use `--verbose` flag to see:
- Colored model outputs (blue=oracle, red=base)
- Real-time accuracy updates
- Detailed analysis statistics

## TODO

1. Implement entropy-based oracle triggering instead of random
2. Analyze PRM800k dataset for correlation between statistics and reasoning quality
3. Explore optimal values for c and k parameters
4. Investigate attention patterns at decision points
