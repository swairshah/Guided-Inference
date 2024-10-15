import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import List, Dict

from guided_inference import Stats, VLLM_TOKENIZER_IDS, prompt_format

def analyze_stats(stats: Stats):

    tokenizer = AutoTokenizer.from_pretrained(VLLM_TOKENIZER_IDS[stats.base_model_id], token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(VLLM_TOKENIZER_IDS[stats.base_model_id], token=os.environ["HF_TOKEN"])
    model.eval()

    analysis_results = []

    partial_response = ""
    for i, output in enumerate(stats.interleaved_outputs):
        if output['model_type'] == "oracle":
            current_prompt = prompt_format(stats.prompt, partial_response)
            
            inputs = tokenizer(current_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits[0, -1, :]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            
            entropy = -torch.sum(torch.exp(log_probs) * log_probs)
            
            var_entropy = torch.sum(torch.exp(log_probs) * (log_probs + entropy) ** 2)
            
            analysis_results.append({
                "position": i,
                "entropy": entropy.item(),
                "var_entropy": var_entropy.item(),
                "log_probs": log_probs.tolist()
            })

        partial_response += output['output']

    return analysis_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.stats_file, "r") as f:
        stats = Stats(**json.load(f))

    analysis_results = analyze_stats(stats)

    print("Analysis Results:")
    for result in analysis_results:
        print(f"Position: {result['position']}")
        print(f"Entropy: {result['entropy']}")
        print(f"Variance of Entropy: {result['var_entropy']}")
        print("---")
