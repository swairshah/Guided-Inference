import os
import random
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from termcolor import colored
from dataclasses import dataclass, field, asdict

from guided_inference import Stats, VLLM_TOKENIZER_IDS, prompt_format

LN_2 = 0.69314718056  # ln(2)

@dataclass
class SamplingStats:
    entropy: float = 0.0
    varentropy: float = 0.0
    top_prob: float = 0.0
    prob_top2_diff : float = 0.0
    
@dataclass
class AttentionStats:
    attn_entropy : float = 0.0
    attn_varentropy : float = 0.0
    agreement : float = 0.0
    interaction_strength : float = 0.0

def calculate_sampling_stats(logits: torch.Tensor, axis: int = -1) -> SamplingStats:
    sampling_stats = SamplingStats()
    
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    
    sampling_stats.prob_top2_diff = abs(probs.topk(k=2).values.diff().item())
    sampling_stats.top_prob = probs.max().item() 
    sampling_stats.entropy = entropy.item()
    sampling_stats.varentropy = varentropy.item()
    
    return sampling_stats

def calculate_attention_stats(attention_scores: torch.Tensor) -> AttentionStats:
    attention_stats = AttentionStats()
    
    attention_probs = F.softmax(attention_scores, dim=-1)                                                                                                       
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)

    attn_varentropy = torch.var(attn_entropy, dim=-1)
    attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    attention_stats.attn_entropy = torch.mean(attn_entropy).item()
    attention_stats.attn_varentropy = torch.mean(attn_varentropy).item()
    attention_stats.agreement = torch.mean(agreement).item()
    attention_stats.interaction_strength = interaction_strength.item()
    
    return attention_stats

def analyze_stats(stats: Stats, model, tokenizer, device, verbose=False):
   
    if verbose:
        for i, output in enumerate(stats.interleaved_outputs):
            if output['model_type'] == "oracle":
                print(colored(output['output'], 'blue'), end=" ")
            elif output['model_type'] == "base":
                print(colored(output['output'], 'red'), end=" ")
        print('\n______________')
    
        if not stats.parsed_answer: stats.parsed_answer = "Null"
    
        print(colored("PRED:" + stats.parsed_answer , 'light_cyan'), end="\n")
        print(colored("LABEL:" + stats.ground_truth, 'light_red'), end="\n")
    
        print('______________')
        if stats.ground_truth == stats.parsed_answer:
            print(colored(str("TRUE"), 'light_green'), end="\n")
        else:
            print(colored(str("FALSE"), 'light_magenta'), end="\n")
        print('______________')
    

    analysis_results = []

    partial_response = ""
    for i, output in enumerate(stats.interleaved_outputs):
        if output['model_type'] == "oracle":
            current_prompt = prompt_format(user_prompt=stats.prompt, assistant_completion=partial_response)
            #print(current_prompt)
            
            inputs = tokenizer(current_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True, output_attentions=True)
           
            logits = outputs.logits[0, -1, :]
            attention_scores = outputs.attentions[-1]
            
            sampling_stats = calculate_sampling_stats(logits=logits.cpu()) 
            attention_stats = calculate_attention_stats(attention_scores=attention_scores.cpu())
            
            # pop is necessary cause we may have cause we have may have consicutive base model generations 
            # so we can't just refer to stats.cheat_positions[i]. 
            cheat_position = stats.cheat_positions.pop(0)
            oracle_tokens = len(tokenizer.encode(output['output']))
            # import ipdb; ipdb.set_trace()
            analysis_stats = {}
            analysis_stats['position'] = cheat_position
            analysis_stats['normalized_position'] = cheat_position/stats.total_tokens_generated
            analysis_stats['oracle_tokens'] = oracle_tokens
            analysis_stats['oracle_tokens_normalized'] = oracle_tokens/stats.total_tokens_generated
            analysis_stats.update(**asdict(sampling_stats))
            analysis_stats.update(**asdict(attention_stats))
            
            #log_probs = torch.log_softmax(logits, dim=-1)
            #entropy = -torch.sum(torch.exp(log_probs) * log_probs)
            #varentropy = torch.sum(torch.exp(log_probs) * (log_probs + entropy) ** 2)
            analysis_results.append(analysis_stats)
            
        partial_response += output['output']

    return analysis_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)

    stats_data = []
    with open(args.stats_file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            stats = Stats(**data)
            stats_data.append(stats)

    entropy_data = []
    varentropy_data = []
    logprob_data = []  

    # Load model using the first stats object
    first_stats = stats_data[0]
    tokenizer = AutoTokenizer.from_pretrained(VLLM_TOKENIZER_IDS[first_stats.base_model_id], token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(VLLM_TOKENIZER_IDS[first_stats.base_model_id], token=os.environ["HF_TOKEN"]).to(device)
    #model_path = "/Users/shahswai/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14/" 
    #tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    for stats in tqdm(stats_data):
        analysis_results = analyze_stats(stats, model, tokenizer, device, args.verbose)
     
        if args.verbose:
            print(f"Analysis Results: {analysis_results}")
               
        # analysis = stats.to_dict()
        analysis = {}
        analysis['analysis_result'] = analysis_results
        if stats.ground_truth == stats.parsed_answer:
            analysis['acc'] = 1
        else:
            analysis['acc'] = 0
        
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(analysis)+"\n")
