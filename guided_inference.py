import os
import json
import argparse
import pandas as pd
import requests
import random
import logging
from infer import VLLM_MODELS_IDS, VLLM_TOKENIZER_IDS, VLLM_PORTS
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from termcolor import colored
from typing import List, Dict
from dataclasses import asdict
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Stats:
    c: int = 0
    k: int = 1
    max_new_tokens: int = 50
    temperature: float = 0
    top_p: float = 1.0
    base_model_id: str = ""
    oracle_model_id: str = ""
    prompt: str = ""
    formatted_prompt: str = ""
    interleaved_outputs: List[Dict] = field(default_factory=list)
    base_model_generations: List[str] = field(default_factory=list)
    oracle_model_generations: List[str] = field(default_factory=list)
    cheat_positions: List[int] = field(default_factory=list)
    total_tokens_generated: int = 0
    
    def to_json(self):
        return json.dumps(asdict(self))
    
def prompt_format(system_prompt, user_prompt, assistant_completion=""):
    if system_prompt:
        prompt="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> {assistant_completion}"""
        return prompt.format(system_prompt=system_prompt, user_prompt=user_prompt, assistant_completion=assistant_completion)
    else:
        prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> {assistant_completion}"""
        return prompt.format(user_prompt=user_prompt, assistant_completion=assistant_completion) 
    
 
def generate_vllm_completion(
    model_id, prompt, max_tokens, temperature, top_p, skip_special_tokens=False
):
    url = f"http://localhost:{VLLM_PORTS[model_id]}/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "skip_special_tokens": skip_special_tokens
    }
    
    response = requests.post(url, headers=headers, json=data)
    try:
        result = response.json()
        completion_tokens = result["usage"]["completion_tokens"]
        return result['choices'][0]['text'].strip(), completion_tokens
    except Exception as e:
        print("Error during vllm completion:", e)
        print(response.text)
        raise


stats = Stats()
def guided_inference(
    model_id,
    oracle_model_id,
    prompt,
    c,
    k,
    step_size=10,
    system_prompt="",
    max_new_tokens=50,
    temperature=0,
    top_p=1.0,
    verbose=False
):
    """
    guided inference using a smaller model with occasional help from a larger oracle model.
    Args:
        model_id (str): The ID of the smaller model to use for primary generation.
        oracle_model_id (str): The ID of the larger model to use as an oracle.
        prompt (str): The initial prompt to start the generation.
        c (int): The number of times to consult the oracle model.
        k (int): The number of tokens to generate from the oracle model each time it's consulted.
        system_prompt (str, optional): A system prompt to guide the model's behavior. Defaults to "".
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 50.
        temperature (float, optional): The temperature for generation. Defaults to 0.
        top_p (float, optional): The top_p value for nucleus sampling. Defaults to 1.0.
    """
    global stats
    
    stats.base_model_id = model_id
    stats.oracle_model_id = oracle_model_id
    stats.c = c
    stats.k = k
    stats.max_new_tokens = max_new_tokens
    stats.temperature = temperature
    stats.top_p = top_p
    stats.prompt = prompt
    current_prompt = prompt_format(system_prompt, prompt)
    stats.formatted_prompt = current_prompt
    
    interleaved_outputs = []
   
    assistant_completion = ""
    total_cheats = 0
    step_size = 10

    i = 0
    while i < max_new_tokens:
        current_prompt = prompt_format(system_prompt, prompt, assistant_completion=assistant_completion)
        if total_cheats < c and random.random() < 0.5:  
            oracle_output, completion_tokens = generate_vllm_completion(
                model_id=oracle_model_id,
                prompt=current_prompt,
                max_tokens=k,
                temperature=0,
                top_p=1.0,
                skip_special_tokens=True
            )
            interleaved_outputs.append({"output": oracle_output, "model_type" : "oracle"})
            
            # bookkeeping
            stats.oracle_model_generations.append(oracle_output)
            stats.cheat_positions.append(i)
            stats.total_tokens_generated += completion_tokens
           
            assistant_completion += oracle_output
            total_cheats += 1
            # Skip k-1 steps since we've already added k tokens
            i += completion_tokens - 1
            
        else:
            base_output, completion_tokens = generate_vllm_completion(
                model_id=model_id,
                prompt=current_prompt,
                max_tokens=step_size,
                temperature=temperature,
                top_p=top_p,
                skip_special_tokens=True
            )
            interleaved_outputs.append({"output": base_output, "model_type" : "base"})
            assistant_completion += base_output
            
            # bookkeeping
            stats.base_model_generations.append(base_output)
            stats.total_tokens_generated += completion_tokens
            
            i += completion_tokens
            
    if verbose:
        for i, output in enumerate(interleaved_outputs):
            if output['model_type'] == "oracle":
                print(colored(output['output'], 'blue'), end=" ")
            elif output['model_type'] == "base":
                print(colored(output['output'], 'red'), end=" ")
        print()  
    
    stats.interleaved_outputs = interleaved_outputs
        
    return assistant_completion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--oracle_model", type=str, required=True)
    parser.add_argument("--c", type=int, default=0, help="Number of times to cheat")
    parser.add_argument("--k", type=int, default=1, help="Number of tokens to generate when cheating from oracle")
    parser.add_argument("--step_size", type=int, default=10, help="Number of tokens to generate in each step of base model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use for inference")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the small model")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top P for the small model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    base_model_id = VLLM_MODELS_IDS[args.base_model]
    oracle_model_id = VLLM_MODELS_IDS[args.oracle_model]
    
    logging.info(f"Base model ID: {base_model_id}")
    logging.info(f"Oracle model ID: {oracle_model_id}")

    result = guided_inference(
        model_id=base_model_id,
        oracle_model_id=oracle_model_id,
        prompt=args.prompt,
        c=args.c,
        k=args.k,
        step_size=args.step_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose
    )
    print(result)
    print(stats)
    with open("stats.json", "w") as f:
        f.write(stats.to_json())
    



