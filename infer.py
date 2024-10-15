import os
import argparse
import requests
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import random
import boto3

MODELS_IDS = {
    "3B":   "meta-llama/Meta-Llama-3.2-3B-Instruct",
    "8B":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70B":  "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "405B": "meta-llama/Meta-Llama-3.1-405B-Instruct"
}

VLLM_MODELS_IDS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct",
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
    "70B": "meta-llama/Llama-3.1-70B-Instruct",
}

VLLM_PORTS = {
    "meta-llama/Llama-3.2-1B-Instruct": 8001,
    "meta-llama/Llama-3.2-3B-Instruct": 8003,
    "meta-llama/Llama-3.1-8B-Instruct": 8008,
    "meta-llama/Llama-3.1-70B-Instruct": 8070,
}

VLLM_TOKENIZER_IDS = {
    "meta-llama/Llama-3.2-1B-Instruct" : "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct" : "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct" : "meta-llama/Llama-3.1-70B-Instruct", 
    "meta-llama/Llama-3.1-70B-Instruct" :"meta-llama/Llama-3.1-70B-Instruct" 
}

BEDROCK_MODELS_IDS = {
    "1B": "us.meta.llama3-2-1b-instruct-v1:0",
    "3B": "us.meta.llama3-2-3b-instruct-v1:0",
    "8B": "meta.llama3-8b-instruct-v1:0",
    "70B": "meta.llama3-70b-instruct-v1:0", 
}

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[dict] = None

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

def generate_chat_completion(
    model_id,
    user_prompt,
    system_prompt="",
    max_tokens=20,
    temperature=0,
    top_p=1.0,
    return_tokens=False,
    provider="hyperbolic"  # New parameter to choose between Hyperbolic and Bedrock
):
    if provider == "hyperbolic":
        return generate_hyperbolic_chat_completion(
            model_id, user_prompt, system_prompt, max_tokens, temperature, top_p, return_tokens
        )
    elif provider == "bedrock":
        return generate_bedrock_chat_completion(
            model_id, user_prompt, system_prompt, max_tokens, temperature, top_p
        )
    elif provider == "vllm":
        return generate_vllm_chat_completion(
            model_id, user_prompt, system_prompt, max_tokens, temperature, top_p
        )
    else:
        raise ValueError("Invalid provider. Choose 'hyperbolic', 'bedrock', or 'vllm'.")


def generate_hyperbolic_chat_completion(
    model_id, user_prompt, system_prompt, max_tokens, temperature, top_p, return_tokens
):
    url = "https://api.hyperbolic.xyz/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["HYPERBOLIC_API_KEY"]
    }
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
    
    data = {
        "messages": messages,
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    response = requests.post(url, headers=headers, json=data)
    try:
        chat_completion = ChatCompletionResponse.model_validate(response.json())
        content = chat_completion.choices[0].message.content
    except Exception as e:
        print(response.json())
        raise e

    if return_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        output_token_ids = tokenizer.encode(content)
        return output_token_ids
    else:
        return content
   
def prompt_format(system_prompt, user_prompt):
    if system_prompt:
        prompt="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)
    else:
        prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt.format(user_prompt=user_prompt) 
    
def generate_vllm_completion(
    model_id, user_prompt, system_prompt, max_tokens, temperature, top_p
):
    url = f"http://localhost:{VLLM_PORTS[model_id]}/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    prompt = prompt_format(system_prompt, user_prompt)
    
    data = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    response = requests.post(url, headers=headers, json=data)
    try:
        result = response.json()
        return result['choices'][0]['text'].strip()
    except Exception as e:
        print("Error during vllm completion:", e)
        print(response.text)
        raise
    
def generate_vllm_chat_completion(
    model_id, user_prompt, system_prompt, max_tokens, temperature, top_p
):
    url = f"http://localhost:{VLLM_PORTS[model_id]}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    data = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    response = requests.post(url, headers=headers, json=data)
    try:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("Error during vllm inference:", e)
        print(response.text)
        raise

def generate_bedrock_chat_completion(
    model_id, user_prompt, system_prompt, max_tokens, temperature, top_p
):
    bedrock = boto3.client('bedrock-runtime')
    
    prompt = prompt_format(system_prompt, user_prompt)
    
    body = json.dumps({
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_gen_len": max_tokens,
    })
    
    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        result = response.get('body').read()
        result_json = json.loads(result.decode('utf-8'))
        return result_json['generation'].strip()
    except Exception as e:
        print("Error during model inference:", e)
        raise



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_size", type=str, required=True, choices=MODELS_IDS.keys())
    parser.add_argument("--model_size", type=str, required=True, choices=BEDROCK_MODELS_IDS.keys())
    parser.add_argument("--user_prompt", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--return_tokens", action="store_true", help="Return tokenized response instead of string")
    parser.add_argument("--provider", type=str, default="bedrock", choices=["hyperbolic", "bedrock", "vllm"])
    args = parser.parse_args()

    if args.provider == "bedrock":
        model_id = BEDROCK_MODELS_IDS[args.model_size]
    elif args.provider == "vllm":
        model_id = VLLM_MODELS_IDS[args.model_size]
    else:
        model_id = MODELS_IDS[args.model_size]
    result = generate_chat_completion(
        model_id=model_id,
        user_prompt=args.user_prompt,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        return_tokens=args.return_tokens,
        provider=args.provider
    )
    print(result)
