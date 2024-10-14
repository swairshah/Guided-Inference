import os
import argparse
import requests
from pydantic import BaseModel, Field
from typing import List, Optional
import random

MODELS_IDS = {
    #"1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct",
    "8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70B": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "405B": "meta-llama/Meta-Llama-3.1-405B-Instruct"
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
    system_prompt="You are a helpful assistant.",
    max_tokens=20,
    temperature=0,
    top_p=1.0,
    return_tokens=False
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



if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True, choices=MODELS_IDS.keys())
    parser.add_argument("--user_prompt", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--return_tokens", action="store_true", help="Return tokenized response instead of string")
    args = parser.parse_args()

    model_id = MODELS_IDS[args.model_size]
    result = generate_chat_completion(
        model_id=model_id,
        user_prompt=args.user_prompt,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        return_tokens=args.return_tokens
    )
    print(result)
