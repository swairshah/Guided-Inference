import argparse
import pandas as pd
import random
from infer import generate_chat_completion, MODELS_IDS

def guided_inference(
    model_id,
    oracle_model_id,
    prompt,
    c,
    k,
    system_prompt="",
    max_new_tokens=50,
    temperature=0,
    top_p=1.0
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

    generated = prompt
    total_cheats = 0
    cheat_positions = sorted(random.sample(range(max_new_tokens), c))
    full_output = ""
    step_size = 50

    i = 0
    while i < max_new_tokens:
        # Check if we should cheat at this position
        if total_cheats < c and i == cheat_positions[total_cheats]:
            # Call the oracle model to generate k tokens
            partial_completion = generated
            oracle_output = oracle_infer(oracle_model_id, prompt, partial_completion, new_tokens=k)
            # Append the oracle output to generated text
            generated += oracle_output
            full_output += oracle_output
            total_cheats += 1
            # Skip k-1 steps since we've already added k tokens
            i += k - 1
            continue

        # Generate next token with the small model
        response = generate_chat_completion(
            model_id=model_id,
            user_prompt=generated,
            system_prompt=system_prompt,
            max_tokens=step_size,
            temperature=temperature,
            top_p=top_p,
            return_tokens=False  # We'll get the generated text
        )
        # Append the generated token to the text
        generated += response
        full_output += response
        i += step_size

    return full_output

def oracle_infer(oracle_model_id, user_prompt, partial_completion, new_tokens=1):
    oracle_response = generate_chat_completion(
        model_id=oracle_model_id,
        user_prompt=partial_completion,
        system_prompt="You are a helpful assistant.",
        max_tokens=new_tokens,
        temperature=0,
        top_p=1.0,
        return_tokens=False
    )
    return oracle_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--oracle_model", type=str, required=True)
    parser.add_argument("--c", type=int, default=0, help="Number of times to cheat")
    parser.add_argument("--k", type=int, default=1, help="Number of tokens to generate when cheating")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use for inference")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the small model")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top P for the small model")

    args = parser.parse_args()

    small_model_id = MODELS_IDS[args.base_model]
    oracle_model_id = MODELS_IDS[args.oracle_model]
    result = guided_inference(
        model_id=small_model_id,
        user_prompt=args.prompt,
        c=args.c,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    print(result)


    #df = pd.read_json(args.dataset, lines=True)[:100]



