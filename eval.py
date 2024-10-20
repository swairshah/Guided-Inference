import pandas as pd
from infer import generate_chat_completion, MODELS_IDS, BEDROCK_MODELS_IDS, VLLM_MODELS_IDS
from guided_inference import guided_inference, get_latest_stats
from tqdm import tqdm
import argparse
import json

from math_utils import normalize_final_answer, process_results, last_boxed_only_string, remove_boxed

def evaluate_accuracy(df, base_model_id, oracle_model_id, c, k, output_file=None, verbose=False):
    correct = 0
    total = 0
    all_stats = []
    
    for idx in tqdm(range(len(df)), desc="Evaluating"):
        prompt = df.iloc[idx]["input_final_prompts"][0]
        ground_truth = df.iloc[idx]["input_correct_responses"][0]
        eval_config = df.iloc[idx]["eval_config"]
        
        # result = generate_chat_completion(
        #     model_id=model_id,
        #     user_prompt=prompt,
        #     max_tokens=int(eval_config["max_gen_len"]),
        #     temperature=float(eval_config["temperature"]),
        #     return_tokens=False,
        #     provider="vllm"
        # )
        
        result = guided_inference(
            model_id=base_model_id,
            oracle_model_id=oracle_model_id,
            prompt=prompt,
            c=c,
            k=k,
            max_new_tokens=int(eval_config["max_gen_len"]),
            temperature=float(eval_config["temperature"]),
            verbose=verbose
        )
        
        stats = get_latest_stats()
        eval_stats = stats.to_dict()
        eval_stats["eval_config"] = eval_config
        eval_stats["prompt"] = prompt
        eval_stats["result"] = result
        eval_stats["ground_truth"] = ground_truth
        eval_stats["parsed_answer"] = None
        all_stats.append(eval_stats)
        
        try:
            final_parsed_answer = normalize_final_answer(remove_boxed(last_boxed_only_string(result)))
            eval_stats["parsed_answer"] = final_parsed_answer
            if final_parsed_answer == ground_truth:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
           
        if output_file:
            with open(output_file, "a") as f:
                f.write(json.dumps(eval_stats) + "\n")
        
        running_accuracy = correct / total if total > 0 else 0
        tqdm.write(f"Running Accuracy: {running_accuracy:.4f}")
    accuracy = correct / total if total > 0 else 0
    return accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model accuracy using guided inference.")
    parser.add_argument("--base_model", choices=VLLM_MODELS_IDS.keys(), default="1B", help="Base model to use")
    parser.add_argument("--oracle_model", choices=VLLM_MODELS_IDS.keys(), default="8B", help="Oracle model to use")
    parser.add_argument("--c", type=int, required=True, help="Number of candidates")
    parser.add_argument("--k", type=int, required=True, help="Number of tokens to reveal")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for evaluation")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default="eval_results.csv", help="File to output results to")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print verbose output")
    args = parser.parse_args()

    base_model_id = VLLM_MODELS_IDS[args.base_model]
    oracle_model_id = VLLM_MODELS_IDS[args.oracle_model]
    
    print(f"Base model: {args.base_model} ({base_model_id})")
    print(f"Oracle model: {args.oracle_model} ({oracle_model_id})")
    print(f"Cheat budget (c): {args.c}")
    print(f"Cheat length (k): {args.k}")
    print(f"Number of samples: {args.num_samples}")
    
    df = pd.read_json("data/llama-3.2-1b-instruct-math-eval.jsonl", lines=True)
    if args.start_index > 0:
        df = df.iloc[args.start_index:]
    elif args.num_samples > 0:
        df = df.iloc[:args.num_samples]
    
    accuracy = evaluate_accuracy(df, base_model_id, oracle_model_id, args.c, args.k, args.output_file, args.verbose)
    print(f"Accuracy: {accuracy:.4f}")

    #idx = 0
    #prompt = df.iloc[idx]["input_final_prompts"][0]
    #eval_config = df.iloc[idx]["eval_config"]
    #base_output = df.iloc[idx]["output_prediction_text"]
    #
    #ground_truth = df.iloc[idx]["input_correct_responses"][0]
    #
    #cheat_count = 10
    #cheat_length = 5
    
    # result = guided_inference(
    #     model_id=model_id,
    #     oracle_model_id=MODELS_IDS["70B"],
    #     prompt=prompt,
    #     c=cheat_count,
    #     k=cheat_length,
    #     max_new_tokens=int(eval_config["max_gen_len"]),
    #     temperature=eval_config["temperature"],
    # )
    
    #result = generate_chat_completion(
    #    model_id=model_id,
    #    user_prompt=prompt,
    #    max_tokens=int(eval_config["max_gen_len"]),
    #    #max_tokens=2048,
    #    temperature=float(eval_config["temperature"]),
    #    return_tokens=False,
    #    provider="vllm"
    #)
    #print(result)
    #try:
    #    final_parsed_answer = normalize_final_answer(remove_boxed(last_boxed_only_string(result)))
    #    print(final_parsed_answer)
    #except Exception as e:
    #    pass
    
    #import IPython; IPython.embed()
