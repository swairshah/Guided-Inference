import pandas as pd
from infer import generate_chat_completion, MODELS_IDS
from guided_inference import guided_inference

from math_utils import normalize_final_answer, process_results, last_boxed_only_string, remove_boxed

model_id = MODELS_IDS["3B"]
print(model_id)
df = pd.read_json("llama-3.2-1b-instruct-math-eval.jsonl", lines=True)

idx = 0
prompt = df.iloc[idx]["input_final_prompts"][0]
eval_config = df.iloc[idx]["eval_config"]
base_output = df.iloc[idx]["output_prediction_text"]

ground_truth = df.iloc[idx]["input_correct_responses"][0]

cheat_count = 10
cheat_length = 5

result = guided_inference(
    model_id=model_id,
    oracle_model_id=MODELS_IDS["70B"],
    prompt=prompt,
    c=cheat_count,
    k=cheat_length,
    max_new_tokens=int(eval_config["max_gen_len"]),
    temperature=eval_config["temperature"],
)

# result = generate_chat_completion(
#     model_id=model_id,
#     user_prompt=prompt,
#     max_tokens=eval_config["max_gen_len"],
#     temperature=eval_config["temperature"],
#     return_tokens=False
# )
# print(result)
try:
    final_parsed_answer = normalize_final_answer(remove_boxed(last_boxed_only_string(result)))
    print(final_parsed_answer)
except Exception as e:
    pass

import IPython; IPython.embed()