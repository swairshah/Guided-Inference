from datasets import load_dataset
data = load_dataset("meta-llama/Llama-3.2-3B-Instruct-evals",
                    name="Llama-3.2-3B-Instruct-evals__math__details",
                    split="latest"
       )

data.to_json('llama-3.2-3b-instruct-math-eval.jsonl')
#import IPython; IPython.embed()

