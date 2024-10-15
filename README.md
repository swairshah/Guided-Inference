In the community the consensus is that : scaling always keeps imporving CELoss but sometimes that adds up to generating the correct tokens and then the final output that you get is right many more times compared to previous version of the model and you get a step change increase in the task performance.

while running eval on a small model if "every once in a while" you call a function oracle_infer(prompt, partial_completion, new_tokens=k) So the small model gets to cheat and ask tokens from a bigger model and continue from there. then the task performance would go up. 

There should be a correlation between c (how many times you cheat) and final score and also k (when you cheat how many tokens do you cheat for) and final score.There should be a correlation between c (how many times you cheat) and final score and also k (when you cheat how many tokens do you cheat for) and final score. 


Evals from Llama website:
```
+--------+-------+-------+-------+-------+-------+
| Model  |   1B  |   3B  |   8B  |  70B  | 405B  |
+--------+-------+-------+-------+-------+-------+
| MATH   |  30.6 |  48.0 |  51.9 |  68.0 |  73.8 |
+--------+-------+-------+-------+-------+-------+
| GPQA   |  27.2 |  32.8 |  32.8 |  46.7 |  51.1 |
+--------+-------+-------+-------+-------+-------+
```
___________

TODO: First establishing that (call the cheat function c times randomly during generation + for a fixed c call cheat f unction with varying k randomly during generation). and second try to figure out if there is a specific signal in the probability or entropy scores of the smaller model where we can get an indication WHEN to call the oracle model.

__________
first 100 MATH dataset:

1B accuracy: 17.05
8B accuracy: 42.05


Experiments: 
1B Base + 8B Oracle
```
+-----+-----+----------+
|  k  |  c  | accuracy |
+-----+-----+----------+
|  0  |  0  |   17.05  |
|  5  |  10 |   18.84  |
|  5  |  20 |   30.00  |
|  5  |  50 |   17.81  |
|  2  | 100 |   19.72  | 
|  5  | 100 |   22.67  | 
| 10  | 100 |   20.55  | 
|  ∞  |  ∞  |   42.04  | 
+-----+-----+----------+
```
______________


full 5000 MATH dataset (we independently evaluated):
1B: 27.93 
8B: 47.51



_____________________

Meta analysis:

on MATH dataset check the answers that 1B model got wrong and 3B model got right. For those check the first token 1B token makes a mistake, 
calculate the prob/entropy. 

