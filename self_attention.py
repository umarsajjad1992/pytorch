import torch

# The inputs
inputs = [[1, 0, 1, 0],
          [0, 2, 0, 2],
          [1, 1, 1, 1]]

inputs = torch.tensor(inputs, dtype=torch.float32)

 # Initializing the weights for the keys, querys and values
w_k = [[0, 0, 1],
       [1, 1, 0],
       [0, 1, 0],
       [1, 1, 0]]

w_q = [[1, 0, 1],
       [1, 0, 0],
       [0, 0, 1],
       [0, 1, 1]]

w_v = [[0, 2, 0],
       [0, 3, 0],
       [1, 0, 3],
       [1, 1, 0]]

w_k = torch.tensor(w_k, dtype=torch.float32)
w_q = torch.tensor(w_q, dtype=torch.float32)
w_v = torch.tensor(w_v, dtype=torch.float32)

# getting keys, querys and values by doing dot product of inputs and weights
keys = inputs @ w_k
querys = inputs @ w_q
values = inputs @ w_v

# Let's get the attention scores by performing dot-prod of querys and keys
attention_scores = querys @ keys.T

# Taking the softmax of the attention scores
from torch.nn.functional import softmax

attention_scores_soft = softmax(attention_scores, dim=-1)

# Let's get the weighted values and sum them
weighted_vals = values[:, None] * attention_scores_soft.T[:, :, None]
outputs =  weighted_vals.sum(dim=0)
print(outputs)