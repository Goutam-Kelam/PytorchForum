import numpy as np
import torch

def NonZeor_1index(tensor,axis,invalid_item =-1):
    mask = tensor!=0
    indices = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_item)
    return indices

x = torch.randn(5, 7)
x[x<0] = 0
x = x.sort(dim=1)

# Convert Tensor to numpy array
# Be careful x is a tuple containing actual data and permuted indices
new_matrix = x[0].numpy()

NonZeor_1index(new_matrix,1,-1)

