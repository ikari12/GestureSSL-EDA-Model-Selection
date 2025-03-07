"""
References:
1. Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding. International Conference on Learning Representations. https://openreview.net/forum?id=8qDwejCuCN
2. https://openreview.net/forum?id=8qDwejCuCN

Acknowledgements:
- https://github.com/sanatonek/TNC_representation_learning?tab=readme-ov-file
- https://seunghan96.github.io/cl/ts/(CL_code3)TNC/
"""

import pandas as pd
import numpy as np
import pickle
import torch

BASE_DIR_UWAVE = 'data/athena/Gesture'

train_uwave = torch.load(f'{BASE_DIR_UWAVE}/train.pt')
test_uwave = torch.load(f'{BASE_DIR_UWAVE}/test.pt')

trainX = train_uwave['samples']
trainy = train_uwave['labels']
testX = test_uwave['samples']
testy = test_uwave['labels']

print(trainX.shape)
print(trainy.shape)
print(np.unique(trainy))

print(testX.shape)
print(testy.shape)
print(np.unique(testy))

## Save signals to file
with open(f'{BASE_DIR_UWAVE}/x_train.pkl', 'wb') as f:
    pickle.dump(trainX, f)
with open(f'{BASE_DIR_UWAVE}/x_test.pkl', 'wb') as f:
    pickle.dump(testX, f)
with open(f'{BASE_DIR_UWAVE}/state_train.pkl', 'wb') as f:
    pickle.dump(trainy, f)
with open(f'{BASE_DIR_UWAVE}/state_test.pkl', 'wb') as f:
    pickle.dump(testy, f)