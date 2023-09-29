import numpy as np
import torch
from geospatial_fm import TemporalViTEncoder

pretrained_path = ""
model = TemporalViTEncoder(pretrained = pretrained_path)
example_input_1 = torch.tensor(np.random.randn(1, 3, 1, 224, 224)).float()
example_input_2 = torch.tensor(np.random.randn(1, 3, 1, 224, 224)).float()

print(model(example_input_1, example_input_2))
