import numpy as np
import torch
from geospatial_fm import TemporalViTEncoder

pretrained_path = "/content/drive/MyDrive/Prithvi_weights/Prithvi_100M.pt"
model = TemporalViTEncoder(pretrained = pretrained_path, embed_dim = 768)
example_input_1 = torch.tensor(np.random.randn(1, 3, 1, 224, 224)).float()
example_input_2 = torch.tensor(np.random.randn(1, 3, 1, 224, 224)).float()

print(model(example_input_1, example_input_2))
