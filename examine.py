import numpy as np
import torch
from geospatial_fm import TemporalViTEncoder, ConvTransformerTokensToEmbeddingNeck

pretrained_path = "/content/drive/MyDrive/Prithvi_weights/Prithvi_100M.pt"
model = TemporalViTEncoder(pretrained = pretrained_path, embed_dim = 768)
nodel = ConvTransformerTokensToEmbeddingNeck(embed_dim = 768, output_embed_dim = 768, Hp = 14, Wp = 14)

example_input_1 = torch.tensor(np.random.randn(1, 6, 1, 224, 224)).float()
#example_input_2 = torch.tensor(np.random.randn(1, 3, 1, 224, 224)).float()

out = model(example_input_1)
#out = nodel(out)
print(out)
print(out.shape)
#temp = out
#out = torch.cat((out[0][0], out[1][0])).reshape(1, 1, -1, 768)
#print(out.shape)

