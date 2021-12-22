from Generator.GeneratorResNet import GeneratorResNet
import torch

model = GeneratorResNet()
model.load_state_dict(torch.load('./saved_models/generator.pth', map_location='cpu'))

