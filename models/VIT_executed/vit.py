# import torch
# import torch_mlir
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import torch.nn as nn
import torch
import torch_mlir
import os

import numpy as np

class prepare(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').vit.embeddings
	def forward(self,x):
		x = self.model(x)
		return x


class vit(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').vit.encoder
		self.layer2 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').vit.layernorm
		self.layer3 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').classifier
	def forward(self,x):
		print("x shape:",x.shape)

		x = self.layer1(x).last_hidden_state
		x = self.layer2(x)
		x = self.layer3(x)
		return x

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# # 将图像转换为numpy数组
# image_array = np.array(image)

# # 将三维数组转换为二维数组
# h, w, c = image_array.shape
# image_2d = image_array.reshape(h * w, c)

# # 保存图像数组为图像文件
# output_path = "/home/tianyi/Torchmlir/Models/MyModel/CGRAVIT/image.jpg"
# Image.fromarray(image_array).save(output_path)
# output_path = '/home/tianyi/Torchmlir/Models/MyModel/CGRAVIT/image_pixels_1d.txt'
# np.savetxt(output_path, image_array.reshape(-1),fmt='%d', delimiter=' ') # (640*480*3)
# output_path = '/home/tianyi/Torchmlir/Models/MyModel/CGRAVIT/image_pixels_2d.txt'
# np.savetxt(output_path, image_2d, fmt='%d', delimiter=' ') # (640*480)*3

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt").pixel_values
input_array = np.array(inputs)

print("input_array.shape:",input_array.shape)

# output_path = "/home/tianyi/Torchmlir/Models/MyModel/CGRAVIT/input.jpg"
# Image.fromarray(input_array).save(output_path)


model = prepare().eval()

example_input = model(inputs)
print("example_input:",example_input.shape)
example_input_array = example_input.squeeze(0).detach().numpy()
# example_input_array_2d = example_input_array.squeeze(0).numpy().tolist()
output_path = './input_pixels_2d.txt'
np.savetxt(output_path, example_input_array, fmt='%d', delimiter=' ') # (197*786)


vit_model = vit().eval()
output = vit_model(example_input)
print(output.shape)

linalg_on_tensors_mlir = torch_mlir.compile(
    vit_model,
    example_input,
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True)
file_path = 'vit.txt'
new_path = "linalg.mlir"
with open(file_path, 'wt') as f:
    print(linalg_on_tensors_mlir.operation.get_asm(), file=f)
os.rename(file_path,new_path)

TORCH_mlir = torch_mlir.compile(
    vit_model,
    example_input,
    output_type=torch_mlir.OutputType.TOSA,
    use_tracing=True)
file_path = 'vit.txt'
new_path = 'tosa.mlir'
with open(file_path, 'wt') as f:
    print(TORCH_mlir.operation.get_asm(large_elements_limit=2), file=new_path)
os.rename(file_path,new_path)
