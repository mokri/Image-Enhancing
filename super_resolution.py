
from Generator.gen import model
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

hr_height = 256
hr_width = 256
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def load_image(image_file):
    img = Image.open(image_file)
    # save_image(img, f"saved_images/image_orginal.png", normalize=False)
    return img


def enhance_image(image_file):
    image = image_loader(image_file)
    generated_image = model(image)
    # lr = nn.functional.interpolate(generated_image, scale_factor=4)
    # save_image(lr, f"saved_images/image_genere.png", normalize=False)
    # image = Image.open('saved_images/image_genere.png')
    image = transforms.ToPILImage()(generated_image.squeeze(0))
    return image




loader = transforms.ToTensor()


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image  # .cuda()  #assumes that you're using GPU


def upload_image(image_file):
    if image_file is not None:
        orignal_image = load_image(image_file)
        enhanced_image = enhance_image(image_file)

        return enhanced_image


def sr(image):
    return upload_image(image)

