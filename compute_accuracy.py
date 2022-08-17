import torch
import cv2
import argparse
# import time
from utils import (
    load_resnet50_model,
    load_efficientnet_v1_model, load_efficientnet_v2_model, preprocess_efficientnet_v2, read_classes, run_through_model
)
# Construct the argumet parser to parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/hog.jpg', 
                    help='path to the input image')
parser.add_argument('-d', '--device', default='cuda', 
                    help='computation device to use', 
                    choices=['cpu', 'cuda'])
args = vars(parser.parse_args())

# Set the computation device.
DEVICE = args['device']

# Initialize the model.
model = load_efficientnet_v2_model()

# Load the ImageNet class names.
categories = read_classes()

# Initialize the image transforms.
transform = preprocess_efficientnet_v2() # NOTE: efficient_v2 전용의 preprocess 함수를 따로 만들었다!! @kookie12

model.to(DEVICE)

if __name__ == "__main__":
    val_images_dir = './ImageNet_val'
    val_grnd_truth_dir = './val.txt'
    top5_accuracy_500 = run_through_model(model, val_images_dir, val_grnd_truth_dir, 500, transform=transform)
    print(top5_accuracy_500)


