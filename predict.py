# imports
import torch
import sys
import json
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn.functional as F
from contextlib import contextmanager
from PIL import Image


def load_checkpoint(file_directory):
    # loading check point
    checkpoint = torch.load(file_directory)
    
    # defining architecture
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Model architecture is not defined.")
        
    # defining model and features parameters
    for x in model.parameters():
        x.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    
    # processes image
    img = Image.open(image)
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    processed_image = transformations(img)
    return processed_image

def predict(img, model, gpu, topk=5):
    model.eval()
    
    # if GPU is enabled, let's copy the relevant contents to GPU
    if gpu:
        model.cuda()
    
    # calculating classes and probabilities
    model_outputs = model.forward(Variable(img.unsqueeze(0), volatile=True))
    probabilities, classes = model_outputs.topk(topk)
    probabilities = probabilities.exp().data.cpu().numpy()[0]
    classes = classes.data.cpu().numpy()[0]
    class_keys = {x: y for y, x in model.class_to_idx.items()}
    classes = [class_keys[i] for i in classes]
    return probabilities, classes


if __name__ == '__main__':
    
    # initialize arguments and variables
    img_dir = "./ImageClassifier/flowers//test" + '/1/' + 'image_06752.jpg'
    check_point_path = "checkpoint_updated.pth"
    arguments = sys.argv[:]
    
    # defining default top_k
    if "--top_k" in arguments:
        top_k = arguments[arguments.index("--top_k") + 1]
    else:
        top_k = 5
        
    # cat_to_name.json
    if "--category_names" in arguments:
        category_names = arguments[arguments.index("--category_names") + 1]
    else:
        category_names = "./ImageClassifier/cat_to_name.json"
        
    if "--gpu" in arguments and torch.cuda.is_available():
        use_gpu = True
        print('\nRunning GPU...')
    elif not torch.cuda.is_available():
        use_gpu = False
        print('\nError: Cuda not available but --gpu was set.')
        print('Running CPU...\n')
    elif torch.cuda.is_available():
        use_gpu = True
        print('\nRunning GPU...')
    else:
        use_gpu = False
        print('\nRunning CPU...\n')

    with open(category_names, 'r') as f:
        category_dict = json.load(f)
        
    model = load_checkpoint(check_point_path)
    img = process_image(img_dir)
    
    if use_gpu:
            img = img.cuda()
            model = model.cuda()
    
    
    probabilities, classes = predict(img, model, use_gpu)
    y = [category_dict.get(i) for i in classes[::]]
    x = np.array(probabilities)
    
    print("\n\n** Computed Results obtained from image {} using model checkpoint {}**".format(img_dir, check_point_path))
    print("\nProbabilies: {}".format(x))
    print("Classes: {}\n\n".format(y))
        
        

    
    