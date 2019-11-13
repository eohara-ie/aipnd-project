import torch
from torch import nn
from collections import OrderedDict
from PIL import Image
from torchvision import datasets, transforms, models
import numpy as np
import json

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(device, filepath):
    
    checkpoint = torch.load(filepath)
    
    # Chosen network   
    arch = 'vgg11' ## Need to ensure this is the same as what is being loaded
    
    model = eval('models.' +  arch + '(pretrained=True)')
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
        ('relu1', nn.ReLU()),
        ('d1', nn.Dropout(0.5)),
        ('output', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
        ('lsoftmax', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    
    img_pil = Image.open(f'{image}' + '.jpg')
   
    # Necessary adjustments
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)

    # Converting to Numpy array 
    img_array = np.array(img_tensor)
    
    return img_array

def predict(device, image_path, model, topk):
    
    # Load model
    loaded_model = load_checkpoint(device, model)
    
    loaded_model.eval()
    loaded_model.to(device);
    
    # Pre-processing image
    img = process_image(image_path)
    # Convert to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)

    # Add dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0).to(device)

    
    # Set model to evaluation mode and turn off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Run image through network
        output = loaded_model.forward(img_add_dim)

    # Calculate probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Convert probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Load index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Invert index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list