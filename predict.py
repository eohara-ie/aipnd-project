import argparse
from utilities import *

parser = argparse.ArgumentParser()

parser.add_argument('img_filepath', action='store', help = 'Enter path to load test image.') # e.g. ./flowers/test/27/image_06887
parser.add_argument('checkpoint_filepath', action='store', help = 'Enter path to load model checkpoint.') # e.g. ./petals/checkpoint_E.pth
parser.add_argument('--top_k', action='store', default = 5, type = int, help = 'Enter top-K most likely classes to return.')
parser.add_argument('--category_names', action='store_true', default = False, help = 'Include to map flower cateogry to real names.')
parser.add_argument('--gpu', action="store_true", default=False, help = 'Include to use GPU for training.')

args = parser.parse_args()
img_filepath = args.img_filepath
checkpoint_filepath = args.checkpoint_filepath
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

# Use GPU if available and requested
if gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# Return predicted top-K classes/names and their probabilities

# Make prediction
probabilities,classes = predict(device, img_filepath, checkpoint_filepath, topk = top_k)

print(img_filepath)

# Converting classes to names if desired
if category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = []
    for i in classes:
        names += [cat_to_name[i]]
    print(names)
else:
    print(classes)
print(probabilities)