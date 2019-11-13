# Imports here
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store', help = 'Enter path to data directory.')
parser.add_argument('--save_dir', action='store', default = '/home/workspace/ImageClassifier/', help = 'Enter path to save model checkpoint.')
parser.add_argument('--arch', action='store', default = 'vgg16', help = 'Enter model architecture(e.g. vgg11).')
parser.add_argument('--learning_rate', action='store', default = 0.002, type = float, help = 'Enter learning rate.')
parser.add_argument('--hidden_units', action='store', default = 512, type = int, help = 'Enter number of units in hidden layer.')
parser.add_argument('--epochs', action='store', default = 6, type = int, help = 'Enter number of epochs for training.')
parser.add_argument('--gpu', action="store_true", default=False, help = 'Include to use GPU for training.')

args = parser.parse_args()
arch = args.arch
save_dir = args.save_dir
data_dir = args.data_dir
epochs = args.epochs
gpu = args.gpu
hidden_units = args.hidden_units
learning_rate = args.learning_rate

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

'''
print(f"Data directory is {data_dir}")
print(f"Checkpoint being saved to {save_dir}")
print(f"Test data folder at {test_dir}")
'''


# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.Resize(255),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Build and train network

# Use GPU if available and requested
if gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
    
# Load pre-trained network
model = eval('models.' +  arch + '(pretrained=True)')

# Define layer sizes and learning rate
input_size = model.classifier[0].in_features

output_size = 102 # Number of flower classes

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define un-trained network
from collections import OrderedDict
model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('d1', nn.Dropout(0.5)),
        ('output', nn.Linear(hidden_units, output_size)),
        ('lsoftmax', nn.LogSoftmax(dim=1))]))

# Define error function
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Move model to correct device if not already there
model.to(device);

# Define hyperparameters for training
steps = 0
running_loss = 0
print_every = 30

for epoch in range(epochs):
    running_loss = 0
        
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad()
        
        # Feed-forward, calculate loss, backpropagate and apply gradient descent step
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
        # Model validation
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            
            # Turn off dropout
            model.eval()
            
            # Turn off gradients as no longer training
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                
                    validation_loss += batch_loss.item()
                
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Training loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            
            # Turn dropout on
            model.train()
            
            
# Save the checkpoint
checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'class_to_idx': train_data.class_to_idx,
              'hidden_layers': hidden_units,
              'state_dict': model.state_dict()}

torch.save(checkpoint, save_dir + '/checkpoint_E.pth')