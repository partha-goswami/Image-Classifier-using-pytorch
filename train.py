# Imports here
import sys
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict

if __name__ == '__main__':
    arguments = sys.argv[:]
 
# Check point is saved at working directory by default, otherwise at the directory specified
if "--save_dir" in arguments:
        save_directory = arguments[arguments.index("--save_dir") + 1]
else:
        save_directory = False
        
# 'densenet121' is used as the default model
if "--arch" in arguments:
    architecture = arguments[arguments.index("--arch") + 1].lower()
    if architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    else:
        print('Model isn\'t found.')
        sys.exit()
else:
    architecture = 'densenet121'
    model = models.densenet121(pretrained=True)
    input_size = 1024
    
# Learning rate is kept by default at 0.001
if "--learning_rate" in arguments:
    learning_rate = arguments[arguments.index("--learning_rate") + 1]
else:
     learning_rate = 0.001
        
# By default, the hidden units are kept at 490
if "--hidden_units" in arguments:
    hidden_units = arguments[arguments.index("--hidden_units") + 1]
    hidden_units = hidden_units.split(',')
    hidden_units = [int(layer) for layer in hidden_units]
else:
    hidden_units = [490]
    
# specifying the output size
hidden_units.append(102)

# epochs is kept at 3, by default
if "--epochs" in arguments:
    epochs = arguments[arguments.index("--epochs") + 1]
else:
    epochs = 3

# Whether to use GPU or CPU
if "--gpu" in arguments and torch.cuda.is_available():
    use_gpu = True
    print('\nExecuting in GPU...\n')
elif not torch.cuda.is_available():
    use_gpu = False
    print('\nError: Cuda not available but --gpu was set.')
    print('Executing CPU...\n')
else:
    use_gpu = True
    print('\nExecuting in GPU...\n')
    
data_dir = "./ImageClassifier/flowers"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"

# local variables
steps = 0
recurring_loss = 0.0
n_print = 10


# transforms, datasets and loaders
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
validation_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
training_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_data_transforms)
training_loaders = DataLoader(training_datasets,batch_size=32,shuffle=True)
validation_loaders = DataLoader(validation_datasets,batch_size=32,shuffle=True)
class_idx = training_datasets.class_to_idx

# defining layers and hidden layers
nn_layers = nn.ModuleList([nn.Linear(input_size, hidden_units[0])])
nn_hidden = zip(hidden_units[:-1], hidden_units[1:])
nn_layers.extend([nn.Linear(x, y) for x, y in nn_hidden])

# gradient tracking isn't required
for x in model.parameters():
    x.requires_grad = False

# parameters and drop out percentage    
parameters = OrderedDict()
dropout = 0.3

for count in range(len(nn_layers)):
    if count == 0:
        parameters.update({'drop{}'.format(count + 1): nn.Dropout(p=dropout)})
        parameters.update({'fc{}'.format(count + 1): nn_layers[count]})
    else:
        parameters.update({'relu{}'.format(count + 1): nn.ReLU()})
        parameters.update({'drop{}'.format(count + 1): nn.Dropout(p=dropout)})
        parameters.update({'fc{}'.format(count + 1): nn_layers[count]})

# using log softmax as output
parameters.update({'output': nn.LogSoftmax(dim=1)})
model.classifier = nn.Sequential(parameters)
model.cuda() if use_gpu else model.cpu()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

for e in range(epochs):
    print('Starting training process...')
    print('Training run {}...'.format((e + 1)))
    model.train()
    
    for images, labels in iter(training_loaders):
        steps += 1
        optimizer.zero_grad()
        inputs = Variable(images)
        training_labels = Variable(labels)
        if use_gpu:
            inputs = inputs.cuda()
            training_labels = training_labels.cuda()
        
        # calculating recurring loss
        output = model.forward(inputs)
        
            
        
        
        loss = criterion(output, training_labels)
        loss.backward()
        recurring_loss += loss.item()
        
        optimizer.step()
        
        if steps % n_print == 0:
            model.eval()
            valid_loss = 0.0
            accuracy = 0.0
            
            for i, (images, labels) in enumerate(validation_loaders):
                inputs = Variable(images, volatile=True)
                valid_labels = Variable(labels, volatile=True)
                
                if use_gpu:
                    inputs = inputs.to('cuda')
                
                output = model.forward(inputs)
                
                if use_gpu:
                    labels = labels.to('cuda')
                
                valid_loss += criterion(output, labels).data[0]
                ps = torch.exp(output).data
                accuracy += (labels.data == ps.max(1)[1]).type_as(torch.FloatTensor()).mean()
                
                print("Caclculated Training Loss: {:.3f}.. ".format(loss))
                print("Calculated Validation Loss: {:.3f}..".format(valid_loss / len(validation_loaders)))
                print("Calculated Validation Accuracy: {:.3f}..\n".format(accuracy / len(validation_loaders)))

                recurring_loss = 0.0
                model.train()

# check point is saved
checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'epochs': epochs,
                  'arch': architecture,
                  'hidden_units': [each.out_features for each in model.classifier if
                                   hasattr(each, 'out_features') == True],
                  'learning_rate': learning_rate,
                  'class_to_idx': class_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}

if save_directory:
    checkpoint_path = save_directory + '/checkpoint_updated.pth'
else:
    checkpoint_path = 'checkpoint_updated.pth'

torch.save(checkpoint, checkpoint_path)


        




