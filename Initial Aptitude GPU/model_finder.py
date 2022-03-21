import torch as th
from torch import nn, optim
from torchvision import datasets, models
from torchvision import transforms as T
import numpy as np
import model_arch
import matplotlib.pyplot as plt

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Download Dataset
transforms = T.Compose([T.ToTensor(), T.Normalize(0.5,0.5)])
train_data = datasets.CIFAR10('CIFAR10_data/', transform=transforms, download=True, train=True)
trainloader = th.utils.data.DataLoader(train_data, shuffle = True, batch_size = 4000)

# Function for Driver   
# Returns the state dict of a model that satisfies the save condition                                                                 
def find_model():
    save_model = False
    #model = model_arch.network()

    model = models.vgg16(pretrained = False)

    model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=10, bias=True)
    )

    model.to(device)

    tries = 0
    uniques_totals = th.zeros(10)
    while not save_model and tries < 3000 :
        tries += 1

        model_arch.randomize_network(model)  
        outputs = th.zeros((10,10))

        new = True
        for images, labels in trainloader:
            if new: 
                new = False
                images, labels = images.to(device), labels.to(device)
                model_out = model(images)
                pred_labels = th.argmax(model_out, dim = 1)

                for l in range(len(labels)):
                    i = labels[l].item()
                    j = pred_labels[l].item()
                    outputs[i][j] += 1
            
        save_model, uniques = model_arch.save_condition(outputs)
        uniques_totals[uniques-1] += 1
        print(f'Try {tries}, Uniques: {uniques}')
        if save_model:
            print('Model Found!')
            print(f'UNIQUES TOTALS: {uniques_totals}')
            return model.state_dict(), tries

    print(f'UNIQUES TOTALS: {uniques_totals}')
