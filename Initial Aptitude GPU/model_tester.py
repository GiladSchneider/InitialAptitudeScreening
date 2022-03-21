# Import dependencies
import torch as th
from torch import nn, optim
from torchvision import datasets, models
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from model_arch import network

def test_model(mdl_file, self_test = False, epochs = 30):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print(f'Device: {device}')
    
    # Transforms and Data Import
    transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    # Pointing to the data w data generators
    train_data = datasets.CIFAR10('CIFAR10_data', transform=transforms, train=True, download=True)
    test_data = datasets.CIFAR10('CIFAR10_data', transform=transforms, train=False, download=True)

    # Variables
    batch_size = 64
    num_workers = 0
    valid_ratio = .2

    # Seporating Validation and Training Set
    num_train = len(train_data)
    indeces = list(range(num_train))
    np.random.shuffle(indeces)
    num_valid = int(np.floor(valid_ratio*num_train))
    valid_indeces, train_indeces = indeces[:num_valid], indeces[num_valid:]

    # Training and Validation Samplers
    train_sampler = th.utils.data.sampler.SubsetRandomSampler(train_indeces)
    valid_sampler = th.utils.data.sampler.SubsetRandomSampler(valid_indeces)

    # Creating iterators for the data generators
    trainloader = th.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    validloader = th.utils.data.DataLoader(train_data, sampler=valid_sampler, batch_size=batch_size, num_workers=num_workers)
    testloader = th.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    # Models
    test_model = models.vgg16(pretrained = False)
    test_model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 10)
    )          
    test_model.load_state_dict(th.load(f'{mdl_file}'))
    test_model.to(device)
    
    model = models.vgg16(pretrained = False)
    model.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, 10)
    )
    if self_test:
        model.load_state_dict(th.load(f'{mdl_file}'))
    model.to(device)

    # Optimizers
    test_model_optimizer = optim.Adam(test_model.parameters(), lr=0.0001)
    model_optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Loss Function
    loss_function = nn.CrossEntropyLoss()

    # Normal Network Training
    # Tracking Validation Losses over Epochs
    model_validation_losses = []
    for e in range(epochs):

        # Training the model on the Training Set
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            model_out = model(images)
            loss = loss_function(model_out, labels)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
        else:
        # Testing the Model on the Validation Set    
            validation_loss = 0
            model.eval()
            with th.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)

                    model_out = model(images)
                    loss = loss_function(model_out, labels)
                    validation_loss += loss.item()
            
            # Adding the Validation loss, corrected for number of batches, to the validation losses
            model_validation_losses.append(validation_loss/len(validloader))
            print(f'Control Model epoch {e+1}, Loss: {validation_loss/len(validloader)}')

    # Recording the natural tendency of the network
    complete_relabel_outputs = th.zeros((10,10))
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        test_model_out = test_model(images)
        class_preds = th.argmax(test_model_out, dim = 1)
        for pred in range(len(class_preds)):
            i = labels[pred].item()
            j = class_preds[pred].item()
            complete_relabel_outputs[i][j] += 1

    used_nums = []
    new_labels = th.zeros(10)
    still_need_labels = []
    # For each class label
    for class_label in range(10):
        # Save the ideal contender for relabel to top_num
        top_num = th.argmax(complete_relabel_outputs[class_label])
        if top_num not in used_nums:
            used_nums.append(top_num)
            new_labels[class_label] = top_num
        # If the ideal contender is unavailable
        else:
            still_need_labels.append(class_label)
    
    # Repeatedly going through all class labels and giving them the best posable class reasignment
    pos = 1
    while len(still_need_labels) != 0:
        for class_label in still_need_labels:
            _, top_k = complete_relabel_outputs.topk(pos+1, dim=1)
            top_num = top_k[class_label][pos]
            if top_num not in used_nums:
                used_nums.append(top_num)
                new_labels[class_label] = top_num
                still_need_labels.remove(class_label)
        pos += 1

    # TODO: reassign data labels according to new labels
    for images, labels in trainloader:
        for label in labels:
            label = new_labels[label]

    # Testing the Test Network
    print()
    test_model_validation_losses = []
    for e in range(epochs):

        # Training the model on the Training Set
        test_model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            test_model_out = test_model(images)
            test_model_loss = loss_function(test_model_out, labels)

            test_model_optimizer.zero_grad()
            test_model_loss.backward()
            test_model_optimizer.step()
        else:
        # Testing the Model on the Validation Set    
            test_validation_loss = 0
            test_model.eval()
            with th.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)

                    test_model_out = test_model(images)
                    test_model_loss = loss_function(test_model_out, labels)
                    test_validation_loss += test_model_loss.item()
            
            # Adding the Validation loss, corrected for number of batches, to the validation losses
            test_model_validation_losses.append(test_validation_loss/len(validloader))
            print(f'Test Model epoch {e+1}, Loss: {test_validation_loss/len(validloader)}')

    plt.plot(model_validation_losses, label = 'Control Validation Loss')
    plt.plot(test_model_validation_losses, label = 'Test Model Validation Loss')
    plt.title(f'{mdl_file}')
    plt.legend(frameon=False)
    if self_test:
        plt.savefig(f'{mdl_file}{epochs}EpochsVSSelf.png')
    else: 
        plt.savefig(f'{mdl_file}{epochs}EpochsVSRandom.png')
    plt.clf()

    file = open('current_trials.txt', 'a')
    if self_test:
            file.write(f'{mdl_file} {epochs} Epochs VS Self.png')
    else: 
        file.write(f'{mdl_file} {epochs} Epochs VS Random.png')
    file.write(f'Test Model Validation Losses: {test_model_validation_losses}')
    file.write(f'Control Model Validation Losses: {model_validation_losses}\n')
    file.close()
