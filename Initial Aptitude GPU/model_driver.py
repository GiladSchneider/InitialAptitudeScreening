import torch as th
from model_finder import find_model
from model_tester import test_model

# Decide which tests to run
find_nets = False
nets = 2

test_nets = True
net_names = ['Model_1.pth']
self_test = False
epochs = 30

# Finding Networks
if find_nets:
    print('FINDING NETWORKS...')
    for net in range(nets):
        found_model, tries = find_model()
        th.save(found_model, f'Model_{net+5}.pth')
        print(f"It took {tries} tries to find model {net+5}") 

# Testing Networks
if test_nets:
    print('TESTING NETWORKS...')
    for net_name in net_names:
        test_model(net_name, self_test=self_test, epochs=epochs)
        print()

