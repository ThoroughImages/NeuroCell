import datetime
import os
import time

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from evolution import create_base_model, Evo, MUTATION_TYPE_TUPLE
from utils import Cutout

GPU_IDS = '8' # GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IDS

NUM_GENERATIONS = 20  # number of model generations
REPEATS = 1  # number of experiments
INIT_SIZE = 15  # number of models in the tournament
BATCH_SIZE = 32  # batch size
RANDOM_SEED = 42  # random seed

start_time = time.time()

print("EVO configurations:")
print("======================")
print("Number of generations: {}".format(NUM_GENERATIONS))
print("Number of repeats: {}".format(REPEATS))
print("Population size: {}".format(INIT_SIZE))
print("Batch size: {}".format(BATCH_SIZE))
print("Mutation types: {}".format(MUTATION_TYPE_TUPLE))
print("======================\n")

# Load CIFAR10 dataset
mean_tuple, std_tuple = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean_tuple, std_tuple),
                                      Cutout(n_holes=1, length=16)
                                      ])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean_tuple, std_tuple)
                                     ])

root = '/home/calvinku/projects/SearchForPath/data/'

training_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
validation_set = datasets.CIFAR10(root=root, train=True, transform=transform_test)
test_set = datasets.CIFAR10(root=root, train=False, transform=transform_test)

kwargs = {'num_workers': 4 * len(GPU_IDS.split(',')), 'pin_memory': False} if torch.cuda.is_available() else {}

valid_size = 0.2
num_train = len(training_set)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE,
                                           sampler=train_sampler, **kwargs)
valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE,
                                           sampler=valid_sampler, **kwargs)
full_train_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, **kwargs)

dataloader_tuple = (train_loader, valid_loader, full_train_loader, test_loader)


# Create evolution object
e = Evo(base_epochs=63, mutation_epochs=15, ft_epochs=63, lr=0.05, init_size=INIT_SIZE)
# e = Evo(base_epochs=1, mutation_epochs=1, ft_epochs=1)

base_model_tuple = create_base_model(num_cells=3, input_dim=32, num_classes=10, dropout_conv=0.7, dropout_fc=0.5)
evo_list, evo_param_list = e.evolve(base_model_tuple, dataloader_tuple=dataloader_tuple, num_generations=NUM_GENERATIONS, repeat=REPEATS)

print("=================================")
print("Entire experiment took {} hours.".format((time.time() - start_time) / 3600))

i = np.argmin(evo_param_list)
j = np.argmax(evo_list)

print("Best EVO: {} ({}M)".format(max(evo_list), evo_param_list[j]))
print("Mean EVO: {}".format(np.mean(evo_list)))
print("Least Param EVO: {} ({}M)".format(evo_list[i], evo_param_list[i]))

print("Time recorded: {}".format(str(datetime.datetime.now()) + "\n"))