from collections import defaultdict
from copy import deepcopy
import datetime
import os
import pdb
import pickle
import random
import time

import adabound
import GPUtil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from graph import Graph, Flatten, layername_generator, get_layername_stem
from utils import get_num_params_from_model, random_weight_init, StableBCELoss, monitor_gpu_usage


MUTATION_TYPE_TUPLE = ('wider', 'deeper', 'wider_kernel', 'skip_connection')
LOG_INTERVAL = 100
TEST_ON_EPOCH = [1, 3, 7, 15, 31, 63]
MODEL_PATH = './models/'


def create_base_model(num_cells, input_dim, num_classes=10, dropout_conv = 0.7, dropout_fc = 0.5):
    last_conv_dim = (input_dim - 2 - 2 * (2**(num_cells - 1) - 1)) / (2**(num_cells - 1)) - 4
    if last_conv_dim < 1:
        raise ValueError("Input size too small!")
        
    flat_dim = int(((input_dim - 2 - 2 * (2**(num_cells - 1) - 1)) / (2**(num_cells - 1)) - 4)**2 * 128)

    # The head network
    head_layer_list = [('head_conv', nn.Conv2d(3, 64, kernel_size=3)),
                       ('head_relu', nn.ReLU(True)),
                       ('head_batchnorm', nn.BatchNorm2d(64))
                       ]
    head_layer_dict = nn.ModuleDict(head_layer_list)
    headnet = Graph(head_layer_dict)
    headnet.add_edge_from_layer_list(head_layer_list)

    # Create cell dicts
    cell_layer_list_dict = {'cell{}'.format(n):
                                               [(str(n) + '_' + 'conv_00000001', nn.Conv2d(128, 128, kernel_size=3)),
                                                (str(n) + '_' + layername_generator('relu'), nn.ReLU(True)),
                                                (str(n) + '_' + layername_generator('batchnorm'), nn.BatchNorm2d(128)),
                                                (str(n) + '_' + layername_generator('dropout'), nn.Dropout2d(dropout_conv))
                                               ]
                            for n in range(2, num_cells + 1)}

    cell_layer_list_dict['cell1'] = [('1' + '_' + 'conv_00000001', nn.Conv2d(64, 128, kernel_size=3)),
                                     ('1' + '_' + layername_generator('relu'), nn.ReLU(True)),
                                     ('1' + '_' + layername_generator('batchnorm'), nn.BatchNorm2d(128)),
                                     ('1' + '_' + layername_generator('dropout'), nn.Dropout2d(dropout_conv))
                                    ]

    cell_dict = {cellname: Graph(nn.ModuleDict(cell_layer_list)) for cellname, cell_layer_list in cell_layer_list_dict.items()}

    # Connect the internal layers for each cell
    for cellname, cell in cell_dict.items():
        cell.add_edge_from_layer_list(cell_layer_list_dict[cellname])

    # Create pooling dicts
    maxpool_layer_dict_dict = {'maxpool{}'.format(n): 
                                                nn.ModuleDict({
                                                               'nonc_' + layername_generator('maxpool'): nn.MaxPool2d(2),
                                                               })
                          for n in range(1, num_cells)}

    maxpool1_layer_dict = nn.ModuleDict({
                                         'nonc_' + layername_generator('maxpool'): nn.MaxPool2d(2),
                                         })

    maxpool_dict = {poolname: Graph(pool_layer_dict) for poolname, pool_layer_dict in maxpool_layer_dict_dict.items()}
    
    # The tail network
    tail_layer_list = [('tail_conv', nn.Conv2d(128, 128, kernel_size=3)),
                       ('tail_relu_1', nn.ReLU(True)),
                       ('tail_batchnorm', nn.BatchNorm2d(128)),
                       ('tail_flatten', Flatten()),
                       ('tail_dropout_1', nn.Dropout(dropout_fc)),
                       ('tail_fc_1', nn.Linear(flat_dim, 128)),
                       ('tail_relu_2', nn.ReLU(True)),
                       ('tail_dropout_2', nn.Dropout(dropout_fc)),
                       ('tail_fc_2', nn.Linear(128, num_classes)),
                       ('tail_logsoftmax', nn.LogSoftmax(dim=1))
                       ]

    tail_layer_dict = nn.ModuleDict(tail_layer_list)
    tailnet = Graph(tail_layer_dict)
    tailnet.add_edge_from_layer_list(tail_layer_list)

    # Combine graphs
    model_dict = {'headnet': headnet,
                  'tailnet': tailnet}

    model_dict = {**model_dict, **cell_dict, **maxpool_dict}

    model = headnet

    for n in range(1, num_cells + 1):
        model = model + cell_dict['cell{}'.format(n)]

        try:
            model = model + maxpool_dict['maxpool{}'.format(n)]
        except KeyError:
            pass
            
    model = model + tailnet

    # He Kaiming Initialization
    _ = random_weight_init(model)

    return model, model_dict


class Evo(object):
    def __init__(self, base_epochs, mutation_epochs, ft_epochs, lr=0.05, init_size=15):
        self.base_epochs = base_epochs
        self.mutation_epochs = mutation_epochs
        self.ft_epochs = ft_epochs
        self.lr = lr
        self.init_size = init_size

    def train(self, model, train_loader, valid_loader, num_epochs, stage, model_number, cuda=True):
        training_start_time = time.time()
        model.train()

        loss_func = nn.NLLLoss()

        cycle = 1

        print("Training started!")

        if torch.cuda.is_available():
            model = model.cuda()

        else:
            model = model.cpu()
        
        num_rows = len(train_loader.sampler) if train_loader.sampler else len(train_laoder.datasets)

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # SGDR Warm start
            if epoch == cycle:
                optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.5, weight_decay=0.0001, nesterov=True)
                optim_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * cycle,
                                                                       eta_min=0, last_epoch=-1)
                cycle = cycle * 2

            for batch_idx, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(async=True), target.cuda(async=True)
                else:
                    data, target = data.cpu(), target.cpu()

                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()

                output = model(data)

                loss = loss_func(output, target)
                loss[(loss != loss) | (loss == float("Inf"))] = 100
                loss_clamped = loss
                loss_sum = loss_clamped.mean()

                loss_sum.backward()

                optimizer.step()
                optim_scheduler.step()

                if (batch_idx + 1) % LOG_INTERVAL == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Validation loss: {:.6f}  LR: {:.6f}'.format(
                        epoch, batch_idx * len(data), num_rows,
                        100. * batch_idx * len(data) / num_rows, loss_sum.data.item(), optim_scheduler.get_lr()[0]))

            print("Epoch {} took {} seconds.".format(epoch, time.time() - epoch_start_time))

            if epoch in TEST_ON_EPOCH:
                print("=====================")
                acc = self.test(model, valid_loader, cuda=True)
                print("=====================")

        print("Training took {} seconds\n".format(round(time.time() - training_start_time, 2)))

        return model, acc

    @staticmethod
    def test(model, data_loader, cuda=True, verbose=True):
        print("Testing...\n")

        start_time = time.time()

        model.eval()

        loss_func = nn.NLLLoss()

        test_loss = 0
        correct = 0
        num_rows = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if torch.cuda.is_available() and cuda:
                    data, target = data.cuda(async=True), target.cuda(async=True)
                else:
                    data, target = data.cpu(), target.cpu()

                data, target = Variable(data), Variable(target)
                

                num_rows += len(data)

                output = model(data)
                

                test_loss += loss_func(output, target).mean()
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= num_rows

            if verbose:
                print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                    test_loss, correct, num_rows, (100. * correct.item()) / num_rows))

        print("Testing took {} seconds.".format(time.time() - start_time))

        return (100. * correct.item()) / num_rows

    def evolve(self, base_model_tuple, dataloader_tuple, num_generations=20, repeat=1, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        evo_list = []
        evo_param_list = []

        for t in range(1, repeat + 1):
            start_time = time.time()

            # Setup the template
            template_model_pair = base_model_tuple
            template_model, template_model_dict = template_model_pair
            num_cells = (len(template_model_dict) - 2) // 2 + 1

            train_loader, valid_loader, full_loader, test_loader = dataloader_tuple
        
            # Initial configurations
            mutation_type_tuple = MUTATION_TYPE_TUPLE

            # Train the template model
            print("Initializing template model...")
            num_params_template = get_num_params_from_model(template_model)
            print("Number of params of the model: {}M".format(num_params_template))

            random_weight_init(template_model)
            template_model, acc = self.train(template_model, train_loader, valid_loader, num_epochs=self.base_epochs, stage='base', model_number=0, cuda=True)
                
            template_model.cpu()
            torch.cuda.empty_cache()

            population = [deepcopy(template_model_pair) for _ in range(self.init_size)]  # All models are on CPU

            performance_dict = defaultdict(list)
            performance_list = []
            generation_best = []
            generation_mean = []

            tournament_size = len(population)
            pick_indices = list(range(tournament_size))

            for gen in range(1, num_generations + 1):
                print("==================")
                print("Generation {}".format(gen))
                print("==================")

                if gen == 1:
                    # Initial training
                    print("Start initialization (mutate and train {} models):".format(self.init_size))
                    for i in pick_indices:
                        model_pair = population[i]
                        model, model_dict = model_pair

                        mutation = np.random.choice(mutation_type_tuple)

                        print("Training model {}/{}".format(i + 1, self.init_size))
                        print("Mutation type: {}".format(mutation))

                        if mutation in {'deeper', 'skip_connection'}:
                            kwargs1 = {'cell_num': 1, 'new_layername_stem': None}

                            layername, new_layername = model.mutate(mutation, model_dict['cell1'], **kwargs1)

                            kwargs_dict = {'kwargs{}'.format(n): {'cell_num': n, 'new_layername_stem': get_layername_stem(new_layername)} for n in range(2, num_cells + 1)}

                            for n in range(2, num_cells + 1):
                                _ = model.mutate(mutation, model_dict['cell{}'.format(n)], layername='{}_'.format(n) + layername, **kwargs_dict['kwargs{}'.format(n)])
                                    
                        elif mutation == 'wider':
                            layername, new_width = model.mutate(mutation, model_dict['cell1'])

                            kwargs_dict = {'kwargs{}'.format(n): {'new_width': new_width} for n in range(2, num_cells + 1)}

                            for n in range(2, num_cells + 1):
                                _ = model.mutate(mutation, model_dict['cell{}'.format(n)], layername='{}_'.format(n) + layername, **kwargs_dict['kwargs{}'.format(n)])

                        else:
                            layername = model.mutate(mutation, model_dict['cell1'])

                            for n in range(2, num_cells + 1):
                                _ = model.mutate(mutation, model_dict['cell{}'.format(n)], layername='{}_'.format(n) + layername)

                        num_params_model = get_num_params_from_model(model)
                        print("Number of params of the model: {}M".format(num_params_model))
                        print("Model to template size ratio: {}".format(num_params_model / num_params_template))

                        model, acc = self.train(model, train_loader, valid_loader, self.mutation_epochs, stage='mutation', model_number=i, cuda=True)
                        performance_list.append(acc)
                        performance_dict[i].append(acc)

                        model.cpu()
                        torch.cuda.empty_cache()

                    # Save all models from population
                    for i, m in enumerate(population):
                        if not os.path.isdir(MODEL_PATH):
                            os.mkdir(MODEL_PATH)
                        torch.save(m, MODEL_PATH + 't{}_gen{}_idx{}.pth'.format(t, gen, i))

                else:
                    print("====================")
                    print("Performance sublist:")
                    performance_sublist = [(i, p) for i, p in enumerate(performance_list) if i in pick_indices]
                    print(performance_sublist)

                    champ_index = max(performance_sublist, key=lambda x: x[1])[0]
                    champ_performance = performance_list[champ_index]
                    print("Winner performance: ({}, {})".format(champ_index, champ_performance))
                    print("====================\n")

                    model_pair = population[champ_index]
                    model, model_dict = model_pair

                    mutation = np.random.choice(mutation_type_tuple)
                    print("Mutation type: {}".format(mutation))

                    if mutation in {'deeper', 'skip_connection'}:
                        kwargs1 = {'cell_num': 1, 'new_layername_stem': None}

                        layername, new_layername = model.mutate(mutation, model_dict['cell1'], **kwargs1)

                        kwargs_dict = {'kwargs{}'.format(n): {'cell_num': n, 'new_layername_stem': get_layername_stem(new_layername)} for n in range(2, num_cells + 1)}

                        for n in range(2, num_cells + 1):
                            _ = model.mutate(mutation, model_dict['cell{}'.format(n)], layername='{}_'.format(n) + layername, **kwargs_dict['kwargs{}'.format(n)])

                    elif mutation == 'wider':
                        layername, new_width = model.mutate(mutation, model_dict['cell1'])

                        kwargs_dict = {'kwargs{}'.format(n): {'new_width': new_width} for n in range(2, num_cells + 1)}

                        for n in range(2, num_cells + 1):
                            _ = model.mutate(mutation, model_dict['cell{}'.format(n)], layername='{}_'.format(n) + layername, **kwargs_dict['kwargs{}'.format(n)])

                    else:
                        layername = model.mutate(mutation, model_dict['cell1'])

                        for n in range(2, num_cells + 1):
                            _ = model.mutate(mutation, model_dict['cell{}'.format(n)], layername='{}_'.format(n) + layername)

                    num_params_model = get_num_params_from_model(model)

                    print("Number of params of the model: {}M".format(num_params_model))
                    print("Model to template size ratio: {}".format(num_params_model / num_params_template))
                    print("Training champ candidate (generation {}/{})".format(gen, num_generations))

                    model, acc = self.train(model, train_loader, valid_loader, self.mutation_epochs, stage='mutation', model_number=champ_index, cuda=True)

                    performance_list[champ_index] = acc
                    performance_dict[champ_index].append(acc)

                    model.cpu()
                    torch.cuda.empty_cache()
 
                tournament_size = int(np.floor(len(population) * 0.15))
                pick_indices = np.random.choice(len(population), tournament_size if tournament_size > 2 else 2,
                                                replace=False)
                generation_best.append(max(performance_list))
                generation_mean.append(round(np.mean(performance_list), 2))

                print("Best of the Generation:")
                print(generation_best)
                print("Mean of the Generation:")
                print(generation_mean)

            # Train with the whole training set
            champ_index = np.argmax(performance_list)
            champ_model, champ_model_dict = population[champ_index]

            print("Training the champ!")
            
            num_params_model = get_num_params_from_model(champ_model)
            print("Number of params of the model: {}M".format(num_params_model))
            print("Model to template size ratio: {}".format(num_params_model / num_params_template))
            print(champ_index, performance_list[champ_index])

            champ_model, acc = self.train(champ_model, full_loader, test_loader, self.ft_epochs, stage='champ', model_number='na', cuda=True)
            acc = self.test(champ_model, test_loader, cuda=True)

            print(" -> {:<15}{}%".format('Model:', acc))

            print("Saving model...(repeat = {})".format(t))

            torch.save(champ_model, MODEL_PATH + "t{}_champ.pth".format(t))

            with open(MODEL_PATH + "t{}_performance_dict.pkl".format(t), 'wb') as handle:
                pickle.dump(performance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("=================================")
            print("Evolution took {} hours. (repeat = {})".format((time.time() - start_time) / 3600, t))

            evo_list.append(acc)
            evo_param_list.append(num_params_model)

        print("Done!")

        return evo_list, evo_param_list
