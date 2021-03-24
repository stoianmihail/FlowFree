import numpy as np
import dimod
import os
import copy

from helpers import Flow as fl
from pyqubo import Binary
from dimod.generators.constraints import combinations
import dwave
import dimod
from dwave.system import LeapHybridSampler
from hybrid.reference.kerberos import KerberosSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from os import path


def get_storage_index():
    file_count = 1
    if path.isdir("results"):
        while path.isfile("results/flow_experiment_"+str(file_count)+".csv"):
            file_count += 1
    else:
        try:
            os.mkdir("results")
        except OSError:
            print("Creation of the directory 'results' failed")
        else:
            print("Successfully created the directory 'results'")
    return file_count


def get_label(posY, posX, color):
    return "{posY},{posX}_{color}".format(**locals())


def get_tensor_index(label):
    name, coord, color = label.split('_')
    x, y = coord.split(',')
    return [int(x), int(y), int(color)]


def tensor_to_sample(tensor):
    sample = {}
    shape = np.shape(tensor)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for c in range(shape[2]):
                sample["x_" + get_label(i, j, c)] = tensor[i, j, c]
    return sample


def get_label_list_x(size, num_colors):
    label_list = []
    for i in range(size):
        for j in range(size):
            for c in range(num_colors):
                label_list.append("x_" + get_label(i, j, c))
    return label_list


def get_label_list_y(flow):
    label_list = []
    for i in range(flow.get_size):
        for j in range(flow.get_size):
            for c in range(flow.get_num_colors):
                for nb in range(flow.get_num_helpers[i, j]):
                    label_list.append("y_" + get_label(i, j, c) + "," + str(nb))
    return label_list


def sample_to_tensor(items, initial_tensor):
    for item in items:
        label, value = item
        if label.split('_')[0] == 'y':
            continue
        x, y, color = get_tensor_index(label)
        initial_tensor[x, y, color] = int(value)

    return initial_tensor


def build_bqm(flow, label_list_x, label_list_y, strength=1):
    x = [Binary(label) for label in label_list_x]
    y = [Binary(label) for label in label_list_y]
    H = 0
    """
    # Constraint: start and end nodes have one neighbour of the same colour
    for p in flow.get_initial_positions(flattened=True):
        nb_coords = flow.get_neighbour_coordinates(p)
        num_nb = len(nb_coords)
        for c in range(flow.get_num_colors):
            H += x[flow.get_x_index(p, c)] * (1 - sum(x[flow.get_x_index(nb, c)] for nb in nb_coords))**2 * (strength)

    # Constraint: all other nodes have two neighbours of the same colour
    for p in flow.get_all_except_initial_positions:
        nb_coords = flow.get_neighbour_coordinates(p)
        num_nb = len(nb_coords)
        for c in range(flow.get_num_colors):
            H += x[flow.get_x_index(p, c)] * (2 - sum(x[flow.get_x_index(nb, c)] for nb in nb_coords))**2 * (strength)
    #model = H.compile()
    print(H)
    """
    for p in flow.get_all_positions:
        num_nb = flow.get_num_helpers[p[0], p[1]]
        nb_coords = flow.get_neighbour_coordinates(p)
        for c in range(flow.get_num_colors):
            H += (sum(n * y[flow.get_y_index(p, c, n)] for n in range(num_nb))
                  - sum(x[flow.get_x_index(nb, c)] for nb in nb_coords))**2

    for p in flow.get_all_except_initial_positions:
        for c in range(flow.get_num_colors):
            H += x[flow.get_x_index(p, c)] * sum((n - 2)**2 * y[flow.get_y_index(p, c, n)] for n in range(num_nb))

    for p in flow.get_initial_positions(flattened=True):
        for c in range(flow.get_num_colors):
            H += x[flow.get_x_index(p, c)] * sum((n - 1)**2 * y[flow.get_y_index(p, c, n)] for n in range(num_nb))

    bqm = H.compile().to_bqm()

    # Constraint: every position can only have one color
    for p in flow.get_all_positions:
        color_labels = [label_list_x[flow.get_x_index(p, c)] for c in range(flow.get_num_colors)]
        print(color_labels)
        color_bqm = combinations(color_labels, 1, strength=3*strength)
        
        print(color_bqm)
        
        bqm.update(color_bqm)
        num_nb = flow.get_num_helpers[p[0], p[1]]
        # Constraint : exactly one helper bit is 1 per position, color
        for c in range(flow.get_num_colors):
            helper_labels = [label_list_y[flow.get_y_index(p, c, n)] for n in range(num_nb)]
            helper_bqm = combinations(helper_labels, 1, strength=2*strength)
            bqm.update(helper_bqm)

    # Constraint: fix known values
    tensor = flow.get_tensor
    print(flow.get_initial_positions())
    for p in flow.get_initial_positions(flattened=True):
        for c in range(flow.get_num_colors):
            print("fix: " + str(get_label(p[0], p[1], c)) + " " + str(tensor[p[0], p[1], c]))
            bqm.fix_variable("x_" + get_label(p[0], p[1], c), tensor[p[0], p[1], c])

    return bqm


def solve_flow(color_pairs, size, num_colors, plot_init=False, plot_error=False, num_reads=1, strength=1):
    flow = fl.Flow(size, num_colors)
    for pair in color_pairs:
        flow.add_color_pair(pair[0], pair[1], pair[2])

    if plot_init:
        flow.plot_flow()

    label_list_x = get_label_list_x(size, num_colors)
    label_list_y = get_label_list_y(flow)
    bqm = build_bqm(flow, label_list_x, label_list_y)
    #print(bqm)
    #sampler = Kerberos()#EmbeddingComposite(DWaveSampler())
    sampler = LeapHybridSampler()
    print("Sending problem...")
    #sample_set = sampler.sample(bqm)#, num_reads=num_reads)
    print("Results from D-Wave:")
    print(sample_set)
    sample_set.to_pandas_dataframe().to_csv("results/flow_experiment_"+str(get_storage_index())+".csv")
    best_solution = sample_set.lowest().first.sample
    for item in best_solution.items():
        print(item)

    # Testing
    """
    tensor = copy.deepcopy(flow.get_tensor)
    tensor[:, 0, 0] = np.ones(5)
    tensor[0:4, 1, 1] = np.ones(4)
    tensor[1:5, 2, 2] = np.ones(4)
    tensor[0:4, 3, 3] = np.ones(4)
    tensor[1:5, 4, 4] = np.ones(4)

    sample = tensor_to_sample(tensor)
    for p in flow.get_initial_positions(flattened=True):
        for c in range(flow.get_num_colors):
            del sample["x_" + get_label(p[0], p[1], c)]
    for p in flow.get_all_positions:
        for c in range(flow.get_num_colors):
            nb_sum = sum(tensor[nb[0], nb[1], c] for nb in flow.get_neighbour_coordinates(p))
            for n in range(flow.get_num_helpers[p[0], p[1]]):
                label = "y_" + get_label(p[0], p[1], c) + "," + str(n)
                if nb_sum == n:
                    sample[label] = 1
                else:
                    sample[label] = 0
    """

    Q, offset = bqm.to_qubo()
    file = open("results/flow_experiment_" + str(get_storage_index()-1) + ".csv", "a")
    file.write("Solution Energy: " + str(dimod.qubo_energy(best_solution, Q, offset)))
    file.close()
    print("Solution Energy: ", dimod.qubo_energy(best_solution, Q, offset))

    solution = sample_to_tensor(best_solution.items(), flow.get_tensor)
    flow.set_solution_tensor(solution)
    if plot_error:
        flow.plot_error(best_solution)

    return flow
