import numpy as np
import matplotlib.pyplot as plt
import os
from os import path


class Flow:
    def __init__(self, size, num_colors):
        self._tensor = np.zeros((size, size, num_colors))
        self._colors = []
        self._initial_positions = []
        self._solved = False
        self._helper_mat = np.zeros((size, size), dtype=np.int)
        for p in self.get_all_positions:
            self._helper_mat[p[0], p[1]] = min(3, len(self.get_neighbour_coordinates(p))) + 1
        self._solution_sample = {}

    @property
    def get_size(self):
        return self._tensor.shape[0]

    @property
    def get_num_colors(self):
        return self._tensor.shape[2]

    @property
    def get_colors(self):
        return self._colors

    def get_initial_positions(self, flattened=False):
        if flattened:
            positions = []
            for pair in self._initial_positions:
                for pos in pair:
                    positions.append(pos)
        else:
            positions = self._initial_positions
        return positions

    @property
    def get_all_positions(self):
        positions = []
        for i in range(self.get_size):
            for j in range(self.get_size):
                positions.append((i, j))
        return positions

    @property
    def get_all_except_initial_positions(self):
        positions = self.get_all_positions
        for pos in self.get_initial_positions(True):
            positions.remove(pos)
        return positions

    @property
    def get_tensor(self):
        return self._tensor

    @property
    def is_solved(self):
        return self._solved

    @property
    def get_num_helpers(self):
        return self._helper_mat

    def get_x_index(self, pos, color):
        return color + self.get_num_colors * pos[1] + self.get_num_colors * self.get_size * pos[0]

    def get_z_index(self, edge, color):
        # Vertical?
        if edge[0]:
            return self.get_num_colors * self.get_size * (self.get_size - 1) + color + self.get_num_colors * edge[2] + self.get_num_colors * self.get_size * edge[1]
        else:
            return color + self.get_num_colors * edge[2] + self.get_num_colors * (self.get_size - 1) * edge[1]

    # add two points of a color, needed for initialization of flow problem tensor,
    # pos1 and pos2 are arrays with x and y coordinate each, color must be a string
    # that python recognizes as a color, e.g. "red"
    def add_color_pair(self, color, pos1, pos2):
        # get index of the new color in the matrix
        color_index = len(self.get_colors)
        self._colors.append(color)
        self._initial_positions.append([pos1, pos2])

        print("pos1=" + str(pos1) + " pos2=" + str(pos2) + " color_index=" + str(color_index))

        self._tensor[pos1][color_index] = 1
        self._tensor[pos2][color_index] = 1

    # tells the Flow object that it has been solved, tensor contains the solution information
    def set_solution_tensor(self, tensor):
        self._tensor = tensor
        self._solved = True

    # takes coordinates as an input and returns an array containing the coordinates
    # of all neighbouring cells
    def get_neighbour_coordinates(self, pos):
        neighbours = []
        for i in range(2):
            if pos[i] > 0:
                neighbours.append(np.array(pos) - np.array([(i + 1) % 2, i % 2]))
            if pos[i] < self.get_size - 1:
                neighbours.append(np.array(pos) + np.array([(i + 1) % 2, i % 2]))
        return np.array(neighbours)

    def get_edges(self, pos):
        edges = []
        for i in range(2):
            if pos[i] > 0:
                edges.append(np.array([1 - i, pos[0], pos[1]]) - np.array([0, (i + 1) % 2, i % 2]))
            if pos[i] < self.get_size - 1:
                edges.append(np.array([1 - i, pos[0], pos[1]]) + np.array([0, 0, 0]))
        print("pos=" + str(pos) + " - neigh=" + str(edges))
        return np.array(edges)

    def get_minus(self, edge):
        # Vertical?
        if edge[0]:
            return np.array([edge[1], edge[2]])
        else:
            return np.array([edge[1], edge[2]])
        
    def get_plus(self, edge):
        # Vertical?
        if edge[0]:
            return np.array([edge[1] + 1, edge[2]])
        else:
            return np.array([edge[1], edge[2] + 1])

    # takes a start position and color as an input and draws the flow from there, always in
    # the direction of the neighbouring field that has a 1 value of the same color. It will
    # return an array called completion, where 1 indicates an error and 0 indicates success.
    # The first number stands for the overall completion, the second stands for whether the
    # tensor provides a unique way to draw the flow and the third indicates, if the flow was
    # interrupted. Note that even though the endposition is optional, one should at least
    # hand it over in the first try of drawing the flow, since otherwise, the flow will be
    # marked as incomplete.
    def draw_flow(self, start_position, color_idx, end_position=None):
        completion = [1, 0, 0]
        current_position = start_position
        last_position = np.array(start_position)
        while completion == [1, 0, 0]:
            next_position = np.array(current_position)
            edge_count = 0
            # look up, which neighbour has the same colour (without going back)
            for nb in self.get_neighbour_coordinates(current_position):
                if self.get_tensor[nb[0], nb[1], color_idx] == 1 and not (nb == last_position).all():
                    edge_count += 1
                    next_position = np.array(nb)

            # flow not optimal (too many neighbours)
            if edge_count > 1:
                completion[1] = 1

            # problem not complete
            if (next_position == current_position).all():
                completion[2] = 1

            plt.plot([current_position[1], next_position[1]], [current_position[0], next_position[0]],
                     c=self.get_colors[color_idx], linewidth=15)
            last_position = np.array(current_position)
            current_position = np.array(next_position)

            if (current_position == end_position).all():
                completion[0] = 0
        return np.array(completion)

    # draws a plot of the flow, the initial points and if the flow is marked as solved, it will also draw
    # the connections. Note that plt.show() has to be called after calling this function
    def plot_flow(self):
        N = self.get_size
        fig = plt.figure(figsize=[5, 5])
        ax = plt.gca()

        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)

        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)

        ax.set_xticks(np.arange(-0.5, N - 0.5, 1))
        ax.set_yticks(np.arange(-0.5, N - 0.5, 1))
        ax.set_xlim([-0.5, N - 0.5])
        ax.set_ylim([-0.5, N - 0.5])
        ax.set_facecolor((0, 0, 0))
        ax.invert_yaxis()
        for i in range(len(self.get_colors)):
            points = np.array(self.get_initial_positions()[i])
            x = points[:, 1]
            y = points[:, 0]
            plt.scatter(x, y, s=500, c=self.get_colors[i])

        plt.grid()

        if self.is_solved:
            for i in range(self.get_num_colors):
                if not (self.draw_flow(self.get_initial_positions()[i][0], i,
                                       self.get_initial_positions()[i][1]) == np.array([0, 0, 0])).all():
                    print(self.draw_flow(self.get_initial_positions()[i][1], i))
        plt.draw()
        plt.savefig("results/flow_plot_" + str(self.get_file_count()) + ".png")

    def get_file_count(self):
        file_count = 1
        if path.isdir("results"):
            while path.isfile("results/flow_experiment_" + str(file_count) + ".csv"):
                file_count += 1
        else:
            try:
                os.mkdir("results")
            except OSError:
                print("Creation of the directory 'results' failed")
            else:
                print("Successfully created the directory 'results'")
        return file_count - 1

    def save_solution(self):
        file_count = self.get_file_count()
        file = open("results/flow_matrix_" + str(file_count) + ".csv", "a")
        for c in range(self.get_num_colors):
            file.write(self.get_colors[c] + "\n")
            for i in range(self.get_size):
                for j in range(self.get_size):
                    file.write(str(self.get_tensor[i, j, c]) + ";")
                file.write("\n")
        file.close()

    def plot_error(self, solution_sample):
        N = self.get_size
        fig = plt.figure(figsize=[5, 5])
        ax = plt.gca()

        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)

        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)

        ax.set_xticks(np.arange(-0.5, N - 0.5, 1))
        ax.set_yticks(np.arange(-0.5, N - 0.5, 1))
        ax.set_xlim([-0.5, N - 0.5])
        ax.set_ylim([-0.5, N - 0.5])
        ax.invert_yaxis()

        colors = ["blue", "green", "yellow", "orange", "red"]

        for p in self.get_all_positions:
            color_error = int(sum(self.get_tensor[p[0], p[1], c] for c in range(self.get_num_colors)))
            if color_error > 4:
                color_error = 4
            plt.scatter(p[1], p[0], marker="s", s=500, c=colors[color_error])

        for p in self.get_all_positions:
            neighbour_error = 0
            num_nb = self.get_num_helpers[p[0], p[1]]
            nb_coords = self.get_neighbour_coordinates(p)
            for c in range(self.get_num_colors):
                neighbour_error += abs(sum(
                    n * solution_sample["y_" + str(p[0]) + "," + str(p[1]) + "_" + str(c) + "," + str(n)] for n in
                    range(num_nb))
                                       - sum(self.get_tensor[nb[0], nb[1], c] for nb in nb_coords))
            if neighbour_error > 3:
                neighbour_error = 3
            neighbour_error = int(neighbour_error)
            plt.scatter(p[1], p[0], marker="s", s=500, c=colors[neighbour_error + 1])

        error_matrix = np.zeros((self.get_size, self.get_size))

        for p in self.get_all_except_initial_positions:
            neighbour_error = 0
            num_nb = self.get_num_helpers[p[0], p[1]]
            for c in range(self.get_num_colors):
                neighbour_error += self.get_tensor[p[0], p[1], c] * sum(abs(n - 2) * solution_sample["y_" + str(p[0]) + "," + str(p[1]) + "_" + str(c) + "," + str(n)] for n in range(num_nb))
            error_matrix[p[0], p[1]] += neighbour_error

        for p in self.get_initial_positions(flattened=True):
            neighbour_error = 0
            num_nb = self.get_num_helpers[p[0], p[1]]
            for c in range(self.get_num_colors):
                neighbour_error += self.get_tensor[p[0], p[1], c] * sum(abs(n - 1) * solution_sample["y_" + str(p[0]) + "," + str(p[1]) + "_" + str(c) + "," + str(n)] for n in range(num_nb))
            error_matrix[p[0], p[1]] += neighbour_error

        for p in self.get_all_positions:
            neighbour_error = int(min(error_matrix[p[0], p[1]], 3))
            plt.scatter(p[1], p[0], marker="s", s=600, c=colors[neighbour_error + 1])

        plt.grid()
        plt.savefig("results/error_plot_" + str(self.get_file_count()) + ".png")

    # wrong atm
    def compute_total_error(self):
        total_error = 0
        for p in self.get_all_positions:
            total_error = (1 - sum(self.get_tensor[p[0], p[1], c] for c in range(self.get_num_colors))) ** 2

        for p in self.get_initial_positions(flattened=True):
            neighbour_error = 0
            for c in range(self.get_num_colors):
                error = (self.get_tensor[p[0], p[1], c]
                         - sum(self.get_tensor[nb[0], nb[1], c]
                               for nb in self.get_neighbour_coordinates(p))) ** 2
                if self.get_tensor[p[0], p[1], c] > 0:
                    neighbour_error += error
            total_error += neighbour_error

        for p in self.get_all_except_initial_positions:
            neighbour_error = 0
            for c in range(self.get_num_colors):
                neighbour_error += (2 * self.get_tensor[p[0], p[1], c]
                                    - sum(self.get_tensor[nb[0], nb[1], c]
                                          for nb in self.get_neighbour_coordinates(p))) ** 2
            total_error += neighbour_error

        return total_error
