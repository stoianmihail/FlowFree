from helpers import helperFunctions as hf
from helpers import Flow as fl


def main():
    size = 6
    num_colors = 6
    #example_flow = [["blue", (0, 0), (3, 0)], ["red", (0, 1), (3, 3)], ["orange", (1, 1), (3, 2)], ["green", (0, 3), (1, 3)]]
    
    #example_flow = [["red", (2, 1), (2, 4)], ["green", (1, 1), (1, 4)], ["blue", (0, 0), (4, 4)], ["yellow", (1, 3), (2, 3)], ["orange", (1, 2), (2, 2)]]
    example_flow = [["red", (0, 0), (4, 1)], ["green", (0, 2), (3, 1)], ["blue", (1, 2), (4, 2)], ["yellow", (0, 4), (3, 3)], ["orange", (1, 4), (4, 3)]]
    #example_flow = [["red", (0, 0), (2, 0)], ["green", (0, 1), (1, 2)], ["blue", (1, 1), (2, 2)]]
    
    example_flow = [["orange", (0, 0), (5, 1)], ["blue", (0, 5), (2, 1)], ["green", (3, 1), (4, 5)], ["yellow", (2, 2), (3, 5)], ["grey", (1, 4), (2, 5)], ["red", (4, 1), (5, 5)]]
    
    solved_flow = hf.solve_flow(example_flow, size, num_colors, num_reads=10, strength=1, plot_error=True, plot_init=True)
    solved_flow.plot_flow()
    #print("Computed error: " + str(solved_flow.compute_total_error()))
    for c in range(solved_flow.get_num_colors):
        print(solved_flow.get_colors[c] + "\n")
        for i in range(solved_flow.get_size):
            out = ""
            for j in range(solved_flow.get_size):
                out += str(solved_flow.get_tensor[i, j, c]) + " "
            print(out)
    solved_flow.save_solution()

    fl.plt.show()


if __name__ == "__main__":
    main()
