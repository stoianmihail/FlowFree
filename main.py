import sys
import json
from helpers import helperFunctions as hf
from helpers import Flow as fl

def parse_flow(flow):
    newFlow = []
    for (color, p1, p2) in flow:
        newFlow.append([color, (p1[0], p1[1]), (p2[0], p2[1])])
    return newFlow

def main(file):
    with open(file) as f:
        data = json.load(f)
        size = data["size"]
        num_colors = data["num_colors"]
        example_flow = parse_flow(data["flow"])
        
        if True:
            print(size)
            print(num_colors)
            print(example_flow)
        
        solved_flow = hf.solve_flow(example_flow, size, num_colors, num_reads=50, strength=1, plot_error=True, plot_init=True)
        solved_flow.plot_flow()
        for c in range(solved_flow.get_num_colors):
            print(solved_flow.get_colors[c] + "\n")
            for i in range(solved_flow.get_size):
                out = ""
                for j in range(solved_flow.get_size):
                    out += str(solved_flow.get_tensor[i, j, c]) + " "
                print(out)
        solved_flow.save_solution()
        fl.plt.show()
    pass

if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        print("Usage: python " + sys.argv[0] + " <file>")
        sys.exit(-1)
    else:
        main(sys.argv[1])
