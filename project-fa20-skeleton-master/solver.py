import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_happiness, calculate_stress_for_room, calculate_happiness_for_room
import sys
from os.path import basename, normpath
import glob
from pulp import *
import itertools

    
def solve(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    """
    num_students = G.number_of_nodes()
    mappings = {}
    total_happiness = {}
    mapping = solve_each_room_number(10, G, s)
#    for i in range(1, num_students + 1):
#        mapping = solve_each_room_number(i, G, s)
#        mappings[i] = mapping
#        total_happiness[i] = calculate_happiness(mapping, G)
#        
#    key_max = max(total_happiness.keys(), key=(lambda k: total_happiness[k]))    
    return mappings[key_max], key_max


def solve_each_room_number(n, G, s):
    """
    Args:
        n:number of breakout rooms
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
    """
    num_students = G.number_of_nodes()
    room_number_range = range(0, n)
    
    pairs = G.edges
    prob = LpProblem("optimize", LpMaximize)
    variables = []
    for e in pairs:
        for x in room_number_range:
            variables += [e + (x,)]
    threshold = s/n
    
    #variables
    all_variables = LpVariable.dicts("all variables", variables , lowBound=0, upBound=1, cat="Integer")
    
    #objective
    prob += lpSum(pairs[i[:2]]["happiness"]*all_variables[i] for i in variables)
    #constraints
    for e in variables:
        #one person cannot appear in multiple rooms
        
            
        room_num = [i for i in range(0,e[2])] + [i for i in range(e[2] + 1, n)]
        if e == (1,9,1):
            print("here:", room_num)
        for i in range(0, num_students):
            if i < e[0]:
                prob += lpSum(all_variables[(i,e[0],j)] for j in room_num) + lpSum(all_variables[e]) <= 1
            if i > e[0] and i != e[1]:
                prob += lpSum(all_variables[(e[0],i,j)] for j in room_num) + lpSum(all_variables[e]) <= 1
            if i < e[1] and i != e[0]:
                prob += lpSum(all_variables[(i,e[1],j)] for j in room_num) + lpSum(all_variables[e]) <= 1
            if i > e[1]:
                prob += lpSum(all_variables[(e[1],i,j)] for j in room_num) + lpSum(all_variables[e]) <= 1
        #print("this stress:", all_variables[e + (0,)]*pairs[e]["stress"])
    #each room threshold
    for j in range(0, n):
        prob += lpSum(all_variables[e + (j,)]*pairs[e]["stress"] for e in pairs) <= threshold
    prob.solve()
    
    print(s, "threshold:", threshold)
    for v in prob.variables():
        print(v.name, "=", v.varValue)
# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G, s = read_input_file(path)
    D, k = solve(G, s)
    assert is_valid_solution(D, G, s, k)
    #D = {7:0, 2:0, 1:0, 5:1, 0:1, 9:1, 8:2, 6:2, 3:2, 4:2}
    #print("Pairing:", D)
    print("Total Happiness: {}".format(calculate_happiness(D, G)))
    print("Total Stress: {}".format(calculate_stress(D, G)))
    write_output_file(D, 'samples/10.out')
    #read_output_file('samples/10.out', G, s)


## For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
#if __name__ == '__main__':
#    inputs = glob.glob('inputs/*')
#    for input_path in inputs:
#        output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#        G, s = read_input_file(input_path)
#        D, k = solve(G, s)
#        assert is_valid_solution(D, G, s, k)
#        happiness = calculate_happiness(D, G)
#        write_output_file(D, output_path)
