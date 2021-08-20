from __future__ import print_function
from ortools.algorithms import pywrapknapsack_solver
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_solution, calculate_happiness, calculate_stress, calculate_stress_for_room, calculate_happiness_for_room, convert_dictionary
import networkx as nx
import sys
import math 
import glob
import os
from collections import defaultdict
import time
import random

def get_subOpt(G, s):
   
        D_greedy, k_greedy = solve_greedy(G,s)
       
        #return whichever is better out of greedy and dp
        if k_greedy < 1 or not is_valid_solution(D_greedy, G, s, k_greedy):
            D_dp, k_dp = solve_dp(G, s)
            if is_valid_solution(D_dp, G, s, k_dp):
                # print("=====dp1======")
                return D_dp, k_dp
            else:
                # print("===basic1====")
                D_basic, k_basic = basic(G, s)
                return D_basic, k_basic
        else:
            D_dp, k_dp = solve_dp(G, s)
            if is_valid_solution(D_dp, G, s, k_dp):
                happiness_dp = calculate_happiness(D_dp, G)
                happiness_greedy = calculate_happiness(D_greedy, G)
                if happiness_dp > happiness_greedy:
            # print("=====dp2======")
                    return happiness_dp
        # print("=====greedy1======")
                return happiness_greedy
            else:
                return D_greedy, k_greedy
        

#========CODE FROM https://stackoverflow.com/questions/18353280/iterator-over-all-partitions-into-k-groups
def clusters(l, K, thresh):
    if l:
        prev = None
        for t in clusters(l[1:], K, thresh):
            tup = sorted(t)
            if tup != prev:
                prev = tup
                for i in range(K):
                    temp = tup[:i] + [[l[0]] + tup[i],] + tup[i+1:]
                    over_limit, i = False, 0
                    while i != len(temp) and not over_limit: 
                        if calculate_stress_for_room(temp[i], G) > thresh: 
                            over_limit = True
                    if not over_limit:
                        yield temp
                    else:
                        yield [[] for _ in range(K)]
                    # j = random.randint(0, len(temp) - 1)
                    # heuristic_room = temp[j]
                    # if calculate_stress_for_room(heuristic_room, G) <= thresh:
                    #     yield temp
    else:
        yield [[] for _ in range(K)]
def neclusters(l, K, G, s):
    for c in clusters(l, K, s/K):
        if all(x for x in c): 
            if calculate_happiness(c, G) > 1.3 * get_subOpt(G, s):
                return c
#END CODE FROM==================


#=======code from stack overflow to get all possible permutations====== 
def sorted_k_partitions(seq, k):
    """Returns a list of all unique k-partitions of `seq`.
    Each partition is a list of parts, and each part is a tuple.
    The parts in each individual partition will be sorted in shortlex
    order (i.e., by length first, then lexicographically).
    The overall list of partitions will then be sorted by the length
    of their first part, the length of their second part, ...,
    the length of their last part, and then lexicographically.
    """
    n = len(seq)
    groups = []  # a list of lists, currently empty
    def generate_partitions(i):
        # now = time.perf_counter()
        # if (now - start) > 1000:
        #     i = n
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > k - len(groups):
                for group in groups:
                    group.append(seq[i])
                    yield from generate_partitions(i + 1)
                    group.pop()

            if n - i == k - len(groups):
                for person in range(n-i):
                    groups.append([seq[person]])
                    yield from generate_partitions(i + 1)
                    groups.pop()

            if len(groups) < k:
                groups.append([seq[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)
    # Sort the parts in each partition in shortlex order
    result = [sorted(ps, key = lambda p: (len(p), p)) for ps in result]
    # Sort partitions by the length of each part, then lexicographically.
    result = sorted(result, key = lambda ps: (*map(len, ps), ps))
    return result
#========end code from stack overflow==========

#global alpha
#checks if this person can go into any of the previous rooms 
#return room if better exists, else return -1
def fit_in_other_room(G, student, room_to_student_dict, curr_room, student_to_room):
    #list of all room numbers where student can validly be added
    all_valid = []
    for room in range(curr_room):
        temp = room_to_student_dict[room].copy()
        temp.append(student)
        curr_load = calculate_stress_for_room(temp, G)
        happiness = calculate_happiness_for_room(temp, G)
        # temp_D = student_to_room.copy()
        # temp_D[student] = room
        # total_stress = calculate_stress(temp_D, G)
        if curr_load <= (s/(curr_room + 1))*alpha: #and total_stress < s:
            all_valid.append((room, happiness))
    if len(all_valid) == 0:
        return -1
    return max(all_valid, key=lambda t:t[1])[0]

#checks if these two people can go into any of the previous rooms 
#return room if better exists, else return -1
def fits_in_other_room(G, students, room_to_student_dict, curr_room, student_to_room):
    #list of all room numbers where student can validly be added
    all_valid = []
    for room in range(curr_room):
        temp = room_to_student_dict[room].copy()
        temp.append(students[0])
        temp.append(students[1])
        curr_load = calculate_stress_for_room(temp, G)
        happiness = calculate_happiness_for_room(temp, G)
        # temp_D = student_to_room.copy()
        # temp_D[students[0]] = room
        # temp_D[students[1]] = room
        #total_stress = calculate_stress(temp_D, G)
        if curr_load <= (s/(curr_room +1))*alpha: #and total_stress < s:
            all_valid.append((room, happiness))
    if len(all_valid) == 0:
        return -1
    return max(all_valid, key=lambda t:t[1])[0]

#add a check to see if any of the existing rooms exceed smax/k where k is number of rooms 
#if so, remove the person contributing most stress/happiness and add them to another exisitng list
#if possible (helper func) otherwise put them in new room  
def check_exceed(room_to_students, G, threshold, student_to_room):
    for room in room_to_students.keys():
        room_stress = calculate_stress_for_room(room_to_students[room], G)
        if room_stress > threshold:
            worst_person = room_to_students[room][0]
            worst_stress = math.inf#100000000000000
            for person in room_to_students[room]:
                temp = room_to_students[room].copy()
                temp.remove(person)
                new_val = calculate_stress_for_room(temp, G)
                if new_val < worst_stress:
                    worst_stress = new_val
                    worst_person = person
            #try to move worst_person to another exisiting room if possible
            new_room = fit_in_other_room(G, worst_person, room_to_students, room, student_to_room)
            if new_room == -1:
                #student_to_move = worst_person
                return (room, worst_person, new_room) 
            else:
                #student_to_move = worst_person
                return (room, worst_person, new_room)
    return None

def check_exceed2(room_to_students, G, threshold, student_to_room):
    for room in room_to_students.keys():
        room_stress = calculate_stress_for_room(room_to_students[room], G)
        if room_stress > threshold:
            worst_person = room_to_students[room][0]
            worst_stress = -math.inf#100000000000000
            for person in room_to_students[room]:
                temp = room_to_students[room].copy()
                temp.remove(person)
                new_val = calculate_stress_for_room(temp, G)
                if new_val > worst_stress:
                    worst_stress = new_val
                    worst_person = person
            #try to move worst_person to another exisiting room if possible
            new_room = fit_in_other_room(G, worst_person, room_to_students, room, student_to_room)
            if new_room == -1:
                #student_to_move = worst_person
                return (room, worst_person, new_room) 
            else:
                #student_to_move = worst_person
                return (room, worst_person, new_room)
    return None

def dp(G, s, num_rooms):

    #create an easy to access dictionary for happy or stress of any pair
    edgeview = G.edges
    pair_to_vals = {}
    for pair in edgeview:
        if pair[1] < pair[0]:
            new_pair = (pair[1], pair[0])
            pair = new_pair
        happy = edgeview[pair]['happiness']
        stress = edgeview[pair]['stress']
        pair_to_vals[pair] = (happy, stress)

    threshold_per_room = s/num_rooms
    all_students = [i for i in range(G.number_of_nodes())]

    student_to_room = {}
    room_to_student = {}
    total_happiness = 0

    for r in range(num_rooms):
        #establish variables 
        total_weight_limit = threshold_per_room
        room = []

        weight_index_to_pair = {}
        value_index_to_pair = {}
        weights = []
        values = []
        index = 0
        for i in all_students:
            for j in all_students:
                if i<j:
                    pair = pair_to_vals[(i,j)]
                    weights.append(pair[1])
                    weight_index_to_pair[index] = (i,j) 
                    values.append(pair[0])
                    value_index_to_pair[index] = (i,j) 
                    index += 1

        #add highest happy pairs to this room while maintaing stress below threshold  
        total_stress_in_room = calculate_stress_for_room(room, G) 
        while total_stress_in_room < threshold_per_room and len(values) > 0:
            #test all people and choose the one who contributes the most happiness 
            #to this room while maintaining that we are below the stress threshold 
            happiness_brought_by_person = {}
            temp_room = room.copy()
            for i in all_students:
                temp_room.append(i)
                hap = calculate_happiness_for_room(temp_room, G)
                stressy = calculate_stress_for_room(temp_room, G)
                if stressy <= threshold_per_room:
                    happiness_brought_by_person[i] = hap
                temp_room.remove(i)

            if len(happiness_brought_by_person.keys()) == 0:
                total_stress_in_room = threshold_per_room
            else:
                if len(room) == 0:
                    #add the person who has min stress value 
                    min_stress = 101
                    min_pair = None
                    for i in all_students:
                        for j in all_students:
                            if i < j:
                                curr_stress = pair_to_vals[(i,j)][1]
                                if curr_stress < min_stress:
                                    min_stress = curr_stress
                                    min_pair = (i,j)
                                elif curr_stress == min_stress:
                                    curr_hap = pair_to_vals[(i,j)][0]
                                    mins_hap = pair_to_vals[(min_pair[0],min_pair[1])][0]
                                    if curr_hap > mins_hap:
                                        min_pair = (i,j)
                    tempy = room.copy()
                    tempy.append(min_pair[0])
                    tempy.append(min_pair[1])
                    stresi = calculate_stress_for_room(tempy, G)
                    if stresi > threshold_per_room:
                        room.append(min_pair[0])
                        all_students.remove(min_pair[0])
                    else:
                        room.append(min_pair[0])
                        room.append(min_pair[1])
                        all_students.remove(min_pair[0])
                        all_students.remove(min_pair[1])
                else:
                    best_person = max(happiness_brought_by_person.keys(), key=lambda x:happiness_brought_by_person[x])
                    room.append(best_person)
                    all_students.remove(best_person)

                # max_happy = values.index(max(valids))
                # max_pair = value_index_to_pair[max_happy]

                # temp1 = room.copy()
                # temp2 = room.copy()
                # temp3 = room.copy()
             
                # #add both to room
                # temp1.append(max_pair[0])
                # temp1.append(max_pair[1])
                # add_both_s = calculate_stress_for_room(temp1, G)
                # add_both_h = calculate_happiness_for_room(temp1, G) 
                # #add first to room
                # temp2.append(max_pair[0])
                # add_first_s = calculate_stress_for_room(temp2, G)
                # add_first_h = calculate_happiness_for_room(temp2, G) 
                # #add second to room
                # temp3.append(max_pair[1])
                # add_second_s = calculate_stress_for_room(temp3, G)
                # add_second_h = calculate_happiness_for_room(temp3, G) 

                # all_valid = {}

                # if total_stress_in_room + add_both_s < threshold_per_room:
                #     all_valid[add_both_h] = [max_pair[0], max_pair[1]]
                # if total_stress_in_room + add_first_s < threshold_per_room:
                #     all_valid[add_first_h] = [max_pair[0]]
                # if total_stress_in_room + add_second_s < threshold_per_room:
                #     all_valid[add_second_h] = [max_pair[1]]

                # if len(all_valid) == 0:
                #     break
                #     #try a different pair
                # else:
                #     best = max(all_valid.keys())
                #     to_add = all_valid[best]
                #     for i in to_add:
                #         room.append(i)
                #         all_students.remove(i)
                #         #remove the value not the person rn we want to swtitch it 
                #         #values.remove(i)
                total_stress_in_room = calculate_stress_for_room(room, G)

            weight_index_to_pair = {}
            value_index_to_pair = {}
            weights = []
            values = []
            index = 0
            for i in all_students:
                for j in all_students:
                    if i<j:
                        pair = pair_to_vals[(i,j)]
                        #stress
                        weights.append(pair[1])
                        weight_index_to_pair[index] = (i,j)
                        #happiness 
                        values.append(pair[0])
                        value_index_to_pair[index] = (i,j) 
                        index += 1
        if len(room) == 0:
            #ADDED FOR SOLVING HARD INPUTS
            if len(all_students) == 1:
                #add this student to a new room an return this sol 
                #where num_rooms=(r+1)
                student = all_students[0]
                room_to_student[r] = student
                student_to_room[student] = r
                return total_happiness, student_to_room, r
            if len(all_students) == 0:
                return total_happiness, student_to_room, r - 1
            #END OF ADDED FOR SOLVING HARD INPUTS
            #print("Here1")
            # return -1, {}, num_rooms
            #ADDED FOR HARD INPUTS
            #add person who provides lowest happiness to room
            total_hap = {}
            for i in all_students:
                total_hap[i] = 0
                for j in all_students:
                    if i != j:
                        if i<j:
                            pair = (i,j)
                        else:
                            pair = (j,i)
                        total_hap[i] += pair_to_vals[pair][0]
            if len(total_hap.keys()) == 0:
                return -1, {}, num_rooms
            best_singleton = min(total_hap.keys(), key=lambda x:total_hap[x])
            room.append(best_singleton)
            all_students.remove(best_singleton)
            #END ADDED FOR HARD INPUTS
        room_to_student[r] = room
        for person in room:
            student_to_room[person] = r
        total_happiness += calculate_happiness_for_room(room, G)
    if  G.number_of_nodes() - len(student_to_room.keys()) >= 1:
        new_room = num_rooms 
        for student in all_students:
            if student not in student_to_room.keys():
                student_to_room[student] = new_room
                new_room += 1
        num_rooms = new_room 
        #go through every room and calcualte new stress per room 
        #if room is above threshold, figure out how to remove
        threshold = s/(num_rooms)
        test = check_exceed(room_to_student, G, threshold, student_to_room)
        while test is not None:
            old_room = test[0]
            student_to_move = test[1]
            the_room = test[2]
            if the_room == -1:
                num_rooms += 1
                student_to_room[student_to_move] = num_rooms - 1
                room_to_student[num_rooms - 1] = [student_to_move]
                room_to_student[old_room].remove(student_to_move)
            else:
                student_to_room[student_to_move] = the_room
                room_to_student[the_room].append(student_to_move)
                room_to_student[old_room].remove(student_to_move)
            threshold = s/(num_rooms)
            test = check_exceed(room_to_student, G, threshold, student_to_room)
    return total_happiness, student_to_room, num_rooms 


    #old dp approach w knapsack that didnt work
    # #each student can either be added to a room or can not be added to the room
    # #we want to add the student that maximizes the happiness of the room while 
    # #maintaining that the total stress levels in the room do not exceed threshold
    # #use knapsack 
    # student_to_room = {}
    # total_happiness = 0
    # for r in range(num_rooms):
    #     #establish variables 
    #     total_weight_limit = [threshold_per_room]
    #     weight_index_to_pair = {}
    #     value_index_to_pair = {}
    #     weights = []
    #     values = []
    #     index = 0
    #     for i in all_students:
    #         for j in all_students:
    #             if i<j:
    #                 pair = pair_to_vals[(i,j)]
    #                 weights.append(pair[1])
    #                 weight_index_to_pair[index] = (i,j) 
    #                 values.append(pair[0])
    #                 value_index_to_pair[index] = (i,j) 
    #                 index += 1
    #     weights =[weights]
    #     #create the solver
    #     solver = pywrapknapsack_solver.KnapsackSolver(
    #     pywrapknapsack_solver.KnapsackSolver.
    #     KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    #     #call the solver to get the solution
    #     solver.Init(values, weights, total_weight_limit)
    #     computed_value = solver.Solve()
    #     packed_items = []
    #     packed_weights = []
    #     total_weight = 0
    #     for i in range(len(values)): #values.keys():
    #         if solver.BestSolutionContains(i):
    #             packed_items.append(value_index_to_pair[i])

    #             #add the happiness of this pair to total happiness
    #             total_happiness += values[i]

    #             packed_weights.append(weights[0][i])
    #             total_weight += weights[0][i]
    #     #add the students assigned to this room into this room
    #     for i in packed_items:
    #         student_to_room[i[0]] = r
    #         student_to_room[i[1]] = r
    #         #remove these students from all_students (students have not yet been assigned a breakout room)
    #         if i[0] in all_students:
    #             all_students.remove(i[0])
    #         if i[1] in all_students:
    #             all_students.remove(i[1])
        # print("-----TIME-------")
        # print(packed_items)
        # print(all_students)
        #print("IN LOOP:", student_to_room)
    #print("DONE:", student_to_room)
    #return total_happiness, student_to_room

def solve(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    """
    # TODO: your code here!
    #print("smax: ", s)

    #hard coded -- solving specific hard inputs
    #return tempe(G, s)
 
    return brute_force(G, s)
    #solve using both greedy and dp
    D_dp, k_dp = solve_dp(G, s)
    D_greedy, k_greedy = solve_greedy(G,s)
    D_basic, k_basic = basic(G, s)
    #return whichever is better out of greedy and dp
    if k_greedy < 1 or not is_valid_solution(D_greedy, G, s, k_greedy):
        if is_valid_solution(D_dp, G, s, k_dp):
            print("=====dp1======")
            return D_dp, k_dp
        else:
            print("===basic1====")
            return D_basic, k_basic
    if not is_valid_solution(D_dp, G, s, k_dp):
        if is_valid_solution(D_greedy, G, s, k_greedy):
            print("======greedy2======")
            return D_greedy, k_greedy
        else:
            print("===basic2===")
            return D_basic, k_basic
    happiness_dp = calculate_happiness(D_dp, G)
    happiness_greedy = calculate_happiness(D_greedy, G)
    if happiness_dp > happiness_greedy:
        print("=====dp2======")
        return D_dp, k_dp
    print("=====greedy1======")
    return D_greedy, k_greedy

# global all_poss_20
# all_poss_20 = []
#solver for small inputs 
def brute_force(G, s):
    #global all_poss_20
    #try every possible arrangement of putting students into rooms and 
    #keep the one that results in highest happiness value
    #all_poss_arrs = []
    #if len(all_poss_20) == 0:
    #for n = 10
    # seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for k in range(1, 10):
    #for n = 20
    seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    #start = time.perf_counter()
    for k in range(3, 4):
    #k = 2
    #for n = 50
    #seq = [i for i in range(50)]
    #k = 49
        #lst = sorted_k_partitions(seq, k, start)
        lst_iter = neclusters(seq, k, G, s)
        # #print("Here")
        #all_poss_arrs.extend(lst)


        # n = len(seq)
        # groups = []  # a list of lists, currently empty
        # def generate_partitions(i):
        #     if i >= n:
        #         yield list(map(tuple, groups))
        #     else:
        #         if n - i > k - len(groups):
        #             for group in groups:
        #                 group.append(seq[i])
        #                 yield from generate_partitions(i + 1)
        #                 group.pop()

        #         if len(groups) < k:
        #             groups.append([seq[i]])
        #             yield from generate_partitions(i + 1)
        #             groups.pop()
        # result = generate_partitions(0)

    # stop = time.perf_counter()
    # print("time_taken: ", )
    #print("here")
        #all_poss_20 = all_poss_arrs
    # else:
    #     all_poss_arrs = all_poss_20

    all_happs = {}
    while True:
        try:
            arrangement = next(lst_iter)
            num_rooms = len(arrangement)
            thresh = s/num_rooms
            index = random.randint(0, num_rooms-1)
#            if calculate_stress_for_room([i for i in arrangement[index]], G) > thresh:
#                allofem.remove(arrangement)
#                break
            total_hap = 0
            ast = {}
            j = 0
            dont = False
            for room in arrangement:
                peeps = [i for i in room]
                stressy = calculate_stress_for_room(peeps, G)
                if stressy <= thresh:
                    hap = calculate_happiness_for_room(peeps, G)
                    total_hap += hap
                    ast[j] = peeps
                    j += 1
                else:
                    dont = True
                    break
            if not dont:
#                allofem.remove(arrangement)
                all_happs[total_hap] = ast
        except StopIteration:
            break
    # all_happs = {}
    # allofem = all_poss_arrs.copy()
    # while len(allofem) > 0:#for i in range(len(allofem)):#arrangement in allofem:
    #     arrangement = allofem[0]
    #     num_rooms = len(arrangement)
    #     thresh = s/num_rooms
    #     index = random.randint(0, num_rooms-1)
    #     if calculate_stress_for_room([i for i in arrangement[index]], G) > thresh:
    #         allofem.remove(arrangement)
    #         break
    #     total_hap = 0
    #     ast = {}
    #     j = 0
    #     dont = False
    #     for room in arrangement:
    #         peeps = [i for i in room]
    #         stressy = calculate_stress_for_room(peeps, G)
    #         if stressy <= thresh:
    #             hap = calculate_happiness_for_room(peeps, G)
    #             total_hap += hap
    #             ast[j] = peeps
    #             j += 1
    #         else:
    #             #if a room w these people together exists in another arrangement, 
    #             #prune this arrangement
    #             all_remove = []
    #             no = False
    #             for ar in allofem:
    #                 if len(ar) >= num_rooms:
    #                     for r in ar:
    #                         peeps2 = [i for i in r]
    #                         #if all the people in peeps are also in peeps2, 
    #                         #add this ar to all_remove
    #                         for per in peeps:
    #                             if per not in peeps2:
    #                                 no = True
    #                         if not no:
    #                             all_remove.append(ar)
    #                             break
    #             for j in all_remove:
    #                 allofem.remove(j)
    #             allofem.remove(arrangement)
    #             dont = True
    #             break
    #     if not dont:
    #         allofem.remove(arrangement)
    #         all_happs[total_hap] = ast

    if len(all_happs) == 0:
        print("failed--just returning default sol")
        return basic(G, s)
    else:
        #print("===============================")
        print("success--yay worked!")
        max_happy = max(all_happs.keys(), key=lambda x:x)
        D = convert_dictionary(all_happs[max_happy])
        k = len(all_happs[max_happy])

    return D,k




#solver for only medium-96 input
def tempe(G, s):
    global alpha
    alpha = 0.5
    #use 10 rooms, call dp_solver
    _, maps, num = dp(G, s, 10)
    return maps, 10

#solver for only medium-96 input (also gives output for large-157 input)
def tempe2(G, s):
    global alpha
    alpha = 0.5
    #use 25 rooms, call dp_solver
    _, maps, num = dp(G, s, 25)
    return maps, 25

def basic(G, s):
    num_rooms = G.number_of_nodes()
    student_to_room = {}
    room = 0
    for student in range(G.number_of_nodes()):
        student_to_room[student] = room
        room += 1
    return student_to_room , num_rooms

#??????NOT PLACING EVERY STUDENT IN A ROOM RN????????
def solve_dp(G, s):
    global alpha 
    alpha = 0.5
    #for every possible number of rooms between 1 and n=total number of students,
    #use dp to assign students to rooms to maximize happiness while maintaining 
    #stress per room is lower than smax/num_rooms 
    #the actual number of rooms is the one that results in the highest happiness
    total_students = G.number_of_nodes()
    possible_num_rooms = [i for i in range(1, total_students + 1)]
    happiness_to_room_pairing = {}
    for num_rooms in possible_num_rooms:
        #call dp to give max happiness with this number of rooms
        happiness, student_to_room_mapping, numb_rooms = dp(G, s, num_rooms)
        happiness_to_room_pairing[happiness] = (student_to_room_mapping, numb_rooms)
    #print(happiness_to_room_pairing)
    max_happiness = max(happiness_to_room_pairing.keys(), key=lambda x:x)
    best_mapping, rooms = happiness_to_room_pairing[max_happiness]
    for i in happiness_to_room_pairing.keys():
        if len(happiness_to_room_pairing[i][0]) < total_students:
            print(happiness_to_room_pairing[i][0])
    # print(max_happiness)
    # print(best_mapping)
    return best_mapping, rooms


def solve_greedy(G, s):
    global alpha 
    room = 0
    alpha = 0.5#0.5 #0.7 #1
    a = room
    update = lambda a1: (1 - math.exp(-a1))
    edgeview = G.edges
    pair_to_ratio = {}
    for pair in edgeview:
        happy = edgeview[pair]['happiness']
        stress = edgeview[pair]['stress']
        #ratio = happy/stress
        if happy == 0:
            happy = 0.001
        if stress == 0:
            stress = 0.001
        ratio = (happy, stress)
        pair_to_ratio[pair] = ratio

    sorted_dict = sorted(pair_to_ratio.items(), key=lambda p: p[1][1]/p[1][0], reverse=False)#p[1][0]/p[1][1], reverse=True)
    #maps student to the room theyre in 
    seen_students = {}
    #maps room to list of students in that room
    room_to_students = {}
    #adjustment = 1.5 
    while len(seen_students) < G.number_of_nodes():
        a = room
        #print("DEBUG:", room_to_students)
        best = sorted_dict[0]
        sorted_dict.remove(best)
        #print("-CONT-", best[0][0], best[0][1])
        thresh = room + 1
        if best[0][0] not in seen_students.keys() or best[0][1] not in seen_students.keys():
            if best[0][0] not in seen_students.keys() and best[0][1] not in seen_students.keys():
                #add each student to the room dict
                #if room is currently not empty,
                i,j = best[0][0], best[0][1]
                if room in room_to_students.keys():
                    #make sure stress levels are valid 
                    #room was curr_room before
                    temp = room_to_students[room].copy()
                    temp.append(i)
                    temp.append(j)
                    curr_load = calculate_stress_for_room(temp, G)
                    if edgeview[pair]['stress'] >= s/(thresh):
                            room += 1
                            room_to_students[room] = [i]
                            seen_students[i] = room
                            room += 1
                            room_to_students[room] = [j]
                            seen_students[j] = room
                    elif curr_load <= (s/(thresh))*alpha:
                        room_to_students[room].append(i)
                        room_to_students[room].append(j)
                        seen_students[i] = room
                        seen_students[j] = room
                    else:
                        check = fits_in_other_room(G, [i, j], room_to_students, room, seen_students)
                        if check != -1:
                            seen_students[best[0][0]] = check
                            seen_students[best[0][1]] = check
                            room_to_students[check].append(best[0][0])
                            room_to_students[check].append(best[0][1])
                        else:
                            room += 1
                            alpha = update(a)
                            room_to_students[room] = [i]
                            room_to_students[room].append(j)
                            seen_students[i] = room
                            seen_students[j] = room
                #if room is empty
                else:
                    room_to_students[room] = [i]
                    room_to_students[room].append(j)
                    seen_students[i] = room
                    seen_students[j] = room
            
            elif best[0][0] not in seen_students.keys():
                #put best[0][0] into the same room as best[0][1]
                
                curr_room = seen_students[best[0][1]]
                #calculating the total amount of added stress if we add student i to this room
                i = best[0][0]
                temp = room_to_students[curr_room].copy()
                temp.append(i)

                curr_load = calculate_stress_for_room(temp, G)
                if curr_load <= (s/(thresh))*alpha:
                    seen_students[best[0][0]] = curr_room
                    room_to_students[curr_room].append(best[0][0])
                    #room += 1
                else:
                    check = fit_in_other_room(G, best[0][0], room_to_students, room, seen_students)
                    if check != -1:
                        seen_students[best[0][0]] = check
                        room_to_students[check].append(best[0][0])
                    else:
                        room += 1
                        alpha = update(a)
                        seen_students[best[0][0]] = room
                        room_to_students[room] = [best[0][0]]
            else:
                #put best[0][1] into the same room as best[0][0]
                curr_room = seen_students[best[0][0]]

                #calculating the total amount of added stress if we add student i to this room
                i = best[0][1]
                temp = room_to_students[curr_room].copy()
                temp.append(i)

                curr_load = calculate_stress_for_room(temp, G)
                if curr_load <= (s/(thresh))*alpha:
                    seen_students[best[0][1]] = curr_room
                    room_to_students[curr_room].append(i)
                    #room += 1
                else:
                    check = fit_in_other_room(G, best[0][1], room_to_students, room, seen_students)
                    if check != -1:
                        seen_students[best[0][1]] = check
                        room_to_students[check].append(best[0][1])
                    else:
                        room += 1
                        alpha = update(a)
                        seen_students[best[0][1]] = room
                        room_to_students[room] = [best[0][1]]
        #print("--TEST---:", room_to_students) 
    threshold = s/thresh
    test = check_exceed2(room_to_students, G, threshold, seen_students)
    numTimes = 2
    #print("test: ", room_to_students)
    while test is not None and numTimes > 0:
        old_room = test[0]
        student_to_move = test[1]
        the_room = test[2]
        if the_room == -1:
            room += 1
            alpha = update(a)
            seen_students[student_to_move] = room
            room_to_students[room] = [student_to_move]
            room_to_students[old_room].remove(student_to_move)
        else:
            seen_students[student_to_move] = the_room
            room_to_students[the_room].append(student_to_move)
            room_to_students[old_room].remove(student_to_move)
        #print(room_to_students)
        threshold = s/thresh
        test = check_exceed2(room_to_students, G, threshold, seen_students)
        numTimes -= 1

    return seen_students, (room + 1)
    #pass




# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G, s = read_input_file(path)
#     D, k = solve(G, s)
#     assert is_valid_solution(D, G, s, k)
#     #D = {7:0, 2:0, 1:0, 5:1, 0:1, 9:1, 8:2, 6:2, 3:2, 4:2}
#     #print("Pairing:", D)
#     print("Total Happiness: {}".format(calculate_happiness(D, G)))
#     print("Total Stress: {}".format(calculate_stress(D, G)))
#     room_to_student = {}
#     for j in D.values():
#         room_to_student[j] = []
#     for i in D.keys():
#         room_to_student[D[i]].append(i)
#     thresh = s/(len(room_to_student.keys()))
#     for room in room_to_student.keys():
#         stress = calculate_stress_for_room(room_to_student[room], G)
#         if stress > thresh:
#             print(room)
#             print("NOT MEETING CONDITION")
#     #print(len(room_to_student.keys()), k)
#     write_output_file(D, 'test.out')
#     #read_output_file('samples/10.out', G, s)



# #For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
if __name__ == '__main__':
    inputs = glob.glob('med_rest/*')
    counter = 1
    for input_path in inputs:
        print(counter, ": ", input_path)
        output_path = 'outputs/' + os.path.basename(os.path.normpath(input_path))[:-3] + '.out'
        G, s = read_input_file(input_path, 100)
        D, k = solve(G, s)
        print("Total Happiness: {}".format(calculate_happiness(D, G)))
        assert is_valid_solution(D, G, s, k)
        #cost_t = calculate_happiness(T)
        write_output_file(D, output_path)
        counter += 1


#file_path/
#file_path/




