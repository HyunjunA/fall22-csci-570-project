from email.mime import base
import sys
from resource import *
import time
import psutil
import numpy as np

def process_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_consumed = int(memory_info.rss/1024)
    return memory_consumed

def time_wrapper(lines):
    start_time = time.time()
    cost, alignment1, alignment2 = spaceEfficientMethod(lines)
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken, cost, alignment1, alignment2

#  DEFINE GLOBAL VARIABLES
P=[]
# Delta value: price of a gap
delta = 30

# B. Input string Generator
# The input to the program would be a text file containing the following information:
# 1. First base string (𝑠1)
# 2. Next 𝑗 lines consist of indices after which the copy of the previous string needs to be inserted in the cumulative string. (eg given below)
# 3. Second base string (𝑠2)
# 4. Next 𝑘 lines consist of indices after which the copy of the previous
# string needs to be inserted in the cumulative string. (eg given below)
# This information would help generate 2 strings from the original 2 base strings. This file could be used as an input to your program and your program could use the base strings and the rules to generate the actual strings. Also note that the numbers 𝑗 and 𝑘 correspond to the first and the second string respectively. Make
# sure you validate the length of the first and the second string to be 2𝑗 * 𝑙𝑒𝑛(𝑠1) and 2𝑘 * 𝑙𝑒𝑛(𝑠2). Please note that the base strings need not have to be of equal
# length and similarly, 𝑗 need not be equal to 𝑘.

def get_indices(lines,start_index):
    lines=lines[start_index:]
    indices = []
    
    for i, line in enumerate(lines):
        # check whether line[0] is  integer string
        if line[0].isdigit() :
            indices.append(int(line))
        elif line[0].isdigit() == False and i > 0:
            break
    return indices, start_index+i

def generate_string(base_string, indices):
    for index in indices:
        base_string = base_string[:index+1] + base_string + base_string[index+1:]    
    return base_string

def alpha(x,y):
    """
    Function to calulcate the mismatch alphas value between 2 characters
    :param x: first string
    :param y: Second string
    """
    if x == y:
        return 0
    elif x == 'A' and y == 'C':
        return 110
    elif x == 'A' and y == 'G':
        return 48
    elif x == 'A' and y == 'T':
        return 94
    elif x == 'C' and y == 'A':
        return 110
    elif x == 'C' and y == 'G':
        return 118
    elif x == 'C' and y == 'T':
        return 48
    elif x == 'G' and y == 'A':
        return 48
    elif x == 'G' and y == 'C':
        return 118
    elif x == 'G' and y == 'T':
        return 110
    elif x == 'T' and y == 'A':
        return 94
    elif x == 'T' and y == 'C':
        return 48
    elif x == 'T' and y == 'G':
        return 110


def forward_alignment(x,y):
    generated_string1_x = x
    generated_string2_y = y

    # in the space efficient method, we only need to keep track of the previous row and the current row
    # define OPT which keep track of the previous row and the current row
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(2)]

    # initialize OPT(0,j) = delta*j
    for j in range(1,len(generated_string2_y)+1):
        OPT[0][j] = delta*j
    # initialize OPT(i,0) = delta*i
    for i in range(1,2):
        OPT[i][0] = i*delta

    for i in range(1,len(generated_string1_x)+1):
        if i > 1:
            # Move current row to previous row
            OPT[0] = OPT[1]
            # set current row to 0. 
            OPT[1] = [0 for x in range(len(generated_string2_y)+1)]
            OPT[1][0] = i*delta

        for j in range(1,len(generated_string2_y)+1):
            OPT[1][j] = min ( 
                OPT[0][j-1] + alpha(generated_string1_x[i-1],generated_string2_y[j-1]), 
                OPT[0][j] + delta, 
                OPT[1][j-1] + delta 
            )

    return OPT[1]

def divConq_align(generated_string1_x,generated_string2_y,x_range,y_range):

    # get string_x from generated_string1_x using x_range
    string_x = generated_string1_x[x_range[0]:x_range[1]]
    string_y = generated_string2_y[y_range[0]:y_range[1]]
    
    if len(string_x) < 2 and len(string_y) < 2:
        return 
    
    string_x_L = string_x[:int(len(string_x)/2)]
    string_x_R = string_x[int(len(string_x)/2):]
    
    for_opt_last=forward_alignment(string_x_L,string_y)
    
    # Backward-Space-Efficient-Alignment(X,Y[n/2+1:n]) by reversing strings
    string_x_R_reversed = string_x_R[::-1]
    string_y_reversed = string_y[::-1]
    back_opt_last=forward_alignment(string_x_R_reversed,string_y_reversed)

    # add for_opt_last and back_opt_last[::-1] element by element
    for_opt_last_np = np.array(for_opt_last)
    back_opt_last_np_reversed = np.array(back_opt_last[::-1])
    
    q = np.argmin(for_opt_last_np+back_opt_last_np_reversed)
    q += y_range[0]
    for_opt_last, back_opt_last = [], []

    # tuple (x:n/2, y:q) to global list P
    n = len(string_x)
    n_2 = int(n/2)
    n_2 = x_range[0] + n_2
    elem_P = [(n_2,q)]
    P.append(elem_P)

    # Divide-and-Conquer-Alignment(X[1 : n/2],Y[1 : q])
    x_range_fi = [x_range[0],n_2]
    y_range_fi = [y_range[0],q]
    
    divConq_align(generated_string1_x,generated_string2_y, x_range_fi, y_range_fi)

    # Divide-and-Conquer-Alignment(X[n/2+1 : n],Y[q+1 : n])
    x_range_se = [n_2,x_range[1]]
    y_range_se = [q,y_range[1]]
    
    divConq_align(generated_string1_x,generated_string2_y, x_range_se, y_range_se)


def reconstructUsingPVerTempReversed(generated_string1_x,generated_string2_y):
    x_seq = ""
    y_seq = ""

    len_P = len(P)
    ind_P=0

    x_seq_loc = len(generated_string1_x)-1
    y_seq_loc = len(generated_string2_y)-1

    for ind_P in range(len_P-1,-1,-1):
        x = P[ind_P][0][0]
        y = P[ind_P][0][1]
        
        x_prev = P[ind_P-1][0][0]
        y_prev = P[ind_P-1][0][1]

        x_diff=x-x_prev
        y_diff=y-y_prev

        if x_diff == 1 and y_diff == 1:
            # add generated_string1_x[x_seq_loc] to most left of x_seq
            x_seq = generated_string1_x[x_seq_loc] + x_seq
            x_seq_loc -= 1
            y_seq = generated_string2_y[y_seq_loc] + y_seq
            y_seq_loc -= 1
            
        elif x_diff ==1 and y_diff == 0:
            x_seq = generated_string1_x[x_seq_loc] + x_seq
            x_seq_loc -= 1
            y_seq = "_" + y_seq
        
        elif x_diff ==0 and y_diff == 1:
            x_seq = "_" + x_seq
            y_seq = generated_string2_y[y_seq_loc] + y_seq
            y_seq_loc -= 1

    #calculate final alignment cost
    for_opt_last_val = np.array(forward_alignment(generated_string1_x,generated_string2_y))
    len_final_string = len(for_opt_last_val) -1
    final_cost = for_opt_last_val[len_final_string]

    return final_cost, x_seq, y_seq


def spaceEfficientMethod(lines):
    # get first part of number string
    first_indices,i_s=get_indices(lines,0)
    generated_string1_x = generate_string(lines[0].strip(), first_indices)

    second_indices,i_final=get_indices(lines,i_s)
    generated_string2_y = generate_string(lines[i_s].strip(), second_indices)

    len_x=len(generated_string1_x)
    len_y=len(generated_string2_y)

    # Divide-and-Conquer-Alignment(X ,Y )
    divConq_align(generated_string1_x,generated_string2_y,[0,len_x],[0,len_y])

    # insert into the P
    if [(len_x,len_y)] not in P:
        P.append([(len_x,len_y)])

    # sort P by x and y
    P.sort(key=lambda x: (x[0][0],x[0][1]))

    # reconstruct the alignment (returns optimal cost, aligned string x, aligned string y)
    return reconstructUsingPVerTempReversed(generated_string1_x, generated_string2_y)
    

if __name__ == "__main__":
    # Read the input file
    with open(sys.argv[1], 'r') as infile:
        lines = infile.readlines()

    # call_algorithm(input_file, output_file)
    elapsed_time, cost, alignment1, alignment2 = time_wrapper(lines)
    consumed_memory = process_memory()
    
    # write results to output file
    with open(sys.argv[2], 'w') as outfile:
        outfile.write(str(cost) + '\n')
        outfile.write(alignment1+ '\n')
        outfile.write(alignment2+ '\n')
        outfile.write(str(elapsed_time)+ '\n')
        outfile.write(str(consumed_memory))