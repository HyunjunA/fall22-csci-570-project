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

def time_wrapper(input_file,output_file):
    start_time = time.time()
    call_algorithm(input_file,output_file)
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken

#  DEFINE GLOBAL VARIABLES
P=[]
# MAKE the variable naive_opt available from any function
naive_opt = 0
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

#def basic_algo(generated_string1_x, generated_string2_y,x_range,y_range):
def basic_algo(generated_string1_x, generated_string2_y):   
    """
    Process generated strings, then runs basic version 
    of the dynamic programming algorithm for sequence alignment
    """
    # define OPT as a 2D array  
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]

    # initialize OPT(0,0) = 0
    OPT[0][0] = 0
    
    # lengths of input strings
    m = len(generated_string1_x)
    n = len(generated_string2_y)

    # optimal costs DP table initialization
    opt = [[0 for x in range(n+1)] for y in range(m+1)]
    # initialize OPT(i,0) = delta*i
    for i in range(1,m+1):
        opt[i][0] = i*delta
    # initialize OPT(0,j) = delta*j
    for j in range(1,n+1):
        opt[0][j] = delta*j
    
    # construction of DP table from bottom-up
    for i in range(1,m+1):
        for j in range(1,n+1):                       
            opt[i][j] = min ( 
                opt[i-1][j-1] + alpha(generated_string1_x[i-1],generated_string2_y[j-1]), 
                opt[i-1][j] + delta, 
                opt[i][j-1] + delta 
            )

    #same as xans and yans
    x_final = []
    y_final = []

    while m!=0 and n!=0:
        if (generated_string1_x[m-1] == generated_string2_y[n-1]) or (opt[m-1][n-1] + alpha(generated_string1_x[m-1],generated_string2_y[n-1])) == opt[m][n]:       
            x_final.append(generated_string1_x[m-1])
            y_final.append(generated_string2_y[n-1])
            m-=1
            n-=1
         
        elif (opt[m][n-1] + delta) == opt[m][n]:       
            x_final.append('_')
            y_final.append(generated_string2_y[n-1])
            n -= 1

        elif (opt[m-1][n] + delta) == opt[m][n]:
            x_final.append(generated_string1_x[m-1])
            y_final.append('_')
            m -= 1

    while m>0:
        x_final.append(generated_string1_x[m-1])
        m-=1

    while n>0:
        y_final.append(generated_string2_y[n-1])
        n-=1

    while len(x_final) < len(y_final):
        x_final.append('_')

    while len(y_final) < len(x_final):
        y_final.append('_')

    x_final.reverse()
    y_final.reverse()

    #aligned_x = ''.join(x_final)
    #aligned_y = ''.join(y_final)  
    return opt  
  
def eachinterationfrom(x,y):
    generated_string1_x = x
    generated_string2_y = y
    # # define OPT as a 2D array  
    # OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]

    # in the space efficient method, we only need to keep track of the previous row and the current row
    # define OPT which keep track of the previous row and the current row
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(2)]

    # initialize OPT(0,0) = 0
    OPT[0][0] = 0
    
    delta = 30

    # initialize OPT(0,j) = delta*j
    for j in range(1,len(generated_string2_y)+1):
        OPT[0][j] = delta*j

    # initialize OPT(i,0) = delta*i
    for i in range(1,2):
        OPT[i][0] = i*delta

    # number of interation in i
    iter_i=0

    for i in range(1,len(generated_string1_x)+1):
        cur_i = i
        if i > 1:
            # Move current row to previous row
            OPT[0] = OPT[1]
            # set current row to 0. 
            OPT[1] = [0 for x in range(len(generated_string2_y)+1)]
            OPT[1][0] = i*delta
        i = i-iter_i
        for j in range(1,len(generated_string2_y)+1):
            if i > 0 and j > 0:
                OPT[i][j] = min ( 
                    OPT[i-1][j-1] + alpha(generated_string1_x[cur_i-1],generated_string2_y[j-1]), 
                    OPT[i-1][j] + delta, 
                    OPT[i][j-1] + delta 
                )
        iter_i+=1
    return OPT[1]

def divConq_align(generated_string1_x,generated_string2_y,x_range,y_range):

    # get string_x from generated_string1_x using x_range
    string_x = generated_string1_x[x_range[0]:x_range[1]]
    string_y = generated_string2_y[y_range[0]:y_range[1]]
    
    if len(string_x) < 2 and len(string_y) <2:
        temp = basic_algo(string_x,string_y)
        #temp = basic_algo(string_x,string_y,x_range,y_range)
        return temp
    
    # professor's
    string_x_L = string_x[:int(len(string_x)/2)]
    string_x_R = string_x[int(len(string_x)/2):]
    
    for_opt_last=eachinterationfrom(string_x_L,string_y)
    
    string_x_R_reversed = string_x_R[::-1]
    string_y_reversed = string_y[::-1]
    # Call Backward-Space-Efficient-Alignment(X,Y[n/2+1:n]) 
    back_opt_last=eachinterationfrom(string_x_R_reversed,string_y_reversed)

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

    # sort P by x
    P.sort(key=lambda x: x[0][0])

    # Divide-and-Conquer-Alignment(X[1 : n/2],Y[1 : q])
    x_range_fi = [x_range[0],n_2]
    y_range_fi = [y_range[0],q]

    l_p=divConq_align(generated_string1_x,generated_string2_y, x_range_fi, y_range_fi)

    # Divide-and-Conquer-Alignment(X[n/2+1 : n],Y[q+1 : n])
    x_range_se = [n_2,x_range[1]]
    y_range_se = [q,y_range[1]]
    
    r_p=divConq_align(generated_string1_x,generated_string2_y, x_range_se, y_range_se)

    return P

def reconstructUsingPVerTempReversed(generated_string1_x,generated_string2_y,P, output_file):
    x_seq = ""
    y_seq = ""

    # x_seq
    len_P = len(P)
    ind_P=0

    x_seq_loc = len(generated_string1_x)-1
    y_seq_loc = len(generated_string2_y)-1

    # reverse for loop
    # for i in range(len_P-1,-1,-1):
    
    # for i < len_P:
    for ind_P in range(len_P-1,-1,-1):
        if ind_P == len_P-1 :
            x = P[ind_P][0][0]
            y = P[ind_P][0][1]

            x_prev = P[ind_P-1][0][0]
            y_prev = P[ind_P-1][0][1]

            x_diff=x-x_prev
            y_diff=y-y_prev

            if x_diff == 1 and y_diff == 0:
                x_seq = generated_string1_x[x_seq_loc] + x_seq
                x_seq_loc -= 1

                y_seq = "_" + y_seq
            
            elif x_diff == 0 and y_diff == 1:
                x_seq = "_" + x_seq

                y_seq = generated_string2_y[y_seq_loc] + y_seq
                y_seq_loc -= 1
            
            elif x_diff == 1 and y_diff == 1:
                x_seq = generated_string1_x[x_seq_loc] + x_seq
                x_seq_loc -= 1

                y_seq = generated_string2_y[y_seq_loc] + y_seq
                y_seq_loc -= 1
        else:
            x = P[ind_P][0][0]
            y = P[ind_P][0][1]

            x_prev = P[ind_P-1][0][0]
            y_prev = P[ind_P-1][0][1]

            x_diff=x-x_prev
            y_diff=y-y_prev

            if x_diff ==1 and y_diff == 1:
                # add generated_string1_x[x_seq_loc] to most left of x_seq
                x_seq = generated_string1_x[x_seq_loc] + x_seq
                # x_seq += generated_string1_x[x_seq_loc]
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
        
    for_opt_last_val = np.array(eachinterationfrom(generated_string1_x,generated_string2_y))
    len_final_string = len(for_opt_last_val) -1
    final_cost = for_opt_last_val[len_final_string]
    print("final cost",final_cost)
    
    # open 
    output_file = output_file
    output="../"+output_file
    file = open(output, "a")

    # 1. Cost of the alignment (Integer)
    # 2. First string alignment ( Consists of A, C, T, G, _ (gap) characters)
    # 3. Second string alignment ( Consists of A, C, T, G, _ (gap) characters )
    # 4. Time in Milliseconds (Float)
    # 5. Memory in Kilobytes (Float)

    # write to file
    output = open(output_file, "a")
    output.write(str(final_cost) + "\n")
    output.write(x_seq + "\n")
    output.write(y_seq + "\n")

    # close output_file
    output.close()

    print("x_seq",x_seq)
    print("y_seq",y_seq)

def spaceEfficientMethod(input_file,output_file):
    # this is the space efficient method
    # read the input file
    print("input_file",input_file)
    file = open(input_file, "r")

    lines = file.readlines()
    file.close()
    # get first part of number string

    first_indices,i_s=get_indices(lines,0)
    generated_string1_x = generate_string(lines[0].strip(), first_indices)

   
    print("Generated String x: ", generated_string1_x)

    second_indices,i_final=get_indices(lines,i_s)
    generated_string2_y = generate_string(lines[i_s].strip(), second_indices)

    print("Generated String y: ", generated_string2_y)

    # 
    len_x=len(generated_string1_x)
    len_y=len(generated_string2_y)

    # Divide-and-Conquer-Alignment(X ,Y )

    divConq_align(generated_string1_x,generated_string2_y,[0,len_x],[0,len_y])

    # insert into the P
    # if [(0,0)] not in P:
    #     P.append([(0,0)])
    if [(len_x,len_y)] not in P:
        P.append([(len_x,len_y)])

        
    # P.insert(0,[(len_x,len_y)])
    # sort P by x and y
    P.sort(key=lambda x: (x[0][0],x[0][1]))

    # reconstruct the alignment
    reconstructUsingPVerTempReversed(generated_string1_x,generated_string2_y,P,output_file)
    
    # get the only x element in P
    P_x = [x[0][0] for x in P]
    # set P_x to set
    P_x_set = set(P_x)
    # list P_x_set  
    P_x_set_list = list(P_x_set)
    # show length of P_x_set_list

    # get the only x element in P
    P_x = [x[0][1] for x in P]
    # set P_x to set
    P_x_set = set(P_x)
    # list P_x_set  
    P_x_set_list = list(P_x_set)

def call_algorithm(input_file,output_file):
    # space efficient approach
    spaceEfficientMethod(input_file,output_file) 

if __name__ == "__main__":
    # Read the input file
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # call_algorithm(input_file, output_file)
    elapsed_time = time_wrapper(input_file,output_file)
    consumed_memory = process_memory()
    
    # write results to output file
    with open(output_file, "a") as f:
        # 4. Time in Milliseconds (Float)
        f.write(f"{elapsed_time}\n")
        # 5. Memory in Kilobytes (Float)
        f.write(f"{consumed_memory}\n")
    # close output_file
    f.close()