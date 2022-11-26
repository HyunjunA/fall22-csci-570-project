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
def time_wrapper():
    start_time = time.time()
    call_algorithm()
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken


#  DEFINE GLOBAL VARIABLES
P=[]


# Implement the basic Dynamic Programming solution to the Sequence Alignment problem. Run the test set provided and show your results.

# A. Algorithm Description
# Suppose we are given two strings 𝑋 and 𝑌, where 𝑋 consists of the sequence of
# symbols 𝑥1, 𝑥2 , ... , 𝑥𝑚 and 𝑌 consists of the sequence of symbols 𝑦1, 𝑦2 , ... , 𝑦𝑛.
# Consider the sets {1, 2, ... , 𝑚} and {1, 2, ... , 𝑛} as representing the different positions in the strings 𝑋 and 𝑌, and consider a matching of these sets; Recall that a matching is a set of ordered pairs with the property that each item occurs in at most one pair. We say that a matching 𝑀 of these two sets is an alignment if there are no “crossing” pairs: if (𝑖, 𝑗), (𝑖', 𝑗') ε 𝑀 and 𝑖 < 𝑖' , then 𝑗 < 𝑗'. Intuitively, an alignment gives a way of lining up the two strings, by telling us which pairs of positions will be lined up with one another.
# Our definition of similarity will be based on finding the optimal alignment between 𝑋 and 𝑌, according to the following criteria. Suppose 𝑀 is a given alignment between 𝑋 and 𝑌:
# 1. First, there is a parameter δ𝑒 > 0 that defines a gap penalty. For each
# position of 𝑋or 𝑌 that is not matched in 𝑀 — it is a gap — we incur a cost of δ.
 
# 2. Second, for each pair of letters 𝑝, 𝑞 in our alphabet, there is a mismatch cost of α𝑝𝑞 for lining up 𝑝 with 𝑞. Thus, for each (𝑖, 𝑗) ε 𝑀, we pay the
# appropriate mismatch cost α𝑥 𝑦 for lining up 𝑥𝑖 with 𝑦𝑗. One generally 𝑖𝑗
# assumes that α𝑝𝑝 = 0 for each letter 𝑝—there is no mismatch cost to line up
# a letter with another copy of itself—although this will not be necessary in
# anything that follows.
# 3. The cost of 𝑀 is the sum of its gap and mismatch costs, and we seek an
# alignment of minimum cost.



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

# MAKE the variable naive_opt available from any function
naive_opt = 0

def get_indices(lines,start_index):
    lines=lines[start_index:]
    indices = []
    # temp_ind, lines
    # for line in lines:
    
    for i, line in enumerate(lines):
        # check whether line[0] is  integer string
        if line[0].isdigit() :
            indices.append(int(line))
        elif line[0].isdigit() == False and i > 0:
            break
    return indices, start_index+i

def generate_string(base_string, indices):
    string = base_string
    # print("base_string", base_string)
    for index in indices:
        base_string = base_string[:index+1] + base_string + base_string[index+1:]
        # print("base_string", base_string)
    
    return base_string

def alpha(x,y):
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

def naiveMethod(input_file, output_file):
    input="SampleTestCases/"+input_file
    file = open(input, "r")

    lines = file.readlines()
    file.close()
    # get first part of number string

    first_indices,i_s=get_indices(lines,0)
    generated_string1_x = generate_string(lines[0].strip(), first_indices)
   
    print("Generated String x: ", generated_string1_x)

    second_indices,i_final=get_indices(lines,i_s)
    generated_string2_y = generate_string(lines[i_s].strip(), second_indices)
   
    print("Generated String y: ", generated_string2_y)


    

    # define OPT as a 2D array  
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]
    
    print(OPT)
    # size of OPT

    # initialize OPT(0,0) = 0
    OPT[0][0] = 0

    delta = 30

    # initialize OPT(i,0) = delta*i
    for i in range(1,len(generated_string1_x)+1):
        OPT[i][0] = i*delta
    # initialize OPT(0,j) = delta*j
    for j in range(1,len(generated_string2_y)+1):
        OPT[0][j] = delta*j



    # for i =1 to m
    for i in range(1,len(generated_string1_x)+1):
        # for j = 1 to n
        for j in range(1,len(generated_string2_y)+1):
            
            if i > 0 and j > 0:
                           
                OPT[i][j] = min ( 
                    OPT[i-1][j-1] + alpha(generated_string1_x[i-1],generated_string2_y[j-1]), 
                    OPT[i-1][j] + delta, 
                    OPT[i][j-1] + delta 
                )

    # last element of the matrix is the answer
    print("The minimum cost of alignment is: ", OPT[len(generated_string1_x)][len(generated_string2_y)])




    # pattern lengths
    M = len(generated_string1_x)
    N = len(generated_string2_y)
    # i = M-1
    # j = N-1
    # alignment1 = ""
    # while True:
    #     if i == 0 or j == 0:
    #         break
    #     elif i <= M-1 and j <= N-1:
            
    #         # find  max among OPT[i-1][j-1], OPT[i-1][j], OPT[i][j-1]
    #         max_val = max(OPT[i-1][j-1], OPT[i-1][j], OPT[i][j-1])

    #         if max_val == OPT[i-1][j-1]:
    #             alignment1 = generated_string1_x[i] + alignment1
    #             i = i-1
    #             j = j-1
                
    #         elif max_val == OPT[i-1][j]:
    #             alignment1 = '_' + alignment1
    #             i = i-1
    #             # generated_string1_x
    #             # generated_string2_y
                
    #         elif max_val == OPT[i][j-1]:
    #             alignment1 = '_' + alignment1
    #             j = j-1
    # print("alignment1", alignment1)
            



    # should make my own "Reconstructing the solution"
    # Reconstructing the solution 
    l = N + M   # maximum possible length
    i = M
    j = N
     
    xpos = l
    ypos = l

    x=generated_string1_x
    y=generated_string2_y
 
    # Final answers for the respective strings
    import numpy as np
    xans = np.zeros(l+1, dtype=int)
    yans = np.zeros(l+1, dtype=int)
     
 
    while not (i == 0 or j == 0):
        #print(f"i: {i}, j: {j}")
        if x[i - 1] == y[j - 1]:       
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
        elif (OPT[i - 1][j - 1] + alpha(x[i-1], y[j-1])) == OPT[i][j]:
         
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
         
        elif (OPT[i - 1][j] + delta) == OPT[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord('_')
            xpos -= 1
            ypos -= 1
            i -= 1
         
        elif (OPT[i][j - 1] + delta) == OPT[i][j]:       
            xans[xpos] = ord('_')
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            j -= 1
         
 
    while xpos > 0:
        if i > 0:
            i -= 1
            xans[xpos] = ord(x[i])
            xpos -= 1
        else:
            xans[xpos] = ord('_')
            xpos -= 1
     
    while ypos > 0:
        if j > 0:
            j -= 1
            yans[ypos] = ord(y[j])
            ypos -= 1
        else:
            yans[ypos] = ord('_')
            ypos -= 1
 
    # Since we have assumed the answer to be n+m long,
    # we need to remove the extra gaps in the starting
    # id represents the index from which the arrays
    # xans, yans are useful
    id = 1
    i = l
    while i >= 1:
        if (chr(yans[i]) == '_') and chr(xans[i]) == '_':
            id = i + 1
            break
         
        i -= 1
 
    # Printing the final answer
    print(f"Minimum Penalty in aligning the genes = {OPT[M][N]}")
    print("The aligned genes are:")   
    # X
    i = id
    x_seq = ""
    while i <= l:
        x_seq += chr(xans[i])
        i += 1
    print(f"X seq: {x_seq}")
 
    # Y
    i = id
    y_seq = ""
    while i <= l:
        y_seq += chr(yans[i])
        i += 1
    print(f"Y seq: {y_seq}")






    # Save OPT
    naive_opt = OPT
    return naive_opt     


def eachinterationfrom(x,y):

    generated_string1_x = x
    generated_string2_y = y

    # # define OPT as a 2D array  
    # OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]


    # in the space efficient method, we only need to keep track of the previous row and the current row
    # define OPT which keep track of the previous row and the current row
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(2)]
    
    print(OPT)
    # size of OPT

    # initialize OPT(0,0) = 0
    OPT[0][0] = 0

    delta = 30

    # initialize OPT(0,j) = delta*j
    for j in range(1,len(generated_string2_y)+1):
        OPT[0][j] = delta*j

    # initialize OPT(i,0) = delta*i
    for i in range(1,2):
        OPT[i][0] = i*delta
    

    # initialize OPT(i,0) = delta*i for i = 1

    



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
        
        # print("i: ", i)
        i = i-iter_i
        # print("i-iter_i: ", i)
        for j in range(1,len(generated_string2_y)+1):
            
            if i > 0 and j > 0:
                           
                OPT[i][j] = min ( 
                    OPT[i-1][j-1] + alpha(generated_string1_x[cur_i-1],generated_string2_y[j-1]), 
                    OPT[i-1][j] + delta, 
                    OPT[i][j-1] + delta 
                )

        iter_i+=1

    # last element of the matrix is the answer
    print("The minimum cost of alignment is: ", OPT[1][len(generated_string2_y)])

    return OPT[1]


def divConq_align(generated_string1_x,generated_string2_y, x_range, y_range):

    if generated_string1_x=="" and generated_string2_y=="":
        return 

    generated_string2_y_L = generated_string2_y[:int(len(generated_string2_y)/2)]
    generated_string2_y_R = generated_string2_y[int(len(generated_string2_y)/2):]


    # Call Space-Efficient-Alignment(X,Y[1:n/2])
    # Call Backward-Space-Efficient-Alignment(X,Y[n/2+1:n]) 

    # Let q be the index minimizing f(q,n/2)+g(q,n/2)
    # Add (q, n/2) to global list P 
    # Divide-and-Conquer-Alignment(X[1 : q],Y[1 : n/2]) 
    # Divide-and-Conquer-Alignment(X[q + 1 : n],Y[n/2 + 1 : n]) 
    # 
    # Return P
    
    # Call Space-Efficient-Alignment(X,Y[1:n/2])
    for_opt_last=eachinterationfrom(generated_string1_x,generated_string2_y_L)
    
    
    # reverse generated_string1_x
    generated_string1_x_reversed = generated_string1_x[::-1]
    # reverse generated_string2_y_R

    generated_string2_y_R_reversed = generated_string2_y_R[::-1]
    # Call Backward-Space-Efficient-Alignment(X,Y[n/2+1:n]) 
    back_opt_last=eachinterationfrom(generated_string1_x_reversed,generated_string2_y_R_reversed)

    q = np.argmin(for_opt_last+back_opt_last[::-1])

    # tuple (x:q, y:n/2) to global list P

    n = len(generated_string2_y)
    n_2 = int(n/2)

    elem_P = [(q,n_2)]
    # Add (q, n/2) to global list P 
    P.append(elem_P)
    print("P",P)

    # Divide-and-Conquer-Alignment(X[1 : q],Y[1 : n/2])
    l_p=divConq_align(generated_string1_x[:q],generated_string2_y[:n_2])

    # Divide-and-Conquer-Alignment(X[q + 1 : n],Y[n/2 + 1 : n])
    r_p=divConq_align(generated_string1_x[q:],generated_string2_y[n_2:])


    # current issue

    # 해당 좌표가 상대적임.
    # 해결책
    # divConq_align(generated_string1_x,generated_string2_y, x_range, y_range):
    # len_x = len(generated_string1_x)
    # len_y = len(generated_string2_y)
    # 즉, divConq_align(generated_string1_x,generated_string2_y,[0,len_x],[0,len_y])


    return P



    




def spaceEfficientMethod(input_file, output_file):
    # this is the space efficient method
    # read the input file

    input="SampleTestCases/"+input_file
    file = open(input, "r")

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

    # divConq_align(generated_string1_x,generated_string2_y,[0,len_x],[0,len_y])
    divConq_align(generated_string1_x,generated_string2_y)
   
    


    print("hello")
    print("P",P)
    # REMOVE DUPLICATES IN P
    # P = list(set(P))
    



    







def call_algorithm(input_file, output_file):
    # Read the input file
    # file = open("SampleTestCases/input3.txt", "r")

    # naive approach
    # naiveMethod(input_file, output_file)

    # space efficient approach
    spaceEfficientMethod(input_file, output_file) 

     





if __name__ == "__main__":

    # `python3 basic_3.py input.txt output.txt`
    
    # input_file = sys.argv[1]
    # output_file = sys.argv[2]

    input_file = "input1.txt"
    output_file = "output1.txt"

    call_algorithm(input_file, output_file)