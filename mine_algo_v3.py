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
# def time_wrapper():
#     start_time = time.time()
#     call_algorithm()
#     end_time = time.time()
#     time_taken = (end_time - start_time)*1000
#     return time_taken

def time_wrapper(input_file,output_file):
    start_time = time.time()
    # call_algorithm()
    call_algorithm(input_file, output_file)
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
    # generated_string1_x = "CGCATC"

    
   
    print("Generated String x: ", generated_string1_x)

    second_indices,i_final=get_indices(lines,i_s)
    generated_string2_y = generate_string(lines[i_s].strip(), second_indices)
    # generated_string2_y ="CACAAT"

    print("Generated String y: ", generated_string2_y)

    # generated_string1_x = "CGCATC"
    # generated_string2_y ="CACAAT"

    # generated_string1_x = "CG"
    # generated_string2_y ="CA"

    generated_string1_x = "ATC"
    generated_string2_y ="AAT"


    

    # define OPT as a 2D array  
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]
    
    # show shape of OPT
    # print("Shape of OPT: ", np.shape(OPT))
    # print(OPT)
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
    # print("The minimum cost of alignment is: ", OPT[len(generated_string1_x)][len(generated_string2_y)])




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
        # if x[i - 1] == y[j - 1]:       
        #     xans[xpos] = ord(x[i - 1])
        #     yans[ypos] = ord(y[j - 1])
        #     xpos -= 1
        #     ypos -= 1
        #     i -= 1
        #     j -= 1
        if (OPT[i - 1][j - 1] + alpha(x[i-1], y[j-1])) == OPT[i][j]:
         
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
    print(f"Similarity in gene alignment = {OPT[M][N]}")
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


    # output OPT[M][N], x_seq, y_seq to output_mine.txt
    with open("output_mine.txt", "w") as f:
        f.write(f"Similarity in gene alignment = {OPT[M][N]}\n")
        f.write("The aligned genes are:\n")
        f.write(f"X seq: {x_seq}\n")
        f.write(f"Y seq: {y_seq}\n")
    







    # Save OPT
    # naive_opt = OPT
    # return naive_opt  
    

    # save the OPT[M][N]
    return OPT[M][N]


def naiveMethod_v3(generated_string1_x, generated_string2_y,x_range,y_range):
    
   
    print("Generated String x: ", generated_string1_x)

   
    print("Generated String y: ", generated_string2_y)


    

    # define OPT as a 2D array  
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]
    
    # print(OPT)
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
            
            
            print("hello")
        elif (OPT[i - 1][j - 1] + alpha(x[i-1], y[j-1])) == OPT[i][j]:
         
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
            print("hello")
        elif (OPT[i - 1][j] + delta) == OPT[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord('_')
            xpos -= 1
            ypos -= 1
            i -= 1
            print("hello")
        elif (OPT[i][j - 1] + delta) == OPT[i][j]:       
            xans[xpos] = ord('_')
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            j -= 1
            print("hello")
    
    # if [(x_range[0]+i, y_range[0]+j)] does not exist in the P
    # then add it to the P

    if [(x_range[0]+i, y_range[0]+j)] not in P:
        P.insert(0, [(x_range[0]+i, y_range[0]+j)])
    # else:
    #     print("Already in P")


    # P.insert(0, [(x_range[0]+i, y_range[0]+j)])
    print("P: ", P)

    

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

    # generated_string1_x = y
    # generated_string2_y = x

    # # define OPT as a 2D array  
    # OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]


    # in the space efficient method, we only need to keep track of the previous row and the current row
    # define OPT which keep track of the previous row and the current row
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(2)]

    # show shape of OPT
    # print("shape of OPT: ", np.shape(OPT))
    
    # print(OPT)
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
    print("Similarity: ", OPT[1][len(generated_string2_y)])

    return OPT[1]


def divConq_align(generated_string1_x,generated_string2_y,x_range,y_range):

    # get string_x from generated_string1_x using x_range
    string_x = generated_string1_x[x_range[0]:x_range[1]]
    string_y = generated_string2_y[y_range[0]:y_range[1]]

    if x_range[0] >=len(generated_string1_x)-2 or y_range[0] >=len(generated_string2_y)-2:
        print("hello")  
    # if generated_string1_x=="" and generated_string2_y=="":
    #     return 


    # generated_string1_x = Y_
    # generated_string2_y = X_L and X_R
    
    # if generated_string1_x length is less than 2 and generated_string2_y length is less than 2 use naive method
    # if len(string_x) < 2 or len(string_y) <2:
    if len(string_x) < 2 and len(string_y) <2:
    # while len(generated_string1_x) <= 2 and len(generated_string2_y) <=2:
        # naive_align(generated_string1_x,generated_string2_y)
        temp = naiveMethod_v3(string_x,string_y,x_range,y_range)




        return temp
        
        

    # original
    # generated_string2_y_L = generated_string2_y[:int(len(generated_string2_y)/2)]
    # generated_string2_y_R = generated_string2_y[int(len(generated_string2_y)/2):]

    # professor's
    string_x_L = string_x[:int(len(string_x)/2)]
    string_x_R = string_x[int(len(string_x)/2):]

    # if string_x_L == "" and string_x_R == "":
    #     return


    # Call Space-Efficient-Alignment(X,Y[1:n/2])
    # Call Backward-Space-Efficient-Alignment(X,Y[n/2+1:n]) 

    # Let q be the index minimizing f(q,n/2)+g(q,n/2)
    # Add (q, n/2) to global list P 
    # Divide-and-Conquer-Alignment(X[1 : q],Y[1 : n/2]) 
    # Divide-and-Conquer-Alignment(X[q + 1 : n],Y[n/2 + 1 : n]) 
    # 
    # Return P


    
    
    # Call Space-Efficient-Alignment(X,Y[1:n/2])
    # Original
    # for_opt_last=eachinterationfrom(generated_string1_x,generated_string2_y_L)
    # Professor's
    for_opt_last=eachinterationfrom(string_x_L,string_y)
    
    
    # reverse generated_string1_x
    string_x_R_reversed = string_x_R[::-1]
    # reverse generated_string2_y_R

    string_y_reversed = string_y[::-1]
    # Call Backward-Space-Efficient-Alignment(X,Y[n/2+1:n]) 
    back_opt_last=eachinterationfrom(string_x_R_reversed,string_y_reversed)

    # add for_opt_last and back_opt_last[::-1] element by element
    for_opt_last_np = np.array(for_opt_last)
    back_opt_last_np_reversed = np.array(back_opt_last[::-1])
    
    q = np.argmin(for_opt_last_np+back_opt_last_np_reversed)
    q += y_range[0]

    for_opt_last, back_opt_last = [], []
    

    # original
    # tuple (x:q, y:n/2) to global list P
    # n = len(generated_string2_y)
    # n_2 = int(n/2)

    # elem_P = [(q,n_2)]
    
    # P.append(elem_P)
    # print("P",P)

    # # Divide-and-Conquer-Alignment(X[1 : q],Y[1 : n/2])
    # l_p=divConq_align(generated_string1_x[:q],generated_string2_y[:n_2])

    # # Divide-and-Conquer-Alignment(X[q + 1 : n],Y[n/2 + 1 : n])
    # r_p=divConq_align(generated_string1_x[q:],generated_string2_y[n_2:])



    # professor's
    # tuple (x:n/2, y:q) to global list P
    n = len(string_x)
    n_2 = int(n/2)
    n_2 = x_range[0] + n_2

    if n_2 == 1:
        print(("Hello"))

    if n_2 == 60:
        print(("Hello"))
    if q == 10:
        print(("Hello"))

    # n_2
    print("n_2: ", n_2)


    elem_P = [(n_2,q)]
    
    P.append(elem_P)

    # sort P by x
    P.sort(key=lambda x: x[0][0])
    print("P",P)

    # Divide-and-Conquer-Alignment(X[1 : n/2],Y[1 : q])
    x_range_fi = [x_range[0],n_2]
    y_range_fi = [y_range[0],q]
    l_p=divConq_align(generated_string1_x,generated_string2_y, x_range_fi, y_range_fi)
    
    
    
    # Divide-and-Conquer-Alignment(X[n/2+1 : n],Y[q+1 : n])
    x_range_se = [n_2,x_range[1]]
    y_range_se = [q,y_range[1]]
    r_p=divConq_align(generated_string1_x,generated_string2_y, x_range_se, y_range_se)

    # current issue

    # 해당 좌표가 상대적임.
    # 해결책
    # divConq_align(generated_string1_x,generated_string2_y, x_range, y_range):
    # len_x = len(generated_string1_x)
    # len_y = len(generated_string2_y)
    # 즉, divConq_align(generated_string1_x,generated_string2_y,[0,len_x],[0,len_y])


    return P



    
def reconstructUsingP(generated_string1_x,generated_string2_y,P):

    print("P",P)
    x_seq = ""
    y_seq = ""

    # 

    # for i in range(len(P)):
    #     print(i)
    #     print("eachPoint i",P[i])
    #     print("eachPoint i+1",P[i+1])
    #     x = P[i][0][0]
    #     y = P[i][0][1]

    #     x_next = P[i+1][0][0]
    #     y_next = P[i+1][0][1]

    #     x_diff=x_next-x
    #     y_diff=y_next-y
    #     if i == 0 :
    #         if x==0 and y==0 and x_diff == 0 and y_diff == 0:
    #             x_seq += ""
    #             y_seq += ""
    #         elif x == 0 and y == 0 and x_diff == 0 and y_diff == 1:
    #             x_seq += ""
    #             y_seq += generated_string2_y[y]
    #         elif x == 0 and y == 0 and x_diff == 1 and y_diff == 0:
    #             x_seq += generated_string1_x[x]
    #             y_seq += ""
            





    #     else:
    #         if x_diff==1 and y_diff==1:
    #             print("diagonal")
    #             # x_seq += generated_string1_x[x]
    #             # y_seq += generated_string2_y[y]
    #             x_seq += generated_string1_x[P[i-1][0][0]]
    #             y_seq += generated_string2_y[P[i-1][0][1]]
    #         elif x_diff==1 and y_diff==0:
    #             print("vertical")
    #             x_seq += generated_string1_x[P[i-1][0][0]]
    #             y_seq += "_"
    #         elif x_diff==0 and y_diff==1:
    #             print("horizontal")
    #             x_seq += "_"
    #             y_seq += generated_string2_y[P[i-1][0][1]]

    # x_seq
    len_P = len(P)
    i=0
    # for i < len_P:
    while i < len_P:
        print(i)
        print("eachPoint i",P[i])
        print("eachPoint i+1",P[i+1])
        x = P[i][0][0]
        y = P[i][0][1]

        x_next = P[i+1][0][0]
        y_next = P[i+1][0][1]

        x_diff=x_next-x
        y_diff=y_next-y
        if i == 0 :
            if x==0 and y==0 and x_diff == 0 and y_diff == 0:
                x_seq += ""
                
            elif x == 0 and y == 0 and x_diff == 0 and y_diff == 1:
                x_seq += ""
                
            elif x == 0 and y == 0 and x_diff == 1 and y_diff == 0:
                x_seq += generated_string1_x[x]
                

        else:
            if x_diff==1 and y_diff==1:
                print("diagonal")
                # x_seq += generated_string1_x[x]
                # y_seq += generated_string2_y[y]
                x_seq += generated_string1_x[P[i-1][0][0]]
                
            elif x_diff==1 and y_diff==0:
                print("vertical")
                x_seq += generated_string1_x[P[i-1][0][0]]
                
            elif x_diff==0 and y_diff==1:
                print("horizontal")
                x_seq += "_"

        i+=1     
        
        print("x_seq",x_seq)
        print("y_seq",y_seq)

    

    print("x_seq",x_seq)
    print("y_seq",y_seq)
    return x_seq,y_seq

def reconstructUsingPVerTemp(generated_string1_x,generated_string2_y,P):

    print("P",P)
    x_seq = ""
    y_seq = ""



    # x_seq
    len_P = len(P)
    ind_P=0

    x_seq_loc = 0
    # for i < len_P:
    while ind_P < len_P:
        print(ind_P)
        print("eachPoint i",P[ind_P])
        if ind_P+1 != len_P:
            print("eachPoint i+1",P[ind_P+1])
            x = P[ind_P][0][0]
            y = P[ind_P][0][1]

            x_next = P[ind_P+1][0][0]
            y_next = P[ind_P+1][0][1]

            x_diff=x_next-x
            y_diff=y_next-y
        else:
            x_seq += generated_string1_x[x_seq_loc]
            x_seq_loc += 1
            print("x_seq",x_seq)
            

        if ind_P == 0 :
            if x==0 and y==0 and x_diff == 0 and y_diff == 0:
                x_seq += ""
                x_seq_loc += 0

                
                
            elif x == 0 and y == 0 and x_diff == 0 and y_diff == 1:
                x_seq += ""
                x_seq_loc += 0
                
            elif x == 0 and y == 0 and x_diff == 1 and y_diff == 0:
                x_seq += generated_string1_x[x]
                x_seq_loc += 1

                

        else:
            if x_diff==1 and y_diff==1 and x_seq_loc < len(generated_string1_x):
                print("diagonal")
                # x_seq += generated_string1_x[x]
                # y_seq += generated_string2_y[y]
                x_seq += generated_string1_x[x_seq_loc]
                x_seq_loc += 1
                
            elif x_diff==1 and y_diff==0 and x_seq_loc < len(generated_string1_x):
                print("vertical")
                x_seq += generated_string1_x[x_seq_loc]
                x_seq_loc += 1
                
            elif x_diff==0 and y_diff==1 and x_seq_loc < len(generated_string1_x):
                print("horizontal")
                x_seq += "_"

        ind_P+=1



    # y_seq
    ind_P=0

    y_seq_loc = 0
    # for i < len_P:
    while ind_P < len_P:
        print(ind_P)
        print("eachPoint i",P[ind_P])
        if ind_P+1 != len_P:
            print("eachPoint i+1",P[ind_P+1])
            x = P[ind_P][0][0]
            y = P[ind_P][0][1]

            x_next = P[ind_P+1][0][0]
            y_next = P[ind_P+1][0][1]

            x_diff=x_next-x
            y_diff=y_next-y
        else:
            y_seq += generated_string2_y[y_seq_loc]
            y_seq_loc += 1
            print("y_seq",y_seq)
            

        if ind_P == 0 :
            if x==0 and y==0 and x_diff == 0 and y_diff == 0:
                y_seq += ""
                y_seq_loc += 0

                
                
            elif x == 0 and y == 0 and x_diff == 0 and y_diff == 1:
                y_seq += generated_string2_y[y]
                
            elif x == 0 and y == 0 and x_diff == 1 and y_diff == 0:
                y_seq += ""
                y_seq_loc += 0

                

        else:
            if x_diff==1 and y_diff==1 and y_seq_loc < len(generated_string2_y):
                print("diagonal")
                # x_seq += generated_string1_x[x]
                # y_seq += generated_string2_y[y]
                y_seq += generated_string2_y[y_seq_loc]
                y_seq_loc += 1
                
            elif x_diff==1 and y_diff==0 and y_seq_loc < len(generated_string2_y):
                print("vertical")
                y_seq += "_"
                
            elif x_diff==0 and y_diff==1 and y_seq_loc < len(generated_string2_y):
                
                y_seq += generated_string2_y[y_seq_loc]
                y_seq_loc += 1

        ind_P+=1       
        
        # print("x_seq",x_seq)
        print("y_seq",y_seq)

    

    print("x_seq",x_seq)
    print("y_seq",y_seq)
    return x_seq,y_seq

def reconstructUsingPVerTempReversed(generated_string1_x,generated_string2_y,P):
    print("P",P)
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
        print(ind_P)
        print("eachPoint i",P[ind_P])
        
        

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
                


            print("x_diff",x_diff)
            print("y_diff",y_diff)
        else:

            x = P[ind_P][0][0]
            y = P[ind_P][0][1]

            x_prev = P[ind_P-1][0][0]
            y_prev = P[ind_P-1][0][1]

            x_diff=x-x_prev
            y_diff=y-y_prev


            if x_diff ==1 and y_diff == 1:
                print("diagonal")
                # add generated_string1_x[x_seq_loc] to most left of x_seq
                x_seq = generated_string1_x[x_seq_loc] + x_seq
                # x_seq += generated_string1_x[x_seq_loc]
                x_seq_loc -= 1
                y_seq = generated_string2_y[y_seq_loc] + y_seq
                y_seq_loc -= 1
                
            elif x_diff ==1 and y_diff == 0:
                print("vertical")
                x_seq = generated_string1_x[x_seq_loc] + x_seq
                x_seq_loc -= 1

                y_seq = "_" + y_seq
            
            elif x_diff ==0 and y_diff == 1:
                print("horizontal")
                x_seq = "_" + x_seq
                y_seq = generated_string2_y[y_seq_loc] + y_seq
                y_seq_loc -= 1
        
            print("x_seq",x_seq)
            print("y_seq",y_seq)
    print("x_seq",x_seq)
    print("y_seq",y_seq)



        


def reconstructUsingPVerTwo(generated_string1_x,generated_string2_y,P):

    # pattern lengths
    M = len(generated_string1_x)
    N = len(generated_string2_y)
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
     
    # for coord in P:
    # reverse order for coord in P:
    for coord in reversed(P):
        pos_x=coord[0][0]
        pos_y=coord[0][1]

    len_P=len(P)
    i_P=0
    while i_P < len_P:
        print(i_P)


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
    print(f"Similarity in gene alignment = {OPT[M][N]}")
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


    # output OPT[M][N], x_seq, y_seq to output_mine.txt
    with open("output_mine.txt", "w") as f:
        f.write(f"Similarity in gene alignment = {OPT[M][N]}\n")
        f.write("The aligned genes are:\n")
        f.write(f"X seq: {x_seq}\n")
        f.write(f"Y seq: {y_seq}\n")
            
            



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
    # generated_string2_y ="CACAAT"
    # generated_string2_y ="CA"


    # generated_string1_x =""
    # generated_string2_y ="C"

    # generated_string1_x ="C"
    # generated_string2_y =""

    # generated_string1_x ="CG"
    # generated_string2_y ="CA"

    generated_string1_x ="ATC"
    generated_string2_y ="AAT"

    # generated_string1_x ="CGCATC"
    # generated_string2_y ="CACAAT"




    print("Generated String y: ", generated_string2_y)


    # 
    len_x=len(generated_string1_x)
    len_y=len(generated_string2_y)


    # Divide-and-Conquer-Alignment(X ,Y )

    # divConq_align(generated_string1_x,generated_string2_y,[0,len_x],[0,len_y])
    divConq_align(generated_string1_x,generated_string2_y,[0,len_x],[0,len_y])
   
    


    # print("hello")
    # print("P",P)
    # sort P by x   
    # P.sort(key=lambda x: x[0][0])
    # print("P",P)






    # insert into the P
    if [(0,0)] not in P:
        P.append([(0,0)])
    if [(len_x,len_y)] not in P:
        P.append([(len_x,len_y)])
    # P.insert(0,[(len_x,len_y)])
    # sort P by x and y
    P.sort(key=lambda x: (x[0][0],x[0][1]))
    print("P",P)

    

    # reconstruct the alignment
    # reconstructUsingP(generated_string1_x,generated_string2_y,P)
    # reconstructUsingPVerTemp(generated_string1_x,generated_string2_y,P)
    reconstructUsingPVerTempReversed(generated_string1_x,generated_string2_y,P)
    # reconstructUsingPVerTwo(generated_string1_x,generated_string2_y,P)
    # print(f"Y seq: {y_seq}")
    # print(f"X seq: {x_seq}")

        
    

    # get the only x element in P
    P_x = [x[0][0] for x in P]
    # set P_x to set
    P_x_set = set(P_x)
    # list P_x_set  
    P_x_set_list = list(P_x_set)
    # show length of P_x_set_list
    print("X",len(P_x_set_list))


    # get the only x element in P
    P_x = [x[0][1] for x in P]
    # set P_x to set
    P_x_set = set(P_x)
    # list P_x_set  
    P_x_set_list = list(P_x_set)
    # show P_x_set_list
    print("P_x_set_list",P_x_set_list)
    print("Y",len(P_x_set_list))

    # # generate list which has 0 to 64
    # list_0_to_64 = [x for x in range(0,65)]
    # # get the difference between list_0_to_64 and P_x_set_list
    # diff_list = list(set(list_0_to_64) - set(P_x_set_list))



    







def call_algorithm(input_file, output_file):
    # Read the input file
    # file = open("SampleTestCases/input3.txt", "r")

    # naive approach
    naiveMethod(input_file, output_file)
    


    # space efficient approach
    spaceEfficientMethod(input_file, output_file) 

     





if __name__ == "__main__":

    # `python3 basic_3.py input.txt output.txt`
    
    # input_file = sys.argv[1]
    # output_file = sys.argv[2]


    file_number = 1
    input_file = "input"+str(file_number)+".txt"
    output_file = "output"+str(file_number)+".txt"

    # call_algorithm(input_file, output_file)
    print(time_wrapper(input_file, output_file))

    
    print(process_memory())