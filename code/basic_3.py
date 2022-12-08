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
    call_algorithm(input_file,output_file)
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken



#  DEFINE GLOBAL VARIABLES
P=[]



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

def naiveMethod(input_file,output_file):
    input="../SampleTestCases/"+input_file
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



    

    # define OPT as a 2D array  
    OPT = [[0 for x in range(len(generated_string2_y)+1)] for y in range(len(generated_string1_x)+1)]
    
   

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
    # print(f"Similarity in gene alignment = {OPT[M][N]}")
    # print("The aligned genes are:")   
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



    
    #calculate final alignment cost
    len_final_string = len(x_seq)
    final_cost = 0
    #iterate through the x and y string comparing the val of each element
    for x in range(len_final_string):
        if x_seq[x] == "_" or y_seq[x] == "_":
            final_cost += 30 #make delta global var
        else:
            final_cost += alpha(x_seq[x],y_seq[x])
    print("final cost",final_cost)

    


    # output 
    # 1. Cost of the alignment (Integer)
    # 2. First string alignment ( Consists of A, C, T, G, _ (gap) characters)
    # 3. Second string alignment ( Consists of A, C, T, G, _ (gap) characters )
    # 4. Time in Milliseconds (Float)
    # 5. Memory in Kilobytes (Float)


    # open output_file and write and add

    output = open(output_file, "a")
    output.write(str(final_cost) + "\n")
    output.write(x_seq + "\n")
    output.write(y_seq + "\n")

    # close output_file
    output.close()




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

    print([(x_range[0]+i, y_range[0]+j)])
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

    # if len(generated_string1_x) == 0:
    #     return OPT[0]

    return OPT[1]



    
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


    output_file = "output1.txt"
    output="SampleTestCases/"+output_file
    file = open(output, "r")

    # get string from file
    string = file.read()
    
    # parse string using space
    string = string.split("\n")



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
            print("X_SEQ",string[1])
            print("y_seq",y_seq)
            print("Y_SEQ",string[2])
            print("Hellp")


    # x_seq = "_A_CA_CACT__G__A_C_TAC_TGACTG_GTGA__C_TACTGACTGGACTGACTACTGACTGGTGACTACT_GACTG_G"
    # y_seq = "TATTATTA_TACGCTATTATACGCGAC_GCG_GACGCGTA_T_AC__G_CT_ATTA_T_AC__GCGAC_GC_GGAC_GCG"

    #calculate final alignment cost
    len_final_string = len(x_seq)
    final_cost = 0
    #iterate through the x and y string comparing the val of each element
    for x in range(len_final_string):
        if x_seq[x] == "_" or y_seq[x] == "_":
            final_cost += 30 #make delta global var
        else:
            final_cost += alpha(x_seq[x],y_seq[x])
    print("final cost",final_cost)


    print("x_seq",x_seq)
    print("y_seq",y_seq)

    

    # read 

    
    
        


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
            
            









def call_algorithm(input_file,output_file):
    # Read the input file
    # file = open("SampleTestCases/input3.txt", "r")

    # naive approach
    naiveMethod(input_file,output_file)
    



     





if __name__ == "__main__":

    # `python3 basic_3.py input.txt output.txt`
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # print("input_file",input_file)


    # file_number = 1
    # input_file = "input"+str(file_number)+".txt"
    # output_file = "output"+str(file_number)+".txt"

    # call_algorithm(input_file, output_file)
    time_calcu=time_wrapper(input_file,output_file)
    pro_mom=process_memory()

    # open output_file
    with open(output_file, "a") as f:
        # 4. Time in Milliseconds (Float)
        f.write(f"{time_calcu}\n")
        # 5. Memory in Kilobytes (Float)
        f.write(f"{pro_mom}\n")
    
    # close output_file
    f.close()
    

    
    


    
