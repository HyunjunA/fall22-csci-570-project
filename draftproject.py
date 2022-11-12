import string
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
    cost, alignment1, alignment2 = test_get_minimum_penalty(lines)
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken, cost, alignment1, alignment2

def get_minimum_penalty(x:str, y:str):
    """
    Function to find out the minimum penalty
 
    :param x: pattern X
    :param y: pattern Y
    :param pxy: penalty of mis-matching the characters of X and Y, mismatch_penalty
    :param pgap: penalty of a gap between pattern elements, gap_penalty
    """
    pgap:int = 30

    # initializing variables
    i = 0
    j = 0
     
    # pattern lengths
    m = len(x)
    n = len(y)
     
    # table for storing optimal substructure answers
    dp = np.zeros([m+1,n+1], dtype=int) #int dp[m+1][n+1] = {0};
    # initializing the table
    dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]
    print(dp)

    # table for storing Delta values = pxy
    delta_vals = np.zeros([4,4], dtype=int)
    delta_vals[0,0] = 0
    delta_vals[0,1] = 110
    delta_vals[0,2] = 48
    delta_vals[0,3] = 94
    delta_vals[1,0] = 110
    delta_vals[1,1] = 0
    delta_vals[1,2] = 118
    delta_vals[1,3] = 48
    delta_vals[2,0] = 48
    delta_vals[2,1] = 118
    delta_vals[2,2] = 0
    delta_vals[2,3] = 110
    delta_vals[3,0] = 94
    delta_vals[3,1] = 48
    delta_vals[3,2] = 110
    delta_vals[3,3] = 0

    #Calculate the mismatch value alpha x y 
    def pxy(first_char:str, sec_char:str):
        p_x = 0
        p_y = 0
        if first_char == "A":
            p_x = 0
        elif first_char == "C":
            p_x = 1
        elif first_char == "G":
            p_x = 2
        elif first_char == "T":
            p_x = 3
        if sec_char == "A":
            p_y = 0
        elif sec_char == "C":
            p_y = 1
        elif sec_char == "G":
            p_y = 2
        elif sec_char == "T":
            p_y = 3
        return delta_vals[p_x,p_y]
 
    # calculating the minimum penalty
    i = 1
    while i <= m:
        j = 1
        while j <= n:
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + pxy(x[i-1],y[j-1]),
                                dp[i - 1][j] + pgap,
                                dp[i][j - 1] + pgap)                
            j += 1
        i += 1
        #print(dp)
    #print(dp) 
    # Reconstructing the solution
    l = n + m   # maximum possible length
    i = m
    j = n
     
    xpos = l
    ypos = l
 
    # Final answers for the respective strings
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

        elif (dp[i - 1][j - 1] + pxy(x[i-1],y[j-1])) == dp[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
         
        elif (dp[i][j - 1] + pgap) == dp[i][j]:       
            xans[xpos] = ord('_')
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            j -= 1

        elif (dp[i - 1][j] + pgap) == dp[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord('_')
            xpos -= 1
            ypos -= 1
            i -= 1
         
    # print(xans)
    # print(x)
    # #print(xpos)
    # print(yans)
    # print(y)
     
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

    #print(xans)
    #print(yans)
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
        #print(xans)
        #print(yans)
 
    # Printing the final answer
    # print(f"Cost of the alignment {dp[m][n]}")
    # X
    i = id
    x_seq = ""
    while i <= l:
        x_seq += chr(xans[i])
        i += 1
    # int(f"First string alignment {x_seq}")
 
    # Y
    i = id
    y_seq = ""
    while i <= l:
        y_seq += chr(yans[i])
        i += 1
    # print(f"Second string alignment {y_seq}")

    return dp[m][n], x_seq, y_seq

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

def test_get_minimum_penalty(lines):
    """
    Test the get_minimum_penalty function
    """
    # get first string
    first_indices,i_s=get_indices(lines,0)
    generated_string1 = generate_string(lines[0].strip(), first_indices)
   
    # get second string
    second_indices,i_final=get_indices(lines,i_s)
    generated_string2 = generate_string(lines[i_s].strip(), second_indices)

    # calling the function to calculate the result
    return get_minimum_penalty(generated_string1, generated_string2)
    
 
if __name__ == "__main__":

    # Read the input file
    with open(sys.argv[1], 'r') as infile:
        lines = infile.readlines()

    # run sequence alignment algo (wrapped by time_wrapper)
    elapsed_time, cost, alignment1, alignment2 = time_wrapper(lines)
    consumed_memory = process_memory()

    print(cost)
    print(alignment1)
    print(alignment2)
    print(elapsed_time)
    print(consumed_memory)

    with open(sys.argv[2], 'w') as outfile:
        outfile.write(str(cost) + '\n')
        outfile.write(alignment1+ '\n')
        outfile.write(alignment2+ '\n')
        outfile.write(str(elapsed_time)+ '\n')
        outfile.write(str(consumed_memory)+ '\n')


 
# This code is contributed by wilderchirstopher. https://www.geeksforgeeks.org/sequence-alignment-problem/
