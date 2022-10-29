import string
import sys
from resource import *
import time
import psutil

import numpy as np
from pyparsing import Char

"""
def process_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_consumed = int(memory_info.rss/1024)
    return memory_consumed

def time_wrapper():
    start_time = time.time()
    test_get_minimum_penalty()
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken    
"""

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
    # initialising the table
    dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]



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
        elif sec_char == "T":
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
         
        elif (dp[i - 1][j] + pgap) == dp[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord('_')
            xpos -= 1
            ypos -= 1
            i -= 1
         
        elif (dp[i][j - 1] + pgap) == dp[i][j]:       
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
    print(f"Minimum Penalty in aligning the genes = {dp[m][n]}")
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
 
def test_get_minimum_penalty():
    """
    Test the get_minimum_penalty function
    """
    # input strings
    gene1 = "AAAAAAGTCGTCAGTCGTCAAGTCGTCAGTCGTCAAAGTCGTCAGTCGTCAAGTCGTCAGTCGTCAAAAGTCGTCAGTCGTCAAGTCGTCAGTCGTCAAAGTCGTCAGTCGTCAAGTCGTCAGTCGTC"
    gene2 = "TATATATATATACGCGTACGCGTATACGCGTACGCGTATATACGCGTACGCGTATACGCGTACGCGTATATATACGCGTACGCGTATACGCGTACGCGTATATACGCGTACGCGTATACGCGTACGCG"

    # calling the function to calculate the result
    get_minimum_penalty(gene1, gene2)
    #print("Memory in Kilobytes: " + str(process_memory()))
 
test_get_minimum_penalty()
#print("Time taken: " + str(time_wrapper()))
 
# This code is contributed by wilderchirstopher. https://www.geeksforgeeks.org/sequence-alignment-problem/