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
    # initializing the table
    dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]
    
    #print(dp)

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

    #Optimal Alignment

    #Space-Efficient-Alignment
    space_eff_table = np.zeros([m+1,2], dtype=int) #int dp[m+1][n+1] = {0};    i = 1
    space_eff_table[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    while i <= m:
        j = 1
        space_eff_table[0][1] = i * pgap
        while j <= n:
            space_eff_table[i][1] = min(space_eff_table[i - 1][0] + pxy(x[i-1],y[j-1]),
                            space_eff_table[i - 1][1] + pgap,
                            space_eff_table[i][0] + pgap)                
            j += 1
            #Move column 1 of B to column ) to make room for the next iteration
            #Update B[i,0]=B[i,1] for each i
            space_eff_table[i][0] = space_eff_table[i][1]
        i += 1  
    print(space_eff_table)  
    #Backward-Space-Efficient-Alignment
    
    g = np.zeros([m+1,n+1], dtype=int) #int dp[m+1][n+1] = {0};
    print(g)
    i = 1
    while i <= m:
        j = 1
        while j <= n:
            g[i][j] = min(g[i + 1][j + 1] + pxy(x[i],y[j]),
                            g[i + 1][j] + pgap,
                            g[i][j + 1] + pgap)                
            j += 1
        i += 1
    print(g)
    
def test_get_minimum_penalty():
    """
    Test the get_minimum_penalty function
    """
    # input strings
    gene1 = "ACTG"
    gene2 = "ATGA"

    # calling the function to calculate the result
    get_minimum_penalty(gene1, gene2)
test_get_minimum_penalty()