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
    print(dp)
    #print(dp[i-1][j-1]) #60 is the min cost of alignment

    #Space-Efficient-Alignment
    space_eff_table = np.zeros([m+1,2], dtype=int) #int dp[m+1][2] = {0};
    space_eff_table[0:(m+1),0] = [ i * pgap for i in range(m+1)]    

    j = 1
    while j <= n:
        i = 1
        space_eff_table[0][1] = j * pgap
        while i <= m:
            space_eff_table[i][1] = min(space_eff_table[i - 1][0] + pxy(x[i-1],y[j-1]),
                            space_eff_table[i - 1][1] + pgap,
                            space_eff_table[i][0] + pgap)                
            i += 1
        #Move column 1 of B to column 0 to make room for the next iteration
        #Update B[i,0]=B[i,1] for each i
        space_eff_table[:(m+1),0] = space_eff_table[:(m+1),1]
        j += 1  
    #print(space_eff_table)  
    
    #Backward-Space-Efficient-Alignmen
    #Prof says reverse strings and re-use alignment algo
    rev_x:str
    rev_y:str
    rev_x = x[::-1]
    rev_y = y[::-1]

    g = np.zeros([m+1,n+1], dtype=int) #int dp[m+1][n+1] = {0};
    g[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    g[0,0:(n+1)] = [ i * pgap for i in range(n+1)]
    #print(g)
    i = 1
    while i <= m:
        j = 1
        while j <= n:
            if rev_x[i - 1] == rev_y[j - 1]:
                g[i][j] = g[i - 1][j - 1]
            else:
                g[i][j] = min(g[i - 1][j - 1] + pxy(rev_x[i-1],rev_y[j-1]),
                                g[i - 1][j] + pgap,
                                g[i][j - 1] + pgap)                
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