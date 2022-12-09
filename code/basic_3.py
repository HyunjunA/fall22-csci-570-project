import string
import sys
from resource import *
import time
import psutil

def process_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_consumed = int(memory_info.rss/1024)
    return memory_consumed

def time_wrapper(lines):
    start_time = time.time()
    alignment1, alignment2, cost  = basic_algo(lines)
    end_time = time.time()
    time_taken = (end_time - start_time)*1000
    return time_taken, cost, alignment1, alignment2

def get_optimal_solution(x:str, y:str):
    """
    Function to get the minimum penalty, and aligned X and Y strings
 
    :param x: pattern X
    :param y: pattern Y
    """
    delta:int = 30

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


    # lengths of input strings
    m = len(x)
    n = len(y)
     
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
                opt[i-1][j-1] + alpha(x[i-1],y[j-1]), 
                opt[i-1][j] + delta, 
                opt[i][j-1] + delta 
            )

    # constructing the optimal strings from top-down of opt table
    x_final = []
    y_final = []
 
    while m!=0 and n!=0:
        if (x[m-1] == y[n-1]) or (opt[m-1][n-1] + alpha(x[m-1],y[n-1])) == opt[m][n]:       
            x_final.append(x[m-1])
            y_final.append(y[n-1])
            m-=1
            n-=1
         
        elif (opt[m][n-1] + delta) == opt[m][n]:       
            x_final.append('_')
            y_final.append(y[n-1])
            n -= 1

        elif (opt[m-1][n] + delta) == opt[m][n]:
            x_final.append(x[m-1])
            y_final.append('_')
            m -= 1

    while m>0:
        x_final.append(x[m-1])
        m-=1

    while n>0:
        y_final.append(y[n-1])
        n-=1

    while len(x_final) < len(y_final):
        x_final.append('_')

    while len(y_final) < len(x_final):
        y_final.append('_')

    x_final.reverse()
    y_final.reverse()

    aligned_x = ''.join(x_final)
    aligned_y = ''.join(y_final)


    return aligned_x, aligned_y, opt[len(x)][len(y)]

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


def basic_algo(lines):
    """
    Process input files to generate the strings, then runs basic version 
    of the dynamic programming algorithm for sequence alignment
    """
    # get first string
    first_indices,i_s=get_indices(lines,0)
    generated_string1 = generate_string(lines[0].strip(), first_indices)
   
    # get second string
    second_indices,i_final=get_indices(lines,i_s)
    generated_string2 = generate_string(lines[i_s].strip(), second_indices)

    # returns optimal cost, aligned string x, and aligned string y
    return get_optimal_solution(generated_string1, generated_string2)
    
 
if __name__ == "__main__":

    # Read the input file
    with open(sys.argv[1], 'r') as infile:
        lines = infile.readlines()

    # run sequence alignment algo (wrapped by time_wrapper) and calculate memory
    elapsed_time, cost, alignment1, alignment2 = time_wrapper(lines)
    consumed_memory = process_memory()

    # write results to output file
    with open(sys.argv[2], 'w') as outfile:
        outfile.write(str(cost) + '\n')
        outfile.write(alignment1+ '\n')
        outfile.write(alignment2+ '\n')
        outfile.write(str(elapsed_time)+ '\n')
        outfile.write(str(consumed_memory))