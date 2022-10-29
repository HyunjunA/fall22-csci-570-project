from email.mime import base
import sys
from resource import *
import time
import psutil
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
    print("base_string", base_string)
    for index in indices:
        base_string = base_string[:index+1] + base_string + base_string[index+1:]
        print("base_string", base_string)
    
    return base_string




def call_algorithm():
    # Read the input file
    file = open("SampleTestCases/input3.txt", "r")
    # file = open("SampleTestCases/input1.txt", "r")
    lines = file.readlines()
    file.close()
    # get first part of number string

    first_indices,i_s=get_indices(lines,0)
    generated_string1 = generate_string(lines[0].strip(), first_indices)
   
    # print("Generated String 1: ", generated_string1)

    second_indices,i_final=get_indices(lines,i_s)
    generated_string2 = generate_string(lines[i_s].strip(), second_indices)
   
    # print("Generated String 2: ", generated_string2)

if __name__ == "__main__":
    call_algorithm()