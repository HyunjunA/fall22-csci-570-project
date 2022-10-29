import numpy as np
from pyparsing import Char

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

print(delta_vals)


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
    print(delta_vals[p_x,p_y])
    #return delta_vals[p_x,p_y]

pxy("A","G") #48
pxy("C","A") #110
pxy("G","C") #118
pxy("A","A") #0
pxy("C","C") #0