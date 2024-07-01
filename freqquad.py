import math
import random

def getFreq(a, b, c, d):
    ab = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) + random.uniform(1e-7, 1e-3)
    bc = math.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2) + random.uniform(1e-7, 1e-3)
    cd = math.sqrt((c[0] - d[0])**2 + (c[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    ad = math.sqrt((a[0] - d[0])**2 + (a[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    ac = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2) + random.uniform(1e-7, 1e-3)
    bd = math.sqrt((b[0] - d[0])**2 + (b[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    
    freq_dict = {}
    
    if ab + cd < ac + bd < ad + bc:
        freq_dict[f'{a, b}'] = 5
        freq_dict[f'{b, c}'] = 1
        freq_dict[f'{c, d}'] = 5
        freq_dict[f'{a, d}'] = 1
        freq_dict[f'{a, c}'] = 3
        freq_dict[f'{b, d}'] = 3
        
    elif ab + cd < ad + bc < ac + bd:
        freq_dict[f'{a, b}'] = 5
        freq_dict[f'{b, c}'] = 3
        freq_dict[f'{c, d}'] = 5
        freq_dict[f'{a, d}'] = 3
        freq_dict[f'{a, c}'] = 1
        freq_dict[f'{b, d}'] = 1
        
    elif ac + bd < ab + cd < ad + bc:
        freq_dict[f'{a, b}'] = 3
        freq_dict[f'{b, c}'] = 1
        freq_dict[f'{c, d}'] = 3
        freq_dict[f'{a, d}'] = 1
        freq_dict[f'{a, c}'] = 5
        freq_dict[f'{b, d}'] = 5
        
    elif ac + bd < ad + bc < ab + cd:
        freq_dict[f'{a, b}'] = 1
        freq_dict[f'{b, c}'] = 3
        freq_dict[f'{c, d}'] = 1
        freq_dict[f'{a, d}'] = 3
        freq_dict[f'{a, c}'] = 5
        freq_dict[f'{b, d}'] = 5
        
    elif ad + bc < ab + cd < ac + bd:
        freq_dict[f'{a, b}'] = 3
        freq_dict[f'{b, c}'] = 5
        freq_dict[f'{c, d}'] = 3
        freq_dict[f'{a, d}'] = 5
        freq_dict[f'{a, c}'] = 1
        freq_dict[f'{b, d}'] = 1
        
    elif ad + bc < ac + bd < ab + cd:
        freq_dict[f'{a, b}'] = 1
        freq_dict[f'{b, c}'] = 5
        freq_dict[f'{c, d}'] = 1
        freq_dict[f'{a, d}'] = 5
        freq_dict[f'{a, c}'] = 3
        freq_dict[f'{b, d}'] = 3
        
    return freq_dict
