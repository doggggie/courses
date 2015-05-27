"""
Merge function for 2048 game.
"""

def merge(line):
    """
    Function that merges a single row or column in 2048.
    """
    merged_line = [0] * len(line)
    prev_elem = -1
    merged_idx = 0
    for elem in line:
        if elem == 0:
            continue
        elif elem == prev_elem:
            merged_line[merged_idx - 1] = 2 * elem
            prev_elem = -1
            continue
        else: # elem != prev_elem
            merged_line[merged_idx] = elem
            prev_elem = elem
            merged_idx += 1
        
    return merged_line
