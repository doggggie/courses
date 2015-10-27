"""
build_scoring_matrix
compute_alignment_matrix
compute_global_alignment
compute_local_alignment
"""

def build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score):
    """
    Input: 
        alphabet: a set of characters
        diag_score, off_diag_score, and dash_score: three scores 
    Output:
        a dictionary of dictionaries whose entries are indexed 
        by pairs in alphabet plus '-'. Score for dash indexed entry
        is dash_score, score for other diagonal entries is diag_score,
        score for other off-diagonal entries is off_diag_score.
    """
    subdict = {}
    for letter in alphabet:
        subdict[letter] = off_diag_score
    subdict['-'] = dash_score
    retdict = {}
    for letter in alphabet:
        retdict[letter] = dict(subdict)
        retdict[letter][letter] = diag_score
    retdict['-'] = {'-': dash_score}
    for letter in alphabet:
        retdict['-'][letter] = dash_score
    return retdict
    
def compute_alignment_matrix(seq_x, seq_y, scoring_matrix, global_flag):
    """
    Input:
        two sequences seq_x and seq_y
        scoring matrix scoring_matrix
        global_flag (boolean) 
    Output: 
        alignment matrix for seq_x and seq_y. 
    """
    mat = []
    submat = [None] * (len(seq_y) + 1)
    for dummy in range(len(seq_x) + 1):
        mat.append(list(submat))
    
    mat[0][0] = 0
    for idx_x in range(1, len(seq_x) + 1):
        mat[idx_x][0] = mat[idx_x - 1][0] + scoring_matrix[seq_x[idx_x - 1]]['-']
        if not global_flag and mat[idx_x][0] < 0:
            mat[idx_x][0] = 0
    for idx_y in range(1, len(seq_y) + 1):
        mat[0][idx_y] = mat[0][idx_y - 1] + scoring_matrix['-'][seq_y[idx_y - 1]]
        if not global_flag and mat[0][idx_y] < 0:
            mat[0][idx_y] = 0
            
    for idx_x in range(1, len(seq_x) + 1):
        for idx_y in range(1, len(seq_y) + 1):
            sc11 = mat[idx_x-1][idx_y-1] + scoring_matrix[seq_x[idx_x-1]][seq_y[idx_y-1]]
            sc10 = mat[idx_x-1][idx_y] + scoring_matrix[seq_x[idx_x-1]]['-']
            sc01 = mat[idx_x][idx_y-1] + scoring_matrix['-'][seq_y[idx_y-1]]
            
            mat[idx_x][idx_y] = max(sc11, sc10, sc01)
            
            if not global_flag and mat[idx_x][idx_y] < 0:
                mat[idx_x][idx_y] = 0
        
    return mat
    
def compute_global_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix): 
    """
    Takes as input two sequences seq_x and seq_y whose elements 
    share a common alphabet with the scoring matrix scoring_matrix. 
    This function computes a global alignment of seq_x and seq_y 
    using the global alignment matrix alignment_matrix.
    The function returns a tuple of the form (score, align_x, align_y) 
    where score is the score of the global alignment align_x and 
    align_y. Note that align_x and align_y should have the same 
    length and may include the padding character '-'.
    """
    idx_x = len(seq_x)
    idx_y = len(seq_y)
    align_x = ''
    align_y = ''
    score = 0
    
    alm = alignment_matrix
    scm = scoring_matrix
    
    while idx_x != 0 and idx_y != 0:
        if alm[idx_x][idx_y] == alm[idx_x-1][idx_y-1] + scm[seq_x[idx_x-1]][seq_y[idx_y-1]]:
            align_x = seq_x[idx_x-1] + align_x
            align_y = seq_y[idx_y-1] + align_y
            score += scm[seq_x[idx_x-1]][seq_y[idx_y-1]]
            idx_x -= 1
            idx_y -= 1
        elif alm[idx_x][idx_y] == alm[idx_x-1][idx_y] + scm[seq_x[idx_x-1]]['-']:
            align_x = seq_x[idx_x-1] + align_x
            align_y = '-' + align_y
            score += scm[seq_x[idx_x-1]]['-']
            idx_x -= 1
        else:
            align_x = '-' + align_x
            align_y = seq_y[idx_y-1] + align_y
            score += scm['-'][seq_y[idx_y-1]]
            idx_y -= 1
    
    while idx_x != 0:
        align_x = seq_x[idx_x-1] + align_x
        align_y = '-' + align_y
        score += scm[seq_x[idx_x-1]]['-']
        idx_x -= 1
        
    while idx_y != 0:
        align_x = '-' + align_x
        align_y = seq_y[idx_y-1] + align_y
        score += scm['-'][seq_y[idx_y-1]]
        idx_y -= 1
        
    return score, align_x, align_y
    
def compute_local_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix): 
    """
    Takes as input two sequences seq_x and seq_y whose elements 
    share a common alphabet with the scoring matrix scoring_matrix.
    This function computes a local alignment of seq_x and seq_y using 
    the local alignment matrix alignment_matrix.
    """
    idx_x = len(seq_x)
    idx_y = len(seq_y)
    align_x = ''
    align_y = ''
    score = 0
    
    alm = alignment_matrix
    scm = scoring_matrix
    
    bestscore = 0
    bestx = 0
    besty = 0
    for idx_x in range(len(alm)):
        for idx_y in range(len(alm[idx_x])):
            if alm[idx_x][idx_y] > bestscore:
                bestscore = alm[idx_x][idx_y]
                bestx = idx_x
                besty = idx_y
    
    idx_x = bestx
    idx_y = besty
    
    score = bestscore
    
    while idx_x != 0 and idx_y != 0:
        if alm[idx_x][idx_y] == 0:
            break
            
        if alm[idx_x][idx_y] == alm[idx_x-1][idx_y-1] + scm[seq_x[idx_x-1]][seq_y[idx_y-1]]:
            align_x = seq_x[idx_x-1] + align_x
            align_y = seq_y[idx_y-1] + align_y
            #score += scm[seq_x[idx_x-1]][seq_y[idx_y-1]]
            idx_x -= 1
            idx_y -= 1
        elif alm[idx_x][idx_y] == alm[idx_x-1][idx_y] + scm[seq_x[idx_x-1]]['-']:
            align_x = seq_x[idx_x-1] + align_x
            align_y = '-' + align_y
            #score += scm[seq_x[idx_x-1]]['-']
            idx_x -= 1
        else:
            align_x = '-' + align_x
            align_y = seq_y[idx_y-1] + align_y
            #score += scm['-'][seq_y[idx_y-1]]
            idx_y -= 1
        
    return score, align_x, align_y
    