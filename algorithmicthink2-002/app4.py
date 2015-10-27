"""
Provide code and solution for Application 4
"""

DESKTOP = True

import math
import random
#import urllib2
import string

import matplotlib.pyplot as plt
#import alg_project4_solution as student
    

# URLs for data files
PAM50_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_PAM50.txt"
HUMAN_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_HumanEyelessProtein.txt"
FRUITFLY_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_FruitflyEyelessProtein.txt"
CONSENSUS_PAX_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_ConsensusPAXDomain.txt"
WORD_LIST_URL = "http://storage.googleapis.com/codeskulptor-assets/assets_scrabble_words3.txt"

PAM50_FILE = "alg_PAM50.txt"
HUMAN_EYELESS_FILE = "alg_HumanEyelessProtein.txt"
FRUITFLY_EYELESS_FILE = "alg_FruitflyEyelessProtein.txt"
CONSENSUS_PAX_FILE = "alg_ConsensusPAXDomain.txt"
WORD_LIST_FILE = "assets_scrabble_words3.txt"


###############################################
# provided code

def read_scoring_matrix(filename):
    """
    Read a scoring matrix from the file named filename.  

    Argument:
    filename -- name of file containing a scoring matrix

    Returns:
    A dictionary of dictionaries mapping X and Y characters to scores
    """
    scoring_dict = {}
    #scoring_file = urllib2.urlopen(filename)
    scoring_file = open(filename)
    ykeys = scoring_file.readline()
    ykeychars = ykeys.split()
    for line in scoring_file.readlines():
        vals = line.split()
        xkey = vals.pop(0)
        scoring_dict[xkey] = {}
        for ykey, val in zip(ykeychars, vals):
            scoring_dict[xkey][ykey] = int(val)
    return scoring_dict




def read_protein(filename):
    """
    Read a protein sequence from the file named filename.

    Arguments:
    filename 
    -- name of file containing a protein sequence

    Returns:
    A string representing the protein
    """
    #protein_file = urllib2.urlopen(filename)
    protein_file = open(filename)
    protein_seq = protein_file.read()
    protein_seq = protein_seq.rstrip()
    return protein_seq


def read_words(filename):
    """
    Load word list from the file named filename.

    Returns a list of strings.
    """
    # load assets
    #word_file = urllib2.urlopen(filename)
    word_file = open(filename)
    
    # read in files as string
    words = word_file.read()
    
    # template lines and solution lines list of line string
    word_list = words.split('\n')
    print "Loaded a dictionary with", len(word_list), "words"
    return word_list



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
    
def compute_agree_percent(seq_x, seq_y):
    if len(seq_x) != len(seq_y) or len(seq_x) == 0:
        return None
    
    count = 0    
    for i in range(len(seq_x)):
        if seq_x[i] == seq_y[i]:
            count += 1
    return count / float(len(seq_x))

def generate_random_protein(size, scoring_matrix):
    res = ""
    letters = scoring_matrix.keys()
    letters.remove("-")
    for i in range(size):
        rn = random.randint(0, len(letters) - 1)
        res += letters[rn]
    return res        

def generate_null_distribution(seq_x, seq_y, scoring_matrix, num_trials):
    distribution = {}
    for i in range(num_trials):
        if i % 10 == 0:
            print "Doing trial #", i
        y_list = list(seq_y)
        random.shuffle(y_list)
        new_y = ""
        for letter in y_list:
            new_y = new_y + letter
        align_mat = compute_alignment_matrix(seq_x, new_y, scoring_matrix, False)
        score, align_x, align_y = compute_local_alignment(seq_x, new_y, 
                                    scoring_matrix, align_mat)
        if score not in distribution:
            distribution[score] = 1
        else:
            distribution[score] += 1
    return distribution

def check_spelling(checked_word, dist, word_list):
    len_checked = len(checked_word)
    ret_list = []
    sc_matrix = build_scoring_matrix(string.ascii_lowercase, 2, 1, 0)
    for word in word_list:
        align_matrix = compute_alignment_matrix(checked_word, word, sc_matrix, True)
        score_edit, alx, aly = compute_global_alignment(checked_word, word, sc_matrix, align_matrix)
        edit_dist = len_checked + len(word) - score_edit
        if edit_dist <= dist:
            ret_list.append(word)
    return ret_list
                
human_protein = read_protein(HUMAN_EYELESS_FILE)
fly_protein = read_protein(FRUITFLY_EYELESS_FILE)
score_mat = read_scoring_matrix(PAM50_FILE)


alm = compute_alignment_matrix(human_protein, fly_protein, score_mat, False)
score1, align_human, align_fly = compute_local_alignment(human_protein, fly_protein, 
                                    score_mat, alm)
                                    
print "score", score1
print "human alignment", align_human
print "fly alignment", align_fly                                    

human_nodash = ""
for letter in align_human:
    if letter != "-":
        human_nodash += letter
fly_nodash = ""
for letter in align_fly:
    if letter != "-":
        fly_nodash += letter
    
dax_protein = read_protein(CONSENSUS_PAX_FILE)    
    
alm21 = compute_alignment_matrix(human_nodash, dax_protein, score_mat, True)
score21, align_human21, align_dax21 = compute_global_alignment(human_nodash, dax_protein, 
                                    score_mat, alm21)
percent21 = compute_agree_percent(align_human21, align_dax21)
alm22 = compute_alignment_matrix(fly_nodash, dax_protein, score_mat, True)
score22, align_fly22, align_dax22 = compute_global_alignment(fly_nodash, dax_protein, 
                                    score_mat, alm22)
percent22 = compute_agree_percent(align_fly22, align_dax22)
print "human agree pencentage", percent21
print "fly agree percentage", percent22


random_protein = generate_random_protein(500, score_mat)
alm3 = compute_alignment_matrix(human_protein, random_protein, score_mat, False)
score3, align_human3, align_random3 = compute_local_alignment(human_protein, random_protein, 
                                           score_mat, alm3)
print "score3", score3
random_nodash = ""
for letter in align_random3:
    if letter != "-":
        random_nodash += letter
alm31 = compute_alignment_matrix(random_nodash, dax_protein, score_mat, True)
score31, align_random31, align_dax31 = compute_global_alignment(random_nodash, dax_protein, 
                                    score_mat, alm31)
percent31 = compute_agree_percent(align_random31, align_dax31)
print "score31", score31, "percent31", percent31


# Q4
#distribution = generate_null_distribution(human_protein, fly_protein, score_mat, 1000)
distribution = {
 37: 1,
 38: 1,
 39: 3,
 40: 5,
 41: 11,
 42: 18,
 43: 27,
 44: 37,
 45: 51,
 46: 47,
 47: 71,
 48: 84,
 49: 54,
 50: 69,
 51: 65,
 52: 62,
 53: 59,
 54: 49,
 55: 36,
 56: 41,
 57: 36,
 58: 28,
 59: 27,
 60: 21,
 61: 11,
 62: 11,
 63: 4,
 64: 10,
 65: 15,
 66: 9,
 67: 5,
 68: 8,
 69: 3,
 70: 2,
 71: 2,
 72: 2,
 73: 2,
 74: 2,
 75: 2,
 76: 1,
 77: 2,
 79: 1,
 80: 1,
 81: 2,
 82: 1,
 83: 1}

nd = {}
sum_distr = sum(distribution.values())
for k in distribution.keys():
    nd[k] = distribution[k] / float(sum_distr)
    
plt.figure()
plt.bar(nd.keys(), nd.values())
plt.hold(True)
plt.title("Score distribution")    
plt.xlabel("Score")
plt.ylabel("Distribution")
plt.legend(loc=1)
plt.show()

sumscore = 0
sumscore2 = 0
sumcount = 0
for score, count in distribution.items():
    sumscore += score * count
    sumscore2 += score ** 2 * count
    sumcount += count
mu = sumscore / float(sumcount)
sigma = math.sqrt(sumscore2 / float(sumcount) - mu ** 2)
zscore = (score1 - mu) / sigma
print "mu, sigma", mu, sigma, "z-score", zscore


# Q8
word_list = read_words(WORD_LIST_FILE)
mylist = check_spelling('humble', 1, word_list)
print "humble:"
for word in mylist:
    print word, ",",
print "\n"
mylist = check_spelling('firefly', 2, word_list)
print "firefly:"
for word in mylist:
    print word, ",", 
print "\n"