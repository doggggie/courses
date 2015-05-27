import re
import numpy as np

Eins = {}
Eouts = {}

with open("hw4q11.txt", "r") as infile:
    for line in infile:
        # M 6 Ein [ 0.96003834] Eout [ 0.94349469]
        m = re.search(r"M\s+(\d+)\s+Ein\s+\[\s+([\d\.]+)\]\s+Eout\s+\[\s+([\d\.]+)\]", line)

        if m:
            M = int(m.group(1))
            Ein = float(m.group(2))
            Eout = float(m.group(3))
        
            if M not in Eins:
                Eins[M] = []
                
            if M not in Eouts:
                Eouts[M] = [] 

            Eins[M].append(Ein)
            Eouts[M].append(Eout)
            
for M in np.sort(Eins.keys()):
    print "M", M, "avgEin", sum(Eins[M])/float(len(Eins[M])),\
        "avgEout", sum(Eouts[M])/float(len(Eouts[M])), \
        "size", len(Eins[M])