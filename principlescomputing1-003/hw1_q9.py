def appendsums(lst):
    """
    Repeatedly append the sum of the current last three elements of lst to lst.
    """
    a = lst[-1] + lst[-2] + lst[-3]
    lst.append(a)
    
mylist = [0,1,2]
for i in range(25):
    appendsums(mylist)