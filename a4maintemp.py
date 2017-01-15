from a2maintemp import paired_t_test
import numpy as np
import sys

# read in data and command-line arguments, and compute X and Y
# processing input files begin
'''
Storing x and y values separately in two files: iris_X.txt and iris_Y.txt
'''

f = open(sys.argv[1],'r').read().strip().split("\"")
out_X = open('iris_X.txt','w')
out_Y = open('iris_Y.txt','w')

Y = f[1::2]
X = f[0::2]
for i in X:
    i = i.replace("\n","")
    i = i.strip(",")
    out_X.write(i)
    out_X.write("\n")

out_X.close()

for i in Y:
    out_Y.write(i)
    out_Y.write("\n")
out_Y.close()

#processing input files end 

Y = np.loadtxt('iris_Y.txt',delimiter="\n",dtype={'names':['e'],'formats':['S20']})
X = np.loadtxt('iris_X.txt',delimiter = ",")
K = int(sys.argv[2])
alpha = float(sys.argv[3])


paired_t_test(X, Y, K, alpha)
