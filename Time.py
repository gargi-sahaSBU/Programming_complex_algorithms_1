import os
import time
import timeit

record_time = open('time.txt','w')
for K in [30,35,40,45,50,55,60,65,70,75,80]:
	record_time.write("K :")
	record_time.write(str(K))
	record_time.write("\n")
	start_time = timeit.default_timer()
	os.system("python3 a4main.py iris.txt " + str(K) + " 0.95")
	elapsed = timeit.default_timer() - start_time
	record_time.write("time taken by a4main: ")
	record_time.write(str(elapsed))

	record_time.write("\n")
	start_time = timeit.default_timer()
	os.system("python3 a4maintemp.py iris.txt " + str(K) + " 0.95")
	elapsed = timeit.default_timer() - start_time
	record_time.write("time taken by a4maintemp: ")
	record_time.write(str(elapsed))
	record_time.write("\n")
record_time.close()
