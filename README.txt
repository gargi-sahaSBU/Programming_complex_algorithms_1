Name: Gargi Saha

a4main.py contains the final optimized program. 
It is executed by running "python a4main.py <file_containing_data_set> <K_value> <alpha_value>".
<file_containing_data_set> is iris.txt , K value is the desired rounds of bootstrap resampling and alpha is the confidence level.
a4main.py scans the dataset given as argument, to make two files: one to store all x values and the other to store corresponding class values for those x. It then creates two numpy arrays,X and Y from those files and passes them,along with K and alpha value, to the paired_t_test function in a2main.py.

The 'paired_t_test' function in a2main.py, computes K random pairs of training sets and test sets. We call the function 'bayes_params' on the training set in order to obtain the value of the mu contained in 'mean',the prior probability 'Pc', the covariance matrix 'cov_full' to be used in full Bayes classifier, and the variance matrix 'var_naive' to be used in naive Bayes classifier.

We then use the 'bayes_test' function to compute predicted class values for the test set using both full Bayes and naive Bayes classifier and we store the results separately in two files 'iris_pred_Y_full.txt' and 'iris_pred_Y_naive.txt'. Once the function returns the result, load the values from the files mentioned above, to two numpy arrays and compare each with the true class values to obtain the number of mismatches and the error rate of each classifier. We store the difference between error rate of each classifier in the diff array and they are used in the end, that is after K rounds of bootstrap resampling, to compute the z-score.

Finally the z score is used to decide between the null hypothesis(both classifiers have same performance) and the alternative hypothesis(both classifiers are different).

OPTIMIZATIONS AND COMPARISON:

a4maintemp.py and a2maintemp.py are the non-optimised programs. They can be invoked in the same way as a4main.py and a2main.py. The optimizations made include:
1) Line 100 in a2main.py: Used the numpy function setdiff1d to get set difference of two numpy arrays instead of running two for loops as in a2maintemp.py
2) Line 130 in a2main.py: Used np.sum(SET1 != SET2) to get the number of mismatches between elements of two sets, instead of doing a per element comparison as in a2maintemp.py

In addition to above, I also have a python script Time.py, which invokes a4main.py and a4maintemp.py 11 times with larger and larger K values and then records the time taken by each in time.txt. A scan of time.txt will reveal a4maintemp.py taking significantly more execution time than a4main.py.