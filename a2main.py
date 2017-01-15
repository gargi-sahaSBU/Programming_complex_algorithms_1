import sys
import numpy as np
import scipy.stats

def bayes_params(X, Y):
    classes = np.unique(Y)
    (n,dim) = X.shape
    Pc = []
    cov_full = []
    var_naive = []
    mean=[]
    for ci in classes:
        # rows will contain row numbers of all values in Y that are equal to ci
        rows = np.nonzero(Y == ci)[0]
        # obtain all corresponding x in X who's y = ci
        Di = X[rows]
        
        ni = len(Di)
        # storing the prior probability
        Pc.append(ni/n)
        
        mean.append(sum(Di)/ni)
        # calculating Di - 1_ni * mu 
        Zi=(Di-(np.ones([ni,dim])*(sum(Di)/ni)))
        # calculating the covariance matrix
        sigma = (Zi.T.dot(Zi)/ni)
        #storing covariance for full Bayes
        cov_full.append(sigma)
        #storing variance for naive Bayes
        var_naive.append(np.diag(sigma))

    return (classes,Pc,mean,cov_full,var_naive)



def multi_var_normal_pdf(x, mean, cov_full, var_naive):
    # x - mu
    x_mean=(x-mean)
    # converting (x-mu) to 2D so transpose can be taken
    x_mean_t = x_mean[np.newaxis]
    # dot product of (x-mu),a 1X4 matrix, with the inverse of covariance matrix, a 4X4 matrix: 
    e_pow = (x_mean.dot(np.linalg.inv(cov_full)))
    # dot product of above with (x-mu).T, then projecting that to a 2D matrix again so determinant can be taken
    e_pow = -(np.linalg.det(e_pow.dot(x_mean_t.T)[np.newaxis])/2)
    # e ^ {the big equation}
    e_pow = np.exp(e_pow)
    # the factor to divide with: 4 * pi * pi * sqrt of determinant of cov matrix
    cov_det = np.sqrt(np.linalg.det(cov_full)) * 4 * np.pi * np.pi
    Bayes = (e_pow/cov_det)
    #-----------------------------------
    #beginning naive bayes calculation
 
    prod_all_dims = 1 # product of fc over all dimensions

    for j in range(0,4): #iteration over the 4 dimensions
        # (x-mu) ^ 2 divided by 2
        e_pow_naive = ((x[j] - mean[j]) * (x[j] - mean[j]))/2
        e_pow_naive = -(e_pow_naive / var_naive[j])
        e_pow_naive = np.exp(e_pow_naive)
        cov_det_naive = np.sqrt(var_naive[j] * 2 * np.pi)
        prod_all_dims = prod_all_dims * (e_pow_naive/cov_det_naive)

    naive_Bayes = prod_all_dims
    return (Bayes, naive_Bayes)



def bayes_test(x, classes, Pc, mean, cov_full, var_naive):
    fc_pc = [] # list of fc * Pc for full Bayes
    fc_pc_naive = []  # list of fc * Pc for naive Bayes
    num_of_classes = len(classes)

    for i in range(0,num_of_classes): #iteration over classes

        Bayes, naive_Bayes = multi_var_normal_pdf(x, mean[i], cov_full[i], var_naive[i]) 
        #computing product of f(c) with P(c)
        fc_pc.append(Bayes * Pc[i])
        
        fc_pc_naive.append(naive_Bayes * Pc[i])
    
    return ( fc_pc.index(max(fc_pc)), fc_pc_naive.index(max(fc_pc_naive)) )
    


def paired_t_test(X, Y, K, alpha):
    n = len(X)
    diff = np.zeros(K)
    for i in range(K):
        # 1 compute training set X_i using bootstrap resampling
        sample_index = np.random.randint(n, size= n)
        X_i = X[sample_index]
        Y_i = Y[sample_index]


        # 2 train both full and naive Bayes on sample X_i
        classes, Pc, mean, cov_full, var_naive = bayes_params(X_i, Y_i)

        # 3 compute testing set X - X_i
        # POINT 1:USING NUMPY FUNCTION setdiff1d INSTEAD OF RUNNING 2 FOR LOOPS 
        test_indices = np.setdiff1d(np.arange(150),sample_index)
        X_minus_Xi = X[test_indices]
        Y_minus_Yi = Y[test_indices]

        # 4 assess both on X - X_i
        # Files to store value predicted classes for the two classifiers
        out_pred_full_Y = open('iris_pred_Y_full.txt','w')
        out_pred_naive_Y = open('iris_pred_Y_naive.txt','w')
        
        for x in X_minus_Xi:
            fc_pc_index, fc_pc_naive_index = bayes_test(x, classes, Pc, mean, cov_full, var_naive)
            
            # writing predicted output of full Bayes to iris_pred_Y_full.txt for easy comparison in future
            out_pred_full_Y.write(str(classes[fc_pc_index]).replace("(b\"b'","").replace("'\",)",""))
            out_pred_full_Y.write("\n")
            
            # writing predicted output of naive Bayes to iris_pred_Y_naive.txt for easy comparison in future
            out_pred_naive_Y.write(str(classes[fc_pc_naive_index]).replace("(b\"b'","").replace("'\",)",""))
            out_pred_naive_Y.write("\n")

        out_pred_full_Y.close()
        out_pred_naive_Y.close()

        #loading predicted output of each classifier to 2 different numpy arrays
        pred_Y = np.loadtxt('iris_pred_Y_full.txt',delimiter="\n",dtype={'names':['e'],'formats':['S20']})
        pred_naive = np.loadtxt('iris_pred_Y_naive.txt',delimiter="\n",dtype={'names':['e'],'formats':['S20']})

        #number of mismatches of predicted class of each classifier with true class
        # POINT 2:USING NUMPY FUNCTION IN PLACE OF REGULAR ARRAY MANIPULATION

        num_err_full = np.sum(Y_minus_Yi != pred_Y)
        num_err_naive = np.sum(Y_minus_Yi != pred_naive)

        print('sample, full, naive:', i, num_err_full, num_err_naive)

        # 5 compute difference in error rates
        err_rate_full = num_err_full / len(Y_minus_Yi)
        err_rate_naive = num_err_naive / len(Y_minus_Yi)

        diff[i] = (err_rate_full - err_rate_naive)
        
    print('all differences:'); print(diff)
    
    
    # compute mean, variance, and z-score
    mean_diff = (sum(diff)/K)

    variance = 0
    for d in diff:
        d_minus_mean = d - mean_diff
        variance = variance + (d_minus_mean * d_minus_mean)
    variance = variance/K

   
    z_score = (np.sqrt(K) * mean_diff) / np.sqrt(variance)

    print('z-score:', z_score)

    # compute interval bound using inverse survival function of t distribution
    bound = scipy.stats.t.isf((1-alpha)/2.0, K-1) 
    print('bound:', bound)

    # output conclusion based on tests
    if -bound < z_score < bound:
        print('accept: classifiers have similar performance')
    else:
        print('reject: classifiers have significantly different performance')


