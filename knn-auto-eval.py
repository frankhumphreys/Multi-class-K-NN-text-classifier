# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 23:14:27 2016

@author: Frank Humphreys
"""
import sys
import math
import random
# We import mmread to read a sparse matrix in Matrix Market format
from scipy.io import mmread
from collections import defaultdict

# getDot takes 2 arrays as parameters, treats them as vectors
# and returns their dot product.
def getDot(a,b):
    # As a and b are arrays, a*b multiplies corresponding elements together
    ab = a*b
    # ab = [a1*b1 a2*b2 ...]
    # To calculate the dot product, a.b, we have to sum all the elements in ab
    dot = ab.sum()
    return dot

# getMagnitude takes an array as a parameter, treats it as a vector
# and returns its magnitude
def getMagnitude(a):
    # For vectors, ||a|| = (a.a)^1/2
    magnitude = math.sqrt(getDot(a,a))
    return magnitude

# getCosineSimilarity takes 2 arrays as parameters, treats them as vectors
# and returns their Cosine similarity.
def getCosineSimilarity(a,b):
    # For vectors, Cos(a,b) = a.b/(||a||*||b||)
    similarity = getDot(a,b)/(getMagnitude(a)*getMagnitude(b))
    return similarity

# If n > 0, shuffleNumberList returns a list of length n
# containing the numbers 0 to n-1 randomly shuffled.
# If n <= 0, shuffleNumberList returns an empty list.
def shuffleNumberList (n):
    shuffled_list = []
    # We randomly add new numbers between 0 and n-1 to shuffled_list
    # until len(shuffled-list) = n and it contains all the numbers up to n-1.
    while len(shuffled_list) < n:
        # r = a random integer between 0 and n -1, inclusive.
        r = random.randint(0,n-1)
        # If r isn't in shuffled_list, we make it the next element.
        if not r in shuffled_list:
            shuffled_list.append(r)
    # shuffled_list contains the values 0 to n-1 randomly shuffled.
    return (shuffled_list)

# splitNumberList takes a number n and returns a list of p partitions
# (also lists) such that the partitions contain the numbers 0 to n-1
# randomly and evenly distributed between them with no duplication.
# The largest partition has at most one more element than the smallest.
#
# Precondition: p > 0.
def splitNumberList (n, p):
    # Create split_list as a list of p empty lists /partitions.
    partitions = []
    for i in range(p):
        partitions.append([]) 
    # Create a list containing the numbers 0 to n-1 randomly shuffled.
    shuffle_list = shuffleNumberList(n)
    # Distribute the elements of shuffle_list in the p partitions.
    x = 0 # x is a counter
    current_partition = 0
    while x < n:
        # Append xth item to the current partition. 
        partitions[current_partition].append(shuffle_list[x])
        # Advance to the next item.
        x += 1        
        # If the current partition is the last one,
        # then make the first partition current
        if current_partition == p-1:
            current_partition = 0
        # otherwise advance to the next partition        
        else:
            current_partition += 1
    return(partitions)

# insertNN inserts neighbour n into a list of 'nearest neighbours', 
# removes the least near neighbour and returns the list.
#
# Precondition: each neighbour is a list, the 2nd element of which is a 
# measure of its similarity to some object.
def insertNN(nn,n):
    l = nn
    # Insert the new neighbour in the list.
    l.append(n)
    # Sort the list in order of increasing similarity.
    ls = sorted(l, key=lambda x: x[1])
    # Return the new list less the least similar neighbour, i.e. least near.
    return ls[1:]

# getLabel takes a row previously extracted from the document labels file and
# returns the label corresponding to that row, i.e. document. This involves
# stripping '\n' from the end and stripping the document number and ','
# preceding the label. E.g., '1,business\n' –> '1,business' –> 'business'.
def getLabel(label_string):
    # Split the string at '\n' and assign the first part to s.
    s=label_string.split('\n')[0]
    # Split the string at ',' and return the 2nd part.
    return s.split(',')[1]
    
# predictLabel predicts a class label from a set of nearest neighbours.
# The other parameter is weight.
# If true, the nearest neighbours are weighted by similarity.
# Otherwise, they are weighted equally and majority voting is applied.
#
# Precondition: each neighbour is a list such that the 2nd element is a 
# measure of its similarity to some object and the 3rd is its label.
def predictLabel(nn, weight):
    # Create an empty dictionary of labels (keys) and weights (values).
    labels_weights =  defaultdict(int)
    # If we are using weighted k-nn…
    if weight:
        # For each neighbour in the list of nearest neighbours, nn
        for (a,b,c) in nn:
            # Each time c appears in an neighbour n in nn, 
            # increment labelled_weights[c] by n's similarity score.
            # Thus neighbours will be weighted by similarity.
            labels_weights[c] += b
    # Else if we are using unweighted k-nn…
    else:
        # For each neighbour in the list of nearest neighbours, nn
        for (a,b,c) in nn:
            # Increment labelled_weights[c] by 1 each time c appears in nn
            labels_weights[c] += 1
    # Finally, we find the label key with the largest value in labels_weights.
    # If weight is False, this will be the one that appeared most often.
    # Otherwise, it will be the one with the greatest sum of similarity scores.
    predicted = max(labels_weights, key=lambda x: labels_weights [x])
    return predicted
    
def main():
    # Take the database file names from the command line if entered.
    # Otherwise choose default values
    if len(sys.argv) > 2:
        mfilename = sys.argv[1]
        lfilename = sys.argv[2]
    else:
        # Assign default values to mfilename and lfilename
        mfilename = 'news_articles.mtx'
        lfilename = 'news_articles.labels'
    # Set the default starting value for k for k-nn (nearest neighbours)
    k_min = 5
    # Set the default final value for k for k-nn (nearest neighbours)
    k_max = 5
    # Set the default number of partitions p for p-fold cross-validation 
    p = 10
    # Set unweighted k-nn (simple majority voting) by default
    weight = False
    # Set the default number of runs for each value of k
    runs = 1
    # Change k_min, k_max, p, weight and runs if they have been set by the user.
    if len(sys.argv) > 3:
        k_min = max(int(sys.argv[3]),1)
    if len(sys.argv) > 4:
        k_max = max(int(sys.argv[4]),k_min)
    if len(sys.argv) > 5:
        p = max(int(sys.argv[5]),2)
    if len(sys.argv) > 6:
        weight = sys.argv[6] in ('True','true','T','t','Y','y','1')
    if len(sys.argv) > 7:
        runs = max(int(sys.argv[7]),1)
        
    # Load the document-term matrix as a sparse matrix in COOrdinate format.
    dtmatrixcoo = mmread(mfilename)
    # Make a dense array representation of this matrix
    dtmatrix = dtmatrixcoo.toarray()
    # Aside: to make a copy in Compressed Sparse Row format (CSR), we would 
    # use .tocsr(). That could also be converted using .toarray().
    
    # Open the file to which we will output the results.
    f = open("out.txt","w")
    # Write the column headings
    f.write('Run\tk(-nn)\tWeight\tP\tP_test\tTest\tTrain\tAccuracy\tAvg.\tAvg_all_runs')

    # Load the document class labels
    dlabelfile = open(lfilename,"r")
    dlabels = dlabelfile.readlines()
    length = len(dlabels)
    k = k_min
    
    # Evaluation.
    # Evaluate the average accuracy of p-fold cross-validation of a k-nn 
    # classifier for values of k between k_min and k_max for 'runs' runs.
    
    # Iterate the evaluation for each k over the specified range.
    for k in range (k_min, k_max+1):
        # For each k, repeat 'runs' times
        # We will use avg_accuracies_sum to sum the average accuracies of the
        # predictions in each of the runs for the current value of k. 
        # We will divide by 'runs' to get the average for all runs for k.
        # First we initialise accuracies_sum for the current value of k.
        avg_accuracies_sum = 0
        for run in range (runs):
            # 0 to length-1 are the indices of dlables and of the rows of 
            # dtmatrix. Rather than partition the data directly we create p
            # even partitions of the randomly shuffled document indices
            index_partition = splitNumberList(length, p)
            # We will use accuracies_sum to sum the accuracies of the
            # predictions in each of the p iterations of p-fold
            # cross-validation. We will divide by p to get the average.
            # First we initialise accuracies_sum for the current run.
            accuracies_sum = 0
            # Generate predictions using each partition i in index_partition
            # for our test set, and everything else for our training set,
            # i.e., the training set equals Union{index_partition[j]:j <> i}.
            for test in range (p):
                # Output newline and the current run, k, weight, and p.
                f.write('\n%d\t%d\t%s\t%d\t' % (run+1, k, weight, p))
                # Output the current test data partition number
                f.write('%d\t' % round(test))
                # Output the current test data partition size
                f.write('%s\t' % repr(len(index_partition [test])))
                # Output the current training data size.
                f.write('%s\t' % repr(length - len(index_partition [test])))

                # For each document with index i in index_partition[test],
                # we find the k nearest neighbours and generate a prediction
                # for its class label, which we check against the actual label.
                correct = 0
                for i in index_partition [test]:
                    test_document = dtmatrix[i,:]
                    # nn holds details of i's k nearest neighbours found so far.
                    nn = []
                    # min_similarity is the similarity to i of the least 
                    # similar document indexed in nn or is -1 if nn is empty.
                    min_similarity = -1
                    # Go through each document indexed in each of the other
                    # partitions looking for nearest neighbours 
                    for j in range (p):
                        # Is j the test partition? If not, j is part of the
                        # training set we search for nearest neighbours.
                        if j != test:
                            # Go through each document indexed in partition j 
                            # looking for nearest neighbours
                            for w in index_partition [j]:
                                training_document = dtmatrix[w,:]
                                # Evaluate the cosine similarity between the
                                # training and test documents
                                similarity = getCosineSimilarity(test_document,training_document)
                                # If the number of nearest neighbours in nn is
                                # less than k, add the training document's 
                                # index, similarity and label to nn.
                                if len(nn) < k:
                                    nn.append([w,similarity,getLabel(dlabels[w])])
                                    # Update min_similarity.
                                    min_similarity = min(b for (a,b,c) in nn)
                                # Else if the similarity is greater than the
                                # least similar document indexed in nn, add
                                # its index, similarity and label to nn.
                                elif similarity > min_similarity:
                                    nn = insertNN(nn,[w,similarity,getLabel(dlabels[w])])
                                    # Update min_similarity.
                                    min_similarity = min(b for (a,b,c) in nn)
                    # Generate prediction for i from its nearest neighbours.
                    predicted = predictLabel(nn, weight)
                    # Get the actual label for i.
                    actual = getLabel(dlabels[i])
                    # If the prediction was accurate, increment the correct count.
                    if actual == predicted:
                        correct += 1

                # Correct is now a count of all the documents indexed in the
                # test partition that were correctly labelled using the
                # training set data. 
        
                # We can calculate accuracy by dividing 'correct' by the total
                # number of test documents.
                accuracy = correct/float(len(index_partition [test])) * 100.0                    
                # Output accuracy to two decimal places
                f.write('%.2f\t' % accuracy)
                # We add 'accuracy' to 'accuracies_sum' which we use below
                # to find the average accuracy.
                accuracies_sum += accuracy                      
            
            # Divide the sum of the accuracy results by the number of tests
            # (i.e., partitions)
            avg_accuracy = accuracies_sum/p
            # Output avg_accuracy to two decimal places
            f.write('%.2f\t' % avg_accuracy)
            # We add 'avg_accuracy' to 'avg_accuracies_sum' which we use below
            # to find the average accuracy for all runs for the current k.
            avg_accuracies_sum += avg_accuracy                      
        avg_avg_accuracy = avg_accuracies_sum/runs
        # Output avg_avg_accuracy to two decimal places
        f.write('%.2f' % avg_avg_accuracy)
    f.write('\n')
    dlabelfile.close()
    f.close()
   
main()