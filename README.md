# Multi-class-K-NN-text-classifier
Multi-class text classification system based on a k-Nearest Neighbour (k-NN) classifier

## INTRODUCTION

This program implements a multi-class text classification system based on a k-Nearest Neighbour (k-NN) classifier. Based on the command line user parameters (or default parameters, see below), the program evaluates weighted or unweighted k-nn for a range of values of k using p-fold cross-validation. The evaluation for each k is repeated over a series of runs. The results are outputted to a text file, "out.txt".

For a given value of k, for each run, the results are outputted showing the accuracy of predicted class labels for documents in the test set for each test partition, based on their nearest neighbours in the training set (made up of all other partitions). These are averaged over the p partitions. This average is then averaged over all the runs for that value of k.

## SAMPLE OUTPUT

Run	k(-nn)	Weight	P	  P_test	Test	Train	Accuracy	Avg.	Avg_all_runs
1	  1	      False	  10	0	      184	  1655	93.48	
1	  1	      False	  10	1	      184	  1655	94.02	
1	  1	      False	  10	2	      184	  1655	95.11	
...
1	  1	      False	  10	9	      183	  1656	96.72		  95.05
2	  1	      False	  10	0	      184	  1655	97.28	
2	  1	      False	  10	1	      184	  1655	92.39	
...
2	  1	      False	  10	9	      183	  1656	95.63		  95.11
3	  1	      False	  10	0	      184	  1655	95.65	
...
...
10	1	      False	  10	9	      183	  1656	92.90		  95.38	95.29

In the output above Run is the run number (numbering starting at 1).
k is the number of nearest neighbours used in the classifier.
P is the number of partitions.
P_test is the number of the current test partition (numbering starting at 0).
Test is the number of documents in the current test partition.
Train is the number of documents in the other partitions, i.e., the training set.
Accuracy is the percentage accuracy of predicted class labels for documents in the test set for each test partition, based on their nearest neighbours in the training set, for the current run and value of k.
Avg. is the average accuracy over the P partitions, for a given run and value of k.
Avg_all_runs is the average over all the runs of Avg. for a given value of k.

Therefore, the estimated accuracy for unweighted 1-nn is 95.29%

## PARAMETERS

Sample commandline:
>python knn-auto-eval.py news_articles.mtx news_articles.labels 1 2 10 True 3

The parameters are optional but must be provided in order as they will be interpreted according to the order in which they are given.

Example optional parameters explained:

news_articles.mtx (default: news_articles.mtx) – the file path of a sparse document-term matrix in the Matrix Market format.

news_articles.labels (default: news_articles.labels) – the file path of a text file which contains the labels for the documents in the matrix. Each line of the text document should take the form document_number, document_label, e.g.: 1,business.

1 (default: 5) – the low value of k

2 (default: 5) - the high value for k (in this example, we evaluate k-nn for k between 1 and 2, inclusive, i.e., 1-nn and 2-nn)

10 (default: 10) – the number of partitions. In this example, we will do 10-fold cross-validation.

True (default: False) – indicates whether we use weighted k-nn. True means yes.

3 (default: 1) – indicates the number of runs
