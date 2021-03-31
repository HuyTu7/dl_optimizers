## 3/24
- slide: https://docs.google.com/presentation/d/1e_-RJp4CJ8t4vwb5PbdYTk5-cykHjNY_eRWqcYVQWaI/edit?usp=sharing

accuracy: (TP+TN)/(TP+TN+FP+FN)

precision: TP/(TP+FP)

recall: predicted TP/ all TP

f1: harmonic mean of precision and recall



## 3/1
- read the TRN paper: confirm the structure if the TRN only update the last layer and what are the benefits. 
- try other datasets (e.g., MNIST)
- rerun without TRN (with or without Layer-wise) so we have 4 settings: TRN alone, Layer-wise alone, and TRN + Layer-wise
- check the HPC to run large-batch training for the layer-wise

## 2/17

TRN and LARS: SGD vs. LAMB: Adam ← mini-batch size 
CIFAR:
batch size: 16 (same as the TRN), so should have the similar performance as Jake’s
batch size: 16- compare LARS and Adam to see whether there is a performance improvement
compare all thee with batch size 64, 512
                     ** maybe accuracy is not the best metric? 
                     ** mnist ?

Layer-wise: can work on larger batch size without performance losing 8K, 32K (capacity: 1K)
LAMB & LARS (mnist or CIFAR)


Literatures folder: MIT, time-series,....
what’s
pros and cons
how can it help us 


Applications: Image classification- CIFAR, mnist

### Note:
For general accuracy metric: as the batch size increases, the performance of LAMB+TRN over SGD+TRN goes from slightly lower to better than.
Best case, at the batch size of 1K, LAMB+TRN performance is 7% relatively better over SGD+TRN.

Not sure if the top-K accuracy (or even the general accuracy) is an appropriate metric as I check the loss outputs at the end of the training (most likely epoch #200):
- Batch size 512, SGD+TRN=247 whereas LAMB+TRN=0.31
- Batch size 512, SGD+TRN=484 whereas LAMB+TRN=4.8

Next steps:
try MNIST dataset and also include precision, recall, and f1.

## 1/27
Go through the word file and TRN 
For TRN: understand does it apply to “deep learning?” how? Yes, Jake’s work uses multi-layer structure.
Find some related work for TRN + deep learning (+ time-awareness/layer-wise?)
Rerun the TRN code can already run his code 
Understand how to do the compression and reconstruction; probably work with some application data
Include the layer-wise code with TRN for time-aware deep learning
Possible tasks: (i) image classification; (ii) QA system





