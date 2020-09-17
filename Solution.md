# Exercise 1 – Imitation Learning
Submitted by: Shubham Arora

## 1.1 Network design

### b) Training
 > Q. Why is is necessary to divide the data into batches?
 
 In most cases where we have large amounts of data for training a model, it is infeasible to feed all the data to the algorithm in one pass. This is due to size of dataset and memory limitations. Hence, data is divided into batches to make it feasible for training. (fit training data into computers memory)

 > Q. What is an epoch?
 
 Epoch is a unit to describe when the entire dataset is passed forward and backward through the neural network exactly once.

 No of epochs is a hyperparameters that states how many times the learning algorithm will work through the entire dataset.

 > Q. What do lines 43 to 48 do?

 


## References
1. https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
2. https://docs.paperspace.com/machine-learning/wiki/epoch

