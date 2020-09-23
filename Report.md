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

### e) Forward pass
>  You can decide whether you want to work with all 3 color channels or convert them togray-scale beforehand.  Motivate your choice briefly.

We went ahead and trained with all 3 color channels.
For all color channels, we got average score:
For grayscale, we got score: 

> Can you achieve better results whenchanging the hyper-parameters?  Can you explain this?

We changed the following hyperparameters:
1. Batch size:
2. Learning rate
3. 

### f) Imitations
>  What is ‘good’ training data? 
Good training data teaches the neural network to do the best possible action that expert would have done in the same situation, as it learns from the training data.

> Is there any problem with only perfect imitations?
Yes! By training on only perfect imitations, the neural net does not learn how to recover from bad or catastrophic situations. 
You can try to mitigate this a little with data augmentation techniques. e.g. image smoothing.
However, you cannot create a lot of bad data, or data that shows how to recover from bad situations, because then it impacts the "good" performance of your system. The "good" performance in this case being lane keeping. 


## 1.1 Network improvements
### a) Observations
>Look  at  the  class  method `extract_sensor_values` in `network.py`. What  does  it  do?

The method `extract_sensor_values` provides us the value of `speed`, `abs_sensors`, `gyroscope` and `steering` from the observation tensors. We get this observation tensor from the environment 


>  Incorporate  it  into  your  network  architecture. How does the performance change?

We incorporated the speed into our network. However, it did not seem to affect performance in any way. I am not sure how adding the other sensor values would help as well.

### b) Multi class predictions
> compare the results to the previous classification approach.

We got a worse result using the multi class aproach.

We believe this is because ....

### c) Classification vs.  regression
> Formulate the current problem as a regression network.  Whichloss function is appropriate?


>  What are the advantages / drawbacks compared to the classification networks?

The regression approach has the following potential advantages:
...

and the following drawbacks:
...

Again, the regression network gives us a worse result than the classification approach.


>  Is it reasonable to use a regression approach given our training data?


Note: We actually changed the top speed, steering values while recording our dataset. This served 2 major purposes:
1. Reducing the top speed of the car makes it easier to control (at least for us)
2. While taking turns (steering), we reduce the acceleration value to `0.2`. Again, this makes it easier for us to control the car, and create better datasets. Also, this prevented the car skidding / spinning on the track.

### d) Data augmentation
>  Investigate two ways to create more training data with synthetically modified data by augmenting the (observation, action) - pairs the simulator provides. 


> Does the overall performance change?

### e) Fine - tuning
> What other tricks can be used to improve the performance of the network?  Please try atleast two ideas, explain your motivation for trying them and whether they improved the result.




## References
1. https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
2. https://docs.paperspace.com/machine-learning/wiki/epoch


