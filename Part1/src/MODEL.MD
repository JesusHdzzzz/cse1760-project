To classify our data, we decided on using the Logistic Regression model for sckit-learn. 
First, however, we need to clean our data and preprocess it so the model can actually use it. 
We first examined the .mat file that had the data on matlab desktop, and saw that there are 
4 sub "datasets", each called train_fea1, train_gnd1, test_fea1, test_gnd1. train_fea1 are the 
feature vectors that we will use as input to our training model, and train_gnd1 are the 
corresponding class labels of each of the train_fea1 vectors. The same goes for train_gnd1 
and test_gnd1, except instead of being used for training, they will be used for testing. 

We then used this information, and created a script to examine the shape of the data using 
numpy's built in .shape() function, which we appropiately named feature.py. We can see that the 
train_fea1 data is organized in a 2D arary of shape (6000, 100); this is a 2D array. 
This is fine and we can pass it into our model as is.  However, our train_gnd1 data is shown to have 
a 2D shape as well; more specifically, of shape (6000,1). This is a natural result of how arrays 
are stored in matlab, but most online python machine learning libraries require ground truth label 
arrays to be in 1D, so we use .flatten to essentially force our class label array to be 1D. 

The next step of our data processing is to filter out only the data we need. Because our 
problem is a binary classification problem 