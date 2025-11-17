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
problem is a binary classification problem consisting only of digits 5 and 6, we need 
to filter out the appropiate images. To do that, we defined something called a mask on 
our labels array.  A mask is a boolean array with defined conditions on a 
corresponding array where if the condition is true for that array's index, then the 
corresponding index in the mask will be true, and will be false otherwise. Our mask's 
defined "condition" is if the label at the label array's index is 5 or 6, then true, and false otherwise. 
We then applied that mask to our train_fea1 array to fully filter out the images with only a label of 5 or 6,
and I also filtered out my class label array, while aso defining 5 to be the negative class while 6 
is the positive class. I then defined some print statements to give a basic overview of the data. Finally,
we standardize our data to a typical gaussian curve of mean 0 and standard deviation 1. 

Next is the fun part; training our model. We want to prevent overfitting of our model, so we need 
to define some kind of regularization. We will use l2 regularization for our model. l2 regularization defines 
a penalty to the magnitude of the weights that it learns. This essentially forces the weights to be as minimal
as possible. Another importnat parameter to consider is the C parameter. This term is the inverse of the 
regularization cost lambda. What this means is that the higher the C parameter, the weaker regularization will be. 
The lower the C parameter, the stronger that regularization will be. This is the parameter that we are trying to 
optimize; we don't want the C parameter too small such that we underfit, but we don't want the C parameter 
so high that we dont regularize at all and we overfit our data. We want to find the perfect tradeoff. To do this, 
we will train our model multiple times using different values for C, and we will use 5 fold cross validation
to test for generalization error. We keep a running variable throughout our run that maintains the model 
with the best average accuracy over the 5 runs in 5 fold cross validation. The C value at the end that produces
the best average accuracy in cross validation will be our final model, which we will then train again on 
our training data and finally testing it with our test data set. We then print out the accuracy given by the test 
run.