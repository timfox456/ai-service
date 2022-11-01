# Lab 3: Training a Model
## Dataset
In this lab, we will use the New York City Airbnb Open Data:

`https://raw.githubusercontent.com/fenago/datasets/main/AirBnB_NYC_2019.csv`

We'll be working with the 'price' variable, and we'll transform it to a classification task.

## Features
For the rest of the lab, you'll need to use the features from below. So the whole feature set will be set as follows:

```text
'neighbourhood_group', 'room_type',
'latitude',
'longitude',
'price',
'minimum_nights',
'number_of_reviews',
'reviews_per_month', 'calculated_host_listings_count',
'availability_365'
```

Select only them and fill in the missing values with 0.

##  Question 1
What is the most frequent observation (mode) for the column 'neighbourhood_group'?
1. Split the data
2. Split your data in train/val/test sets, with 60%/20%/20% (or 80/20) distribution. 
3. Use Scikit-Learn for that (the train_test_split function) and set the seed to 42. 
4. Make sure that the target value ('price') is not in your dataframe.


## Question 2
Create the correlation matrix for the numerical features of your train dataset.

In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.

What are the two features that have the biggest correlation in this dataset?

Example of a correlation matrix for the car price dataset (I know this is not your dataset):

###  Make price binary
We need to turn the price variable from numeric into binary.
Let's create a variable above_average which is 1 if the price is above (or equal to) 152.

## Question 3
Calculate the mutual information score with the (binarized) price for the two categorical variables that we have. Use the training set only.
Which of these two variables has bigger score?

Round it to 2 decimal digits using `round(score, 2)`

## Question 4
Now let's train a logistic regression
Remember that we have two categorical variables in the data. Include them using one-hot encoding.
Fit the model on the training dataset.
To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:

```python
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
```
Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

## Question 5

We have 9 features: 7 numerical features and 2 categorical.
Let's find the least useful one using the feature elimination technique.
