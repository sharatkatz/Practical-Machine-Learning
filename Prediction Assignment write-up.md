# Practical-Machine-Learning
Coursera

I modeled, in this assignment, to classify the activity performed by an individual based on the data given. I utilized Random Forest method to build, validate and submit predicted classes for 20 cases in the test data set provided.

1.	First step after loading relevant libraries was to read in data taking care of how missing data should be read.
2.	Since, I did not intend to do any missing imputation, I excluded the variables whichever had any missings. I also excluded those variables which seemed like information of how the information was gathered and who participated in it.
3.	I noticed there were some ‘total’-prefix variables in the data which had components in X, Y and Z space dimensions also in data. I excluded them as they did not bring in additional information to help in classification.
4.	Then, I created partition in data to train and validate the model as 70-30 split.
5.	Training data was scaled before modeling
6.	I used random forest technique to build the model; built 4 forests of 50 trees each.
7.	And then produced results from calculating accuracy on training and validation datasets.

NOTE: Contents of html file uploaded in repository (Prediction_Assignment.html) can be copied locally in notepad and saved as .htm to look at the knitted document. 	



