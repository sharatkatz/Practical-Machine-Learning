---
title: "Prediction Assignment"
output: html_document
---



# R Markdown
##### I modeled, in this assignment, to classify the activity performed by an individual based on the data given. I utilized Random Forest method to build, validate and submit predicted classes for 20 cases in the test data set provided


Clear workspace and load libraries

```r
rm(list=ls()); gc();
```

```
##           used  (Mb) gc trigger  (Mb)  max used  (Mb)
## Ncells 2819144 150.6    4703850 251.3   4703850 251.3
## Vcells 5052088  38.6   28994262 221.3 110600901 843.9
```

```r
library(data.table)
library(caret)
library(ggplot2)
library(tabplot)
library(Hmisc)
library(randomForest)
library(foreach)
```

Input Files

```r
inD <- read.csv("./pml-training.csv", header=T, sep=",", na.strings = c("NA","","#DIV/0!"),strip.white = T)
testing  <- read.csv("./pml-testing.csv", header=T, sep=",", na.strings = c("NA","","#DIV/0!"),strip.white = T)

table(inD$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
##### Since, I did not intend to do any missing imputation, I excluded the variables whichever had any missings. I also excluded those variables which seemed like information of how the information was gathered and who participated in it


Leaving out variables having missings
Not doing missing value treatment/imputation

```r
featWithMissings <- names(which(sapply(inD, function(x) any(is.na(x)|x==""))==TRUE))
```

Take a look at the potential list of features to model

```r
inTr <- inD[,!(names(inD) %in% (featWithMissings))]
```

Quickly visualize data using tableplot function from tabplot package
excluded for HTML file generation as it was getting big in size

```r
for (i in 1:6) {
  jfrom <- (i-1)*10 + 1
  jto <- jfrom + 9
   if (jto <= ncol(inTr)) {
     tableplot(inTr, jfrom:jto)
   }
    else { ( table(inTr, jfrom:ncol(inTr)))
  }
}
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png)![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-2.png)![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-3.png)![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-4.png)![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-5.png)![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-6.png)

Above shows need to exclude first 8 variables for predictors

```r
pred_set1 <- inTr[, -(1:8)]
```

#####I noticed there were some 'total'-prefix variables in the data which had components in X, Y and Z space dimensions also in data. I excluded them as they did not bring in additional information to help in classification
There is a clear pattern between total-prefix variables and its _x, _y, _z counterparts/components
Both kind should not be included
Taking out the total-prefix variables from set of predictors

```r
qplot(data=pred_set1, accel_dumbbell_y, total_accel_dumbbell, facets=.~classe)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png)

```r
pred_set <- pred_set1[, !grepl("^total_", tolower(names(pred_set1)))]
```


#####Then, I created partition in data to train and validate the model as 70-30 split.
#####Training data was scaled before modeling
#####I used random forest technique to build the model; built 4 forests of 50 trees each.
#####And then produced results from calculating accuracy on training and validation datasets.


Partition

```r
set.seed(123)
Part <- createDataPartition(pred_set$classe, p=0.7, list=F) 
training <- pred_set[Part, ]
trainingVal <- pred_set[-Part, ]

R <- training[, -dim(training)[2]]
L <- training$classe
```

Scaling the predictors

```r
normalize <- preProcess(R)
R1 <- as.data.frame(predict(normalize, R))
```

Apply the scaling to validation data set

```r
trainingValR1 <- as.data.frame(predict(normalize, trainingVal[, -dim(trainingVal)[2]]))
trainingValL <- trainingVal$classe
```

The combine function in the randomForest package will bind together different trained forests

```r
set.seed(1234)
quickRF <- randomForest(R1, L, ntree=40, norm.votes=F)
```


Look for convergence of error rates for different classes of 'classe' and Out-of-bag error rate
This helps in keeping a cap on number of trees to build for good prediction (in step above)
by looking at where OOB stabilizes and reaches minimum

```r
plot(quickRF, main="Random Forest")
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png)
Also take a look at the variable importance plot

```r
varImpPlot(quickRF, main="Random Forest")
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

Replicate number of trees for 4 forests and combine

```r
model_rf <- foreach(ntree=rep(50,4), .combine=randomForest::combine) %dopar% {
  rfM <- randomForest(R1, L, ntree=ntree, norm.votes=F, keep.inbag=TRUE) 
}
```

Check accuracy of prediction on training and trainingVal datasets

```r
predictions1 <- predict(model_rf, newdata=R1)
confusionMatrix(predictions1,L)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predictions2 <- predict(model_rf, newdata=trainingValR1)
confusionMatrix(predictions2,trainingValL)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    6    0    0    0
##          B    1 1133   14    0    0
##          C    0    0 1012   18    0
##          D    0    0    0  945    0
##          E    2    0    0    1 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9929          
##                  95% CI : (0.9904, 0.9949)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.991           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9947   0.9864   0.9803   1.0000
## Specificity            0.9986   0.9968   0.9963   1.0000   0.9994
## Pos Pred Value         0.9964   0.9869   0.9825   1.0000   0.9972
## Neg Pred Value         0.9993   0.9987   0.9971   0.9962   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1925   0.1720   0.1606   0.1839
## Detection Prevalence   0.2850   0.1951   0.1750   0.1606   0.1844
## Balanced Accuracy      0.9984   0.9958   0.9913   0.9901   0.9997
```

Apply the scaling to test data set and do the predictions

```r
testingR1 <- as.data.frame(predict(normalize, testing[, names(testing) %in% names(pred_set)]))
testPredictions <- predict(model_rf, newdata=testingR1)
testPredictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


Decision taken

This study shows that in-sample accuracy turned out to be perfect with 95% CI in the range of (0.9997, 1).
Out of Sample accuracy was 0.9929 with 95% CI in the range of (0.9904, 0.9949). Error rate is reported as 0.71%.
Since, Random Forest technique gave me near perfect accuracy, I did not attempt to test Gradient Boosting or Naive Bayes methods.


