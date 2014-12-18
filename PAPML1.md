# Practical Machine Learning: Peer Assessment
Guido Gallopyn  
12/17/2014  



# Introduction

In this project, our goal is to predict the manner in which subjects perform a weightlifting exercise using as predictors data from accelerometers on the belt, forearm, arm and dumbbell. We use the Weight Lifting Exercise Data-set that contains data from 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har). see [ref 1.]

This report describes how a set of prediction model was build with the caret package [ref 2.], how we used cross validation to estimate out of sample error and selected the best model, and we explain the choices that were made. We have used the best prediction model to predict 20 different test cases as part of this assignment. 

# Explory Analysis and Data Processing

## Observation

* The pml-training.csv data set consists of a set of time series with in total 19622 observations of 160 variables. The pml-testing.csv contains only 20 discrete points in time. This is preventing the prediction model to use features derived from time windows on the data set as the authors in [ref 1] did in their prediction modeling. We have removed all time related variables from the observations. 

* Many variables are left blank for a majority of the observations, we remove variables with more than 50% blank observations.

* Although some prediction models are robust to this, we remove variables with near zero variance primarily to reduce the training data size and model training time.

* Some data from the accellerometers come in the form of (x,y,z) components of 3D vectors. We calculate magnitudes of these vectors as covariates and we remove the individual x, y and z components.

* There are outliers in  magnet\_dumbbell\_mag and gyros\_dumbbell\_mag and gyros\_forearm\_mag variables, each have  observations as far as 50 or more standard deviations from the mean. We have removed them for graphical scaling purposes, but we left them intact for model training as we plan to use non-parametric based prediction that is robust against outliers.

* With these pre-processing steps we obtain a data set of 19622 observations of 25 variables.

## Data Partitioning

* To train, test and evaluate our models we partition the data in 3 subsets as outlined below. We use the training set to train all prediction models and estimate accuracy trough cross validation, we use the test set for error analysis and tuning of parameters when needed, and we use the evaluation set to finally predict the to be expected accuracy. As the final evaluation will be done on discrete time points, we will partition the data with the random partitioning as provided by caret createDataPartition function. Evaluation set is 30% of the data, training set 49% and the test set 21%.  


```r
inTrain  <- createDataPartition(y=data$classe, p=0.7, list=FALSE)
training <- data[inTrain,]
evaluation <- data[-inTrain,]
inTrain  <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
testing  <- training[-inTrain,]
training <- training[inTrain,]
```

## Exploratory Data Analysis

Below are 2 characteristic plots of a graphical analysis of predictor variables in the training set. 

* The first plot shows 2 principle components of the predictor variables with data points colored according to the classe. Observe that the data is quite noisy and that the classes are not confined to simple regions in this 2D sub-space, and that there are no obvious decision boundaries for classification.


```r
preProc <- preProcess(training[,-17], method="pca", pcaComp=2)
pcdata <- transform(predict(preProc,training[,-17]),classe=training$classe)
ggplot(pcdata,aes(PC1,PC2)) +  xlim(-5,5) + ylim(-5,5) + geom_point(aes(colour=classe),alpha=0.15,size=2)
```

![2 principal components of the predictors](report/plot1.png)

* The plot below is a caret featurePlot with the predictor variables chosen by a simple classification tree trained on the data, data points are again colored according to the classe. Observe again that there are no obvious decision boundaries between the 5 classes.  


```r
cart <- train(classe ~ . , method="rpart", data=training)
featurePlot(x=training[,setdiff(cart$finalModel$frame$var,c("<leaf>"))], y=training$classe, plot="pairs")
```

![CART predictors](report/plot2.png)

In conclusion, the exploratory analysis of training data reveals a complex and noisy data set with a high high degree of overlap between the 5 classes A, B, C, D and E, and non-obvious decision boundaries for the classes A, B, C, D and E.

# Model Selection, Training and Cross-validation

The exploratory analysis points to a hard classification problem that will best be tackled by prediction techniques such as classification trees and more advanced statistical classification techniques. We will use the caret package to explore a variety of modeling techniques and in addition build a combination of prediction models.

## Simple Classification Tree

As a baseline, we train a classification tree using the training data with the rpart method in the caret train function. We use no pre-processing (not needed for CART) and use default training parameters, which means repeated bootstrapped resampeling (25 reps) with optimal model selection based on accuracy.


```r
modFit1 <- train(classe ~ . , method="rpart",data=training)
fancyRpartPlot(modFit1$finalModel)
```
![Classification Tree](report/plot3.png)

This classification tree has an accuracy of 48% determined by the default cross validation based on resampeling. As can be seen, the tree lacks a leaf node to predict class B hence the sensitivity (recall) of B class prediction is zero. The purity of the rightmost (red) leaf is great, leading to a positive predictive value (precision) of 99.8% for the E class by using just one simple split on roll-belt. The A class has two leaf nodes resulting from two split criteria on roll-belt and pitch-forearm that achieve a recall (sensitivity) of 80%, but only with a precision of 57%. The leaf node for class D is the node where a lot of the data is classified (43%) and with low purity. Apparently the rpart training doesn't find a way to further split this node and improve accuracy. With these low accuracy results we can concluded that a simple classification tree will not lead to good results, and we have not explored various rpart training options, but instead we explored other and more advanced prediction methods.

## More complex classifiers

Given the complexity of the data, we proceed to build four more classifiers on the training set

* a k-Nearest Neighbors non-parametric model with knn method 
* a Bagged CART model using treebag method
* a Random Forest model using rf method
* a Stochastic Gradient Boosting model with gbm method

For all these model training we don't use pre-processing and we use the caret default parameters, for cross validation this means repeated bootstrapped resampeling (25 reps) with optimal model selection based on accuracy. The training based on the code below takes considerable time (multiple hours of CPU time)


```r
modFit0 <- train(classe ~ . , method="knn", data=training)
modFit2 <- train(classe ~ . , method="treebag", data=training)
modFit3 <- train(classe ~ . , method="rf", data=training)
modFit4 <- train(classe ~ . , method="gbm", data=training, verbose=FALSE)
resamps <- resamples(list(KNN = modFit0, TreeBag = modFit2, RF = modFit3, GBM = modFit4))
summary(resamps)
```


```
## 
## Call:
## summary.resamples(object = resamps)
## 
## Models: KNN, TreeBag, RF, GBM 
## Number of resamples: 25 
## 
## Accuracy 
##          Min. 1st Qu. Median  Mean 3rd Qu.  Max. NA's
## KNN     0.810   0.820  0.827 0.827   0.833 0.840    0
## TreeBag 0.938   0.950  0.957 0.955   0.959 0.964    0
## RF      0.974   0.977  0.978 0.979   0.980 0.984    0
## GBM     0.928   0.932  0.935 0.935   0.938 0.948    0
## 
## Kappa 
##          Min. 1st Qu. Median  Mean 3rd Qu.  Max. NA's
## KNN     0.760   0.772  0.782 0.781   0.789 0.797    0
## TreeBag 0.922   0.937  0.946 0.943   0.948 0.954    0
## RF      0.967   0.971  0.973 0.973   0.975 0.980    0
## GBM     0.909   0.914  0.917 0.918   0.922 0.935    0
```

Observations

* The best accuracy measured through cross validation on the training set is obtained by the Random Forest model, it reaches 97.86% which is comparable to the recognition performance of 98.03% in [ref.1][1]. The result in [ref.1][1] was obtained with a classifier constructed as an ensemble of 10 random forests of 10 trees using a bagging method, with predictor variables extracted from an optimal 2.5 sec time windows of the sensors data. The random forest constructed by caret in this project used predictor variables with discrete time points but contains 500 trees. 

* The Bagged CART and Stochastic Gradient Boosting models give somewhat lower accuracy then the random forest model.

* Surprising is that a simple non-parametric approach as k nearest neighbors is able to reach 82.66 accuracy.

# A Combined Classifier 

As a final step in pursuit of even higher accuracy, we build a combined classifier using an ensemble of the models from the previous section, with a random forest to combine the predictions from the ensemble on the test set to provide a final prediction. 


```r
pred <- data.frame( pred0 = predict(modFit0,testing),  # prediction of knn model
                    pred2 = predict(modFit2,testing),  # prediction of tree bagging model
                    pred3 = predict(modFit3,testing),  # prediction of random forest model
                    pred4 = predict(modFit4,testing),  # prediction of boosting with trees model
                    classe = testing$classe)
modFit6 <- train(classe ~ . , method="rf", data=pred)
print(modFit6)
```


```
## Random Forest 
## 
## 4118 samples
##    4 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 4118, 4118, 4118, 4118, 4118, 4118, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1         1      0.002        0.003   
##    9    1         1      0.002        0.003   
##   16    1         1      0.002        0.003   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 9.
```

The estimated accuracy of the combined classifier on the test set through cross validation is 98.8%

On the test set there are only a few observations where the final combined model provides a different prediction compared to the random forest predictor from the previous section.


```r
confusionMatrix(predict(modFit3,testing),predict(modFit6,pred))$table
```


```
##           Reference
## Prediction    A    B    C    D    E
##          A 1174    1    0    0    0
##          B    0  784    9    0    1
##          C    0    5  715    3    0
##          D    0    0    4  666    1
##          E    0    0    0    1  754
```

# Final Evaluation and Conclusions

To compare the accuracy of the combined predictor with the models explored in this project, we use the evaluation data set. We predict outcomes on the evaluation data using the models, and calculate a confusion matrix comparing predicted outcomes with the evaluation reference, and calculate derived statistics.  


```r
pred <- data.frame( pred0 = predict(modFit0,evaluation),  # knn model
                    pred1 = predict(modFit1,evaluation),  # simple tree model
                    pred2 = predict(modFit2,evaluation),  # tree bagging model
                    pred3 = predict(modFit3,evaluation),  # random forest model
                    pred4 = predict(modFit4,evaluation))  # boosting with trees model
pred <- transform(pred,pred6=predict(modFit6,pred),       # prediction of combined model
                  classe=evaluation$classe)               # reference
print(confusionMatrix(pred$pred6,pred$classe))

acc <- data.frame(ModelInfo=sapply(c(0:4,6),function(x) eval(parse(text = paste0("modFit",x,"$modelInfo$label")))),
                  Method=sapply(c(0:4,6),function(x) eval(parse(text = paste0("modFit",x,"$method")))),
                  Accuracy=sapply(paste0("pred",c(0:4,6)), 
                           function(x) confusionMatrix(pred[[x]],pred$classe)$overall[["Accuracy"]]))
print(acc[order(acc$Accuracy),])
```


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669   18    0    0    0
##          B    3 1105   12    0    0
##          C    2   16 1009   24    3
##          D    0    0    5  938    7
##          E    0    0    0    2 1072
## 
## Overall Statistics
##                                         
##                Accuracy : 0.984         
##                  95% CI : (0.981, 0.987)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.98          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.970    0.983    0.973    0.991
## Specificity             0.996    0.997    0.991    0.998    1.000
## Pos Pred Value          0.989    0.987    0.957    0.987    0.998
## Neg Pred Value          0.999    0.993    0.996    0.995    0.998
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.188    0.171    0.159    0.182
## Detection Prevalence    0.287    0.190    0.179    0.161    0.182
## Balanced Accuracy       0.996    0.983    0.987    0.985    0.995
```

```
##                      ModelInfo  Method Accuracy
## 2                         CART   rpart   0.4783
## 1          k-Nearest Neighbors     knn   0.8732
## 5 Stochastic Gradient Boosting     gbm   0.9405
## 3                  Bagged CART treebag   0.9760
## 6                Random Forest      rf   0.9844
## 4                Random Forest      rf   0.9852
```

Line number 6 is the accuracy from the final model on the evaluation set, note that it is slightly lower than the random forest model from the previous section, but is well within the 95% confidence interval. 

I found the accuracy obtained surprisingly high, especially that a single large random forest and a also a more complex combined model trained on noisy data taken randomly from time series is able to match the accuracy of the approach in [ref.1][1]

I decided to use the combined model for the submission of the 20 point testing set of this project, because although the accuracy is equivalent as the random forest model, I would hope for slightly more robustness to new data due to a larger number of predictors 

Note: The high accuracy measured here, may be an overestimate due to the artifact of the data set partitioning and cross validation method of resampeling used to measure accuracy on the data set that is a set of time series. Data partitioning and resampeling procedures mix samples from all time series in the data-set. A better way to cross validate may be to hold out complete time series for training, test and evaluation sets and also for accuracy measurement on the training set via a more sophisticated cross validation. In this way there would be a higher independence of the training, test and evaluation data sets. This was not pursued further in this project, as the final model created in this project obtained a perfect 20/20 score on the Coursera submission pages already.

# References
1: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3M5zysvol

2: the caret package http://topepo.github.io/caret/


