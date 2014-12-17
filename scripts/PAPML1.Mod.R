# R script fprPeer assignement 1 of practical machine learning 

# loads the preprocessed and partitioned data sets
# does all the model building on the training set and saves the models to file
# list all confusion matices and accuract on the evaluation set 

# install and load packages

#install.packages("caret")
#install.packages("rattle")
#install.packages("Hmisc") # for cut2()
#install.packages("rpart") # for train(method='rpart')
#install.packages("rpart.plot") # for fancyRpartPlot()
#install.packages("e1071") # required by rpart
#install.packages("ipred") # train(method='treebag')
#install.packages("ada") # train(method='ada'), not used

library(caret)

# set working directory appropriately
setwd("~/Documents/Courses/DataScience/projects/PracticalML")

# run prep-pocessing script to create file if needed
load("data/data.RData")

# model0: nearest neigbors
if (file.exists("models/modFit0.RData")) { 
  load("models/modFit0.RData")
} else {
  set.seed(127)
  modFit0 <- train(classe ~ . , method="knn", data=training)
  save(modFit0,file="models/modFit0.RData")
} 
confusionMatrix(predict(modFit0,evaluation),evaluation$classe) # accuracy is 87.5%
cm0 <- confusionMatrix(predict(modFit0,testing),testing$classe) # accuracy is 87.5%
t(cm0$byClass)

# model1: classification  tree with rpart package
if (file.exists("models/modFit1.RData")) { 
  load("models/modFit1.RData")
} else {
  set.seed(127)
  modFit1 <- train(classe ~ . , method="rpart",data=training, control=rpart.control(cp=0.1))
  save(modFit1,file="models/modFit1.RData")
} 
confusionMatrix(predict(modFit1,evaluation),evaluation$classe) # accuracy is 47.8%

# not working well, depending on seed, accuracy varries widely

# model2: Bagging of trees
if (file.exists("models/modFit2.RData")) { 
  load("models/modFit2.RData")
} else {
  set.seed(1235)
  modFit2 <- train(classe ~ . , method="treebag", data=training)
  save(modFit2,file="models/modFit2.RData")
}
confusionMatrix(predict(modFit2,testing),testing$classe) # accuracy is %97.5

# model3: Random forrest
if (file.exists("models/modFit3.RData")) { 
  load("models/modFit3.RData")
} else {
  set.seed(1235)
  modFit3 <- train(classe ~ . , method="rf", data=training)
  save(modFit3,file="models/modFit3.RData")
}
confusionMatrix(predict(modFit3,evaluation),evaluation$classe) # accuracy is %98.5

# model4: Boosting with trees gbm
if (file.exists("models/modFit4.RData")) { 
  load("models/modFit4.RData")
} else {
  set.seed(123)
  modFit4 <- train(classe ~ . , method="gbm", data=training, verbose=FALSE)
  save(modFit4,file="models/modFit4.RData")
}
confusionMatrix(predict(modFit4,evaluation),evaluation$classe) # accuracy is %94


# model6:combined predictor model
if (file.exists("models/modFit5.RData")) { 
  load("models/modFit6.RData")
} else {
  set.seed(127)
  pred0 <- predict(modFit0,testing) # knn model   
  pred2 <- predict(modFit2,testing) # tree bagging model
  pred3 <- predict(modFit3,testing) # random forest model
  pred4 <- predict(modFit4,testing) # boosting with trees model
  predDF <- data.frame(pred0, pred2, pred3, pred4, classe=testing$classe)
  modFit6 <- train(classe ~ . , method="rf", data=predDF)
  save(modFit6,file="models/modFit6.RData")
}

confusionMatrix(predict(modFit6,data.frame(pred0=predict(modFit0,evaluation),
                                           pred2=predict(modFit2,evaluation),
                                           pred3=predict(modFit3,evaluation),
                                           pred4=predict(modFit4,evaluation))),
                evaluation$classe) # accuracy is %98.5



capture.output(print(confusionMatrix(predict(modFit0,evaluation),evaluation$classe)), 
               file = "results/eval.modFit0.txt")

capture.output(print(confusionMatrix(predict(modFit01evaluation),evaluation$classe)), 
               file = "results/eval.modFit1.txt")

capture.output(print(summary(modFit1)), 
               file = "models/summary.modFit1.txt")

resamps <- resamples(list(KNN = modFit0, TreeBag = modFit2, RF = modFit3, GBM = modFit4))
summary(resamps)
save(resamps, file="report/resamps.RData") 

