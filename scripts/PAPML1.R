
library(caret)
library(rattle)
library(ggplot2)
library(plyr)
library(caret)
library(rattle)
library(ggplot2)
library(plyr)
library(Hmisc)


#functions
acc <- function(model,data) round(100*confusionMatrix(predict(model,data),data$classe)$overall[["Accuracy"]],0)
magnitude <- function(x,y,z) sqrt(x^2 + y^2 + z^2)
add.magnitudes <- function(df) transform(df, 
                                         accel_belt_mag = magnitude(accel_belt_x, accel_belt_y, accel_belt_z),
                                         gyros_belt_mag = magnitude(gyros_belt_x, gyros_belt_y, gyros_belt_z),
                                         magnet_belt_mag = magnitude(magnet_belt_x, magnet_belt_y, magnet_belt_z),
                                         accel_arm_mag = magnitude(accel_arm_x,accel_arm_y,accel_arm_z),
                                         gyros_arm_mag = magnitude(gyros_arm_x,gyros_arm_y,gyros_arm_z),
                                         magnet_arm_mag = magnitude(magnet_arm_x,magnet_arm_y,magnet_arm_z),
                                         accel_forearm_mag = magnitude(accel_forearm_x,accel_forearm_y,accel_forearm_z),
                                         gyros_forearm_mag = magnitude(gyros_forearm_x,gyros_forearm_y,gyros_forearm_z),
                                         magnet_forearm_mag = magnitude(magnet_forearm_x,magnet_forearm_y,magnet_forearm_z),
                                         accel_dumbbell_mag = magnitude(accel_dumbbell_x,accel_dumbbell_y,accel_dumbbell_z),
                                         gyros_dumbbell_mag = magnitude(gyros_dumbbell_x,gyros_dumbbell_y,gyros_dumbbell_z),
                                         magnet_dumbbell_mag = magnitude(magnet_dumbbell_x,magnet_dumbbell_y,magnet_dumbbell_z))


# set working directory appropriately
setwd("~/Documents/Courses/DataScience/projects/PracticalML")

if (file.exists("data.RData")) { load("data.RData")
} else {
  # Read the csv file for data set
  data <- read.csv("pml-training.csv")
  
  # throw out variables with more then 50% NA or near-zero-variance in data set, remove index, username and time related info
  vars <- names(data)[sapply(names(data), function(x) (mean(is.na(data[[x]])) < 0.5))]
  vars <- vars[-nearZeroVar(data[vars],freqCut = 90/10)]
  vars <- setdiff(vars,c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window"))
  data <- data[vars]
  
  # covariate creation: calculate vector magnitudes, and remove x,y,z components
  vec <- sub("(.)_x","\\1",grep("_x",names(data),value=TRUE)) # find features that are vectors
  
  data <- add.magnitudes(data)
  data <- data[,-grep("_(x|y|z)",names(data))]  # remove (xyz) features
  
  # dataset partitioning in train, test sets
  set.seed(1235)
  inTrain  <- createDataPartition(y=data$classe, p=0.7, list=FALSE)
  training <- data[inTrain,]
  evaluation <- data[-inTrain,]
  inTrain  <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
  testing  <- training[-inTrain,]
  training <- training[inTrain,]  
  save(training,testing,evaluation,file="data.RData")
}

# preprocessing (normalizing? box-cox?)
preProc <- preProcess(training[,-17],method="pca",pcaComp=2)
pcdata <- predict(preProc,training[,-17])
ggplot(aes(PC1,PC2),data=pcdata) + geom_point(alpha=0.1,size=1,colour=as.numeric(training$classe))


# plotting predictors => not much to see?!
featurePlot(x=training[,grep("dumbbell",names(data))], y=training$classe, plot="box")
featurePlot(x=training[,grep("forearm",names(data))], y=training$classe, plot="box")
featurePlot(x=training[,grep("_arm",names(data))], y=training$classe, plot="box")
featurePlot(x=training[,grep("belt",names(data))], y=training$classe, plot="box")

featurePlot(x=training[,grep("dumbbell",names(data))], y=training$classe, plot="pairs")

qplot(magnet_belt_mag,colour=classe,geom="density",data=training)
qplot(accel_arm_mag,accel_dumbbell_mag,colour=classe,data=training)

# model0: nearest neigbors
if (file.exists("modFit0.RData")) { 
  load("modFit0.RData")
} else {
  set.seed(127)
  modFit0 <- train(classe ~ . , method="knn", data=training)
  save(modFit0,file="modFit0.RData")
} 
confusionMatrix(predict(modFit0,testing),testing$classe) # accuracy is 87%



# model1: classification  tree with rpart package
if (file.exists("modFit1.RData")) { 
  load("modFit1.RData")
} else {
  set.seed(127)
  modFit1 <- train(classe ~ . , method="rpart",data=training, control=rpart.control(cp=0.1))
  save(modFit1,file="modFit1.RData")
} 
fancyRpartPlot(modFit1$finalModel)
confusionMatrix(predict(modFit1,testing),testing$classe) # accuracy is 48%

# not working well, depending on seed, accuracy varries

# model2: Bagging of trees
if (file.exists("modFit2.RData")) { 
  load("modFit2.RData")
} else {
  set.seed(1235)
  modFit2 <- train(classe ~ . , method="treebag", data=training)
  save(modFit2,file="modFit2.RData")
}
confusionMatrix(predict(modFit2,testing),testing$classe) # accuracy is %97.5

# model3: Random forrest
if (file.exists("modFit3.RData")) { 
  load("modFit3.RData")
} else {
  set.seed(1235)
  modFit3 <- train(classe ~ . , method="rf", data=training)
  save(modFit3,file="modFit3.RData")
}
confusionMatrix(predict(modFit3,testing),testing$classe) # accuracy is %98

# model4: Boosting with trees gbm
if (file.exists("modFit4.RData")) { 
  load("modFit4.RData")
} else {
  set.seed(123)
  modFit4 <- train(classe ~ . , method="gbm", data=training, verbose=FALSE)
  save(modFit4,file="modFit4.RData")
}
confusionMatrix(predict(modFit4,testing),testing$classe) # accuracy is %94
print(modFit4)

# model5: adaBoost Boosting with additive logistic models,  WORKING only for binary resonse
if (file.exists("modFit5.RData")) { 
  load("modFit5.RData")
} else {
  set.seed(127)
  binaryOutcomes <- dummyVars(~ classe, data = training)
  training2 <- predict(binaryOutcomes, training)
  trainPred <- subset(training,select=-classe)
  testing2  <- predict(binaryOutcomes, testing)
  testPred  <- subset(testing,select=-classe)
  fitControl <- trainControl(classProbs = TRUE)
  modFit5A <- train(as.factor(training2[,1]) ~ . , method="ada", data=trainPred, trControl=fitControl)
  modFit5B <- train(as.factor(training2[,2]) ~ . , method="ada", data=trainPred, trControl=fitControl)
  modFit5C <- train(as.factor(training2[,3]) ~ . , method="ada", data=trainPred, trControl=fitControl)
  modFit5D <- train(as.factor(training2[,4]) ~ . , method="ada", data=trainPred, trControl=fitControl)
  modFit5E <- train(as.factor(training2[,5]) ~ . , method="ada", data=trainPred, trControl=fitControl)
  
  confusionMatrix(predict(modFit5A,subset(testing,select=-classe)),as.factor(testing2[,1])) # 95%
  confusionMatrix(predict(modFit5B,subset(testing,select=-classe)),as.factor(testing2[,2])) # 91%
  confusionMatrix(predict(modFit5C,subset(testing,select=-classe)),as.factor(testing2[,3])) # 90%
  confusionMatrix(predict(modFit5D,subset(testing,select=-classe)),as.factor(testing2[,4])) # 94%
  confusionMatrix(predict(modFit5E,subset(testing,select=-classe)),as.factor(testing2[,5])) # 96%
  
  predict(modFit5A,head(testPred),type="prob")
  
  
  save(modFit5A,modFit5B,modFit5C,modFit5D,modFit5E,file="modFit5.RData")
}

confusionMatrix(predict(modFit5A,subset(testing,select=-classe)),as.factor(testing2[,1]))


# model6:combined predictor model
if (file.exists("modFit5.RData")) { 
  load("modFit6.RData")
} else {
  set.seed(127)
  pred0 <- predict(modFit0,testing) # knn model   
  pred2 <- predict(modFit2,testing) # tree bagging model
  pred3 <- predict(modFit3,testing) # random forest model
  pred4 <- predict(modFit4,testing) # boosting with trees model
  predDF <- data.frame(pred0, pred2, pred3, pred4, classe=testing$classe)
  modFit6 <- train(classe ~ . , method="rf", data=predDF)
  save(modFit6,file="modFit6.RData")
}

confusionMatrix(predict(modFit6,data.frame(pred0=predict(modFit0,evaluation),
                                           pred2=predict(modFit2,evaluation),
                                           pred3=predict(modFit3,evaluation),
                                           pred4=predict(modFit4,evaluation))),
                evaluation$classe) # accuracy is %98.4





