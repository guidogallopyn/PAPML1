# R script for Peer assignement 1 of practical machine learning

# loads the scoring dataset
# loads the prediction models
# generates all plots and tables for report
# saves files 

library(caret)
library(ggplot2)
library(AppliedPredictiveModeling)

# set working directory appropriately
setwd("~/Documents/Courses/DataScience/projects/PracticalML")

load("data/data.RData")

# remove total features (duplicate with mag features)
training <- training[,-grep("total_",names(training))] 

# looking for outliers
outthere <- function(x) (max(x)-mean(x))/sd(x)
sapply(setdiff(names(training),c("classe")), function(x) outthere(training[[x]]))
#we remove outliers that are 10 signma from the mean (fill in NA)
#remove.outlier <- function(x) (training[[x]][ training[[x]] > mean(training[[x]])+10*sd(training[[x]]) ] <- NA)
#sapply(setdiff(names(training),c("classe")), function(x) remove.outlier(x) )
training$magnet_dumbbell_mag[ training$magnet_dumbbell_mag > 800] <-NA
training$magnet_dumbbell_mag[ training$magnet_dumbbell_mag < 200] <-NA
training[["gyros_dumbbell_mag"]][ training$gyros_dumbbell_mag > 3] <-NA
training[["gyros_forearm_mag"]][ training$gyros_forearm_mag > 10] <-NA


# plotting predictors => not much to see?!
transparentTheme(trans = .05,pchSize = 0.5)
featurePlot(x=training[,grep("dumbbell",names(training))], y=training$classe, plot="pairs")
featurePlot(x=training[,grep("forearm",names(training))], y=training$classe, plot="pairs")
featurePlot(x=training[,grep("_arm",names(training))], y=training$classe, plot="pairs") # nice one
featurePlot(x=training[,grep("belt",names(training))], y=training$classe, plot="pairs") # nice one

# use predictors of model1 classification tree 
cart <- train(classe ~ . , method="rpart",data=training)
png(filename = "report/plot2.png", width = 640, height = 480)

transparentTheme(trans = .05,pchSize = 1)
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x=training[,setdiff(cart$finalModel$frame$var,c("<leaf>"))], y=training$classe, plot="pairs",auto.key = list(columns = 5))

featurePlot(x=training[,setdiff(cart$finalModel$frame$var,c("<leaf>"))], y=training$classe, plot="box")


featurePlot(x=training[,grep("dumbbell",names(training))], y=training$classe, plot="box")
featurePlot(x=training[,grep("forearm",names(training))], y=training$classe, plot="box")
featurePlot(x=training[,grep("_arm",names(training))], y=training$classe, plot="box")
featurePlot(x=training[,grep("belt",names(training))], y=training$classe, plot="box")



qplot(magnet_belt_mag,colour=classe,geom="density",data=training)
qplot(accel_arm_mag,accel_dumbbell_mag,colour=classe,data=training)





