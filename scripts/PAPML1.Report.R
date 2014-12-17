# R script for Peer assignement 1 of practical machine learning

# loads the scoring dataset
# loads the prediction models
# generates all plots and tables for report
# saves files 

library(caret)
library(rattle)
library(AppliedPredictiveModeling)

#functions

# set working directory appropriately
setwd("~/Documents/Courses/DataScience/projects/PracticalML")

load("data/data.RData")
load("models/modFit0.RData")
load("models/modFit1.RData")
load("models/modFit2.RData")
load("models/modFit3.RData")
load("models/modFit4.RData")
load("models/modFit6.RData")


# figure 1: 2 principal components of the predictors
preProc <- preProcess(training[,-17], method="pca", pcaComp=2)
pcdata <- transform(predict(preProc,training[,-17]),classe=training$classe)
png(filename = "report/plot1.png", width = 640, height = 480)
ggplot(pcdata,aes(PC1,PC2)) + 
  geom_point(aes(colour=classe),alpha=0.15,size=2) +
  xlim(-5,5) + ylim(-5,5)
dev.off()

# figure 2: Feature plot of predictors used by a classification tree trained on data
# need to control color better A and E are both redish
set.seed(127)
cart <- train(classe ~ . , method="rpart",data=training)
png(filename = "report/plot2.png", width = 640, height = 480)
transparentTheme(trans = .05,pchSize = 0.5)
featurePlot(x=training[,setdiff(cart$finalModel$frame$var,c("<leaf>"))], y=training$classe, plot="pairs",,auto.key = list(columns = 5))
dev.off()

# figure 3: Classification tree
png(filename = "report/plot3.png", width = 640, height = 480)
fancyRpartPlot(modFit1$finalModel)
dev.off()

# accuracy report on resamples
resamps <- resamples(list(KNN = modFit0, TreeBag = modFit2, RF = modFit3, GBM = modFit4))
summary(resamps)
save(resamps, file="report/resamps.RData") 

# acccuracy of combined model on test set
save(print(modFit6),file="report/print6.RData")

# where does combined model differ from rf model
pred <- data.frame( pred0 = predict(modFit0,testing),  # knn model
                    pred2 = predict(modFit2,testing),  # tree bagging model
                    pred3 = predict(modFit3,testing),  # random forest model
                    pred4 = predict(modFit4,testing))  # boosting with trees model
pred <- transform(pred,pred6=predict(modFit6,pred),    # prediction of combines model
                       classe=testing$classe)          # reference
cm6 <- confusionMatrix(pred$pred3,pred$pred6)
save(cm6,file="report/cm6.RData")

# final evaluation
pred <- data.frame( pred0 = predict(modFit0,evaluation),  # knn model
                    pred1 = predict(modFit1,evaluation),  # simple tree model
                    pred2 = predict(modFit2,evaluation),  # tree bagging model
                    pred3 = predict(modFit3,evaluation),  # random forest model
                    pred4 = predict(modFit4,evaluation))  # boosting with trees model
pred <- transform(pred,pred6=predict(modFit6,pred),       # prediction of combines model
                  classe=evaluation$classe)               # reference
cm6e <- confusionMatrix(pred$pred6,pred$classe)

acc <- data.frame(
           ModelInfo=sapply(c(0:4,6),function(x) eval(parse(text = paste0("modFit",x,"$modelInfo$label")))),
           Method=sapply(c(0:4,6),function(x) eval(parse(text = paste0("modFit",x,"$method")))),
           Accuracy=sapply(paste0("pred",c(0:4,6)), 
                           function(x) confusionMatrix(pred[[x]],pred$classe)$overall[["Accuracy"]])
          )
rownames(acc)<-NULL
save(acc,file="report/acc.RData")

save(resamps,modFit6,cm6,cm6e,acc,file="report/report.RData")
