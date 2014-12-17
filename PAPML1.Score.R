# R script for Peer assignement 1 of practical machine learning

# loads the scoring dataset
# loads the prediction models
# predicts outcome for 20 data points with all models
# displays scores
# saves files 

library(caret)

#functions

# set working directory appropriately
setwd("~/Documents/Courses/DataScience/projects/PracticalML")

load("data/scoring.RData")

load("models/modFit0.RData")
load("models/modFit1.RData")
load("models/modFit2.RData")
load("models/modFit3.RData")
load("models/modFit4.RData")
load("models/modFit6.RData")

score <- data.frame(problem_id=scoring$problem_id, 
                    pred0 = predict(modFit0,scoring),
                    pred1 = predict(modFit1,scoring), 
                    pred2 = predict(modFit2,scoring), 
                    pred3 = predict(modFit3,scoring), 
                    pred4 = predict(modFit4,scoring)) 
           
score<- transform(score,pred6=predict(modFit6,score))
colnames(score) <- c(" problem_id","knn","rpart","treebag","rf","gbm","combine")

print(score)

# we choose the combined model for the final submission

for(i in 1:nrow(score)){
    filename = paste0("score/problem_id_",score[i,1],".txt")
    write.table(as.character(score$combine[i]),file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}

# score submitted on 12/14/2014 and are all correct
