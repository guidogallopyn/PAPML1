# R script fprPeer assignement 1 of practical machine learning 


# loads the pml-training.csv dataset
# pre-processes pml-training.csv dataset (removing unneeded variables)
# partitions in training, testing and evaluaiton data sets
# saves the preprocessed and partitioned data sets
# loads the pml-testing.csv dataset
# pre-processes pml-testing.csv dataset (removing unneeded variables)
# saves the preprocessed scoring set

#functions
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
setwd("~/Documents/Courses/DataScience/projects/PracticalML/data")

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
  
  # remove "total" features (duplicate with mag features)
  training <- training[,-grep("total_",names(training))] 
  
  # dataset partitioning in train, test sets
  set.seed(1235)
  inTrain  <- createDataPartition(y=data$classe, p=0.7, list=FALSE)
  training <- data[inTrain,]
  evaluation <- data[-inTrain,]
  inTrain  <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
  testing  <- training[-inTrain,]
  training <- training[inTrain,]
  
  save(training,testing,evaluation,file="data.RData")

  # preparing the scoring set
  data <- read.csv("pml-testing.csv")
  vars <- c(setdiff(vars,c("classe")),"problem_id")
  data <- add.magnitudes(data[vars])
  scoring <- data[,-grep("_(x|y|z)",names(data))]  # remove (xyz) features  
  save(scoring,file="scoring.RData")
}


