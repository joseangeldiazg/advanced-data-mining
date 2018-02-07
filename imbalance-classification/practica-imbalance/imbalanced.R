# imbalanced.R
# Implementation and evaluation of imbalanced classification techniques 
# Programming code courtesy by Sarah Vluymans, Sarah.Vluymans@UGent.be

## load the subclus dataset
subclus <- read.table("subclus.txt", sep=",")
colnames(subclus) <- c("Att1", "Att2", "Class")

# determine the imbalance ratio
unique(subclus$Class)
nClass0 <- sum(subclus$Class == 0)
nClass1 <- sum(subclus$Class == 1)
IR <- nClass1 / nClass0
IR

# visualize the data distribution
plot(subclus$Att1, subclus$Att2)
points(subclus[subclus$Class==0,1],subclus[subclus$Class==0,2],col="red")
points(subclus[subclus$Class==1,1],subclus[subclus$Class==1,2],col="blue")  

# Set up the dataset for 5 fold cross validation.
# Make sure to respect the class imbalance in the folds.
pos <- (1:dim(subclus)[1])[subclus$Class==0]
neg <- (1:dim(subclus)[1])[subclus$Class==1]

CVperm_pos <- matrix(sample(pos,length(pos)), ncol=5, byrow=T)
CVperm_neg <- matrix(sample(neg,length(neg)), ncol=5, byrow=T)

CVperm <- rbind(CVperm_pos, CVperm_neg)

# Base performance of 3NN
library(class)
knn.pred = NULL
for( i in 1:5){
  predictions <- knn(subclus[-CVperm[,i], -3], subclus[CVperm[,i], -3], subclus[-CVperm[,i], 3], k = 3)
  knn.pred <- c(knn.pred, predictions)
}
acc <- sum((subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) 
           | (subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2)) / (nClass0 + nClass1)
tpr <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean <- sqrt(tpr * tnr)


# 1. ROS
knn.pred = NULL
for( i in 1:5){
  
  train <- subclus[-CVperm[,i], -3]
  classes.train <- subclus[-CVperm[,i], 3] 
  test  <- subclus[CVperm[,i], -3]
  
  # randomly oversample the minority class (class 0)
  minority.indices <- (1:dim(train)[1])[classes.train == 0]
  to.add <- dim(train)[1] - 2 * length(minority.indices)
  duplicate <- sample(minority.indices, to.add, replace = T)
  for( j in 1:length(duplicate)){
    train <- rbind(train, train[duplicate[j],])
    classes.train <- c(classes.train, 0)
  }  
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.ROS <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.ROS <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.ROS <- sqrt(tpr.ROS * tnr.ROS)

# 2. RUS
knn.pred = NULL
for( i in 1:5){
  
  train <- subclus[-CVperm[,i], -3]
  classes.train <- subclus[-CVperm[,i], 3] 
  test  <- subclus[CVperm[,i], -3]
  
  # randomly undersample the minority class (class 1)
  majority.indices <- (1:dim(train)[1])[classes.train == 1]
  to.remove <- 2* length(majority.indices) - dim(train)[1]
  remove <- sample(majority.indices, to.remove, replace = F)
  train <- train[-remove,] 
  classes.train <- classes.train[-remove]
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.RUS <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.RUS <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.RUS <- sqrt(tpr.RUS * tnr.RUS)

# Visualization (RUS on the full dataset)
subclus.RUS <- subclus
majority.indices <- (1:dim(subclus.RUS)[1])[subclus.RUS$Class == 1]
to.remove <- 2 * length(majority.indices) - dim(subclus.RUS)[1]
remove <- sample(majority.indices, to.remove, replace = F)
subclus.RUS <- subclus.RUS[-remove,] 

plot(subclus.RUS$Att1, subclus.RUS$Att2)
points(subclus.RUS[subclus.RUS$Class==0,1],subclus.RUS[subclus.RUS$Class==0,2],col="red")
points(subclus.RUS[subclus.RUS$Class==1,1],subclus.RUS[subclus.RUS$Class==1,2],col="blue") 


# 1.4.1 Distance function
distance <- function(i, j, data){
  sum <- 0
  for(f in 1:dim(data)[2]){
    if(is.factor(data[,f])){ # nominal feature
      if(data[i,f] != data[j,f]){
        sum <- sum + 1
      }
    } else {
      sum <- sum + (data[i,f] - data[j,f]) * (data[i,f] - data[j,f])
    }
  }
  sum <- sqrt(sum)
  return(sum)
}