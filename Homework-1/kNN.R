euclideanDist <- function(a, b){
  d = 0
  for(i in c(1:(length(a)) ))
  {
    d = d + (a[[i]]-b[[i]])^2
  }
  d = sqrt(d)
  return(d)
}

knn_predict2 <- function(test_data, train_data, k_value, labelcol){
  pred <- c()  #empty pred vector 
  #LOOP-1
  for(i in c(1:nrow(test_data))){   #looping over each record of test data
    eu_dist =c()          #eu_dist & eu_char empty  vector
    eu_char = c()
    good = 0              #good & bad variable initialization with 0 value
    bad = 0
    
    #LOOP-2-looping over train data 
    for(j in c(1:nrow(train_data))){
 
      #adding euclidean distance b/w test data point and train data to eu_dist vector
      eu_dist <- c(eu_dist, euclideanDist(test_data[i,-c(labelcol)], train_data[j,-c(labelcol)]))
 
      #adding class variable of training data in eu_char
      eu_char <- c(eu_char, as.character(train_data[j,][[labelcol]]))
    }
    
    eu <- data.frame(eu_char, eu_dist) #eu dataframe created with eu_char & eu_dist columns
 
    eu <- eu[order(eu$eu_dist),]       #sorting eu dataframe to gettop K neighbors
    eu <- eu[1:k_value,]               #eu dataframe with top K neighbors
 
    tbl.sm.df<-table(eu$eu_char)
    cl_label<-  names(tbl.sm.df)[[as.integer(which.max(tbl.sm.df))]]
    
    pred <- c(pred, cl_label)
    }
    return(pred) #return pred vector
  }
  

accuracy <- function(test_data,labelcol,predcol){
  correct = 0
  for(i in c(1:nrow(test_data))){
    if(test_data[i,labelcol] == test_data[i,predcol]){ 
      correct = correct+1
    }
  }
  accu = (correct/nrow(test_data)) * 100  
  return(accu)
}

#load data
knn.df <- read.csv("D:\\MS Data Science\\CUNY\\CUNY\\CUNY MSDS\\Spring 2020\\DATA 622\\Homework-1\\icu.csv",
                   header = TRUE)

head(knn.df)

summary(knn.df)
str(knn.df)
knn.df$STA <- as.factor(knn.df$STA)

### Adding the new column - COMA
knn.df$COMA <- ifelse(knn.df$LOC == 2, 1, 0)

View(knn.df)
dim(knn.df)
##sum((knn.df$LOC == 1 | knn.df$LOC == 0) & knn.df$COMA == 0)
##sum(knn.df$LOC == 2 & knn.df$COMA == 1)
##unique(knn.df$LOC)

### Selecting only the required columns
knn.df <- knn.df[, c("TYP", "COMA", "AGE", "INF", "STA")]
summary(knn.df)

labelcol <- 5 
predictioncol<-labelcol+1
# create train/test partitions
n <- nrow(knn.df)

set.seed(2)
knn.df<- knn.df[sample(n),]

train.df <- knn.df[1:as.integer(0.7*n),]
test.df <- knn.df[as.integer(0.7*n +1):n,]

table(train.df[,labelcol]) / nrow(train.df)

table(test.df[,labelcol]) / nrow(test.df)

K = c(3,5,7,15,25,50) # number of neighbors to determine the class

predictions_K3 <- knn_predict2(test.df, train.df, K[1],labelcol) #calling knn_predict()

predictions_K5 <- knn_predict2(test.df, train.df, K[2],labelcol) #calling knn_predict()

predictions_K7 <- knn_predict2(test.df, train.df, K[3],labelcol) #calling knn_predict()

predictions_K15 <- knn_predict2(test.df, train.df, K[4],labelcol) #calling knn_predict()

predictions_K25 <- knn_predict2(test.df, train.df, K[5],labelcol) #calling knn_predict()

predictions_K50 <- knn_predict2(test.df, train.df, K[6],labelcol) #calling knn_predict()

test.df$predction_K3 <- predictions_K3
test.df$predction_K5 <- predictions_K5
test.df$predction_K7 <- predictions_K7
test.df$predction_K15 <- predictions_K15
test.df$predction_K25 <- predictions_K25
test.df$predction_K50 <- predictions_K50

##test.df[,predictioncol] <- predictions #Adding predictions in test data as 7th column
K_df <- cbind(as.data.frame(K), as.data.frame(c(6,7,8,9,10,11)))
colnames(K_df) <- c("K", "colnum")

accuracy_df <- data.frame()
for (i in 1:nrow(K_df))
  {
    accuracy_df <- rbind(accuracy_df, data.frame(K_df[i,1], accuracy(test.df, labelcol, K_df[i,2])))
  }


colnames(accuracy_df) <- c("K", "accuracy_K")

print(accuracy_df)

ggplot(accuracy_df, aes(x = K, y = accuracy_K)) + geom_point() +
  ylim(0,100) + scale_x_continuous(breaks = seq(0,50,by = 5)) +
  geom_line()


xtabs(~STA + predction_K3, data = test.df)
xtabs(~STA + predction_K5, data = test.df)
xtabs(~STA + predction_K7, data = test.df)
xtabs(~STA + predction_K15, data = test.df)
xtabs(~STA + predction_K25, data = test.df)
xtabs(~STA + predction_K50, data = test.df)

table(test.df[,labelcol])


##### Summary:
### As we see above in the accuracy graph, the accuracy is maxiumm for K=15, K=25 and K=50
### Upon looking further, we see that K=15, 25 and 50 do not predict 1s at all, but as 
### the data is highly imbalanced with a huge number of 0's and only 7 1's. Hence the value of
### K=15, K=25 and K=50 do no make any sense with this dataset.
### Even from above, K=3,5 and 7 do not give a good prediction as they predict 1's very poorly
### even if they give a very high accuracy, due to imbalanced data.
### So overall, knn does not seem to be a good fit overall with this imbalanced data.

### Trying with k=1
test.df2 <- test.df[,1:5]
predictions_K1 <- knn_predict2(test.df2, train.df, 1,labelcol) #calling knn_predict()
test.df2$predction_K1 <- predictions_K1

accuracy(test.df2, labelcol, 6)
xtabs(~STA + predction_K1, data = test.df2)

### As we see above k=1 still does a better job at predicting 1's than k=3,5,7,15,25,50 that were used earlier