library(dplyr)
library(ggplot2)
library(stringr)
library(class)
library(caret)
library(pROC)
library(ROCR)
library(rpart)
library(rpart.plot)
library(MASS)
library(e1071)
library(bnlearn)

hw2_data <- read.csv("https://raw.githubusercontent.com/deepakmongia/Data622/master/Homework-2/Data/HW-2_data.csv",
                     header = TRUE)

head(hw2_data)
#View(hw2_data)
str(hw2_data)
summary(hw2_data)
dim(hw2_data)
any(is.na(hw2_data))

hw2_data$Y <- str_remove_all(hw2_data$Y, "[ ]") %>% as.factor()
hw2_data$label <- str_remove_all(hw2_data$label, "[ ]") %>% as.factor()
summary(hw2_data)

ggplot(data = hw2_data, aes(Y,as.factor(X))) + 
  geom_point(color=as.character(tolower(hw2_data$label))) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

#sum(hw2_data$label == "BLACK")
#sum(hw2_data$label == "BLUE")


#### Split data into training and test
set.seed(123)
train.rows <- createDataPartition(y = hw2_data$label, p = 0.7, list = FALSE)
train_data2 <- hw2_data[train.rows,]
test_data2 <- hw2_data[-train.rows,]

### Building a separate copy of data for kNN as kNN needs the independent features to be numeric,
### as it uses the distance to get the predictions
hw2_data_knn <- hw2_data
head(hw2_data_knn)
hw2_data_knn$Y <- as.integer(hw2_data_knn$Y)
head(hw2_data_knn)

set.seed(123)
train.rows.knn <- createDataPartition(y = hw2_data_knn$label, p = 0.7, list = FALSE)
train_data2_knn <- hw2_data_knn[train.rows.knn,]
test_data2_knn <- hw2_data_knn[-train.rows.knn,]

### Accuracy, tpr and fpr functions - will be used to calculate these metrics for all the algorithms that will be used
### Taking black as negative and blue as positive

accuracy_func <- function(actual_labels, model_labels) {
  True_positive_plus_negative <- sum(actual_labels == model_labels)
  False_positive_plus_negative <- sum(actual_labels != model_labels)
  
  Accuracy <- True_positive_plus_negative / (True_positive_plus_negative + False_positive_plus_negative)
  return(Accuracy)
}

tpr_func <- function(actual_labels, model_labels) {
  True_positive <- sum(actual_labels == 'BLUE' & model_labels == 'BLUE')
  False_negative <- sum(actual_labels == 'BLUE' & model_labels == 'BLACK')
  
  true_positive_rate <- True_positive / (True_positive + False_negative)
  
  return(true_positive_rate)
}

fpr_func <- function(actual_labels, model_labels) {
  True_negative <- sum(actual_labels == 'BLACK' & model_labels == 'BLACK')
  False_positive <- sum(actual_labels == 'BLACK' & model_labels == 'BLUE')
  
  false_positive_rate <- False_positive / (True_negative + False_positive)
  
  return(false_positive_rate)
}

### Creating a dataframe to store the metrics from the algorithms
algorithm_metrics <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(algorithm_metrics) <- c("algorithm", "AUC", "ACCURACY", "TPR", "FPR")

#### RUNNING THE ALGORITHMS

### Run kNN model

predict_knn_prob <- knn(train = train_data2_knn[,-3], test = test_data2_knn[,-3], 
                   cl = train_data2_knn$label, k = 3, prob = TRUE)

predict_knn_prob

prob_knn <- attr(predict_knn_prob, "prob")

prob_knn <- ifelse(predict_knn_prob == "BLUE", prob_knn, 1 - prob_knn)

roc_knn <- roc(response = test_data2_knn$label, predictor = prob_knn)

plot(roc_knn)

table(predict_knn_prob, test_data2_knn$label)

auc_knn <- as.numeric(auc(roc_knn))
accuracy_knn <- accuracy_func(test_data2_knn$label, predict_knn_prob)
tpr_knn <- tpr_func(test_data2_knn$label, predict_knn_prob)
fpr_knn <- fpr_func(test_data2_knn$label, predict_knn_prob)

algorithm_metrics <- rbind(algorithm_metrics, setNames(data.frame("KNN", auc_knn, accuracy_knn, tpr_knn, fpr_knn), names(algorithm_metrics)))

#### TREE algorithm
tree_model <- rpart(formula = label ~ ., data = train_data2, 
                    method = "class")

rpart.plot(tree_model)

tree_predict_prob <- predict(object = tree_model, 
                        newdata = test_data2,
                        type = "prob")

tree_predict_prob <- as.data.frame(tree_predict_prob)$BLUE
tree_predict <- ifelse(tree_predict_prob > 0.5, "BLUE", "BLACK")


auc_knn <- as.numeric(auc(roc_knn))
accuracy_knn <- accuracy_func(test_data2_knn$label, predict_knn_prob)
tpr_knn <- tpr_func(test_data2_knn$label, predict_knn_prob)
fpr_knn <- fpr_func(test_data2_knn$label, predict_knn_prob)

roc_tree <- roc(response = test_data2$label, predictor = tree_predict_prob)

plot(roc_tree)

table(tree_predict, test_data2$label)

summary(tree_model)

auc_tree <- as.numeric(auc(roc_tree))
accuracy_tree <- accuracy_func(test_data2$label, tree_predict)
tpr_tree <- tpr_func(test_data2$label, tree_predict)
fpr_tree <- fpr_func(test_data2$label, tree_predict)

algorithm_metrics <- rbind(algorithm_metrics, setNames(data.frame("TREE", auc_tree, accuracy_tree, tpr_tree, fpr_tree), names(algorithm_metrics)))

### LDA
lda_model <- lda(formula = label~.,
                 data = train_data2)

lda_predict <- predict(lda_model,
                       newdata = test_data2)

lda_predict$class

xtabs(~lda_predict$class + test_data2$label)

prob_lda <- as.data.frame(lda_predict$posterior)$BLUE

roc_lda <- roc(response = test_data2_knn$label, predictor = prob_lda)

plot(roc_lda)

auc_lda <- as.numeric(auc(roc_lda))
accuracy_lda <- accuracy_func(test_data2$label, lda_predict$class)
tpr_lda <- tpr_func(test_data2$label, lda_predict$class)
fpr_lda <- fpr_func(test_data2$label, lda_predict$class)

algorithm_metrics <- rbind(algorithm_metrics, setNames(data.frame("LDA", auc_lda, accuracy_lda, tpr_lda, fpr_lda), names(algorithm_metrics)))

### SVM

svm_model <- svm(label~., data = train_data2, probability = TRUE)
summary(svm_model)

svm_model

svm_predict <- predict(svm_model,
                       newdata = test_data2,
                       probability = TRUE)

svm_predict
prob_svm <- as.data.frame(attr(svm_predict, "probabilities"))$BLUE

roc_svm <- roc(response = test_data2$label, predictor = prob_svm)

plot(roc_svm)

accuracy_func(test_data2$label, svm_predict)
tpr_func(test_data2$label, svm_predict)
fpr_func(test_data2$label, svm_predict)

table(Predicted = svm_predict, Actual = test_data2$label)

### Let's make better SVM model
svm_model_2 <- svm(label~., data = train_data2, probability = TRUE,
                   kernel = 'linear')
summary(svm_model_2)

svm_predict_2 <- predict(svm_model_2,
                       newdata = test_data2,
                       probability = TRUE)
svm_predict_2
accuracy_func(test_data2$label, svm_predict_2)
tpr_func(test_data2$label, svm_predict_2)
fpr_func(test_data2$label, svm_predict_2)

table(Predicted = svm_predict_2, Actual = test_data2$label)

### Even better
svm_model_3 <- svm(label~., data = train_data2, probability = TRUE,
                   kernel = 'polynomial')
summary(svm_model_3)

svm_predict_3 <- predict(svm_model_3,
                         newdata = test_data2,
                         probability = TRUE)
svm_predict_3

accuracy_func(test_data2$label, svm_predict_3)
tpr_func(test_data2$label, svm_predict_3)
fpr_func(test_data2$label, svm_predict_3)

table(Predicted = svm_predict_3, Actual = test_data2$label)

prob_svm3 <- as.data.frame(attr(svm_predict_3, "probabilities"))$BLUE
roc_svm3 <- roc(response = test_data2$label, predictor = prob_svm3)

auc_svm <- as.numeric(auc(roc_svm3))
accuracy_svm <- accuracy_func(test_data2$label, svm_predict_3)
tpr_svm <- tpr_func(test_data2$label, svm_predict_3)
fpr_svm <- fpr_func(test_data2$label, svm_predict_3)

algorithm_metrics <- rbind(algorithm_metrics, setNames(data.frame("SVM", auc_svm, accuracy_svm, tpr_svm, fpr_svm), names(algorithm_metrics)))


### Logistic Regession

lr_model <- glm(label~., data = train_data2,
                family = binomial(link = 'logit'))

summary(lr_model)

lr_predict_prob <- predict(lr_model,
                           newdata = test_data2,
                           type = 'response')

lr_predict <- ifelse(lr_predict_prob > 0.5, "BLUE", "BLACK")
lr_predict
table(Predicted = lr_predict, Actual = test_data2$label)

roc_lr <- roc(response = test_data2$label, predictor = lr_predict_prob)

plot(roc_lr)

auc_lr <- as.numeric(auc(roc_lr))
accuracy_lr <- accuracy_func(test_data2$label, lr_predict)
tpr_lr <- tpr_func(test_data2$label, lr_predict)
fpr_lr <- fpr_func(test_data2$label, lr_predict)

algorithm_metrics <- rbind(algorithm_metrics, setNames(data.frame("LR", auc_lr, accuracy_lr, tpr_lr, fpr_lr), names(algorithm_metrics)))


### Naive Bayes
nb_model <- naiveBayes(label~., data = train_data2, prob = TRUE)

nb_model

nb_predict_prob <- predict(nb_model,
                      newdata = test_data2,
                      "raw")

summary(nb_model)

nb_predict_prob <- as.data.frame(nb_predict_prob)$BLUE
nb_predict <- if_else(nb_predict_prob > 0.5, "BLUE", "BLACK")

table(Predicted = nb_predict, Actual = test_data2$label)


roc_nb <- roc(response = test_data2$label, predictor = nb_predict_prob)

plot(roc_nb)

auc_nb <- as.numeric(auc(roc_nb))
accuracy_nb <- accuracy_func(test_data2$label, nb_predict)
tpr_nb <- tpr_func(test_data2$label, nb_predict)
fpr_nb <- fpr_func(test_data2$label, nb_predict)

algorithm_metrics <- rbind(algorithm_metrics, setNames(data.frame("NB", auc_nb, accuracy_nb, tpr_nb, fpr_nb), names(algorithm_metrics)))


###### CONCLUSIONS:
print(algorithm_metrics)

head(hw2_data, 10)

### As we see above, there is no fixed pattern in the data, that is the reason KNN performed the best
### with this data. 
### The final best algorithm that works the best or any data set will depend a lot on the data. We did
### not expect Tree / LDA / SVM / Logistic Regression / Naive Bayes algorithms to perfomr poorly as compared
### to the kNN model. But again as the data does not have a fixed pattern, and kNN works on the nearest
### neigbhor, this seems to be the best algorithm for the data we have at hand.