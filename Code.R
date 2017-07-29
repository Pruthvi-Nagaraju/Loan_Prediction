#Analytics Vidya Loan Prediction Hackathon

library(VIM)
library(DMwR)
library(rpart)
library(mice)
library(caret)
library(randomForest)
library(PRROC)
library(pROC)


train= read.csv("FakePath/train.csv")
test=read.csv("FakePath/test.csv")

table(train$Loan_Status)

#function to find number of missing values in each column
missing<-function(x){
  n=ncol(x)
  missing=0
  cname=1
  for(i in 1:n){
    missing[i]=sum(is.na(x[,i]) | x[,i]=="")
    cname[i]=colnames(x)[i]
  }
  df=data.frame(cname,missing)
  return(df)
}


summary(train )
missing(train)

#Columns with missing values
#Gender-13
#Married-3
#Dependents-15
#Self_Employed-32
#LoanAmount- 22
#Loan_Amount_Term- 14
#Credit_History- 50

Mode <- function(num) {
  unique_num <- unique(num)
  unique_num [which.max(tabulate(match(num, unique_num )))]
}

#Imputing missing values in trian data
train$Gender[train$Gender==""]= Mode(train$Gender)
train$Married[train$Married==""]= Mode(train$Married)
train$Dependents[train$Dependents==""]=Mode(train$Dependents)
train$LoanAmount[is.na(train$LoanAmount)]= mean(train$LoanAmount, na.rm= TRUE)
train$Self_Employed=ifelse(train$Self_Employed=="", "unknown", train$Self_Employed )
train$Loan_Amount_Term[is.na(train$Loan_Amount_Term)]= Mode(train$Loan_Amount_Term)

#Imputing missing values in Credit_History using mice( multivariate imputation by chained equations)

miceMod <- mice(train[, !names(train) %in% "medv"], method="rf")  # perform mice imputation, based on random forests.
train <- complete(miceMod) 
missing(test)


#Imputing missing values in test data
test$Gender[test$Gender==""]= Mode(test$Gender)
test$Dependents[test$Dependents==""]=Mode(test$Dependents)
test$LoanAmount[is.na(test$LoanAmount)]= mean(test$LoanAmount, na.rm= TRUE)
test$Self_Employed=ifelse(test$Self_Employed=="", "unknown", test$Self_Employed )
test$Loan_Amount_Term[is.na(test$Loan_Amount_Term)]= Mode(test$Loan_Amount_Term)
miceMod <- mice(test[, !names(test) %in% "medv"], method="rf")  # perform mice imputation, based on random forests.
test <- complete(miceMod) 
missing(test)

#Building a base Logistic Regression model

trainIndex=createDataPartition(train$Loan_Status, p = 0.7,list = F,times = 1)

train1= train[trainIndex,]
test1= train[-trainIndex,]


train1=subset(train1, select= -Loan_ID)
train1
Model1= glm(train1$Loan_Status ~ . , data= train1, family = binomial)
summary(Model1)


test1=subset(test1, select= -Loan_ID)

Pred1= predict(Model1, test1)
PredClass= ifelse(Pred1 > 0.5, 'Y','N')

table(PredClass, test1$Loan_Status)
caret::confusionMatrix(PredClass,test1$Loan_Status)

# using k fold cross validation technique

k=10
n=floor(nrow(train)/k)
log_accuracy<-c()
train=subset(train, select= -Loan_ID)
#using 10-fold cross validation
for (i in 1:k){
  s1 = ((i-1)*n+1)
  s2 = (i*n)
  subset = s1:s2
  log_train<- train[-subset,]
  log_test<- train[subset,]
  log_fit<-glm(Loan_Status ~ ., family=binomial, data = log_train)
  log_pred <- predict(log_fit, log_test)
  log_pred_class <- ifelse(log_pred>0.5, 'Y', 'N')
  print(paste("Logistic Accuracy: ",1-sum(log_test$Loan_Status!=log_pred_class)/nrow(log_test)))
  log_accuracy[i]<- 1- (sum(log_test$Loan_Status!=log_pred_class)/nrow(log_test))
}
#taking the mean of all the 10 model to estimate the accuracy of the model
print(paste("The accuracy of the logistic Model is: ",mean(log_accuracy)))

#Accuracy= 0.7918


#Random Forest

#Tuning Parameters
ntree = c(500,600,700,800,900,1000)
mtry = c(11,12,13)
nodesize = c(4,5,6)
k=10
n=floor(nrow(train)/k)
rf_accuracy_poss=c()
rf_accuracy_all=data.frame("No_of_Trees" = integer(0),"No_of_features"=integer(0),
                           "Nodesize" = integer(0), "Accuracy"= numeric(0))

train$Self_Employed=as.factor(train$Self_Employed)
#using 10-fold cross validation
for (t in ntree){
  for (m in mtry){
    for (n in nodesize){
      for (i in 1:k){
        s1 = ((i-1)*n+1)
        s2 = (i*n)
        subset = s1:s2
        rf_train<- train[-subset,]
        rf_test<- train[subset,]
        rf_fit<-randomForest(x=rf_train[,-c(12)], y = rf_train[,c(12)], 
                             ntree = t, mtry = m, nodesize = n)
        rf_pred <- predict(rf_fit, rf_test, type = "prob")[,2]
        rf_pred_class <- ifelse(rf_pred>0.5, 'Y', 'N')
        rf_accuracy_poss[i]<- 1 - sum(rf_test$Loan_Status!=rf_pred_class)/nrow(rf_test)
      } 
      print(paste("number of trees: ",t,"number of features: ", m, "nodesize :", n,
                  "Cross-Validation mean Accuracy",mean(rf_accuracy_poss)))
      rf_accuracy_all<- rbind(rf_accuracy_all, data.frame(t,m,n,mean(rf_accuracy_poss)))
    }
  }
}


print("The best parameters and the accuracies are :")
rf_accuracy_all[rf_accuracy_all$mean.rf_accuracy_poss. == max(rf_accuracy_all$mean.rf_accuracy_poss.),]

#Building the model using the best parameters
k=10
n=floor(nrow(train)/k)
rf_accuracy<-c()
for (i in 1:k){
  s1 = ((i-1)*n+1)
  s2 = (i*n)
  subset = s1:s2
  rf_train<- train[-subset,]
  rf_test<- train[subset,]
  rf_fit<-randomForest(x=rf_train[,-c(12)], y = rf_train[,c(12)],
                       ntree = 600,mtry = 12,nodesize = 4)
  rf_pred <- predict(rf_fit, rf_test, type = "prob")[,2]
  rf_pred_class <- ifelse(rf_pred>0.5, 'Y', 'N')
  print(paste("RF Accuracy: ",1 - sum(rf_test$Loan_Status!=rf_pred_class)/nrow(rf_test)))
  rf_accuracy[i]<- 1- (sum(rf_test$Loan_Status!=rf_pred_class)/nrow(rf_test))
}

print(paste("The accuracy of the Random Forest is: ", mean(rf_accuracy)))
# The final accuracy from Random Forest is 78.36%



















