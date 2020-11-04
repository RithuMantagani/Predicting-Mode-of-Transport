library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(lattice)
library(DataExplorer)
library(grDevices)
library(factoextra)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ranger)
library(Metrics)
library(ROCit)
library(kableExtra)
library(DataExplorer)
library(fpc)
library(NbClust)
library(caTools)
library(rattle)
library(RColorBrewer)
library(data.table)
library(ROCR)
library(cluster)
library(DMwR)
library(corrplot)
library(car)
library(mice)
library(xgboost)
library(ineq)
library(e1071)
library(ipred)

##EDA##
setwd("F:/GREAT LEARNING/MACHINE LEARNING/Project -Predicting mode of Transport (ML)/GL- Solution")
getwd()
cars=read.csv("Cars_edited.csv",header = TRUE)
View(cars)
summary(cars)
cars$Gender=as.factor(cars$Gender)
cars$Transport=as.factor(cars$Transport)
str(cars)
sum(is.na(cars))

##Univariate Analysis##
colnames(cars[,sapply(cars, is.numeric)]) 
ggplot(cars, aes(x = Age)) + geom_histogram(fill = "lightgreen", col = "cyan")
ggplot(cars, aes(x = Engineer)) + geom_histogram(fill = "lightgreen", col = "cyan")
ggplot(cars, aes(x = MBA)) + geom_histogram(fill = "lightgreen", col = "cyan")
ggplot(cars, aes(x = Work.Exp)) + geom_histogram(fill = "lightgreen", col = "cyan")
ggplot(cars, aes(x = Salary)) + geom_histogram(fill = "lightgreen", col = "cyan")
ggplot(cars, aes(x = Distance)) + geom_histogram(fill = "lightgreen", col = "cyan")
ggplot(cars, aes(x = license)) + geom_histogram(fill = "lightgreen", col = "cyan")

colnames(cars[,sapply(cars, is.factor)])
ggplot(cars, aes(x = Gender, fill = Gender)) + geom_bar()
ggplot(cars, aes(x = Transport, fill = Transport)) + geom_bar()
ggplot(cars, aes(x = as.factor(Engineer), fill = as.factor(Engineer))) + geom_bar()
ggplot(cars, aes(x = as.factor(MBA), fill = as.factor(MBA))) + geom_bar()
ggplot(cars, aes(x = as.factor(license), fill = as.factor(license))) + geom_bar()

ggplot(cars, aes(y = Salary)) + geom_boxplot()
ggplot(cars, aes(y = Distance)) + geom_boxplot()
ggplot(cars, aes(y = Work.Exp)) + geom_boxplot()

##Bi-Variate Analysis##
boxplot(cars$Work.Exp ~ cars$Gender)
boxplot(cars$Salary~cars$Transport, main="Salary vs Transport")
boxplot(cars$Age~cars$Transport, main="Age vs Transport")
boxplot(cars$Distance~cars$Transport, main="Distance vs Transport")
table(cars$license,cars$Transport)
cor(cars$Age, cars$Work.Exp)
table(cars$Gender,cars$Transport)

ggplot(cars, aes(x = Gender)) + geom_bar(aes(fill = Transport), position = "dodge")
ggplot(cars, aes(x = Engineer)) + geom_bar(aes(fill = Transport), position = "dodge")
ggplot(cars, aes(x = MBA)) + geom_bar(aes(fill = Transport), position = "dodge")
ggplot(cars, aes(x = license)) + geom_bar(aes(fill = Transport), position = "dodge")


##Outlier Treatment##
plot_boxplot(cars, by="Transport" , geom_boxplot_args = list("outlier.color" = "red",
                                                             fill="blue"))

##Missing Values##
sapply(cars,function(x) sum(is.na(x)))
cars = knnImputation(cars, 5)
any(is.na(cars))

##Multicollinearity for Numerical Value##
cars.new = cars %>% select_if(is.numeric)
a=round(cor(cars.new),2)
corrplot(a,method="number",type ="upper")

##All variables##
carsnew1=cars
carsnew1$Gender=as.integer(carsnew1$Gender)
carsnew1$Transport=as.integer(carsnew1$Transport)
b= round(cor(carsnew1),2)
corrplot(b,method = "number",type = "upper")

##Multicollinearity Treatment##
carsnew1$Work.Exp<-NULL
lModel=lm(Transport~.,data = carsnew1)
summary(lModel)
vif(lModel)


##Checking the proportion of target variable in actual dataset##
cars$Transport= ifelse(cars$Transport=='Car',1,0)
table(cars$Transport)

sum(cars$Transport==1)/nrow(cars)

cars$Transport=as.factor(cars$Transport)
cars$Engineer=as.factor(cars$Engineer)
cars$MBA=as.factor(cars$MBA)
cars$license=as.factor(cars$license)

##Checking Target variable proportion in overall data set##
prop.table(table(cars$Transport))

##Applying SMOTE for Data Balancing##
balanced.cars=SMOTE(Transport~.,perc.over = 300,cars,k=3,perc.under = 370)
table(balanced.cars$Transport)

##Lets create a subset and create Train and Test Dataset##
cars2=balanced.cars
##Split the data into Train and Test 
set.seed(500)
spl=sample.split(balanced.cars$Transport,SplitRatio = 0.7)
train=subset(balanced.cars,spl==TRUE)
test=subset(balanced.cars,spl=FALSE)
##Checking Correlation##
cdata=carsnew1
cdata$Gender=as.integer(cdata$Gender)
cdata$Transport=as.integer(cdata$Transport)
corrplot(cor(cdata))


##Modeling##
#Logistic Regression#

##Building Logistic Regression Model Based on all variables ##
logreg=glm(balanced.cars$Transport~.,data=balanced.cars,family=binomial)
summary(logreg)
#Checking for LOgistic Regression Model Multicollinearity#
vif(logreg)
## LR after removing highly correlated variables - Salary&Work.Exp##
logreg2=glm(balanced.cars$Transport~.-Salary-Work.Exp,data=balanced.cars,family=binomial)
summary(logreg2)
vif(logreg2)

## LR built after removing all insignificant variables ##
logreg3=glm(balanced.cars$Transport~.-Salary-Work.Exp-MBA-license-Gender-Engineer,data=balanced.cars,family=binomial)
summary(logreg3)
vif(logreg3)

##Regression Model Performance on Train and Test Data##
##Confusion Matrix of Train Data with 0.5 Threshold##
ctrain=predict(logreg3,newdata=train[,-9],type="response")
tab1=table(train$Transport,ctrain>0.5)
sum(diag(tab1))/sum(tab1)

##CM on Test Data##
ctest=predict(logreg3,newdata=test[,-9],type="response")
tab2=table(test$Transport,ctest>0.5)
sum(diag(tab2))/sum(tab2)

##ROC on Train Data##
predictroc1=predict(logreg3,newdata=train)
pred1=prediction(predictroc1,train$Transport)
perf1=performance(pred1,"tpr","fpr")
plot(perf1,colorize=T)
as.numeric(performance(pred1,"auc")@y.values)

##Roc on Test Data##
predictroc2=predict(logreg3,newdata=test)
pred2=prediction(predictroc2,test$Transport)
perf2=performance(pred2,"tpr","fpr")
plot(perf2,colorize=T)
as.numeric(performance(pred2,"auc")@y.values)

##KS-Chart##
##Ks ON Train##
KSLRTrain=max(attr(perf1,'y.values')[[1]]-attr(perf1,'x.values')[[1]])
KSLRTrain

##KS On Test##
KSLRTest=max(attr(perf2,'y.values')[[1]]-attr(perf2,'x.values')[[1]])
KSLRTest

##Gini Chart##
#Gini for Train#
giniLRTrain=ineq(ctrain,type="Gini")
giniLRTrain

##Gini for Test##
giniLRTest=ineq(ctest,type="Gini")
giniLRTest

##K-NN Classification##
#Loading Original Data#
cdata1=cars
str(cdata1)

#Converting all factor variable into integer#
cdata1$Gender=as.integer(cdata1$Gender)
cdata1$Transport=as.integer(cdata1$Transport)
cdata1$MBA=as.integer(cdata1$MBA)
cdata1$Engineer=as.integer(cdata1$Engineer)
cdata1$license=as.integer(cdata1$license)


knn.data=cdata1[,-9]
norm.data=scale(knn.data)
usable.data=cbind(Transport=cdata1[,9],norm.data)
usable.data=as.data.frame(usable.data)

##Splitting data into Test and Train set in 70:30 ratio##
set.seed(500)
spl=sample.split(usable.data$Transport,SplitRatio = 0.7)
train1=subset(usable.data,spl==TRUE)
test1=subset(usable.data,spl==FALSE)

##KNN - Test and Train Dataset##
library(class)
knn_fit_test<- knn(train = train1[,1:8], test = test1[,1:8], cl= train1[,9],k = 3,prob=TRUE) 
knn_fit_train<- knn(train = train1[,1:8],  train1[,1:8], cl= train1[,9],k = 3,prob=TRUE) 

table(train1[,9],knn_fit_train)
table(test1[,9],knn_fit_test)

##Confusion matrix on Train Data##
table.knn3=table(train1[,9],knn_fit_train)
sum(diag(table.knn3))/sum(table.knn3)

##Confusion matrix on Test Data##
table.knn3=table(test1[,9],knn_fit_test)
sum(diag(table.knn3))/sum(table.knn3)

##ROC Curve on Train Dataset##
predRoc3=ROCR::prediction(train1[,9],knn_fit_train)
perf3=performance(predRoc3,"tpr","fpr")
plot(perf3,colorize=T)
as.numeric(performance(predRoc3,"auc")@y.values)
##ROC Curve for Test Dataset##
predRoc4=ROCR::prediction(test1[,9],knn_fit_test)
perf4=performance(predRoc4,"tpr","fpr")
plot(perf3,colorize=T)
as.numeric(performance(predRoc4,"auc")@y.values)

##KS on Train##
KSLRTrain1=max(attr(perf3,'y.values')[[1]]-attr(perf3,'x.values')[[1]])
KSLRTrain1
plot(perf3,main=paste0(' KSLRTrain1=',round(KSLRTrain1*100,1),'%'))
##KS on Test##
KSLRTest1=max(attr(perf4,'y.values')[[1]]-attr(perf4,'x.values')[[1]])
KSLRTest1
plot(perf4,main=paste0(' KSLRTest1=',round(KSLRTest1*100,1),'%'))

##K-NNGini for Train##
giniKnnTrain=ineq(knn_fit_train,type="Gini")
giniKnnTrain

##K-NNGini for Test##
giniKnnTest=ineq(knn_fit_test,type="Gini")
giniKnnTest

##Creating Naive Bayes Model##
NB1=naiveBayes(as.factor(train1$Transport)~.,data=train1,method="class")
NB2=naiveBayes(as.factor(test1$Transport)~.,data=test1,method="class")
##Performing Classification Model Performance Measures for NB##
##Confusion Matrix on Train Data##
predNB1=predict(NB1,newdata=train1,type="class")
table.NB1=table(train1$Transport,predNB1)
sum(diag(table.NB1))/sum(table.NB1)
##Confusion Matrix on Test Data##
predNB2=predict(NB2,newdata=test1,type="class")
table.NB2=table(test1$Transport,predNB2)
sum(diag(table.NB2))/sum(table.NB2)

##Area under the ROC Curve on Train Dataset##
predROC7=ROCR::prediction(train1[,9],predNB1)
perf7=performance(predROC7,"tpr","fpr")
plot(perf7,colorize=T)
as.numeric(performance(predROC7,"auc")@y.values)

##Area under the ROC Curve on Test Dataset##
predROC8=ROCR::prediction(test1[,9],predNB2)
perf8=performance(predROC8,"tpr","fpr")
plot(perf8,colorize=T)
as.numeric(performance(predROC8,"auc")@y.values)

##KS On Train##
KSLRTrain2=max(attr(perf7,'y.values')[[1]]-attr(perf7,'x.values')[[1]])
KSLRTrain2
plot(perf7,main=paste0(' KSLRTrain2=',round(KSLRTrain2*100,1),'%'))

##KS On Test##
KSLRTest2=max(attr(perf8,'y.values')[[1]]-attr(perf8,'x.values')[[1]])
KSLRTest2
plot(perf8,main=paste0(' KSLRTest2=',round(KSLRTest2*100,1),'%'))


##Applying Bagging Model##

BAGmodel=bagging(as.numeric(Transport)~.,data=train1,control=rpart.control(maxdepth=10,minsplit=50))
BAGpredTest=predict(BAGmodel,test1)
tabBAG=table(test1$Transport,BAGpredTest>0.5)
tabBAG

##Convert the dependent variable to a numeric##
train1$Gender=as.numeric(train1$Gender)
train1$Transport=as.numeric(train1$Transport)
train1$Engineer=as.numeric(train1$Engineer)
train1$MBA=as.numeric(train1$MBA)
train1$license=as.numeric(train1$license)
str(train1)

test1$Gender=as.numeric(test1$Gender)
test1$Transport=as.numeric(test1$Transport)
test1$Engineer=as.numeric(test1$Engineer)
test1$MBA=as.numeric(test1$MBA)
test1$license=as.numeric(test1$license)
str(test1)

##Boosting Model##
##All Numeric Variables to Matrix##
features_train1=as.matrix(train1[,1:8])
label_train1=as.matrix(train1[,9])
features_test1=as.matrix(test1[,1:8])

##XGBoost Model##
XGBmodel=xgboost(data=features_train1,label=label_train1,
                 eta=.01,
                 max_depth=5,
                 min_child_weight=3,
                 nrounds=10,
                 nfold=5,
                 objective="reg:linear",verbose = 0,
                 early_stopping_rounds = 10)

##Confusion Matrix Output##
XGBpredTest=predict(XGBmodel,features_test1)
tabXGB=table(test1$Transport,XGBpredTest>0.5)
tabXGB

######################################THANK YOU############################################

