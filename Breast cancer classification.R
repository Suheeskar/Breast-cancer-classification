
library(tidyverse)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)

breast.data=read.csv("data.csv")
str(breast.data)

sum(is.na(breast.data))

names(breast.data) = tolower(names(breast.data))

boxplot(breast.data[-2],col='red')

#Data visualisation

ggplot(breast.data,aes(concave.points_mean,concave.points_worst,col=diagnosis))+
  geom_point()

ggplot(breast.data,aes(diagnosis,perimeter_worst))+
  geom_boxplot()

ggplot(breast.data,aes(diagnosis,radius_worst))+
  geom_bar(stat = "identity")

ggplot(breast.data,aes(diagnosis,texture_worst))+
  geom_bar(stat = "identity")

ggplot(breast.data,aes(radius_worst,texture_worst,col=diagnosis))+
  geom_point()

ggplot(breast.data,aes(symmetry_mean,perimeter_se,col=diagnosis))+
  geom_point()

ggplot(breast.data,aes(diagnosis,perimeter_se))+
  geom_bar(stat = "identity")

ggplot(breast.data,aes(diagnosis,smoothness_worst))+
  geom_bar(stat = "identity")

ggplot(breast.data,aes(smoothness_worst,area_worst,col=diagnosis))+
  geom_point()

breast.data1 = breast.data[,-c(1,2)]

#Finding Correlated variables and doing PCA for reducing multicolinearity

cor(breast.data1)
corrplot(cor(breast.data1))

corr.matrix = cor(breast.data1)

breast.data_highcorr = findCorrelation(corr.matrix,cutoff = 0.7)

colnames(breast.data1[,breast.data_highcorr])

breast.data_pca = prcomp(breast.data1,center = TRUE,scale = TRUE,
                         retx = T)
summary(breast.data_pca)

dim(breast.data_pca$x)

biplot(breast.data_pca,scale=0)

# Compute standard deviation
breast.data_pca$sdev

# Compute variance
breast.data_pca.var=breast.data_pca$sdev^2
breast.data_pca.var

# To compute the proportion of variance explained by each component, we simply divide
breast.data.prop = breast.data_pca.var/sum(breast.data_pca.var)
breast.data.prop

# Plot variance explained for each principal component
plot(breast.data.prop, xlab = "principal component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b",
     main = "Scree Plot")

# Plot the cumulative proportion of variance explained
plot(cumsum(breast.data.prop),
     xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

# Find Top n principal component
# which will atleast cover 90 % variance of dimension
which(cumsum(breast.data.prop) >=0.9)[1]

breast.data.final = data.frame(Diagnosis = breast.data$diagnosis,breast.data_pca$x[,1:7])
str(breast.data.final)

breast.data.final$Diagnosis = factor(breast.data.final$Diagnosis,levels = c("M", "B"))

plot_qq(breast.data.final)

# Splitting the dataset to train and test
set.seed(123)
breast.split = createDataPartition(breast.data.final$Diagnosis,
                                        p=0.75,list = FALSE)
x=breast.data.final[,-1]
y=breast.data.final[1]
xtrain=x[breast.split,]
str(xtrain)
xtest =x[-breast.split,]
str(xtest)

ytrain=as.factor(ytrain)
ytest=as.factor(ytest)

ytrain=y[breast.split]
str(ytrain)
ytest =y[-breast.split]
str(ytest)

# create data folds for cross validation
myFolds = createFolds(ytrain, k = 2)

f1 = function(data, lev = NULL, model = NULL) {
  f1val = MLmetrics::F1_Score(y_pred = data$pred,
                              y_true = data$obs,
                              positive = lev[1])
  c(F1 = f1val)
}


# Create reusable trainControl object: myControl{classprobs=for binary classification}
# verboseiter=we want to see the process
set.seed(123)
myControl = trainControl(
  method = "cv", 
  number = 3, 
  summaryFunction = f1,
  classProbs = TRUE, 
  verboseIter = TRUE,
  savePredictions = "final",
  index = myFolds
)

# Fit a simple baseline model
modelbaseline = train(
  diagnosis~.,
  data=breast.data,
  metric = "F1",
  method = "glm",
  family = "binomial",
  trControl = myControl
)

sample.predict = predict.train(modelbaseline,breast.data.final,type = 'raw')

confusionMatrix(sample.predict,breast.data.final$Diagnosis,mode='prec_recall')

modelbaseline
summary(modelbaseline)

#linear model
lg = train(
  x=xtrain,
  y=ytrain,
  method = "glm",
  family = "binomial",
  metric= "F1",
  trControl = myControl
)

#xgb boost model
xgb = train(
  x=xtrain,
  y=ytrain,
  method = "xgbTree",
  metric= "F1",
  trControl = myControl
)

#Xgb boost linear model
xgb_linear = train(
  x=xtrain,
  y=ytrain,
  method = "xgbLinear",
  metric= "F1",
  trControl = myControl
)

#naive Baye's model
nb = train(
  x=xtrain,
  y=ytrain,
  method = "naive_bayes",
  metric= "F1",
  trControl = myControl
)

#svm model
svm = train(
  x=xtrain,
  y=ytrain,
  method = "svmRadial",
  metric= "F1",
  trControl = myControl
)

# Create model_list
modellist = list(baseline = modelbaseline,naivebayes = nb,glmnet = lg,xgboost= xgb,
                 xgblinear=xgb_linear,svmmodel=svm)

# Pass model_list to resamples(): resamples
resamples =resamples(modellist)
# Summarize the results
summary(resamples)
bwplot(resamples, metric = "F1")

# create confusion matrix for basline model
Pred_lg = predict.train(lg,xtest, type = "raw")
Pred_nb = predict.train(nb,xtest, type = "raw")
Pred_xgb = predict.train(xgb,xtest, type = "raw")
Pred_xgblinear = predict.train(xgb_linear,xtest, type = "raw")
Pred_svm = predict.train(svm,xtest, type = "raw")

confusionMatrix(Pred_lg,ytest, mode = "prec_recall")
confusionMatrix(Pred_nb,ytest, mode = "prec_recall")
confusionMatrix(Pred_xgb,ytest, mode = "prec_recall")
confusionMatrix(Pred_xgblinear,ytest, mode = "prec_recall")
confusionMatrix(Pred_svm,ytest, mode = "prec_recall")


xgb1 = train(
  diagnosis~.,
  data=breast.data,
  method = "xgbTree",
  metric= "F1",
  trControl = myControl
)

breast.data$diagnosis=as.factor(breast.data$diagnosis)

xxxx=predict.train(xgb1,breast.data, type = "raw")
confusionMatrix(xxxx,breast.data$diagnosis,mode = "prec_recall")

plot(varImp(xgb))

