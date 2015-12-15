setwd('/Users/jason/Desktop/Quick-Data-Science-Experiments-2015/kaggle_titanic/R')
train<-read.csv("../data/train.csv",na.strings=c('NA',''),stringsAsFactors=F)
drops <- c("Cabin","PassengerId")
train <- train[,!(names(train) %in% drops)]
train$Survived <- factor(train$Survived)
train$Sex <- factor(train$Sex)
train$Embarked <- factor(train$Embarked)
test<-read.csv("../data/test.csv",na.strings=c('NA',''),stringsAsFactors=F)

library(ggplot2)
library(Amelia)
library(rpart)
library(rpart.plot)
library(randomForest)

summary(train)

train$Title <- sapply(train$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
train$Title <- sub(' ', '', train$Title)
train$Title <- factor(train$Title)
prop.table(table(train$Title))

# backfill age and embarked
missmap(train, main = "Age missing before backfill")
Agefit <- rpart(Age ~ Pclass + Title + Sex + SibSp + Parch + Fare + Embarked, data=train[!is.na(train$Age),], method="anova")
train$Age[is.na(train$Age)] <- predict(Agefit, train[is.na(train$Age),])
train$Embarked[is.na(train$Embarked)] <- 'S'
missmap(train, main = "Age missing after backfill")

fit <- randomForest(Survived ~ Pclass + Title + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, importance=TRUE, ntree=1000)
fit$importance
fit$err.rate
fit$confusion
