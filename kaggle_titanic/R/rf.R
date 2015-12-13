setwd('/Users/jason/Desktop/Quick-Data-Science-Experiments-2015/kaggle_titanic/R')
train<-read.csv("../data/train.csv",na.strings=c('NA',''),stringsAsFactors=F)
test<-read.csv("../data/test.csv",na.strings=c('NA',''),stringsAsFactors=F)

library(randomForest)
library(party)
library(rpart)

check.missing<-function(x) return(paste0(round(sum(is.na(x))/length(x),4)*100,'%'))
data.frame(sapply(train,check.missing))
data.frame(sapply(test,check.missing))

train$Cat<-'train'
test$Cat<-'test'
test$Survived<-NA
full<-rbind(train,test)

table(full$Embarked)
full$Embarked[is.na(full$Embarked)]<-'S'
full$Title<-sapply(full$Name,function(x) strsplit(x,'[.,]')[[1]][2])
full$Title<-gsub(' ','',full$Title)
aggregate(Age~Title,full,median)
full$Title[full$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
full$Title[full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

aggregate(Age~Title,full,summary, digits=2)
full$FamilySize<-full$Parch+full$SibSp+1
fit.Fare<-rpart(Fare[!is.na(Fare)]~Pclass+Title+Sex+SibSp+Parch,data=full[!is.na(full$Fare),],method='anova')
printcp(fit.Fare) 
