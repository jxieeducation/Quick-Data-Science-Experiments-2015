setwd('/Users/jason/Desktop/Quick-Data-Science-Experiments-2015/kaggle_titanic/R')
train<-read.csv("../data/train.csv",na.strings=c('NA',''),stringsAsFactors=F)
test<-read.csv("../data/test.csv",na.strings=c('NA',''),stringsAsFactors=F)

library(ggplot2)
library(Amelia)

summary(train)
head(train)
missmap(train, main = "Missingness Map Train")

### look at distribution

prop.table(table(train$Survived))
prop.table(table(train$Embarked))

### look at 2 variable relations
ggplot(aes(x=Age, y=Survived), data=train) + geom_point()
cor.test(train$Age, train$Survived, method='spearman') #-0.0525653
train$Child <- 0
train$Child[train$Age < 18] <- 1
aggregate(Survived ~ Child + Sex, data=train, FUN=sum)



ggplot(aes(x=Fare), data=train) + geom_histogram() + xlim(0, 300) + scale_x_continuous(seq(0, 300, 10))
cor.test(train$Fare, train$Survived, method='pearson') #0.2573065
ggplot(aes(x=log(Fare), y=Survived), data=train) + geom_point()
cor.test(log(train$Fare), train$Survived, method='spearman') # 0.3237361
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

aggregate(Survived ~ Pclass, data=train, FUN=function(x) {sum(x)/length(x)}) # OHHHHH SHIT, nice
cor.test(train$Pclass, train$Survived, method='pearson') #0.2573065

### looking at feature importance using trees
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

fit <- rpart(Survived ~ Sex, data=train, method="class")
fancyRpartPlot(fit) # OOOOOO SO FANCYYYYY

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
fancyRpartPlot(fit)

### 

train$Title <- sapply(train$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
train$Title <- sub(' ', '', train$Title)
train$Title[train$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
train$Title[train$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
train$Title[train$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
train$Title <- factor(train$Title)


