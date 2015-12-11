setwd('/Users/jason.xie/Desktop/Quick-Data-Science-Experiments-2015/udacity_R_class/L2')
stateInfo <- read.csv('stateData.csv')
sub1 <- subset(stateInfo, state.region == 1)
sub2 <- stateInfo[stateInfo$state.region == 1, ]
# sub3 <- stateInfo[stateInfo$highSchoolGrad >= 50, stateInfo$X]

reddit <- read.csv('reddit.csv')
reddit <- head(reddit, 40) 
levels(reddit$income.range)
write.csv(reddit, 'reddit.csv')
reddit[13,]

library(ggplot2)
reddit$age.range <- factor(reddit$age.range, levels(reddit$age.range), labels=c(1, 2, 3, 4, 5, 6, 0))
reddit$age.range <- factor(reddit$age.range, levels=c("1", "2", "3", "4", "5", "6", "0"), labels=c(1, 2, 3, 4, 5, 6, 0))
qplot(data = reddit, x = age.range, geom="histogram")

