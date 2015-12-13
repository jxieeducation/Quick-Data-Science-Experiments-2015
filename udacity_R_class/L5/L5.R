setwd('/Users/jason/Desktop/Quick-Data-Science-Experiments-2015/udacity_R_class/L5')
library(ggplot2)
library(dplyr)

facebook <- read.csv('../L3/facebook.tsv', sep='\t')

ggplot(aes(x=gender, y=age), data=subset(facebook, !is.na(gender))) + geom_boxplot()+ stat_summary(fun.y=mean, geom='point', shape=4)
ggplot(aes(x=age, y=friend_count), data=subset(facebook, !is.na(gender))) + geom_line(aes(color=gender), stat='summary', fun.y=median)
