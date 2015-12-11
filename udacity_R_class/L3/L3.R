setwd('/Users/jason.xie/Desktop/Quick-Data-Science-Experiments-2015/udacity_R_class/L3')
facebook <- read.csv('facebook.tsv', sep='\t')
library(ggplot2)
names(facebook)
graph <- qplot(data=facebook, x=dob_day) + scale_x_discrete(breaks=1:31) 
graph_wrap <- graph + facet_wrap(~dob_month, ncol = 3)

qplot(data=facebook, x=friend_count, xlim=c(0,1000), binwidth=1)
summary(facebook$gender)
facebook_notNA <- subset(facebook, gender!="female")
summary(facebook_notNA$gender)
by(facebook$friend_count, facebook$gender, summary)



