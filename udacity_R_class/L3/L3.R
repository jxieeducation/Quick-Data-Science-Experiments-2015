setwd('/Users/jason.xie/Desktop/Quick-Data-Science-Experiments-2015/udacity_R_class/L3')
facebook <- read.csv('facebook.tsv', sep='\t')
library(ggplot2)
names(facebook)
graph <- qplot(data=facebook, x=dob_day) + scale_x_discrete(breaks=1:31) 
graph_wrap <- graph + facet_wrap(~dob_month, ncol = 3)

qplot(data=facebook, x=friend_count, xlim=c(0,1000), binwidth=1, xlab="friendcount", ylab="count")
summary(facebook$gender)
facebook_notNA <- subset(facebook, gender!="female")
summary(facebook_notNA$gender)
by(facebook$friend_count, facebook$gender, summary)

qplot(data=facebook, x=age, xlab='age', ylab='count', binwidth=3) + scale_x_continuous(breaks=seq(10, 40, 3))  + theme_gray()
facebook$likes_received <- log10(facebook$likes_received + 1)
qplot(data=facebook, x=likes_received) + scale_x_discrete(breaks=seq(0, 0.5, 0.05))
qplot(data=facebook, x=likes + 1) + scale_x_log10()

qplot(geom="boxplot", data=subset(facebook, !is.na(gender)), x=gender, y=age)
