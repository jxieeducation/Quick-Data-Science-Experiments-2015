setwd('/Users/jason/Desktop/Quick-Data-Science-Experiments-2015/udacity_R_class/L4')
library(ggplot2)
library(dplyr)

facebook <- read.csv('../L3/facebook.tsv', sep='\t')
names(facebook)
qplot(x=age, y=friend_count, data=facebook)

ggplot(aes(x=age, y=friend_count), data=facebook) + geom_point()

age_groups <- group_by(facebook, age)
facebook.fc_by_age <- summarise(age_groups, friend_count_mean=mean(friend_count), friend_count_median=median(friend_count), n=n())
ggplot(aes(x=age, y=friend_count), data=facebook) + ylim(0, 3000) + xlim(13, 90) + geom_point(alpha=0.05, position=position_jitter(h=0), color='yellow') + geom_line(stat='summary', fun.y=mean) + geom_line(stat='summary', fun.y=quantile, probs=0.1) + geom_line(stat='summary', fun.y=mean) + geom_line(stat='summary', fun.y=quantile, probs=0.9)

not_senior_facebook <- subset(facebook, age <= 70)
cor(x=not_senior_facebook$age, y=not_senior_facebook$friend_count)
cor.test(x=not_senior_facebook$age, y=not_senior_facebook$friend_count)

cor.test(x=facebook$likes_received, y=facebook$www_likes_received)
ggplot(aes(x=likes_received, y=www_likes_received), data=facebook) + geom_point() + xlim(0, quantile(facebook$likes_received, 0.95)) + ylim(0, quantile(facebook$www_likes_received, 0.95)) + geom_smooth(method='lm', color='red')

library(alr3)
data(Mitchell)
summary(Mitchell)
cor.test(Mitchell$Month, Mitchell$Temp)
ggplot(aes(x=Month, y=Temp), data=Mitchell) + scale_x_continuous(breaks=seq(0, 300, 12)) + geom_point()




