setwd('/Users/jason.xie/Desktop/Quick-Data-Science-Experiments-2015/udacity_R_class/L3')
library(ggplot2)

facebook <- read.csv('facebook.tsv', sep='\t')
names(facebook)
qplot(x=age, y=friend_count, data=facebook)

ggplot(aes(x=age, y=friend_count), data=facebook) + geom_point()
