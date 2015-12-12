data(diamonds)
summary(diamonds)
dim(diamonds)

library(ggplot2)

qplot(data=diamonds, x=price) + facet_wrap(~ cut, scales="free_y")
dim(subset(diamonds, diamonds$price >= 15000))

qplot(data=diamonds, x=price/carat, binwidth=1) + scale_x_log10() + facet_wrap(~ cut)

qplot(data=diamonds, x=color, y=price, geom="boxplot")

DDiamonds <- subset(diamonds, color='D')
summary(DDiamonds$price)

