setwd('/Users/jason/Desktop/Quick-Data-Science-Experiments-2015/udacity_R_class/L2')
stateInfo <- read.csv('stateData.csv')
sub1 <- subset(stateInfo, state.region == 1)
sub2 <- stateInfo[stateInfo$state.region == 1, ]
# sub3 <- stateInfo[stateInfo$highSchoolGrad >= 50, stateInfo$X]
