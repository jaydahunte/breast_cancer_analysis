
# Ten real-valued features are computed for each cell nucleus:
#   
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry 
# j) fractal dimension ("coastline approximation" - 1)
# 
# Several of the papers listed above contain detailed descriptions of
# how these features are computed. 
# 
# The mean, standard error, and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features.  For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.

#install.packages("ggplot2")
#install.packages("caTools")
#install.packages("corrplot")
#install.packages("dplyr")
#install.packages("caret")

library("ggplot2")
library("caTools")
library("corrplot")
library("dplyr")
library("caret")

data = read.csv("data.csv", header=T) # reads in the data

View(data) # gives a visual representation of all the data
View(head(data)) # shows the first 6 rows

str(data) # shows the structure of the dataset - the datatype of each row

dim(data) # returns the row by col representation of the dataset

summary(data) # returns a statistical summary of all the numeric cols of 
              # the dataset 

data = data[-33] # removes the last col of the data, since it only has NaN vals

summary(data)

# The goal is to predict whether the cancer is benign or malignant (harmful)

data %>% count(diagnosis) # returns the count for malignant vs benign
dim(data)[1]

data %>% count(diagnosis) %>% group_by(diagnosis) %>% 
  summarize(perc_dx = round((n/dim(data)[1]) * 100, 2)) # returns the percentage
# of malignant (M) vs benign (B) results

# Data Viz

# Frequency of cancer diagnosis

diagnosis.table = table(data$diagnosis)
colors = terrain.colors(2)

# create pie chart

diagnosis.prop.table = prop.table(diagnosis.table) * 100
diagnosis.prop.df = as.data.frame(diagnosis.table)
pielabels = sprintf("%s - %3.1f%s", diagnosis.prop.df[, 1], diagnosis.prop.table, "%")
pie(diagnosis.prop.table, labels = pielabels, clockwise = T, col = colors, 
    border = "gainsboro", radius = 0.8, cex = 0.8, main = "Frequency of Cancer Diagnosis")
legend(1, .4, legend = diagnosis.prop.df[, 1], cex = 0.7, fill = colors)

# correlation plot

c = cor(data[, 3:32])
corrplot(c, order = "hclust", tl.cex = 0.7)


# comparing the radius, area, and concavity of the benign and malignant state

ggplot(data, aes(x = diagnosis, y = radius_mean)) + geom_boxplot() + ggtitle("Radius of Malignant vs Benign")
ggplot(data, aes(x = diagnosis, y = area_mean)) + geom_boxplot() + ggtitle("Area of Malignant vs Benign")
ggplot(data, aes(x = diagnosis, y = concavity_mean)) + geom_boxplot() + ggtitle("Concavity of Malignant vs Benign")

# we can see that the malignant cells have a higher radius, area, and concavity mean than the benign cells


# Barplot for analyzing the stages of the affected women
ggplot(data, aes(x = diagnosis, fill=texture_mean)) + geom_bar() + ggtitle("Women affected in Malignant and Benign stages")

# women affected at higher levels based on mean from analysis of boxplots
data_r = data[data$radius_mean > 10 & data$radius_mean < 15 & data$compactness_mean > 0.1,]
ggplot(data_r, aes(x = diagnosis, y = radius_mean, fill = diagnosis)) + geom_col() + ggtitle("Women Affected in Higher Levels, based on Mean")

# density plot based on texture_mean
ggplot(data, aes(x = texture_mean, fill = as.factor(diagnosis))) + geom_density(alpha=0.4) + ggtitle("Texture Mean Density of Malignant vs Benign")

# barplot for area_se > 15
ggplot(data, aes(x = area_se > 15, fill = diagnosis)) + geom_bar(position="fill") + ggtitle("area_se for Malignant vs Benign")


# Train an Ml model on the dataset

# split data into training and test sets
data$diagnosis = factor(data$diagnosis, levels = c("B", "M"), labels = c(0, 1))
split = sample.split(data$diagnosis, SplitRatio = 0.7)
data = data[-33]
training_set = subset(data, split==T)
View(training_set)
test_set = subset(data, split==F)
View(test_set)

# normalization process - so that the model fits the data better
training_set[,3:32] = scale(training_set[,3:32])
View(training_set)
test_set[,3:32] = scale(test_set[, 3:32])
View(test_set)

# create model
reg = glm(formula = diagnosis ~ ., family = quasibinomial(), data = training_set)
#reg = glm(formula = diagnosis ~ ., family = quasibinomial(), Dinfo = training_set)
summary(reg)

# get predictions
pred = predict(object = reg, type = "response", newdata = test_set[-2])
View(pred)

# separate the pred vals
y_pred = ifelse(pred > 0.5, 1, 0)
View(y_pred)

# construct confusion matrix
tab = table(test_set[, 2], y_pred)
tab

# Accuracy
acc = sum(diag(tab)) / sum(tab)
err = 1 - acc
acc
err
# Save model to RDS file
saveRDS(reg, "model.rds")


