### *****************North Point Software List Company*********************************************
##reading all packages from library.
library(ggplot2)
library(dplyr)
library(GGally)
library(gains)
library(forecast)
library(MASS)
library(purrr)
North_Point_List<- read.csv("C:/Users/deeps/OneDrive/Documents/WEBSTER/Analytics Practicum/North Point Project/North-Point List.csv")
dim(North_Point_List)
class(North_Point_List)
head(North_Point_List)
tail(North_Point_List)
names(North_Point_List)
str(North_Point_List)

# We can calculate how many NAs there are in each variable by using the map() in the purrr package
North_Point_List %>% 
  map(is.na) %>%
  map(sum)

missing_values <- sum(is.na(North_Point_List))
missing_values
summary(North_Point_List)

###check for zero
colSums(North_Point_List==0)
library(ggplot2)

### Data Analysis
library(corrplot)
correlation_matrix <- cor(North_Point_List[-1]) #removed sequence number.
correlation_matrix
# Create a basic correlation plot
corrplot(correlation_matrix, method = "number")

ggpairs(North_Point_List[, c("Freq", "last_update_days_ago", "X1st_update_days_ago","source_w","Purchase", "Spending")])

library(ggplot2)
# Histogram for 'Purchase'
ggplot(North_Point_List, aes(x = Purchase)) +
  geom_bar(fill = "brown", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Purchase",
       x = "Purchase",
       y = "Count")
ggplot(data=North_Point_List) + geom_histogram(fill = "brown", color = "black",mapping = aes(x = Spending))
ggplot(data=North_Point_List) + geom_histogram(fill = "pink", color = "black",mapping = aes(x = Freq))
barplot(table(North_Point_List$US), main = "US Distribution", xlab = "US", col = "brown", border = "black")
ggplot(data=North_Point_List) + geom_histogram(fill = "skyblue", color = "black",mapping = aes(x = last_update_days_ago))
ggplot(data=North_Point_List) + geom_histogram(mapping = aes(x = X1st_update_days_ago),fill = "skyblue",color = "black")
barplot(table(North_Point_List$Web.order), main = "Distribution of Web Order ", xlab = "Web Order", col = "brown", border = "white")
ggplot(data=North_Point_List) + geom_point(mapping = aes(x = Freq, y = Spending))
barplot(table(North_Point_List$Address_is_res), main = "Distribution of Address_is_res", xlab = "Address_is_res", col = "pink", border = "black")
barplot(table(North_Point_List$Gender), main = "Gender Distribution", xlab = "Gender", col = "brown", border = "black")

plot_density <- density(North_Point_List$source_u)
plot(plot_density)

colnames(North_Point_List)

#####*************Data Preparation************

###Here In a specific case, I am setting spending value as 0 when purchase was 0.
North_Point_List <- within(North_Point_List, Spending[Spending == 1 & Purchase == 0] <- 0)
North_Point_List[711,]

##removing sequence_number column
North_Point_List<- North_Point_List[,-c(1)]
names(North_Point_List)
str(North_Point_List)

pca_result <- prcomp(North_Point_List, scale. = T)
names(pca_result)
#Performing PCA
# PCA 
plot(pca_result)
library(factoextra)
fviz_eig(pca_result, 
         addlabels = TRUE,
         ylim = c(0, 40),
         main="PCA plot")


# Get loadings from the rotation matrix
loadings <- pca_result$rotation
# Summary
summary(pca_result)
pca_result
###  [ Hence, Not using PCA, as the pca is giving the 22 pc's for 22 variable so no dimension reduction is achieved.]
library(dplyr)
library(tidyverse)
North_Point_List <- North_Point_List %>% mutate(Purchase = factor(Purchase, levels=c(0,1), labels=c("0","1")))
str(North_Point_List)
####*** Data partitioning methods ****
# use set.seed() to get the same partitions when re-running the R code.
set.seed(1)

##Partition the data randomly into training (800 records), validation (700records), and holdout set (500 records)
## partitioning into training (40%), validation (35%), holdout (25%)
# randomly sample 40% of the row IDs for training
train.rows <- sample(rownames(North_Point_List), nrow(North_Point_List)*0.4)

# sample 35% of the row IDs into the validation set, drawing only from records
# not already in the training set
# use setdiff() to find records not already in the training set
valid.rows <- sample(setdiff(rownames(North_Point_List), train.rows),
                     nrow(North_Point_List)*0.35)

# assign the remaining 25% row IDs serve as holdout
holdout.rows <- setdiff(rownames(North_Point_List), union(train.rows, valid.rows))

# create the 3 data frames by collecting all columns from the appropriate rows
train.df <- North_Point_List[train.rows, ]
train_spend <- subset(train.df, Purchase == 1)
valid.df <- North_Point_List[valid.rows, ]
valid_spend <- subset(valid.df, Purchase == 1)

holdout.df <- North_Point_List[holdout.rows, ]

###What is the percentage of customers who actually the purchased the product? in training and testing partitions, respectively?
prop.table(table(train.df$Purchase)) [2]*100  
prop.table(table(valid.df$Purchase))  [2]*100
prop.table(table(holdout.df$Purchase))[2]*100

### Fitting Logistic Regression*********************
library(caret)
logistic.regression <- glm(Purchase ~ .-Spending ,data = train.df, family="binomial")
summary(logistic.regression)
logit.regresspred <- predict(logistic.regression, valid.df,type = "response")
logit.regress_pred <- ifelse(logit.regresspred >0.5, 1, 0)
confusionMatrix(data = as.factor(logit.regress_pred),valid.df$Purchase, positive = "1")

###A confusion matrix in R is a table that will categorize the predictions against the actual values.
##It includes two dimensions, among them one will indicate the predicted values and another one will represent the actual values.

###Freq and Web.order have positive coefficient. Based on the coefficient of the important variable of logit. Source_u, source_h. 
## Source h have -coeff it means there are less chance of customer. Address_is_res has negative coeff which tells that the probability of purchase for being a resident addresses is lower.

####********************NAive bayes Model*********************************
library(tidyverse)
library(caret)
library(e1071)
NorthPoint.NB <- naiveBayes(Purchase ~ .-Spending, data=train.df, laplace = 1)
NorthPoint.NB

##Evaluating performance on validation dataset.
# evaluate
pred_NB <- predict(NorthPoint.NB, valid.df)
confusionMatrix(data= as.factor(pred_NB), as.factor(valid.df$Purchase), positive = "1")

##************************Classification tree**********************************
#install.packages("rpart")
library(rpart)
library(rpart.plot)
# classification tree
default.northpoint <- rpart(Purchase ~ .-Spending, data=train.df, method="class")
default.northpoint

# plot tree
rpart.plot(default.northpoint, extra=1, fallen.leaves=FALSE)
###Evaluating performance on validation dataset
pred.Ctree<- predict(default.northpoint,valid.df,type = "class")
confusionMatrix(pred.Ctree, as.factor(valid.df$Purchase), positive = "1")

#####*********************fit the Random Forest model**************************

library(randomForest)
randomP.model <- randomForest(Purchase ~ .-Spending,  data = train.df)
randomP.model

##Evaluating performance of validation data using confusion matrix
rf.pred <- predict(randomP.model, valid.df)
confusionMatrix(data= as.factor(rf.pred), as.factor(valid.df$Purchase), positive="1")

library(forecast)
## variable importance plot
var_importance <- varImp(randomP.model)
# Print variable importance summary
print(var_importance)

# Plot the variable importance
randomForest::varImpPlot(randomP.model, 
                         sort=TRUE, 
                         main="Variable Importance Plot")

#### Will choose Logistic model as it gave better performance & also gives probability of occurrence of each class.


####*****************************For spending outcome variable******************************************
###---------Regression Model------------

# Fitting a Linear regression model
fit.lmmodel<-lm(Spending~., data=train_spend[,-23])
summary(fit.lmmodel)

## Evaluating Linear regression model Performance
library(gains)
# use predict() with type = "response" to compute predicted probabilities.
fit.lmmodelpred <- predict(fit.lmmodel, valid_spend ,type = "response")
accuracy(fit.lmmodelpred, valid_spend$Spending)

###Improving model performance using STEPWISE REGRESSION MODEL*************
stepfit.lmmodel <- step(fit.lmmodel,direction = "backward")
stepfit.lmmodelpred<- predict(stepfit.lmmodel, valid_spend ,type = "response")
accuracy(stepfit.lmmodelpred, valid_spend$Spending)
summary(stepfit.lmmodel)

##************ Training a Model On the Regression Tree *******************************
#install.packages("rpart")
library(rpart)
spend.rpart<-rpart(Spending~., data=train_spend)
spend.rpart

# Visualizing Regression Trees
# install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(spend.rpart, digits=3)
rpart.plot(spend.rpart, digits = 4, fallen.leaves = T, type=3, extra=101)
library(rattle)
fancyRpartPlot(spend.rpart, cex = 0.8)

# use predict() with type = "response" to compute predicted probabilities.
spend.rpartpred <- predict(spend.rpart, valid_spend)
summary(spend.rpartpred)
accuracy(spend.rpartpred,valid_spend$Spending) ###

###Since the correlation of stepwise is better than linear model & rpart regression tree model. So we are going to take stepwise backward regression model.

##*******Holdout Accuracy*******
#Compute holdout accuracy on prediction set for Logistic Regression model on Purchase variable.
logit.reg.predhold <- predict(logistic.regression, holdout.df,type = "response")
logit.reg.predholdout <- ifelse(logit.reg.predhold >0.5, 1, 0)
confusionMatrix(data = as.factor(logit.reg.predholdout), (holdout.df$Purchase), positive = "1")

logit.cf <- caret::confusionMatrix(data = as.factor(logit.reg.predholdout), as.factor(holdout.df$Purchase), positive = "1")
###A confusion matrix in R is a table that will categorize the predictions against the actual values.
##It includes two dimensions, among them one will indicate the predicted values and another one will represent the actual values.
fourfoldplot(as.table(logit.cf),color=c("red", "green"),main = "Logistic Confusion Matrix on holdout")

# Compute holdout accuracy on prediction set for Linear Regression model on Spending variable.
lmfit.holdpred<- predict(fit.lmmodel, holdout.df ,type = "response")
accuracy(lmfit.holdpred, holdout.df$Spending)

# Compute holdout accuracy on prediction set for Stepwise BAckwaRD Regression model on Spending variable.
stepwise.lmholdpred<- predict(stepfit.lmmodel, holdout.df ,type = "response")
accuracy(stepwise.lmholdpred, holdout.df$Spending)

pred_residuals <- holdout.df$Spending- stepwise.lmholdpred
## Comparing actual vs predicted data.
(data.frame("Predicted" = stepwise.lmholdpred, "Actual" = holdout.df$Spending, "Residual" = pred_residuals))

#A. ##Add a column to the data frame with the predicted probability of purchase from Logistic Regression
holdout.df$Purchase_Predicted_prob <- predict(logistic.regression, holdout.df,type = "response")
tail(holdout.df)

#B.  ####Add another column with predicted spending value from Stepwise regression model.
holdout.df$Spending_Pred_value <- predict(stepfit.lmmodel, holdout.df ,type = "response")
tail(holdout.df)

#C. ####Add a column for “adjusted probability of purchase” to adjust for oversampling the purchaser 
holdout.df$Adjusted_Prob_Purchase <- 0.1065*(holdout.df$Purchase_Predicted_prob)
tail(holdout.df)

#### d. Expected spending
holdout.df$Expected_Spending <- (holdout.df$Spending_Pred_value) * (holdout.df$Adjusted_Prob_Purchase)
tail(holdout.df)

### Estimating gross profit.
sum(holdout.df$Expected_Spending)
sum(holdout.df$Spending)  # actual spending for 2000 customers.

### Calculating estimate profit on 180,000 customers
estimatespend.ontotal<- (180000*sum(holdout.df$Expected_Spending))/500
estimatespend.ontotal
### Subtract it by emailing expense
estimateProfit <- estimatespend.ontotal-360000
estimateProfit

##***********************Cumulative chart****************************
library(gains)
gain <- gains(holdout.df$Expected_Spending, holdout.df$Adjusted_Prob_Purchase, groups=nrow(holdout.df))
gain <- gains(holdout.df$Expected_Spending, holdout.df$Adjusted_Prob_Purchase, groups=10)
result <- data.frame(
  ncases=c(0, gain$cume.obs),
  cumulative=sum(holdout.df$Expected_Spending)*c(0, gain$cume.pct.of.total)
)
g1<-ggplot(result, aes(x=ncases, y=cumulative)) +
  geom_line() +
  geom_segment(aes(x=0, y=0, xend=nrow(holdout.df), yend=sum(holdout.df$Expected_Spending)),
               color="gray", linetype=2) + # adds baseline
  labs(x="# Cases", y="# Expected Spending", title="Cumulative gains chart")

g1
# use gains() to compute deciles.
library(ggplot2)
barplot(gain$mean.resp / mean(holdout.df$Expected_Spending), names.arg=seq(10, 100, by=10),
        xlab="Percentile", ylab="Decile mean / global mean")

# Decile-wise lift chart
liftresult <- data.frame(
  percentile=gain$depth,
  meanResponse=gain$mean.resp /mean(holdout.df$Expected_Spending
                                    ))
  
g2 <- ggplot(liftresult, aes(x=percentile, y=meanResponse)) +
  geom_bar(stat="identity") +
  labs(x="decile", y=" ", title="Decile-wise lift chart")
library(gridExtra)
grid.arrange(g1, g2, ncol=2)

