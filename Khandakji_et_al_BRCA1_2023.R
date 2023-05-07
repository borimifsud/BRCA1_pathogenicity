########## Khadakji et al. 2023 ###################
### BRCA1-specific machine learning model predicts variant pathogenicity with high accuracy#####

# install and load required packages
install.packages("data.table") 
install.packages("mlr")
install.packages("xgboost")
install.packages("caret") 
library(data.table) 
library(mlr)
library(xgboost) 
library(caret)


# download TrainingBrca1_2668.txt and TestingBrca1_667.txt to your computer and set the working directory to the folder that contains them.
train <- read.table("TrainingBrca1_2668.txt", header = T, sep = "\t", stringsAsFactors = F) 
test <- read.table("TestingBrca1_667.txt", header = T, sep = "\t", stringsAsFactors = F) 
setDT(train) 
setDT(test) 

train$id <- NULL 
test$id <- NULL

# create outcome column based on Pathogenicity expert category
train$outcome[train$Pathogenicity_expert_Cat == "1"] = "1" 
train$outcome[train$Pathogenicity_expert_Cat == "0"] = "0" 
table(train$outcome)

test$outcome[test$Pathogenicity_expert_Cat == "1"] = "1" 
test$outcome[test$Pathogenicity_expert_Cat == "0"] = "0" 
table(test$outcome)

train$Pathogenicity_expert_Cat <- NULL 
test$Pathogenicity_expert_Cat <- NULL

# assign "benign" to out come 0 and "Pathogenic" to outcome 1
train$outcome <- factor(train$outcome,levels = c(0,1),labels = c("Benign", "Pathogenic"))
test$outcome <- factor(test$outcome,levels = c(0,1),labels = c("Benign", "Pathogenic"))
labels <- train$outcome 
ts_label <- test$outcome

new_tr <- model.matrix(~.+0,data = train[,-c("outcome"),with=F],) 
new_ts <- model.matrix(~.+0,data = test[,-c("outcome"),with=F])
labels <- as.numeric(labels)-1 
ts_label <- as.numeric(ts_label)-1
dtrain <- xgb.DMatrix(data = new_tr,label = labels,missing =999) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label,missing=999)

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

min(xgbcv$evaluation_log$test_logloss_mean)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds =100, watchlist = list(val=dtest,train=dtrain), print_every_n = 10, early_stopping_rounds = 10, maximize = F , eval_metric = "error")

colnames(dtest) <- NULL

xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#make xbgpred and ts_label factors for calculating the confusion matrix
nxgbpred <- as.factor(xgbpred) 
nts_label <- as.factor(ts_label) 
confusionMatrix (nxgbpred, nts_label)
