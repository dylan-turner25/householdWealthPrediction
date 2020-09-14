rm(list = ls()) # clear console
options(scipen = 999) # forces R to avoid exponential notation
system.info <- Sys.info()
setwd("/home/dylan/Dropbox/Research/Estimating Wealth")

# load libraries --------------------------
library(xgboost) 
library(ggplot2) 
library(mlr) # for hyper tuning parameter
library(Ckmeans.1d.dp)

# load survey of consumer finance data from txt file ---------------- 
scf_data <- read.csv("./SCF/sub-data.txt", header = TRUE, sep = ",", dec = ".")


# clean the data ----------------------------

# creating age categories in a way surveys typicaly use
scf_data$age <- NA
scf_data$age <- replace(scf_data$age, scf_data$AGE <= 24, 1)
scf_data$age <- replace(scf_data$age, scf_data$AGE >= 25 & scf_data$AGE <= 34, 2)
scf_data$age <- replace(scf_data$age, scf_data$AGE >= 35 & scf_data$AGE <= 44, 3)
scf_data$age <- replace(scf_data$age, scf_data$AGE >= 45 & scf_data$AGE <= 54, 4)
scf_data$age <- replace(scf_data$age, scf_data$AGE >= 55 & scf_data$AGE <= 64, 5)
scf_data$age <- replace(scf_data$age, scf_data$AGE >= 65 & scf_data$AGE <= 74, 6)
scf_data$age <- replace(scf_data$age, scf_data$AGE >= 75 , 7)

# household size
scf_data$hh_size <- 1
scf_data$hh_size <- replace(scf_data$hh_size, scf_data$married == 1, 2)
scf_data$hh_size <- scf_data$hh_size + scf_data$KIDS

# education level
scf_data$highschool <- as.numeric(scf_data$EDUC == 8)
scf_data$bachelors <- as.numeric(scf_data$EDUC == 12)
scf_data$masters <- as.numeric(scf_data$EDUC == 13)
scf_data$doctorate <- as.numeric(scf_data$EDUC == 14)
scf_data$grad_school <- scf_data$doctorate+scf_data$masters

# married
scf_data$married <- as.numeric(scf_data$MARRIED == 1)

# race
scf_data$white <- as.numeric(scf_data$RACE == 1)
scf_data$black <- as.numeric(scf_data$RACE == 2)
scf_data$hispanic <- as.numeric(scf_data$RACE == 3)

# income as an ordered categorical variable (this is common on surveys)
scf_data$hh_inc_cats <- NA
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME < 35000, 1 ) # less than 35
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME >= 35000 & scf_data$INCOME <= 49999, 2 ) # 35 - 49999
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME >= 50000 & scf_data$INCOME <= 74999, 3 ) # 50 - 74999
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME >= 75000 & scf_data$INCOME <= 99999, 4 ) # 75 - 99999
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME >= 100000 & scf_data$INCOME <= 149999, 5 ) # 100 - 149,999
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME >= 150000 & scf_data$INCOME <= 199999, 6 ) # 150 - 199999
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME >= 200000 & scf_data$INCOME <= 249999, 7 ) # 200 - 249,999
scf_data$hh_inc_cats <- replace(scf_data$hh_inc_cats, scf_data$INCOME >= 250000 , 8 ) # 250 + 

#employement status
scf_data$employed <- as.numeric(scf_data$OCCAT1 == 1)
scf_data$self_employed <- as.numeric(scf_data$OCCAT1 == 2)
scf_data$retired_homemaker_disabled_student <- as.numeric(scf_data$OCCAT1 == 3)
scf_data$not_working <- as.numeric(scf_data$OCCAT1 == 4)

# race
scf_data$white <- as.numeric(scf_data$RACE == 1)
scf_data$black <- as.numeric(scf_data$RACE == 2)

#other real estate
scf_data$other_RE_100k <- as.numeric(scf_data$ORESRE >= 100000)


# subset the data 
data_subset <- scf_data[,c("NETWORTH","age","highschool",
                             "bachelors","grad_school",
                             "hh_size","HOUSES","other_RE_100k",
                             "hh_inc_cats",
                             "employed","self_employed",
                             "not_working","white","black")]
colnames(data_subset) <- c("networth","age","highschool","bachelors",
                        "grad_school","hh_size","home_value","re_100k",
                        "hh_inc_cats","employed","self_employed","not_working",
                        "white","black")


# throw out extreme outliers
bottom_pct <- round(.01*nrow(data_subset)) # observation number corresponding to the 1st percentile 
data_subset <- data_subset[order(data_subset$networth),] # order the data by networth
data_subset <- data_subset[which(data_subset$home_value > 0),] # keep only homeowners
data_subset <- data_subset[bottom_pct:nrow(data_subset),] # drop the bottom 1 percenet of the distribution
hist(data_subset$networth)

# experiment with dropping high networth households using differenct thresholds 
data_subset_unconstrained <- data_subset
data_subset_10mil <- data_subset[data_subset$networth < 10000000,] # drop households with net worth over 10 million
data_subset_5mil <- data_subset[data_subset$networth < 5000000,] # drop households with net worth over 5 million
data_subset_2mil <- data_subset[data_subset$networth < 2000000,] # drop households with net worth over 2 million
data_subset_1mil <- data_subset[data_subset$networth < 1000000,] # drop households with net worth over 1 million
data_subset_500k <- data_subset[data_subset$networth < 500000,] # drop households with net worth over 500k

# loop through differnt data subsets to see how accuracy changes as high networth house holds are dropped at various thresholds
subsets <- c("unconstrained","10mil","5mil","2mil","1mil","500k")
for(k in 1:length(subsets)){
  data_subset = eval(parse(text = paste0("data_subset_",subsets[k])))

# define the target variable
outcome_variable <- "networth"

# number of random hyperparameter sets to test during hyper parameter tuning
hyper_tuning_reps <- 10 

# number of folds used in cross validation
cv_folds <- 3 
# run time is going to be a function of (hyper_tuning_reps*cv_folds)

# Split data in to Train/Test 
train_test_ratio <- .5
smp_size <- floor(train_test_ratio * nrow(data_subset))
train_ind <- sample(seq_len(nrow(data_subset)), size = smp_size)
train <- data_subset[train_ind, ]
test <- data_subset[-train_ind, ]

# train and test data with target variable still attached
train_with_target <-  train
test_with_target <-  test

# define tasks for mlr
traintask <- makeRegrTask(data = train_with_target , target = outcome_variable)
testtask <- makeRegrTask(data = test_with_target , target = outcome_variable)

# create learner using the mlr library
lrn <- makeLearner("regr.xgboost",predict.type = "response")
lrn$par_vals <- list( objective = "reg:squarederror", eval_metric = "rmse")

# create parameter space to search for optimal hyper parameters
params <- makeParamSet( 
  makeIntegerParam("max_depth",lower = 2L,upper = 20L), 
  makeNumericParam("min_child_weight",lower = 1L,upper = 20L), 
  makeNumericParam("subsample",lower = 0.5,upper = 1), 
  makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
  makeNumericParam("gamma", lower = 0, upper = 5),
  makeNumericParam("eta", lower = 0, upper = .5),
  makeIntegerParam("nrounds", lower = 10L, upper = 1000L),
  makeIntegerParam("early_stopping_rounds", lower = 50L, upper = 200L))


# set resampling strategy
rdesc <- makeResampleDesc("CV" , stratify = F , iters = cv_folds)

#search strategy
ctrl <- makeTuneControlRandom(maxit = hyper_tuning_reps) # using random search
#ctrl <- makeTuneControlGrid(resolution = 5L) # using grid search

# tune the hyperparameters by searching the paramater space
mytune <- tuneParams(learner = lrn, task = traintask, 
                     resampling = rdesc, 
                     par.set = params, control = ctrl, 
                     show.info = T)


# put training data into a Dmatrix
TrainingSet_XGB <- as.matrix(train[,-which(colnames(test) == outcome_variable)])
class(TrainingSet_XGB) <- "numeric"
train_label <- train[,outcome_variable]
TrainingSet_XGB <- xgb.DMatrix(data = TrainingSet_XGB, label = train_label)

# put test data into a Dmatrix
TestSet_XGB <- as.matrix(test[,-which(colnames(test) == outcome_variable)])
class(TestSet_XGB) <- "numeric"
test_label <- test[,outcome_variable]
TestSet_XGB <- xgb.DMatrix(data = TestSet_XGB, label = test_label)

# define for XGBoost which data is training and which is for testing
watchlist = list(train = TrainingSet_XGB, test = TestSet_XGB)

# define the hyperparameter values using the best set found during hyper parameter tuning
tuning_results <- mytune
params <- list(
  objective = 'reg:squarederror', # regression task
  booster = "gbtree",
  max_depth = tuning_results$x$max_depth,
  min_child_weight = tuning_results$x$min_child_weight,
  subsample  = tuning_results$x$subsample,
  colsample_bytree = tuning_results$x$colsample_bytree,
  gamma = tuning_results$x$gamma,
  eta = tuning_results$x$eta
)

#run the XGBoost algorithm
bst <- xgb.train(
  params = params,
  data = TrainingSet_XGB,
  watchlist = watchlist,
  print_every_n = 1, 
  nrounds = tuning_results$x$nrounds,
  early_stopping_rounds = tuning_results$x$early_stopping_rounds,
  maximize = F, eval_metric = "rmse"
)

#predict on test set
predTest <- predict(bst, TestSet_XGB)

#predict on training set
predTrain <- predict(bst, TrainingSet_XGB)

# merge predictions with test data 
test$predicted_networth <- predTest
test$networth_diff <- test$networth - test$predicted_networth
test$networth_diff_pct <- (test$networth_diff/test$networth)
test$networth_diff_pct <- replace(test$networth_diff_pct, abs(test$networth) < 10000, NA )
hist(test$networth_diff, breaks = 30)
oos_accuracy <- data.frame(matrix(nrow = 6, ncol = 1))
colnames(oos_accuracy) <- c("percent within")
rownames(oos_accuracy) <- c("10000","20000","30000","40000","50000","100000")
oos_accuracy$`percent within`[1] <- nrow(test[which(abs(test$networth_diff) <= 10000),])/nrow(test)
oos_accuracy$`percent within`[2] <- nrow(test[which(abs(test$networth_diff) <= 20000),])/nrow(test)
oos_accuracy$`percent within`[3] <- nrow(test[which(abs(test$networth_diff) <= 30000),])/nrow(test)
oos_accuracy$`percent within`[4] <- nrow(test[which(abs(test$networth_diff) <= 40000),])/nrow(test)
oos_accuracy$`percent within`[5] <- nrow(test[which(abs(test$networth_diff) <= 50000),])/nrow(test)
oos_accuracy$`percent within`[6] <- nrow(test[which(abs(test$networth_diff) <= 100000),])/nrow(test)
oos_accuracy

assign(paste0("oos_accuaracy_",subsets[k]),oos_accuracy)


# Importance plot
importance_matrix <- xgb.importance(colnames(TrainingSet_XGB), model = bst)
gg <- xgb.ggplot.importance(importance_matrix[1:nrow(importance_matrix)], measure = NULL , rel_to_first = TRUE, n_clusters = 4 )
gg + ggplot2::ylab("Relative Importance") + theme(legend.position = "none") 

assign(paste0("importance_matrix_",subsets[k]),importance_matrix)



# plot the train vs test error
bst.log <- data.frame(bst$evaluation_log)
library(ggplot2)

ggplot() + 
  geom_line(data = bst.log, aes(x = iter, y = train_rmse, color = "Train Error"), show.legend = T) +
  geom_line(data = bst.log, aes(x = iter, y = test_rmse, color = "Test Error"), show.legend = T) +
  scale_color_manual(values = c("brown","blue4")) +
  xlab('Itteration') +
  ylab('Error') +
  ggtitle("Mean Squared Error") +  theme(plot.title = element_text(hjust = 0.5)) +
  labs(color = "") #this line changes the lengend title

} 

# put out of sample accuaracy from each data subset into one data frame
oos_accuracy <- data.frame(rownames(oos_accuracy),oos_accuaracy_unconstrained, oos_accuaracy_10mil, oos_accuaracy_5mil, oos_accuaracy_2mil, oos_accuaracy_1mil, oos_accuaracy_500k)
colnames(oos_accuracy) <- c("percent within $X","Unconstrained","Less than 10mil","Less than 5mil","Less than 2mil","Less than 1mil","Less than 500k")


