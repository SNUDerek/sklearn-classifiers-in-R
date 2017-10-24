# demo of reticulate plus sklearn - train and test on different data
# https://rstudio.github.io/reticulate/index.html

# installation:
# set python env and install:

# Sys.setenv(RETICULATE_PYTHON='/usr/bin/python3')
# install.packages("reticulate")

setwd("/home/derek/PycharmProjects/sklearn-classifiers-in-R")

# import reticulate and check python info
Sys.setenv(RETICULATE_PYTHON='/home/derek/anaconda3/bin/python')
library(reticulate)
use_condaenv('/home/derek/anaconda3/bin/python')
py_config()

# basic print functionality
py <- import_builtins()

# read in and test toy csv data
# NEEDS the stringsAsFactors=FALSE
train <- read.csv('miniconvo-train.csv', sep=',', stringsAsFactors=FALSE)
test <- read.csv('miniconvo-test.csv', sep=',', stringsAsFactors=FALSE)
names(train)
names(test)

# run my script
script <- py_run_file("rpythontest2.py")

# call 'classify' function with sklearn pipeline : returns [model, labelencoder]
model_package <- py_call(script$r_classify, train$sentence, train$bucket)

# call 'predict' function with sklearn pipeline: returns preds as np array
preds <- py_call(script$r_predict, model_package, test$sentence, test$bucket)

# convert array to R format
preds = py_to_r(preds)
head(preds)
head(test$bucket)



