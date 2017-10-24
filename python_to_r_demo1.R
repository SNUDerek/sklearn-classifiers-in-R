# demo of reticulate plus sklearn
# https://rstudio.github.io/reticulate/index.html

# installation:
# set python env and install:

# Sys.setenv(RETICULATE_PYTHON='/usr/bin/python3')
# install.packages("reticulate")

# import reticulate and check python info
Sys.setenv(RETICULATE_PYTHON='/usr/bin/python3')
library(reticulate)
use_python('/usr/bin/python3')
py_config()

# basic print functionality
py <- import_builtins()

# read in and test toy csv data
# NEEDS the stringsAsFactors=FALSE
brown <- read.csv('brown.csv', sep='\t', stringsAsFactors=FALSE)
names(brown)

# run my script
script <- py_run_file("rpythontest.py")

# testing variables
script$test_string
script$test_list

# call 'addone' function that takes one list of ints
py_call(script$addone, c(1,2,3,4,5))

# call 'classify' function with sklearn pipeline : returns np array
preds <- py_call(script$classify, brown$document, brown$topic)

# convert to R format
preds = py_to_r(preds)
head(preds)




