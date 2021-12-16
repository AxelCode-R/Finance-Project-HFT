hallo
.libPaths("D:/R-Portable/App/R-Portable/library")

install.packages("htmltools")
install.packages("reticulate")

Sys.setenv(RETICULATE_PYTHON = "D:\\WinPython2\\WPy64-3950\\python-3.9.5.amd64\\")
reticulate::py_config()
install.packages("formatR")
install.packages("prettydoc")
downcute
install.packages("rmdformats")
# how to code fold https://stackoverflow.com/questions/14127321/how-to-hide-code-in-rmarkdown-with-option-to-see-it


getwd()


.libPaths(new="D:/R-Portable/App/R-Portable/library")
options(StringsAsFactor=FALSE)
options(scipen=999)
library(reticulate)
Sys.setenv(RETICULATE_PYTHON = "D:\\WinPython2\\WPy64-3950\\python-3.9.5.amd64\\")



Sys.unsetenv("R_LIBS")
Sys.setenv(R_LIBS = "D:/R-Portable/App/R-Portable/library")

