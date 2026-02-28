library(evmix)
library(readr)
data = read_csv("File path for AQI data")
train = head(data, 1460)
u = 60 # Threshold

fitting_start <- Sys.time()
for(i in 1:(ncol(train))){
  alpha[i]=fgpd(train[[colnames(train)[i]]],u)$xi
  sig[i]=fgpd(train[[colnames(train)[i]]],u)$sigmau
  
}
fitting_end <- Sys.time()

as.numeric(difftime(fitting_end, fitting_start, units = "secs"))

df=data.frame(cbind(alpha,sig))