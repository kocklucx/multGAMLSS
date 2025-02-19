#Code to fit the bivariate Gaussian distribution with mvngam as in Section 3.1. See the code for "Additive Covariance Matrix Models: Modelling Regional Electricity Net-Demand in Great Britain" 
#by Gioia, Fasiolo, Browell and Bellio available throught https://zenodo.org/record/7315106.

library(covmodUK)
library(mgcv)

pos1 = head(seq(1,6,0.01),-1)
pos2 = head(seq(-3,3,0.01),-1)
pos3 = head(seq(-1,1,0.01),-1)

for (n in list(250,500,1000)){ #n=100 does not work to little data for model
  for(j in 1:250) {
    fileName = paste(".../data/bivariate_normal_n",toString(n),"_",toString(j),".csv",sep = '')
    df <- data.frame(read.csv(fileName))
    
    mean_formula_int <- list(y1 ~ 1+s(x1)+s(x2)+s(x3), y2 ~ 1+s(x1)+s(x2)+s(x3))
    d <- length(mean_formula_int)
    theta_formula <- lapply(1:(d*(d+1)/2), function(nouse) ~ 1+s(x1)+s(x2)+s(x3))
    global_formula <- c(mean_formula_int,  theta_formula)
    
    fit <- gam(global_formula,family=mvn_mcd(d=2),data = df, control=list(trace=TRUE))
    
    results <- data.frame(predict(fit))
    colnames(results) <- c('eta1','eta2','eta3','eta4','eta5')
    
    fileName = paste(".../mvn_gam/bivariate_normal_predict_n",toString(n),"_",toString(j),".csv",sep = '')
    write.csv(data.frame(results),fileName, row.names = FALSE)
  }
}

