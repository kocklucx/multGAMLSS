#Code to fit the bivariate Gaussian distribution with bamlss as in Section 3.1. Fitted effects including 95% credible intervalls
#evaluated on a fine grid are saved for all effects.

library(MASS)
library(bamlss)

for (n in list(100,250,500,1000)){
  for(j in 1:250) {
    fileName = paste(".../data/bivariate_normal_n",toString(n),"_",toString(j),".csv",sep = '')
    df <- data.frame(read.csv(fileName))
    
    f <- list(y1 ~ s(x1, k = 20)+s(x2, k = 20)+s(x3, k = 20),y2~s(x1, k = 20)+s(x2, k = 20)+s(x3, k = 20),sigma1~s(x1, k = 20)+s(x2, k = 20)+s(x3, k = 20),sigma2~s(x1, k = 20)+s(x2, k = 20)+s(x3, k = 20),rho~s(x1, k = 20)+s(x2, k = 20)+s(x3, k = 20))
    b <- bamlss(f, data = df, family = "mvnorm")
    
    pos1 = head(seq(1,6,0.01),-1)
    pos2 = head(seq(-3,3,0.01),-1)
    pos3 = head(seq(-1,1,0.01),-1)
    
    mu1_f1 <- predict(b, newdata = data.frame('x1'=pos1), model = "mu1", term ="s(x1)", FUN = c95)
    mu1_f2 <- predict(b, newdata = data.frame('x2'=pos2), model = "mu1", term ="s(x2)", FUN = c95)
    mu1_f3 <- predict(b, newdata = data.frame('x3'=pos3), model = "mu1", term ="s(x3)", FUN = c95)
    
    sigma1_f1 <- predict(b, newdata = data.frame('x1'=pos1), model = "sigma1", term ="s(x1)", FUN = c95)
    sigma1_f2 <- predict(b, newdata = data.frame('x2'=pos2), model = "sigma1", term ="s(x2)", FUN = c95)
    sigma1_f3 <- predict(b, newdata = data.frame('x3'=pos3), model = "sigma1", term ="s(x3)", FUN = c95)
    
    mu2_f1 <- predict(b, newdata = data.frame('x1'=pos1), model = "mu2", term ="s(x1)", FUN = c95)
    mu2_f2 <- predict(b, newdata = data.frame('x2'=pos2), model = "mu2", term ="s(x2)", FUN = c95)
    mu2_f3 <- predict(b, newdata = data.frame('x3'=pos3), model = "mu2", term ="s(x3)", FUN = c95)
    
    sigma2_f1 <- predict(b, newdata = data.frame('x1'=pos1), model = "sigma2", term ="s(x1)", FUN = c95)
    sigma2_f2 <- predict(b, newdata = data.frame('x2'=pos2), model = "sigma2", term ="s(x2)", FUN = c95)
    sigma2_f3 <- predict(b, newdata = data.frame('x3'=pos3), model = "sigma2", term ="s(x3)", FUN = c95)
    
    rho_f1 <- predict(b, newdata = data.frame('x1'=pos1), model = "rho", term ="s(x1)", FUN = c95)
    rho_f2 <- predict(b, newdata = data.frame('x2'=pos2), model = "rho", term ="s(x2)", FUN = c95)
    rho_f3 <- predict(b, newdata = data.frame('x3'=pos3), model = "rho", term ="s(x3)", FUN = c95)
    
    fileName = paste(".../bamlss/bivariate_normal_f1_n",toString(n),"_",toString(j),".csv",sep = '')
    write.csv(data.frame('mu1'=mu1_f1,'sigma1'=sigma1_f1,'mu2'=mu2_f1,'sigma2'=sigma2_f1,'rho'=rho_f1),fileName, row.names = FALSE)
    
    fileName = paste(".../bamlss/bivariate_normal_f2_n",toString(n),"_",toString(j),".csv",sep = '')
    write.csv(data.frame('mu1'=mu1_f2,'sigma1'=sigma1_f2,'mu2'=mu2_f2,'sigma2'=sigma2_f2,'rho'=rho_f2),fileName, row.names = FALSE)
    
    fileName = paste(".../bamlss/bivariate_normal_f3_n",toString(n),"_",toString(j),".csv",sep = '')
    write.csv(data.frame('mu1'=mu1_f3,'sigma1'=sigma1_f3,'mu2'=mu2_f3,'sigma2'=sigma2_f3,'rho'=rho_f3),fileName, row.names = FALSE)
  }
}