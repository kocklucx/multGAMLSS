#Code to fit the bivariate Gaussian distribution with vgam as in Section 3.1. 
library(MASS)
library(VGAM)

for (n in list(100,250,500,1000)){
  for(j in 1:250) {
    fileName = paste(".../data/bivariate_normal_n",toString(n),"_",toString(j),".csv",sep = '')
    df <- data.frame(read.csv(fileName))
    
    pos1 = head(seq(1,6,0.01),-1)
    pos2 = head(seq(-3,3,0.01),-1)
    pos3 = head(seq(-1,1,0.01),-1)
    
    fit <- vglm(cbind(y1,y2)~sm.ps(x1,outer.ok = TRUE)+sm.ps(x2,outer.ok = TRUE)+sm.ps(x3,outer.ok = TRUE),binormal(lsd1   = "loglink",     lsd2   = "loglink",lrho   = "rhobitlink",zero=NULL), data = df, trace = TRUE)
    
    pred <- predict(fit,type='link',newdata=df)
    prediction <- data.frame('x1'=df$x1,'x2'=df$x2,'x3'=df$x3)
    prediction$mu1 <- pred[,'mean1']
    prediction$mu2 <- pred[,'mean2']
    prediction$sigma1 <- loglink(pred[,'loglink(sd1)'],inverse=TRUE)
    prediction$sigma2 <- loglink(pred[,'loglink(sd2)'],inverse=TRUE)
    prediction$rho <- rhobitlink(pred[,'rhobitlink(rho)'],inverse=TRUE)
    
    fileName = paste(".../vgam/bivariate_normal_prediction_n",toString(n),"_",toString(j),".csv",sep = '')
    write.csv(prediction,fileName, row.names = FALSE)
    }
}






