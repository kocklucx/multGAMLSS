#Code to fit the five dimension non-Gaussian distribution with mctm as in Section 3.2. 

library(MASS)
library(tram)

fit_mctm <- function(data){
  By <- lapply(c("y1","y2", "y3", "y4","y5"), function(y) {
    v <- numeric_var(y, support = quantile(df[[y]], prob = c(.1, .9)),
                     bounds = c(0, Inf)) 
    Bernstein_basis(var = v, order = 6, ui = "increasing")
  })
  
  Bx_shift <- Bernstein_basis(numeric_var("x", support = c(-.8, .8)), order = 6, 
                              ui = "zero")
  Bx_lambda <- Bernstein_basis(numeric_var("x", support = c(-.8, .8)), order = 6)
  
  ### marginal models
  ctm_y1 <- ctm(By[[1]], shift = Bx_shift, todistr = "Normal")
  m_y1 <- mlt(ctm_y1, data = df) 
  ctm_y2 <- ctm(By[[2]], shift = Bx_shift, todistr = "Normal")
  m_y2 <- mlt(ctm_y2, data = df)
  ctm_y3 <- ctm(By[[3]], shift = Bx_shift, todistr = "Normal")
  m_y3 <- mlt(ctm_y3, data = df)
  ctm_y4 <- ctm(By[[4]], shift = Bx_shift, todistr = "Normal")
  m_y4 <- mlt(ctm_y4, data = df)
  ctm_y5 <- ctm(By[[5]], shift = Bx_shift, todistr = "Normal")
  m_y5 <- mlt(ctm_y5, data = df)
  
  ### full model
  m <- mmlt(m_y1, m_y2, m_y3, m_y4, m_y5, formula = Bx_lambda, data = df)
  return(m)
}

return_summaries <- function(mctm){
  xseq <- head(seq(-0.9,0.9,0.01),-1)
  df_predict <- 6*asin(0.5*coef(mctm, newdata = data.frame(x = xseq), type = "Corr"))/pi
  colnames(df_predict) <- c('rho21','rho31','rho32','rho41','rho42','rho43','rho51','rho52','rho53','rho54')
  return(cbind(df_predict,x=xseq))
}


for(j in 1:250) {
  fileName = paste(".../data/high_dims_",toString(j),".csv",sep = '')
  df <- data.frame(read.csv(fileName))
  
  m_true <- fit_mctm(df)
  rho <- return_summaries(m_true)
  
  # parametric bootstap
  len_bootsrap <- 500
  rho_bootstrap <- vector(mode = "list", length = len_bootsrap)
  for (b in 1:len_bootsrap) {
    # generate bootstrap data 
    df_bootsrap <- df
    pred_corr <- coef(m_true, newdata =df['x'], type = "Corr")
    pred_trafo <- predict(m_true,type="trafo",marginal=c(1,2,3,4,5),newdata=df['x'])
    for(h in 1:500){
      omega <- matrix(0,ncol=5,nrow=5)
      omega[upper.tri(omega,diag=F)] <- pred_corr[h,]
      omega <- omega + t(omega) + diag(5)
      z <- mvrnorm(1,rep(0, 5),omega)
      for(j in 1:5){
        pred <- pred_trafo[[j]][,h]
        predx <- unname(pred)
        predx[1] <- qnorm(0.00001)
        predy <- as.numeric(names(pred))
        z[j]<-approx(predx,predy, xout=z[j])$y
      }
      df_bootsrap[h,c('y1','y2','y3','y4','y5')] <- z
    }
    m_bootstrap <- fit_mctm(df_bootsrap)
    rho <- rbind(rho, return_summaries(m_bootstrap))
  }
  
  # fast alternative
  #xseq <- head(seq(-0.9,0.9,0.01),-1)
  #nd <- data.frame(x = xseq)
  #nsamp <- 1000
  #V <- vcov(m_true)
  #V <- (V + t(V)) / 2
  #P <- rmvnorm(nsamp, mean = coef(m_true), sigma = V)
  #m_tmp <- m_true
  #for (i in 1:nsamp) {
  #  cf <- P[i, ]
  #  mi <- 1:length(m_true$pars$mpar)
  #  mcf <- cf[mi]
  #  vcf <- matrix(cf[-mi], nrow = nrow(m_true$pars$cpar))
  #  m_tmp$par <- cf
  #  m_tmp$pars <- list(mpar = mcf, cpar = vcf)
  #  rho <- rbind(rho, return_summaries(m_tmp))
  #}
  
  ### safe results
  fileName = paste(".../tram/high_dims_tram_",toString(j),".csv",sep ='')
  write.csv(rho,fileName, row.names = FALSE)
}
