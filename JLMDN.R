#----------------------------------------------------------------------------------------
#  Joint Learning of Multiple Differential Networks with fMRI data for Brain Connectivity Alteration Detection
#  Input:
#       X11_w ------ A list with N11 elements corresponding to N11 subjects from the case group in first dataset and
#                    each element in this list is a q x p_o matrix.
#       X12_w ------ A list with N12 elements corresponding to N12 subjects from the control group in first dataset and
#                    each element in this list is a q x p_o matrix.
#       X21_w ------ A list with N21 elements corresponding to N21 subjects from the case group in second dataset and
#                    each element in this list is a q x p_o matrix.
#       X22_w ------ A list with N22 elements corresponding to N22 subjects from the control group in second dataset and
#                    each element in this list is a q x p_o matrix.
#   sis1,sis2 ------ Give the variables number in the data obtained after screening for first dataset and second dataset.      
#
#  boot_strap ------ bootstrap times for this method.
#       
#       p_o, q, N11, N12, N21, N22
#
#  Output:
#   beta_int1 ------ A boot_strap x p_o*(p_o-1)/2 matrix of coefficients for first dataset, stored in sparse matrix format.  
#   beta_int2 ------ A boot_strap x p_o*(p_o-1)/2 matrix of coefficients for second dataset, stored in sparse matrix format. 
#----------------------------------------------------------------------------------------
library(DensParcorr)
library(mvtnorm)
library(MASS)
library(huge)
library(Matrix)
library(expm)
library(lpSolve)
library(fastclime)
library(mnormt)
library(clime)
library(glmnet)
library(foreach)
library(doParallel)
library(JointNets)
library(SIS)

set.seed(777)
p_o <- p_o
q1 <- q1
q2 <- q2
N11 <- N11
N12 <- N12
N21 <- N21
N22 <- N22
sis1 <- sis1
sis2 <- sis2
boot_strap <- boot_strap

cores <- detectCores()
cl <- makeCluster(cores)
registerDoParallel(cl)

X11_omega <- list()
X12_omega <- list()
X11_omega <- foreach(i = 1:N11, .errorhandling = 'pass', .packages = "DensParcorr") %dopar% {
        X11.clime <- DensParcorr(X11_w[[i]],dens.level =.5,select=TRUE)
        X11_omega <- X11.clime$selected.precision
        rm(X11.clime)
        gc()
        return(X11_omega)
}
print('X11_omega completed')

X12_omega <- foreach(i = 1:N12, .errorhandling = 'pass', .packages = "DensParcorr") %dopar% {
        X12.clime <- DensParcorr(X12_w[[i]],dens.level =.5,select=TRUE)
        X12_omega <- X12.clime$selected.precision
        rm(X12.clime)
        gc()
        return(X12_omega)
}
print('X12_omega completed')

X21_omega <- list()
X22_omega <- list()
X21_omega <- foreach(i = 1:N21, .errorhandling = 'pass', .packages = "DensParcorr") %dopar% {
        X21.clime <- DensParcorr(X21_w[[i]],dens.level =.5,select=TRUE)
        X21_omega <- X21.clime$selected.precision
        rm(X21.clime)
        gc()
        return(X21_omega)
}
print('X21_omega completed')

X22_omega <- foreach(i = 1:N22, .errorhandling = 'pass', .packages = "DensParcorr") %dopar% {
        X22.clime <- DensParcorr(X22_w[[i]],dens.level =.5,select=TRUE)
        X22_omega <- X22.clime$selected.precision
        rm(X22.clime)
        gc()
        return(X22_omega)
}
print('X22_omega completed')

W11 <- list()
W12 <- list()
W21 <- list()
W22 <- list()

X11_vec<-matrix(nrow=N11,ncol=p_o*(p_o-1)/2)
W11_vec <- array()
X12_vec<-matrix(nrow=N12,ncol=p_o*(p_o-1)/2)
W12_vec <- array()

X21_vec<-matrix(nrow=N21,ncol=p_o*(p_o-1)/2)
W21_vec <- array()
X22_vec<-matrix(nrow=N22,ncol=p_o*(p_o-1)/2)
W22_vec <- array()

#Mutual information Calculation
mut_in_f<-function(x){
        1/2*log((1+x)/(1-x))
}

W11<-lapply(lapply(X11_omega,cov2cor),mut_in_f)
W12<-lapply(lapply(X12_omega,cov2cor),mut_in_f)
W21<-lapply(lapply(X21_omega,cov2cor),mut_in_f)
W22<-lapply(lapply(X22_omega,cov2cor),mut_in_f)

#Generate data for logistic regression
for (i in 1:N11) {
        W11_vec <- as.vector(W11[[i]][upper.tri(W11[[i]], diag = FALSE)])
        X11_vec[i,] <- c(W11_vec)
}
for (i in 1:N12) {
        W12_vec <- as.vector(W12[[i]][upper.tri(W12[[i]], diag = FALSE)])
        X12_vec[i,] <- c(W12_vec)
}
for (i in 1:N21) {
        W21_vec <- as.vector(W21[[i]][upper.tri(W21[[i]], diag = FALSE)])
        X21_vec[i,] <- c(W21_vec)
}
for (i in 1:N22) {
        W22_vec <- as.vector(W22[[i]][upper.tri(W22[[i]], diag = FALSE)])
        X22_vec[i,] <- c(W22_vec)
}

beta_int1 <- matrix(ncol=p_o*(p_o-1)/2,nrow=boot_strap)
beta_int2 <- matrix(ncol=p_o*(p_o-1)/2,nrow=boot_strap)

a_1 <- matrix(nrow=boot_strap,ncol=p_o*(p_o-1)/2)
a_2 <- matrix(nrow=boot_strap,ncol=p_o*(p_o-1)/2)
class <- list()

for (j in 1:boot_strap) {
        print(j)
        ids11 <- sample(1:N11, N11, replace=T)
        ids12 <- sample(1:N12, N12, replace=T)
        ids21 <- sample(1:N21, N21, replace=T)
        ids22 <- sample(1:N22, N22, replace=T)
        
        X1_vec <- rbind(X11_vec[ids11,],X12_vec[ids12,])
        Y1_vec <- array(c(rep(0,N11),rep(1,N12)),dim = c(N11+N12,1))
        sis_result1 <- SIS(X1_vec, Y1_vec, family = "binomial",penalty='lasso', tune='cv', 
                             type.measure='class', iter=FALSE, nsis = sis1)
        p_index1 <- sis_result1$sis.ix0
        X1_vec <- X1_vec[,p_index1]
        
        X2_vec <- rbind(X21_vec[ids21,],X22_vec[ids22,])
        Y2_vec <- array(c(rep(0,N21),rep(1,N22)),dim = c(N21+N22,1))  
        sis_result2 <- SIS(X2_vec, Y2_vec, family = "binomial",penalty='lasso', tune='cv', 
                             type.measure='class', iter=FALSE, nsis = sis2)
        p_index2 <- sis_result2$sis.ix0
        X2_vec <- X2_vec[,p_index2]
        
        X1 <- X1_vec
        X2 <- X2_vec
        xt=rbind(X1,X2)
        y1=c(rep(0,N11),rep(1,N12))
        y2=c(rep(0,N21),rep(1,N22))
        y=c(y1,y2)
        
        source("JLMDN cv.R")
        print('cv completed')
        
        n1=length(y1)
        n2=length(y2)
        
        n.t=c(n1,n2)
        pe=cumsum(n.t)
        ps=pe-n.t+1
        
        m=2
        ntotal=sum(n.t)
        p=ncol(X1)
        standardize<-function(j,m,ps,pe,x)
        {
                x.t <- x[,j]
                rem <- j%%m
                if ( rem == 0 )
                        rem = m
                
                x.non <- x.t[ps[rem]:pe[rem]]
                n <- length(x.non)
                x.mean <- mean(x.non)
                x.std <- (x.non-x.mean)/sd(x.non)*n^0.5/(n-1)^0.5
                x.t[ps[rem]:pe[rem]] = x.std
                list(x.t=x.t)
        }
        
        process <- function(xt,m,n)
        {
                ntotal=nrow(xt)
                p=ncol(xt)
                pe=cumsum(n)
                ps=pe-n+1
                
                x=matrix(rep(0,ntotal*p*m),nrow=ntotal)
                for( j in 1:p )
                        for ( k in 1:m )
                        {
                                start=ps[k]
                                end = pe[k]
                                x[start:end,m*(j-1)+k]=xt[start:end,j]
                        }
                
                index=seq(1,p*m)
                x=matrix(unlist(lapply(index,standardize,m,ps,pe,x)),nrow=ntotal)
                x0=matrix(rep(0,ntotal*m),nrow=ntotal)
                for ( k in 1:m )
                {
                        start=ps[k]
                        end = pe[k]
                        x0[start:end,k]=rep(1,end-start+1)
                }
                list(x0=x0,x=x,ps=ps,pe=pe)
        }
        
        n=n.t
        data=process(xt,m,n)
        x=data$x
        x0=data$x0
        ps=data$ps
        pe=data$pe
        
        d_cal <- function(x,pos_s,pos_e)
        {
                p <- length(pos_s)
                n=nrow(x)
                index <- seq(1,p)
                trans<-function(i,x,pos_s,pos_e)
                {
                        n=nrow(x)
                        x_t = matrix(x[,pos_s[i]:pos_e[i]],nrow=n)
                        d=diag(t(x_t)%*%x_t)/n
                        list(d=d)
                }
                D=unlist(lapply(index,trans,x,pos_s,pos_e))
                
                list(D=as.vector(D))
        }
        group=rep(m,p)
        pos_e <- cumsum(group)
        pos_s <- pos_e - group + 1
        n.i <- as.double(unlist(d_cal(x,pos_s,pos_e)$D)*nrow(x))
        n.fold=5
        epsilon=0.02
        n.step=10
        gamma=10
        alpha=0.01
        
        source("Logistic_SGM_MM.r")
        fit_inte <- sgroup(x,y,x0,group,n.i,ps,pe,lambda1.opt,lambda2.opt,gamma,gamma)
        beta_raw <- fit_inte$beta
        
        for (a_t2 in 1:(p_o*(p_o-1)/2)) {
                beta_int1[j,a_t2] <- 0
                beta_int2[j,a_t2] <- 0
        }
        
        for (a_t in 1:sis_ny) {
                beta_int1[j,p_index1[a_t]] <- beta_raw[2*a_t-1] 
                beta_int2[j,p_index2[a_t]] <- beta_raw[2*a_t] 
        }
        
        for (a_t3 in 1:(p_o*(p_o-1)/2)) {
                if (beta_int1[j,a_t3]!=0){
                        beta_int1[j,a_t3]<-1                        
                }
                if (beta_int2[j,a_t3]!=0){
                        beta_int2[j,a_t3]<-1                        
                }
        }
        print("JLMDN completed!")
}
parallel::stopCluster(cl)



