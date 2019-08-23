getMyFweighted <- function(actualData,PredData) {
  P11<-0
  P12<-0
  P21<-0
  P22<-0
  
  
  for (i in seq(1,length(actualData))){
    if(actualData[i]==1){
      if(PredData[i]==1){
        P11<-P11+1
      }
      else
      {
        P12<-P12+1
      }
    }
    else if (actualData[i]==2){
      if(PredData[i]==2){
        P22<-P22+1
      }
      else
      {
        P21<-P21+1
      }
    }
  }
  
  
  
  Prec_1<-(P11/(P11+P21))
  Prec_2<-(P22/(P22+P12))
  Recall_1 <-(P11/(P11+P12))
  Recall_2 <-(P22/(P22+P21))
  
  F1_1 <-(2*Prec_1*Recall_1)/(Prec_1+Recall_1)
  F1_2 <-(2*Prec_2*Recall_2)/(Prec_2+Recall_2)
  
  F_weighted <- (F1_1*nc1 +F1_2*nc2)/(nc1+nc2)
  return(F_weighted)
}



dropMyCols <- function(BestK,dataset){
  D<-310-BestK
  s<- ranked_list[1:D] 
  cols.dont.want <- c(names(s)) 
  
  Result <- dataset[, !names(dataset) %in% cols.dont.want, drop = T]
  return(Result)
}


rebalanceMyData <- function(dataset){
  rebalanced<- resample(class ~., data = dataset, control = Weka_control(B=b, Z=z ), na.action=NULL ) 
  return(rebalanced)
}



