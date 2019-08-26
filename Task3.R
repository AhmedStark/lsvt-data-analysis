source("/home/ahmed/Desktop/RStudio/Assignment1/Task1and2.R")


resample <- make_Weka_filter("weka.filters.supervised.instance.Resample") # register the Resample filter
#Task 3 


#gets the best features for each algorithm
BestTrainLSVTDroppedOneR<-dropMyCols(BestKOneR,TrainLSVT)
BestTestLSVTDroppedOneR<-dropMyCols(BestKOneR,TestLSVT)

BestTrainLSVTDroppedJ48<-dropMyCols(BestKJ48,TrainLSVT)
BestTestLSVTDroppedJ48<-dropMyCols(BestKJ48,TestLSVT)

BestTrainLSVTDroppedNB<-dropMyCols(BestKNB,TrainLSVT)
BestTestLSVTDroppedNB<-dropMyCols(BestKNB,TestLSVT)

BestTrainLSVTDroppedIBk<-dropMyCols(BestKIBk,TrainLSVT)
BestTestLSVTDroppedIBk<-dropMyCols(BestKIBk,TestLSVT)

Z <- seq(100,1000,100)
B <- seq(0.3,1,0.1)

OneRTable<-matrix(1,10,8)
rownames(OneRTable)<-Z
colnames(OneRTable)<-B

J48Table<-matrix(1,10,8)
rownames(J48Table)<-Z
colnames(J48Table)<-B

NBTable<-matrix(1,10,8)
rownames(NBTable)<-Z
colnames(NBTable)<-B

IBkTable<-matrix(1,10,8)
rownames(IBkTable)<-Z
colnames(IBkTable)<-B

zIndex<-1
bIndex<-1

for (b in B){
  
  for (z in Z){
    #balancing data
    rebalanceForOneR <- rebalanceMyData(BestTrainLSVTDroppedOneR)
    rebalanceForJ48 <- rebalanceMyData(BestTrainLSVTDroppedJ48)
    rebalanceForNB <- rebalanceMyData(BestTrainLSVTDroppedNB)
    rebalanceForIBk <- rebalanceMyData(BestTrainLSVTDroppedIBk)
    
    #making OneR table
    OneRModel <- oneR(class ~ ., data = rebalanceForOneR , na.action=NULL)
    
    predOneR <- predict(OneRModel,BestTestLSVTDroppedOneR, na.action=NULL,seed=1) 
    
    OneRTable[zIndex,bIndex]=getMyFweighted(actual,predOneR)
    
    #making J48 table
    J48Model <- j48(class ~ ., data = rebalanceForJ48 , na.action=NULL)
    
    predJ48 <- predict(J48Model,BestTestLSVTDroppedJ48, na.action=NULL,seed=1) 
    
    J48Table[zIndex,bIndex]=getMyFweighted(actual,predJ48)
    
    #making Naive Bayes table
    NBModel <- NB(class ~ ., data = rebalanceForNB , na.action=NULL)
    
    predNB <- predict(NBModel,BestTestLSVTDroppedNB, na.action=NULL,seed=1) 
    
    NBTable[zIndex,bIndex]=getMyFweighted(actual,predNB)
    
    #making 1NN table
    IBkRModel <- ibk(class ~ ., data = rebalanceForIBk , na.action=NULL)
    
    predIBk <- predict(IBkRModel,BestTestLSVTDroppedIBk, na.action=NULL,seed=1) 
    
    IBkTable[zIndex,bIndex]=getMyFweighted(actual,predIBk)
    
    if(zIndex <= length(Z)){
      zIndex<-zIndex+1
    }
    
    if(zIndex >length(Z)){
      zIndex<-1
    }
  }
  
  if(bIndex < length(B)){
    bIndex<-bIndex+1
  }
  
}

OneRBestZandB <- which(OneRTable == max(OneRTable), arr.ind = TRUE)[1,]
J48BestZandB <- which(J48Table == max(J48Table), arr.ind = TRUE)[1,]
NBBestZandB <- which(NBTable == max(NBTable), arr.ind = TRUE)[1,]
IBkBestZandB <- which(IBkTable == max(IBkTable), arr.ind = TRUE)[1,]



AllZandBTable <-matrix(1,2,4)
colnames(AllZandBTable)<-Classifiers
rownames(AllZandBTable)<-c("Best Z","Best B")

AllZandBTable[1,1] <- Z[OneRBestZandB[1]]
AllZandBTable[2,1] <- B[OneRBestZandB[2]]

AllZandBTable[1,2] <- Z[J48BestZandB[1]]
AllZandBTable[2,2] <- B[J48BestZandB[2]]

AllZandBTable[1,3] <- Z[NBBestZandB[1]]
AllZandBTable[2,3] <- B[NBBestZandB[2]]

AllZandBTable[1,4] <- Z[IBkBestZandB[1]]
AllZandBTable[2,4] <- B[IBkBestZandB[2]]



j=data.frame(actual)

write.csv(j,'/home/ahmed/Desktop/RStudio/Assignment1/file.csv')
