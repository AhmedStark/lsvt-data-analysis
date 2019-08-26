library(rJava)
library(RWeka)
library(RWekajars)


NB <- make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
oneR <- make_Weka_classifier("weka/classifiers/rules/OneR")
ibk <- make_Weka_classifier("weka/classifiers/lazy/IBk")
j48 <- make_Weka_classifier("weka/classifiers/trees/J48")
GainRatio <- make_Weka_attribute_evaluator("weka/attributeSelection/GainRatioAttributeEval")


source("/home/ahmed/Desktop/RStudio/Assignment1/functions.R")  #gets the functions used

#Task1

TrainLSVT<-read.arff("/home/ahmed/Desktop/RStudio/Assignment1/data/LSVT_train.arff")
colnames(TrainLSVT)[ncol(TrainLSVT)] <- "class"  

TestLSVT<-read.arff("/home/ahmed/Desktop/RStudio/Assignment1/data/LSVT_test.arff")
colnames(TestLSVT)[ncol(TestLSVT)] <- "class"  
actual<-TestLSVT[, ncol(TestLSVT)]  


D <- vector()
accuracy<-vector()
F1_1<-vector()
F1_2<-vector()
Prec_1<-vector()
Prec_2<-vector()
Recall_1<-vector()
Recall_2<-vector()
nc1 <- 0
nc2 <- 0


F_weightedOneR<- vector()
F_weightedJ48 <-vector()
F_weightedNB<-vector()
F_weightedIBk<-vector()

for (i in actual){
  if (i==1){
    nc1 <- nc1 + 1
  }
  else{
    nc2 <- nc2 + 1
  }
}

features_number <- seq(305,5,-5)
n <- 1

A <- GainRatio(class ~ . , data = TrainLSVT,na.action=NULL ) 
ranked_list<- A[order(A)] 



for (K in features_number){

  D[n]=310-K    #Numb of attributes to drop
  s<- ranked_list[1:D[n]] 
  cols.dont.want <- c(names(s)) 
  
  TrainLSVTDropped <- TrainLSVT[, !names(TrainLSVT) %in% cols.dont.want, drop = T]
  TestLSVTDropped <- TestLSVT[, !names(TestLSVT) %in% cols.dont.want, drop = T]
  
  #oneR with K number of attributes
  OneRModel <- oneR(class ~ ., data = TrainLSVTDropped , na.action=NULL)
  
  predOneR <- predict(OneRModel,TestLSVTDropped, na.action=NULL,seed=1) 
  
  F_weightedOneR[n]=getMyFweighted(actual,predOneR)
  
  
  #J48 with K number of attributes
  J48Model <- j48(class ~ ., data = TrainLSVTDropped , na.action=NULL)
  
  predJ48 <- predict(J48Model,TestLSVTDropped, na.action=NULL,seed=1) 
  
  F_weightedJ48[n]=getMyFweighted(actual,predJ48)
  
  
  #Naive Bayes with K number of attributes
  
  NBModel <- NB(class ~ ., data = TrainLSVTDropped , na.action=NULL)
  
  predNB <- predict(NBModel,TestLSVTDropped, na.action=NULL,seed=1) 
  
  F_weightedNB[n]=getMyFweighted(actual,predNB)
  
  
  #1NN with K number of attributes
  IBkModel <- ibk(class ~ ., data = TrainLSVTDropped , na.action=NULL)
  
  predIBk <- predict(IBkModel,TestLSVTDropped, na.action=NULL,seed=1) 
  
  F_weightedIBk[n]=getMyFweighted(actual,predIBk)
  
  
  n <- n+1
}

maxFOneR=max(F_weightedOneR)
BestKOneR= features_number[which.max(F_weightedOneR)]

maxFJ48=max(F_weightedJ48)
BestKJ48=features_number[which.max(F_weightedJ48)]

maxFNB=max(F_weightedNB)
BestKNB=features_number[which.max(F_weightedNB)]

maxFIBk=max(F_weightedIBk)
BestKIBk=features_number[which.max(F_weightedIBk)]


#----------------------------------------------------------
#Task2

#oneR with 310 attributes
OneRModel <- oneR(class ~ ., data = TrainLSVT , na.action=NULL)

predOneR <- predict(OneRModel,TestLSVT, na.action=NULL,seed=1) 

F_weightedOneRAlldata=getMyFweighted(actual,predOneR)


#J48 with 310 attributes
J48Model <- j48(class ~ ., data = TrainLSVT , na.action=NULL)

predJ48 <- predict(J48Model,TestLSVT, na.action=NULL,seed=1) 

F_weightedJ48Alldata=getMyFweighted(actual,predJ48)


#Naive Bayes with 310 attributes

NBModel <- NB(class ~ ., data = TrainLSVT , na.action=NULL)

predNB <- predict(NBModel,TestLSVT, na.action=NULL,seed=1) 

F_weightedNBAlldata=getMyFweighted(actual,predNB)


#1NN with 310 attributes
IBkModel <- ibk(class ~ ., data = TrainLSVT , na.action=NULL)

predIBk <- predict(IBkModel,TestLSVT, na.action=NULL,seed=1) 

F_weightedIBkAlldata=getMyFweighted(actual,predIBk)

Cases=c("Before selection","After selection")
Classifiers=c("OneR","J48","Naive Bayes","1NN")
classifiersTable<-matrix(1,2,4)
rownames(classifiersTable)<-Cases
colnames(classifiersTable)<-Classifiers

classifiersTable[1,1]<-F_weightedOneRAlldata
classifiersTable[1,2]<-F_weightedJ48Alldata
classifiersTable[1,3]<-F_weightedNBAlldata
classifiersTable[1,4]<-F_weightedIBkAlldata

classifiersTable[2,1]<- toString(c(maxFOneR,BestKOneR))
classifiersTable[2,2]<- toString(c(maxFJ48,BestKJ48))
classifiersTable[2,3]<- toString(c(maxFNB,BestKNB))
classifiersTable[2,4]<- toString(c(maxFIBk,BestKIBk))
