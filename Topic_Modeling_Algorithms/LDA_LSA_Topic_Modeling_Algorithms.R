rm(list=ls())
gc()
getwd()
#install.packages('pals')
#install.packages("tm")
library(tm)
library(wordcloud)
#library(pals)
#library(reshape2)
library(ggplot2)
#library(stringr)
#library(stringi)
#library(SnowballC)
library(topicmodels)
#library(quanteda)
library(topicdoc)
library(topicmodels)

#***************** THIS IS THE CODE BY WHICH WE REPLACE THE NAME OF FILE AS SAME BEFORE GETTING IT CORPUS
#*********** AFTER GETTING FILES CONTAINGN PROPERTIES, CLASS AND OTHER FROM ONTOLOY IN PYTHON FILE NAME ALSO CHANGES 
#***** BUT HERE ONLY WE RUN THIS CODE FOR 1st time to getting the same name as ontolgy
#******* ye code label mean file name ko label karne ki liye be use ho sake gaa at the end

# pathfolder="G:/Synopsis Tahir/LDATOPICMODELLING/ontoloy courpus for lda//"
# 
# file_names_old <- list.files(pathfolder)              # Get current file names
# for (j in seq(file_names_old)) {
#  
#    x=file_names_old[j]
#   print(x)
# 
# y=substr(x, 35, nchar(x))
#   print(y)
#   file.rename(paste0(pathfolder,x),paste0(pathfolder,y))
# 
#   }


#*******************
#**********************



docs <- Corpus(DirSource("G:/Synopsis Tahir/LDATOPICMODELLING/ontoloy courpus for lda//"))
#this read the
#files in text from our textming forlder
#and store results in docs VARIABLE
summary(docs)# this give detail about our corpus

# Count stopwords BEFORE removal
stopword_list <- stopwords("english")
total_stopwords_before <- 0
found_stopwords_before <- c()

for (doc in docs) {
  words <- unlist(strsplit(tolower(as.character(doc)), "\\s+"))
  clean_words <- gsub("[[:punct:]]", "", words)  # Remove punctuation attached to words
  found <- clean_words[clean_words %in% stopword_list]
  found_stopwords_before <- c(found_stopwords_before, found)
  total_stopwords_before <- total_stopwords_before + length(found)
}

cat("🔹 Total stop words before removal:", total_stopwords_before, "\n")
cat("🔹 Unique stop words found:\n")
print(unique(found_stopwords_before))
# Count frequency of each stopword before removal
stopword_freq_table <- table(found_stopwords_before)
cat("🔢 Frequency of each stopword before removal:\n")
print(stopword_freq_table)

#inspect a particular document
#PRE-PROCESSING OF DATA
#library(stringr)
#library(stringi) 
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
writeLines(as.character(docs[[141]]))#single document file detail all content this show

#PRE-PROCESSING OF DATA
getTransformations()#[1] "removeNumbers" "removePunctuation" "removeWords" "stemDocument" "stripWhitespace"
getTransformations()# its give list of availabe transformation in tm

#create the toSpace content transformer is used to apply custom transformation on data
#content_tranro takes input funtion which we want to apply transfoion type
#Create content transformers, i.e., functions which modify the content of an R object.
toSpace <- content_transformer(function(x, pattern) {return (gsub(pattern," ", x))})
#to space etc are contet_trans which we appy on our data
docs <- tm_map(docs, toSpace, "-")

docs <- tm_map(docs, toSpace, ":")          
#Remove punctuation - replace punctuation marks with " "
#docs <- tm_map(docs, removePunctuation) #THIS REMOVE ALL / SO ALL DATA MERGERD WHICH IS NOT USEFUL                              

docs <- tm_map(docs, toSpace, "'_\ '")
inspect(docs[71]) #single document file detail all content this show (SAME AS WRITE LINE)
docs <- tm_map(docs, toSpace, "'_\ '")
    docs <- tm_map(docs, toSpace, "'")
docs <- tm_map(docs, toSpace, "'-\'")
docs <- tm_map(docs, toSpace, " ,")

#docs <- tm_map(docs, toSpace, "\\")

#Transform to lower case (need to wrap in content_transformer)
##docs <- tm_map(docs,content_transformer(tolower))THIS ALSO MAKE PROBLEM SO WE NOT USE IT
#Strip digits (std transformation, so no need for content_transformer)
docs <- tm_map(docs, removeNumbers)
writeLines(as.character(docs[[91]]))#single document file detail all content this show
# Count stopwords AFTER removal
total_stopwords_after <- 0
found_stopwords_after <- c()

for (doc in docs) {
  words <- unlist(strsplit(tolower(as.character(doc)), "\\s+"))
  clean_words <- gsub("[[:punct:]]", "", words)
  found <- clean_words[clean_words %in% stopword_list]
  found_stopwords_after <- c(found_stopwords_after, found)
  total_stopwords_after <- total_stopwords_after + length(found)
}

cat("🔻 Total stop words after removal:", total_stopwords_after, "\n")
cat("🔻 Unique stop words still remaining:\n")
print(unique(found_stopwords_after))
summary(docs)
docs <- tm_map(docs, toSpace, "Synopsis Tahir")
docs <- tm_map(docs, toSpace, "LDATOPICMODELLING")
docs <- tm_map(docs, toSpace, "ontology data set")
#docs <- tm_map(docs, toSpace, "FoodDrink")
docs <- tm_map(docs, toSpace, "G")
docs <- tm_map(docs, toSpace, "LDATOPICMODELLIN")
writeLines(as.character(docs[[133]]))#single document file detail all content this show


writeLines(as.character(docs[[141]]))#single document file detail all content this show
for (j in seq(docs)) {
  docs[[j]] <- gsub("/", " ", docs[[j]])
  docs[[j]] <- gsub("@", " ", docs[[j]])
  docs[[j]] <- gsub("\\|", " ", docs[[j]])
  docs[[j]] <- gsub("\\|", " ", docs[[j]])
  
  docs[[j]] <- gsub("'\'", " ", docs[[j]])
  
  #  docs[[j]] <- gsub("\u2028",docs " ", docs[[j]])
  
  #  docs[[j]] <- gsub("[^[:alnum:][:blank:]?&/\\-]", "", docs[[j]])
  docs[[j]] <- gsub("(?<=[A-Za-z])(?=[A-Z])", " ", docs[[j]], perl = TRUE)
  docs[[j]] <- gsub("(?<=\\s)(\\w{1,3}\\s)"," ", docs[[j]], perl = TRUE)
  
  docs[[j]] <-gsub("#", " ", docs[[j]], fixed = TRUE)
  docs[[j]] <-gsub(",", " ", docs[[j]], fixed = TRUE)
  docs[[j]] <-gsub(".", " ", docs[[j]], fixed = TRUE)
  docs[[j]] <-gsub("_", " ", docs[[j]], fixed = TRUE)
  
  docs[[j]] <-gsub("-", " ", docs[[j]], fixed = TRUE)
  docs[[j]] <-gsub("[", " ", docs[[j]], fixed = TRUE)
  docs[[j]] <-gsub("]", " ", docs[[j]], fixed = TRUE)
  docs[[j]] <-gsub("\\", " ", docs[[j]], fixed = TRUE)
  
  
  #my_corpus[[j]] <- gsub('\\.', " ", my_corpus[[j]], perl = TRUE)
}

writeLines(as.character(docs[[91]]))#single document file detail all content this show

docs <- tm_map(docs, toSpace, "'-\'")

writeLines(as.character(docs[[111]]))
docs <- tm_map(docs, toSpace, "'\ \ \ \'")

writeLines(as.character(docs[[133]]))

docs <- tm_map(docs, toSpace, " \n")

writeLines(as.character(docs[[133]]))

writeLines(as.character(docs[[133]]))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))


docs <- tm_map(docs, toSpace, " E ")
docs <- tm_map(docs, toSpace, " f ")

writeLines(as.character(docs[[133]]))

docs <- tm_map(docs, toSpace, " ")
docs <- tm_map(docs, toSpace, "fccb")
writeLines(as.character(docs[[133]]))
docs <- tm_map(docs, removeWords, stopwords("english"))

docs <- tm_map(docs, toSpace, " computer 10")
docs <- tm_map(docs, toSpace, "computercomputer")
docs <- tm_map(docs, toSpace, "bioasthmavMOCHA")


writeLines(as.character(docs[[133]]))#single document file detail all content this show

writeLines(as.character(docs[[61]]))
for (j in seq(docs)) {
  docs[[j]] <- gsub("(?<=\\s)(\\w{1,2}\\s)"," ", docs[[j]], perl = TRUE)
  

  #my_corpus[[j]] <- gsub('\\.', " ", my_corpus[[j]], perl = TRUE)
}
writeLines(as.character(docs[[61]]))


summary(docs)
inspect(docs[1])

writeLines(as.character(docs[[133]]))#single document file detail all content this show

docs <- tm_map(docs, removePunctuation) #THIS REMOVE ALL / SO ALL DATA MERGERD WHICH IS NOT USEFUL                              
inspect(docs[133])
# for (j in seq(docs)) {
#  docs[[j]] <- gsub("/", " ", docs[[j]])
#  docs[[j]] <- gsub("@", " ", docs[[j]])
#  docs[[j]] <- gsub("\\|", " ", docs[[j]])
# 
#  
#  #  docs[[j]] <- gsub("\u2028",docs " ", docs[[j]])
#   
# #  docs[[j]] <- gsub("[^[:alnum:][:blank:]?&/\\-]", "", docs[[j]])
# docs[[j]] <- gsub("(?<=[A-Za-z])(?=[A-Z])", " ", docs[[j]], perl = TRUE)
# docs[[j]] <- gsub("(?<=\\s)(\\w{1,3}\\s)"," ", docs[[j]], perl = TRUE)
#   
#     #my_corpus[[j]] <- gsub('\\.', " ", my_corpus[[j]], perl = TRUE)
# }
writeLines(as.character(docs[[133]]))#single document file detail all content this show

#load library
#library(SnowballC)
#Stem document
#docs <- tm_map(docs,stemDocument)

#Transform to lower case (need to wrap in content_transformer)
###docs <- tm_map(docs,content_transformer(tolower))
###Strip whitespace (cosmetic?)
###docs <- tm_map(docs, stripWhitespace)
#Stem document
#There is a more sophisticated procedure called lemmatization that takes grammatical context into account. 
# organiz and organis are actually variants of the same stem organ.

#####we will not do stem bcz its mixs some term so here we are skiping it 25/march/2021
#docs <- tm_map(docs,stemDocument)

writeLines(as.character(docs[[111]]))
docs <- tm_map(docs, content_transformer(gsub), pattern = "organiz", replacement = "organ")
docs <- tm_map(docs, content_transformer(gsub), pattern = "organis", replacement = "organ")
docs <- tm_map(docs, content_transformer(gsub), pattern = "andgovern", replacement = "govern")
docs <- tm_map(docs, content_transformer(gsub), pattern = "inenterpris", replacement = "enterpris")
docs <- tm_map(docs, content_transformer(gsub), pattern = "team-", replacement = "team")
#docs <- tm_map(docs, content_transformer(gsub), pattern = " \ \ \ \ ", replacement = " ")
writeLines(as.character(docs[[133]]))

#Note that I have removed the stop words and and in in the 3rd and 4th transforms above.
#There are definitely other errors that need to be cleaned up,
#but I'll leave these for you to detect and remove.
#define and eliminate all custom stopwords

###we esckip it 25/03/2021
#docs <- tm_map(docs, removeNumbers)
#docs <- tm_map(docs, removeWords, stopwords("english"))
#docs <- tm_map(docs, removeWords, 'Tahir')
#docs <- tm_map(docs, removeWords, 'Synopsis')
####
#The document term matrix
writeLines(as.character(docs[[58]]))
inspect(docs[58])







########################################
#Cross validation  KKKKKKKKK

#here we find the no of desired topic automaticaly we try to trase it use perplexity
#here we use DTM but DTM is created in later code so the purpose of this is to only use it once
#when we Dtm created thn if want to find k mean perplexity thn we run this code
# this code need to run only once.
library(parallel)
library(tm)
#library(pdftools)
#library(tidyverse)
library(topicmodels)
#library(tidytext)
#library(ggraph)
#library(igraph)
#library(kableExtra)
library(doParallel)
library(doParallel)
inspect(DTM)
n <- nrow(DTM)
n
#Create training and test dataset
#in this case 75% is in the training set and 25% in the testset
splitter <- sample(1:n, round(n * 0.75))
train_set <- DTM[splitter, ]
test_set <- DTM[-splitter, ]
#----------------5-fold cross-validation, different numbers of topics----------------
#Use multiple cores for faster runtime
cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(topicmodels)
})
#select parameters for cross validation
burnin <- 1000 
iter <- 1000
thin <- 500
seed <-list(34,447,6798,1000457,2244)
keep <- 50
best <- TRUE
folds <- 5
splitfolds <- sample(1:folds, n, replace = TRUE)
candidate_k <- c(2, 5, 10, 20, 30, 40, 50, 75)#c(100,200)  #, 100, 200, 300 candidates for how many topics
clusterExport(cluster, c("DTM", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k", "LDA"))
# we parallelize by the different number of topics.  A processor is allocated a value
# of k, and does the cross-validation serially.  This is because it is assumed there
# are more candidate values of k than there are cross-validation folds, hence it
# will be more efficient to parallelise
system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("k", "perplexity")
    for(i in 1:folds){
      train_set <- DTM[splitfolds != i , ]
      test_set <- DTM[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(burnin = burnin, iter = iter, keep=keep ) )
      # inspect(fitted)
       writeLines(as.character(fitted))
      
      results_1k[i,] <- c(k, perplexity(fitted, newdata = DTM))
      
    }
    return(results_1k)
  }
})


 #Elapsed Time is the time charged to the CPU(s) for the expression.

#User Time is the wall clock time. The time that you as a user experienced.

#Usually both times are relatively close. But they may vary in some other situations. For example:

# If elapsed time > user time, this means that the CPU is waiting around for some other operations (may be external) to be done.
#If elapsed time < user time, this means that your machine has multiple cores and is able to use them
stopCluster(cluster)
results_df <- as.data.frame(results)
inspect(results)
#Export results to csv'
write.csv(results_df,file=paste("cluster analysis.csv"))
#Plot
#remove.packages("ggplot2") # Unisntall ggplot
#install.packages("ggplot2") # Install it again
library(ggplot2) # Load the librarie (you have to do this one on each new session)

ggplot(results_df, aes(x = k, y = perplexity)) +
  geom_point(color='black') +
  geom_smooth(color="RED", se = FALSE) +
  ggtitle("5-Fold Cross Validation of Topic Modelling") +
  labs(x = "Number  of topics", y = "Perplexity") +
  theme_bw() +
  


##########################above all perplexity to find BEST KKKKKKKKKKKK





#################################################LDA topic Modelling##############################
str(docs[[13]])
# compute document term matrix with terms >= minimumFrequency
minimumFrequency <- 1
DTM <- DocumentTermMatrix(docs, control = list(bounds = list(global = c(minimumFrequency, Inf))))
DTM
inspect(DTM)
# have a look at the number of documents and terms in the matrix
dim(DTM)
write.csv(as.matrix(DTM),file=paste("DTM.csv"))
#141 466
# due to vocabulary pruning, we have empty rows in our DTM
# LDA does not like this. So we remove those docs from the
# DTM and the metadata
sel_idx <- slam::row_sums(DTM) > 0
DTM <- DTM[sel_idx, ]
#docs <- docs[sel_idx, ]
# number of topics
inspect(DTM)
K <- 14
# set random number generator seed
set.seed(1253)
#install.packages('topicmodels')
#library(topicmodels)
# compute the LDA model, inference via 1000 iterations of Gibbs sampling
topicModel <- LDA(DTM, K, method="Gibbs", control=list(iter = 500, verbose = 25))

################################## THIS CODE IS TO FIND LOGLIKEHOOD OF THE MODEL###################
best.model <- lapply(seq(2,20, by=1), function(k){LDA(DTM, K, method="Gibbs", control=list(iter = 100, verbose = 100))})
best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik)))

best.model.logLik.df <- data.frame(topics=c(2:20), LL=as.numeric(as.matrix(best.model.logLik)))

library(ggplot2)
ggplot(best.model.logLik.df, aes(x=topics, y=LL)) + 
  xlab("Number of topics") + ylab("Log likelihood of the model") + 
  geom_line() + 
  theme_bw() 
best.model.logLik.df[which.max(best.model.logLik.df$LL),]



#######################################ABOVE CODE PORTION IS TO FIND LOG LIKELIHOOD OF THE
str(topicModel)
terms(topicModel, 5)

help("logLik,TopicModel-method")

plot(logLik(topicModel))

# # have a look a some of the results (posterior distributions)
# tmResult <- posterior(topicModel)
# # format of the resulting object
# attributes(tmResult)
# nTerms(DTM)              # lengthOfVocab




#we cover topics and this code  give top word of each topic
#we can go thorough individually one by one from all 15
#we can use loop to see all 15 at once

#The posterior function gives the posterior distribution of words and documents to topics, 
#which can be used to plot a word cloud of terms proportional to their occurrence:

  
library(wordcloud)
#IF PROBLEM IN POSTEREIO SOLUTION IS TO INCLUDE OTHER POSTER I.E 1POSTERTIER(TEXTMINR) 
#R=posterior(topicModel)
#R

words = posterior(topicModel)$terms[1, ]
topwords = head(sort(words, decreasing = T), n=50)
head(topwords)

words = posterior(topicModel)$terms[2, ]
topwords = head(sort(words, decreasing = T), n=50)
head(topwords)


words = posterior(topicModel)$terms[3, ]
topwords = head(sort(words, decreasing = T), n=50)
head(topwords)
#str(topicModel)
################ in this we get the most top word in this topic
#mean as we discuss above separately here we get one by one that what 
#top word are included in it
#here wordcloud display top in single document
##### probably that will be topic we can visualize them 

library(wordcloud)
wordcloud(names(topwords), topwords)


logLik(topicModel)



#in this code we use loop to see all 15 at once and together
#############################

#We can also look at the topics per document, to find the top documents per topic:
#this tell that this topic no is represented by whom documents
#this give no of topics in which this topic exist which is representing it
#all the document from which this topic came weill be shown their name
#We can also look at the topics per document, to find the top documents per topic:
for (x in seq(K)) {
  print(x)


topicname<-terms(topicModel, 1)

topicname
topicname[x]

  #my_corpus[[j]] <- gsub('\\.', " ", my_corpus[[j]], perl = TRUE)

  
topic=x
topic.docs = posterior(topicModel)$topics[, topic] 
#topic.docs
topic.docs = sort(topic.docs, decreasing=T)
y<-head(topic.docs, n=10)
y
  write.table(topicname[x],file=paste("Topicfiles.csv"),append = TRUE, col.names = FALSE )
write.table(as.matrix(y),file=paste("Topicfiles.csv"),append = TRUE, col.names = FALSE)
}

#code to exclue file less than .6 probabilty all other values and code will
# be same just include only those ontologies whose proballity is .6 or grater


for (x in seq(K)) {
  print(x)

  topicname<-terms(topicModel, 1)
  
  topicname
  topicname[x]
  topic=x
  
  topic.docs = posterior(topicModel)$topics[, topic] 
  #topic.docs
  topic.docs = sort(topic.docs, decreasing=T)
  y<-head(topic.docs, n=10)
  y
 
  write.table(topicname[x],file=paste("Topicfilesincluded.csv"),append = TRUE,col.names = FALSE )
  
  # pathfolder="G:/Synopsis Tahir/LDATOPICMODELLING/ontoloy courpus for lda//"
  #
  # file_names_old <- list.files(pathfolder)              # Get current file names
  for (j in seq(y)) {

     p=y[j]
    print(p)
    print("ppppp")

   z=substr(p, 1, nchar(p))
   print(z)
   if(z>.6){
       write.table(as.matrix(y[j]),file=paste("Topicfilesincluded.csv"),append = TRUE, col.names = FALSE)
    
  }
  
  }

}








#write.table((topicname),file=paste("Topicfiles.csv"))



#lda_ap4 <- LDA(DTM,
#               control = list(seed = 33), k = 12)
# See the top 10 terms associated with each of the topics
# Calculate all diagnostics for each topic in the topic model

#install.packages("magrittr") # package installations are only needed the first time you use it
#install.packages("dplyr")    # alternative installation of the %>%
#install.packages("tidyr")    # alternative installation of the %>%
#install.packages("stringr")    # alternative installation of the %>%



library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(ggplot2)
library(tidyr)
library(stringr)




diag_df <- topic_diagnostics(topicModel, DTM)
diag_df
topic_size(topicModel)

# Not using the pipe %>%, this code would return the same as your code:
# words <- colnames(as.matrix(dtm))
# words <- words[nchar(words) < 20]
# words

#diag_df[order(diag_df)] <- diag_df %>%
  
  diag_df<- diag_df %>%
      mutate(topic_label = terms(topicModel, 5) %>%
           apply(2, paste, collapse = "- "),
         topic_label = paste(topic_num, topic_label, sep = " - "))

diag_df %>% 
  gather(diagnostic, value, -topic_label, -topic_num) %>%
  ggplot(aes(x = topic_num, y = value,
             fill = str_wrap(topic_label, 25))) +
  geom_bar(stat = "identity") +
  facet_wrap(~diagnostic, scales = "free") +
  labs(x = "Topic Number", y = "Diagnostic Value",
       fill = "Topic Label", title = "All Topic Model Diagnostics")


#plot(topicModel@log_likelihood)
#is me hum topic coherence krte hn kud se verify b ho gea he ki oper diagonistic se value teak aye he
#hum isko chek kr skte hn ki oper wali aur ye value same aye hn hum isko draw b kar skte hn .ok 
#hum inko alag alag graph be kr skte he 



#histo bata rhe he ke zadia topic kis me he mena cohreance ki zaida value kis ki darmian he
# hamri cohere ki zaida value 80 aur 110 ki darmaina hea so 
# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones

topic_coherence(topicModel, DTM, top_n_tokens = 10)
hist(topic_coherence(topicModel, DTM, top_n_tokens = 10))
doc_prominence(topicModel)
dist_from_corpus(topicModel, DTM)
tf_df_dist(topicModel, DTM, top_n_tokens = 10)
topic_exclusivity(topicModel, top_n_tokens = 10)
topic_size(topicModel)
hist(topic_size(topicModel))
plot(topic_size(topicModel), type = "c")
mean_token_length(topicModel, top_n_tokens = 10)

###################################


##### LSI MODEL FOR SAME DATA SET 
### WE COMPARE WHICH IS BEst LDA OR LSA
#The workflow for LSA is largely the same for LDA.
#Two key differences: we will use the IDF vector mentioned
#above to create a TF-IDF matrix and we cannot get 
#an R-squared for LSA as it is non-probabilistic.


 #install.packages('textmineR')    # alternative installation of the %>%
 library(textmineR)
#library(lsa)

#install.packages('LSAfun')
#install.packages('lsa')
library(LSAfun)
library(lsa)
library("ggplot2")  

dim(DTM)
DTM
inspect(DTM)
DTMMATRIX<-as.matrix(DTM)
DTMMATRIX
tf_sample <- TermDocFreq(DTMMATRIX)

tf_sample
getwd()
write.csv(tf_sample, "LSITermFrequ.csv")
tf_sample$idf[ is.infinite(tf_sample$idf) ] <- 0 # fix idf for missing words
tf_sample
#write_xlsx(tf_sample,"G:\\Synopsis Tahir\\LDATOPICMODELLING\\LSITermFrequency.xlsx")#G:\Synopsis Tahir\LDATOPICMODELLING
DTMMARTIX=as.matrix(DTM)
DTMMARTIX
DTMMARTIXROWSOM=(rowSums(DTMMARTIX))
DTMMARTIXROWSOM
trnsp<-t((DTMMARTIX / DTMMARTIXROWSOM))
trnsp


tf_idf <- trnsp * tf_sample$idf

tf_idf

tf_idf <- t(tf_idf)
tf_idf

# Fit a Latent Semantic Analysis model
# note the number of topics is arbitrary here
# see extensions for more info
lsa_model <- FitLsaModel(dtm = tf_idf,k=14,optimize_alpha = TRUE,
                         calc_likelihood = TRUE,
                         calc_coherence = TRUE,
                         calc_r2 = TRUE)

lsa_model






#> Warning in fun(A, k, nu, nv, opts, mattype = "dgCMatrix"): all singular
#> values are requested, svd() is used instead

# objects: 
# sv = a vector of singular values created with SVD
# theta = distribution of topics over documents
# phi = distribution of words over topics
# gamma = predition matrix, distribution of topics over words
# coherence = coherence of each topic
# data = data used to train model
str(lsa_model)

lsa_model$log_likelihood
# probabilistic coherence, a measure of topic quality
# - can be used with any topic lsa_model, e.g. LSA
# probabilistic coherence, a measure of topic quality
# - can be used with any topic lsa_model, e.g. LSA
lsa_model$coherence <- CalcProbCoherence(phi = lsa_model$phi, dtm = DTMMARTIX, M = 10)
summary(lsa_model$coherence)
#>     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#> -0.05107  0.18792  0.29267  0.34172  0.38925  0.99000

hist(lsa_model$coherence, col= "blue")


# Get the top terms of each topic
lsa_model$top_terms <- head(GetTopTerms(phi = lsa_model$phi, M =5 ))
#HERE WE 
(lsa_model$top_terms)

lsa_model$top_terms[1:1,1:14]





# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
lsa_model$prevalence <- colSums(lsa_model$theta) / sum(lsa_model$theta) * 100

# textmineR has a naive topic labeling tool based on probable bigrams
lsa_model$labels <- LabelTopics(assignments = lsa_model$theta > 0.05, 
                                DTMMARTIX,
                                M = 15)
head(lsa_model$labels)

head(lsa_model$labels)





# put them together, with coherence into a summary table
lsa_model$summary <- data.frame(topic = rownames(lsa_model$phi),
                                label = lsa_model$labels,
                                coherence = round(lsa_model$coherence, 3),
                                prevalence = round(lsa_model$prevalence,3),
                                top_terms = apply(lsa_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)
#we can sort this topic and label so easilyl by using sort as above in lda

View(lsa_model$summary[ order(lsa_model$summary$prevalence, decreasing = TRUE) , ])

write.csv(lsa_model$summary, "LSAMODELSUMMARY.csv")


library(ggplot2) # Load the librarie (you have to do this one on each new session)

ggplot(data.frame(lsa_model$coherence)), aes(x = k, y = perplexity)) +
  geom_point(color='black') +
  geom_smooth(color="RED", se = FALSE) +
  ggtitle("5-Fold Cross Validation of Topic Modelling") +
  labs(x = "Number  of topics", y = "Perplexity") +
  theme_bw() +
  

ggplot(data.frame(lsa_model$coherence))##, aes(x=year, y=gdpPercap)) + geom_point()
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(ggplot2)
library(tidyr)
library(stringr)








# Get topic predictions for all 2,000 documents

# first get a prediction matrix,
lsa_model$phi_prime <- diag(lsa_model$sv) %*% lsa_model$phi

lsa_model$phi_prime <- t(MASS::ginv(lsa_model$phi_prime))

# set up the assignments matrix and a simple dot product gives us predictions
lsa_assignments <- t(DTMMARTIX) * tf_sample$idf

lsa_assignments <- t(lsa_assignments)

lsa_assignments <- lsa_assignments %*% t(lsa_model$phi)

lsa_assignments <- as.matrix(lsa_assignments) # convert to regular R dense matrix
barplot(lsa_model$theta[ rownames(DTMMARTIX)[ 1 ] , ], las = 2,
        main = "Topic Assignments While Fitting LSA")

barplot(lsa_assignments[ rownames(DTMMARTIX)[ 1 ] , ], las = 2,
        main = "Topic Assignments Predicted Under the Model")
Extensions




# extra code of lda ye hum ik lda phle kr chukey hn magr
# ye ab wala same ho ga jese for same like lsa

# hum ne 2 lda implement kiye hn ik textmier ka aur i topicmodel ka

set.seed(12345)




m <-  Matrix::sparseMatrix(i=DTM$i, 
                           j=DTM$j, 
                           x=DTM$v, 
                           dims=c(DTM$nrow, DTM$ncol),
                           dimnames = DTM$dimnames)



dim(DTM)
dim(m)
DTM
m
inspect(DTM)


model <- FitLdaModel(dtm = m, method = "gibbs",
                     k = 14,
                     iterations = 500, # I usually recommend at least 500 iterations or more
                     burnin =400,
                     alpha = 0.1,
                     beta = 0.05,
                     optimize_alpha = TRUE,
                     calc_likelihood = TRUE,
                     calc_coherence = TRUE,
                     calc_r2 = TRUE,
                     cpus = 2) 


str(model)

plot(model$log_likelihood, type = "l")
# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
summary(model$coherence)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>  0.0060  0.1188  0.1543  0.1787  0.2187  0.4117

hist(model$coherence, 
     col= "blue", 
     main = "Histogram of probabilistic coherence")

# Get the top terms of each topic
model$top_terms <- GetTopTerms(phi = model$phi, M = 5)
/head((model$top_terms))
model$top_terms
# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
model$prevalence <- colSums(model$theta) / sum(model$theta) * 100

# prevalence should be proportional to alpha
plot(model$prevalence, model$alpha, xlab = "prevalence", ylab = "alpha")

model$prevalence
model$log_likelihood
model$coherence
# textmineR has a naive topic labeling tool based on probable bigrams
model$labels <- LabelTopics(assignments = model$theta > 0.05, 
                            dtm = m,
                            M = 8)

head(model$labels)
#>     label_1                      
#> t_1 "radiation_necrosis"         
#> t_2 "kda_fragment"               
#> t_3 "cardiovascular_disease"     
#> t_4 "mast_cell"                  
#> t_5 "radiation_necrosis"         
#> t_6 "laparoscopic_fundoplication"

# put them together, with coherence into a summary table
model$summary <- data.frame(topic = rownames(model$phi),
                            label = model$labels,
                            coherence = round(model$coherence, 3),
                            prevalence = round(model$prevalence,3),
                            top_terms = apply(model$top_terms, 2, function(x){
                              paste(x, collapse = ", ")
                            }),
                            stringsAsFactors = FALSE)
model$summary[ order(model$summary$prevalence, decreasing = TRUE) , ][ 1:13 , ]

write.csv(model$summary, "LDAModelSumayTextminer.csv")

model$coherence
#ABOVE ALL CLEAR BY ME ON 22 june  21
############################## this was final code file 2
