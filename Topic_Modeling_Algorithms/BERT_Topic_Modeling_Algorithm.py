# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:02:53 2023

@author: TAHIR BASHIR KAYANI
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:02:23 2023

@author: TAHIR BASHIR KAYANI
"""
import os
from bertopic import BERTopic
from top2vec import Top2Vec
#from tqdm import tqdm
import pandas as pd
# Specify the directory where your text files are located
#folder_path = "C:/Users/nabee/Downloads/datasetbert"
folder_path = "G:/Synopsis Tahir/LDATOPICMODELLING/final bert dataset after preprocessing"


# Initialize a list to store the contents of the text files
dataset = []

# Loop through all the files in the specified folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file (you can adjust the condition as needed)
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # Open the file and read its contents
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            
            # Append the file content to the dataset
            dataset.append(file_content)
words_to_remove=["\n","list","(language",")","\nlist()\n","list(language ="," list(language =",'list(language = "en") list()','list(language = "en") list()']
filtered_dataset=[" ".join([word for word in text.split() if word not in words_to_remove]) for text in dataset]
#print(filtered_dataset)
#print(len(dataset))
#print(dataset)
# for item in dataset:
#     #print(dataset[101])
#    print(item)   
    #break
#HERE ALL IS GOOD SO WE CAN GO WITH THIS


# # #all top2vec but issue its gets only 2 topics why? just look it
# print("test")  
# model=Top2Vec(dataset)#, embedding_model='universal-sentence-encoder')
# model.topic_merging=0.3  
# #model.min_topic_size=1
# #model.nr_topics=10
# print("no of topics which we get are in TOP2VEC=")

# #= model.get_num_topics()
# print(model.get_topics())
# print(model.get_num_topics())
# #print(model.get_num_topics())


#here issue its get only two topic we will will put some argument as we set 
#in BERTopic minsize4 and nr 14
# topic_size, topic_nums=model.get_topic_sizes()
# print(topic_size)
# print(topic_nums)

# #all above @* top2vec but issue its gets only 2 topics why? just look it


topic_model = BERTopic(min_topic_size=3,nr_topics=14,embedding_model="bert-base-nli-stsb-mean-tokens")
topics, probs = topic_model.fit_transform(filtered_dataset)
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from bertopic import BERTopic

# Extract the topic assignments for each document
topics = topic_model.transform(filtered_dataset)  # Replace your_text_corpus with your actual text data
topics

# Tokenize your text corpus if it's not already tokenized
tokenized_corpus = [text.split() for text in filtered_dataset]

# Create a Gensim dictionary from the tokenized_corpus
dictionary = Dictionary(tokenized_corpus)

# Extract topic words
topic_words = topic_model.get_topics()  
topic_words[12]


# Extract the topic assignments for each document
topics = topic_model.transform(filtered_dataset)

# Tokenize your text corpus if it's not already tokenized
tokenized_corpus = [text.split() for text in filtered_dataset]

# Create a Gensim dictionary from the tokenized_corpus
dictionary = Dictionary(tokenized_corpus)

# Extract topic words
# Extract topic words
topic_words_ids = topic_model.get_topics()
 
topic_words_ids
# Convert word IDs back to tokens using the Gensim dictionary
topic_words_tokens = [
    [word[0] for word in word_ids] for word_ids in topic_words_ids.values()
]
topic_words_tokens
i=0
topicandcoherencevalue=[]
topicandcoherencevalue.clear()

for i in range(10):
    # Choose a specific topic, for example, the first topic (index 0)
    selected_topic = topic_words_tokens[i]#here pass one by one till 13-14 topics reached
    selected_topic[i]
   #print(selected_topic[i])
    # Calculate the coherence score using the c_v coherence measure for the selected topic
    coherence_model = CoherenceModel(
        topics=[selected_topic],  # Note: Treat the selected topic as a separate list of tokens
        texts=tokenized_corpus,
        dictionary=dictionary,
        coherence="c_v"
        )
    coherence_score = coherence_model.get_coherence()
    coherence_score
   # print(f"Coherence Score for Selected Topic: {coherence_score}")
    topicandcoherencevalue.append([selected_topic[0], coherence_score])
    #print(topicandcoherencevalue)
print("here is the cohence value for our each topic which model get")
topicandcoherencevalue

#after 9 loop range stop so we get and write code for 10,11,12,13 and append one by one okkkkkk? clear?

    # Choose a specific topic, for example, the first topic (index 0)
selected_topic = topic_words_tokens[10]#here pass one by one till 13-14 topics reached
selected_topic
#print(selected_topic[i])
# Calculate the coherence score using the c_v coherence measure for the selected topic
coherence_model = CoherenceModel(
        topics=[selected_topic],  # Note: Treat the selected topic as a separate list of tokens
        texts=tokenized_corpus,
        dictionary=dictionary,
        coherence="c_v"
        )
coherence_score = coherence_model.get_coherence()
coherence_score
   # print(f"Coherence Score for Selected Topic: {coherence_score}")
topicandcoherencevalue.append(['people', coherence_score])
    #print(topicandcoherencevalue)
print("here is the cohence value for our each topic which model get")
topicandcoherencevalue

selected_topic = topic_words_tokens[11]#here pass one by one till 13-14 topics reached
selected_topic
#print(selected_topic[i])
# Calculate the coherence score using the c_v coherence measure for the selected topic
coherence_model = CoherenceModel(
        topics=[selected_topic],  # Note: Treat the selected topic as a separate list of tokens
        texts=tokenized_corpus,
        dictionary=dictionary,
        coherence="c_v"
        )
coherence_score = coherence_model.get_coherence()
coherence_score
   # print(f"Coherence Score for Selected Topic: {coherence_score}")
topicandcoherencevalue.append(['family', coherence_score])
    #print(topicandcoherencevalue)
print("here is the cohence value for our each topic which model get")
topicandcoherencevalue

selected_topic = topic_words_tokens[12]#here pass one by one till 13-14 topics reached
selected_topic
#print(selected_topic[i])
# Calculate the coherence score using the c_v coherence measure for the selected topic
coherence_model = CoherenceModel(
        topics=[selected_topic],  # Note: Treat the selected topic as a separate list of tokens
        texts=tokenized_corpus,
        dictionary=dictionary,
        coherence="c_v"
        )
coherence_score = coherence_model.get_coherence()
coherence_score
   # print(f"Coherence Score for Selected Topic: {coherence_score}")
topicandcoherencevalue.append(['uni', coherence_score])
    #print(topicandcoherencevalue)
print("here is the cohence value for our each topic which model get")
topicandcoherencevalue

selected_topic = topic_words_tokens[13]#here pass one by one till 13-14 topics reached
selected_topic
#print(selected_topic[i])
# Calculate the coherence score using the c_v coherence measure for the selected topic
coherence_model = CoherenceModel(
        topics=[selected_topic],  # Note: Treat the selected topic as a separate list of tokens
        texts=tokenized_corpus,
        dictionary=dictionary,
        coherence="c_v"
        )
coherence_score = coherence_model.get_coherence()
coherence_score
   # print(f"Coherence Score for Selected Topic: {coherence_score}")
topicandcoherencevalue.append(['closure', coherence_score])
    #print(topicandcoherencevalue)
print("here is the cohence value for our each topic which model get")
topicandcoherencevalue








import matplotlib.pyplot as plt


# Extract topic numbers and coherence scores
topics = [item[0] for item in topicandcoherencevalue]
coherence_scores = [item[1] for item in topicandcoherencevalue]

# Plotting the bar chart
plt.bar(topics, coherence_scores, color='blue')
plt.xlabel('Topic')
plt.ylabel('Coherence Score')
plt.title('Coherence Scores for Topics')
plt.show()











#after getting this value store it ia a array for every topics and than you may 
#draw the chart.
#just need to draw the chrat for all coherece values stored in array.

#In this code, we've made sure that topics is in the correct format, and we use a list comprehension to transform it into the required format for the CoherenceModel. This should resolve the "TypeError: 'int' object is not iterable" issue. Make sure to replace "your_bertopic_model" and "your_text_corpus" with your actual model and corpus data.





print(filtered_dataset)#all dataset in single file text so we can get topics form it
print(topic_model.fit_transform(filtered_dataset))


#need to run this code on other pc bcz its not working here why test it.ok
# import matplotlib.pyplot as plt
# figur=topic_model.visualize_barchart()
# plt.figure(figsize=(18,14))
# plt.imshow(figur)



topic_model.visualize_documents(filtered_dataset)
all_document_data=topic_model.fit_transform(filtered_dataset)
all_document_data_df=pd.DataFrame(all_document_data)
all_document_data_df.to_excel('BERT_Top_which_DOC_Cov_Whi_To.xlsx')
print(filtered_dataset.get_document_info)
print((topic_model.get_topic_freq()))
topic_model.get_topics() #get all topic top n words their ctf_idf score
all_topic_top=pd.DataFrame(topic_model.get_topics())
all_topic_top.to_excel('BERT_Top_all_tOP_WORD_CDTIDF.xlsx') #get all topic top n words their ctf_idf score

(topic_model.get_topic_freq()).to_excel('BERT_TOPIC_COUT_DOCUMENT.xlsx')
type(print((topic_model.get_topic_freq())))
print(topic_model.get_topic_info(1))
TOPIC1_DETAILS=topic_model.get_topic_info(1)
print(TOPIC1_DETAILS)
TOPIC1_DETAILS.to_excel('BERT_TOPIC1_DETAILS_INFO.xlsx')
TOPIC2_DETAILS=topic_model.get_topic_info(2)
print(TOPIC2_DETAILS)
TOPIC2_DETAILS.to_excel('BERT_TOPIC2_DETAILS_INFO.xlsx')
print(topic_model.get_topic_info())
topics_all=topic_model.get_topic_info()#here get all details about topic
topics_all.to_excel('BERT_TOPICS_ALL_DETAILS_INFO.xlsx')

print(TOPIC1_DETAILS)#to print all deatil
#TOPIC1_DETAILS.to_excel('TOPIC1_DETAILS_ONLY.xlsx')#to converrt and save in excel for anlysis completely
#print(TOPIC1_DETAILS.get('Count'))#getting columns and all data in it
#GETTING INDIVIDAUL COLUMNS


#all realated wordcloud for topic one and topic 2 
#for topic1 cloud
print(TOPIC1_DETAILS.get('Representative_Docs'))#getting columns and all data in it
items=[TOPIC1_DETAILS.get('Representative_Docs')]
label='my topic 1'

print(items)
type(items)
data_text=items[0].to_string(index=False)
print(data_text)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud =WordCloud(width=400, height=400, background_color='white').generate(data_text)
plt.figure(figsize=(8,4))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.imshow(wordcloud)
wordcloud.to_file('BERT_TOPICS1_WORDCLOUD.png')
#plt.savefig('wordcloud.png', bbox_inches='tight')

id="tahir"
#for topic2 cloud
TOPIC2_DETAILS=topic_model.get_topic_info(2)
print(TOPIC2_DETAILS)
print(TOPIC2_DETAILS.get('Representative_Docs'))#getting columns and all data in it
items=[TOPIC2_DETAILS.get('Representative_Docs')]
label='my topic 2'

print(items)
type(items)
data_text=items[0].to_string(index=False)
print(data_text)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud =WordCloud(width=400, height=400, background_color='white').generate(data_text)
plt.figure(figsize=(8,4))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.imshow(wordcloud)
wordcloud.to_file('BERT_TOPICS2_WORDCLOUD.png')
 #plt.savefig('wordcloud.png', bbox_inches='tight')
 
 
 #for all topic cloud
 

 
 
TOPICall_DETAILS=topic_model.get_topic_info()
print(TOPICall_DETAILS)
print(TOPICall_DETAILS.get('Representative_Docs'))#getting columns and all data in it
items=[TOPICall_DETAILS.get('Representative_Docs')]
label='my topic all'
print(items)
type(items)
data_text=items[0].to_string(index=False)
print(data_text)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud =WordCloud(width=400, height=400, background_color='white').generate(data_text)
plt.figure(figsize=(8,4))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.imshow(wordcloud)
wordcloud.to_file('BERT_TOPICSall_WORDCLOUD.png')
  #plt.savefig('wordcloud.png', bbox_inches='tight')
id="tahir"
#till above all ok just now read the specific colmn and pass it into word
#cloud we already read but just convet it in , separtion so we can 
#draw cloud ok )
#example also whatsapp today wordcloud

# this show topic 1 and topic 2 all in term of 0-1 that how many near to 1 
#mean most suitable one
topic_word=topic_model.get_topic(1)
print(topic_word)
topic_word.to_excel('BERT_TOPIC1_DETAILS_in_term_number_INFO.xlsx')
#above line is not woking please solve this issue
#AttributeError: 'list' object has no attribute 'to_excel'
#
#Coherence Score TOPIC FOR BERT.
from bertopic import BERTopic
from bertopic.evaluation import C_v
from gensim.models import CoherenceModel  # Import CoherenceModel
import numpy as np
from gensim.models import CoherenceModel
from bertopic import BERTopic


# Assuming you already have your BERTopic model and documents
your_topic_model = BERTopic.load("your_model")
your_documents = ["document 1", "document 2", ...]  # Replace with your actual documents

# Calculate coherence using BERTopic's C_v metric
coherence_calculator = C_v(your_topic_model, your_documents)
coherence_score = coherence_calculator.get_coherence()

print("Coherence Score:", coherence_score)



 
##coherence_score=model.


# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#wordcloud =WordCloud(width=400, height=400, background_color='white').generate(text_data)




# print(topic1.get_Count)
# print(topic1.get_topic)
# topic_info=topic_model.get_topic_info()
# topic_model.get_document_info(8)

    
#best result here  
#runfile('G:/NEWMODEL/NEWMODEL/finaltop2vec2310.py', wdir='G:/NEWMODEL/NEWMODEL')
#     Topic  Count
# 5      -1     27
# 7      10     23
# 4       4     13
# 11      8     12
# 0       1     10
# 2       2     10
# 1       7      9
# 3       9      9
# 6       3      8
# 8       5      7
# 9       6      7
# 10      0      7




  



#print(topic_model.get_topic_info())
#print(topic_model.get_topic(10))
# #print(topic_model.get_representative_docs(0))
# #topic_model.visualize_topics()
# df =pd.DataFrame({"topic": topics, "document":dataset})
# topic_model.visualize_topics()
# topic_model.visualize_barchart()


