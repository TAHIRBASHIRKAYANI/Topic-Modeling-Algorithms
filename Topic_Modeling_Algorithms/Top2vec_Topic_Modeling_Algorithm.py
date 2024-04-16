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





from gensim.models import CoherenceModel




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
print("test")
#print(filtered_dataset) 




  
model=Top2Vec(filtered_dataset, workers=4,)#, embedding_model='universal-sentence-encoder')
model=Top2Vec(filtered_dataset, speed="fast-learn",workers=4, verbose=True)

#topic_word, word_scrores, topic_num(model.get_topics(10))
#model=Top2Vec(filtered_dataset, embedding_model='universal-sentence-encoder', workers=4)

#model.hierarchical_topic_reduction(min_topic_size=2, method='average')
# model.topic_merging=0.3  
# model.min_topic_size=10
# model.nr_topics=10
print("no of topics which we get are in TOP2VEC=")

total_topic_get=model.get_num_topics()
print(total_topic_get)
topics = model.get_topics(2)
#get no of topics wiht each topic
topics
topic_sizes, topic_nums=model.get_topic_sizes()
topic_sizes

# Extract topic words
topic_words, _, _ = model.get_topics()  
topic_words

# Calculate coherence
import gensim
corpus = [doc.split() for doc in filtered_dataset]
dictionary = gensim.corpora.Dictionary(corpus)
corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]

coherence_model = CoherenceModel(
    topics=topic_words,
    texts=corpus,
    dictionary=dictionary,
    coherence='c_v'  # You can choose a different coherence measure if needed
)

coherence_score = coherence_model.get_coherence()

print(f"Coherence Score: {coherence_score}")

# Print topics and coherence scores
for i, (topic, coherence) in enumerate(zip(topic_words, coherence_model.get_coherence_per_topic())):
    print(f"\nTopic {i + 1}:\nWords: {topic}\nCoherence: {coherence}")

import matplotlib.pyplot as plt

# Plot bar chart
topics_range = range(1, len(topic_words) + 1)

#plt.bar(topics_range, coherence_model.get_coherence_per_topic(), align='center', alpha=0.7)

plt.bar(topics_range, coherence_model.get_coherence_per_topic(), align='center', alpha=0.7)
plt.xlabel('Topics').set_ylabel('Coherence Score')

plt.bar(topics_range, coherence_model.get_coherence_per_topic(), align='center', alpha=0.7)
plt.savefig('top2vec_coherence_plot.jpg')
plt.xlabel('Topics')
plt.ylabel('Coherence Score')
plt.title('Coherence Scores for Each Topic')
plt.show()



topic_nums
topic_words, word_scores, topic_nums=model.get_topics()
topic_words
topic_nums
topic_sizes
for words, scores, num in zip(topic_words, word_scores,topic_nums):
    print(num)
    print(f"Words:{words}")
for topic in topic_nums:
    model.generate_topic_wordcloud(topic, background_color='white')
# topic_model = BERTopic(min_topic_size=4,nr_topics=14,embedding_model="bert-base-nli-stsb-mean-tokens")
# topics, probs = topic_model.fit_transform(filtered_dataset)

#They shows two topics for the and detais here like figure 4
toic_sizes, topic_nums =model.get_topic_sizes()
topic_nums
toic_words, word_scrores, topic_nums =model.get_topics(1)
topic1=pd.DataFrame(toic_words)
topic1.to_excel('top2vectopic1.xlsx')
topic1
word_scrores
wordscore=pd.DataFrame(word_scrores)
wordscore.to_excel('top2vectopic1wordsore.xlsx')
toic_words, word_scrores, topic_nums =model.get_topics(2)
topic2=pd.DataFrame(toic_words)
topic2.to_excel('top2vectopic2.xlsx')
word_scrores
wordscore=pd.DataFrame(word_scrores)
wordscore.to_excel('top2vectopic2wordsore.xlsx')


#here till above we have save top 1 and top 2 in excell after 
#we get now wordcloud for these 2 here down above all cearl
#top2vec   
