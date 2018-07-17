from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import pandas as pd
import gensim
import json
import operator
import csv
from nltk.corpus import wordnet
from itertools import chain
import time,datetime,os

#Prereq variables
path='/Users/kaushik/Desktop/MS CS/Research/annotation task/'
corpus_location = "/Users/kaushik/Desktop/MS CS/Research/annotation task/final/"

ne_filename_df = pd.read_csv(path+'peopleNameDisambiDocs_top50.csv', header=None)

en_stop = None
dictionary_words_set = None
professionDict = None

with open("/Users/kaushik/Desktop/MS CS/Research/annotation task/mallet-stopwords-en.txt") as f:
    en_stop=set(f.read().split())
    
with open("/Users/kaushik/Desktop/MS CS/Research/annotation task/professionDict_new.json") as f:
    professionDict=json.loads(f.read())

dictionary_words_set = set(chain(*[syn.lemma_names() for syn in wordnet.all_synsets()]))

window_size_N=10
no_of_topics=3



def extract_words_in_window(tokens, named_entity, window_size_N):
    '''
    returns the set of words in the word window of size window_size_N around each occurance of named_entity in tokens
    '''
    ne_words = named_entity.split()
    token_count = len(tokens)
    ne_word_count = len(ne_words)
    document = []
    match_count=0
    i=0
    while i < token_count:
        match=True
        for j in range(ne_word_count):
            if(i+j >= token_count or tokens[i+j]!=ne_words[j]):
                match=False
                continue
        if(not match):
            document.append(tokens[i])
            i=i+1
        else:
            match_count = match_count + 1
            i= i + ne_word_count
    return document
                
def get_window_idxs(token_count, window_size_N, ne_pos, ne_word_count):
    '''
    returns the start indices and the end indices of the word window
    '''
    if (2 * window_size_N) > (token_count - ne_word_count):
        return [(0, token_count), (0, 0)]
    if ne_pos - window_size_N < 0:
        return [(0, ne_pos), (ne_pos + 2, ne_pos + ne_word_count + window_size_N + (window_size_N - ne_pos))]
    if ne_pos + ne_word_count + window_size_N > token_count:
        return [(ne_pos - window_size_N - (token_count - ne_pos - ne_word_count - window_size_N), ne_pos), (ne_pos + ne_word_count, token_count)]
    return [(ne_pos - window_size_N, ne_pos), (ne_pos + ne_word_count, ne_pos + ne_word_count + window_size_N)]

def generate_lda_model(file_names, named_entity, en_stop, tokenizer, window_size_N,no_of_topics,file,
                       dictionary_words_set,professionSet,wordsProfessiondict,csv_writer,throwaway_csv_writer,profession_words_writer):
    '''
    main method to generate the lda model
    '''
    ner_data={}
    texts = []
    document_set=[]
    file_name_doc_list=[]
    profession_match_count = 0
    profession_file_count_dict={}
    for filename in file_names:
        profession_count_dict={}
        document = None

        filename = corpus_location+filename.strip()
        with open(filename) as f:
            f=open(filename)
            content= (f.read())
            raw = content.lower()
            tokens = tokenizer.tokenize(raw)
            document = extract_words_in_window(tokens,named_entity,window_size_N)

        document_set = document_set + document
        token_list_reduced = [i for i in document if len(i)>2]
        token_list_thowaway = [i for i in document if len(i)<=2]
        stopped_tokens = [i for i in token_list_reduced if not i in en_stop]
        token_list_thowaway.extend([i for i in token_list_reduced if i in en_stop])
        tokens_noise_removed = [i for i in stopped_tokens if i in dictionary_words_set]
        token_list_thowaway.extend([i for i in stopped_tokens if i not in dictionary_words_set])
        texts.append(tokens_noise_removed)

        docs_for_file = tokens_noise_removed
        profession_words = [i for i in tokens_noise_removed if i in professionSet]
        profession_match_count = profession_match_count if not profession_words else profession_match_count+1

        for word in profession_words:
            if word not in profession_count_dict:
                profession_count_dict[word] = 0
            profession_count_dict[word] = profession_count_dict[word]+1
        writeRowToFile(csv_writer,st,named_entity,tokens_noise_removed)
        writeRowToFile(throwaway_csv_writer,st,named_entity,token_list_thowaway)

        file_name_doc_list.append((st,docs_for_file)) #hack X
        profession_file_count_dict[st] = profession_count_dict
    file_count = len(file_names)
    profession_words_writer.writerow([named_entity,file_count,profession_match_count])
    for filename in profession_file_count_dict.keys():
        profession_count_dict = profession_file_count_dict[filename]
        for profession in profession_count_dict.keys():
            document_profession_writer.writerow([named_entity,filename,profession,profession_count_dict[profession]])
        
    print("Done tokenizing and stemming")
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("Rows: ",len(corpus));
    print("columns: ",len(corpus[0]));
    print("Starting LDA")
    
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=no_of_topics, id2word=dictionary, passes=20)

    file.write("LDA MODEL:\n")
    file.write(str(ldamodel.print_topics(num_topics=no_of_topics)) + "\n")
    file.write("\n")
    topic_vs_profession = profession_analyzer(ldamodel,file,named_entity)
    file.write("\n")
    file.write("per doc topic probability\n")
    
    ner_data["topic_map"] = topic_vs_profession
    
    file.write(str(topic_vs_profession)+'\n')
    document_vs_topic = {}

    for file_name_and_docs in file_name_doc_list:
        document_name = file_name_and_docs[0]
        bow = dictionary.doc2bow(file_name_and_docs[1])
        topic_probs = ldamodel.get_document_topics(bow)
        topic = max(topic_probs, key=operator.itemgetter(1))[0]
        document_vs_topic[document_name] = (str(topic), topic_vs_profession[topic])
        final_profession_writer.writerow([named_entity,str(document_name), topic_vs_profession[topic]])
        file.write(str(document_name) + " - Topic: "+ str(topic) + " - "+ topic_vs_profession[topic] +" - " + str(topic_probs) +"\n")
    
    ner_data["document_profession_map"] = document_vs_topic
    return ner_data

def calculate_average_probabilities(x,no_of_topics):
    avg_topic_probabilitis=[]
    for i in range(no_of_topics):
        avg_topic_probabilitis.append([i,0])
    for i in range(len(x)):
        topic_probabilities = x[i]
        for j in range(len(topic_probabilities)):
            avg_topic_probabilitis[j][1] = avg_topic_probabilitis[j][1] + topic_probabilities[j][1]
    for i in range(no_of_topics):
        avg_topic_probabilitis[i][1] = avg_topic_probabilitis[i][1]/len(x) if len(x) > 0 else -1
    return avg_topic_probabilitis

def print_dict_sorted_by_values(dictionary,file,dictionary_name,sort_type):
    file.write("###########Start of Dictionary printing "+dictionary_name+"###########\n")
    sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(sort_type), reverse=True)
    for key, value in sorted_dictionary:
        file.write(str(key) + " - " + str(value) + "\n")
    file.write("###########End of Dictionary printing "+ dictionary_name+"###########\n")

def profession_analyzer(ldamodel,file,named_entity):
    topics_str=ldamodel.show_topics()
    topics = []
    for t in topics_str:
        words_with_prob = t[1].split('+')
        words_prob_dict={}
        for word_prob_str in words_with_prob:
            word_prob_str = word_prob_str.strip()
            word_prob = word_prob_str.split('*"')
            prob = float(word_prob[0])
            word = word_prob[1]
            word= word[:len(word)-1]
            words_prob_dict[word] = prob
        topics.append(words_prob_dict)
    file.write(str(topics))
    file.write("\n")
    count = 0
    topic_vs_profession={}
    topic_id=0
    professions_found_set=set([])
    profession_for_topics = set([])
    for topic in topics:
        file.write("########"+"Topic "+str(count)+"########\n")
        count = count + 1
        profession={}
        professions=[]
        profession_prob_dict={}
        for word in topic:
            professions_for_word = ''
            probability = topic[word]
            probability_for_each = 0
            if word in professionSet:
                professions_for_word = [word]
                probability_for_each = probability
            elif word in wordsProfessiondict.keys():
                professions_for_word = list(wordsProfessiondict[word])
                probability_for_each = probability/len(professions_for_word)
            else:
                professions_for_word=[]
            #print(word,professions_for_word)
            professions_found_set.update(professions_for_word)
            file.write(word + " = " +str(professions_for_word) + "\n")
            #print(probability_for_each)
            profession[word] = professions_for_word
            professions.extend(professions_for_word)
            for matched_profession in professions_for_word:
                if(matched_profession not in profession_prob_dict):
                    profession_prob_dict[matched_profession] = 0
                profession_prob_dict[matched_profession] = profession_prob_dict[matched_profession] + probability_for_each
        print_dict_sorted_by_values(profession_prob_dict,file,"Profession Probability",1)
        sorted_professions = sorted(profession_prob_dict.items(), key=operator.itemgetter(1), reverse=True)
        profession_for_topic = ''
        if len(sorted_professions) > 0:
            profession_for_topic = sorted_professions[0][0]
        topic_vs_profession[topic_id] = profession_for_topic
        profession_for_topics.add(profession_for_topic)
        topic_id = topic_id + 1
        intersection = set(professions)
        for word in topic:
            intersection = intersection & set(profession[word])
        count_profession={}
        for prof in set(professions):
            count = professions.count(prof)
            if count not in count_profession.keys():
                count_profession[count] = []
            count_profession[count].append(prof)
    profession_collection_writer.writerow([named_entity," ".join(list(professions_found_set)),len(professions_found_set)])
    lda_profession_collection_writer.writerow([named_entity," ".join(list(profession_for_topics)),len(profession_for_topics)])
    print(named_entity," ".join(list(profession_for_topics)),len(profession_for_topics))
    file.write("####################\n")    
    return topic_vs_profession

def writeRowToFile(csv_writer,filename,named_entity,document_words):
    words_as_str = ""
    for word in document_words:
        words_as_str = words_as_str + "\""+ word+"\","
    words_as_str = words_as_str[:-1]
    csv_writer.writerow([filename,named_entity,words_as_str])

print("Strat!!!")

#Creating the set of professions
wordsProfessiondict={}
professionSet=set([])
for profession in professionDict:
    for word in professionDict[profession]:
        if(word not in wordsProfessiondict):
            wordsProfessiondict[word] = set([])
        professionList = wordsProfessiondict[word]
        professionList.update({profession})
    professionSet.update({profession})

p_stemmer = PorterStemmer()
named_entity_profession_dict={}


#Creating new folder to store the results for this run
path=path +'LDA_Output_Partial/'
folder_name = 'LDA_Output_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
directory = path+folder_name
os.makedirs(directory)
directory = directory+'/'


csv_file=open(directory+'documents.csv', 'w+')
csv_writer = csv.writer(csv_file, delimiter=',')

throwaway_csv_file=open(directory+'throwaway.csv', 'w+')
throwaway_csv_writer = csv.writer(throwaway_csv_file, delimiter=',')

profession_words_csv=open(directory+'profession_words.csv', 'w+')
profession_words_writer=csv.writer(profession_words_csv,delimiter=',')
profession_words_writer.writerow(["Entity Name","Document Count","Documents Count that Contain Profession"])

file = open(directory+'output_dict.txt', 'w+')
file.write("OUTPUT FOR LDA TOPIC MODELLING\n")

final_profession=open(directory+'final_profession.csv', 'w+')
final_profession_writer = csv.writer(final_profession, delimiter=',')
final_profession_writer.writerow(["Named Entity","Filename","Profession"])

document_profession_csv=open(directory+'profession_stats.csv', 'w+')
document_profession_writer=csv.writer(document_profession_csv,delimiter=',')
document_profession_writer.writerow(["Entity Name","Filename","Profession","Count"])

profession_collection_csv = open(directory+'lds_profession_collection.csv', 'w+')
profession_collection_writer = csv.writer(profession_collection_csv,delimiter=',')
profession_collection_writer.writerow(["Named Entity","Profession Words", "Count"])

lda_profession_collection_csv = open(directory+'lda_model_profession.csv', 'w+')
lda_profession_collection_writer = csv.writer(lda_profession_collection_csv,delimiter=',')
lda_profession_collection_writer.writerow(["Named Entity","Profession Words", "Count"])

final_data = {}

for i in range(ne_filename_df.shape[0]):
        named_entity = ne_filename_df[0][i]
        file.write("******************************************************************************************************\n")
        file.write("named_entity: " + str(named_entity) + "\n")
        print((ne_filename_df[1][i].split()))
        file_names = sorted(set(ne_filename_df[1][i].split()))
        final_data[named_entity] = generate_lda_model(file_names,named_entity,en_stop,tokenizer,window_size_N,no_of_topics,
                           file,dictionary_words_set,professionSet,wordsProfessiondict,csv_writer,throwaway_csv_writer,profession_words_writer)
        file.write("******************************************************************************************************\n")



json_file = open(directory+'lda_data.json','w+')
json_file.write(json.dumps(final_data))
json_file.close()

file.close()
csv_file.close()
throwaway_csv_file.close()
profession_words_csv.close()
document_profession_csv.close()
final_profession.close()
profession_collection_csv.close()
lda_profession_collection_csv.close()
