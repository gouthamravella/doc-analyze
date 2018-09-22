from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import io
import sys
from gensim import corpora
import re
import os
import pandas as pd
import pickle

import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


class DocToWord:
    def __init__(self, userDoc):
        self.userDoc = userDoc


    def trainModelWithTopics(self, forceTrain=False):
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        if forceTrain or not os.path.isfile(os.path.join(SITE_ROOT, 'data', 'docToWord_WithoutTopics_Model.sav')):
            datauRL = os.path.join(SITE_ROOT, 'data', 'finalTopicsWithIntentData.csv')
            dataSet = pd.read_csv(datauRL)

            pipeline = Pipeline([
                ('bow', CountVectorizer(analyzer=self.text_process)),
                ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
                ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
            ])
            pipeline.fit(dataSet["Topics"], dataSet["Intent"])

            filename = os.path.join(SITE_ROOT, 'data', 'docToWord_WithTopics_Model.sav')
            pickle.dump(pipeline, open(filename, 'wb'))


    def trainModelWithoutTopics(self, forceTrain=False):
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))

        if forceTrain or not os.path.isfile(os.path.join(SITE_ROOT, 'data', 'docToWord_WithoutTopics_Model.sav')):

            datauRL = os.path.join(SITE_ROOT, 'data', 'withoutTopics.csv')
            dataSet = pd.read_csv(datauRL)

            pipeline = Pipeline([
                ('bow', CountVectorizer(analyzer=self.text_process)),
                ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
                ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
            ])
            pipeline.fit(dataSet["Article"], dataSet["Intent"])

            filename = os.path.join(SITE_ROOT, 'data', 'docToWord_WithoutTopics_Model.sav')
            pickle.dump(pipeline, open(filename, 'wb'))

    def getWordWithoutTopics(self):
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        filename = os.path.join(SITE_ROOT, 'data', 'docToWord_WithoutTopics_Model.sav')
        loaded_model = pickle.load(open(filename, 'rb'))

        prediction = loaded_model.predict([self.userDoc])
        return str(prediction)

    def getWordWithTopics(self):
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        filename = os.path.join(SITE_ROOT, 'data', 'docToWord_WithTopics_Model.sav')
        loaded_model = pickle.load(open(filename, 'rb'))
        ftList = self.getTopicsList()
        predictionList = list()
        for ft in ftList:
            prediction = loaded_model.predict(ft)
            predictionList.append(prediction)
        return predictionList

        #Faltoo Extra Block to consume Memory :( - BTW wrote for testing you can remove it by creating string above.
        #prediction = str()
        #for pre in predictionList:
            #prediction = prediction + " " + pre.item(0)
        #return prediction

        #result = loaded_model.score(X_test, Y_test)

        #print("hello")
    def getWordWithoutTopics(self):
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        filename = os.path.join(SITE_ROOT, 'data', 'docToWord_WithoutTopics_Model.sav')
        loaded_model = pickle.load(open(filename, 'rb'))

        prediction = loaded_model.predict([self.userDoc])
        return str(prediction)

    def clean(self, doc):
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        lemma = WordNetLemmatizer()

        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

        return normalized

    def text_process(self, mess):
        nopunc = [char for char in mess if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


    def getTopicsList(self):
        doc_complete = re.sub("\\d+", "", self.userDoc)

        doc_clean = [self.clean(self.userDoc).split()]
        finalWords = list()
        junkWords = list()
        i = 1
        print(type(doc_clean))
        for words in doc_clean:
            for word in words:
                word.replace('â•', '')
                if '/' in word or 'â£' in word or 'â’' in word or 'â•' in word or '\x94' in word or '\x95' in word or '\x92' in word or '\x97' in word or '\x96' in word or '\'' in word:
                    junkWords.append(word)
                    print(i)
                    i = i + 1
                else:
                    finalWords.append(word)
        dictionary = corpora.Dictionary([finalWords])
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in [finalWords]]

        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix, num_topics=10, id2word=dictionary, passes=50)

        ftList = list()
        for topics in ldamodel.print_topics(num_topics=10, num_words=5):
            for topic in topics:
                if isinstance(topic, str):
                    topicWords = topic.split("+")
                    ft = str()
                    for topicWord in topicWords:
                        finalTopics = topicWord.split("*")
                        # print(type(finalTopics[1]))
                        ft = ft + " " + finalTopics[1].replace('"', '')
                        # print(ft)
                        # nft = nft + ft
                        # if len(ft) != 1:
                        # ftList.append(ft)
                    ftList.append([ft])  # Final Test File
        return ftList