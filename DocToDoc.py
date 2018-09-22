import gensim
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from collections import Counter
from DocToWord import DocToWord

class DocToDoc:
    def __init__(self, doc1, doc2):
        self.doc1 = doc1
        self.doc2 = doc2

    def getDocToDocSimilarity(self):
        doc1ToWordObj = DocToWord(self.doc1)
        doc2ToWordObj = DocToWord(self.doc2)

        doc1ToWordObj.trainModelWithTopics()
        doc2ToWordObj.trainModelWithTopics()

        doc1WithoutTopicIntent = doc1ToWordObj.trainModelWithoutTopics()
        doc2WithoutTopicIntent = doc2ToWordObj.trainModelWithoutTopics()

        #doc1PredictionsList = doc1ToWordObj.getWordWithTopics()
        doc1PredictionsList = list()
        for pre in doc1ToWordObj.getWordWithTopics():
            doc1PredictionsList.append(pre.item(0))
        doc1MostCommonIntentList = Counter(doc1PredictionsList).most_common(3)

        doc2PredictionsList = list()
        for pre in doc2ToWordObj.getWordWithTopics():
            doc2PredictionsList.append(pre.item(0))
        doc2MostCommonIntentList = Counter(doc2PredictionsList).most_common(3)


        docMostCommonIntentList = [doc1MostCommonIntent for doc1MostCommonIntent in doc1MostCommonIntentList if doc1MostCommonIntent[0] in [doc2MostCommonIntent[0] for doc2MostCommonIntent in doc2MostCommonIntentList]]

        if doc1WithoutTopicIntent == doc1WithoutTopicIntent and len(docMostCommonIntentList) > 0:
            return "highly Similar"

        elif len(docMostCommonIntentList) > 0:
            return "Similar"
        else:
            return "Not Similar"

#DocToDocObj = DocToDoc("Skin, Health, goal, banker, life, banking, doctor, investment", "Skin, Health, goal, banker, life, banking, doctor, investment")
#DocToDocObj.getDocToDocSimilarity()



