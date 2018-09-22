from flask import Flask, request
from DocToWord import DocToWord
from DocToDoc import DocToDoc
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/findProfession", methods =["GET","POST"])
def findProfession():
    testArticle = str()
    if request.method == "POST":
        postData = request.get_json()
        if postData.get("article") and postData.get("type"):
            testArticle = postData["article"]
            testType = postData["type"]
            docToWordObj = DocToWord(testArticle)

            prediction = "no Profession here"
            if (testType == "withTopics"):
                docToWordObj.trainModelWithTopics()
                predictionList = docToWordObj.getWordWithTopics()

                prediction = str()
                for pre in predictionList:
                    prediction = prediction + " " + pre.item(0)

            elif (testType == "withoutTopics"):
                docToWordObj.trainModelWithoutTopics()
                prediction = docToWordObj.getWordWithoutTopics()

            return str(prediction)
        return "No Profession"

@app.route("/findDocSimilarity", methods =["GET","POST"])
def findDocSimilarity():
    if request.method == "POST":
        postData = request.get_json()
        if postData.get("doc1") and postData.get("doc2"):
            doc1 = postData["doc1"]
            doc2 = postData["doc2"]
            docToDocObj = DocToDoc(doc1, doc2)
            return docToDocObj.getDocToDocSimilarity()

if __name__ == '__main__':
    app.run()
