import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
import jpype
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


names = ["trendyol","hepsiburada","amazon","n11","yemeksepeti"]

WPT = nltk.WordPunctTokenizer()

pre_data = []
pre_data_labels = []

def get_data(name):
    data = pd.read_csv("dataset_"+name+".csv",sep=",")
    labels = pd.read_csv("labels_"+name+".csv",header=None,sep=",")
    itemList = list(data.Comment)
    labelList = list(labels[0])
    pre_data.extend(itemList)
    pre_data_labels.extend(labelList)


def clean_doc(doc):
    stop_words = stopwords.words("turkish")
    filtered_token_list = []
    single_doc_list = []
    for i in range(len(doc)):
        single_doc = re.sub(" \d+", " ", doc[i])
        # Getting rid of the comments with links.
        single_doc = re.sub(r'^https?:\/\/.*[\r\n]*', '', single_doc, flags=re.MULTILINE)
        pattern = r"[{}]".format(",.;")
        single_doc = re.sub(pattern, "", single_doc)
        single_doc = single_doc.lower()
        single_doc = re.sub(r'^br$', ' ', single_doc)
        single_doc = re.sub(r'\s+br\s+', ' ', single_doc)
        single_doc = re.sub(r'\s+[a-z]\s+', ' ', single_doc)
        single_doc = re.sub(r'^b\s+', '', single_doc)
        single_doc = re.sub(r'\s+', ' ', single_doc)
        single_doc = re.sub(r'\s+', ' ', single_doc)
        tokens = WPT.tokenize(single_doc)
        filtered_tokens = [token for token in tokens if not token in stop_words]
        # Stemming with Zemberek.
        try:
            filtered_tokens = [zemberek.kelimeCozumle(token)[0].kok().icerik() for token in filtered_tokens if
                               zemberek.kelimeDenetle(token) == True]
        except Exception as e:
            print(e)
        str_filtered_tokens = [str(i) for i in filtered_tokens]
        filtered_token_list.extend(str_filtered_tokens)
        single_doc = ' '.join(str_filtered_tokens)
        single_doc_list.append(single_doc)
    return filtered_token_list,single_doc_list


if __name__ == '__main__':
    for i in range(len(names)):
        get_data(names[i])

    jpype.startJVM(jpype.getDefaultJVMPath(),
                   "-Djava.class.path=C:/Users/t23463int/PycharmProjects/nlpProject/zemberek-tum-2.0.jar", "-ea")
    tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
    tr = tr()
    Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
    zemberek = Zemberek(tr)

    tokens,single_docs = clean_doc(pre_data)

    df = pd.DataFrame(single_docs, columns=["Comment"])
    df["Sentiment"] = pre_data_labels
    # Clean empty entries.
    df = df[df["Comment"] != ""]

    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000)
    bow = bow_vectorizer.fit_transform(df['Comment'])
    train_df = df[:800]

    train_bow = bow[:800, :]
    test_bow = bow[800:, :]
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_df['Sentiment'], random_state=42,
                                                              test_size=0.3)
    classifier = LogisticRegression(solver="lbfgs", multi_class="ovr")

    classifier.fit(xtrain_bow, ytrain)
    sent_pred = classifier.predict(xvalid_bow)

    print(accuracy_score(yvalid, sent_pred))