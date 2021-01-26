import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import pickle
nltk.download('stopwords')
import jpype
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix

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

# Some text processing and corpus creation.
def norm_doc(single_doc):

    single_doc = re.sub(" \d+", " ", single_doc)
    # Getting rid of the comments with links.
    single_doc = re.sub(r'^https?:\/\/.*[\r\n]*', '', single_doc, flags=re.MULTILINE)
    pattern = r"[{}]".format(",.;")
    single_doc = re.sub(pattern, "", single_doc)
    single_doc = single_doc.lower()
    single_doc = re.sub(r'^br$', ' ', single_doc)
    single_doc = re.sub(r'\s+br\s+',' ',single_doc)
    single_doc = re.sub(r'\s+[a-z]\s+', ' ',single_doc)
    single_doc = re.sub(r'^b\s+', '', single_doc)
    single_doc = re.sub(r'\s+', ' ', single_doc)
    single_doc = re.sub(r'\s+', ' ', single_doc)
    tokens = WPT.tokenize(single_doc)
    filtered_tokens = [token for token in tokens]
    # Stemming with Zemberek.
    try:
        filtered_tokens = [zemberek.kelimeCozumle(token)[0].kok().icerik() for token in filtered_tokens if zemberek.kelimeDenetle(token) == True ]
    except Exception as e:
        print(e)
    str_filtered_tokens = [str(i) for i in filtered_tokens]
    single_doc = ' '.join(str_filtered_tokens)
    return single_doc


def drawFullOpinionChart(df):
    very_pos = df[df["Sentiment"] == 5].iloc[:, 0:1].size
    pos = df[df["Sentiment"] == 4].iloc[:, 0:1].size
    neut = df[df["Sentiment"] == 3].iloc[:, 0:1].size
    neg = df[df["Sentiment"] == 2].iloc[:, 0:1].size
    very_neg = df[df["Sentiment"] == 1].iloc[:, 0:1].size
    plt.bar(["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"], [very_pos, pos, neut, neg, very_neg])
    plt.ylabel("Number")
    plt.title("Sentiment Distribution")
    plt.show()

def drawNarrowedOpinionChart(df):
    df_ternary = df.copy()
    df_ternary['Sentiment'] = df_ternary['Sentiment'].map({5: 4, 4: 4, 3: 3, 2: 2, 1: 2})

    tern_pos = df_ternary[df_ternary["Sentiment"] == 4].iloc[:, 0:1].size
    tern_neut = df_ternary[df_ternary["Sentiment"] == 3].iloc[:, 0:1].size
    tern_neg = df_ternary[df_ternary["Sentiment"] == 2].iloc[:, 0:1].size

    plt.bar(["Positive", "Neutral", "Negative"], [tern_pos, tern_neut, tern_neg])
    plt.ylabel("Number")
    plt.title("Sentiment Distribution")
    plt.show()
    return df_ternary

if __name__ == '__main__':
    for i in range(len(names)):
        get_data(names[i])

    jpype.startJVM(jpype.getDefaultJVMPath(),
                   "-Djava.class.path=C:/Users/emir/PycharmProjects/nlpProject/zemberek-tum-2.0.jar", "-ea")
    tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
    tr = tr()
    Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
    zemberek = Zemberek(tr)

    norm_docs = np.vectorize(norm_doc)
    corpus = norm_docs(pre_data)

    df = pd.DataFrame(corpus, columns=["Comment"])
    df["Sentiment"] = pre_data_labels
    # Clean empty entries.
    df = df[df["Comment"] != ""]

    drawFullOpinionChart(df)
    df_ternary = drawNarrowedOpinionChart(df)

    stop_words = stopwords.words("turkish")

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.2, stop_words=stop_words, use_idf=True)
    X = vectorizer.fit_transform(df["Comment"]).toarray()
    features = vectorizer.get_feature_names()
    tfidf_df = pd.DataFrame(np.round(X, 3), columns=features)

    text_train, text_test, sent_train, sent_test = train_test_split(X, df_ternary["Sentiment"], test_size=0.1,
                                                                    random_state=0)
    classifier = LogisticRegression(solver="lbfgs", multi_class="ovr")

    classifier.fit(text_train, sent_train)
    sent_pred = classifier.predict(text_test)

    print("Logistic Regression")
    print(accuracy_score(sent_test, sent_pred))
    disp = plot_confusion_matrix(classifier, text_test, sent_test)
    disp.ax_.set_title("Logistic Regression")
    plt.show()

    q = cross_val_score(classifier, text_train, sent_train, cv=10)

    plt.plot(np.arange(10), q)
    plt.xlabel("Number of folds")
    plt.ylabel("Accuracy")
    plt.show()

    cross_val_score = sum(q) / len(q)
    print(cross_val_score)

    gnb = GaussianNB()
    gnb.fit(text_train, sent_train)

    sent_gnn_predict = gnb.predict(text_test)
    print("Gaussian Naive Bayes")
    print(accuracy_score(sent_test, sent_gnn_predict))
    disp = plot_confusion_matrix(gnb, text_test, sent_test)
    disp.ax_.set_title("Gaussian Naive Bayes")
    plt.show()

    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(text_train, sent_train)

    sent_knn_predict = knn.predict(text_test)
    print("K-nn")
    print(accuracy_score(sent_test, sent_knn_predict))
    disp = plot_confusion_matrix(knn, text_test, sent_test)
    disp.ax_.set_title("K-nn")
    plt.show()

    dct = tree.DecisionTreeClassifier()
    dct.fit(text_train, sent_train)

    sent_dct_predict = dct.predict(text_test)
    print("Desicion Tree")
    print(accuracy_score(sent_test, sent_dct_predict))
    disp = plot_confusion_matrix(dct, text_test, sent_test)
    disp.ax_.set_title("Desicion Tree")
    plt.show()

    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(text_train, sent_train)

    sent_rf_predict = rf.predict(text_test)
    print("Random Forest")
    print(accuracy_score(sent_test, sent_rf_predict))
    disp = plot_confusion_matrix(rf, text_test, sent_test)
    disp.ax_.set_title("Random Forest")
    plt.show()



    q = cross_val_score(classifier, text_train, sent_train, cv=6)

    plt.plot(np.arange(6), q)
    plt.xlabel("Number of folds")
    plt.ylabel("Accuracy")
    plt.show()

    cross_val_score = sum(q) / len(q)
    print(cross_val_score)



