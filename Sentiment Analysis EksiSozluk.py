
# coding: utf-8

# # Importing Libraries

# In[99]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import string
import re
import nltk
import pickle
from numpy import genfromtxt
from nltk.corpus import stopwords
nltk.download('stopwords')
from unidecode import unidecode
import jpype


# # Web Scraping

# In[100]:


headers = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36"}
url_list = ["https://eksisozluk.com/mustafa-kemal-ataturk--34712?p=","https://eksisozluk.com/enes-batur--2825586?p=","http://eksisozluk.com/filmi-varken-gidip-500-sayfa-roman-okuyan-tip--5830909?p="]


# In[101]:


# Retrieving the data from URL's.
pre_data = []
def get_data(url_link):
    for x in range(1,22):
        page = requests.get(url_link+str(x),headers=headers)
        soup = BeautifulSoup(page.content,"html.parser")
        for entry in soup.find_all("div",{"class" : "content"}):
            pre_data.append(entry.get_text())
for url in url_list:
    get_data(url)


# # Moving unprocessed data to df

# In[102]:


pre_data = np.array(pre_data)
# Sentiment data is taken from csv.
y = pd.read_csv("labels63.csv",header=None,sep=";")
df = pd.DataFrame(pre_data,columns=["Comment"])
df["Sentiment"] = y
df.head(10)


# # Data preprocessing with Zemberek

# In[103]:


# Setting up JVM for Zemberek.
jpype.startJVM(jpype.getDefaultJVMPath(),"-Djava.class.path=/Users/fikretefe/Downloads/zemberek-tum-2.0.jar","-ea")
tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
tr = tr()
Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
zemberek = Zemberek(tr)


# In[104]:


WPT = nltk.WordPunctTokenizer()
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
    filtered_tokens = [zemberek.kelimeCozumle(token)[0].kok().icerik() for token in filtered_tokens if zemberek.kelimeDenetle(token) == True ]
    single_doc = ' '.join(filtered_tokens)
    return single_doc
norm_docs = np.vectorize(norm_doc)
corpus = norm_docs(pre_data)


# # Moving processed data to df

# In[105]:


df = pd.DataFrame(corpus,columns=["Comment"])
df["Sentiment"] = y
# Clean empty entries.
df = df[df["Comment"]!= ""]
df.head(10)


# # Exploring the sentiment distribution

# In[106]:


very_pos = df[df["Sentiment"] == 5].iloc[:,0:1].size
pos = df[df["Sentiment"] == 4].iloc[:,0:1].size
neut = df[df["Sentiment"] == 3].iloc[:,0:1].size
neg = df[df["Sentiment"] == 2].iloc[:,0:1].size
very_neg = df[df["Sentiment"] == 1].iloc[:,0:1].size
plt.bar(["Very Positive","Positive","Neutral","Negative","Very Negative"],[very_pos,pos,neut,neg,very_neg])
plt.ylabel("Number")
plt.title("Sentiment Distribution")
plt.show()


# # Ternary Classification 

# In[109]:


# This is for trial with 3 class classification.

df_ternary = df.copy()
df_ternary['Sentiment'] = df_ternary['Sentiment'].map({5: 4, 4: 4,3: 3, 2: 2, 1: 2})

tern_pos = df_ternary[df_ternary["Sentiment"] == 4].iloc[:,0:1].size
tern_neut = df_ternary[df_ternary["Sentiment"] == 3].iloc[:,0:1].size
tern_neg = df_ternary[df_ternary["Sentiment"] == 2].iloc[:,0:1].size

plt.bar(["Positive","Neutral","Negative"],[tern_pos,tern_neut,tern_neg])
plt.ylabel("Number")
plt.title("Sentiment Distribution")
plt.show()
df_ternary.head(10),df.head(10)


# In[110]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop_words = stopwords.words("turkish")

vectorizer = TfidfVectorizer(min_df=3,max_df=0.6,stop_words=stop_words,use_idf=True)
X = vectorizer.fit_transform(df["Comment"]).toarray()
features = vectorizer.get_feature_names()
tfidf_df = pd.DataFrame(np.round(X,3),columns=features)


# In[140]:


from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X,df_ternary["Sentiment"],test_size = 0.20,random_state=30)
text_train.shape


# In[141]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver="lbfgs",multi_class="ovr")

classifier.fit(text_train,sent_train)
sent_pred = classifier.predict(text_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(sent_test,sent_pred))


# In[123]:


from sklearn.model_selection import cross_val_score
q = cross_val_score(classifier,text_train,sent_train,cv = 10)


plt.plot(np.arange(10),q)
plt.xlabel("Number of folds")
plt.ylabel("Accuracy")
plt.show()

cross_val_score = sum(q) / len(q)
print(cross_val_score)


# # Implementing TF-IDF Model

# In[139]:


vectorizer = TfidfVectorizer(min_df=3,max_df=0.6,stop_words=stop_words,use_idf=True)
X = vectorizer.fit_transform(df["Comment"]).toarray()
features = vectorizer.get_feature_names()
tfidf_df = pd.DataFrame(np.round(X,3),columns=features)
tfidf_df.head(8)


# # Train test split

# In[125]:


text_train, text_test, sent_train, sent_test = train_test_split(X,df["Sentiment"].values.ravel(),test_size = 0.2,random_state=33)


# # Logistic Regression

# In[137]:


classifier = LogisticRegression(solver="lbfgs",multi_class="ovr")
classifier.fit(text_train,sent_train)

sent_pred = classifier.predict(text_test)
print(accuracy_score(sent_test,sent_pred))


# # Gaussian Naive Bayes

# In[127]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(text_train,sent_train)

sent_gnn_predict = gnb.predict(text_test)
print(accuracy_score(sent_test,sent_gnn_predict))


# # KNN

# In[128]:


from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(text_train,sent_train)

sent_knn_predict = knn.predict(text_test)
print(accuracy_score(sent_test,sent_knn_predict))


# # Decision Tree

# In[129]:


from sklearn import tree
dct = tree.DecisionTreeClassifier()
dct.fit(text_train,sent_train)

sent_dct_predict = dct.predict(text_test)
print(accuracy_score(sent_test,sent_dct_predict))


# # Random Forest

# In[130]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200)
rf.fit(text_train,sent_train)

sent_rf_predict = rf.predict(text_test)
print(accuracy_score(sent_test,sent_rf_predict))


# # Cross Validation

# In[134]:


from sklearn.model_selection import cross_val_score
q = cross_val_score(classifier,text_train,sent_train,cv = 6)


plt.plot(np.arange(6),q)
plt.xlabel("Number of folds")
plt.ylabel("Accuracy")
plt.show()


cross_val_score = sum(q) / len(q)
print(cross_val_score)


# In[305]:


df.to_csv('data.csv')

