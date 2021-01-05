import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import jpype
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot

#reference file : https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/

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

# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def process_docs(line_list,vocab):
    lines = []
    for line in line_list:
        tokens = line.split()
        tokens = [w for w in tokens if w in vocab]
        filtered_line = ' '.join(tokens)
        lines.append(filtered_line)
    return lines


def evaluate_mode(Xtrain,ytrain,Xtest,ytest):
    scores = []
    n_repeats = 30
    n_words = Xtest.shape[1]
    for i in range(n_repeats):
        model = Sequential()
        model.add(Dense(50,input_shape=(n_words,),activation='relu'))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])

        model.fit(Xtrain,ytrain,epochs=50,verbose=2)

        loss,acc = model.evaluate(Xtest,ytest,verbose=0)
        scores.append(acc)
        print('%d accuracy: %s'%((i+1),acc))
    return scores

if __name__ == '__main__':
    for i in range(len(names)):
        get_data(names[i])

    jpype.startJVM(jpype.getDefaultJVMPath(),
                   "-Djava.class.path=C:/Users/t23463int/PycharmProjects/nlpProject/zemberek-tum-2.0.jar", "-ea")
    tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
    tr = tr()
    Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
    zemberek = Zemberek(tr)

    vocab = Counter()
    tokens,single_docs = clean_doc(pre_data)
    vocab.update(tokens)
    print(vocab.most_common(50))

    df = pd.DataFrame(single_docs, columns=["Comment"])
    df["Sentiment"] = pre_data_labels
    # Clean empty entries.
    df = df[df["Comment"] != ""]

    # save tokens to a vocabulary file
    # save_list(tokens, 'vocab.txt')
    # vocab_filename = "vocab.txt"
    # vocab = load_doc(vocab_filename)
    # vocab = vocab.split()
    min_occurance = 2
    tokens = [k for k,c in vocab.items() if c >= min_occurance]
    print(len(tokens))
    vocab = set(tokens)

    very_pos_lines = process_docs(list(df[df["Sentiment"] == 5].iloc[:, 0:1].Comment),vocab)
    pos_lines = process_docs(list(df[df["Sentiment"] == 4].iloc[:, 0:1].Comment),vocab)
    neut_lines = process_docs(list(df[df["Sentiment"] == 3].iloc[:, 0:1].Comment),vocab)
    neg_lines = process_docs(list(df[df["Sentiment"] == 2].iloc[:, 0:1].Comment),vocab)
    very_neg_lines = process_docs(list(df[df["Sentiment"] == 1].iloc[:, 0:1].Comment),vocab)
    print(len(very_neg_lines),len(neg_lines),len(neut_lines),len(pos_lines),len(very_pos_lines))

    tokenizer = Tokenizer()

    linesplitters =[int(len(very_pos_lines)*0.8),int(len(pos_lines)*0.8),int(len(neut_lines)*0.8),int(len(neg_lines)*0.8),int(len(very_neg_lines)*0.8)]
    test_docs = very_pos_lines[0:linesplitters[0]]+pos_lines[0:linesplitters[1]]+neut_lines[0:linesplitters[2]]+neg_lines[0:linesplitters[3]]+very_neg_lines[0:linesplitters[4]]
    print(len(test_docs))
    train_docs = very_pos_lines[linesplitters[0]:]+pos_lines[linesplitters[1]:]+neut_lines[linesplitters[2]:]+neg_lines[linesplitters[3]:]+very_neg_lines[linesplitters[4]:]
    print(len(train_docs))

    tokenizer.fit_on_texts(train_docs)
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode='freq')
    print(Xtrain.shape)
    Xtest = tokenizer.texts_to_matrix(test_docs,mode='freq')
    print(Xtest.shape)
    n_words = Xtest.shape[1]
    ytrain = ([5 for _ in range(linesplitters[0])]+
                   [4 for _ in range(linesplitters[1])]+
                   [3 for _ in range(linesplitters[2])]+
                   [2 for _ in range(linesplitters[3])]+
                   [1 for _ in range(linesplitters[4])])
    ytest = ([5 for _ in range((len(very_pos_lines) - linesplitters[0]))]+
                   [4 for _ in range((len(pos_lines) - linesplitters[1]))]+
                   [3 for _ in range((len(neut_lines) - linesplitters[2]))]+
                   [2 for _ in range((len(neg_lines) - linesplitters[3]))]+
                   [1 for _ in range((len(very_neg_lines) - linesplitters[4]))])
    print(ytest)

    modes = ['binary','count','tfidf','freq']
    results = pd.DataFrame()
    for mode in modes:
        tokenizer.fit_on_texts(train_docs)
        Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
        print(Xtrain.shape)
        Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
        print(Xtest.shape)
        results[mode] = evaluate_mode(Xtrain,ytrain,Xtest,ytest)

    print(results.describe())
    results.boxplot()
    pyplot.show()




