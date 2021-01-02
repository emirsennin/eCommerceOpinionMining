import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests


headers = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36"}
url_list = ["https://eksisozluk.com/trendyol--2361186","https://eksisozluk.com/hepsiburada-com--93181","https://eksisozluk.com/amazon-com-tr--1434702","https://eksisozluk.com/n11-com--3698262","https://eksisozluk.com/yemeksepeti-com--199510"]
page = [680,1396,725,340,626]
names = ["trendyol","hepsiburada","amazon","n11","yemeksepeti"]

pre_data = []

def get_data(url_link,page,name):
    pre_data = []
    for x in range(page-20,page):
        print(url_link+"?p="+str(x))
        page = requests.get(url_link+"?p="+str(x),headers=headers,verify = False)
        soup = BeautifulSoup(page.content,"html.parser")
        for entry in soup.find_all("div",{"class" : "content"}):
            pre_data.append(entry.get_text())
    pre_data = np.array(pre_data)
    df = pd.DataFrame(pre_data, columns= ["Comment"])
    df.to_csv("dataset_"+str(name)+".csv")




if __name__ == '__main__':
    for i in range(len(url_list)):
        get_data(url_list[i], page[i], names[i])

