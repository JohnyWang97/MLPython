# Python机器学习笔记

## 第六章 处理文本

### 6.1 文本清洗方法

``` python
1.string.strip()		#清除文本首尾的空格
eg: text_data=["……"]
  strip_Whitespace = [string.strip() for string in text_data]
  
2.string.replace(",","")  #移除文本中的逗号
eg:remove_periods =[string.replace(",","") for string in strip_Whitespace]

3.创建自定义函数
def capitalizer(string : str) -> str:
  return string.upper()
upper_func = [capitalizer(string) for string in remove_periods]
```

### 6.2 解析并清洗HTML

``` python
# BeautifulSoup库解析HTML
from bs4 import BeautifulSoup
html ="""<div class='full_name'><span style=''>
			Masego Azra"""
soup = BeautifulSoup.(html,'html')
soup.find("div",{"class":"full_name"}).text
```

### 6.3 移除标点符号

``` python
#使用translate方法将标点符号替换成空，构建标点符号的字典，标点字符为key,None为value
import unicodedata
import sys
text_data=['Hi!!!I.Love.This.Song....']
#创建标点字典
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
[string.translate(punctuation)for string in text_data]
#构建字典方法
dict.fromkeys(键值，value(默认为None))
```

### 6.4 文本分词

``` python
#NLTK包的使用,word_tokenize自动按空格将句子切分为单词，sent_tokenize按.将语段切分为句子
from nltk.tokenize import word_tokenize
from nltk.toenize import sent_tokenize
string = "The Science of today is the technology of tomorrow"
word_tokenize(string)
['The','Science','of'...'tomorrow']

```

### 6.5 停止词

``` python
#使用NLTK的stopwords
from nltk.corpus import stopwords
#首次使用需要先下载停止词集
import nltk
nltk.download('stopwords')
tokenized_word =['i','am','going'...'park']
stop_words = stopwords.words('english') #英文停止词
[word for word in tokenized_word if word not in stop_words] #	删除停止词
```

### 6.6 提取词干

``` python
from nltk.stem.porter import PorterStemmer  #波特词干算法，将分词转换为词根形式
tokenized_word =['i','am','going'...'park']
porter = PorterStemmer()
[porter.stem(word) for word in tokenized_word]
```

### 6.7 标注词性

```python
from nltk import pos_tag
from nltk import word_tokenize
text = "Chris loved outdoor running"
text_tagged = pos_tag(word_tokenize(text))
text_tagged
[('Chris','NNP'),('loved','VBD')...]

#将词性特征标签用one-hot编码表示
tweets= ["I am eating a burrito for breakfast",
        	"Political science is an amazing field",
        	"San Francisco is an awesome city"]
#创建列表
tagged_tweets =[]
#为每条推文打上词性标签
for tweet in tweets:
  tweet_tag = nltk.pos_tag(word_tokenize(tweet))
  tagged_tweets.append([tag for word,tag in tweet_tag])

#使用one-hot编码将标签转化为0-1特征
one_hot_multi = MultiLabelBinarizer() #参数为空默认根据标签数量生成特征列表
one_hot_multi.fit_transform(tagged_tweets)

#查看特征名
one_hot_multi.classes_
  
```

## 6.8 将文本编码成词袋

``` python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#创建文本
text_data = np.array(['I love Brazil. Brazil!',
                     	'Sweden is best',
                     	'Germany beats both'])
#创建一个词袋特征矩阵
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
#查看词频统计矩阵
bag_of_words.toarray()
#产看每个特征对应的单词
count.get_feature_names()

#CounterVectorizer参数说明（ngram_range:n元模型，stop_words：停止词类型，vocabulary:限定观察值）
count_2gram = CountVectorizer(ngram_range=(1,2),
                                stop_words='english',
                                vocabulary=['brazil'])

bag = count_2gram.fit_transform(text_data)
#查看特征矩阵
bag.toarray()
#查看一元模型和二元模型
count_2gram.vocabulary_
```

## 6.9 按单词重要性加权

``` python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#创建文本
text_data = np.array(['I love Brazil country. Brazil!',
                     	'Sweden is best',
                     	'Germany beats both'])
#创建TF-IDF特征矩阵
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
#查看矩阵格式
feature_matrix
#查看TF-IDF特征矩阵的稠密矩阵形式
feature_matrix.toarray()
array([[0.        , 0.        , 0.        , 0.81649658, 0.40824829,
        0.        , 0.        , 0.40824829, 0.        ],
       [0.        , 0.57735027, 0.        , 0.        , 0.        ,
        0.        , 0.57735027, 0.        , 0.57735027],
       [0.57735027, 0.        , 0.57735027, 0.        , 0.        ,
        0.57735027, 0.        , 0.        , 0.        ]])
#查看特征名字
tfidf.vocabulary_
{'love': 7,
 'brazil': 3,
 'country': 4,
 'sweden': 8,
 'is': 6,
 'best': 1,
 'germany': 5,
 'beats': 0,
 'both': 2}
```

tf(term frequency)词频:某个单词在文档中出现次数
df(document frequency)文档频率：某个单词在多少个文档中出现过

``` math
TF- IDF输出值：tf-idf(t, d) = tf(t, d) * idf(t)
tf(t,d):单词t在文档d中出现次数
```

$$
idf(t)=log \frac{1+n_d}{1+df(d,t)}+1
$$

$$
n_d表示文档的数量，df(d,t)表示单词t的频率（单词在多少文档中出现过），上述公式log底数为e
$$

一般来说会对结果进行L2范数的正则化处理，得到的结果越大，则该单词对文档就越重要

