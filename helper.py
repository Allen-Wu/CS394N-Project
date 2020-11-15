import numpy
import string
import nltk
import csv
import re
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 

def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


def normalize_text(text):
    '''
    Perform text normalization
    '''
    text = text.lower()

    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    table = str.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)

    text = text.replace('0',' zero ')
    text = text.replace('1',' one ')
    text = text.replace('2',' two ')
    text = text.replace('3',' three ')
    text = text.replace('4',' four ')
    text = text.replace('5',' five ')
    text = text.replace('6',' six ')
    text = text.replace('7',' seven ')
    text = text.replace('8',' eight ')
    text = text.replace('9',' nine ')
    # https://stackoverflow.com/questions/1546226/simple-way-to-remove-multiple-spaces-in-a-string
    text = ' '.join(text.split())

    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    stemmer= PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    result = [stemmer.stem(lemmatizer.lemmatize(i)) for i in tokens if not i in stop_words]
    resString = ""
    for w in result:
        resString += w + " "
    return resString

with open('helper.csv', 'w', newline = '') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['type', 'posts'])
    with open('mbti_1.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        firstLine = True
        for row in readCSV:
            if firstLine:
                firstLine = False
                continue
            text = row[1]
            sentences = text.split("|||")
            paragraph = ""
            for sentence in sentences:
                sentence = remove_urls(sentence)
                paragraph += normalize_text(sentence) + " "
            filewriter.writerow([row[0], paragraph])
            break
