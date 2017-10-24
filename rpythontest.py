def printthis():
    return("hello world")

def addone(lst):
    for i, n in enumerate(lst):
        lst[i] = n+1
    return lst

def checklist(lst):
    lst = list(lst)
    return lst

def getfirstitem(lst):
    lst = list(lst)
    return lst[0]

def splitstring(st):
    return st.split(' ')

def encodelabels(labels):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return labels

def classify(sents, labels):

    sents = list(sents)
    labels = list(labels)

    import codecs, re
    import numpy as np
    # preprocessing
    # from nltk.stem.snowball import SnowballStemmer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.preprocessing import LabelEncoder
    # classifiers
    from sklearn.linear_model import LogisticRegression
    # from sklearn.svm import LinearSVC
    # model pipeline stuff
    from sklearn.pipeline import Pipeline


    def tokenize(sentence):
        # stemmer = SnowballStemmer("english")
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        wordlist = sentence.strip('\n').split(' ')
        # result = [stemmer.stem(word) for word in wordlist]
        result = wordlist
        return result

    vectorizer = CountVectorizer(tokenizer=tokenize)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    
    pipeline1 = Pipeline([
        ('vect', vectorizer),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ])

    pipeline1.fit(sents, labels)

    preds = pipeline1.predict(sents)

    preds = [encoder.inverse_transform(i) for i in preds]

    return np.asarray(preds)

# test

test_string = "hello world"
test_list = [1,3,3,7]

'''
sents = ["Do you like", "Green eggs and ham", "I do not like them",
         "Sam I am", "I do not like", "Green eggs and ham",
         "It's getting dark too dark to see",
         "Feels like I'm knockin' on heaven's door",
         "Knock knock knockin on heaven's door"]

labels = ["Seuss", "Seuss", "Seuss", "Seuss", "Seuss", "Seuss", "Dylan", "Dylan", "Dylan"]

test = classify(sents, labels)
print(test)
'''
