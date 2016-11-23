from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


train = pd.read_csv('labeledTrainData.tsv', header=0, quoting=3, delimiter='\t')

print 'checking the train cols:', train.columns.values

# cleaning the reviews for ML
# preprocessing
lemmatizer = WordNetLemmatizer()

def create_lexicon(words):

    lexicon = [lemmatizer.lemmatize(word) for word in words]
    return lexicon

def clean_text(review_text):

    # 1 removing HTML tags
    soup = BeautifulSoup(review_text, 'html.parser')
    no_html = soup.getText()

    # 2 removing puntuations
    no_punc = re.sub("[^a-zA-Z]", " ", no_html)

    #sliting sentence into words
    words = TextBlob(no_punc).words

    # 3 remove stop words like the, is, an..
    stop_words = set(stopwords.words('english'))

    meaningful_words = [w for w in words if not w in stop_words]

    #4 lemmatising
    # print set(meaningful_words) - set(create_lexicon(meaningful_words))
    return ' '.join(create_lexicon(meaningful_words))


def create_clean_train():
    '''
    :return: pickle the trained set
    '''
    cleaned_trained_reviews = []
    num_reviews = train['review'].size
    for i in xrange(0, num_reviews):
        cleaned_trained_reviews.append(clean_text(train['review'][i]))
        if ((i+1) % 1000) == 0:
            print 'batch of 1000 is done!'

    with open('clean_train_data.pickle', 'wb') as f:
        pickle.dump(cleaned_trained_reviews, f)

    print 'pickling is done!'

# create_clean_train()

def create_logits_batch(batch):

    # 1 --> [1, 0]
    # 0 --> [0, 1]
    cleaned_batch = []
    for i in xrange(0, len(batch)):

        if batch[i] == 1 :
            cleaned_batch.append([1, 0])
        else:
            cleaned_batch.append([0, 1])
    return cleaned_batch


def get_cleaned_data_pickle():
    with open('clean_train_data.pickle', 'rb') as f:
        data = pickle.load(f)
        return data


print 'creating bag of words..'

def get_vectorizer():
    vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None,
                                stop_words=None, max_features=5000)

    train_data_features = vectorizer.fit_transform(get_cleaned_data_pickle())
    train_data_features = train_data_features.toarray()

    print 'preprocessing is done!'
    return train_data_features, vectorizer

# print train_data_features.shape
# vocab = vectorizer.get_feature_names()







