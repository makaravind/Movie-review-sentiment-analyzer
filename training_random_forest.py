from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from movie_review_sentiment_analysis_preprocessing import get_vectorizer, clean_text
train_cleaned_data, vectorizer = get_vectorizer()

# Initialize a Random Forest classifier with 100 trees


print 'start - training the model'
# def train():
forest = RandomForestClassifier(n_estimators=100)
train = pd.read_csv('labeledTrainData.tsv', header=0, quoting=3, delimiter='\t')
# train the forest
forest = forest.fit(train_cleaned_data, train['sentiment'])

# return forest


print 'end-training the model'
# forest = train()

# testing the model against test data
# Read the test data
test_data = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

def test():

    clean_test_reviews = []
    num_reviews = test_data['review'].__len__()
    print 'cleaning and parsing the test doc..'

    for i in xrange(0, num_reviews):
        clean_test_reviews.append(clean_text(test_data['review'][i]))

        if (i+1) % 5000 == 0:
            print 'batch of 5000 is done!'

    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    print 'starting predicting..'
    result = forest.predict(test_data_features)

    out = pd.DataFrame(data={'id': test_data['id'], 'sentiment': result})
    out.to_csv('bag_of_words_model.csv', index=False, quoting=3)
    

test()
