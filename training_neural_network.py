import tensorflow as tf
import pandas as pd
import numpy as np
from movie_review_sentiment_analysis_preprocessing import get_vectorizer, clean_text, create_logits_batch
train_cleaned_data, vectorizer = get_vectorizer()

vocab = 5000
num_reviews = 25000

x = tf.placeholder('float', [None, vocab])
y = tf.placeholder('float')

n_classes = 2
batch_size = 100

hl1_nodes = 500
hl2_nodes = 500

hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([vocab, hl1_nodes])),
                 'biases': tf.Variable(tf.random_normal([hl1_nodes]))}

hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                  'biases': tf.Variable(tf.random_normal([hl2_nodes]))}

output_layer = {'weights': tf.Variable(tf.random_normal([hl2_nodes, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])
    return output


saver = tf.train.Saver()

def train_neural_network(x):

    prediction = neural_network_model(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdadeltaOptimizer(0.5).minimize(cross_entropy)

    hm_epochs = 5

    train = pd.read_csv('labeledTrainData.tsv', header=0, quoting=3, delimiter='\t')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for each_epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < num_reviews:
                start = i
                end = i + batch_size

                batch_x = np.array(train_cleaned_data[start:end])
                batch_y = np.array(train['sentiment'][start:end])
                batch_y = create_logits_batch(batch_y)

                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print(each_epoch, 'is done out of', hm_epochs, 'loss:', epoch_loss)

        print('training is done!')
        saver.save(sess, 'sentiment_model.ckpt')

# train_neural_network(x)

test_data = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

def get_test_data():

    to_predict = []
    num_reviews = len(test_data)

    for i in xrange(0, num_reviews):
        to_predict.append(clean_text(test_data['review'][i]))

        if i % 5000 == 0:
            'cleaning test data of 5000 done!'

    return to_predict


def test_neural_network():

    prediction = neural_network_model(x)
    to_predict = get_test_data()

    test_data_features = vectorizer.transform(to_predict)
    test_data_features = test_data_features.toarray()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "sentiment_model.ckpt")

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: test_data_features}), 1)))
        print result[:10]

        out = pd.DataFrame(data={'id': test_data['id'], 'sentiment': result})
        out.to_csv('bag_of_words_model_neural_network.csv', index=False, quoting=3)

test_neural_network()