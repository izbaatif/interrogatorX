from flask import Flask, render_template, request
import pickle
import numpy as np


#Load bag of words pickle file
with open('pickleFiles/bagOfWords.pickle', 'rb') as f:
    lr_bw = pickle.load(f)
    mapping = pickle.load(f)

# Load the TF-IDF vectorizer and the trained model from the pickle files
with open('pickleFiles/tfidf.pickle', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    lr_tfidf = pickle.load(f)

with open('pickleFiles/word2vec.pickle', 'rb') as f:
    lr_w2v = pickle.load(f)
    word2vec = pickle.load(f)


def preprocess_text_bag(headline, body):
    # Combine headline and body into a single string
    text = f"{headline} {body}"
    # Gets the mapping dictionary from data
    count_vector = np.zeros(len(mapping))
    #makes the bag of word from that mapping dictionary
    for word in text.split():
        if word in mapping:
            count_vector[mapping[word]] += 1
    #reshapes data
    new_text_vector=count_vector.reshape(1, -1)
    # Make a prediction using the trained classifier
    predicted_label = lr_bw.predict(new_text_vector)

    # Make predictions of confidence
    predicted_confidence = lr_bw.predict_proba(new_text_vector)

    # Get the confidence 
    confidence = round(predicted_confidence.max()*100 ,2 )

    #return both
    return [predicted_label[0],confidence]


def preprocess_text_tfidf(headline, body):
    # Preprocess the text data
    text = f"{headline} {body}"
    tfidf_text = tfidf_vectorizer.transform([text])

    # Predict the label using the trained model
    predicted_label = lr_tfidf.predict(tfidf_text)

    # Make predictions of confidence
    predicted_confidence = lr_tfidf.predict_proba(tfidf_text)

    # Get the confidence 
    confidence = round(predicted_confidence.max()*100 ,2 )

    #return both
    return [predicted_label[0],confidence]


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if nwords > 0:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def preprocess_text_word2vec(headline, body):
    # Combine headline and body into a single string
    text = f"{headline} {body}"
    #tokenize text
    tokenized_text = text.split()
    features = average_word_vectors(tokenized_text, word2vec, word2vec.wv.key_to_index, 1000)

    # Predict using LogisticRegression model
    lr_w2v_pred = lr_w2v.predict([features])

    # Make predictions of confidence
    predicted_confidence = lr_w2v.predict_proba([features])

    # Get the confidence 
    confidence = round(predicted_confidence.max()*100 ,2 )

    #return both
    return [lr_w2v_pred[0],confidence]


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['titleText']
        body = request.form['bodyText']
        method = request.form['method']

        if method == 'bagOfWords':
            predicted_label = preprocess_text_bag(title, body)[0]
            confidence = preprocess_text_bag(title, body)[1]
        elif method == 'tfidf':
            predicted_label = preprocess_text_tfidf(title, body)[0]
            confidence = preprocess_text_tfidf(title, body)[1]
        else:
            predicted_label = preprocess_text_word2vec(title, body)[0]
            confidence = preprocess_text_word2vec(title, body)[1]
            
        if predicted_label == 'real':
            return real(confidence)
        else:
            return fake(confidence)

    else:
        return render_template('index.html')

@app.route('/real', methods=['GET'])
def real(confidence):
    return render_template('real.html', conf=confidence)

@app.route('/fake', methods=['GET'])
def fake(confidence):
    return render_template('fake.html' , conf=confidence)

if __name__ == '__main__':
    app.run(host='localhost', port=3000)
