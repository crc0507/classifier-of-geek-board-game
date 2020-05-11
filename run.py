from flask import Flask, request, render_template
import re
import nltk
import string
from nltk.corpus import stopwords


app = Flask(__name__)


prior_list = [0.023256179208138498, 0.07930746509217226, 0.28983922523090455, 0.4642061400515805, 0.14339099041720418]
f = open('./static/data.txt', 'r')
vlist = f.read()
vocabulary_list = eval(vlist)
label_count = [42940, 146435, 535168, 857125, 264761]
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


def clean_text(text):
    # Make text lowercase, remove text in square brackets,remove links,remove punctuation
    # remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


def conditional_prob(word, i):
    all_count = label_count[i]
    if word in vocabulary_list:
        return (vocabulary_list[word][i]+1)/(all_count+5)
    if word not in vocabulary_list:
        return 1/(all_count+5)


def classify(s):
    s = clean_text(s)
    s = tokenizer.tokenize(s)
    s = remove_stopwords(s)
    pred_list = []
    for i in range(5):
        pred = prior_list[i]
        for word in s:
            pred *= conditional_prob(word, i)
        pred_list.append(pred)
    max_prob = max(pred_list)
    return pred_list.index(max_prob), pred_list


@app.route('/run_classify', methods=['POST', 'GET'])
def run_classify():
    if request.method == 'POST':
        comment = request.form['comment']
    else:
        comment = request.args.get('comment')
    rating = ' '
    probability = ''
    if comment:
        rating, probability = classify(comment)
    else:
        comment = ''
    return render_template('input.html', rating=rating, probability=probability)


if __name__ == '__main__':
    app.run()
