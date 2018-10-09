from models import *
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib

#import data
df = pd.read_csv('sample_tweets.csv')

def tokenize(tweet):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr.tokenize(tweet)

def nlp(train_data, test_data):
    bow = TfidfVectorizer(tokenizer=tokenize, max_features=10000)
    train_text = bow.fit_transform(train_data['text'])
    test_text = bow.transform(test_data['text'])
    dftrain = pd.DataFrame(train_text.toarray())
    dftrain.columns = bow.get_feature_names()
    dftest = pd.DataFrame(test_text.toarray())
    dftest.columns = bow.get_feature_names()
    sub_train = train_data.drop(['text'], axis=1).reset_index()
    sub_test = test_data.drop(['text'], axis=1).reset_index()
    x_train = pd.concat([dftrain, sub_train], axis=1)
    x_test = pd.concat([dftest, sub_test], axis=1)
    return x_train, x_test

def logistic_regression(x_train, y_train):
    logreg = LogisticRegression()
    model_log = logreg.fit(x_train, y_train)
    return model_log

def random_forest(x_train, y_train):
    forest = RandomForestClassifier(n_estimators=100, max_depth= 5)
    forest.fit(x_train, y_train)
    return forest

def svm(x_train, y_train):
    clf = SVC(kernel='linear')
    svm_model = clf.fit(x_train, y_train)
    return svm_model

def metrics(x_test, y_test, model):
    y_hat = model.predict(x_test)
    precision = precision_score(y_test, y_hat)
    accuracy = accuracy_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    return pd.DataFrame({'precision': precision, 'accuracy':accuracy, 'recall':recall})
#
# tweets_df = pd.read_sql_query('SELECT * FROM tweets', con=engine)
# users_df = pd.read_sql_query('SELECT id,description FROM users', con=engine)
# df = pd.merge(tweets_df, users_df, left_on = 'user_id', right_on = 'id')

x_vals = df[['text','created_hour', 'compound', 'polarity']]
labels = df['city']

train_data, test_data, y_train, y_test = train_test_split(data, labels)
x_train, x_test = nlp(train_data, test_data)
x_train.to_csv('x_train.csv')
x_test.to_csv('x_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

log = logistic_regression(x_train, y_train)
forest = random_forest(x_train, y_train)
svm = svm(x_train, y_train)
d1 = metrics(x_test, y_test, log)
d2 = metrics(x_test, y_test, forest)
d3 = metrics(x_test, y_test, svm)
metric_df = pd.concat([d1,d2,d3])
#save data to disk
metric_df.to_csv('model_metrics.csv')
joblib.dump(log, 'lr.pkl')
joblib.dump(forest, 'rf.pkl')
joblib.dump(svm, 'svm.pkl')
