import dash_core_components as dcc
import dash_html_components as html
from twitter_package import *
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import json
import dash
from dash.dependencies import Input, Output, State
from sklearn.externals import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split, KFold
from twitter_package.config import *
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff

#import data, models, and metrics
log = joblib.load('log.pickle')
forest = joblib.load('forest.pickle')
# svm = joblib.load('svm.pkl')
# metric_df = pd.read_csv('model_metrics.csv')
df = pd.read_csv('sample_tweets.csv', low_memory=False)
map_df = pd.read_csv('city_mapping.csv')

# x_train = pd.read_csv('x_train.csv', low_memory=False)
# x_test = pd.read_csv('x_test.csv', low_memory=False)
# y_train = pd.read_csv('y_train.csv', low_memory=False)
# y_test = pd.read_csv('y_test.csv', low_memory=False)

# image_filename = 'my-image.png' # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())

def generate_map():
    data = [go.Scattermapbox(lat=df['centroid_lat'],
                            lon=df['centroid_long'],
                            mode='markers',
                            marker=dict(size=6),
                            text=df['text'])
                            ]
    layout = go.Layout(width=1300, height=900,
                    hovermode='closest',
                    mapbox=dict(
                    accesstoken=mapbox_access_token,
                    bearing=0,
                    center=dict(lat=39.8283,lon=-99.5795),
                    pitch=0,
                    zoom=3.5),)
    return {'data': data, 'layout': layout}

app.layout = html.Div(style={'fontFamily': 'Sans-Serif'}, children=[
    html.H1('Twitter Geolocation', style={'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'Sans-Serif'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Map Overview', children=[
            html.Div([
                # html.Div(dcc.Input(id='input-box', type='text')),
                html.Button('Submit', id='button'),
                html.Div(id='output-container-button', children='Enter a value and press submit'),
                dcc.Graph(id='map',figure=generate_map())
                        ])
                        ]),
        dcc.Tab(label='Exploratory Analysis', children=[
            html.Div([
                dcc.Dropdown(
                id='select-xvar',
                options=[{'label': 'Tweet Counts', 'value': 'tweets'},
                # {'label': 'Peak Active Hours', 'value': 'hour'},
                {'label': 'Mean Compound', 'value': 'sentiment'},
                {'label': 'Mean Polarity', 'value': 'polarity'},
        #         # {'label': 'Compound vs. Polarity', 'value': 'com_pol'}
                        ]),
                html.Div(id='plot-container')
                        ])
                        ]),
        dcc.Tab(label='Models Overview', children=[
        #     html.Div(
        #         html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image))]),
        #         html.Div([dcc.Graph(id='table', figure={generate_table()})
        #                 ]),
                dcc.Dropdown(
                id='select-model',
                options=[{'label': 'Logistic Regression', 'value': 'log'},
                {'label': 'Random Forest Classifier', 'value': 'forest'},
        #         {'label': 'Support Vector Machine', 'value': 'svm'}
                        ],
                placeholder="Select a Model", value ='Model'),
                html.Div(id='cm-container')
                        ]),
                        ])
                        ])

#
# def generate_table():
#     trace = go.Table(header=dict(values=list(metric_df.columns),
#                     fill = dict(color='#C2D4FF'),
#                     align = ['left'] * 5),
#                     cells=dict(values=[metric_df.Accuracy, metric_df.Precision, metric_df.Recall],
#                     fill = dict(color='#F5F8FF'),
#                     align = ['left'] * 5))
#     layout = dict(width=500, height=300)
#     return {'data': [trace], 'layout':layout}

def tokenize(tweet):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr.tokenize(tweet)

def nlp(train_data, test_data):
    bow = TfidfVectorizer(tokenizer=tokenize, max_features=2000)
    bow1 = TfidfVectorizer(tokenizer=tokenize, max_features = 800)
    train_text = bow.fit_transform(train_data['text'])
    test_text = bow.transform(test_data['text'])
    train_loc = bow1.fit_transform(train_data['location'])
    test_loc = bow1.fit_transform(test_data['location'])
    dftrain = pd.DataFrame(train_text.toarray())
    dftrain.columns = bow.get_feature_names()
    dftest = pd.DataFrame(test_text.toarray())
    dftest.columns = bow.get_feature_names()
    dftrain_loc = pd.DataFrame(train_loc.toarray())
    dftrain_loc.columns = bow1.get_feature_names()
    dftest_loc = pd.DataFrame(test_loc.toarray())
    dftest_loc.columns = bow1.get_feature_names()
    sub_train = train_data.drop(columns = ['text', 'location'], axis=1).reset_index()
    sub_test = test_data.drop(columns = ['text', 'location'], axis=1).reset_index()
    x_train1 = pd.concat([dftrain, sub_train], axis=1)
    x_test1 = pd.concat([dftest, sub_test], axis=1)
    x_train = pd.concat([x_train1, dftrain_loc], axis=1)
    x_test = pd.concat([x_test1, dftest_loc], axis=1)
    return x_train, x_test

df['location'] = df['location'].astype('str')
x_vals = df[['text', 'location','created_hour']]
labels = df['region_id']

train_data, test_data, y_train, y_test = train_test_split(x_vals, labels)
x_train, x_test = nlp(train_data, test_data)

def region_label(region_id):
    if region_id=='1.0':
        return 'Northeast'
    elif region_id=='2.0':
        return 'Midwest'
    elif region_id=='3.0':
        return 'South'
    elif region_id=='4.0':
        return 'West'

@app.callback(Output('output-container-button', 'children'),
              [Input('button','n_clicks')],
              # [State('input1', 'value')]
              )
def make_prediction(n_clicks):
    rand_tweet = df.sample(1)
    return 'You have generated a random tweet: ' + rand_tweet['text']
    # x_vals = rand_tweet[['text','created_hour', 'compound_score', 'polarity']]
    # y_actual = rand_tweet['region_id']
    #
    # city_value = log.predict(x_vals)
    # probability = log.predict_proba(text)
    # region = region_label(region_id)
    # return """ The submitted tweet was '{}'. The model predicts the region to be {}, with a
    #             probability of {}""".format(text, region, probability)
    # return 'The button has been clicked {} times'.format(n_clicks)


def generate_x_y(data_series):
    x = []
    y = []
    for i, v in data_series.items():
        x.append(i)
        y.append(v)
    return x,y

def generate_tweet_plot():
    city_counts = df['region_name'].value_counts()
    x,y = generate_x_y(city_counts)
    trace = [{'x': x, 'y': y, 'type': 'bar'}]
    layout = go.Layout(
    xaxis={'title': 'Cities'},
    yaxis={'title': 'Tweets'})
    return dcc.Graph(id='tweet-graph', figure={'data': trace, 'layout':layout})
#
# def generate_hour_plot():
#     pass
#
def generate_sentiment_plot():
    city_compound_means = df.groupby('region_name')['compound_score'].mean()
    x,y = generate_x_y(city_compound_means)
    trace = [{'x': x, 'y': y, 'type': 'bar'}]
    layout = go.Layout(xaxis={'title': 'Cities'}, yaxis={'title': 'Mean Sentiment'})
    return dcc.Graph(id ='mean-sentiment', figure = {'data': trace, 'layout': layout})

def generate_polarity_plot():
    city_pol_means = df.groupby('region_name')['polarity'].mean()
    x,y = generate_x_y(city_pol_means)
    trace = [{'x': x, 'y': y, 'type': 'bar'}]
    layout = go.Layout(xaxis={'title': 'Cities'}, yaxis={'title': 'Mean Polarity'})
    return dcc.Graph(id ='mean-polarity', figure = {'data': trace, 'layout': layout})

# def generate_com_pol_plot():

#
@app.callback(Output(component_id = 'plot-container', component_property ='children'),
[Input(component_id = 'select-xvar',component_property = 'value')])
def generate_plot(input_value):
    if input_value == 'tweets':
        return generate_tweet_plot()
    # elif input_value == 'hour':
    #     return generate_hour_plot()
    elif input_value == 'sentiment':
        return generate_sentiment_plot()
    elif input_value == 'polarity':
        return generate_polarity_plot()
    # elif input_value == 'com_pol':
    #     return generate_com_pol_plot()

def check_model(model):
    if model=='log':
        return log
    elif model=='forest':
        return forest
    # else:
    #     return svm

@app.callback(Output(component_id = 'cm-container', component_property ='children'),
[Input(component_id = 'select-model',component_property = 'value')])
def confusion_matrix_plot(input_value):
    model = check_model(input_value)
    y_hat = log.predict(x_test)
    labels = [1,2,3,4]
    cm = confusion_matrix(y_test, y_hat,labels=labels)
    trace = ff.create_annotated_heatmap(z=cm,
                    x=['Northeast', 'Midwest', 'South', 'West'],
                    y=['Northeast', 'Midwest', 'South', 'West'])
    return trace
