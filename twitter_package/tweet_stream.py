import tweepy
import dataset
import json
from tweepy import StreamListener
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from models import *
import pandas as pd
import numpy as np
from config import *

analyser = SentimentIntensityAnalyzer()
def sentiment_score(text):
    return analyser.polarity_scores(text)

api = tweepy.API(auth)

def calculate_centroid(box):
    avg_lat = (box[1][1] + box[0][1])/2
    avg_long = (box[2][0] + box[1][0])/2
    return avg_lat, avg_long

#LOCATIONS = [-124.7771694, 24.520833, -66.947028, 49.384472,       # Contiguous US
#             -164.639405, 58.806859, -144.152365, 71.76871,         # Alaska
#             -160.161542, 18.776344, -154.641396, 22.878623]        # Hawaii
LOCATION = [-164.639405, 18.776344, -66.947028, 71.76871]

cities = [('New York', 40.7127837, -74.00594129999999),
 ('Los Angeles', 34.0522342, -118.24368490000002),
 ('Chicago', 41.8781136, -87.62979820000001),
 ('Houston', 29.7604267, -95.36980279999999),
 ('Philadelphia', 39.9525839, -75.1652215),
 ('Phoenix', 33.4483771, -112.07403729999999),
 ('San Antonio', 29.4241219, -98.4936282),
 ('San Diego', 32.715738, -117.1610838),
 ('Dallas', 32.7766642, -96.7969879),
 ('San Jose', 37.338208200000004, -121.88632859999998),
 ('Austin', 30.267153000000004, -97.7430608),
 ('Indianapolis', 39.768403, -86.158068),
 ('Jacksonville', 30.3321838, -81.65565099999998),
 ('San Francisco', 37.7749295, -122.4194155),
 ('Columbus', 39.9611755, -82.99879419999998),
 ('Charlotte', 35.2270869, -80.8431267),
 ('Fort Worth', 32.7554883, -97.3307658),
 ('Detroit', 42.331427000000005, -83.0457538),
 ('El Paso', 31.7775757, -106.44245590000001),
 ('Memphis', 35.1495343, -90.0489801),
 ('Seattle', 47.6062095, -122.33207079999998),
 ('Denver', 39.739235799999996, -104.990251),
 ('Washington', 38.9071923, -77.03687070000001),
 ('Boston', 42.360082500000004, -71.0588801),
 ('Nashville-Davidson', 36.1626638, -86.78160159999999),
 ('Baltimore', 39.2903848, -76.6121893),
 ('Oklahoma City', 35.4675602, -97.5164276),
 ('Louisville/Jefferson County', 38.252664700000004, -85.7584557),
 ('Portland', 45.523062200000005, -122.67648159999999),
 ('Las Vegas', 36.169941200000004, -115.13982959999998)]

def add_item(item):
    db.session.add(item)
    db.session.commit()

def find_closest_city(centroid_lat, centroid_long, cities=cities):
    smallest = 10000
    point = (centroid_lat, centroid_long)
    for city in cities:
        dist = np.sqrt((city[1]-point[0])**2 + (city[2]-point[1])**2)
        if dist < smallest:
            smallest = dist
            closest = city
    return closest

def get_city_id(lat, long):
    closest = find_closest_city(lat, long, cities=cities)
    if closest[0] not in [city.name for city in City.query.all()]:
        # print('New City! {}'.format(closest))
        city = City(name=closest[0], lat=closest[1], long=closest[2])
        add_item(city)
        city_id = city.id
    else:
        # print('Old City!')
        city = City.query.filter_by(name = closest[0]).all()
        city_id=city[0].id
    return city_id

def check_user(uid, created, retweets, centroid_lat, centroid_long,
loc, user_created, description, followers, friends, statuses, positivity,
negativity, compound, polarity, twitter_id, text, city_id):
    if uid not in [user.user_id for user in User.query.all()]:
        # print('New User!')
        # print('UID: {}'.format(uid))
        user = User(user_id=uid, created=created, description=description, followers=followers, friends=friends,
        statuses=statuses, location=loc)
        # print('User.id: {}'.format(user.id))
        # print('User.user_id: {}'.format(user.user_id))
        add_item(user)
        add_item(Tweet(twitter_id=twitter_id, text=text, created=created, retweets=retweets, centroid_lat=centroid_lat,
        centroid_long=centroid_long, positivity=positivity, negativity=negativity, compound=compound, polarity=polarity, user_id=user.id, city_id=city_id))
#        user = User.query.filter_by(user_id == str(uid)).all()[0]
#        print('User.id: {}'.format(user.id))
#        add_item(Tweet(twitter_id=twitter_id, text=text, created=created, retweets=retweets, centroid_lat=centroid_lat,
#        centroid_long=centroid_long, positivity=positivity, negativity=negativity, compound=compound, polarity=polarity, user_id=user.id, city_id=city_id))

# db = dataset.connect("postgresql://chris@localhost:5432/app")
engine = create_engine('postgresql://chris:@localhost5432/app')
Session = sessionmaker(bind=engine)
session = Session()

class StreamListener(tweepy.StreamListener):
    def on_connect(self):
        print('Now were saving from twitter!')

    def on_status(self, status):
    #avoids retweets, non-geolocated
        if status.retweeted:
            return
        if not status.place:
            return
        if status.lang == 'en':
            if status.truncated == True:
                text = status.extended_tweet['full_text']
                if len(text) > 320:
                    text = text[:320]
            else:
                text=status.text
            id_str = status.id_str
            created = status.created_at
            created_hr = status.created_at.hour
            retweets = status.retweet_count
            box = status.place.bounding_box.coordinates[0]
            centroid_lat, centroid_long = calculate_centroid(box)
            coords = status.coordinates
            if coords is not None:
                coords = json.dumps(coords)
            loc = status.user.location
            user_created = status.user.created_at
            description = status.user.description
            followers = status.user.followers_count
            friends = status.user.friends_count
            user_id = str(status.user.id)
            statuses = status.user.statuses_count
            sentiment = sentiment_score(text)
            positivity = round(sentiment['pos'], 4)
            negativity = round(sentiment['neg'], 4)
            compound = round(sentiment['compound'], 4)
            polarity = round((TextBlob(text)).sentiment.polarity, 4)
            city_id = get_city_id(centroid_lat, centroid_long)
            # print('city_id: {}'.format(city_id))
            check_user(user_id, created, retweets, centroid_lat, centroid_long,
            loc, user_created, description, followers, friends, statuses, positivity,
            negativity, compound, polarity, id_str, text, city_id)

    def on_exception(self, exception):
           print(exception)
           return

    def on_error(self, status_code):
        if status_code == 420:
            return False
stop = ['the','i','to','a','and','is','in','it','you','of','for','on','my','that','at','with','me','do','have','just','this','be','so','are','not','was','but','out','up','what','now','new','from','your','like','good','no','get','all','about','we','if','time','as','day','will','one','twitter','how','can','some','an','am','by','going','they','go','or','has','know','today','there','love','more','work','=','too','got','he','2','back','think','did','lol','when','see','really','had','great','off','would']
stream_listener = StreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
test = stream.filter(locations=LOCATION, track=stop)
