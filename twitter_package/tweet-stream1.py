import tweepy
import dataset
import json
from tweepy import StreamListener
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from models import *
from config import *

analyser = SentimentIntensityAnalyzer()
def sentiment_score(text):
    return analyser.polarity_scores(text)

api = tweepy.API(auth)

def calculate_centroid(box):
    avg_lat = (box[1][1] + box[0][1])/2
    avg_long = (box[2][0] + box[1][0])/2
    return avg_lat, avg_long

LOCATIONS = [-124.7771694, 24.520833, -66.947028, 49.384472,       # Contiguous US
            -164.639405, 58.806859, -144.152365, 71.76871,         # Alaska
            -160.161542, 18.776344, -154.641396, 22.878623]        # Hawaii

def add_item(item):
    db.session.add(item)
    db.session.commit()

def check_user(id, created, retweets, centroid_lat, centroid_long,
loc, user_created, description, followers, friends, statuses, positivity,
negativity, compound, polarity, twitter_id, text, utc_offset):
    if id not in [user.user_id for user in User.query.all()]:
        # print('id {} not present in db'.format(id))
        user = User(user_id=id, created=created, description=description, followers=followers, friends=friends,
        statuses=statuses, utc_offset=utc_offset,location=loc)
        add_item(user)
        user_id = user.id
        add_item(Tweet(twitter_id=twitter_id, text=text, created=created, retweets=retweets, centroid_lat=centroid_lat,
        centroid_long=centroid_long, positivity=positivity, negativity=negativity, compound=compound, polarity=polarity, user_id=user_id))
    else:
        add_item(Tweet(twitter_id=twitter_id, text=text, created=created, retweets=retweets, centroid_lat=centroid_lat,
        centroid_long=centroid_long,positivity=positivity, negativity=negativity, compound=compound, polarity=polarity, user_id=id))

# db = dataset.connect("postgresql://chris@localhost:5432/app")
engine = create_engine('postgresql://chris:@localhost5432/app')
Session = sessionmaker(bind=engine)
session = Session()

class StreamListener(tweepy.StreamListener):
    def on_connect(self):
        print('Now were listening to twitter!')

    def on_status(self, status):
    #avoids retweets, non-geolocated
        if status.retweeted:
            return
        if not status.place:
            return
        id_str = status.id_str
        if status.truncated == True:
            text = status.extended_tweet['full_text']
        else:
            text=status.text
        created = status.created_at
        retweets = status.retweet_count
        box = status.place.bounding_box.coordinates[0]
        centroid_lat, centroid_long = calculate_centroid(box)
        coords = status.coordinates
        if coords is not None:
            coords = json.dumps(coords)
        loc = status.user.location
        user_created = status.user.created_at
        utc_offset = status.user.utc_offset
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
        check_user(user_id, created, retweets, centroid_lat, centroid_long,
        loc, user_created, description, followers, friends, statuses, positivity,
        negativity, compound, polarity, id_str, text, utc_offset)

    def on_error(self, status_code):
        if status_code == 420:
            return False

stop = ['the','i','to','a','and','is','in','it','you','of','for','on','my','that','at','with','me',
'do','have','just','this','be','so','are','not','was','but','out','up','what','now','new','from','your',
'like','good','no','get','all','about','we','if','time','as','day','will','one','twitter','how','can','some',
'an','am','by','going','they','go','or','has','know','today','there','love','more','work','=','too','got',
'he','2','back','think','did','lol','when','see','really','had','great','off','would']

stream_listener = StreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
test = stream.filter(locations=LOCATIONS, languages=["en"], track=stop)
