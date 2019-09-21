from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json

#consumer key, consumer secret, access token, access secret.
ckey="1owGswjGGWp9bm5JkktnWECQn"
csecret="1vINrhQxT5ctZ2bW3PQKC2ho9gpbTYni3fob0iXzlEylMTT2kY"
atoken="399667865-QYYYNMKbIHRDFH7ao71AJyOvJW0ywaUIw5cDUVM5"
asecret="BfZ09Ry1Zz7Du16hTsysSxT60X3ERYdH5pyDWMfaUoyAj"

words = ["bear",'crash','sell','dump','pump','surge','demand','break','profit','trust','btc']

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        sentiment = 0
        for x in range(0,len(words)):
            print(words[x])
            if words[x] in all_data['text']:
                sentiment = sentiment + x - 5
                print('true')
        followers = all_data.get(u'followers_count')
        print(all_data['user']['followers_count'])
        print("start")
        print(all_data['text'])
        print("middle")
        print(sentiment)
        
        if followers:
            print(followers)
        print("end")

        return(True)

    def on_error(self, status):
        print(status)

    def on_status(self, status):
        print(status.text)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
print('here')
twitterStream.filter(track=["BTC"])
print('here2')



