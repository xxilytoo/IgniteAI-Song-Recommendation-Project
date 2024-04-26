from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.utils import register_keras_serializable
import keras
import json
import random

app = Flask(__name__)

df = pd.read_csv("Song List with Emotion Values.csv")
df = df.drop(columns = "Unnamed: 0")
dl = pd.read_csv("finalsongurls.csv")
dl = dl.drop(columns = "Unnamed: 0")
dl['title'] = dl['title'].str.strip().str.lower()
dl.reset_index(drop=True, inplace=True)
dl.drop_duplicates(subset='title', keep='first', inplace=True)

def getSongs(emotion):
    rando = []
    songnames = []
    while (len(songnames)<5):
        num = random.randint(0,len(df))
        if(num not in rando):
            rando.append(num)
            val = df.iloc[num][emotion]
        if(val > 0):
            songnames.append(df.iloc[num]["Title"])
    return songnames

def geturls(songnames):
    urls = []
    for name in songnames:
        # Check if there's any match for the current song name
        if not dl[dl["title"] == name].empty:
            # Get the index of the first occurrence of the song name
            index = dl[dl["title"] == name].index[0]
            # Use the index to fetch the corresponding "uri" value
            link = dl.loc[index, "uri"]
            #print(link)
            urls.append(link)
        else:
            #print("No match found for", name)
            urls.append("No match found for" + name)
    return urls


def clean_text(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, 'http\S+', '')
    text = tf.strings.regex_replace(text, '([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)', ' ')
    text = tf.strings.regex_replace(text, '[/(){}\[\]\|@,;]', ' ')
    text = tf.strings.regex_replace(text, '[^0-9a-z #+_]', '')
    text = tf.strings.regex_replace(text, '[\d+]', '')
    return text


@keras.saving.register_keras_serializable(package='my_package', name='TextCleaner')
class TextCleaner:
    def __init__(self):
        self.clean_text = clean_text

    def __call__(self, text):
        return self.clean_text(text)

    def get_config(self):
        return {}
    @classmethod
    def from_config(cls, config):
        return cls()

model_test = keras.saving.load_model("nlptest.keras")

def emotiondet(outputarr):
    store = list(outputarr.flatten())
    emotions = ['Fun', 'Love', 'sadness', 'anticipation']
    maxindex = store.index(max(store))
    return emotions[maxindex]




@app.route('/')
def home():
    return render_template('index.html')


@app.route('/model', methods=['POST'])
def prediction():
    dictionary = request.get_json(force=True)
    
    #debugging json
    json_str = json.dumps(dictionary, indent=4)
    print(json_str)
    
    sentence = dictionary.get("input")
    if(len(sentence) < 1):
        return jsonify("")
    #print("the sentence type is:" + str(type(sentence)))
    pre = pd.Series(sentence)
    #print("the pre type is:" + str(type(pre)))
    #print(pre)
    result = model_test.predict(pre)
    emotion = emotiondet(result)
    #emotion = "hello"
    recsongs = getSongs(emotion)
    recurls = geturls(recsongs)
    songrecdict = {
        "songnames" : recsongs,
        "songurls" : recurls
    }
    songrecjson = json.dumps(songrecdict, indent=4)
    return songrecjson
    """
    """


if __name__ == '__main__':
    app.run(port=5000, debug=True)
