from waitress import serve
from flask import Flask, render_template, request
from time import strftime
import time
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
import datetime
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import regex
import nltk
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
import string

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():
    """Return the main page."""
    time_str = strftime("%m/%d/%Y %H:%M")
    print(time_str)
    return render_template("index.html", time_info=time_str)

# Functions necessary for the website model to run

def get_clean_text_pattern(recomposed_note):
    """Function that filters through the notes, retrieves those that match
     the specified pattern and removes stopwords."""
    pattern = "([a-zA-Z0-9\\\]+(?:'[a-z]+)?)"
    recomposed_note_raw = nltk.regexp_tokenize(recomposed_note, pattern)
    # Create a list of stopwords and remove them from our corpus
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    # additional slang and informal versions of the original words had to be added to the corpus.
    stopwords_list += (["im", "ur", "u", "'s", "n", "z", "n't", "brewskies", "mcd’s", "Ty$",
                        "Diploooooo", "thx", "Clothessss", "K2", "B", "Comida", "yo", "jobby",
                        "F", "jus", "bc", "queso", "fil", "Lol", "EZ", "RF", "기프트카드", "감사합니다",
                        "Bts", "youuuu", "X’s", "bday", "WF", "Fooooood", "Yeeeeehaw", "temp",
                        "af", "Chipoodle", "Hhuhhyhy", "Yummmmers", "MGE", "O", "Coook", "wahoooo",
                        "Cuz", "y", "Cutz", "Lax", "LisBnB", "vamanos", "vroom", "Para", "el", "8==",
                        "bitchhh", "¯\\_(ツ)_/¯", "Ily", "CURRYYYYYYY", "Depósito", "Yup", "Shhhhh"])

    recomposed_note_stopped = ([w.lower() for w in recomposed_note_raw if w not in stopwords_list])
    return recomposed_note_stopped


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_notes(recomposed_note_stopped):
    "Function that lemmatizes the different notes."
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_notes = []
    for sentence in recomposed_note_stopped:
        for word in nltk.word_tokenize(sentence):
            lem = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            lemmatized_notes.append(lem)
    return lemmatized_notes


def get_user_df(when_did_you_open_your_account, n_transactions_made_last_week, how_many_were_during_the_week_end,
                how_many_were_yesterday, n_transactions_made_to_diff_users, trans_pending_of_those_made,
                max_time_between_transactions, average_time_between_transactions, text_description):
    "Function that returns a specific user dataframe with the relevant inputs"
    user_details = {}
    user_details['time_since_account_inception'] = (
        datetime.datetime.today() - when_did_you_open_your_account).total_seconds()
    user_details['n_transactions_made_last_week'] = n_transactions_made_last_week
    user_details['n_transactions_made_during_weekend'] = how_many_were_during_the_week_end
    user_details['n_transactions_made_during_week'] = (
        n_transactions_made_last_week - how_many_were_during_the_week_end)
    user_details['n_transactions_made_yesterday'] = how_many_were_yesterday
    user_details['n_transactions_made_to_diff_users'] = n_transactions_made_to_diff_users
    user_details['max_time_between_transactions'] = max_time_between_transactions
    user_details['mean_time_between_transactions'] = average_time_between_transactions   
    user_details['unsuccesful_transactions'] = trans_pending_of_those_made
    # Convert dict with responses to dataframe
    user_details_df = pd.DataFrame([user_details])
    # Dealing with the text aspect
    recomposed_note_stopped = get_clean_text_pattern(text_description)
    lemmatized_notes = lemmatize_notes(recomposed_note_stopped)
    # Load the vectorizer model
    vectorizer = Doc2Vec.load("d2v.model")
    # Find the vectors for each note in the whole note corpus
    _vectrs = [np.array(vectorizer.infer_vector(lemmatized_notes))]
    _vectrs_df = pd.DataFrame(_vectrs)
    
    user_details_combined = pd.concat([user_details_df, _vectrs_df], axis=1)
    return user_details_combined

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)