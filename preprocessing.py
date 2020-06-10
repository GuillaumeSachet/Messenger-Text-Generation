import re
import string
import numpy as np
from json_to_dict import Json

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001F923"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_punct(text):
    punct = string.punctuation[:6]+string.punctuation[7:] # remove all punct except '
    table=str.maketrans('','',punct)
    return text.translate(table)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


def detect_messenger_sentence(text):
    """Ban French sentences created by Messenger (like make someone admin, changing nickname, polls...)."""
    ban_words = ['administrateur',
                 'sondage',
                 'discussion vidéo',
                 'a voté pour',
                 'a rejoint l’appel',
                 'a surnommé',
                 'a commencé à partager une vidéo',
                 'a changé la photo du groupe']
    
    ret = False
    for i in ban_words:
        if re.search(i,text):
            ret = True
    return ret
    
def clean(text):
    text = text.lower()
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_punct(text)
    text = re.sub(r'[" "]+', " ", text) # convert multiple whitespaces to 1 whitespace
    text = text.strip() # strip whitespaces at the beginning and end
#     text = '< ' + text + ' >'
    if detect_messenger_sentence(text):
        return []
    return text

def sentence_split(message):
    """Take message as input, output list of sentences."""
    
    n_split = re.split(r'[\n]',message) # split by \n
    ret = []
    for s in n_split:
        s = s.strip()
        raw_split = re.split(r'([?.!]+)',s) # split sentences when they end
        raw_split = list(filter(None, raw_split))
        if len(raw_split)>1:
            for i in range(0,len(raw_split)-1,2):
                raw_split_ponct = raw_split[i]+raw_split[i+1]
                ret.append(raw_split_ponct)
    return ret

def preprocess(list_json):
    """Take list of json and output dictionnary with senders as keys and sentences as values."""
    
    dic_out = dict()
    for json_name in list_json:
        ojb_json = Json(json_name)
        raw_dict_senders = ojb_json.load_messages()
        for key in raw_dict_senders.keys():
            dic_out.setdefault(key, [])
            for raw_data in raw_dict_senders[key]:
                for raw_sentence in sentence_split(raw_data):
                    s = clean(raw_sentence)
                    if s:
                        dic_out[key].append(s)
    return dic_out