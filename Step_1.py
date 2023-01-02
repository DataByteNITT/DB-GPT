import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#nltk.download('all') # use this if any Lookup error occurs.

input = "This is Surendra's with ?/// Punctuations ;' = + code running"
def preprocess(text):
    # Removal of Punctuation.
    text_without_punc = ""
    for word in text:
        if word not in string.punctuation:
            text_without_punc+=word

    # Lower case every word : 
    text_without_punc  =text_without_punc.lower()

    # Tokenization:
    tokenlist = nltk.word_tokenize(text_without_punc)

    # Removal of stopwords:
    tokens_without_sw = [ word for word in tokenlist if word not in stopwords.words()]

    # Lemmatization : 
    lemmas = WordNetLemmatizer()
    Lemmatokens = [lemmas.lemmatize(token) for token in tokens_without_sw]

    return Lemmatokens

print(preprocess(input))




