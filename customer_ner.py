import pandas as pd
import numpy as np
import unicodedata
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense
from keras.layers import TimeDistributed, Dropout, Bidirectional

def load_data(train_loc, test_loc):
    train_data = pd.read_csv(train_loc, encoding="latin1")
    test_data = pd.read_csv(test_loc, encoding="latin1")
    return (train_data, test_data)

def remove_duplicates(train_data, test_data):
    def drop_duplicates(data):
        if(train_data.nunique()['id'] < len(data)):
            data.drop_duplicates(inplace = True, keep = 'first')
        return data
    return (drop_duplicates(train_data), drop_duplicates(test_data))
        
def unique_words(train_data, test_data):
    temp_list = list(set(train_data["Word"].append(test_data["Word"]).values))
    temp_list.append('ENDPAD')
    return temp_list

def remove_nan(list_data):
    list_data.remove(np.nan)
    return list_data

def convert_latin_to_ascii(word, greek_code = 'NFKD'):
    return unicodedata.normalize(greek_code, str(word)).encode('ascii','ignore')

def sentence_to_markuptags(data, for_set, group_by = 'Sent_ID'):
    if for_set == 'train':
        agg_func = lambda x:[(convert_latin_to_ascii(word).lower(), tag) for word, tag in zip(x['Word'].values, x['tag'].values) ]
        return [sentence for sentence in data.groupby(group_by).apply(agg_func)]
    agg_func = lambda x: [(convert_latin_to_ascii(word).lower()) for word in x['Word'].values ]
    return [sentence for sentence in data.groupby(group_by).apply(agg_func)]

def convert_sentences_to_uinque_ids(sentences, word2idx, for_type):
    if for_type == 'train':
        return [[word2idx[word[0]] for word in sentence] for sentence in sentences]
    return [[word2idx[word] for word in sentence] for sentence in sentences]

def pad_words(sentences, vector_length, maxlen_threshold = 180):
    return pad_sequences(maxlen = maxlen_threshold, sequences=sentences, padding='post', value = vector_length)

def runner(train_dataset_url, test_dataset_url):
    train_data, test_data = load_data(train_dataset_url, test_dataset_url)
    train_data, test_data = remove_duplicates(train_data, test_data)
    words = list(set([convert_latin_to_ascii(word).lower() for word in remove_nan(unique_words(train_data, test_data))]))
    tags = list(set(train_data["tag"].values))
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i:t for i, t in enumerate(tags)}
    train_sentences = sentence_to_markuptags(train_data, for_set = 'train')
    test_sentences = sentence_to_markuptags(train_data, for_set = 'test')
    x_train = pad_words(convert_sentences_to_uinque_ids(train_sentences, word2idx, 'train'), vector_length=len(words)-1)
    x_test  = pad_words(convert_sentences_to_uinque_ids(test_sentences, word2idx, 'test'), vector_length=len(words)-1)
    y_train = [to_categorical(label, num_classes=len(tags)) for label in pad_words([[ tag2idx[word[1]] for word in sentence] for sentence in train_sentences], vector_length=len(tags)-1)]
    return x_train, y_train, x_test, word2idx, idx2word, tag2idx, idx2tag

def model_built_up(maxlen, input_dimension, output_dim, lstm_units = 250, activation = "softmax"):
    input = Input(shape=(maxlen,))
    model = Embedding(input_dim=input_dimension, output_dim=maxlen, input_length=maxlen)(input)
    model = Dropout(0.2)(model)
    model = Bidirectional(LSTM(units=lstm_units, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(output_dim, activation="softmax"))(model) # softmax output layer
    model = Model(input, out)
    return model

def compile_train(model, x_train, y_train, optmz = "adam", loss_metric = "categorical_crossentropy", batch_size = 50, epochs = 500, validation_split = 0.4):
    model.compile(optimizer=optmz, loss=loss_metric, metrics = ["accuracy"])
    trained_model = model.fit(x_train, np.array(y_train), batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)
    return trained_model

def test(x_test, trained_model, idx2word):
    return [[idx2word[word_id] for word_id in sentence] for sentence in np.argmax(trained_model.predict(x_test), axis = -1)] 
    
def predict_point(query_sentence, word2idx, idx2word, trained_model):
    query_point = [convert_latin_to_ascii(word) for word in query_sentence.split(' ')]
    return test(pad_words(sentences=np.array([word2idx[word] for word in query_point]).reshape(1, len(query_point)), vector_length=len(word2idx.keys())-1), trained_model=trained_model, idx2word=idx2word)


x_train, y_train, x_test, word2idx, idx2word, tag2idx, idx2tag = runner('/Users/deepak/Downloads/hackathon_disease_extraction/train.csv', '/Users/deepak/Downloads/hackathon_disease_extraction/test.csv')
model = model_built_up(180, len(word2idx.keys()), 3)
trained_model = compile_train(model, x_train, y_train)

sentence = input('Enter a sentence')

predict_point(sentence, word2idx, idx2word, trained_model)