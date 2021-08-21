from src.perlex_project.utils import save_obj
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def crate_ft_embedding(ft, tokenizer , path_embeddings):
    ft_embeddings = {}
    for token in tokenizer.word_index:
        vector = ft.get_word_vector(token)
        ft_embeddings[token] = vector
    save_obj(ft_embeddings, 'fatsttext_embeddings')
    del ft_embeddings


def crate_word2vec_embedding(path, tokenizer,path_embeddings):
    word2vec = {}
    files = open(path, errors='replace', mode='r', encoding='utf-8')
    lines = files.readlines()[1:]
    files.close()
    for line in lines:
        word, *vector = line.split()
        if word in tokenizer.word_index:
            word2vec[word] = np.array(vector, dtype=np.float32)
    save_obj(word2vec, 'word2vec_embeddings')
    del word2vec


def create_embeddings_matrix(w2v, ft, tokenizer):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 400))
    for word in tokenizer.word_index:
        w2v_vector = w2v.get(word) if w2v.get(word) is not None else np.zeros((1, 100))
        ft_vector = ft.get(word) if ft.get(word) is not None else np.zeros((1, 300))
        idx = tokenizer.word_index[word]
        embedding_matrix[idx] = np.concatenate((ft_vector, w2v_vector), axis=None)
    return embedding_matrix


def tokenize_padd(df, tokenizer, max_len):
    sentence = df['sentence'].values
    # labels = pd.get_dummies(df['label']).values
    labels = df['nlabel'].values
    sequences = tokenizer.texts_to_sequences(sentence)
    sequences_matrix = pad_sequences(sequences, padding='post', maxlen=max_len)
    return sequences, sequences_matrix, labels
