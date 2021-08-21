import re
import pandas as pd
from keras.preprocessing.text import Tokenizer


class DataSet:
    def __init__(self, train_path_e, test_path_e, train_path, test_path, label2index, normalizer,tokenizer):
        self.train_path_e = train_path_e
        self.test_path_e = test_path_e
        self.train_path = train_path
        self.test_path = test_path
        self.label2index = label2index
        self.normalizer = normalizer
        self.tokenizer = tokenizer



    def get_e1(self, text):
        text = re.sub('[E21]', '', text)
        text = re.sub('[E22]', '', text)
        e = re.findall("[E11](.*?)[E12]", text)
        pre_process = set()
        for ent in e:
            if ent != ' ' and len(ent.strip()) > 1:
                pre_process.add(ent.strip())
        return list(pre_process)

    def get_e2(self, text):
        text = re.sub('[E11]', '', text)
        text = re.sub('[E12]', '', text)
        e = re.findall("[E21](.*?)[E22]", text)
        pre_process = set()
        for ent in e:
            if ent != ' ' and len(ent.strip()) > 1:
                pre_process.add(ent.strip())
        return list(pre_process)

    def process_sentence(self, sentence, normilizer):
        sentence = normilizer.normalize(sentence)
        sentence = sentence.replace('<e1>', '[E11]').replace('</e1>', '[E12]').replace('<e2>', '[E21]').replace('</e2>',
                                                                                                                '[E22]')
        return sentence

    def get_e1p(self, text):
        sentence = self.tokenizer.tokenize(text)
        E11_p = [i + 2 for i, e in enumerate(sentence) if e == '[E11]']
        E12_p = [i for i, e in enumerate(sentence) if e == '[E12]']
        return zip(E11_p, E12_p)

    def get_e2p(self, text):
        sentence = self.tokenizer.tokenize(text)
        E21_p = [i + 2 for i, e in enumerate(sentence) if e == '[E21]']
        E22_p = [i for i, e in enumerate(sentence) if e == '[E22]']
        return zip(E21_p, E22_p)

    def label_spliter(self, label):
        order = re.findall("\((.*?)\)", label)[0].split(',')
        label_names = label.split("(")[0].split("-")
        if order[0] == "e1":
            return label_names
        else:
            return list(reversed(label_names))

    def make_dataframe_row(self, sentence, label):
        if label == "Other":
            labels = [label, label]
        else:
            labels = self.label_spliter(label=label)

        result = []
        clean_sentence = self.process_sentence(sentence, self.normalizer)
        e1s_p = self.get_e1p(clean_sentence)
        e2s_p = self.get_e2p(clean_sentence)

        for e11, e12 in e1s_p:
            for e21, e22 in e2s_p:
                result.append({
                    "e1_label": labels[0],
                    "e2_label": labels[1],
                    "label": label,
                    "e1_start": e11,
                    "e1_end": e12,
                    "e2_start": e21,
                    "e2_end": e22,
                    "nlabel": self.label2index[label],
                    "sentence": clean_sentence
                })
        return result

    def make_dataframe(self, path):
        f = open(path, 'r')
        data = [x.rstrip() for x in f]
        data_set_rows = []
        clean_sentences = []
        for i in range(0, len(data) - 4, 4):
            item = data[i].split('\t')
            sentence = re.sub('[!@#$،.()]', '', item[1])
            label = data[i + 1]
            rows = self.make_dataframe_row(sentence, label)
            data_set_rows += rows
        return pd.DataFrame(data_set_rows)

    def tokenize_sentences(self, df, df_test):
        all_sentence = df['sentence'].tolist() + df_test['sentence'].tolist()
        keras_tokenizer = Tokenizer(num_words=200000)
        keras_tokenizer.fit_on_texts(all_sentence)
        return all_sentence, keras_tokenizer

    def get_dataset_entity(self, df):
        X_train = df.sentence.tolist()
        X_train_e1_se = zip(df.e1_start.tolist(), df.e1_end.tolist())
        X_train_e2_se = zip(df.e2_start.tolist(), df.e2_end.tolist())
        y_train = df.nlabel.tolist()

        return X_train, X_train_e1_se, X_train_e2_se, y_train
