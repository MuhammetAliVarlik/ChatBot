# -*- coding:utf-8 -*-
import numpy as np
from tensorflow.python.framework import ops
import json
import tensorflow as tf
import random
import trnlp
from trnlp import TrnlpWord
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

class ChatBot:
    def __init__(self):
        with open("intents.json", encoding='utf8') as file:
            self.data = json.load(file)
        with open("data.pickle", "rb") as f:
            self.words_last, self.labels, self.training, self.output_last = pickle.load(f)
        try:
            self.model = load_model("model1.model")
        except:
            self.retrain_model()

    def retrain_model(self):
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in self.data["intents"]:
            for pattern in intent["pattern"]:
                wrds = trnlp.word_token(pattern)
                words.extend(wrds)
                docs_x.append(pattern)
                docs_y.append(intent["tag"])
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
        words2 = []
        for w in words:
            obj = TrnlpWord()
            obj.usepron = False
            obj.setword(trnlp.helper.to_lower(w))
            if w != "?":
                words2.append(obj.get_stem)

        words2 = sorted(list(set(words2)))
        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]
        wrd = []
        for x, doc in enumerate(docs_x):
            bag = []
            if " " in doc:
                t = doc.split()
                for i in t:
                    obj = TrnlpWord()
                    obj.usepron = False
                    obj.setword(trnlp.helper.to_lower(i))
                    # print(i)
                    wrd.append(obj.get_stem)
                    # print(wrd)
            else:
                obj = TrnlpWord()
                obj.usepron = False
                obj.setword(trnlp.helper.to_lower(doc))
                wrd.append(obj.get_stem)
            for w in words2:
                if w in wrd:
                    bag.append(1)
                else:
                    bag.append(0)
            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)
        training = np.array(training)
        output = np.array(output)
        with open("data.pickle", "wb") as f:
            pickle.dump((words2, labels, training, output), f)
        ops.reset_default_graph()
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(output[0]), activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.fit(training, output, epochs=800, verbose=1, batch_size=7)
        self.model.save("model1.model")
        print("Tamamdır")

    @staticmethod
    def bag_words(s, words):
        bag = [0 for _ in range(len(words))]
        # print(words)
        s_words = trnlp.word_token(s)
        s_words2 = []
        for w in s_words:
            obj = TrnlpWord()
            obj.usepron = False
            obj.setword(trnlp.helper.to_lower(w))
            if w != "?":
                s_words2.append(obj.get_stem)
        for se in s_words2:
            for i, w in enumerate(words):
                obj = TrnlpWord()
                obj.usepron = False
                obj.setword(trnlp.helper.to_lower(w))
                w = obj.get_stem
                if w == se:
                    bag[i] = 1
        return np.array(bag)

    def chat(self):

        while True:
            self.inp = input("Siz:")
            if self.inp.lower() == "çıkış yap":
                break
            if self.inp.lower() == "modeli yeniden eğit":
                self.retrain_model()
            tensor = tf.convert_to_tensor(np.array([self.bag_words(self.inp, self.words_last)]))
            results = self.model.predict(tensor)
            results_index = np.argmax(results)
            if (results[0][results_index] * 60) >= 0:
                # for i in range(len(results[0])):
                # print(labels[i], (results[0][i]) * 100)
                self.tag = self.labels[results_index]
                for tg in self.data["intents"]:
                    if tg["tag"] == self.tag:
                        self.responses = tg["responses"]
                        self.response=random.choice(self.responses)
            else:
                self.response="Bunu anlamadım"
            K.clear_session()
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            return  self.tag,self.response,self.inp.lower()
