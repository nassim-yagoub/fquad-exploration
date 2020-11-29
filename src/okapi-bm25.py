import os
import random
import math
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords')

def bow_len(bow):
    return sum([bow[key] for key in bow])

def average(lst): 
    return sum(lst) / len(lst)

def freq_in_doc(word, doc_bow):
    """
    Returns the number of iterations of a given word in a bag of words.
    """
    return doc_bow.get(word, 0)

def remove_stopwords(text):
    stop = stopwords.words('french')
    ans = []
    
    for word in text:
        if not word in stop:
            for char in [',', '.', ':', ';']:
                word = word.replace(char, '')
            ans.append(word)
            
    return ans

def process_dataset(df):

    # Create a list of documents
    contexts = []

    for i in range(len(df)):
        for j in range(len(df.iloc[i]['data']['paragraphs'])):
            text = remove_stopwords(df.iloc[i]['data']['paragraphs'][j]['context'].split())
            contexts.append(text)

    # Stemming the text
    stemmer = SnowballStemmer("french")
    contexts = [[stemmer.stem(word) for word in context] for context in contexts]

    # Change contexts into a bags of words
    bows = []

    for text in contexts:
        bows.append({})
        for word in text:
            bows[-1][word] = bows[-1].get(word, 0) + 1

    questions = []
    answers = []
    id_to_ans = {}
    for i in range(len(df)):
        for j in range(len(df.iloc[i]['data']['paragraphs'])):
            for k in range(len(df.iloc[i]['data']['paragraphs'][j]['qas'])):
                questions.append(df.iloc[i]['data']['paragraphs'][j]['qas'][k]['question'])
                answers.append([i,j])
            id_to_ans[(i,j)] = len(id_to_ans)
                

    questions = [[stemmer.stem(word) for word in remove_stopwords(question.split())] for question in questions]

    random.Random(42).shuffle(questions)
    random.Random(42).shuffle(answers)

    return bows, questions, answers, id_to_ans

class OkapiBM25:

    def __init__(self):
        # Load or create datasets
        # Load or calculate idf

        self.k1 = 1.6
        self.b = 0.75

        self.train_df = pd.read_json("../data/train.json")
        self.valid_df = pd.read_json("../data/valid.json")

        self.train_bows, self.train_q, self.train_a, self.train_id_to_ans = process_dataset(self.train_df)
        self.valid_bows, self.valid_q, self.valid_a, self.valid_id_to_ans = process_dataset(self.valid_df)

        self.nb_doc_train = 0

        for i in range(len(self.train_df)):
            for _ in range(len(self.train_df.iloc[i]['data']['paragraphs'])):
                self.nb_doc_train += 1

        self.nb_doc_valid = 0

        for i in range(len(self.valid_df)):
            for _ in range(len(self.valid_df.iloc[i]['data']['paragraphs'])):
                self.nb_doc_valid += 1

        self.create_idf_stores()

        self.avg_dl_train = average([bow_len(bow) for bow in self.train_bows])
        self.avg_dl_valid = average([bow_len(bow) for bow in self.valid_bows])

    def create_idf_stores(self):
        if not os.path.exists("../data/idf_store_train.json"):
            self.idf_store_train = {}
            print("Computing IDF values for the training set...")
            for question in tqdm(self.train_q):
                for word in question:
                    if word in self.idf_store_train:
                        pass
                    else:
                        self.idf_store_train[word] = self.idf_store_train.get(word, self.idf(word, "train"))

            print("Training IDF computed !")

            with open("../data/idf_store_train.json", 'w') as file:
                json.dump(self.idf_store_train, file)

        else:
            print("Loading precomputed IDF values for the training set...")
            with open("../data/idf_store_train.json", 'r') as file:
                self.idf_store_train = json.load(file)
            print("Values loaded !")

        if not os.path.exists("../data/idf_store_valid.json"):
            self.idf_store_valid = {}
            print("Computing idf values for the validation set...")
            for question in tqdm(self.valid_q):
                for word in question:
                    if word in self.idf_store_valid:
                        pass
                    else:
                        self.idf_store_valid[word] = self.idf_store_valid.get(word, self.idf(word, "valid"))

            print("Validation IDF computed !")

            with open("../data/idf_store_valid.json", 'w') as file:
                json.dump(self.idf_store_valid, file)

        else:
            print("Loading precomputed IDF values for the validation set...")
            with open("../data/idf_store_valid.json", 'r') as file:
                self.idf_store_valid = json.load(file)
            print("Values loaded !")

    def nb_of_docs(self, word, dataset_type):
        """
        Returns the number of documents containing a given word.
        """
        ans = 0

        if dataset_type == "train":
            bows = self.train_bows
        else:
            bows = self.valid_bows
        
        for bow in bows:
            if word in bow:
                ans += 1
        
        return ans

    def idf(self, word, dataset_type):
        """
        Computes the idf of a word in the entire corpus.
        """
        docs_nb = self.nb_of_docs(word, dataset_type)

        if dataset_type == "train":
            total_docs = self.nb_doc_train
        else:
            total_docs = self.nb_doc_valid

        return math.log((total_docs - docs_nb + 0.5) / (docs_nb + 0.5) + 1)

    def single_okapi(self, word, bow, dataset_type):
        """
        Computes the OKAPI BM25 score relative to a single word.
        """
        freq = freq_in_doc(word, bow)
        nb_of_words = bow_len(bow)

        if dataset_type == "train":
            idf_store = self.idf_store_train
            avg_dl = self.avg_dl_train
        else:
            idf_store = self.idf_store_valid
            avg_dl = self.avg_dl_valid
        
        return idf_store[word] * (self.k1 + 1) * freq / (freq + self.k1 * (1 - self.b + self.b * nb_of_words / avg_dl))

    def okapi_score(self, question, bow, dataset_type):
        """
        Computes the final OKAPI BM25 score relative to an entire question.
        """
        return sum([self.single_okapi(word, bow, dataset_type) for word in question])

    def get_best_fits(self, question, prediction_nb, dataset_type, text = False):

        if dataset_type == "train":
            bows = self.train_bows
        else:
            bows = self.valid_bows

        scores = []
        for i in range(len(bows)):
            scores.append([i, self.okapi_score(question, bows[i], dataset_type)])

        scores.sort(key = (lambda x: -x[1]))
        
        if text:
            return [(scores[i][1], ' '.join(bows[scores[i][0]].keys())) for i in range(prediction_nb)]

        return scores[:prediction_nb]

    def top_k_precision(self, dataset_type, prediction_nb):
        # Computes the precision on the train or the test set, which are treated the same way with this approach

        if dataset_type == "train":
            questions = self.train_q
            answers = self.train_a
            id_to_ans = self.train_id_to_ans
        else:
            questions = self.valid_q
            answers = self.valid_a
            id_to_ans = self.valid_id_to_ans

        iterator = tqdm(range(len(questions)))

        correct = 0

        for i in iterator:
            question = questions[i]
            good_ids = [self.get_best_fits(question, prediction_nb, dataset_type)[i][0] for i in range(prediction_nb)]
            
            if id_to_ans[tuple(answers[i])] in good_ids:
                correct += 1

            iterator.set_description("The right context is found {: .3f}% of the time".format(100*correct/(i+1)))

        return correct / len(iterator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default='precision')
    parser.add_argument("--dataset", type=str, default='train')
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--question", type=str)

    args = parser.parse_args()

    okapi = OkapiBM25()

    if args.task == 'precision':
        okapi.top_k_precision(args.dataset, args.number)

    elif args.task == 'selection':
        stemmer = SnowballStemmer("french")
        question = [stemmer.stem(word) for word in remove_stopwords(args.question.split())]
        print(okapi.get_best_fits(question, args.number, args.dataset, text = True))
    