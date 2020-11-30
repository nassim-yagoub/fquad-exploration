import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from transformers import CamembertForQuestionAnswering, CamembertTokenizer

def create_context(df):
    contexts = []
    for i in range(len(df)):
        for j in range(len(df.iloc[i]['data']['paragraphs'])):
            contexts.append(df.iloc[i]['data']['paragraphs'][j]['context'])
            
    return contexts

def create_qa(df):
    questions, answers = [], []
    id_to_ans = {}
    for i in range(len(df)):
        for j in range(len(df.iloc[i]['data']['paragraphs'])):
            for k in range(len(df.iloc[i]['data']['paragraphs'][j]['qas'])):
                questions.append(df.iloc[i]['data']['paragraphs'][j]['qas'][k]['question'])
                answers.append((i,j))
            id_to_ans[(i,j)] = len(id_to_ans)
            
    return questions, answers, id_to_ans

class CamembertSelector:

    def __init__(self, cuda = False, model_size = "base"):

        self.model = CamembertForQuestionAnswering.from_pretrained("illuin/camembert-{}-fquad".format(model_size), return_dict = True)
        self.tokenizer = CamembertTokenizer.from_pretrained("illuin/camembert-{}-fquad".format(model_size))

        self.train_df = pd.read_json("../data/train.json")
        self.valid_df = pd.read_json("../data/valid.json")

        self.train_contexts = create_context(self.train_df)
        self.valid_contexts = create_context(self.valid_df)

        self.train_q, self.train_a, self.train_id_to_ans = create_qa(self.train_df)
        self.valid_q, self.valid_a, self.valid_id_to_ans = create_qa(self.valid_df)

        self.cuda = cuda

        if cuda:
            self.model = self.model.cuda()

    def choose_best_contexts(self, question, dataset_type, number = 1):

        scores = []

        if dataset_type == "train":
            contexts = self.train_contexts
        else:
            contexts = self.valid_contexts

        for i in range(len(contexts)):

            context = contexts[i]

            inputs = self.tokenizer(question, context, return_tensors='pt')

            if self.cuda:
                inputs['input_ids'] = inputs['input_ids'][:,:510].cuda()
                inputs['attention_mask'] = inputs['attention_mask'][:,:510].cuda()
            else:
                inputs['input_ids'] = inputs['input_ids'][:,:510]
                inputs['attention_mask'] = inputs['attention_mask'][:,:510]

            outputs = self.model(**inputs)
                
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            score = float(max(list(start_scores[0])).data) + float(max(list(end_scores[0])).data)

            scores.append((i, score))

        scores.sort(key = (lambda x: -x[1]))

        return scores[:number]

    def get_performance(self, dataset_type, prediction_nb):

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
            best_contexts = self.choose_best_contexts(question, dataset_type, prediction_nb)
            good_ids = [best_contexts[i][0] for i in range(prediction_nb)]

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

    selector = CamembertSelector()

    if args.task == 'precision':
        selector.get_performance(args.dataset, args.number)

    elif args.task == 'selection':
        pass