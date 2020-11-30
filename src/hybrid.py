import argparse

from tqdm import tqdm

import camembert
import okapi_bm25

class HybridSelector:

    def __init__(self, cuda = False, model_size = 'base'):
        self.okapi = okapi_bm25.OkapiBM25()
        self.camembert = camembert.CamembertSelector(cuda, model_size)

    def topk_predictions(self, dataset_type, prediction_nb, okapi_number = 20):
        if dataset_type == "train":
            questions_o = self.okapi.train_q
            questions_c = self.camembert.train_q
            answers = self.okapi.train_a
            id_to_ans = self.okapi.train_id_to_ans
        else:
            questions_o = self.okapi.valid_q
            questions_c = self.camembert.valid_q
            answers = self.okapi.valid_a
            id_to_ans = self.okapi.valid_id_to_ans

        iterator = tqdm(range(len(questions_o)))

        correct = 0

        for i in iterator:
            question_o = questions_o[i]
            question_c = questions_c[i]
            best_fits = self.okapi.get_best_fits(question_o, okapi_number, dataset_type)
            good_ids = [best_fits[i][0] for i in range(okapi_number)]

            best_ids = self.camembert.choose_best_contexts(
                question_c,
                dataset_type,
                prediction_nb,
                limited_context = good_ids
                )

            if id_to_ans[tuple(answers[i])] in best_ids:
                correct += 1

            iterator.set_description("The right context is found {: .3f}% of the time".format(100*correct/(i+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default='precision')
    parser.add_argument("--dataset", type=str, default='train')
    parser.add_argument("--number", type=int, default=5)
    parser.add_argument("--question", type=str)

    args = parser.parse_args()

    selector = HybridSelector()
    selector.topk_predictions(args.dataset, args.number)
