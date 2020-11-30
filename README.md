# fquad-exploration
FQuAD is the first French Question Answering Dataset, it has been developped by Illuin Technology and presented in this paper: https://arxiv.org/pdf/2002.06071.pdf
This package allows the user to select parts of the FQuAD dataset that are likely to respond to a given question.

## Okapi-BM25

Okapi-BM25 is a ranking function used by search engines to determine the relevance of documents relative to a given query.
We use this method in order to select the contexts most likely to answer to a given query.
By testing this method on the Training and Validation sets of the FQuAD dataset we obtain the following results:

![alt text](https://github.com/nassim-yagoub/fquad-exploration/blob/main/okapi_results.png?raw=true)

## CamemBERT

We also used the CamemBERT model with the following weigths: https://huggingface.co/illuin/camembert-base-fquad
This is the weights obtained by training the CamemBERT base model on the FQuAD dataset, making it especially adapted to our problem.
Our approach is to apply this model to every question/answer pair and look at how confident the model is about finding the right extract.

However, the complexity of this approach is very high, it takes several minutes to compute every question / answer couple without a GPU.
It makes it impossible to obtain trustable results, we can assume from the first results that this approach is very effective on the training set but does not work very well with the validation set.

Another possibility not implemented yet is to use CamemBERT to obtain embeddings of the questions and answers and create a metric of similarity for those embeddings. For example by computing the cosine of those vectors. However, the interesting part of the context might be very limited and most of the context may not present a strong similarity with the question.

This approach would reduce the complexity by limiting the number of calls to the model from Q * C et Q + C with Q the number of questions and C the number of contexts.
