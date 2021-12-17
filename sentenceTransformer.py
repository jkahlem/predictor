from sentence_transformers import SentenceTransformer, util
import numpy as np

def testSentenceTransformer():
    # https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
    methodname = "update contract: "
    model = SentenceTransformer('stsb-roberta-large')
    corpus = [methodname + "string", methodname+"integer", methodname+"short", methodname+"number", methodname+"list", methodname+"person", methodname+"document",
    methodname+"contract", methodname+"offer", methodname+"business partner"]
    # encode corpus to get corpus embeddings
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    sentence = methodname+"id"
    # encode sentence to get sentence embeddings
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    # top_k results to return
    top_k=2
    # compute similarity scores of the sentence with the corpus
    cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
    # Sort the results in decreasing order and get the first top_k
    top_results = np.argpartition(-cos_scores.cpu(), range(top_k))[0:top_k]
    print("Sentence:", sentence, "\n")
    print("Top", top_k, "most similar sentences in corpus:")
    for idx in top_results[0:top_k]:
        print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx]))
    
    # one option for finetuning:
    # - a dataset of different "test sets"
    # - a test set contains:
    #   - the target parameter name
    #   - maybe also the method name
    #   - the actual parameter type
    #   - other optional parameter types derived from inheritance (A)
    #   - other optional parameter types derived from context (B)
    #
    # Example:
    # import com.example.SalesContract;
    # import java.util.List;
    # ...
    # setPrice(BigInteger amount)
    # - actual parameter type: "big integer"
    # - (A) other optional inherited parameter types: "number", "object"
    # - (B) other optional parameter types: "sales contract", "list", "integer"
    # - all these types should be distinct.
    #
    # in this case, use following rules for the label (similarity)
    # - actual parameter type: always 1
    # - (A): use some normalized log-likelihood thing to derive a label? Use amount of steps until target class as base value ("big integer" = 0, "number" = 1, "object" = 2 ...)
    # - (B): for each type:
    #   - if they share a parent class (integer extends number), use the label derived at (A) for that parent class
    #   - otherwise, use 0.
    #
    # fine tuning similarity for programming languages makes sense, as for example:
    # - in natural language, "name" is a "word" (or a "string", but string is also a synonym for the yarn thing ...), but it is also something very related to a "person".
    #   in programming languages or for types, "name" is a "string", but it is not
    #
    # Instead of sbert, also see:
    # https://simpletransformers.ai/docs/sentence-pair-classification/
    #
    # Other approach: T5 model with data rows like:
    # - prefix: "type picking: "
    # - input: "set price - amount. big integer. number. object. sales contract. list. integer."
    # - output: "big integer"
    #
    # maybe other keywords of interest: "unsupervised text classification", "clustering", "word embeddings"
    # https://medium.com/@ai_medecindirect/unsupervised-text-classification-695392c6fac7

