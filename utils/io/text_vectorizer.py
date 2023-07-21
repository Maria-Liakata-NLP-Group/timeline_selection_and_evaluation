import numpy as np


class TextVectorizer:
    def __init__(self, embedding_type="sentence-bert"):
        from sentence_transformers import SentenceTransformer

        if embedding_type == "sentence-bert":
            self.sentence_bert_model = SentenceTransformer(
                "paraphrase-distilroberta-base-v1"
            )

    def vectorize(
        self, post, embedding_type="sentence-bert", apply_preprocessing=False
    ):
        if apply_preprocessing:
            post = self.pre_process(post)

        # Sentence Bert
        if embedding_type == "sentence-bert":
            vectorised_post = self.sentence_bert_model.encode([post]).reshape(1, -1)
        else:
            print(
                "Error, {} is not a vectorizer we have code for.".format(embedding_type)
            )

        return vectorised_post

    def pre_process(self, text):
        text = text.lower()
        return text
