"""
scikit-learn Model Wrapper
--------------------------
"""


import pandas as pd

from .model_wrapper import ModelWrapper


class SklearnModelWrapper(ModelWrapper):
    """Loads a scikit-learn model and tokenizer (tokenizer implements
    `transform` and model implements `predict_proba`).

    May need to be extended and modified for different types of
    tokenizers.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list, batch_size=None):
        encoded_text_matrix = self.tokenizer.transform(text_input_list)
        propa = self.model.predict_proba(encoded_text_matrix)
        return propa

    def get_grad(self, text_input):
        raise NotImplementedError()
