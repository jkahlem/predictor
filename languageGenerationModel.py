import model

class LanguageGenerationModel(model.Model):
    def __init__(self):
        pass

    # initializes a new model
    def init_new_model(self) -> None:
        pass

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        pass

    # Trains the model using the given training set
    def train_model(self, training_set) -> None:
        pass

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set) -> dict:
        pass
