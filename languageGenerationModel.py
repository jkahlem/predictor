import model
from util import copyTo
from uuid import uuid4
from config import get_model_config, get_resource_path, is_cuda_available
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs
from simpletransformers.language_generation import LanguageGenerationModel, LanguageGenerationArgs

# TODO: rework model loading... (method generation needs two models - language modeling model and generator, but model holder supports only one model)
#       -> the underlying model should do the needed stuff. However, the model holder should call the loading/initializing methods instead of the message script.
#       -> maybe rename to "initialize/load for training" and "initialize/load for prediction"...

class MethodGenerationModel(model.Model):
    def __init__(self):
        self.languageModeling = None
        self.languageGenerator = None

    # prints a message if cuda is used or not
    def __print_model_initialization(self) -> None:
        if is_cuda_available():
            print('Initialize language modelling model using CUDA')
        else:
            print('Initialize language modelling model without CUDA')
    
    def __languageModelingArgs(self) -> LanguageModelingArgs:
        return LanguageModelingArgs(cache_dir=self.cacheDirName(), output_dir=self.outputsDirName(), mlm=False)

    def __languageGenerationArgs(self) -> LanguageGenerationArgs:
        return LanguageGenerationArgs(cache_dir=self.cacheDirName(), output_dir=self.outputsDirName())

    # initializes a new model
    def init_new_model(self) -> None:
        self.__print_model_initialization()
        used_model_type, used_model = get_model_config()
        self.languageModeling = LanguageModelingModel('gpt2', 'gpt2', use_cuda=is_cuda_available(), args=self.__languageModelingArgs())

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        self.languageGenerator = LanguageGenerationModel('gpt2', self.outputsDirName(), use_cuda=is_cuda_available(), args=self.__languageGenerationArgs())

    # Trains the model using the given training set
    def train_model(self, training_set: list) -> None:
        self.languageModeling.train_model(training_set)

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set: list) -> dict:
        return self.languageModeling.eval_model(evaluation_set)

    # Not needed for parameter generation
    def load_additional(self, additional_data) -> None:
        pass

    # Makes predictions
    def predict(self, predictionData: list) -> list:
        output = list()
        for methodName in predictionData:
            print("Generate for '"  + methodName + "'")
            generated = self.languageGenerator.generate(methodName)[0]
            print(generated)
            output.append(generated)
        return output

    # path addition for the cacheDir
    def cacheDirName(self) -> str:
        return 'cache_dir/'

    # path addition for the outputs dir 
    def outputsDirName(self) -> str:
        return 'outputs/'

    # formats input data
    def dataFormatter(self, data):
        filename = 'lmd_' + str(uuid4())
        filepath = get_resource_path(filename)
        copyTo(data, filepath)
        return filepath
