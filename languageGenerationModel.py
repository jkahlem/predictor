import os
import shutil
import model
import pandas as pd
from random import randint
from io import StringIO
from config import get_labels_path, get_model_config, is_cuda_available, is_test_mode
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs
from simpletransformers.language_generation import LanguageGenerationModel, LanguageGenerationArgs
from sklearn.metrics import f1_score, accuracy_score

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
        return LanguageModelingArgs(cache_dir=self.cacheDirName(), output_dir=self.outputsDirName())

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
            output.append(self.languageGenerator.generate(methodName))
        return output

    # path addition for the cacheDir
    def cacheDirName(self) -> str:
        return 'cache_dir/'

    # path addition for the outputs dir 
    def outputsDirName(self) -> str:
        return 'outputs/'

    # formats input data into a list
    def dataFormatter(self, data) -> list:
        if not isinstance(data, list):
            return data.split("\n")
        return data
