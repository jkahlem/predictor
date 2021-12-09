from simpletransformers.t5 import T5Model, T5Args
from transformers.training_args import TrainingArguments
import model
from util import copyTo
from uuid import uuid4
import pandas as pd
from config import get_model_config, get_resource_path, get_script_dir, is_cuda_available
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs
from simpletransformers.language_generation import LanguageGenerationModel, LanguageGenerationArgs
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.trainer import Trainer
from transformers.models.auto import AutoModel
import torch
import math

# TODO: rework model loading... (method generation needs two models - language modeling model and generator, but model holder supports only one model)
#       -> the underlying model should do the needed stuff. However, the model holder should call the loading/initializing methods instead of the message script.
#       -> maybe rename to "initialize/load for training" and "initialize/load for prediction"...
#       - Also: clear temporary files (train/eval messages ...)
#
#  Using current T5 model works pretty good! example predictions:
#  - "getName" : "void."
#  - "createPerson" : "void."
#  - "setPosition" : "position."
#  - "withName" : "name."
#  - "findByNameOrAge" : "name, age."
#  - "compare" : "a, b."

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
        #self.languageModeling = LanguageModelingModel('gpt2', 'gpt2', use_cuda=is_cuda_available(), args=self.__languageModelingArgs())

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        pass
        #self.languageGenerator = LanguageGenerationModel('gpt2', self.outputsDirName(), use_cuda=is_cuda_available(), args=self.__languageGenerationArgs())

    # https://huggingface.co/blog/how-to-generate
    # https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0
    # https://link.springer.com/chapter/10.1007/978-3-030-91699-2_15
    # https://ieeexplore.ieee.org/document/9402114
    # https://jaketae.github.io/study/keyword-extraction/
    # https://www.lighttag.io/blog/sequence-labeling-with-transformers/
    # github.com/abhimishra91/transformers-tutorials/

    # Trains the model using the given training set
    def train_model(self, training_set: str) -> None:
        trainDf = pd.read_csv(training_set, header=None, names=['prefix', 'input_text', 'target_text'], sep=';', na_filter=False)
        args = T5Args()
        args.num_train_epochs = 3
        model = T5Model('t5', 't5-small', args=args, use_cuda=True)
        model.train_model(trainDf)
        

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set: list) -> dict:
        pass
        #return self.languageModeling.eval_model(evaluation_set)

    # Not needed for parameter generation
    def load_additional(self, additional_data) -> None:
        pass

    # Makes predictions
    def predict(self, predictionData: list) -> list:
        model = T5Model('t5', 'outputs')
        inputs = list()
        for methodName in predictionData:
            inputs.append("generate parameters: " + methodName)
        outputs = model.predict(inputs)
        print(outputs)
        return outputs

    # path addition for the cacheDir
    def cacheDirName(self) -> str:
        return 'cache_dir/'

    # path addition for the outputs dir 
    def outputsDirName(self) -> str:
        return 'outputs/'

    # formats input data
    def dataFormatter(self, data):
        if not isinstance(data, pd.DataFrame):
            return pd.read_csv(data, header=None, names=['prefix', 'input_text', 'target_text'], sep=';', na_filter=False)
        return data
