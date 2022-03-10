from methods import Method, MethodContext, Parameter
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

GenerateParametersTask = 'generate parameters'
AssignReturnTypeTask = 'assign returntype'
AssignParameterTypeTask = 'assign parametertype'

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
        #trainDf = pd.read_csv(training_set, header=None, names=['prefix', 'input_text', 'target_text'], sep=';', na_filter=False)
        args = T5Args()
        args.num_train_epochs = 3
        model = T5Model('t5', 't5-small', args=args, use_cuda=True)
        model.train_model(training_set)

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set: list) -> dict:
        pass
        #return self.languageModeling.eval_model(evaluation_set)

    # Not needed for parameter generation
    def load_additional(self, additional_data) -> None:
        pass

    # Makes predictions
    def predict(self, predictionData: list[Method]) -> list[str]:
        model = T5Model('t5', 'outputs')
        inputs = list()
        for method in predictionData:
            inputs.append(GenerateParametersTask + ': ' + self.__getGenerateParametersInput(method))
        outputs = model.predict(inputs)
        print(outputs)
        return outputs

    # path addition for the cacheDir
    def cacheDirName(self) -> str:
        return 'cache_dir/'

    # path addition for the outputs dir 
    def outputsDirName(self) -> str:
        return 'outputs/'
    
    # Returns a method identifier for the method for caching. Methods with the same identifier won't be predicted again.
    def methodIdentifier(self, method: MethodContext) -> str:
        static = ''
        if method.isStatic:
            static = 'static,'
        return static + method.className + "," + method.methodName

    # formats input data
    def dataFormatter(self, data: list[Method]):
        frame = pandas.DataFrame(columns=['prefix', 'input_text', 'target_text'])
        for method in data:
            self.__addMethodToFrame(method, frame)
        return data
    
    def __addMethodToFrame(self, method: Method, frame: pandas.DataFrame):
        self.__addGenerateParametersTask(method, frame)
        self.__addGenerateReturnTypeTask(method, frame)
        self.__addGenerateParameterTypeTask(method, frame)

    def __addGenerateParametersTask(self, method: Method, frame: pandas.DataFrame):
        self.__addTask(GenerateParametersTask, self.__getGenerateParametersInput(method.context), self.__parameterNames(method.values.parameters), frame)
    
    def __getGenerateParametersInput(self, context: MethodContext) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '.'
    
    def __parameterNames(self, parameters: list[Parameter]) -> str:
        if len(parameters) == 0:
            return 'void'
        output = ''
        for par in parameters:
            output += par.name
        return output

    def __addGenerateReturnTypeTask(self, method: Method, frame: pandas.DataFrame):
        self.__addTask(AssignReturnTypeTask, self.__getGenerateReturnTypeInput(method.context), method.values.returnType, frame)

    def __getGenerateReturnTypeInput(self, context: MethodContext) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '.'

    def __addGenerateParameterTypeTask(self, method: Method, frame: pandas.DataFrame):
        for par in method.values.parameters:
            self.__addTask(AssignParameterTypeTask, self.__getGenerateParameterTypeInput(method.context, par), par.type, frame)
            
    def __getGenerateParameterTypeInput(self, context: MethodContext, par: Parameter) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '. parameter: ' + par.name

    def __addTask(self, prefix, input_text, target_text, frame: pandas.DataFrame):
        frame.append(pd.Series({'prefix': prefix, 'input_text': input_text, 'target_text': target_text}))

## Type assignment task
# prefix: "type assignment"
# input_text: "name: <parameter name> context: <types in context>"
# target_text: "<expected parameter type>" OR "0" 
#   - "0" if type is not available in context. As its a number, it is not allowed as a type in Java and therefore can be used without collision.
#
## Type classification task
#
# prefix: "type classification"
# input_text: "<parameter name>"

# TODO: Create resource directory if not existing ... if this is still needed ...