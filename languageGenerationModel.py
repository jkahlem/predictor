from messages import Options
from methods import Method, MethodContext, MethodValues, Parameter
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

class MethodGenerationModel(model.Model):
    options: Options
    def __init__(self):
        self.model = None
        self.options = None

    # prints a message if cuda is used or not
    def __print_model_initialization(self) -> None:
        if is_cuda_available():
            print('Initialize language modelling model using CUDA')
        else:
            print('Initialize language modelling model without CUDA')

    def __t5Args(self) -> T5Args:
        return T5Args(cache_dir=self.cache_dir_name(), output_dir=self.outputs_dir_name(), num_train_epochs=3)

    # initializes a new model
    def init_new_model(self) -> None:
        self.__print_model_initialization()
        #used_model_type, used_model = get_model_config()
        self.model = T5Model('t5', 't5-small', args=self.__t5Args(), use_cuda=is_cuda_available())

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        self.model = T5Model('t5', self.outputs_dir_name(), args=self.__t5Args(), use_cuda=is_cuda_available())

    # https://huggingface.co/blog/how-to-generate
    # https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0
    # https://link.springer.com/chapter/10.1007/978-3-030-91699-2_15
    # https://ieeexplore.ieee.org/document/9402114
    # https://jaketae.github.io/study/keyword-extraction/
    # https://www.lighttag.io/blog/sequence-labeling-with-transformers/
    # github.com/abhimishra91/transformers-tutorials/

    # Trains the model using the given training set
    def train_model(self, training_set: str) -> None:
        self.model.train_model(training_set)

    # Evaluates the model using the given evaluation set
    def eval_model(self, _: list) -> dict:
        raise Exception("Unsupported method: The language generation model has no evaluation method.")
    
    # sets options for the model
    def set_options(self, options: Options) -> None:
        self.options = options

    # Makes predictions
    def predict(self, predictionData: list[MethodContext]) -> list[str]:
        parameters = self.__predict_parameter_list(predictionData)
        returntypes = self.__predict_return_types(predictionData)
        parametertypes = self.__predict_parameter_types(predictionData, parameters)

        paroffset = 0
        result: list[MethodValues] = list()
        for i, _ in enumerate(predictionData):
            value = MethodValues()
            value.returnType = returntypes[i]
            parameter_names = parameters[i].split(',')
            for name in parameter_names:
                parameter = Parameter()
                parameter.name = name
                parameter.type = parametertypes[paroffset]
                value.parameters.append(parameter)
                paroffset += 1
            result.append(value)
        return result
    
    def __predict_parameter_list(self, data: list[MethodContext]) -> list[str]:
        inputs = list()
        for method in data:
            inputs.append(GenerateParametersTask + ': ' + self.__getGenerateParametersInput(method))
        return self.model.predict(inputs)
    
    def __predict_return_types(self, data: list[MethodContext]) -> list[str]:
        inputs = list()
        for method in data:
            inputs.append(AssignReturnTypeTask + ': ' + self.__getGenerateReturnTypeInput(method))
        return self.model.predict(inputs)

    def __predict_parameter_types(self, data: list[MethodContext], parameterLists: list[str]) -> list[str]:
        inputs = list()
        for i, method in enumerate(data):
            parameters = parameterLists[i].split(',')
            for par in parameters:
                inputs.append(AssignParameterTypeTask + ': ' + self.__getGenerateParameterTypeInput(method, par))
        return self.model.predict(inputs)
        

    # path addition for the cacheDir
    def cache_dir_name(self) -> str:
        return self.__parent_dir() + 'cache_dir/'

    # path addition for the outputs dir 
    def outputs_dir_name(self) -> str:
        return self.__parent_dir() +'outputs/'

    # the main directory for this model
    def __parent_dir(self) -> str:
        return 'models/language_generation/'+self.options.identifier+'/'
    
    # Returns a method identifier for the method for caching. Methods with the same identifier won't be predicted again.
    def get_identifier_for_method(self, method: MethodContext) -> str:
        static = ''
        if method.isStatic:
            static = 'static,'
        return static + method.className + "," + method.methodName

    # converts the input data to a pandas frame
    def convert_methods_to_frame(self, data: list[Method]) -> pd.Dataframe:
        frame = pd.DataFrame(columns=['prefix', 'input_text', 'target_text'])
        for method in data:
            self.__addMethodToFrame(method, frame)
        return data
    
    def __addMethodToFrame(self, method: Method, frame: pd.DataFrame):
        self.__addGenerateParametersTask(method, frame)
        self.__addGenerateReturnTypeTask(method, frame)
        self.__addGenerateParameterTypeTask(method, frame)

    def __addGenerateParametersTask(self, method: Method, frame: pd.DataFrame):
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

    def __addGenerateReturnTypeTask(self, method: Method, frame: pd.DataFrame):
        self.__addTask(AssignReturnTypeTask, self.__getGenerateReturnTypeInput(method.context), method.values.returnType, frame)

    def __getGenerateReturnTypeInput(self, context: MethodContext) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '.'

    def __addGenerateParameterTypeTask(self, method: Method, frame: pd.DataFrame):
        for par in method.values.parameters:
            self.__addTask(AssignParameterTypeTask, self.__getGenerateParameterTypeInput(method.context, par.name), par.type, frame)
            
    def __getGenerateParameterTypeInput(self, context: MethodContext, parName: str) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '. parameter: ' + parName

    def __addTask(self, prefix, input_text, target_text, frame: pd.DataFrame):
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