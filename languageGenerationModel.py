from sqlite3 import paramstyle
import tempfile
import typing
from messages import Options
from methods import Method, MethodContext, MethodValues, Parameter
from simpletransformers.t5 import T5Model, T5Args
import model
import pandas as pd
from config import is_cuda_available
from os.path import exists

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
        model_options = self.options.model_options
        args = T5Args(cache_dir=self.cache_dir_name(), output_dir=self.outputs_dir_name(), num_train_epochs=1)
        if model_options.num_of_epochs > 0:
            args.num_train_epochs = model_options.num_of_epochs
        if model_options.batch_size > 0:
            args.train_batch_size = model_options.batch_size
            args.eval_batch_size = model_options.batch_size
        if model_options.max_sequence_length > 0:
            args.max_seq_length = model_options.max_sequence_length
        if model_options.num_return_sequences > 1:
            args.num_return_sequences = model_options.num_return_sequences
            args.do_sample = True
        return args
    
    def exists(self) -> bool:
        return exists(self.outputs_dir_name())

    # initializes a new model
    def init_new_model(self) -> None:
        self.__print_model_initialization()
        #used_model_type, used_model = get_model_config()
        self.model = T5Model('t5', 't5-small', args=self.__t5Args(), use_cuda=is_cuda_available())

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        self.model = T5Model('t5', self.outputs_dir_name(), args=self.__t5Args(), use_cuda=is_cuda_available())

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
    def predict(self, prediction_data: list[MethodContext]) -> list[list[MethodValues]]:
        tasks = self.options.model_options.generation_tasks
        results = list()
        parameters = self.__predict_parameter_list(prediction_data)
        return_types = list()
        if tasks.return_type:
            return_types = self.__predict_return_types(prediction_data)
        for i, generated_parameters in enumerate(parameters):
            values = list()
            if not isinstance(generated_parameters, list):
                generated_parameters = [generated_parameters]
            for j, parlist in enumerate(generated_parameters):
                value = MethodValues()
                sentences = parlist.strip().split('. returns')
                if not (sentences[0] == 'void' or sentences[0] == 'void.'):
                    for p in sentences[0].split(','):
                        p = p.strip().split(' ', maxsplit=1)
                        parameter_type = 'any'
                        parameter_name = p[-1]
                        if len(p) == 2:
                            parameter_type = p[0]

                        value.add_parameter(parameter_name, parameter_type)
                if len(return_types) == len(parameters):
                    if isinstance(return_types[i], list):
                        value.returnType = return_types[i][j]
                    else:
                        value.returnType = return_types[i]
                elif len(sentences) == 2 and tasks.parameter_names.with_return_type:
                    value.returnType = sentences[1]
                values.append(value)
            results.append(values)
        return results
    
    def __predict_parameter_list(self, data: list[MethodContext]) -> list:
        inputs = list()
        for method in data:
            inputs.append(GenerateParametersTask + ': ' + self.__getGenerateParametersInput(method))
        return self.model.predict(inputs)
    
    def __predict_return_types(self, data: list[MethodContext]) -> list:
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
    def convert_methods_to_frame(self, data: list[Method]) -> pd.DataFrame:
        print("Convert list of " + str(len(data)) + " methods to frame ...")
        i, n = 0, 1000
        temp_fd = tempfile.TemporaryFile()
        for method in data:
            if i > n:
                print("[" + str(i) + "/" + str(len(data)) + "] Converting methods ...")
                n += 1000
            self.__addMethodToFrame(method, temp_fd)
            i += 1
        temp_fd.seek(0)
        frame: pd.DataFrame = pd.read_csv(temp_fd, header=None, names=['prefix', 'input_text', 'target_text'], sep=";", na_filter=False)
        temp_fd.close()
        print("Done.")
        return frame
    
    def __addMethodToFrame(self, method: Method, temp_fd):
        tasks = self.options.model_options.generation_tasks
        self.__addGenerateParametersTask(method, temp_fd)
        if tasks.return_type:
            self.__addGenerateReturnTypeTask(method, temp_fd)
        if tasks.parameter_types:
            self.__addGenerateParameterTypeTask(method, temp_fd)

    def __addGenerateParametersTask(self, method: Method, temp_fd):
        self.__addTask(GenerateParametersTask, self.__getGenerateParametersInput(method.context), self.__getCompoundTaskOutput(method.values), temp_fd)
    
    def __getGenerateParametersInput(self, context: MethodContext) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '.'
    
    def __getCompoundTaskOutput(self, values: MethodValues) -> str:
        tasks = self.options.model_options.generation_tasks.parameter_names
        output = self.__getGenerateParametersOutput(values.parameters, tasks.with_parameter_types)
        if tasks.with_return_type:
            output += " returns " + values.returnType + "."

        return output

    def __getGenerateParametersOutput(self, parameters: list[Parameter], with_types: bool = False) -> str:
        if len(parameters) == 0:
            return 'void.'
        output = ''
        for i, par in enumerate(parameters):
            if i > 0:
                output += ", "
            if with_types:
                output += par.type + " "
            output += par.name
        return output + "."

    def __addGenerateReturnTypeTask(self, method: Method, temp_fd):
        self.__addTask(AssignReturnTypeTask, self.__getGenerateReturnTypeInput(method.context), method.values.returnType, temp_fd)

    def __getGenerateReturnTypeInput(self, context: MethodContext) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '.'

    def __addGenerateParameterTypeTask(self, method: Method, temp_fd):
        for par in method.values.parameters:
            self.__addTask(AssignParameterTypeTask, self.__getGenerateParameterTypeInput(method.context, par.name), par.type, temp_fd)

    def __getGenerateParameterTypeInput(self, context: MethodContext, parName: str) -> str:
        static = ''
        if context.isStatic:
            static = 'static '
        return 'method: ' + static + context.methodName + ". class: " + context.className + '. parameter: ' + parName

    def __addTask(self, prefix, input_text, target_text, temp_fd):
        s: str = prefix + ";" + input_text + ";" + target_text + "\n"
        temp_fd.write(s.encode('utf-8'))
