from sqlite3 import paramstyle
import tempfile
import typing
from messages import MethodGenerationTaskOptions, Options
from methods import Method, MethodContext, MethodValues, Parameter
from simpletransformers.t5 import T5Model, T5Args
import model
import pandas as pd
from config import is_cuda_available
from os.path import exists

GenerateParametersTask = 'generate parameters'
AssignReturnTypeTask = 'assign returntype'
AssignParameterTypeTask = 'assign parametertype'
TypePrefix = 'type_'

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
        parameters = self.__predict_parameter_list(prediction_data)

        # predict return types in separate task if configured
        return_types = list()
        if self.__generation_tasks().return_type:
            return_types = self.__predict_return_types(prediction_data)

        # predict parameter types in separate task if configured
        parameter_types = list()
        if self.__generation_tasks().parameter_types:
            parameter_types = self.__predict_parameter_types(prediction_data, parameters)

        return self.__map_predictions_to_method_values(parameters, return_types, parameter_types)
    
    def __map_predictions_to_method_values(self, predictions: list[list[str]], return_types: list[list[str]], parameter_types: list[list[list[str]]]) -> list[list[MethodValues]]:
        results = list()
        # iterate through result. result might be list[str] or list[list[str]] depending on num return sequences. 
        for i, generated_parameters in enumerate(predictions):
            value_suggestions = list()
            suggestions = set()

            # iterate through the predicted sequences.
            for j, parlist in enumerate(generated_parameters):
                value = MethodValues()
                sentences = parlist.strip().split('. returns:')
                par_types = parameter_types[i][j] if len(parameter_types) > 0 else None
                self.__add_parameters_to_method_values(value, sentences[0], par_types)

                # if return types are predicted in a separate task
                if len(return_types) == len(predictions):
                    value.set_return_type(return_types[i][j])
                # else if return type prediction is a compound task, then pick the second sentence for it.
                elif len(sentences) == 2 and self.__generation_tasks().parameter_names.with_return_type:
                    value.set_return_type(sentences[1])

                # get the hash for the current state to prevent adding the same suggestions multiple times
                value_hash = value.current_state_hash()
                if not value_hash in suggestions:
                    value_suggestions.append(value)
                    suggestions.add(value_hash)

            results.append(value_suggestions)
        return results
    
    def __add_parameters_to_method_values(self, value: MethodValues, parlist: str, types: list[str]) -> MethodValues:
        # the sequence should be a parameter list (<type>-<name>, <type>-<name>. returns: <type>.)
        # the parameter list can be "void."
        if not (parlist == 'void' or parlist == 'void.'):
            # if the parameter list is not empty, iterate through the parameter list
            for i, p in enumerate(parlist.split(',')):
                p = p.split('-', maxsplit=1)
                parameter_type = 'Object'
                parameter_name = p[-1]

                # if types are predicted in a separate task, use that type
                if types is not None and len(types) > 0:
                    parameter_type = types[i]
                # otherwise if the parameter can be splitted into two values, the first value is the type.
                elif len(p) == 2:
                    parameter_type = p[0]

                value.add_parameter(parameter_name.strip(), parameter_type.strip())

    # wraps the model output in list, even if only one suggestion exists to make the code easier  
    def __wrap_model_output_in_lists(predictions: list) -> list[list[str]]:
        for i in enumerate(predictions):
            if not isinstance(predictions[i], list):
                predictions[i] = [predictions[i]]
        return predictions
    
    def __predict_parameter_list(self, data: list[MethodContext]) -> list[list[str]]:
        inputs = list()
        for method in data:
            inputs.append(GenerateParametersTask + ': ' + self.__getGenerateParametersInput(method))
        return self.__wrap_model_output_in_lists(self.model.predict(inputs))
    
    def __predict_return_types(self, data: list[MethodContext]) -> list[list[str]]:
        inputs = list()
        for method in data:
            inputs.append(AssignReturnTypeTask + ': ' + self.__get_generate_return_type_input(method))
        return self.__wrap_model_output_in_lists(self.model.predict(inputs))

    # parameter_names can be a list of single predictions or a list of suggestion (multiple predictions per input)
    # the return value is structured as: list[MethodIndex][SuggestionIndex][ParameterIndex]
    # therefore list[x][y][z] returns the type for the z-th parameter in the y-th suggestion for the x-th method.
    def __predict_parameter_types(self, data: list[MethodContext], parameter_names: list[list[str]]) -> list[list[list[str]]]:
        inputs = list()
        for i, method in enumerate(data):
            parameter_suggestions = parameter_names[i]
            for suggestion in parameter_suggestions:
                parameters = suggestion.split(',')
                for par in parameters:
                    inputs.append(AssignParameterTypeTask + ': ' + self.__get_generate_parameter_type_input(method, par))
        
        # to make things not more complicated, parameter type predictions should have only one suggestion
        t = self.model.args.num_return_sequences
        self.model.args.num_return_sequences = 1
        predictions = self.model.predict(inputs)
        self.model.args.num_return_sequences = t

        # because the prediction input and output is a flattened list of strings (parameter types), we need to map
        # each parameter type to it's name. The output (parameter names) per input (method name) can also consist
        # of multiple suggestions
        outputs = list()
        offset = 0
        for i in enumerate(data):
            suggestions = list()
            # iterate through each suggestion and increment the offset
            for _ in enumerate(parameter_names[i]):
                types_per_suggestion = list()
                for _ in enumerate(parameter_names[i].split(',')):
                    types_per_suggestion.append(predictions[offset])
                    offset = offset + 1
                suggestions.append(types_per_suggestion)
            outputs.append(suggestions)

        return predictions
    
    def __prefix(self, task_type: str, context: MethodContext) -> str:
        if context.is_static:
            return task_type + " static"
        return task_type

    # path addition for the cacheDir
    def cache_dir_name(self) -> str:
        return self.__parent_dir() + 'cache_dir/'

    # path addition for the outputs dir 
    def outputs_dir_name(self) -> str:
        return self.__parent_dir() + 'outputs/'

    # the main directory for this model
    def __parent_dir(self) -> str:
        return 'models/language_generation/'+self.options.identifier+'/'

    # converts the input data to a pandas frame
    def convert_methods_to_frame(self, data: list[Method]) -> pd.DataFrame:
        print("Convert list of " + str(len(data)) + " methods to frame ...")
        i, n = 0, 1000
        temp_fd = tempfile.TemporaryFile()
        for method in data:
            if i > n:
                print("[" + str(i) + "/" + str(len(data)) + "] Converting methods ...")
                n += 1000
            self.__add_method_to_frame(method, temp_fd)
            i += 1
        temp_fd.seek(0)
        frame: pd.DataFrame = pd.read_csv(temp_fd, header=None, names=['prefix', 'input_text', 'target_text'], sep=";", na_filter=False)
        temp_fd.close()
        print("Done.")
        return frame
    
    def __add_method_to_frame(self, method: Method, temp_fd):
        tasks = self.options.model_options.generation_tasks
        self.__add_generate_parameters_task(method, temp_fd)
        if tasks.return_type:
            self.__add_generate_return_type_task(method, temp_fd)
        if tasks.parameter_types:
            self.__add_generate_parameter_type_task(method, temp_fd)

    def __add_generate_parameters_task(self, method: Method, temp_fd):
        self.__add_task(self.__prefix(GenerateParametersTask, method.context), self.__getGenerateParametersInput(method.context), self.__get_compound_task_output(method.values), temp_fd)
    
    def __getGenerateParametersInput(self, context: MethodContext) -> str:
        return 'method: ' + context.methodName + ". class: " + context.className + '.' + self.__get_context_parameter(context)
    
    def __get_context_parameter(self, context: MethodContext) -> str:
        default_context = self.options.model_options.default_context
        if not context.types and not default_context:
            return ""
        context_types = default_context + context.types
        if self.options.model_options.use_type_prefixing:
            context_types = [TypePrefix+x for x in context_types]
        return " context: " + ", ".join(context_types) + "."
    
    def __get_compound_task_output(self, values: MethodValues) -> str:
        tasks = self.options.model_options.generation_tasks.parameter_names
        output = self.__get_generate_parameters_output(values.parameters, tasks.with_parameter_types)
        if tasks.with_return_type:
            output += " returns: " + values.returnType + "."

        return output

    def __get_generate_parameters_output(self, parameters: list[Parameter], with_types: bool = False) -> str:
        if len(parameters) == 0:
            return 'void.'
        output = ''
        use_type_prefix = self.options.model_options.use_type_prefixing
        for i, par in enumerate(parameters):
            if i > 0:
                output += ", "
            if with_types:
                output += (TypePrefix if use_type_prefix else "") + par.type +  ("[] - " if par.is_array else " - ")
            output += par.name
        return output + "."

    def __add_generate_return_type_task(self, method: Method, temp_fd):
        self.__add_task(self.__prefix(AssignReturnTypeTask, method.context), self.__get_generate_return_type_input(method.context), method.values.returnType, temp_fd)

    def __get_generate_return_type_input(self, context: MethodContext) -> str:
        return 'method: ' + context.methodName + ". class: " + context.className + '.'

    def __add_generate_parameter_type_task(self, method: Method, temp_fd):
        for par in method.values.parameters:
            self.__add_task(self.__prefix(AssignParameterTypeTask, method.context), self.__get_generate_parameter_type_input(method.context, par.name), par.type, temp_fd)

    def __get_generate_parameter_type_input(self, context: MethodContext, parName: str) -> str:
        return 'method: ' + context.methodName + ". class: " + context.className + '. parameter: ' + parName

    def __add_task(self, prefix, input_text, target_text, temp_fd):
        s: str = prefix + ";" + input_text + ";" + target_text + "\n"
        temp_fd.write(s.encode('utf-8'))
    
    def __generation_tasks(self) -> MethodGenerationTaskOptions:
        return self.options.model_options.generation_tasks

    # method generation is not cacheable (or rather it makes no sense to cache it) as the input sequences get very complex
    # (due to class name, context types, static state etc.) and it happens very rarely that predictions on the same input are requested.
    def is_cacheable(self) -> None:
        return False
