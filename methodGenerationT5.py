import tempfile
from filewrapper import FileWrapper
from messages import Adafactor, MethodGenerationTaskOptions, Options
from methods import Method, MethodContext, MethodValues, Parameter
from simpletransformers.t5 import T5Model, T5Args
import model
import pandas as pd
from config import is_cuda_available
from os.path import exists
import re

from modelConsts import ArrayToken, EmbeddedParameterSeparator, EmbeddedReturnSeparator, EmbeddedTypeSeparator, ParameterSeparatorToken, ReturnSeparatorToken, TypeSeparatorToken

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
        args = T5Args(cache_dir=self.cache_dir_name(), output_dir=self.outputs_dir_name(), num_train_epochs=1, save_steps=5000)

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

        if model_options.num_beams is not None and model_options.num_beams > 1:
            args.num_beams = model_options.num_beams

        if model_options.top_k is not None:
            args.top_k = model_options.top_k
            args.do_sample = True

        if model_options.top_p is not None:
            args.top_p = model_options.top_p
            args.do_sample = True

        if model_options.length_penalty is not None:
            args.length_penalty = model_options.length_penalty

        self.__set_adafactor_settings_to_args(model_options.adafactor, args)

        return args

    def __set_adafactor_settings_to_args(self, adafactor: Adafactor, args: T5Args):
        if adafactor.beta is not None:
            args.adafactor_beta1 = adafactor.beta
        if adafactor.clip_threshold is not None:
            args.adafactor_clip_threshold = adafactor.clip_threshold
        if adafactor.decay_rate is not None:
            args.adafactor_decay_rate = adafactor.decay_rate
        if adafactor.eps is not None:
            args.adafactor_eps = adafactor.eps
        if adafactor.scale_parameter is not None:
            args.adafactor_scale_parameter = adafactor.scale_parameter
        if adafactor.relative_step is not None:
            args.adafactor_relative_step = adafactor.relative_step
        if adafactor.warmup_init is not None:
            args.adafactor_warmup_init = adafactor.warmup_init

    def exists(self) -> bool:
        return exists(self.outputs_dir_name())

    # initializes a new model
    def init_new_model(self) -> None:
        self.__print_model_initialization()
        #used_model_type, used_model = get_model_config()
        self.model = T5Model('t5',
            self.options.model_options.model_name,
            args=self.__t5Args(),
            use_cuda=is_cuda_available())
        self.model.tokenizer.add_tokens([TypeSeparatorToken, ReturnSeparatorToken, ParameterSeparatorToken, ArrayToken])
        self.model.model.resize_token_embeddings(len(self.model.tokenizer))

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        self.model = T5Model('t5', self.outputs_dir_name(), args=self.__t5Args(), use_cuda=is_cuda_available())

    # Trains the model using the given training set
    def train_model(self, training_set: str) -> None:
        self.model.train_model(training_set)

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set: str) -> dict:
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

    def __map_predictions_to_method_values(self, predictions: list[list[str]],
        return_types: list[list[str]],
        parameter_types: list[list[list[str]]]) -> list[list[MethodValues]]:
        results = list()
        order = self.options.model_options.output_order

        # iterate through result. result might be list[str] or list[list[str]] depending on num return sequences. 
        for i, generated_parameters in enumerate(predictions):
            value_suggestions = list()
            suggestions = set()

            # iterate through the predicted sequences.
            for j, parlist in enumerate(generated_parameters):
                value = MethodValues()
                sentences = parlist.strip().split(ReturnSeparatorToken)

                if len(sentences) == 2:
                    # TODO: this does not work with non-compound tasks. Maybe, remove even non-compound tasks?
                    return_type = sentences[0 if order.return_type < order.parameter_name else 1]
                    parameter_list = sentences[0 if order.return_type > order.parameter_name else 1]
                    self.__add_parameters_to_method_values(value, parameter_list, None)
                    value.set_return_type(return_type)
                else:
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
        order = self.options.model_options.output_order
        # the sequence should be a parameter list (<type>-<name>, <type>-<name>. returns: <type>.)
        # the parameter list can be "."
        if not self.__is_parameter_list_empty(parlist):
            # if the parameter list is not empty, iterate through the parameter list
            for i, p in enumerate(parlist.split(ParameterSeparatorToken)):
                p = p.split(TypeSeparatorToken, maxsplit=1)
                parameter_type, parameter_name = 'Object', ''

                if len(p) == 2:
                    parameter_name = p[0 if order.parameter_name < order.parameter_type else 1]
                    parameter_type = p[0 if order.parameter_name > order.parameter_type else 1]
                elif types is not None and len(types) > 0:
                    # if types are predicted in a separate task, use that type
                    parameter_type = types[i]

                parameter_type = parameter_type.strip()
                value.add_parameter(parameter_name.replace('.', '').strip(), parameter_type.replace('.', '').strip())

    def __is_parameter_list_empty(self, parlist: str) -> bool:
        if self.options.model_options.empty_parameter_list_by_keyword:
            return parlist.strip() in {'void', 'void.'}
        else:
            return not re.search('[a-zA-Z]', parlist)

    # wraps the model output in list, even if only one suggestion exists to make the co[e easier 
    def __wrap_model_output_in_lists(self, predictions: list) -> list[list[str]]:
        for i, _ in enumerate(predictions):
            if not isinstance(predictions[i], list):
                predictions[i] = [predictions[i]]
        return predictions

    def __predict_parameter_list(self, data: list[MethodContext]) -> list[list[str]]:
        inputs = list()
        for method in data:
            inputs.append(self.__prefix(GenerateParametersTask, method) + ': ' + self.__get_generate_parameters_input(method))
        return self.__wrap_model_output_in_lists(self.model.predict(inputs))

    def __predict_return_types(self, data: list[MethodContext]) -> list[list[str]]:
        inputs = list()
        for method in data:
            inputs.append(self.__prefix(AssignReturnTypeTask, method) + ': ' + self.__get_generate_return_type_input(method))
        return self.__wrap_model_output_in_lists(self.model.predict(inputs))

    # parameter_names can be a list of single predictions or a list of suggestion (multiple predictions per input)
    # the return value is structured as: list[MethodIndex][SuggestionIndex][ParameterIndex]
    # therefore list[x][y][z] returns the type for the z-th parameter in the y-th suggestion for the x-th method.
    def __predict_parameter_types(self, data: list[MethodContext], parameter_names: list[list[str]]) -> list[list[list[str]]]:
        inputs = list()
        for i, method in enumerate(data):
            parameter_suggestions = parameter_names[i]
            for suggestion in parameter_suggestions:
                parameters = suggestion.split(ParameterSeparatorToken)
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
        for i, _ in enumerate(data):
            suggestions = list()
            # iterate through each suggestion and increment the offset
            for _ in enumerate(parameter_names[i]):
                types_per_suggestion = list()
                for _ in enumerate(parameter_names[i].split(ParameterSeparatorToken)):
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
        temp_fd = FileWrapper(is_temp=False)
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
        self.__add_task(self.__prefix(GenerateParametersTask, method.context),
            self.__get_generate_parameters_input(method.context),
            self.__get_compound_task_output(method.values),
            temp_fd)

    def __get_generate_parameters_input(self, context: MethodContext) -> str:
        compound_task = self.options.model_options.generation_tasks.parameter_names
        if compound_task.with_parameter_types or compound_task.with_return_type:
            return 'method: ' + context.methodName + " . class: " + context.className + ' .' + self.__get_context_parameter(context)
        return 'method: ' + context.methodName + " . class: " + context.className + ' .'

    def __get_context_parameter(self, context: MethodContext) -> str:
        default_context = self.options.model_options.default_context
        if not context.types and not default_context:
            return ""
        context_types = default_context + context.types
        return " context: " + EmbeddedParameterSeparator.join(context_types)

    def __get_compound_task_output(self, values: MethodValues) -> str:
        tasks = self.options.model_options.generation_tasks.parameter_names
        parameter_list = self.__get_generate_parameters_output(values.parameters, tasks.with_parameter_types)
        if not tasks.with_return_type:
            return parameter_list

        order = self.options.model_options.output_order
        if order.return_type < order.parameter_name:
            return values.returnType + EmbeddedReturnSeparator + parameter_list
        else:
            return parameter_list + EmbeddedReturnSeparator + values.returnType

    def __get_generate_parameters_output(self, parameters: list[Parameter], with_types: bool = False) -> str:
        if len(parameters) == 0:
            return 'void .' if self.options.model_options.empty_parameter_list_by_keyword else '.'
        output = ''
        order = self.options.model_options.output_order

        for i, par in enumerate(parameters):
            if i > 0:
                output += EmbeddedParameterSeparator
            if not with_types:
                output += par.name
                continue

            typestring = par.type + (" " + ArrayToken if par.is_array else "")
            if order.parameter_name < order.parameter_type:
                output += par.name + EmbeddedTypeSeparator + typestring
            else:
                output += typestring + EmbeddedTypeSeparator + par.name
        return output + " ."

    def __add_generate_return_type_task(self, method: Method, temp_fd):
        self.__add_task(self.__prefix(AssignReturnTypeTask, method.context),
            self.__get_generate_return_type_input(method.context),
            method.values.returnType,
            temp_fd)

    def __get_generate_return_type_input(self, context: MethodContext) -> str:
        return 'method: ' + context.methodName + " . class: " + context.className + ' .' + self.__get_context_parameter(context)

    def __add_generate_parameter_type_task(self, method: Method, temp_fd):
        for par in method.values.parameters:
            self.__add_task(self.__prefix(AssignParameterTypeTask, method.context),
                self.__get_generate_parameter_type_input(method.context, par.name),
                par.type,
                temp_fd)

    def __get_generate_parameter_type_input(self, context: MethodContext, parName: str) -> str:
        return ' parameter: ' + parName + 'method: ' + context.methodName + " . class: " + context.className + ' .' + self.__get_context_parameter(context)

    def __add_task(self, prefix, input_text, target_text, temp_fd):
        s: str = prefix + ";" + input_text + ";" + target_text + "\n"
        temp_fd.write(s.encode('utf-8'))
    
    def __generation_tasks(self) -> MethodGenerationTaskOptions:
        return self.options.model_options.generation_tasks

    # method generation is not cacheable (or rather it makes no sense to cache it) as the input sequences get very complex
    # (due to class name, context types, static state etc.) and it happens very rarely that predictions on the same input are requested.
    def is_cacheable(self) -> None:
        return False
