import os
from filewrapper import FileWrapper
from messages import Adafactor, Options
from methods import Method, MethodContext, MethodValues, Parameter
from simpletransformers.t5 import T5Model, T5Args
import model
import pandas as pd
from config import is_cuda_available, multiprocessed_decoding, num_save_steps, num_workers
from os.path import exists
import re
import json

from modelConsts import ArrayToken, EmbeddedClassSeparator, EmbeddedParameterSeparator, EmbeddedReturnSeparator, EmbeddedTypeSeparator, ParameterSeparatorToken, ReturnSeparatorToken, SentenceFormattingOptionsFile, TypeSeparatorToken

GenerateParametersTask = 'generate parameters'

class MethodGenerationModel(model.Model):
    options: Options
    def __init__(self):
        self.model = None
        self.options = None

    # prints a message if cuda is used or not
    def __print_model_initialization(self) -> None:
        if is_cuda_available():
            print('Initialize T5-based method generation model using CUDA')
        else:
            print('Initialize T5-based method generation model without CUDA')

    def __t5Args(self) -> T5Args:
        model_options = self.options.model_options
        args = T5Args(cache_dir=self.cache_dir_name(), output_dir=self.outputs_dir_name(), num_train_epochs=1,
            dataloader_num_workers=num_workers(),
            save_steps=num_save_steps(),
            use_multiprocessed_decoding=multiprocessed_decoding())

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
        self.model = T5Model('t5',
            self.options.model_options.model_name,
            args=self.__t5Args(),
            use_cuda=is_cuda_available())
        self.model.tokenizer.add_tokens([TypeSeparatorToken, ReturnSeparatorToken, ParameterSeparatorToken, ArrayToken])
        self.model.model.resize_token_embeddings(len(self.model.tokenizer))
        self.save_sentence_formatting_options()

    def save_sentence_formatting_options(self) -> None:
        if self.options.sentence_formatting_options is not None:
            os.makedirs(os.path.dirname(self.sentence_formatting_options_path()), exist_ok=True)
            with open(self.sentence_formatting_options_path(), 'w') as file:
                json.dump(self.options.sentence_formatting_options, file)

    def sentence_formatting_options_path(self) -> str:
        return self.__parent_dir() + SentenceFormattingOptionsFile

    # Loads an already created/trained classification model
    def load_model(self, for_continuation: bool = False) -> None:
        outputs_dir = self.outputs_dir_name()
        if for_continuation:
            last_epoch = model.get_last_epoch_path(outputs_dir)
            if last_epoch is not None:
                outputs_dir = last_epoch

        if for_continuation:
            T5Model('t5', outputs_dir, args=dict(overwrite_output_dir = True), use_cuda=is_cuda_available())
        else:
            self.model = T5Model('t5', outputs_dir, use_cuda=is_cuda_available())

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
        return self.__map_predictions_to_method_values(parameters)

    def __map_predictions_to_method_values(self, predictions: list[list[str]]) -> list[list[MethodValues]]:
        results = list()
        order = self.options.model_options.output_order

        # iterate through result. result might be list[str] or list[list[str]] depending on num return sequences. 
        for _, generated_parameters in enumerate(predictions):
            value_suggestions = list()
            suggestions = set()

            # iterate through the predicted sequences.
            for _, parlist in enumerate(generated_parameters):
                value = MethodValues()
                sentences = parlist.strip().split(ReturnSeparatorToken)

                if len(sentences) == 2:
                    return_type = sentences[0 if order.return_type < order.parameter_name else 1]
                    parameter_list = sentences[0 if order.return_type > order.parameter_name else 1]
                    self.__add_parameters_to_method_values(value, parameter_list)
                    value.set_return_type(return_type)
                else:
                    self.__add_parameters_to_method_values(value, sentences[0])
                    if len(sentences) == 2:
                        value.set_return_type(sentences[1])

                # get the hash for the current state to prevent adding the same suggestions multiple times
                value_hash = value.current_state_hash()
                if not value_hash in suggestions:
                    value_suggestions.append(value)
                    suggestions.add(value_hash)

            results.append(value_suggestions)
        return results

    def __add_parameters_to_method_values(self, value: MethodValues, parlist: str) -> MethodValues:
        order = self.options.model_options.output_order
        # the sequence should be a parameter list (<type>-<name>, <type>-<name>. returns: <type>.)
        # the parameter list can be "."
        if not self.__is_parameter_list_empty(parlist):
            # if the parameter list is not empty, iterate through the parameter list
            for _, p in enumerate(parlist.split(ParameterSeparatorToken)):
                p = p.split(TypeSeparatorToken, maxsplit=1)
                parameter_type, parameter_name = 'Object', ''

                if len(p) == 2:
                    parameter_name = p[0 if order.parameter_name < order.parameter_type else 1]
                    parameter_type = p[0 if order.parameter_name > order.parameter_type else 1]

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

    def __prefix(self, task_type: str, context: MethodContext) -> str:
        if context.is_static:
            return task_type + " static"
        return task_type

    # path addition for the cacheDir
    def cache_dir_name(self) -> str:
        return self.__parent_dir() + 'cache_dir/'

    # path addition for the outputs dir 
    def outputs_dir_name(self) -> str:
        return self.__parent_dir() + 'outputs/' +  (self.options.checkpoint + '/' if len(self.options.checkpoint) > 0 else '')

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
        frame: pd.DataFrame = pd.read_csv(temp_fd.file_descriptor(), header=None, names=['prefix', 'input_text', 'target_text'], sep=";", na_filter=False)
        temp_fd.close()
        print("Done.")
        return frame

    def __add_method_to_frame(self, method: Method, temp_fd):
        self.__add_generate_parameters_task(method, temp_fd)

    def __add_generate_parameters_task(self, method: Method, temp_fd):
        self.__add_task(self.__prefix(GenerateParametersTask, method.context),
            self.__get_generate_parameters_input(method.context),
            self.__get_compound_task_output(method.values),
            temp_fd)

    def __get_generate_parameters_input(self, context: MethodContext) -> str:
        return 'method: ' + context.methodName + " . class: " + self.__join_classes(context.className) + ' .' + self.__get_context_parameter(context)

    def __get_context_parameter(self, context: MethodContext) -> str:
        default_context = self.options.model_options.default_context
        if not context.types and not default_context:
            return ""
        context_types = default_context + context.types
        return " context: " + EmbeddedParameterSeparator.join(context_types)

    def __get_compound_task_output(self, values: MethodValues) -> str:
        parameter_list = self.__get_generate_parameters_output(values.parameters)
        order = self.options.model_options.output_order
        if order.return_type < order.parameter_name:
            return values.returnType + EmbeddedReturnSeparator + parameter_list
        else:
            return parameter_list + EmbeddedReturnSeparator + values.returnType

    def __get_generate_parameters_output(self, parameters: list[Parameter]) -> str:
        if len(parameters) == 0:
            return 'void .' if self.options.model_options.empty_parameter_list_by_keyword else '.'
        output = ''
        order = self.options.model_options.output_order

        for i, par in enumerate(parameters):
            if i > 0:
                output += EmbeddedParameterSeparator

            typestring = par.type + (" " + ArrayToken if par.is_array else "")
            if order.parameter_name < order.parameter_type:
                output += par.name + EmbeddedTypeSeparator + typestring
            else:
                output += typestring + EmbeddedTypeSeparator + par.name
        return output + " ."

    def __add_task(self, prefix, input_text, target_text, temp_fd):
        s: str = prefix + ";" + input_text + ";" + target_text + "\n"
        temp_fd.write(s)

    def __join_classes(self, classes: list[str]) -> str:
        return EmbeddedClassSeparator.join(classes)

    # method generation is not cacheable (or rather it makes no sense to cache it) as the input sequences get very complex
    # (due to class name, context types, static state etc.) and it happens very rarely that predictions on the same input are requested.
    def is_cacheable(self) -> None:
        return False
