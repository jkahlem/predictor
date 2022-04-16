from filewrapper import FileWrapper
from messages import Adafactor, Options
from methods import Method, MethodContext, MethodValues, Parameter
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import model
import pandas as pd
from config import is_cuda_available
from os.path import exists
import re
from modelConsts import *

class MethodGenerationModelBart(model.Model):
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

    def __seq2seq_args(self) -> Seq2SeqArgs:
        model_options = self.options.model_options
        args = Seq2SeqArgs(cache_dir=self.cache_dir_name(), output_dir=self.outputs_dir_name(), num_train_epochs=1,
            save_steps=5000)

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
        
        self.__set_adafactor_settings_to_args(model_options.adafactor, args)

        return args
    
    def __set_adafactor_settings_to_args(self, adafactor: Adafactor, args: Seq2SeqArgs):
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
        self.model = Seq2SeqModel(encoder_decoder_type="bart", encoder_decoder_name="facebook/bart-base", args=self.__seq2seq_args(), use_cuda=is_cuda_available())
        self.model.encoder_tokenizer.add_tokens([TypeSeparatorToken, ReturnSeparatorToken, ParameterSeparatorToken, ArrayToken])
        self.model.model.resize_token_embeddings(len(self.model.encoder_tokenizer))
        self.model.decoder_tokenizer.add_tokens([TypeSeparatorToken, ReturnSeparatorToken, ParameterSeparatorToken, ArrayToken])

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        self.model = Seq2SeqModel(encoder_decoder_type="bart", encoder_decoder_name=self.outputs_dir_name(), args=self.__seq2seq_args(), use_cuda=is_cuda_available())

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
        # iterate through result. result might be list[str] or list[list[str]] depending on num return sequences. 
        for _, generated_parameters in enumerate(predictions):
            value_suggestions = list()
            suggestions = set()

            # iterate through the predicted sequences.
            for _, parlist in enumerate(generated_parameters):
                if len(predictions) < 50:
                    print(parlist)
                value = MethodValues()
                sentences = parlist.strip().split(ReturnSeparatorToken)
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
        # the sequence should be a parameter list (<type>-<name>, <type>-<name>. returns: <type>.)
        # the parameter list can be "."
        if not self.__is_parameter_list_empty(parlist):
            # if the parameter list is not empty, iterate through the parameter list
            for _, p in enumerate(parlist.split(ParameterSeparatorToken)):
                p = p.split(TypeSeparatorToken, maxsplit=1)
                parameter_type = 'Object'
                parameter_name = p[-1]

                if len(p) == 2:
                    parameter_type = p[0]

                parameter_type = parameter_type.strip()
                    
                if self.options.model_options.use_type_prefixing and parameter_type.startswith(TypePrefix):
                    parameter_type = (parameter_type[len(TypePrefix):]).strip()
                value.add_parameter(parameter_name.replace('.', '').strip(), parameter_type)

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
            inputs.append(self.__get_generate_parameters_input(method))
        return self.__wrap_model_output_in_lists(self.model.predict(inputs))

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
        temp_fd = FileWrapper(is_temp=True)
        for method in data:
            if i > n:
                print("[" + str(i) + "/" + str(len(data)) + "] Converting methods ...")
                n += 1000
            self.__add_method_to_frame(method, temp_fd)
            i += 1
        temp_fd.seek(0)
        frame: pd.DataFrame = pd.read_csv(temp_fd.file_descriptor(), header=None, names=['input_text', 'target_text'], sep=";", na_filter=False)
        temp_fd.close()
        print("Done.")
        return frame

    def __add_method_to_frame(self, method: Method, temp_fd):
        self.__add_generate_parameters_task(method, temp_fd)

    def __add_generate_parameters_task(self, method: Method, temp_fd):
        self.__add_task(self.__get_generate_parameters_input(method.context),
            self.__get_compound_task_output(method.values),
            temp_fd)

    def __get_generate_parameters_input(self, context: MethodContext) -> str:
        static = 'static ' if context.is_static else ''
        compound_task = self.options.model_options.generation_tasks.parameter_names
        if compound_task.with_parameter_types or compound_task.with_return_type:
            return static + context.className + ' ' + context.methodName + ' ' + self.__get_context_parameter(context)
        return static + context.className + ' ' + context.methodName
            #return 'method: ' + static + context.methodName + " . class: " + context.className + ' .' + self.__get_context_parameter(context)
        #return 'method: ' + static + context.methodName + " . class: " + context.className + ' .'

    def __get_context_parameter(self, context: MethodContext) -> str:
        default_context = self.options.model_options.default_context
        if not context.types and not default_context:
            return ""
        context_types = default_context + context.types
        if self.options.model_options.use_type_prefixing:
            context_types = [TypePrefix+x for x in context_types]
        return " context: " + (" " + ParameterSeparatorToken + " ").join(context_types)

    def __get_compound_task_output(self, values: MethodValues) -> str:
        tasks = self.options.model_options.generation_tasks.parameter_names
        output = self.__get_generate_parameters_output(values.parameters, tasks.with_parameter_types)
        if tasks.with_return_type:
            output += " " + ReturnSeparatorToken + " " + values.returnType + ""

        return output

    def __get_generate_parameters_output(self, parameters: list[Parameter], with_types: bool = False) -> str:
        if len(parameters) == 0:
            return 'void' if self.options.model_options.empty_parameter_list_by_keyword else ''
        output = ''
        use_type_prefix = self.options.model_options.use_type_prefixing
        for i, par in enumerate(parameters):
            if i > 0:
                output += " " + ParameterSeparatorToken + " "
            if with_types:
                array_extension = ((ArrayToken + " ") if par.is_array else '')
                output += (TypePrefix if use_type_prefix else "") + par.type + " " + array_extension + TypeSeparatorToken + " "
            output += par.name
        return output

    def __add_task(self, input_text, target_text, temp_fd):
        s: str = input_text + ";<s> " + target_text + "\n"
        temp_fd.write(s)

    # method generation is not cacheable (or rather it makes no sense to cache it) as the input sequences get very complex
    # (due to class name, context types, static state etc.) and it happens very rarely that predictions on the same input are requested.
    def is_cacheable(self) -> None:
        return False
