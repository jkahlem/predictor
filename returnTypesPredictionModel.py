from messages import Options
from methods import Method, MethodContext, MethodValues
import model
import pandas as pd
from util import copyTo
from random import randint
from config import get_labels_path, get_model_config, is_cuda_available, is_test_mode
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import f1_score, accuracy_score
from io import StringIO
from os.path import exists

class ReturnTypesPredictionModel(model.Model):
    model: ClassificationModel
    options: Options
    def __init__(self):
        self.model = None
        self.labels = None
        self.options = None

    # prints a message if cuda is used or not
    def __print_model_initialization(self) -> None:
        if is_cuda_available():
            print('Initialize model with ' + str(self.labels.index.size) + ' label types and using CUDA')
        else:
            print('Initialize model with ' + str(self.labels.index.size) + ' label types without CUDA')

    def exists(self) -> bool:
        return exists(self.outputs_dir_name())

    # the arguments to use in this model
    def __args(self) -> ClassificationArgs:
        model_options = self.options.model_options
        args = ClassificationArgs(cache_dir=self.cache_dir_name(), output_dir=self.outputs_dir_name())
        if model_options.num_of_epochs > 0:
            args.num_train_epochs = model_options.num_of_epochs
        if model_options.batch_size > 0:
            args.train_batch_size = model_options.batch_size
            args.eval_batch_size = model_options.batch_size
        return args
    
    # initializes a new classification model
    def init_new_model(self) -> None:
        if self.labels is None:
            return
        self.__print_model_initialization()
        used_model_type, used_model = get_model_config()
        self.model = ClassificationModel(used_model_type, used_model, num_labels=self.labels.index.size, use_cuda=is_cuda_available(), args=self.__args())

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        self.__load_labels(get_labels_path())
        if not self.labels is None:
            self.__load_model()
    
    # loads a previously trained model
    def __load_model(self) -> None:
        self.__print_model_initialization()
        used_model_type, _ = get_model_config()
        self.model = ClassificationModel(used_model_type, self.outputs_dir_name(), num_labels=self.labels.index.size, use_cuda=is_cuda_available(), args=self.__args())

    # Loads labels
    def load_labels(self, labels) -> None:
        if not is_test_mode():
            copyTo(labels, get_labels_path())
        self.__load_labels(labels)
    
    def set_options(self, options: Options) -> None:
        self.options = options
        self.load_labels(self, StringIO(options.labels))

    # loads labels from a csv file
    def __load_labels(self, filepath_or_buffer) -> None:
        self.labels = pd.read_csv(filepath_or_buffer, header=None, sep=';', na_filter=False)

    # Trains the model using the given training set
    def train_model(self, training_set: pd.DataFrame) -> None:
        if self.model is None:
            return

        self.model.train_model(training_set)

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set: pd.DataFrame) -> dict:
        if self.model is None:
            return dict()
        def f1_multiclass(l, preds):
            return f1_score(l, preds, average='micro')
        result, _, _ = self.model.eval_model(evaluation_set, f1=f1_multiclass, acc=accuracy_score)
        return result

    # Makes predictions for the expected return type of each of the given method names (uses cached values if exist)
    def predict(self, methods: list[MethodContext]) -> list[MethodValues]:
        if is_test_mode():
            # In test mode, return a list of random values
            types = list()
            for _ in methods:
                types.append(self.__get_type_by_label(randint(0, 1)))
            return types

        if self.model is None:
            return list()
 
        inputs: list[str] = list()
        for method in methods:
            inputs.append(method.methodName)

        predictions, _ = self.model.predict(inputs)

        predicted_types: list[MethodValues] = list()
        for p in predictions:
            value = MethodValues()
            value.returnType = self.__get_type_by_label(p)
            predicted_types.append(value)
        return predicted_types

    # returns the type name for the given label
    def __get_type_by_label(self, label: int) -> str:
        if is_test_mode():
            # In test mode, return predefined values
            if label == 0:
                return "object"
            return "void"

        if self.labels is None:
            return ""

        row_for_label = self.labels.loc[self.labels.iloc[:, 1] == label]
        return str(row_for_label.iloc[0, 0])
    
    # returns the type name for the given label
    def __get_label_by_type(self, type: str) -> str:
        if self.labels is None:
            return ""

        row_for_label = self.labels.loc[self.labels.iloc[:, 0] == type]
        return str(row_for_label.iloc[0, 1])

    # path addition for the cacheDir
    def cache_dir_name(self) -> str:
        return self.__parent_dir() + 'cache_dir/'

    # path addition for the outputs dir 
    def outputs_dir_name(self) -> str:
        return self.__parent_dir() +'outputs/'

    # the main directory for this model
    def __parent_dir(self) -> str:
        return 'models/returntypes/'+self.options.identifier+'/'

    # converts the input data to a pandas frame
    def convert_methods_to_frame(self, data: list[Method]) -> pd.DataFrame:
        frame = pd.DataFrame(columns=['text', 'labels'])
        for method in data:
            frame.append(pd.Series({'text': method.context.methodName, 'labels': self.__get_label_by_type(method.values.returnType)}))
        return data
    
    # returns a method identifier for the method for caching. Methods with the same identifier won't be predicted again.
    def get_identifier_for_method(self, method: MethodContext) -> str:
        return method.methodName
