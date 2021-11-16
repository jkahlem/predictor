from enum import Enum
import os
from random import randint
import shutil
import threading

import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import f1_score, accuracy_score

from config import get_labels_path, get_model_config, get_script_dir, is_cuda_available, is_test_mode

class ModelState(Enum):
    NONE = 1
    INITIALIZED = 3

# Wraps the classification model functionality and is thread safe
class Model:
    def __init__(self):
        self.model_state = ModelState.NONE
        self.model = None
        self.labels = None
        self.mutex = threading.Lock()
        self.prediction_cache = dict()

    # returns true if the model is starting
    def is_starting(self) -> bool:
        return self.model_state != ModelState.NONE

    # prints a message if cuda is used or not
    def __print_model_initialization(self) -> None:
        if is_cuda_available():
            print('Initialize model with ' + str(self.labels.index.size) + ' label types and using CUDA')
        else:
            print('Initialize model with ' + str(self.labels.index.size) + ' label types without CUDA')

    # the arguments to use in this model
    def __args(self) -> ClassificationArgs:
        args = ClassificationArgs()
        return args

    # Creates a new classification model using the loaded labels (if labels are loaded, otherwise nothing happens)
    def create_new_model(self) -> None:
        if is_test_mode():
            # Do nothing in test mode
            return
        self.mutex.acquire()

        if self.is_starting() or self.labels is None:
            self.mutex.release()
            return

        self.__remove_previous_model_files()
        self.__init_new_model()

        self.mutex.release()

    # removes the model files used by a previously trained model    
    def __remove_previous_model_files(self) -> None:
        self.__remove_dir_if_exists(os.path.join(get_script_dir(), 'outputs'))
        self.__remove_dir_if_exists(os.path.join(get_script_dir(), 'cache_dir'))

    # removes a directory failsafe
    def __remove_dir_if_exists(self, path: str) -> None:
        try:
            for root, dirs, files in os.walk(path, topdown=True):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        except Exception:
            pass
    
    # initializes a new classification model
    def __init_new_model(self) -> None:
        self.__print_model_initialization()
        used_model_type, used_model = get_model_config()
        self.model = ClassificationModel(used_model_type, used_model, num_labels=self.labels.index.size, use_cuda=is_cuda_available(), args=self.__args())
        self.model_state = ModelState.INITIALIZED

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        if is_test_mode():
            # Do nothing in test mode
            return
        self.mutex.acquire()

        if self.is_starting():
            self.mutex.release()
            return

        self.__load_labels(get_labels_path())
        if not self.labels is None:
            self.__load_model()
    
        self.mutex.release() 
    
    # loads a previously trained model
    def __load_model(self) -> None:
        self.__print_model_initialization()
        used_model_type, _ = get_model_config()
        self.model = ClassificationModel(used_model_type, 'outputs', num_labels=self.labels.index.size, use_cuda=is_cuda_available(), args=self.__args())
        self.model_state = ModelState.INITIALIZED
    
    # loads the labels from a csv file
    def load_labels(self, filepath_or_buffer) -> None:
        if is_test_mode():
            # Do nothing in test mode
            return
        self.mutex.acquire()

        self.__load_labels(filepath_or_buffer)

        self.mutex.release()

    # loads labels from a csv file
    def __load_labels(self, filepath_or_buffer) -> None:
        if self.is_starting():
            return
        self.labels = pd.read_csv(filepath_or_buffer, header=None, sep=';', na_filter=False)

    # Trains the model using the given training set
    def train_model(self, training_set) -> None:
        if is_test_mode():
            # Do nothing in test mode
            return
        self.mutex.acquire()

        if self.model is None:
            self.mutex.release()
            return

        if not isinstance(training_set, pd.DataFrame):
            training_set = pd.read_csv(training_set, header=None, sep=';', na_filter=False)
        self.model.train_model(training_set)

        self.mutex.release()

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set) -> dict:
        if is_test_mode():
            # In test mode, return a predefined evaluation response
            return dict(acc=0.1, eval_loss=0.2, f1=0.3, mcc=0.4)
        self.mutex.acquire()

        if self.model is None:
            self.mutex.release()
            return
    
        if not isinstance(evaluation_set, pd.DataFrame):
            evaluation_set = pd.read_csv(evaluation_set, header=None, sep=';', na_filter=False)

        def f1_multiclass(l, preds):
            return f1_score(l, preds, average='micro')
        result, _, _ = self.model.eval_model(evaluation_set, f1=f1_multiclass, acc=accuracy_score)

        self.mutex.release()
        return result

    # Makes predictions for the expected return type of each of the given method names (uses cached values if exist)
    def predict(self, method_names: list) -> list:
        if is_test_mode():
            # In test mode, return a list of random values
            types = list()
            for m in method_names:
                types.append(randint(0, 1))
            return types
        self.mutex.acquire()

        if self.model is None:
            self.mutex.release()
            return list()

        unpredicted = self.__get_unpredicted(method_names)
        self.__predict_and_save_in_cache(unpredicted)

        result = list()
        for m in method_names:
            result.append(self.prediction_cache[m])

        self.mutex.release()
        return result
    
    # makes predictions and saves them in the prediction cache
    def __predict_and_save_in_cache(self, method_names: list):
        if len(method_names) > 0:
            predictions, _ = self.model.predict(method_names)
            for i in range(len(predictions)):
                self.prediction_cache[method_names[i]] = predictions[i]

    # returns a list of method names from the passed method names list which are not in the predictions cache
    def __get_unpredicted(self, method_names: list) -> list:
        unpredicted = list()
        for m in method_names:
            if not m in self.prediction_cache:
                unpredicted.append(m)
        return unpredicted

    # returns the type name for the given label
    def get_type_by_label(self, label: int) -> str:
        if is_test_mode():
            # In test mode, return predefined values
            if label == 0:
                return "object"
            return "void"

        if self.labels is None:
            return ""

        row_for_label = self.labels.loc[self.labels.iloc[:, 1] == label]
        return str(row_for_label.iloc[0, 0])