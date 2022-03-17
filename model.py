from enum import Enum
from optparse import Option
import os
import shutil
import threading

import pandas as pd

from config import get_script_dir, is_test_mode
from messages import Options
from methods import Method, MethodContext, MethodValues

class ModelState(Enum):
    NONE = 1
    INITIALIZED = 3

# Wraps the classification model functionality and is thread safe
class Model:
    def __init__(self):
        pass

    # initializes a new model
    def init_new_model(self) -> None:
        pass

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        pass

    # Trains the model using the given training set
    def train_model(self, training_set: pd.DataFrame) -> None:
        pass

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set: pd.DataFrame) -> dict:
        pass

    # Makes predictions
    def predict(self, predictionData: list[MethodContext]) -> list[MethodValues]:
        pass

    # path addition for the cacheDir
    def cache_dir_name(self) -> str:
        pass

    # path addition for the outputs dir 
    def outputs_dir_name(self) -> str:
        pass

    # converts the input data to a pandas frame
    def convert_methods_to_frame(self, data: list[Method]) -> pd.DataFrame:
        return data
    
    # returns a method identifier for the method for caching. Methods with the same identifier won't be predicted again.
    def get_identifier_for_method(self, method: MethodContext) -> str:
        pass

    # Sets the options for the model
    def set_options(self, options: Options) -> None:
        pass

    # Returns true if the model already exists (and therefore is for example not available for training if not retraining the model)
    def exists(self) -> bool:
        return False

    # Removes all files which belong to this model
    def remove_model(self) -> None:
        pass

class ModelHolder():
    def __init__(self, model: Model):
        self.model_state = ModelState.NONE
        self.model = model
        self.model_identifier = ''
        self.options = Options()
        self.mutex = threading.Lock()
        self.prediction_cache = dict()

    # returns true if the model is starting
    def is_starting(self) -> bool:
        return self.model_state != ModelState.NONE

    # Creates a new classification model using the loaded labels (if labels are loaded, otherwise nothing happens)
    def create_new_model(self) -> None:
        if is_test_mode():
            # Do nothing in test mode
            return
        self.mutex.acquire()

        if self.is_starting():
            self.mutex.release()
            return
        elif self.model.exists() and not self.options.retrain:
            self.mutex.release()
            raise Exception('Cannot train model: Model is already trained')

        self.__remove_previous_model_files()
        self.__init_new_model()

        self.mutex.release()

    # removes the model files used by a previously trained model    
    def __remove_previous_model_files(self) -> None:
        self.__remove_dir_if_exists(os.path.join(get_script_dir(), self.model.outputs_dir_name()))
        self.__remove_dir_if_exists(os.path.join(get_script_dir(), self.model.cache_dir_name()))

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
    
    # initializes a new model
    def __init_new_model(self) -> None:
        self.model.init_new_model()
        self.model_state = ModelState.INITIALIZED

    # Loads an already created/trained model
    def load_model(self) -> None:
        if is_test_mode():
            # Do nothing in test mode
            return
        self.mutex.acquire()

        if self.is_starting():
            self.mutex.release()
            return

        self.model.load_model()
        self.model_state = ModelState.INITIALIZED
    
        self.mutex.release()

    # Trains the model using the given training set
    def train_model(self, training_set: list[MethodContext]) -> None:
        if is_test_mode():
            # Do nothing in test mode
            return
        self.mutex.acquire()

        if self.model is None:
            self.mutex.release()
            return

        self.model.train_model(self.model.convert_methods_to_frame(training_set))

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

        result = self.model.eval_model(self.model.convert_methods_to_frame(evaluation_set))

        self.mutex.release()
        return result
        
    # Makes predictions
    def predict(self, methods: list[MethodContext]) -> list[MethodValues]:
        self.mutex.acquire()

        if self.model is None:
            self.mutex.release()
            return list()

        unpredicted = self.__get_unpredicted(methods)
        self.__predict_and_save_in_cache(unpredicted)

        result = list()
        for m in methods:
            result.append(self.prediction_cache[self.model.get_identifier_for_method(m)])

        self.mutex.release()
        return result

    # makes predictions and saves them in the prediction cache
    def __predict_and_save_in_cache(self, methods: list[MethodContext]):
        if len(methods) > 0:
            predictions = self.model.predict(methods)
            for i in range(len(predictions)):
                self.prediction_cache[self.model.get_identifier_for_method(methods[i])] = predictions[i]

    # returns a list of method names from the passed method names list which are not in the predictions cache
    def __get_unpredicted(self, methods: list[MethodContext]) -> list[MethodContext]:
        unpredicted = list()
        for m in methods:
            if not self.model.get_identifier_for_method(m) in self.prediction_cache:
                unpredicted.append(m)
        return unpredicted
    
    def set_options(self, options: Options) -> None:
        self.mutex.acquire()

        if self.model is None:
            self.mutex.release()
            return

        self.model.set_options(options)
        self.model_identifier = options.identifier

        self.mutex.release()