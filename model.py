from enum import Enum
import os
import shutil
import threading

import pandas as pd

from config import get_script_dir, is_test_mode

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
    def train_model(self, training_set) -> None:
        pass

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set) -> dict:
        pass

    # Loads additional, model relevant data
    def load_additional(self, additional_data) -> None:
        pass

    # Makes predictions
    def predict(self, predictionData) -> list:
        pass

    # path addition for the cacheDir
    def cacheDirName(self) -> str:
        pass

    # path addition for the outputs dir 
    def outputsDirName(self) -> str:
        pass

class ModelHolder():
    def __init__(self, model: Model):
        self.model_state = ModelState.NONE
        self.model = model
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

        if self.is_starting() or self.labels is None:
            self.mutex.release()
            return

        self.__remove_previous_model_files()
        self.__init_new_model()

        self.mutex.release()

    # removes the model files used by a previously trained model    
    def __remove_previous_model_files(self) -> None:
        self.__remove_dir_if_exists(os.path.join(get_script_dir(), self.model.outputsDirName()))
        self.__remove_dir_if_exists(os.path.join(get_script_dir(), self.model.cacheDirName()))

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

        self.model.__load_model()
        self.model_state = ModelState.INITIALIZED
    
        self.mutex.release()

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

        result = self.model.eval_model(evaluation_set)

        self.mutex.release()
        return result

    # Evaluates the model using the given evaluation set
    def load_additional(self, additional_data) -> None:
        if is_test_mode():
            return
        self.mutex.acquire()

        if self.model is None:
            self.mutex.release()
            return

        self.model.load_additional(additional_data)

        self.mutex.release()

        
    # Makes predictions
    def predict(self, method_names: list) -> list:
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
            predictions = self.model.predict(method_names)
            for i in range(len(predictions)):
                self.prediction_cache[method_names[i]] = predictions[i]

    # returns a list of method names from the passed method names list which are not in the predictions cache
    def __get_unpredicted(self, method_names: list) -> list:
        unpredicted = list()
        for m in method_names:
            if not m in self.prediction_cache:
                unpredicted.append(m)
        return unpredicted