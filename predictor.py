import socket
import logging
import errno
import os
import threading
from enum import Enum
from io import StringIO
from languageGenerationModel import MethodGenerationModel

from messages import Message, parse_message_from_fd, SupportedModels, TrainMessage
from config import get_port, get_script_dir, is_cuda_available, load_config
from model import ModelHolder
from returnTypesPredictionModel import ReturnTypesPredictionModel
from sentenceTransformer import testSentenceTransformer

class JsonRpcErrorCodes(str, Enum):
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32602
    InternalError = -32603
    ServerError = -32000

# Handles a connection for rpc messages
class ConnectionHandler:
    def __init__(self, connection):
        self.connection = connection
        self.fd = connection.makefile(newline='')

    # handles the connection
    def handle(self) -> None:
        try:
            while True:
                msg = parse_message_from_fd(self.fd)
                # msg is none if there is no message to parse because the connection was clsoed
                if msg is None:
                    self.connection.close()
                    return
                self.__handle_message(msg.body)
        except socket.error as e:
            if e.errno != errno.ECONNRESET:
                logging.exception("Error")
        except Exception:
            logging.exception("Error")
            self.__send_error_msg(JsonRpcErrorCodes.ParseError, "Parser error")
        finally:
            print("connection closed")
            self.connection.close()

    # handles a message
    def __handle_message(self, msg: dict) -> None:
        print("Received message: " + msg['method'])
        
        if msg['method'] == "train":
            self.__handle_train_message(TrainMessage(msg))
        elif msg['method'] == "predict":
            self.__handle_predict_message(msg)
        else:
            self.__send_error_msg(JsonRpcErrorCodes.MethodNotFound, "Method not found: " +msg['method'])

    # Handles a train message which trains a new model
    def __handle_train_message(self, msg: TrainMessage) -> None:
        model = get_model(msg.targetModel)
        model.load_additional(msg.additional)
        model.create_new_model()
        model.train_model(msg.training_set)

        result = model.eval_model(msg.evaluation_set)

        self.__send_evaluation_msg(msg.id, result)
    
    # Sends an evaluation object as response to a train message
    def __send_evaluation_msg(self, id, result: dict) -> None:
        eval = dict(accScore=None, evalLoss=None, f1Score=None, mcc=None)
        if "acc" in result:
            eval["accScore"] = result["acc"]
        if "eval_loss" in result:
            eval["evalLoss"] = result["eval_loss"]
        if "f1" in result:
            eval["f1Score"] = result["f1"]
        if "mcc" in result:
            eval["mcc"] = result['mcc']
        print(result)
        
        msg = self.__create_jsonrpc_response(id, eval)
        
        self.__send_str_to_conn(str(Message(None, msg)))
    
    # Handles a predict message which makes predictions to the given method names
    def __handle_predict_message(self, msg: dict) -> None:
        words = msg['params']['predictionData']
        model = get_model(SupportedModels(msg['params']['targetModel']))
        model.load_model()

        prediction = model.predict(words)
        if len(prediction) != len(words):
            self.__send_error_msg("Internal error")
            print("Only " + str(len(prediction)) + " predicted, expected to predict " + str(len(words)) + " types")
            return

        self.__send_predictions_msg(msg['id'], prediction)

    # Sends predictions to method names in response to a predict message
    def __send_predictions_msg(self, id, predicted_types: list) -> None:
        print("Send predictions as response to message with id " + str(id))
        msg = self.__create_jsonrpc_response(id, predicted_types)
        self.__send_str_to_conn(str(Message(None, msg)))

    # Creates a response using the jsonrpc protocol
    def __create_jsonrpc_response(self, id, result) -> dict:
        return dict(jsonrpc="2.0", id=id, result=result)
    
    # Sends a jsonrpc response error with the given message
    def __send_error_msg(self, code, msg: str) -> None:
        print("Send error message with code " + str(code) + ": " + msg)
        response_error = dict(jsonrpc="2.0", code=code, msg=msg)
        self.__send_str_to_conn(str(Message(None, response_error)))

    # Sends a plain string to the connection
    def __send_str_to_conn(self, str: str) -> None:
        self.connection.sendall((str).encode("utf-8"))


# Creates a connection handler for the connection
def handle_connection(connection) -> None:
    handler = ConnectionHandler(connection)
    handler.handle()


# Starts the server
def start_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('', get_port())

    print("starting up on port " + str(get_port()))
    sock.bind(server_address)
    sock.listen()

    try: 
        while True:
            connection, client_address = sock.accept()
            print("accept connection to " + str(client_address))
            connection_thread = threading.Thread(target=handle_connection, args=(connection,))
            connection_thread.start()
    except KeyboardInterrupt:
        print('Closing server...')
        sock.close()


# Gets the prediction model (global singleton) currently in use
def get_model(target: SupportedModels) -> ModelHolder:
    global prediction_models, prediction_model_lock
    prediction_model_lock.acquire()
    if not target in prediction_models:
        if target == SupportedModels.ReturnTypesPrediction:
            prediction_models[target] = ModelHolder(ReturnTypesPredictionModel())
        elif target == SupportedModels.MethodGenerator:
            prediction_models[target] = ModelHolder(MethodGenerationModel())
        else:
            prediction_model_lock.release()
            raise Exception("Unsupported target model: " + target)
    prediction_model_lock.release()
    return prediction_models[target]

# Startup script for the server
if __name__ == '__main__':
    os.chdir(get_script_dir())
    global prediction_models, prediction_model_lock

    prediction_model_lock = threading.Lock()
    prediction_models = dict()

    load_config()
    if not is_cuda_available():
        print("CUDA is not available on the machine. Use CPU for machine learning tasks (which may take more time).")

    start_server()
