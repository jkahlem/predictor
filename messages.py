import json
from enum import Enum
from typing import Tuple

from methods import Method, MethodContext, MethodEncoder

class SupportedModels(str, Enum):
    ReturnTypesPrediction = "ReturnTypesPrediction"
    MethodGenerator = "MethodGenerator"

# defines a stream message (in simplified http) wrapping the jsonrpc messages 
class Message:
    def __init__(self, header, body):
        self.header = header
        self.body = body
        if header is None and type(body) is dict:
            body_as_str = json.dumps(body, cls=MethodEncoder)
            self.header = MessageHeader(len(body_as_str), "application/json")
            self.bodystr = body_as_str

    def __str__(self) -> str:
        return str(self.header)+str(self.bodystr)

    def __repr__(self) -> str:
        return str(self.header)+str(self.bodystr)

# Defines the header of a stream message
class MessageHeader:
    def __init__(self, length: int, mediaType: str):
        self.length = length
        self.mediaType = mediaType

    def __str__(self) -> str:
        return "Content-Length: "+str(self.length)+"\r\nContent-Type: "+str(self.mediaType)+"\r\n\r\n"

    def __repr__(self) -> str:
        return "Content-Length: "+str(self.length)+"\r\nContent-Type: "+str(self.mediaType)+"\r\n\r\n"

# parses a message from a file descriptor
def parse_message_from_fd(fd) -> Message:
    header = __parse_header(fd)
    if header is None:
        return

    body = fd.read(header.length)
    if len(body) == 0:
        return
    return Message(header, json.loads(body))

# parses the header of a message
def __parse_header(fd) -> MessageHeader:
    length, mediaType = 0, ""
    while True:
        line: str = fd.readline()
        # if the line is empty, return nothing and stop parsing (connection closed)
        if len(line) == 0:
            return

        # when encoutering the header seperator ("\r\n" as whole line) stop parsing the header
        if line == "\r\n":
            break
        if not line.endswith("\r\n"):
            raise Exception("Unexpected header line: `" + line + "`, len: " + str(len(line)))

        # remove the line seperator
        line: str = line[:-2]

        # split the header line at colon
        parts = line.split(": ")
        if len(parts) != 2:
            raise Exception("Unexpected header line: `" + line + "`, len: " + str(len(line)))

        # parse header line key-value pair
        if parts[0] == "Content-Type":
            if not parts[1].startswith("application/json"):
                raise Exception("Unexpected content-type: " + parts[1])
            else:
                mediaType = "application/json"
        elif parts[0] == "Content-Length":
            length = int(parts[1])
        else:
            raise Exception("Unexpected header option: " + parts[0])

    return MessageHeader(length, mediaType)

class Adafactor:
    beta: float
    clip_threshold: float
    decay_rate: float
    eps: Tuple[float, float]
    relative_step: bool
    warmup_init: bool
    scale_parameter: bool

    def __init__(self, options: dict = dict()):
        self.beta = None
        self.clip_threshold = None
        self.decay_rate = None
        self.eps = None
        self.relative_step = None
        self.warmup_init = None
        self.scale_parameter = None
        if 'beta' in options:
            self.beta = (options['beta'])
        if 'clipThreshold' in options:
            self.clip_threshold = options['clipThreshold']
        if 'decayRate' in options:
            self.decay_rate = options['decayRate']
        if 'eps' in options:
            self.eps = options['eps'][0], options['eps'][1]
        if 'relativeStep' in options:
            self.relative_step = options['relativeStep']
        if 'warmupInit' in options:
            self.warmup_init = options['warmupInit']
        if 'scaleParameter' in options:
            self.scale_parameter = options['scaleParameter']

    def __str__(self) -> str:
        return "Options..."

    def __repr__(self) -> str:
        return "Options..."

class Adam:
    learning_rate: float
    eps: float

    def __init__(self, options: dict = dict()):
        self.learning_rate = None
        self.eps = None
        if 'eps' in options:
            self.eps = options['eps']
        if 'learningRate' in options:
            self.learning_rate = options['learningRate']

    def __str__(self) -> str:
        return "Options..."

    def __repr__(self) -> str:
        return "Options..."

class OutputComponentOrder:
    return_type: int
    parameter_type: int
    parameter_name: int

    def __init__(self, order: dict = dict()) -> None:
        self.parameter_type = 1
        self.parameter_name = 2
        self.return_type = 3
        if 'parameterType' in order:
            self.parameter_type = order['parameterType']
        if 'parameterName' in order:
            self.parameter_name = order['parameterName']
        if 'returnType' in order:
            self.return_type = order['returnType']

    def is_valid(self) -> bool:
        return len({self.parameter_name, self.parameter_type, self.return_type}) == 3

class ModelOptions:
    batch_size: int
    num_of_epochs: int
    max_sequence_length: int
    num_return_sequences: int
    num_beams: int
    default_context: list[str]
    empty_parameter_list_by_keyword: bool
    adafactor: Adafactor
    adam: Adam
    model_type: str
    model_name: str
    length_penalty: float
    top_k: float
    top_p: float
    output_order: OutputComponentOrder

    def __init__(self, options: dict = dict()):
        self.batch_size = 0
        self.num_of_epochs = 0
        self.max_sequence_length = 0
        self.num_return_sequences = 0
        self.default_context = list()
        self.empty_parameter_list_by_keyword = False
        self.adafactor = Adafactor()
        self.adam = Adam()
        self.model_type = "t5"
        self.model_name = "t5-small"
        self.num_beams = 0
        self.length_penalty = None
        self.top_k = None
        self.top_p = None
        self.output_order = OutputComponentOrder()
        if 'batchSize' in options:
            self.batch_size = options['batchSize']
        if 'numOfEpochs' in options:
            self.num_of_epochs = options['numOfEpochs']
        if 'maxSequenceLength' in options:
            self.max_sequence_length = options['maxSequenceLength']
        if 'numReturnSequences' in options:
            self.num_return_sequences = options['numReturnSequences']
        if 'defaultContext' in options:
            self.default_context = options['defaultContext']
        if 'emptyParameterListByKeyword' in options:
            self.empty_parameter_list_by_keyword = options['emptyParameterListByKeyword']
        if 'adafactor' in options:
            self.adafactor = Adafactor(options['adafactor'])
        if 'adam' in options:
            self.adam = Adam(options['adam'])
        if 'modelType' in options:
            self.model_type = options['modelType']
            if self.model_type == 'bart':
                self.model_name = 'facebook/bart-base'
        if 'modelName' in options:
            self.model_name = options['modelName']
        if 'numBeams' in options:
            self.num_beams = options['numBeams']
        if 'lengthPenalty' in options:
            self.length_penalty = options['lengthPenalty']
        if 'topK' in options:
            self.top_k = options['topK']
        if 'topP' in options:
            self.top_p = options['topP']
        if 'outputOrder' in options:
            self.output_order = OutputComponentOrder(options['outputOrder'])

    def __str__(self) -> str:
        return "Model Options..."

    def __repr__(self) -> str:
        return "Model Options..."

class Options:
    labels: str
    target_model: SupportedModels
    identifier: str
    retrain: bool
    model_options: ModelOptions
    checkpoint: str

    def __init__(self, options: dict = dict()):
        self.retrain = False
        self.checkpoint = ''
        if 'type' in options:
            self.target_model = SupportedModels(options['type'])
        if 'identifier' in options:
            self.identifier = options['identifier']
        if 'retrain' in options:
            self.retrain = options['retrain']
        if 'labels' in options:
            self.labels = options['labels']
        if 'modelOptions' in options:
            self.model_options = ModelOptions(options['modelOptions'])
        if 'checkpoint' in options:
            self.checkpoint = options['checkpoint']

    def __str__(self) -> str:
        return "Options..."

    def __repr__(self) -> str:
        return "Options..."

class TrainMessage:
    training_data: list[Method]
    options: Options
    continue_training: bool

    def __init__(self, msg: dict):
        self.training_data = [Method(data) for data in msg['params']['trainData']]
        self.options = Options(msg['params']['options'])
        self.continue_training = msg['params']['continueTraining']
        self.id = msg['id']

    def __str__(self) -> str:
        return "Train message..."

    def __repr__(self) -> str:
        return "Train message..."

class EvaluateMessage:
    evaluation_data: list[Method]
    options: Options

    def __init__(self, msg: dict):
        self.evaluation_data = [Method(data) for data in msg['params']['evaluationData']]
        self.options = Options(msg['params']['options'])
        self.id = msg['id']

    def __str__(self) -> str:
        return "Evaluation message..."

    def __repr__(self) -> str:
        return "Evaluation message..."

class PredictMessage:
    def __init__(self, msg: dict):
        self.prediction_data = [MethodContext(data) for data in msg['params']['predictionData']]
        self.options = Options(msg['params']['options'])
        self.id = msg['id']

    def __str__(self) -> str:
        return "Predict message..."

    def __repr__(self) -> str:
        return "Predict message..."

class ExistsMessage:
    def __init__(self, msg: dict):
        self.options = Options(msg['params']['options'])
        self.id = msg['id']

    def __str__(self) -> str:
        return "Exists message..."

    def __repr__(self) -> str:
        return "Exists message..."

class GetCheckpointsMessage:
    def __init__(self, msg: dict) -> None:
        self.options = Options(msg['params']['options'])
        self.id = msg['id']

    def __str__(self) -> str:
        return "GetCheckpoints message..."

    def __repr__(self) -> str:
        return "GetCheckpoints message..."

class GetModelsMessage:
    def __init__(self, msg: dict) -> None:
        self.model_type = msg['params']['modelType']
        self.id = msg['id']

    def __str__(self) -> str:
        return "GetCheckpoints message..."

    def __repr__(self) -> str:
        return "GetCheckpoints message..."