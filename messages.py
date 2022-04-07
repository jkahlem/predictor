import json
from enum import Enum

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

class CompoundTaskOptions:
    with_return_type: bool
    with_parameter_types: bool

    def __init__(self, options: dict = dict()) -> None:
        self.with_return_type = False
        self.with_parameter_types = False

        if 'withReturnType' in options:
            self.with_return_type = options['withReturnType']
        if 'withParameterTypes' in options:
            self.with_parameter_types = options['withParameterTypes']

class MethodGenerationTaskOptions:
    parameter_names: CompoundTaskOptions
    parameter_types: bool
    return_type: bool

    def __init__(self, options: dict = dict()):
        self.parameter_names = CompoundTaskOptions()
        self.parameter_types = False
        self.return_type = False
        if 'parameterNames' in options:
            self.parameter_names = CompoundTaskOptions(options['parameterNames'])
        if 'parameterTypes' in options:
            self.parameter_types = options['parameterTypes']
            #if self.parameter_names.with_parameter_types and self.parameter_types:
            #    raise Exception('Cannot generate method with parameter types as compound task and as separate task.')
            if self.parameter_types:
                raise Exception('Parameter type generation as separate task is currently not supported.')
        if 'returnType' in options:
            self.return_type = options['returnType']
            if self.parameter_names.with_return_type and self.return_type:
                raise Exception('Cannot generate method with return type as compound task and as separate task.')

class ModelOptions:
    batch_size: int
    num_of_epochs: int
    generation_tasks: MethodGenerationTaskOptions
    max_sequence_length: int
    num_return_sequences: int
    default_context: list[str]
    use_type_prefixing: bool
    empty_parameter_list_by_keyword: bool

    def __init__(self, options: dict = dict()):
        self.batch_size = 0
        self.num_of_epochs = 0
        self.max_sequence_length = 0
        self.num_return_sequences = 0
        self.generation_tasks = MethodGenerationTaskOptions()
        self.default_context = list()
        self.use_type_prefixing = False
        self.empty_parameter_list_by_keyword = False
        if 'batchSize' in options:
            self.batch_size = options['batchSize']
        if 'numOfEpochs' in options:
            self.num_of_epochs = options['numOfEpochs']
        if 'maxSequenceLength' in options:
            self.max_sequence_length = options['maxSequenceLength']
        if 'numReturnSequences' in options:
            self.num_return_sequences = options['numReturnSequences']        
        if 'generationTasks' in options:
            self.generation_tasks = MethodGenerationTaskOptions(options['generationTasks'])
        if 'defaultContext' in options:
            self.default_context = options['defaultContext']
        if 'useTypePrefixing' in options:
            self.use_type_prefixing = options['useTypePrefixing']
        if 'emptyParameterListByKeyword' in options:
            self.empty_parameter_list_by_keyword = options['emptyParameterListByKeyword']

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

    def __init__(self, options: dict = dict()):
        self.retrain = False
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

    def __str__(self) -> str:
        return "Options..."

    def __repr__(self) -> str:
        return "Options..."

class TrainMessage:
    training_data: list[Method]
    options: Options

    def __init__(self, msg: dict):
        self.training_data = [Method(data) for data in msg['params']['trainData']]
        self.options = Options(msg['params']['options'])
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
        return "Predict message..."

    def __repr__(self) -> str:
        return "Predict message..."
