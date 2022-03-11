import json

from methods import Method, MethodContext

class SupportedModels(str, Enum):
    ReturnTypesPrediction = "ReturnTypesPrediction"
    MethodGenerator = "MethodGenerator"

# defines a stream message (in simplified http) wrapping the jsonrpc messages 
class Message:
    def __init__(self, header, body):
        self.header = header
        self.body = body
        if header is None and type(body) is dict:
            body_as_str = json.dumps(body)
            self.header = MessageHeader(len(body_as_str), "application/json")
            self.body = body_as_str

    def __str__(self) -> str:
        return str(self.header)+str(self.body)

    def __repr__(self) -> str:
        return str(self.header)+str(self.body)

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


class Options:
    labels: str
    targetModel: SupportedModels
    identifier: str

    def __init__(self, options: dict):
        self.targetModel = SupportedModels(options['targetModel'])
        self.identifier = options['identifier']
        if 'labels' in options:
            self.labels = options['labels']

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
        self.prediction_data = [MethodContext(data) for data in msg['params']['predictionData']]# Todo: convert to MethodContext?
        self.options = Options(msg['params']['options'])
        self.id = msg['id']

    def __str__(self) -> str:
        return "Predict message..."

    def __repr__(self) -> str:
        return "Predict message..."