from enum import Enum

class JsonRpcErrorCodes(Enum):
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32602
    InternalError = -32603
    ServerError = -32000

    def __str__(self) -> str:
        return str(self.value)