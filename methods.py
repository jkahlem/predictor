import json
from jsonrpcErrorCodes import JsonRpcErrorCodes

class Parameter:
    name: str
    type: str
    def __init__(self, parameter: dict = dict()) -> None:
        if 'name' in parameter:
            self.name = parameter['name']
        if 'type' in parameter:
            self.type = parameter['type']

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class MethodContext:
    className: str
    is_static: bool
    methodName: str
    types: list[str]
    def __init__(self, context: dict = dict()) -> None:
        self.className = context['className']
        self.is_static = context['isStatic']
        self.methodName = context['methodName']
        self.types = context['types']

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class MethodValues:
    returnType: str
    parameters: list[Parameter]
    def __init__(self, values: dict = dict()) -> None:
        self.returnType = ''
        if 'returnType' in values:
            self.returnType = values['returnType']
        self.parameters = list()
        if 'parameters' in values and values['parameters'] is not None:
            for rawParameter in values['parameters']:
                self.parameters.append(Parameter(rawParameter))

    def add_parameter(self, name: str, type: str) -> None:
        p = Parameter()
        p.name = name
        p.type = type
        self.parameters.append(p)

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class Method:
    context: MethodContext
    values: MethodValues
    def __init__(self, method: dict = dict()) -> None:
        self.context = MethodContext(method['context'])
        self.values = MethodValues(method['values'])

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class MethodEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MethodValues):
            return dict({'returnType': obj.returnType, 'parameters': obj.parameters})
        if isinstance(obj, Parameter):
            return dict({'name': obj.name, 'type': obj.type})
        if isinstance(obj, JsonRpcErrorCodes):
            return obj.value
        return json.JSONEncoder.default(self, obj)