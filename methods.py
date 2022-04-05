import json
from jsonrpcErrorCodes import JsonRpcErrorCodes

class Parameter:
    name: str
    type: str
    is_array: bool
    def __init__(self, parameter: dict = dict()) -> None:
        self.name = ''
        self.type = ''
        self.is_array = False
        if 'name' in parameter:
            self.name = parameter['name']
        if 'type' in parameter:
            self.type = parameter['type']
        if 'isArray' in parameter:
            self.is_array = parameter['isArray']

    def __str__(self) -> str:
        return self.type + ' ' + self.name

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
        if type.endswith('[]'):
            p.is_array = True
            p.type = type[:-2]
        else:
            p.type = type
        self.parameters.append(p)
    
    def set_return_type(self, type: str) -> None:
        type = type.strip()
        if type.endswith('.'):
            type = type[:-1]
        self.returnType = type

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

    def current_state_hash(self) -> int:
        return hash((self.returnType, ", ".join([str(x) for x in self.parameters])))

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