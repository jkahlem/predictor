import json
from jsonrpcErrorCodes import JsonRpcErrorCodes
from modelConsts import ArrayToken

class SentenceFormattingOptions:
    method_name: bool
    type_name: bool
    parameter_name: bool

    def __init__(self, options: dict = dict()):
        self.method_name = False
        self.type_name = False
        self.parameter_name = False
        if 'methodName' in options:
            self.method_name = options['methodName']
        if 'parameterName' in options:
            self.parameter_name = options['parameterName']
        if 'typeName' in options:
            self.type_name = options['typeName']

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
    className: list[str]
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
        if type.endswith(ArrayToken):
            p.is_array = True
            p.type = type[:-len(ArrayToken)]
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

class Model:
    model_name: str
    checkpoints: list[str]
    sentence_formatting_options: SentenceFormattingOptions
    def __init__(self) -> None:
        self.checkpoints = list()
        self.sentence_formatting_options = SentenceFormattingOptions()

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
        if isinstance(obj, Model):
            sentence_formatting_options = dict()
            if obj.sentence_formatting_options is not None:
                sentence_formatting_options = dict({
                    'methodName': obj.sentence_formatting_options.method_name,
                    'parameterName': obj.sentence_formatting_options.parameter_name,
                    'typeName': obj.sentence_formatting_options.type_name,
                })
            return dict({'modelName': obj.model_name,
                'checkpoints': obj.checkpoints,
                'sentenceFormattingOptions': sentence_formatting_options})
        return json.JSONEncoder.default(self, obj)