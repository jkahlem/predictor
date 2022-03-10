class Parameter:
    name: str
    type: str
    def __init__(self, parameter: dict) -> None:
        self.name = parameter['name']
        self.type = parameter['type']

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class MethodContext:
    className: str
    isStatic: bool
    methodName: str
    types: list[str]
    def __init__(self, context: dict) -> None:
        self.className = context['className']
        self.isStatic = context['isStatic']
        self.methodName = context['methodName']
        self.types = context['types']

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class MethodValues:
    returnType: str
    parameters: list[Parameter]
    def __init__(self, values: dict) -> None:
        self.returnType = values['returnType']
        self.parameters = list()
        for rawParameter in values['parameters']:
            self.parameters.append(Parameter(rawParameter))

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class Method:
    context: MethodContext
    values: MethodValues
    def __init__(self, method: dict) -> None:
        self.context = MethodContext(method['context'])
        self.values = MethodValues(method['values'])

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass
