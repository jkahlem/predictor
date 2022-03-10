class Method:
    def __init__(self, method: dict) -> None:
        self.context = MethodContext(method['context'])
        self.values = MethodValues(method['values'])

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

class MethodContext:
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
    def __init__(self, values: dict) -> None:
        self.returnTypes = values['returnTypes']
        self.parameters = values['parameters']

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass