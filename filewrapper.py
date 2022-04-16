import tempfile


class FileWrapper():
    def __init__(self, is_temp: bool) -> None:
        self.is_temp = is_temp
        if is_temp:
            self.fd = tempfile.TemporaryFile()
        else:
            self.fd = open("c:\\test\\dataset.csv", "w+")

    def file_descriptor(self):
        return self.fd
    
    def seek(self, offset: int) -> None:
        self.fd.seek(offset)
    
    def write(self, contents: str) -> None:
        if self.is_temp:
            self.fd.write(contents.encode('utf-8'))
        else:
            self.fd.write(contents)

    def close(self):
        self.fd.close()
