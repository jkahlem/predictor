import os
import shutil
from io import StringIO

# Copies the content of a string (wrapped in StringIO) to the target path
def copyTo(strio: StringIO, target_path) -> None:
    containing_dir = os.path.dirname(target_path)
    if not os.path.exists(containing_dir):
        os.makedirs(containing_dir)

    with open(target_path, 'w') as fd:
        strio.seek(0)
        shutil.copyfileobj(strio, fd)
        strio.seek(0)