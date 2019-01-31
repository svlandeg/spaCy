import os
from pathlib import Path
from spacy.compat import path2str, symlink_remove


def target_local_path():
    return Path("./foo-target")


def link_local_path():
    return Path("./foo-symlink")


def exists():
    print(target_local_path().exists())


if __name__ == "__main__":
    exists()
    if not target_local_path().exists():
        os.mkdir(path2str(target_local_path()))
    exists()
    symlink_remove(link_local_path())
    os.rmdir(path2str(target_local_path()))
    exists()
