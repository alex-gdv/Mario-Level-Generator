import os

def load_level_names():
    files = []
    res = []
    for r, d, f in os.walk(f"./super_mario_python/levels"):
        for file in f:
            files.append(os.path.join(r, file))
    for f in files:
        res.append(os.path.split(f)[1].split(".")[0])
    return res
