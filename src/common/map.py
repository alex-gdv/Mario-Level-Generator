import json
import os

from root import rootpath

MAP = { "X":(False, "ground"), "S":(False, "coinBrick"), "-":(False, "sky"),
        "?":(True, "RandomBox"), "Q":(True,"CoinBox"), "E":(True, "Goomba"),
        "<":(False, "Pipe_left_top"),">":(False, "Pipe_right_top"),
        "[":(False, "Pipe_left_body"), "]":(False, "Pipe_right_body"),
        "o":(True, "coin")}

def map_string_to_json(lvl_string, lvl_name, generated=False, path=""):
    lvl = lvl_string.split("\n")
    lvl = list(filter(None, lvl))
    height = len(lvl)
    width = len(lvl[0])

    data = {}
    data["id"] = lvl_name
    data["length"] = width
    objects = {"bush":[], "sky":[], "cloud":[], "pipe":[], "ground":[]}
    layers = { "sky":{"x":[0,width], "y":[0,13]}, "ground":{"x":[0,width], "y":[14,16]}}
    entities = {"CoinBox":[], "coinBrick":[], "coin":[], "Goomba":[], "Koopa":[],
                "RandomBox":[]}
    level = {"objects":objects, "layers":layers, "entities":entities}
    data["level"] = level

    for y in range(height-1):
        for x in range(width):
            if lvl[y][x] in MAP:
                if MAP[lvl[y][x]][0]:
                    data["level"]["entities"][MAP[lvl[y][x]][1]].append([x,y])
                elif MAP[lvl[y][x]][1] == "ground":
                    data["level"]["objects"]["ground"].append([x,y])
                elif MAP[lvl[y][x]][1] == "Pipe_left_top":
                    data["level"]["objects"]["pipe"].append([x,y,1])

    for x in range(width):
        if lvl[height-1][x] in MAP and MAP[lvl[height-1][x]][1] == "sky":
            data["level"]["objects"]["sky"].append([x,height-1])
            data["level"]["objects"]["sky"].append([x,height])

    if generated:
        os.makedirs(path, exist_ok=True)
        p = f"{path}/Level{lvl_name}.json"
    else:
        p = f"{rootpath}/super_mario_python/levels/Level{lvl_name}.json"
    with open(p, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    path = rootpath + "/mariopuzzle/mario_level_repairer/levels/"
    file_names = os.listdir(path)
    for name in file_names:
        with open(path + name, "r") as f:
            s = f.read()
            map_string_to_json(s, name.split("mario-")[-1][:-4])