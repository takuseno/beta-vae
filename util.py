import numpy as np
import json
import re


def tile_images(images, row=4):
    shape = images.shape[1:]
    column = int(images.shape[0] / row)
    height = shape[0]
    width = shape[1]
    tile_height = row * height
    tile_width = column * width
    output = np.zeros((tile_height, tile_width, shape[-1]), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            image = images[i*column+j]
            output[i*height:(i+1)*height,j*width:(j+1)*width] = image
    return output

def dump_constants(constants, path):
    data = {}
    for name in dir(constants):
        if re.match(r'^([A-Z]|_|[0-9])+$', name):
            data[name] = getattr(constants, name)
    json_str = json.dumps(data)
    with open(path, 'w') as f:
        f.write(json_str + '\n')

def restore_constants(path):
    # dummy class
    class Constant:
        pass
    constants = Constant()
    with open(path, 'r') as f:
        json_obj = json.loads(f.read())
        for key, value in json_obj.items():
            setattr(constants, key, value)
    return constants
