import numpy as np
import json

with open('night_1.json') as f:
    data = json.load(f)

obj = {k:v for k, v in data.items() if k.startswith('objects')}
slice = list(obj.values())[0]

label_map = np.zeros((600, 400))

for i in range(len(slice)):
    group = {k:v for k, v in slice[i].items() if k.startswith('group')}
    seg = {k:v for k, v in slice[i].items() if k.startswith('segmentation')}
    seg_coor = list(seg.values())
    label_index = list(group.values())[0]

    for j in range(len(seg_coor[0])):
        print(f'x: {int(seg_coor[0][j][0])}, y: {int(seg_coor[0][j][1])}, {label_index}')
        label_map[int(seg_coor[0][j][0])][int(seg_coor[0][j][1])] = label_index
        print(f'label_classes: {np.unique(label_map)}')