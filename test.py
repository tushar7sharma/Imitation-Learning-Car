from imitations import load_imitations
from network import ClassificationNetwork
import numpy as np


dataFolder = "data/teacher/"
observations, actions = load_imitations(dataFolder)

# print("obs len: ", len(obs), "acts len: ", len(acts))
"""
acts : steer, gas, brake
steer values:  {0.0, 1.0, -1.0}
gas values:  {0.0, 0.5}
brake values:  {0.0, 0.8}
9 classes of actions
action values:  {
    0, (0.0, 0.0, 0.0), no_op 
    1, (0.0, 0.5, 0.0), gas
    2, (-1.0, 0.5, 0.0), gas_left
    3, (-1.0, 0.0, 0.0), steer_left
    4, (-1.0, 0.0, 0.8), brake_left
    5, (0.0, 0.0, 0.8), brake
    6, (1.0, 0.0, 0.8), brake_right
    7, (1.0, 0.0, 0.0), steer_right
    8, (1.0, 0.5, 0.0), gas_right 
    }
"""

steerValueSet = set()
gasValueSet = set()
brakeValueSet = set()
actionSet = set()

# for i in range(len(actions)):
#     steerValueSet.add(actions[i][0])
#     gasValueSet.add(actions[i][1])
#     brakeValueSet.add(actions[i][2])
#     actionSet.add( (actions[i][0],actions[i][1],actions[i][2] ))
    # print(i, acts[i])

# print("steer values: ", steerValueSet)
# print("gas values: ", gasValueSet)
# print("brake values: ", brakeValueSet)
# print("action values: ", actionSet)

testNet = ClassificationNetwork()
actionClasses = testNet.actions_to_classes(actions)
# print(actionClasses[0])
for i in range(len(actionClasses)):
    print(i, actionClasses[i])


# import os
# import numpy as np
# import glob
# from imitations import load_imitations
# import matplotlib.pyplot as plt
# import torch
# from network import ClassificationNetwork





# device = torch.device('cpu')
# data_folder = "data/teacher/"
# obs , act = load_imitations(data_folder)

# model = ClassificationNetwork()
# print(model)
# model.to(device)
# ip = torch.tensor(obs)

# ip.shape
# ip = torch.reshape(ip,(599,3,96,96))
# preds = model(ip)