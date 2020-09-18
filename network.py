import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4,stride=2)
        self.conv2_drop = nn.Dropout2d()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 9)


    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        x = observation
        x = x.permute(0,3,1,2)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        #print(x.shape)
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
             

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        actions_np = []
        for i in actions:
            actions_np.append(i.numpy())
        self.values, inverse,count = np.unique(actions_np ,return_inverse=True, return_counts=True, axis=0)
        onehot = np.eye(self.values.shape[0])[inverse]
        print(count)
        
        class_tensor=[]
        for item in onehot:
            class_tensor.append(torch.tensor(item,dtype=torch.float32))
            
        return class_tensor

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        
        #print(scores)
        preds = torch.argmax(scores,dim=1)
        #print(preds.shape)
        class_labels = self.values[preds]
        #print(class_labels)
        
        return class_labels
        

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
