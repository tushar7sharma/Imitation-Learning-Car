import torch
import numpy as np


class ClassificationNetwork(torch.nn.Module):
    def __init__(self, actions):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')
        self.unique_actions = np.unique([str(v[0])+"_" + str(v[1])+"_"+str(v[2]) for v in actions]).tolist()
        self.num_classes = len(self.unique_actions)
        #self.num_classes = 4
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)

        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.drp = torch.nn.Dropout()

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.act = torch.nn.ReLU()

        self.lin1 = torch.nn.Linear(in_features=32*22*22, out_features=256)
        self.lin2 = torch.nn.Linear(in_features=256+7, out_features=self.num_classes)

        self.stfmx = torch.nn.Softmax()

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        res = []
        batch_size = len(observation)
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, batch_size)
        x = observation.permute(0, 3, 1, 2)
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.bn3(self.lin1(x)))
        x = torch.cat((x, speed, steering, gyroscope, abs_sensors), dim=1)
        x = self.lin2(x)

        if not self.training:
            x = self.stfmx(x)

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
        acts = [action.numpy() for action in actions]
        action_classes = np.array([self.unique_actions.index(str(v.numpy()[0])+"_" + str(v.numpy()[1])+"_"+str(v.numpy()[2])) for v in  actions])
        one_hot = np.zeros((action_classes.size, action_classes.max()+1))
        one_hot[np.arange(action_classes.size),action_classes] = 1
        return [torch.tensor(x) for x in one_hot]

    def actions_to_classes_multiclass(self, actions):
        classes = []
        for action in actions:
            _1 = action[0] == -1
            _2 = action[0] == 1
            _3 = action[1] > 0
            _4 = action[2] > 0
            classes.append(torch.stack([_1, _2, _3, _4]).float())
        return classes

    def scores_to_action_multiclass(self, scores):
        _1 = 0.0
        _2 = 0.0
        _3 = 0.0
        score = torch.round(scores[0])

        if (score[0] > 0).tolist():
            _1 = -1.0
        elif (score[1] > 0).tolist():
            _1 = 1.0
        if (score[2] > 0).tolist():
            _2 = 0.5
        if (score[3] > 0).tolist():
            _3 = 0.8
        return _1, _2, _3

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        values = self.unique_actions[torch.argmax(scores[0])].split('_')
        return float(values[0]), float(values[1]), float(values[2])

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
