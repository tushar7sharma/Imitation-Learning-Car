import torch
import random
import time
from network import ClassificationNetwork
from imitations import load_imitations
import torch.nn as nn
import tqdm


def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    gpu =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action = ClassificationNetwork().to(gpu)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-4)  #-4
    observations, actions = load_imitations(data_folder)
    #print(len(actions))
    #print(len(observations))
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]
    
    nr_epochs = 300 # 400
    batch_size = 64
    number_of_classes = 9 # needs to be changed
    start_time = time.time()
    
    for epoch in range(nr_epochs):
        
        random.shuffle(batches)
        
        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(gpu))
            batch_gt.append(batch[1].to(gpu))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                #print(number_of_classes)
                #print(batch_gt)
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))

                batch_out = infer_action(batch_in)
                #print(batch_out.shape)
                loss = cross_entropy_loss(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []
                
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        
        if epoch%30==0:
            torch.save(infer_action, trained_network_file)

    #torch.save(infer_action, trained_network_file)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    #print(f'Shape of ground truth: {batch_gt.shape} | shape of preds = {batch_out.shape}')
    #print(batch_out[0])
#    print(batch_gt[0])
    loss_f = nn.CrossEntropyLoss()
    #loss = loss_f(batch_gt,torch.argmax(batch_out,dim=0)[0])
    loss =  loss_f(batch_out, torch.max(batch_gt, 1)[1])
    #print(loss)
    return loss
