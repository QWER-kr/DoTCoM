import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from cosine_annealing_warmup import *
from utils import PolyLR

class SupervisedLearning():
    def __init__(self, trainloader, testloader, model, path, model_name, device):

        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.path = path
        self.model_name = model_name

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self, epochs, opt, lr_policy, lr, l2, momentum, step_size, retrain):

        if retrain:
            checkpoint = torch.load(self.path + self.model_name + '_checkpoint.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']
                print(param_group['lr'], param_group['initial_lr'])
            print("Restart training the model.")
        else:
            start_epoch = 0
            print("Start training the model.")

        if opt == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=l2)
        elif opt == 'adam':
            optimizer = optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
        elif opt == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=l2)

        if lr_policy == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, last_epoch=start_epoch-1)
        elif lr_policy == 'poly':
            scheduler = PolyLR(optimizer, epochs, power=0.9, last_epoch=start_epoch-1)
        elif lr_policy == 'cosinewm':
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=100, cycle_mult=0.5, max_lr=lr, min_lr=lr*0.001, warmup_steps=10, gamma=0.8, last_epoch=start_epoch-1)

        train_loss_list = []
        test_loss_list = []
        n = len(self.trainloader)
        m = len(self.testloader)
        test_loss = 10

        for epoch in tqdm(range(start_epoch, epochs)):
            running_loss = 0.0
            correct = 0.0
            train_acc = 0.0
            test_acc = 0.0
            total = 0.0

            for data in tqdm(self.trainloader):
                
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                _, train_pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (train_pred == labels).sum().item()

            train_acc = correct / total * 100

            train_cost = running_loss / n
            train_loss_list.append(train_cost)
            scheduler.step()

            running_loss = 0.0
            correct = 0.0
            total = 0.0
            self.model.eval()
            with torch.no_grad():
                for data in tqdm(self.testloader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
                    
                    _, test_pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (test_pred == labels).sum().item()

            test_acc = correct / total * 100

            test_cost = running_loss / m
            test_loss_list.append(test_cost)
            self.model.train()

            if epoch % 1 == 0:
                print("\nEpoch: {}, LR: {}, Train Loss: {}, Test Loss: {}, Train Accuracy: {}, Test Accuracy: {}".format(epoch, optimizer.param_groups[0]['lr'], train_cost, test_cost, train_acc, test_acc))
                tqdm.write("Epoch: {}, LR: {}, Train Loss: {}, Test Loss: {}, Train Accuracy: {}, Test Accuracy: {}".format(epoch, optimizer.param_groups[0]['lr'], train_cost, test_cost, train_acc, test_acc), file=open("./results/" + self.model_name + "_result.txt", "a"))
                torch.save({'epoch':epoch, 'model_state_dict':self.model.state_dict(), 'optimizer':optimizer.state_dict()}, self.path + self.model_name + '_checkpoint.pth')
                
            if test_acc >= test_loss:
                torch.save(self.model.state_dict(), self.path + self.model_name + '_best.pth')
                test_loss = test_acc

        torch.save(self.model.state_dict(), self.path + self.model_name + '_last.pth')
        print('Finished training.')
