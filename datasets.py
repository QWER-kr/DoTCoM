import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader

def dataloader(data_root, image_size, crop_size, batch_size):
    transf = tr.Compose([tr.RandomResizedCrop(crop_size), tr.RandomHorizontalFlip(), tr.ToTensor(), tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transf = tr.Compose([tr.Resize(image_size), tr.CenterCrop(crop_size), tr.ToTensor(), tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    trainset = torchvision.datasets.ImageNet(data_root, split='train', transform=transf)
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)
 
    testset = torchvision.datasets.ImageNet(data_root, split='val', transform=test_transf)
    testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory = True)

    return trainloader, testloader
