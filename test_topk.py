import torch
from utils import *

def eval(testloader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
        
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            pred1, pred5 = accuracy(outputs, labels, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)
        
        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    acc = 100 * correct / total 

    return acc, losses.avg, top1.avg, top5.avg
