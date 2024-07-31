import torch
import torch.nn as nn
import argparse
import training
import datasets
import test_topk
import dotcom
from timm.models import create_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DoTCoM train and eval pytorch script')
    parser.add_argument("--data_root", type=str, default='/home/qwer/nn/data/ImageNet/',help="path to dataset")
    parser.add_argument("--image_size", type=int, default=288, help="num classes (default: 288)")
    parser.add_argument("--crop_size", type=int, default=256, help="num classes (default: 256)")
    parser.add_argument("--num_classes", type=int, default=1000, help="num classes (default: 1000)")
    parser.add_argument('--batch_size', default = 128, type = int, help = 'batch size (default: 128)')
    parser.add_argument('--epoch', default = 100, type = int, help = 'training epoch (default: 100)')
    parser.add_argument('--optimizer', default = 'adamw', type = str, choices=['sgd', 'adam', 'adamw'], help = 'optimization algorithms (default: adamw)')
    parser.add_argument("--lr_policy", default='cosinewm', choices=['poly', 'step', 'cosinewm'], type=str, help="learning rate scheduler policy (default: cosinewm)")
    parser.add_argument('--step_size', default = 30, type = int, help = 'step size (default: 30)')
    parser.add_argument('--lr', default = 1e-3, type = float, help = 'learning rate (default: 1e-3)')
    parser.add_argument('--l2', default = 5e-3, type = float, help = 'weight decay (default: 5e-3)')
    parser.add_argument('--momentum', default = 0.9, type =float, help = 'momentum')
    parser.add_argument('--model', default = 'dotcom_large', type = str, help = 'model name')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument("--path", type=str, default='./results/', help="path to weight")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--retrain', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--dp', action='store_true', default=False)
    parser.add_argument('--ddp', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = datasets.dataloader(args.data_root, args.image_size, args.crop_size, args.batch_size)
    print('Completed loading your datasets. ')

    print(f"Creating model: {args.model}.")
    model = create_model(args.model, pretrained=False, num_classes=args.num_classes)
    print(model)
    if args.dp:
        model = nn.DataParallel(model)
    elif args.ddp:
        raise NotImplementedError("Distributed Data Parallel not yet supported.")
    else:
        model = model.to(device)
    
    learning = training.SupervisedLearning(trainloader, testloader, model, args.path, args.model, device)

    if args.pretrained:
        model.load_state_dict(torch.load(args.path + args.model))
        print('Completed you pretrained model.')

    if args.train:
        learning.train(args.epoch, args.optimizer, args.lr_policy, args.lr, args.l2, args.momentum, args.step_size, args.retrain)

    if args.eval:
        test_acc, test_loss, test_top1_acc, test_top5_acc = test_topk.eval(testloader, model)
        
        print('test_acc {test_acc:.3f} test_loss {test_loss:.3f} test_acc@1 {test_top1_acc:.3f} test_acc@5 {test_top5_acc:.3f}'
                .format(test_acc=test_acc, test_loss=test_loss, test_top1_acc=test_top1_acc, test_top5_acc=test_top5_acc))
        
