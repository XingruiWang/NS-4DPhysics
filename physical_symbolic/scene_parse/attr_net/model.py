import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models


PYTORCH_VER = torch.__version__



class AttributeNet(nn.Module):

    def __init__(self, opt, output_dim, input_channels=3):
        super(AttributeNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        
        # remove the last layer
        layers.pop()

        # change the channel of first layer
        if input_channels != 3:
            layers.pop(0)
            layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)

        self.fc_shape = nn.Linear(512, output_dim['shape'])
        self.fc_color = nn.Linear(512, output_dim['color'])
        self.fc_size = nn.Linear(512, output_dim['size'])
        self.fc_material = nn.Linear(512, output_dim['material'])

        self.shape_ce = nn.CrossEntropyLoss()
        self.color_ce = nn.CrossEntropyLoss()
        self.material_ce = nn.CrossEntropyLoss()
        self.size_ce = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        shape = self.fc_shape(x)
        color = self.fc_color(x)
        size = self.fc_size(x)
        material = self.fc_material(x)
        return (shape, color, material, size)

    def loss(self, y, target):
        shape, color, material, size = y
        shape_gt, color_gt, material_gt, size_gt = target
        shape_gt = shape_gt.cuda()
        color_gt = color_gt.cuda()
        material_gt = material_gt.cuda()
        size_gt = size_gt.cuda()


        loss_shape = self.shape_ce(shape, shape_gt)
        loss_color = self.color_ce(color, color_gt)
        loss_material = self.material_ce(material, material_gt)
        loss_size = self.size_ce(size, size_gt)
        
        loss = loss_shape + loss_color + loss_material + loss_size

        return loss

    def accuracy(self, y, target):
        bs = y[0].size(0)
        shape, color, material, size = y
        shape_gt, color_gt, material_gt, size_gt = target

        shape_acc = sum(torch.argmax(shape.data, dim=1) == shape_gt.cuda()) / bs
        color_acc = sum(torch.argmax(color.data, dim=1) == color_gt.cuda()) / bs
        material_acc = sum(torch.argmax(material.data, dim=1) == material_gt.cuda()) / bs
        size_acc = sum(torch.argmax(size.data, dim=1) == size_gt.cuda()) / bs
        
        return (shape_acc, color_acc, material_acc, size_acc)

    def save_checkpoint(self, save_path, best_acc, optimizer):
        checkpoint = {
            'model_state': self.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)



def get_model(opt):
    output_dim = {
        'shape': 129,
        'color': 8,
        'material': 2,
        'size': 2,
    }
    model = AttributeNet(opt, output_dim, input_channels=4)

    return model

if __name__ == '__main__':
    output_dim = {
        'shape': 129,
        'color': 8,
        'material': 2,
        'size': 2,
    }
    model = get_model(None)

    x = torch.randn((1, 3, 224, 224))

    
    target = (torch.tensor([0]).long(), torch.tensor([3]).long(), torch.tensor([1]).long(), torch.tensor([0]).long())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for e in range(10):
        optimizer.zero_grad()
        y = model(x)
        loss = model.loss(y, target)
        print(loss.data)
        loss.backward()
        optimizer.step()
        
