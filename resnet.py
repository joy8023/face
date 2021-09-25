import torch
import numpy as np
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout, MaxPool2d, Sequential, Module
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Support: ['ResNet_50', 'ResNet_101', 'ResNet_152']
class Fawkes(Dataset):
    def __init__(self, path):

        #self.path = path
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = np.load(path)

        images = self.dataset['images']
        fawkes = self.dataset['fawkes']
        self.labels = self.dataset['labels']
        
        self.images = images/255.0
        self.fawkes = fawkes/225.0

    def get_label(self):
        return self.labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img, target = self.images[index], self.fawkes[index]
        img = self.transform(img)
        target = self.transform(target)
        return img, target

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""

    return Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)
def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""

    return Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)

class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(Module):

    def __init__(self, input_size, block, layers, zero_init_residual = True):
        super(ResNet, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.bn_o1 = BatchNorm2d(2048)
        self.dropout = Dropout()
        if input_size[0] == 112:
            self.fc = Linear(2048 * 4 * 4, 512)
        else:
            self.fc = Linear(2048 * 8 * 8, 512)
        self.bn_o2 = BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_o1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_o2(x)

        return x

def ResNet_18(input_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, Bottleneck, [2, 2, 2, 2], **kwargs)

    return model
def ResNet_50(input_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model
def ResNet_101(input_size, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model
def ResNet_152(input_size, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 8, 36, 3], **kwargs)

    return model
    
# normalize image to [-1,1]
normalize = transforms.Compose([
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def normalize_batch(imgs_tensor):
    normalized_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        normalized_imgs[i] = normalize(img_ten)

    return normalized_imgs

def l2_norm(input, axis = 1):
    # normalizes input with respect to second norm
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class feature_extractor(nn.Module):
    def __init__(self, model):
        super(feature_extractor, self).__init__()
        self.model = model

    def forward(self, input):
        self.model.eval()
        batch_normalized = normalize_batch(input)
        features = l2_norm(self.model(batch_normalized))
        return features

def load_resnet(model_root, input_size = [112,112]):

    model = ResNet_152(input_size)
    print("Loading Attack Backbone Checkpoint '{}'".format(model_root))
    model.load_state_dict(torch.load(model_root))
        
    feature_extractor_model = nn.DataParallel(
            nn.Sequential(feature_extractor(model=model))).to(device)

    return feature_extractor_model, device


def get_feature_resnet(datapath):

    model, device = load_resnet('model/Backbone_ResNet_152_Arcface_Epoch_65.pth')
    batch_size = 128
    #images, fawkes, labels = load_data(datapath)
    
    dataset = Fawkes(datapath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    #print(images.shape)
    labels = dataset.get_label()

    image_features = []
    fawkes_features = []

    model.eval()
    with torch.no_grad():
        for batch, (images, fawkes) in enumerate(loader):
            images = images.to(device, dtype=torch.float)
            fawkes = fawkes.to(device, dtype=torch.float)

            image_f = model(images).to('cpu').numpy()
            fawkes_f = model(fawkes).to('cpu').numpy()

            image_features.append(image_f)
            fawkes_features.append(fawkes_f)

    image_features = np.concatenate(image_features, axis = 0)
    fawkes_features = np.concatenate(fawkes_features, axis = 0)
    
    print(image_features.shape)

    return image_features, fawkes_features, labels

