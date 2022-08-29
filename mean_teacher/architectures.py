from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls
from torch.nn.utils import weight_norm
from .utils import export
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


@export
class MyResNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=pretrained, isL2=False, double_output=False, ):
        super(MyResNet, self).__init__()
        model = resnet50(pretrained=pretrained)
        self.double_output = double_output
        self.backbone_block = nn.Sequential(*list(model.children())[:-3])
        self.cov = nn.Sequential(*list(model.children())[-3:-1])
        self.fc1 = weight_norm(nn.Linear(2048, num_classes, bias=True))
        self.fc2 = weight_norm(nn.Linear(2048, num_classes, bias=True))
        # MLP = project1+fc1
        self.project1 = nn.Sequential(
            nn.Linear(2048, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, iss=False):
        if iss:
            return self.fc1(self.project1(x))
        x = self.backbone_block(x)
        x = self.cov(x)
        out = x.reshape(x.size(0), -1)

        z = self.fc1(self.project1(out))
        if self.double_output:
            return self.fc1(out), self.fc2(out), z, out
        else:
            return self.fc1(out), out


@export
def cifar_cnn(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN(**kwargs)
    return model


class CNN(nn.Module):
    """
    CNN from Mean Teacher paper
    """

    def __init__(self, num_classes=10, isL2=False, double_output=False, isSwitch=False):
        super(CNN, self).__init__()

        self.isL2 = isL2
        self.double_output = double_output
        self.isSwitch = isSwitch

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        #
        self.drop1 = nn.Dropout(0.3)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop2 = nn.Dropout(0.5)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        # MLP = project1+fc1
        self.project1 = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # nn.Linear(128, 128, bias=True),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 128, bias=True),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),

        )

        self.fc1 = weight_norm(nn.Linear(128, num_classes))

        if self.double_output:
            self.fc2 = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x, x2=None, debug=False, iss=False):
        if iss:
            return self.fc1(self.project_1(x))
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.drop1(x)
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.drop1(x)
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.drop1(x)
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.drop1(x)
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop1(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.drop1(x)
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.drop1(x)
        x = self.activation(self.bn3c(self.conv3c(x)))
        out1 = self.ap3(x)
        out1 = out1.view(-1, 128)
        if self.isL2:
            out1 = F.normalize(out1)
        z = self.fc1(self.project1(out1))
        if self.double_output:
            return self.fc1(out1), self.fc2(out1), z, out1
        else:
            return self.fc1(out1), out1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, isL2=False, double_output=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.double_output = double_output
        self.isL2 = isL2
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        if self.double_output:
            self.fc2 = weight_norm(nn.Linear(512 * block.expansion, num_classes))
        # MLP
        self.project1 = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x, iss=False):
        if iss:
            return self.fc1(self.project1(x))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        out1 = x.reshape(x.size(0), -1)
        if self.isL2:
            out1 = F.normalize(out1)

        z = self.fc1(self.project1(out1))
        if self.double_output:
            return self.fc1(out1), self.fc2(out1), z, out1
        else:
            return self.fc1(out1), out1


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


@export
def resnet18_(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
