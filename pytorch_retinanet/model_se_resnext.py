from collections import OrderedDict

import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from .utils import BBoxTransform, ClipBoxes
from .anchors import Anchors
from . import losses

from pretrainedmodels.models import senet
from .model import PyramidFeatures, RegressionModel, ClassificationModel, nms


class SeResNetXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        block = senet.SEResNeXtBottleneck
        layers = [3, 4, 23, 3]
        groups = 32
        reduction = 16
        inplanes = 64
        downsample_kernel_size = 1
        downsample_padding = 0

        self.inplanes = inplanes

        layer0_modules = [
            ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(inplanes)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))
        ]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        fpn_sizes = [
            self.layer1[layers[0] - 1].conv3.out_channels,
            self.layer2[layers[1] - 1].conv3.out_channels,
            self.layer3[layers[2] - 1].conv3.out_channels,
            self.layer4[layers[3] - 1].conv3.out_channels
        ]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors(pyramid_levels=[2, 3, 4, 5, 6, 7])

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_encoder(self):
        for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            layer.eval()

    def forward(self, inputs, return_loss, return_boxes):

        if return_loss:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = torch.cat([img_batch, img_batch, img_batch], dim=1)
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x1, x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        res = []

        if return_loss:
            res += list(self.focalLoss(classification, regression, anchors, annotations))

        if return_boxes:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.4)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            res += [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

        return res


def se_resnext101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SeResNetXt(num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            senet.pretrained_settings['se_resnext101_32x4d']['imagenet']['url'], model_dir='models'), strict=False)
    return model

