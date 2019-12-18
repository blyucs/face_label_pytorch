import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        '''
        # SSD network
        self.vgg = nn.ModuleList(base)
        '''
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

#        if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
'''
        # apply vgg up to conv4_3 relu U should define de forward of modulelist yourself
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
'''
        C1,C2,C3,C4,C5 = resnet_graph_18(x,stage5=True)
        # apply extra layers and cache source layer outputs

        P5 = KL.Conv2D(64, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p5transpose")(P5),
            KL.Conv2D(64, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p4transpose")(P4),
            KL.Conv2D(64, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p3transpose")(P3),
            KL.Conv2D(64, (1, 1), name='fpn_c2p2')(C2)])
        PF = P1 = KL.Add(name="fpn_p1add")([
            KL.Conv2DTranspose(64, (2, 2), strides=2, kernel_initializer='he_normal', name="fpn_p2transpose")(P2),
            KL.Conv2D(64, (1, 1), name='fpn_c1p1')(C1)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        P1 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p1")(P1)  # 256*256
        P2 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p2")(P2)  # 128*128
        P3 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p3")(P3)  # 64*64
        # P4 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_p4")(P4) #32*32

        PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF1", activation='relu')(PF)
        PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF2", activation='relu')(PF)
        PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF3", activation='relu')(PF)
        PF = KL.Conv2DTranspose(64, (2, 2), strides=2, name='DPF1', activation="relu")(PF)
        PF = KL.Conv2D(64, (3, 3), padding="SAME", name="PF4", activation='relu')(PF)
        PF = KL.Conv2D(3, (1, 1), padding="SAME", name="PF_out", activation='softmax')(PF)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  # permute(pailie) the dimensions of this tensor
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
'''     if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),  # permute to n (batch)*any*4(box)
                conf.view(conf.size(0), -1, self.num_classes),  #permute to n(batch) * any * (classes)  confidence
                self.priors
            )
'''
        output = self.detect(
            loc.view(loc.size(0), -1, 4),  # loc preds
            self.softmax(conf.view(conf.size(0), -1,
                                   self.num_classes)),  # conf preds
            self.priors.type(type(x.data))  # default boxes
        )
        return output

'''
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers
'''

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    #remove vgg
    '''
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        #conv2d(input_channel,output_channel,kernel_size,padding)
        #cfg[k] is the box number of the layer
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    '''
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return extra_layers, (loc_layers, conf_layers)

'''
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
'''
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}
# move inside of the class
def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):

    nb_filter1, nb_filter2 = filters

    x = nn.Conv2d(nb_filter1, (1, 1), strides=strides,)(input_tensor)

    x = nn.ReLU(x)

    x = nn.Conv2d(nb_filter2, (kernel_size, kernel_size))(x)

    shortcut = nn.Conv2d(nb_filter2, (1, 1), strides=strides)(input_tensor)

    x += shortcut
    x = nn.ReLU(x)
    return x

#move inside of the class
def identity_block(input_tensor, kernel_size, filters,stage, block,
                   use_bias=True):

    nb_filter1, nb_filter2 = filters

    x = nn.Conv2d(nb_filter1, (1, 1), )(input_tensor)
    x = nn.ReLU(x)

    x = nn.Conv2d(nb_filter2, (kernel_size, kernel_size))(x)
    x += input_tensor
    x = nn.ReLU(x)
    return x

def resnet_graph_18(input_image, stage5=False):

    # Stage 1
    x = nn.ZeroPad2d((3))(input_image)
    x = nn.Conv2d(3, 32, (7, 7), stride=(2, 2))(x)
    C1 = x = nn.ReLU(x)

    x = nn.MaxPool2d(3, strides=2)(x)
    # Stage 2
    x = conv_block(x, 3, [32, 32], stage=2, block='a',strides=(1,1))
    C2 = x = identity_block(x, 3, [32, 32], stage=2, block='b')

    # Stage 3
    x = conv_block(x, 3, [64, 64], stage=3, block='a')
    C3 = x = identity_block(x, 3, [64, 64], stage=3, block='b')
    # Stage 4
    x = conv_block(x, 3, [128, 128], stage=4, block='a')
    C4 = x = identity_block(x, 3, [128, 128], stage=4, block='b')

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [256, 256], stage=5, block='a')
        C5 = x = identity_block(x, 3, [256, 256], stage=5, block='b')
    else:
        C5 = None
    return C1, C2, C3, C4, C5

# this can be move inside of the SSD class
def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    #base is vgg  to get output  ---vgg has been removed
    #extras_ is the extra conv layer after vgg  got by add_extras(300/512) to get PC1/PC2/PC3/PC4/PC5/PC6/PC7
    # head_ is (loc_layers, conf_layers)
    extras_, head_ = multibox(add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, extras_, head_, num_classes)
