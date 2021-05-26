from __future__ import absolute_import

from .ResNet import *

__factory = {
    'ResNet50TA_BT_video' : ResNet50TA_BT_video,
    'ResNet50TA_BT_image': ResNet50TA_BT_image,
}


def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
