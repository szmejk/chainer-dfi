# Copyright (c) 2016 Tomoto Yusuke
# Released under MIT license.
# https://github.com/yusuketomoto/chainer-fast-neuralstyle

from chainer import link
from chainer.links.caffe import CaffeFunction
from chainer import serializers
import sys
from net import VGG19

def copy_model(src, dst):
    global test
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        child.name = child.name.replace("/", "_")
        child.name = child.name.replace("expand", "ex")
        child.name = child.name.replace("scale", "sc")
        child.name = child.name.replace("linear", "ln")
        child.name = child.name.replace("dwise", "dw")
        if child.name not in dst.__dict__:
                print "Child name not found in destination + " + str(child.name)
                continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                # print(a[1].data)
                b[1].data = a[1].data
            print 'Copy %s' % child.name

print 'load MobileNet caffemodel'
ref = CaffeFunction('mobilenet_v2.caffemodel')
vgg = VGG19()
print 'copy weights'
copy_model(ref, vgg)

print 'save "mobilenetv2.model"'
serializers.save_npz('mobilenetv2.model', vgg)
