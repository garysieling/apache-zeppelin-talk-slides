from mxnet import gluon
from mxnet import nd
from mxboard import SummaryWriter
import argparse
import logging
import datetime

parser = argparse.ArgumentParser(description='Appliance Recognizer')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Train on GPU with CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

opt = parser.parse_args()

#sw = SummaryWriter(logdir='/data/logs-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), flush_secs=5)
logging.basicConfig(level=logging.DEBUG)

import os
idx = 0
category_to_idx = {}
for root, dirs, files in sorted(os.walk('/data/household/train/')):
  category = root.split("/")[-1]
  print(category)
  if category not in category_to_idx and category != '':
      category_to_idx[category] = idx
      idx = idx + 1
      
print(category_to_idx)

train_map = {}

for key in category_to_idx:
  train_map[key] = []

import copy
test_map = copy.deepcopy(train_map)
validation_map = copy.deepcopy(train_map)
sample_map = copy.deepcopy(train_map)

def counts(data_dir, map):
  counter = 0
  for root, dirs, files in os.walk(data_dir):
    category = root.split("/")[-1]
    for f in files:
      if (category in category_to_idx):
        map[category].append({
          'idx': counter,
          'label': category_to_idx[category],
          'filename': f,
          'filepath': root + '/' + f
        })
    counter += 1

counts('/data/household/sample/', sample_map)
counts('/data/household/train/', train_map)
counts('/data/household/test/', test_map)
counts('/data/household/validation/', validation_map)

f = open("/data/classes.tsv", "w")
print("%table type\tclass\tcount", file=f)
for k in train_map:
    print("train\t" + k + "\t" + str(len(train_map[k])), file=f)
    
for k in test_map:
    print("test\t" + k + "\t" + str(len(test_map[k])), file=f)
    
for k in validation_map:
    print("validation\t" + k + "\t" + str(len(validation_map[k])), file=f)
    
for k in sample_map:
    print("sample\t" + k + "\t" + str(len(sample_map[k])), file=f)    
f.close()

#from mxnet.gluon.model_zoo.vision import mobilenet_v2_1_0
from mxnet.gluon.model_zoo.vision import vgg16
#from mxnet.gluon.model_zoo.vision import resnet50_v1
#from mxnet.gluon.model_zoo.vision import resnet152_v2
#pretrained_net = resnet50_v1(pretrained=True)
#initNet = mobilenet_v2_1_0
initNet = vgg16
pretrained_net = initNet(pretrained=True)
print(pretrained_net)

print("classes: " + str(len(train_map)))
net = initNet(classes=len(train_map))

from mxnet import init

net.features = pretrained_net.features
net.output.initialize(init.Xavier())

from mxnet.image import color_normalize
from mxnet import image

eigval = [55.46, 4.794, 1.148]
eigvec = [[-0.5675, 0.7192, 0.4009],[-0.5808, -0.0045, -0.8140],[-0.5836, -0.6948, 0.4203]]

train_augs = [
    image.HorizontalFlipAug(0.5),
    image.LightingAug(1, eigval, eigvec),
    image.BrightnessJitterAug(.3),
    image.HueJitterAug(.05),
    image.ForceResizeAug((224, 224))
]

test_augs = [
    image.HorizontalFlipAug(0.5),
    image.LightingAug(1, eigval, eigvec),
    image.BrightnessJitterAug(.3),
    image.HueJitterAug(.05),
    image.ForceResizeAug((224, 224)),
    image.CenterCropAug((224, 224))
]


def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))

    return data, nd.array([label]).asscalar().astype('float32')

from mxnet.gluon.data.vision import ImageFolderDataset

train_rec = '/data/household/train/'
validation_rec = '/data/household/validation/'
test_rec = '/data/household/test/'
sample_rec = '/data/household/sample/'

trainIterator = ImageFolderDataset(
    root=train_rec, 
    transform=lambda X, y: transform(X, y, train_augs)
)

validationIterator = ImageFolderDataset(
    root=validation_rec,
    transform=lambda X, y: transform(X, y, test_augs)
)

testIterator = ImageFolderDataset(
    root=test_rec,
    transform=lambda X, y: transform(X, y, test_augs)
)

sampleIteratorTrain = ImageFolderDataset(
    root=sample_rec,
    transform=lambda X, y: transform(X, y, train_augs)
)

sampleIteratorTest = ImageFolderDataset(
    root=sample_rec,
    transform=lambda X, y: transform(X, y, test_augs)
)

import time
from mxnet.image import color_normalize
from mxnet import autograd
import mxnet as mx
from mxnet import nd

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        data = color_normalize(data/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))

        output = net(data)
        prediction = nd.argmax(output, axis=1)
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]

def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])

def log(text, f):
    print(text)
    print(text, file=f)
    f.flush()

def train_util(net, train_iter, test_iter, validation_iter, loss_fn, trainer, ctx, epochs, batch_size):
    global_step = 0
    f = open("/data/training.csv", "w")
    log("Epoch\tBatch\tSpeed\tTraining Accuracy\tTest Accuracy\tValidation Accuracy", f)
    metric = mx.metric.create(['acc'])
    for epoch in range(epochs):
        print("Starting epoch " + str(epoch))
        for i, (data, label) in enumerate(train_iter):
            print("Epoch " + str(epoch) + ", Iteration " + str(i))
            st = time.time()
            
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            data = color_normalize(data/255,
                                   mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                                   std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))

            #if epoch == 0 and i < 100:
            #    print("Recording sample images")
            #    img = data.clip(0, 1)
            #    sw.add_image('appliances_minibatch_' + str(epoch) + '_' + str(i), img.reshape((opt.batch_size, 3, 224, 224)), epoch)

            print("Computing loss")
            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)

            #print("Saving cross_entropy")
            #sw.add_scalar(tag='cross_entropy', value=loss.mean().asscalar(), global_step=global_step)
            #global_step += 1

            print("Backpropagation")
            loss.backward()
            trainer.step(data.shape[0])
            
            print("Updating metrics")
            metric.update([label], [output])
            names, accs = metric.get()
            log('%d\t%d\t%f\t%s'%(epoch, i, batch_size/(time.time()-st), str(accs[0])), f)


            if i%10 == 0 and i > 0:
                print("Computing periodic metrics")
                #train_acc = evaluate_accuracy(train_iter, net)
                #test_acc = evaluate_accuracy(test_iter, net)
                validation_acc = evaluate_accuracy(validation_iter, net)
                log('%d\t%d\t%f\t%s\t%s'%(epoch, i, batch_size/(time.time()-st), str(accs[0]), validation_acc), f)
#               print("%s\t%d\t%s | test_acc %s " % (epoch, i, train_acc, test_acc), file = f)
                net.collect_params().save('/data/checkpoints/%d-%d.params'%(epoch, i))
                #net.save_parameters('/data/checkpoints/%d-%d.params'%(epoch, i))
                #net.hybridize()
                #net.export("/data/checkpoints/%d-%d.params-symbol.json")

        #if epoch == 0:
        #    print("Saving graph")
        #    sw.add_graph(net)

        print("Saving histogram")
        params = net.collect_params()
        #grads = [i.grad() for i in net.collect_params().values() if i.grad_req != 'null']
        #assert len(grads) == len(param_names)
        # logging the gradients of parameters for checking convergence
        #for name in params:
        #    grad = params[name].data()
        #    sw.add_histogram(tag=name, values=grad, global_step=epoch, bins=1000)

        #print("Logging metrics")
        #name, train_acc = metric.get()
        #print('[Epoch %d] Training: %s=%f' % (epoch, name[0], train_acc[0]))
        #sw.add_scalar(tag='accuracy_curves', value=('train_acc', train_acc[0]), global_step=epoch)

        #validation_acc = evaluate_accuracy(validation_iter, net)
        print('[Epoch %d] Validation: %s=%f' % (epoch, 'validation_acc', validation_acc))
        # logging the validation accuracy
        #sw.add_scalar(tag='accuracy_curves', value=('validation_acc', validation_acc), global_step=epoch)

        print("Saving checkpoint")
        net.collect_params().save('/data/checkpoints/%d.params'%(epoch))
        net.save_parameters('/data/checkpoints/%d.params'%(epoch))
        print("Checkpoint saved")

    #sw.export_scalars('scalar_dict.json')
    #sw.close()
    f.close()

def train_model(net, ctx, 
          batch_size=8, epochs=10, learning_rate=0.01, wd=0.001):
    print("Creating data iterators")
    train_data = gluon.data.DataLoader(
        trainIterator, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(
        testIterator, batch_size)
    validation_data = gluon.data.DataLoader(
        validationIterator, batch_size)

    print("Network setup")
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})
    
    print("Starting training")
    train_util(net, train_data, test_data, validation_data,
               loss, trainer, ctx, epochs, batch_size)


import mxnet as mx

ctx = None
if opt.cuda:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu()

train_model(net, ctx, batch_size=opt.batch_size, epochs=opt.epochs, learning_rate=opt.lr)


