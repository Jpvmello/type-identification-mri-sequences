import torchvision
import torch
import torch.nn as nn
from torchvision.models.video.resnet import BasicStem

def select_net(architecture, n_slices, is_3d, consider_other_class):
	num_classes = 5 if consider_other_class else 4
	if architecture == 'resnet18':
		return resnet18(n_slices, num_classes, is_3d)
	if architecture == 'alexnet':
		return alexnet(n_slices, num_classes)
	if architecture == 'vgg':
		return vgg(n_slices, num_classes)
	if architecture == 'squeezenet':
		return squeezenet(n_slices, num_classes)
	if architecture == 'mobilenet':
		return mobilenet(n_slices, num_classes)

def resnet18(n_slices, num_classes, is_3d):
	if is_3d:
		net = torchvision.models.video.r3d_18(num_classes = num_classes)
		net.stem = nn.Sequential(
		nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
		nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		nn.ReLU(inplace=True)
	)
	else:
		net = torchvision.models.resnet18(num_classes = num_classes)
		net.conv1 = nn.Conv2d(n_slices, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 
	#net.layer3 = nn.Identity() 
	#net.layer4 = nn.Identity()
	net.fc = nn.Linear(in_features = 512, out_features = num_classes, bias = True)
	return net

def alexnet(n_slices, num_classes):
	net = torchvision.models.alexnet(num_classes = num_classes)
	net.features[0] = nn.Conv2d(n_slices, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
	net.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
	return net
	
def vgg(n_slices, num_classes):
	net = torchvision.models.vgg16(num_classes = num_classes)
	net.features[0] = nn.Conv2d(n_slices, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	net.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
	return net
	
def squeezenet(n_slices, num_classes):
	net = torchvision.models.vgg16(num_classes = num_classes)
	net.features[0] = nn.Conv2d(n_slices, 64, kernel_size=(3, 3), stride=(2, 2))
	#net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
	return net
	
def mobilenet(n_slices, num_classes):
	net = torchvision.models.mobilenet_v2(num_classes = num_classes)
	net.features[0][0] = nn.Conv2d(n_slices, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
	#net.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
	return net


class Net(nn.Module):
	def __init__(self, n_slices):
		super(Net, self).__init__()
		self.axial = torchvision.models.resnet18(num_classes = 5)
		self.axial.conv1 = nn.Conv2d(n_slices, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.axial.layer3 = nn.Identity()
		self.axial.layer4 = nn.Identity()
		self.axial.fc = nn.Identity()

		self.longitud = torchvision.models.resnet18(num_classes = 5)
		self.longitud.conv1 = nn.Conv2d(200, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.longitud.layer3 = nn.Identity()
		self.longitud.layer4 = nn.Identity()
		self.longitud.fc = nn.Identity()

		self.fc = nn.Linear(in_features = 256, out_features = 5, bias = True)

	def forward(self, x):
		ax = self.axial(x)
		lg = self.longitud(torch.transpose(x, 1, 3))
		return self.fc(torch.cat((ax, lg), 1))
