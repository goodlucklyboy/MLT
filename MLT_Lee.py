# -*- coding: utf-8 -*-
#########################################################################
###                                                                   ###
###                                                                   ###
###                                                                   ###
###    此版本多任务学习代码有误，源领域和目标领域共享一套输出神经元，需修改         ###
###                                                                   ###
#########################################################################
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset #All other datasets should subclass it .and override _len_ and _getitem
from torch.autograd import Variable

import torchvision
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image

from Assistant import format_time, loss_curve, show_sample_classification
plt.ion()   # interactive mode

print('===>> Program start, running ID :MTL001 <<===')
print('===>> Program start, running ID :MTL001 <<===')
print('===>> Program start, running ID :MTL001 <<===')
print('===>> Program start, running ID :MTL001 <<===')
##########################################################################
#################    Result saved setting       ##########################
##########################################################################
save_name= '01_multi_task_basic.txt'
save_dir = './result/'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

##########################################################################
#################   1/  HYPER PARAMETERS        ##########################
##########################################################################

LR = 0.0001
STEP_SIZE = 700
EPOCH_NUM = 2000
best_acc  = 0
BATCH_SIZE_SOURCE = 48
BATCH_SIZE_TARGET = 24
result = np.zeros([EPOCH_NUM,12])
torch.cuda.empty_cache()

if torch.cuda.is_available():
	device = "cuda"
	torch.cuda.empty_cache()
else:
	device = "cpu"
##########################################################################
#################       DATASET         ##########################
##########################################################################
print('===>> Preparing data <<===')

class MTL_Dataset(Dataset):#这个类，并没有用到。
	"""
	Train: For each sample creates randomly a positive or a negative pair
	Test: Creates fixed pairs for testing
	"""

	def __init__(self, dataset_source,dataset_target):
		self.dataset_source = dataset_source
		self.dataset_target = dataset_target

		self.data_len_s = len(self.dataset_source)
		self.data_len_t = len(self.dataset_target)

		self.sample_size = self.dataset_source[0][0].shape
		self.train_data = [0]*self.data_len
		self.train_labels = torch.Tensor(self.data_len)
		self.transform = self.dataset.transform

		for i in range(len(self.dataset)):
			self.train_data[i] = self.dataset.imgs[i][0]
			self.train_labels[i] = self.dataset[i][1]

		self.labels_set = set(self.train_labels.numpy())

		self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
								 for label in self.labels_set}
		print('haha')
		print(self.label_to_indices)

##覆盖定义数据集的迭代器
	def __getitem__(self, index):

		target = np.random.randint(0, 2)
		img1, label1 = self.train_data[index], self.train_labels[index].item()
		if target == 1:
			siamese_index = index
			while siamese_index == index:
				siamese_index = np.random.choice(self.label_to_indices[label1])
		else:
			siamese_label = np.random.choice(list(self.labels_set - set([label1])))
			siamese_index = np.random.choice(self.label_to_indices[siamese_label])
		img2 = self.train_data[siamese_index]


		img1 = Image.open(img1)
		img2 = Image.open(img2)
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
		return (img1, img2), target

	def __len__(self):
		return len(self.dataset)


Source_Data_Dir = './Datasets/image'
Target_Data_Dir = './Datasets/ori_1_expand'

transform_train = transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

transform_val = transforms.Compose((
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
))

Source_train_dataset = torchvision.datasets.ImageFolder(root=Source_Data_Dir+'/train/', transform=transform_train)
Source_valid_dataset = torchvision.datasets.ImageFolder(root=Source_Data_Dir+'/val/', transform=transform_train)

Target_train_dataset = torchvision.datasets.ImageFolder(root=Target_Data_Dir+'/train/', transform=transform_train)
Target_valid_dataset = torchvision.datasets.ImageFolder(root=Target_Data_Dir+'/val/', transform=transform_train)


Source_train_loader = torch.utils.data.DataLoader(Source_train_dataset, batch_size=BATCH_SIZE_SOURCE, shuffle=True, num_workers=0)
Source_val_loader   = torch.utils.data.DataLoader(Source_valid_dataset, batch_size=BATCH_SIZE_SOURCE, shuffle=True, num_workers=0)

Target_train_loader = torch.utils.data.DataLoader(Target_train_dataset, batch_size=BATCH_SIZE_TARGET, shuffle=True, num_workers=0)
Target_val_loader   = torch.utils.data.DataLoader(Target_valid_dataset, batch_size=BATCH_SIZE_TARGET, shuffle=True, num_workers=0)

use_gpu = True

train_data_len = len(Target_train_loader)
val_data_len = len(Target_val_loader)

##########################################################################
###########       Network architecture and optimizer        ##############
##########################################################################
class M_VGG(nn.Module):
	def __init__(self, features, num_classes_s=1000,num_classes_t = 1000, init_weights=True):
		super(M_VGG, self).__init__()
		self.features = features
		self.classifier_s = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1024),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(1024, 256),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(256, 64),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(64, num_classes_s),
		)
		self.classifier_t = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1024),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(1024, 256),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(256, 64),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(64, 10),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(10, num_classes_s)
		)
		if init_weights:
			self._initialize_weights()
	
	def forward(self, x_s, x_t):
		x_s = self.features(x_s)
		x_t = self.features(x_t)
		x_s = x_s.view(x_s.size(0),-1)
		x_t = x_t.view(x_t.size(0),-1)
		x_s = self.classifier_s(x_s)
		x_t = self.classifier_t(x_t)
		return x_s, x_t

	def _initialize_weights(self):
		return models.VGG._initialize_weights(self)
	
	
def vgg11(source_class_num =1000,target_class_num = 1000):
	model = M_VGG(models.vgg.make_layers(models.vgg.cfg['A'], True),source_class_num,target_class_num)
	return model

net = vgg11(10,2)
# for param in net.parameters():
#     param.requires_grad = False

#net.classifier._modules['6']= nn.Linear(4096,52)   #输出神经元个数为源领域类别数目+目标领域类别数目

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(net.parameters(), lr=LR, momentum=0.9) # only parameters of final layer are being optimized as
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=STEP_SIZE, gamma=0.1)      # Decay LR by a factor of 0.1 every 7 epochs


##########################################################################
###################       Assistant function           ###################
##########################################################################


##########################################################################
###################       Training and validate        ###################
##########################################################################

def Train_MTL(model, epoch, criterion, optimizer, scheduler):
	curve = loss_curve(epoch)
	since = time.time()
	print('===>> Start to Training <<===')

	for current_epoch_num in range(epoch):
		print('\nCurrent Epoch: %d' % current_epoch_num)

		#############################################################################
		########            Training stage    ######## ######## ######## ########
		model.train()
		if scheduler is not None:
			scheduler.step()
		train_loss = 0
		train_loss_s = 0
		train_loss_t = 0
		correct = 0
		correct_s = 0
		correct_t = 0
		total = 0
		total_s = 0
		total_t = 0

		Source_iter = iter(Source_train_loader)
		Target_iter = iter(Target_train_loader)


		for batch_idx in range(train_data_len):

			input_s, label_s = next(Source_iter)
			input_t, label_t = next(Target_iter)

			input_s ,label_s = input_s.to(device) , label_s.to(device)
			input_t, label_t = input_t.to(device) , label_t.to(device)
			start_time = time.time()

			# inputs = torch.cat((inputs_s,inputs_t),0)
			# targets = torch.cat((targets_s,targets_t),0)

			optimizer.zero_grad()
			output_s,output_t = model(input_s,input_t)
			loss_s = criterion(output_s, label_s)
			loss_t = criterion(output_t, label_t)
			loss = loss_s+loss_t

			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			train_loss_s += loss_s.item()
			train_loss_t += loss_t.item()

			_, predicted_s = output_s.max(1)
			_, predicted_t = output_t.max(1)

			total += label_s.size(0)+label_t.size(0)
			total_s += label_s.size(0)
			total_t += label_t.size(0)

			correct += predicted_s.eq(label_s).sum().item()
			correct += predicted_t.eq(label_t).sum().item()

			correct_s += predicted_s.eq(label_s).sum().item()
			correct_t += predicted_t.eq(label_t).sum().item()

			batch_time = time.time() - start_time
			print('Training| Epoch: %d | batch num: %d | Loss: %.3f | Loss_s: %.3f | Loss_t: %.3f | Acc: %.3f | Acc_s: %.3f | Acc_t: %.3f | Time: %s'
				  % (current_epoch_num, batch_idx, train_loss / (batch_idx + 1), train_loss_s / (batch_idx + 1), train_loss_t / (batch_idx + 1),
					 100. * correct / total, 100. * correct_s / total_s, 100. * correct_t / total_t,
					 format_time(batch_time)))

		curve.train_loss[current_epoch_num] = train_loss / (train_data_len+1)
		curve.train_acc[current_epoch_num] = correct / total
		# progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		#     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

		#############################################################################
		########            Validating stage
		global best_acc

		model.eval()
		val_loss = 0
		val_loss_s = 0
		val_loss_t = 0
		correct = 0
		correct_s = 0
		correct_t = 0
		total = 0
		total_s = 0
		total_t = 0

		Source_iter = iter(Source_val_loader)
		Target_iter = iter(Target_val_loader)

		with torch.no_grad():
			for batch_idx in range(val_data_len):
				input_s, label_s = next(Source_iter)
				input_t, label_t = next(Target_iter)

				input_s, label_s = input_s.to(device), label_s.to(device)
				input_t, label_t = input_t.to(device), label_t.to(device)

				start_time = time.time()

				output_s, output_t = model(input_s,input_t)
				loss_s = criterion(output_s, label_s)
				loss_t = criterion(output_t, label_t)

				loss = loss_s + loss_t

				val_loss += loss.item()
				val_loss_s += loss_s.item()
				val_loss_t += loss_t.item()

				_, predicted_s = output_s.max(1)
				_, predicted_t = output_t.max(1)

				total += output_s.size(0) + output_t.size(0)
				total_s += label_s.size(0)
				total_t += label_t.size(0)

				correct += predicted_s.eq(label_s).sum().item()
				correct += predicted_t.eq(label_t).sum().item()

				correct_s += predicted_s.eq(label_s).sum().item()
				correct_t += predicted_t.eq(label_t).sum().item()

				batch_time = time.time() - start_time
				print('validating | Epoch: %d | batch num: %d | Loss: %.3f | Loss_s: %.3f | Loss_t: %.3f | Acc: %.3f | Acc_s: %.3f | Acc_t: %.3f | Time: %s'
				  % (current_epoch_num, batch_idx, val_loss / (batch_idx + 1), val_loss_s / (batch_idx + 1), val_loss_t / (batch_idx + 1),
					 100. * correct / total, 100. * correct_s / total_s, 100. * correct_t / total_t,
					 format_time(batch_time)))
				# progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				#     % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
		# Save checkpoint.
		acc = 100. * correct / total
		if acc > best_acc:
			print('Saving..')
			state = {
				'net': net.state_dict(),
				'acc': acc,
				'epoch': epoch,
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')
			torch.save(state, './checkpoint/ckpt.t7')
			best_acc = acc
		#
		curve.val_loss[current_epoch_num] = val_loss / (train_data_len+1)
		curve.val_acc[current_epoch_num] = correct / total
		if current_epoch_num % 10 == 0 :
			curve.display()
	total_time = time.time()-since
	print("总时间为：",total_time)
	return [model, curve]
##########################################################################
#########################       Main        ##############################
##########################################################################


torch.cuda.empty_cache()

######################################################################

print('===>> Start to trainning <<===')
print('===>> Start to trainning <<===')
print('===>> Start to trainning <<===')


# Train and evaluate
time_start = time.time()
net, _ = Train_MTL(net, EPOCH_NUM,criterion, optimizer_conv, exp_lr_scheduler)

time_cost_final = time.time()-time_start


torch.save(net,save_dir+save_name+'.pkl')
# plt.ion()
# plt.figure(5)
# plt.title('fine-tune the finally layer')
# plt.xlabel('epoch num')
# plt.ylabel('loss value value')
# plt.plot(curveft_loss_tr,'r',label='train loss')
# plt.plot(curveft_loss_val,'deeppink',label='val loss')
# plt.plot(curveft_acc_tr,'g',label='train accuracy')
# plt.plot(curveft_acc_val,'greenyellow',label='val accuracy')
# plt.legend()
# plt.show()

torch.cuda.empty_cache()


##########################################################################################################
##########   Training finished, saving result and display result     #####################################
##########################################################################################################

torch.cuda.empty_cache()
plt.waitforbuttonpress()
