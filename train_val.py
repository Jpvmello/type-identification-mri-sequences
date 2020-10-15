#scikit-learn must be at version <= 0.17

import os
import argparse
import numpy
import random
import cv2
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from pandas_ml import ConfusionMatrix

from models import select_net, Net
from time_util import time_format
from MedicalDataset import MedicalDataset

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', dest='train_data_path', type = str, required = True,
		help = 'Path containing data to be trained.')
	parser.add_argument('-v', dest='val_data_path', type = str, required = True,
		help = 'Path containing validation data.')
	parser.add_argument('-m', dest='model_file', type = str, required = True,
		help = 'Name of the trained model file.')
	parser.add_argument('-bs', dest='batch_size', type = int, default = 1,
		help = 'Number of samples per batch.')
	parser.add_argument('-lr', dest='learning_rate', type = float, default = 0.001,
		help = 'Learning rate.')
	parser.add_argument('-ep', dest='epochs', type = int, default = 70,
		help = 'Number of training epochs.')
	parser.add_argument('-w', dest='weight_decay', type = float, default = 0,
		help = '')
	parser.add_argument('-sl', dest='slices', type = int, default = 10,
		help = 'Number of central slices to be considered.')
	parser.add_argument('-3d', dest='tridim', action = 'store_true',
		help = 'Determine that 3D convolution will be applied. If not specified, 2D convolution is applied considering slices as channels.')
	parser.add_argument('--no-other', dest='no_other', action = 'store_true',
		help = 'If specified, "Other" class is not considered.')
	parser.add_argument('--net', dest='net', type = str, default = 'resnet18',
		help = 'Network architecture to be used.')
	return parser.parse_args()

def fix_random_seeds():
	torch.backends.cudnn.deterministic = True
	random.seed(1)
	torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	numpy.random.seed(1)

if __name__ == '__main__':
	abs_start = time.time()
	args = parse_args()
	train_data_path = args.train_data_path
	val_data_path = args.val_data_path
	model_file = args.model_file
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	n_epochs = args.epochs
	n_slices = args.slices
	tridim = args.tridim
	decay = args.weight_decay
	consider_other_class = not args.no_other
	architecture = args.net
	
	assert(architecture in ['resnet18', 'alexnet', 'vgg', 'squeezenet', 'mobilenet'])
	os.makedirs('models', exist_ok = True)
	fix_random_seeds()
	
	#train_set = datasets.DatasetFolder(train_data_path, load_image, extensions = ('nii.gz', 'mha'), transform = augmentator())
	train_set = MedicalDataset(train_data_path, min_slices = n_slices, consider_other_class = consider_other_class)
	test_set = MedicalDataset(val_data_path, min_slices = n_slices, consider_other_class = consider_other_class, test = True)
	#test_set = datasets.DatasetFolder(val_data_path, load_image, extensions = 'nii.gz', transform = normalizer())
	train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)
	test_loader = data.DataLoader(test_set, num_workers = 8, pin_memory = True)
	#train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True)
	#test_loader = data.DataLoader(test_set, pin_memory = True)
	
	classes = ['FLAIR', 'T1', 'T1c', 'T2', 'OTHER'] #train_set.classes

	#net = Net(n_slices)
	net = select_net(architecture, n_slices, tridim, consider_other_class)
	#net = torchvision.models.vgg16(num_classes = 5)
	#net = torchvision.models.resnet101(num_classes = 5)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay = decay)
	if torch.cuda.is_available():
		net = net.cuda()
		criterion = criterion.cuda()
	else:
		exit()
		
	running_loss = 0
	epochs = n_epochs

	prev_accuracy = -1
	prev_train_loss = float('inf')
	prev_val_loss = float('inf')
	best_accuracy = {'value': -1, 'epoch': 0}
	epochs_list = range(1, epochs + 1)
	train_losses = []
	val_losses = []
	accuracies = []
	start_time = time.time()
	smallest_training_loss = {'value': float('inf'), 'non_decr_epochs': 0}
	for epoch in range(epochs):
		#train
		net.train()
		print('=== TRAINING EPOCH', str(epoch + 1), '===')
		for i, (pixel_data, label, _) in enumerate(train_loader):
			if tridim:
				pixel_data = pixel_data.view(-1, 1, 10, 200, 200)
			optimizer.zero_grad()
						
			output = net(pixel_data.cuda())
			loss = criterion(output, label.cuda())
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
		
		train_loss = running_loss/len(train_loader)
		print('Mean training loss: %.3f' % (train_loss),
			'(DECREASED!)' if (train_loss < prev_train_loss) else '(increased...)' if epoch > 0 else '')
		running_loss = 0.0
		'''if train_loss < smallest_training_loss['value']:
			smallest_training_loss['value'] = train_loss
			smallest_training_loss['non_decr_epochs'] = 0
		else:
			smallest_training_loss['non_decr_epochs'] += 1
			if smallest_training_loss['non_decr_epochs'] > 15:
				print('Train loss seems not to decrease anymore. Stopping...')
				epochs_list = range(1, epoch + 1)
				break'''
		
		#validation
		net.eval()
		correct = 0
		total = 0
		correct_per_class = [0] * len(classes)
		total_per_class = [0] * len(classes)
		actual_classes = []
		predicted_classes = []
		wrong = []
		with torch.no_grad():
			for pixel_data, label, _ in test_loader:
				label_as_num = label.numpy()[0]
				if tridim:
					pixel_data = pixel_data.view(-1, 1, 10, 200, 200)
				outputs = net(pixel_data.cuda())
				loss = criterion(outputs, label.cuda())
				running_loss += loss.item()
			
				_, predicted = torch.max(outputs.data, 1)
				total += label.size(0)
				correct += (predicted == label.cuda()).sum().item()
				total_per_class[label_as_num] += label.size(0)
				correct_per_class[label_as_num] += (predicted == label.cuda()).sum().item()
				#if predicted != label.cuda():
				#	wrong.append(path)
				
				actual_classes.append(classes[label_as_num])
				predicted_classes.append(classes[predicted.cpu().numpy()[0]])
		
		micro_accuracy = 100 * correct / total
		macro_accuracy = 0
		sampled_classes = 0
		for i in range(len(classes)):
			 if total_per_class[i] > 0:
					macro_accuracy += correct_per_class[i]/total_per_class[i]
					sampled_classes += 1
		macro_accuracy = 100 * macro_accuracy/sampled_classes
		
		accuracy = macro_accuracy
		val_loss = running_loss/len(test_loader)
		print('Validation macro-accuracy: %.3f %%' % accuracy,
			'(INCREASED!)' if (accuracy > prev_accuracy) else '(decreased...)' if epoch > 0 else '')
		print('Mean validation loss: %.3f' % (val_loss),
			'(DECREASED!)' if (val_loss < prev_val_loss) else '(increased...)' if epoch > 0 else '')
		running_loss = 0.0
		
		#time
		end_time = time.time()
		elapsed_time = end_time - start_time
		remaining_time = (epochs/(epoch+1) - 1) * elapsed_time
		print('Elapsed time:', time_format(elapsed_time))
		print('Estimated remaining time:', time_format(remaining_time))
		print()
		
		#update
		train_losses.append(train_loss)
		val_losses.append(val_loss)
		accuracies.append(accuracy)
		prev_train_loss = train_loss
		prev_val_loss = val_loss
		prev_accuracy = accuracy
		if accuracy > best_accuracy['value']:
			best_accuracy = {'value': accuracy, 'epoch': epoch + 1}
			torch.save(net.state_dict(), os.path.join('models', model_file))
			print('MODEL SAVED: Best macro-accuracy up to now!\n')
			confusion_matrix = ConfusionMatrix(actual_classes, predicted_classes)
			#best_wrong = wrong

	print()
	#for item in best_wrong:
	#	print(item[0])
	print('Best macro-accuracy obtained:', str(best_accuracy['value']) + '% (' + str(best_accuracy['epoch']), 'epochs). Details (considering MICRO-accuracy):')
	confusion_matrix.print_stats()
	
	plt.subplot(2, 1, 1)
	plt.plot(epochs_list, train_losses, label = 'Train')
	plt.plot(epochs_list, val_losses, label = 'Validation')
	plt.title('RESULTS ' + model_file.replace('.pth', '').upper())
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.plot(epochs_list, accuracies)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (%)')

	os.makedirs('graphics', exist_ok = True)
	plt.savefig(os.path.join('graphics', 'results_' + model_file.replace('.pth', '.png')))
	
	total_time = time_format(time.time() - abs_start)
	print('Total elapsed time:', total_time)
	#plt.show()
	
	print(train_losses)
	print(val_losses)
	
	os.makedirs(os.path.join('results', 'train_val'), exist_ok = True)
	with open(os.path.join('results', 'train_val', model_file.replace('.pth', '.txt')), 'w') as results_txt:
		results_txt.write('Best macro-accuracy obtained: ' + str(best_accuracy['value']) + '% (' + str(best_accuracy['epoch']) + ' epochs). Details (considering MICRO-accuracy):\n\n')
		results_txt.write(str(confusion_matrix.stats()))
		results_txt.write('\n\nTime: ' + total_time)
