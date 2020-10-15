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

from models import select_net
from time_util import time_format
from MedicalDataset import MedicalDataset

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', dest='test_data_path', type = str, required = True,
		help = 'Path containing data to be tested.')
	parser.add_argument('-m', dest='model_file', type = str, required = True,
		help = 'Name of the trained model file.')
	parser.add_argument('-sl', dest='slices', type = int, default = 10,
		help = 'Number of central slices considered by the trained model.')
	parser.add_argument('-3d', dest='tridim', action = 'store_true',
		help = 'Use if the trained model used tridimensional convolution.')
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
	args = parse_args()
	test_data_path = args.test_data_path
	model_file = args.model_file
	n_slices = args.slices
	tridim = args.tridim
	consider_other_class = not args.no_other
	architecture = args.net
	
	assert(architecture in ['resnet18', 'alexnet', 'vgg', 'squeezenet', 'mobilenet'])

	fix_random_seeds()
	
	test_set = MedicalDataset(test_data_path, min_slices = n_slices, consider_other_class = consider_other_class, test = True)
	test_loader = data.DataLoader(test_set, num_workers = 8, pin_memory = True)
	#test_loader = data.DataLoader(test_set, pin_memory = True)
	
	n_test_files = test_set.__len__()
	classes = ['FLAIR', 'T1', 'T1c', 'T2', 'OTHER'] #train_set.classes

	net = select_net(architecture, n_slices, tridim, consider_other_class)

	if torch.cuda.is_available():
		net = net.cuda()

	start_time = time.time()
		
	#test
	net.load_state_dict(torch.load(os.path.join('models', model_file)))
	net.eval()
	correct = 0
	total = 0
	correct_per_class = [0] * len(classes)
	total_per_class = [0] * len(classes)
	actual_classes = []
	predicted_classes = []
	wrong_predictions = []
	with torch.no_grad():
		for i, (pixel_data, label, path) in enumerate(test_loader):
			label_as_num = label.numpy()[0]
			if tridim:
				pixel_data = pixel_data.view(-1, 1, 10, 200, 200)

			outputs = net(pixel_data.cuda())
			_, predicted = torch.max(outputs.data, 1)

			total += label.size(0)
			correct += (predicted == label.cuda()).sum().item()
			total_per_class[label_as_num] += label.size(0)
			correct_per_class[label_as_num] += (predicted == label.cuda()).sum().item()
			
			actual_classes.append(classes[label_as_num])
			predicted_classes.append(classes[predicted.cpu().numpy()[0]])
			
			if predicted != label.cuda():
				wrong_predictions.append((path[0], classes[label.numpy()[0]], classes[predicted.cpu().numpy()[0]]))
				
			print('Tested', i + 1, 'of', n_test_files, 'files.')
			
	micro_accuracy = 100 * correct / total
	macro_accuracy = 0
	sampled_classes = 0
	for i in range(len(classes)):
		 if total_per_class[i] > 0:
			  macro_accuracy += correct_per_class[i]/total_per_class[i]
			  sampled_classes += 1
	macro_accuracy = 100 * macro_accuracy/sampled_classes
	 
	accuracy = macro_accuracy
	
	confusion_matrix = ConfusionMatrix(actual_classes, predicted_classes)
			
	print()
	print('Macro-accuracy:', str(accuracy) + '%. Details (considering MICRO-accuracy):')
	confusion_matrix.print_stats()
	
	#time
	print()
	end_time = time.time()
	elapsed_time = time_format(end_time - start_time)
	print('Testing elapsed time:', elapsed_time)
	
	os.makedirs(os.path.join('results', 'test'), exist_ok = True)
	with open(os.path.join('results', 'test', test_data_path.replace(os.sep, '_').replace('.', '_') + '--' + model_file.replace('.pth', '.txt')), 'w') as results_txt:
		results_txt.write('Macro-accuracy: ' + str(accuracy) + '%. Details (considering MICRO-accuracy):\n\n')
		results_txt.write(str(confusion_matrix.stats()))
		results_txt.write('\n\nWRONG PREDICTIONS:\n\n')
		for wrong_prediction in wrong_predictions:
			path, label, prediction = wrong_prediction
			results_txt.write(path + ' is ' + label+ ' and was predicted as ' + prediction + '\n')
		results_txt.write('\n\nTime: ' + elapsed_time)
