# Import any ML library here (eg torch, keras, tensorflow)
# Start Editing

# End Editing

import argparse
import random
import numpy as np
from dataLoader import Loader
import os
import cv2

# (Optional) If you want to define any custom module (eg a custom pytorch module), this is the place to do so
# Start Editing
# End Editing


# This is the class for training our model
class Trainer:
	def __init__(self):

		# Seed the RNG's
		# This is the point where you seed your ML library, eg torch.manual_seed(12345)
		# Start Editing
		np.random.seed(12345)
		random.seed(12345)

		# End Editing

		# Set hyperparameters. Fiddle around with the hyperparameters as different ones can give you better results
		# (Optional) Figure out a way to do grid search on the hyperparameters to find the optimal set
		# Start Editing
		self.batch_size = 64 # Batch Size
		self.num_epochs = 20 # Number of Epochs to train for
		self.lr = 0.01       # Learning rate
		# End Editing

		# Init the model, loss, optimizer etc
		# This is the place where you define your model (the neural net architecture)
		# Experiment with different models
		# For beginners, I suggest a simple neural network with a hidden layer of size 32 (and an output layer of size 10 of course)
		# Don't forget the activation function after the hidden layer (I suggest sigmoid activation for beginners)
		# Also set an appropriate loss function. For beginners I suggest the Cross Entropy Loss
		# Also set an appropriate optimizer. For beginners go with gradient descent (SGD), but others can play around with Adam, AdaGrad and you can even try a scheduler for the learning rate
		# Start Editing
		self.model = None
		self.loss = None
		self.optimizer = None
		# End Editing

	def load_data(self):
		# Load Data
		self.loader = Loader()

		# Change Data into representation favored by ML library (eg torch.Tensor for pytorch)
		# This is the place you can reshape your data (eg for CNN's you will want each data point as 28x28 tensor and not 784 vector)
		# Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
		# Start Editing
		


		# End Editing
		pass

	def save_model(self):
		# Save the model parameters into the file 'assets/model'
		# eg. For pytorch, torch.save(self.model.state_dict(), 'assets/model')
		# Start Editing
		


		# End Editing
		pass

	def load_model(self):
		# Load the model parameters from the file 'assets/model'
		if os.path.exists('assets/model'):
			# eg. For pytorch, self.model.load_state_dict(torch.load('assets/model'))
			pass
		else:
			raise Exception('Model not trained')

	def train(self):
		if not self.model:
			return

		print("Training...")
		for epoch in range(self.num_epochs):
			train_loss = self.run_epoch()

			# For beginners, you can leave this alone as it is
			# For others, you can try out splitting the train data into train + val data, and use the validation loss to determine whether to save the model or not
			# Start Editing
			self.save_model()
			# End Editing

			print(f'	Epoch #{epoch+1} trained')
			print(f'		Train loss: {train_loss:.3f}')
		print('Training Complete')

	def test(self):
		if not self.model:
			return 0

		print(f'Running test...')
		# Initialize running loss
		running_loss = 0.0

		# Start Editing

		# Set the ML library to freeze the parameter training

		i = 0 # Number of batches
		correct = 0 # Number of correct predictions
		for batch in range(0, self.test_data.shape[0], self.batch_size):
			batch_X = self.test_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
			batch_Y = self.test_labels[batch: batch+self.batch_size] # shape [batch_size,]

			# Find the predictions
			# Find the loss
			# Find the number of correct predictions and update correct

			# Update running_loss

			i += 1
		
		# End Editing

		print(f'	Test loss: {(running_loss/i):.3f}')
		print(f'	Test accuracy: {(correct*100/self.test_data.shape[0]):.2f}%')

		return correct/self.test_data.shape[0]

	def run_epoch(self):
		# Initialize running loss
		running_loss = 0.0

		# Start Editing

		# Set the ML library to enable the parameter training

		# Shuffle the data (make sure to shuffle the train data in the same permutation as the train labels)
		
		i = 0 # Number of batches
		for batch in range(0, self.train_data.shape[0], self.batch_size):
			batch_X = self.train_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
			batch_Y = self.train_labels[batch: batch+self.batch_size] # shape [batch_size,]


			# Zero out the grads for the optimizer
			
			# Find the predictions
			# Find the loss
			# Backpropagation

			# Update the running loss
			i += 1
		
		# End Editing

		return running_loss / i

	def predict(self, image):
		prediction = 0
		if not self.model:
			return prediction

		# Start Editing

		# Change image into representation favored by ML library (eg torch.Tensor for pytorch)
		# This is the place you can reshape your data (eg for CNN's you will want image as 28x28 tensor and not 784 vector)
		# Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
		
		# Predict the digit value using the model

		# End Editing
		return prediction

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Trainer')
	parser.add_argument('-train', action='store_true', help='Train the model')
	parser.add_argument('-test', action='store_true', help='Test the trained model')
	parser.add_argument('-preview', action='store_true', help='Show a preview of the loaded test images and their corresponding labels')
	parser.add_argument('-predict', action='store_true', help='Make a prediction on a randomly selected test image')

	options = parser.parse_args()

	t = Trainer()
	if options.train:
		t.load_data()
		t.train()
		t.test()
	if options.test:
		t.load_data()
		t.load_model()
		t.test()
	if options.preview:
		t.load_data()
		t.loader.preview()
	if options.predict:
		t.load_data()
		try:
			t.load_model()
		except:
			pass
		i = np.random.randint(0,t.loader.test_data.shape[0])

		print(f'Predicted: {t.predict(t.loader.test_data[i])}')
		print(f'Actual: {t.loader.test_labels[i]}')

		image = t.loader.test_data[i].reshape((28,28))
		image = cv2.resize(image, (0,0), fx=16, fy=16)
		cv2.imshow('Digit', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()