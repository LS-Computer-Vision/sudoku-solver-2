import cv2
import numpy as np
import inspect, sys, re, operator
from model import Trainer
from solver import Solver

class Detector:
	def __init__(self):
		p = re.compile("stage_(?P<idx>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)")

		self.stages = list(sorted(
		map(
			lambda x: (p.fullmatch(x[0]).groupdict()['idx'], p.fullmatch(x[0]).groupdict()['name'], x[1]),
			filter(
				lambda x: inspect.ismethod(x[1]) and p.fullmatch(x[0]),
				inspect.getmembers(self))),
		key=lambda x: x[0]))

		# For storing the recognized digits
		self.digits = [ [None for i in range(9)] for j in range(9) ]

	# Takes as input 9x9 array of numpy images
	# Combines them into 1 image and returns
	# All 9x9 images need to be of same shape
	def makePreview(images):
		assert isinstance(images, list)
		assert len(images) > 0
		assert isinstance(images[0], list)
		assert len(images[0]) > 0
		assert isinstance(images[0], list)

		rows = len(images)
		cols = len(images[0])

		cellShape = images[0][0].shape

		padding = 10
		shape = (rows * cellShape[0] + (rows + 1) * padding, cols * cellShape[1] + (cols + 1) * padding)
		
		result = np.full(shape, 255, np.uint8)

		for row in range(rows):
			for col in range(cols):
				pos = (row * (padding + cellShape[0]) + padding, col * (padding + cellShape[1]) + padding)

				result[pos[0]:pos[0] + cellShape[0], pos[1]:pos[1] + cellShape[1]] = images[row][col]

		return result


	# Takes as input 9x9 array of digits
	# Prints it out on the console in the form of sudoku
	# None instead of number means that its an empty cell
	def showSudoku(array):
		cnt = 0
		for row in array:
			if cnt % 3 == 0:
				print('+-------+-------+-------+')

			colcnt = 0
			for cell in row:
				if colcnt % 3 == 0:
					print('| ', end='')
				print('. ' if cell is None else str(cell) + ' ', end='')
				colcnt += 1
			print('|')
			cnt += 1
		print('+-------+-------+-------+')

	# Runs the detector on the image at path, and returns the 9x9 solved digits
	# if show=True, then the stage results are shown on screen
	# Corrections is an array of the kind [(1,2,9), (3,3,4) ...] which implies
	# that the digit at (1,2) is corrected to 9
	# and the digit at (3,3) is corrected to 4
	def run(self, path='assets/sudokus/sudoku1.jpg', show = False, corrections = []):
		self.path = path
		self.original = cv2.imread(path)

		self.run_stages(show)
		result = self.solve(corrections)


		if show:
			self.showSolved()
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return result

	# Runs all the stages
	def run_stages(self, show):
		results = [('Original', self.original)]

		for idx, name, fun in self.stages:
			image = fun().copy()
			results.append((name, image))

		if show:
			for name, img in results:
				cv2.imshow(name, img)
		

	# Stages
	# Stage function name format: stage_[stage index]_[stage name]
	# Stages are executed increasing order of stage index
	# The function should return a numpy image, which is shown onto the screen
	# In case you have 81 images of 9x9 sudoku cells, you can use makePreview()
	# to create a single image out of those
	# You can pass data from one stage to another using class member variables
	def stage_1_example1(self):
		image = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)
		image = cv2.GaussianBlur(image, (9,9), 0)

		self.image1 = image

		return image

	def stage_2_example2(self):
		
		image = cv2.resize(self.image1, (28, 28))

		cells = [[image.copy() for i in range(9)] for j in range(9)]

		return Detector.makePreview(cells)


	# Solve function
	# Returns solution
	def solve(self, corrections):
		# Only upto 3 corrections allowed
		assert len(corrections) < 3

		# Apply the corrections

		# Solve the sudoku
		self.answers = [[ self.digits[j][i] for i in range(9) ] for j in range(9)]
		s = Solver(self.answers)
		if s.solve():
			self.answers = s.digits
			return s.digits

		return [[None for i in range(9)] for j in range(9)]

	# Optional
	# Use this function to backproject the solved digits onto the original image
	# Save the image of "solved" sudoku into the 'assets/sudoku/' folder with
	# an appropriate name
	def showSolved(self):
		pass


if __name__ == '__main__':
	d = Detector()
	result = d.run('assets/sudokus/sudoku1.jpg', show=True)
	print('Recognized Sudoku:')
	Detector.showSudoku(d.digits)
	print('\n\nSolved Sudoku:')
	Detector.showSudoku(result)