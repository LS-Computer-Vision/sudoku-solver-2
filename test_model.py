from sudoku import Detector
from model import Trainer

def testSolver():
	d = Detector()

	with open('assets/sudokus/sudoku1.txt', 'r') as file:
		answer = [list(map(int, line.strip('\n'))) for line in file.readlines()]

	## Add correction array
	result = d.run(path='assets/sudokus/sudoku1.jpg', corrections=[])

	for i in range(9):
		for j in range(9):
			assert result[i][j] == answer[i][j]

	with open('assets/sudokus/sudoku2.txt', 'r') as file:
		answer = [list(map(int, line.strip('\n'))) for line in file.readlines()]

	d = Detector()

	# Add correction array
	result = d.run(path='assets/sudokus/sudoku2.jpg', corrections=[])

	for i in range(9):
		for j in range(9):
			assert result[i][j] == answer[i][j]