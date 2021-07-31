# Initialize this with the 9x9 array of digits
# None element in the 9x9 array represents empty cell
# solve() solves the sudoku and saves it in self.digits
# The solved sudoku can be accessed by the digits member of this class
class Solver:
	def __init__(self, digits):
		self.digits = digits

	# Solve function
	# Returns True if sudoku admits a solution
	# False otherwise
	# Solved sudoku can be found in self.digits
	def solve(self):
		return False