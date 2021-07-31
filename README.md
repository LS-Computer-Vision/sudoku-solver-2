The updated problem statement for this assignment can be found at https://github.com/LS-Computer-Vision/sudoku-solver-2

# Sudoku Solver 2

We will use a combination of OpenCV and Deep Learning to build a Sudoku generator and solver

## Part 0: Setup

Open up your terminal and execute the following commands:

	pip install virtualenv
	python -m virtualenv venv
	venv/Scripts/activate      # For Windows Users
	source venv/bin/activate   # For OSX/Linux Users
	pip install -r requirements.txt
	# Pip install the required ML library
	pip freeze > requirements.txt

## Part 1: Grid detector
First we will detect the sudoku grid from the image

The image of sudokus are available in ```assets/sudokus/sudoku*.jpg```

```sudoku.py``` contains the relevant classes

It is very helpful to carry out the task in stages, and view the output of every stage as you go along. This will help in debugging the process

The class ```Detector``` contains the functions needed to carry out the processing

All you need to do is add methods corresponding to stages. The methods will be executed automatically during processing.

To add a stage to the processing, just add a member function (taking no parameters) whose name is of the format ```stage_[idx]_[name]```. The stages are executed in the order of increasing ```idx```.

Each stage should take **no parameters** (except ```self```) and **return a numpy image** for debugging purposes (it can be any data you want which will help you visualize the stage). The numpy image is displayed to the screen for debugging.

If you want to share data between stages (or pass data from one stage to another) use class member variables (eg you might store the output of one stage into ```self.image``` and access it in the next stage using ```self.image```)

One example might be, we have the methods ```stage_1_preprocess(self)``` and ```stage_2_transform(self)```. Then the preprocess stage is carried out first followed by transform stage.

For some stages, you might want have 81 different images corresponding to the 81 cells you extracted from the sudoku grid. To display them, use the ```Detector.makePreview()``` method. This takes as input a 9x9 array of similar sized images, and returns a single combined image which you can return from your stage to be displayed

A typical pipeline (set of stages) might go like this:

* Preprocess the image

	![Imgur](https://imgur.com/WH62exT.png)
* Detect largest rectangle and apply perspective transform

	![Imgur](https://imgur.com/zMiX2Vl.png)
* Extract all cells

	![Imgur](https://imgur.com/FwNW7oM.png)
* Remove cell borders

	![Imgur](https://imgur.com/UoOyJVz.png)

Now, there are several different ways to do all of this stuff, and you may want to explore some of this on your own. You may even design an ML based pipeline to extract the grid if you want. Some alternative ideas you can explore

* Use ```Hough Line Detection``` to find all the lines, find their intersection points to find the grid and each cell corner. This method is robust to some warped sudoku images

* The method of finding the largest rectangle to get the grid corners fails when there are other larger rectangles in the image. A more robust method may be to use structural elements to find the most likely location of the grid, and extract the cell centers based on this information

* Removing the cell borders can be done via removing parts touching the boundaries, or removing everything within some margin of the boundaries

* A more robust method to remove cell borders is to find the largest connected blob at the centre of the image, which will obviously be the digit

### Resources

* [The simplest sudoku OCR pipeline](https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/)

* [Another pipeline](https://aakashjhawar.medium.com/sudoku-solver-using-opencv-and-dl-part-1-490f08701179)

* [Hough transform to detect grid](https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python)

* [Alternative approach to cell extraction](https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv)

* [Remove borders by finding the largest blob](https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2)

* [Another approach](https://aishack.in/tutorials/sudoku-grabber-opencv-detection/)

* [Cell extraction using contour detection](https://becominghuman.ai/part-3-solving-the-sudoku-ai-solver-13f64a090922)

## Part 2: Digit Recognizer

You might think that we are done once we have the image of each digit, but sadly we still have a long way to go.

First you need to resize your cell images to ```28x28``` and possibly denoise it

The problems start to arise when you apply the model on each of your cells (with digits in it) to find the digits. You will often find mispredictions.

We need more accuracy. ```90%``` might be fine for a simple ```MNIST``` project, but it will simply not cut it in this case.

For the sudoku to be solved correctly, **all** the digits have to be recognized correctly. If you have a model with ```90%``` accuracy, assuming we need to recognize 30 digits, that means that the chances that all the digits are recognized correctly is ```0.9^30 = 4.2%``` ! This is way too low. Even an accuracy of ```99%``` implies only a ```74%``` chance of correct sudoku detection.

* One way of getting your accuracy up is to use a better model, example a CNN. Now that you are more confident in ML, use your knowledge to explore more complicated networks and try to get your accuracy as high as possible

* ```MNIST``` is a handwritten digit dataset. This is very different from the type of data we are trying to classify here, which is printed digits. A model trained on handwritten digits will perform worse on printed digits compared to the test accuracy which was found on handwritten digits. One way around is to **augment** the dataset or even use our own custom **alternative** dataset by constructing images of printed digits on the fly. We can do this using the ```PIL``` library. Training your model on this kind of a dataset you can achieve even ```99.6%``` accuracy which translates to an ```89%``` chance of detecting the sudoku correctly which is quite good

* Even if you are not able to achieve such high accuracies and an error free sudoku detection, you are allowed upto 3 hardcoded corrections to the recognition. ```Detector.run()``` takes in the array parameter ```correction```, which you can hardcode for each sudoku inside ```test_model.py```. Each element of this array is a tuple of the kind ```(x,y,dig)``` where ```x,y``` is the location of the correction and ```dig``` is the correct digit value. You need to apply these corrections yourself in the ```Detector.solve()``` function, before calling the ```Solver.solve()``` method to solve the sudoku

### Resources

* [PyTorch CNN on MNIST](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118)

* [Alternative Printed digit Dataset using PIL](https://towardsdatascience.com/ruining-sudoku-a-data-science-project-part-3-digits-recognition-and-sudoku-solver-5271e6acd81f)

## Part 3: Solving the Sudoku

This part is pretty simple and does not contain too many roadblocks. While there are many ways to implement this, none of them have any hidden gotchas you need to take care of (unlike the detection tasks which are very fiddly)

```solver.py``` contains the ```Solver``` class. Initialize the objects of the ```Solver``` instance with the `9x9` array of digits you want to recognize, with ```None``` representing empty cells

You need to implement the ```Solver.solve()``` function which will solve the sudoku and store it in ```self.digits```

Some ideas you may explore:

* [Recursive Backtracking](https://stackoverflow.com/questions/1697334/algorithm-for-solving-sudoku/35500280)

* [Backtracking - GFG](https://www.geeksforgeeks.org/sudoku-backtracking-7/)

* [Another implementation](https://dev.to/aspittel/how-i-finally-wrote-a-sudoku-solver-177g)

* [Using a constraint solver like Z3 to solve Sudoku](https://sites.google.com/site/modante/sudokusolver)

* [Another Z3 based implementation](https://stackoverflow.com/questions/23451388/z3-sudoku-solver/23452244)

## Part 4: Reproject Solution (Optional)

The hardest part is over. Everything you have done so far will make you pass the testcases.

But why stop here? Let's be a little fancy and reproject the solution back onto the original image, and that way we have a full augmented reality sudoku solver!

Implement the ```Detector.showSolved()``` method to reproject the solved digits onto the original image, show it to the screen using opencv and save the image in a file inside ```assets/sudoku/``` directory (give the file an appropriate name).

The reprojected image may look something like this:

![Imgur](https://imgur.com/oGLjtRw.jpg)

## Part 4: Analysis (Mandatory)

We have 2 sudoku images in the assets folder, and the corresponding solutions in the respective ```txt``` files.

You can add more sudoku images (and their solutions) to this folder and to the test cases in ```test_model.py```, and achieve successful solving in all of them

* If you needed to hardcode corrections into any test image, then analyse what went wrong in detecting those cells. You might try fiddling with parameters like ```threshold``` of the adaptive thresholding step, or the erosion or dilation kernel of the preprocessing step, or any other parameter you feel like. Sometimes some sudokus work with some parameters and not with others, it is very hard to get it right for the general case. Note down all the experiments that you tried

* If some of your sudoku images straight up do not work, then put them inside the ```assets/sudokus/bad/``` folder. Analyse why these don't work, and suggest improvements on your pipeline to make them work

Put all the conclusions and analysis, experiments, charts and graphs (if any) into ```explanation.pdf```

## Other Resources

* [Yixin Wang's paper (uses structural elements for cell centering)](https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Wang.pdf)

## Submission Instructions

Your assignment repository (https://github.com/LS-Computer-Vision/sudoku-solver-2-{username}) should have the following contents pushed to it

	repository root
	├── assets
	│   ├── sudokus
	│   │   ├── sudoku*.jpg (test images)
	│   │   ├── sudoku*.txt (test solutions)
	│   │   ├── images of reprojected solutions
	│   │   └── bad
	│   │       └── images of sudokus that don't pass
	│   ├── any other resources like Fonts (for PIL)
	│   ├── all the data files
	│   └── model
	├── .gitignore
	├── README.md
	├── requirements.txt
	├── dataLoader.py
	├── model.py
	├── solver.py
	├── sudoku.py
	├── test_model.py
	├── explanations.pdf
	└── (Not to be pushed, ignored by git) venv

## Deadline
The deadline for this assignment is kept at 5 August 11:59 PM
