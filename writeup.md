# Data Set Summary & Exploration

## 1. Basic summary of the data set. 
I used numpy to calculate summary statistics of the traffic signs data set:
* The size of the training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of the testing set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43.

## 2. Visualization of the dataset.
I first looked at random samples of each class to get a better understanding of the data. For each class, I took 10 samples and displayed them. This allowed me to have a qualitative familiarity with the dataset.

The ten samples from the class 'Speed limit (50km/h) is shown below. For samples of all the classes, please see the attached html files.

![Sample of Speel limit 50kmph](report/Speed_limit_50kmph.png)

I looked at the distribution of labels in the training, validation and test sets. The distributions of the training, validation and test sets are shown below.

![Training Set Distribution](report/training_set_distribution.png)
![Validation Set Distribution](report/validation_set_distribution.png)
![Test Set Distribution](report/test_set_distribution.png)


# Design and Test a Model Architecture

## 1. Preprocessing Techniques.

I used two techniques for preprocessing.
1. The first technique I used is to normalize the image values to the range [0, 1].

	Normalization of input is important for the performance of convolutions
	and other processing units. 
2. I decided not to convert the image into greyscale as I wanted to explore how the network handles colored images and I was hoping if there are certain patterns which are augmented by color, the network can learn those.
3. The second technique is image augmentation. In image augmentation, for each class, I picked a random sample of 200 images and then generated 10 augmentations for each of the image. This technique was crucial for pushing the accuracy of my model beyond 94%. It helped increase the accuracy to 96.2%. Augmentations of the image include rotation, translation, shear and brightness. Here are examples of images generated from augmenting an image for speed limit 20kmph.

![Augmentation Source](report/20kmpg_augmentation_input.png)
![Augmentation Source](report/20kmph_augmentation_output.png)


## 2. Model Architecture

For the basic architecture, I considered several variations of the Lenet architecture before iterating on the one I describe below. In contrast to the basic Lenet architecture my model had the following differences
	* Used drop out units after the two fully connected layers to reduce over fitting.
	* Used an additional convolutional layer. The idea was to train a bigger network which first overfits the training set and then use dropout and pooling to regularize.
	* To enable me to use an additional layer easily, I used the 'same' padding option in contrast to the 'valid' padding option in Lenet.
	* I used much wider FC layers than Lenet to allow the model to fit the training set well.
	
Layer | Description
------------ | -------------
Input | 32x32x3 RGB image
Convolution 5x5 | 1x1 stride, same padding, outputs 32x32x16
Max Pooling | 2x2 stride, outputs 16x16x16
Convolution 5x5 | 1x1 stride, same padding, outputs 16x16x32
Max Pooling | 2x2 stride, outputs 8x8x32
Convolution 5x5 | 1x1 stride, same padding, outputs 8x8x64
Max Pooling | 2x2 stride, outputs 4x4x64
Flatten | outputs 1024
Fully Connected | outputs 512
Dropout | Training keep_prob = 0.4
Fully Connected | outputs 512
Dropout | Training keep_prob = 0.4
Fully Connected | outputs 43
