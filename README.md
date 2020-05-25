# Handwritten Digit Classifier
Java implementation of a 2-layer, feed-forward neural network trained with back-propagation algorithm for recognizing images of handwritten digits.

## Dataset
The dataset is called Semeion (https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit). It contains 1,593 binary images of size 16 x 16 that each contain one handwritten digit. Each example image is classified as one of the three possible digits: 6, 8 or 9.
Each dataset will begin with a header that describes the dataset: First, there may be several lines starting with “//” that provide a description and comments about the dataset. The line starting with “**” lists the digits. The line starting with " ##" lists the number of attributes, i.e., the number of input values in each instance (in our case, the number of pixels). 

Dataset is pre-processed to contains instances of digits for 6, 8 or 9.

The first output node should output a large value when the instance is determined to be in class 1 (here meaning it is digit 6). 

The second output node should output a large value when the instance is in class 2 (i.e., digit 8) 

and, similarly, the third output node corresponds to class 3 (i.e., digit 9). 
Following these header lines, there will be one line for each instance, containing the values of each attribute 
followed by the target/teacher values for each output node. For example, if the last 3 values for an instance are: 0 0 1 then this means the instance is the digit 9.

**Optimizer used is Stochastic Gradient Descent with Cross - Entropy Loss.**

**Nodes of hidden layer use ReLu activation function while nodes of output layer use Softmax activation function.**

## Usage
`javac *.java`

`java DigitClassifier <numHidden> <learnRate> <maxEpoch> <trainFile> <testFile>`

where trainFile, and testFile are the names of training and testing datasets, respectively. 

numHidden specifies the number of nodes in the hidden layer (excluding the bias node at the hidden layer). 

learnRate and maxEpoch are the learning rate and the number of epochs that the network will be trained, respectively. 

For example,

`java DigitClassifier 5 0.01 100 train1.txt test1.txt`
