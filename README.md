EN3150 Assignment 03 : Simple CNN for classification
====================================================

This project demonstrates the implementation and training of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset.

Description
-----------

The notebook covers the complete machine learning workflow:

*   Loading and preprocessing the MNIST dataset from its original IDX file format.
    
*   Defining a custom CNN architecture using TensorFlow/Keras.
    
*   Training the model and evaluating its performance using metrics like accuracy, precision, recall, and F1-score.
    
*   Comparing the performance of different optimizers: Adam, SGD, and SGD with Momentum.
    
*   Benchmarking the custom CNN against pre-trained models (DenseNet121 and ResNet50).
    

Prerequisites
-------------

*   Python 3.8+
    
*   Jupyter Notebook
    
*   Git

or 

*  Google colab
    

Steps to Run
------------

### Local machine

1.  **Clone the repository**
    
```bash
git clone https://github.com/RashmikaDushan/EN3150_CNN.git
cd EN3150_CNN
```
2.  **Install Dependencies**
- Using Conda (Recommended):
Create the conda environment with:
```bash
conda env create -f environment.yaml
conda activate pattern-rec
```
- Using Pip(Not required if using conda):
Install the required packages using pip:
```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

3.  **Download the MNIST Dataset**
a. Create a folder named `dataset` in the same directory as your `cnn.ipynb` notebook.
b. Download and extract the files of the MNIST dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets) inside the `dataset` folder.
c. Update the dataset paths in the `cnn.ipynb`.
    
```
    init_train_images = load_images('dataset/train-images.idx3-ubyte')
    init_train_labels = load_labels('dataset/train-labels.idx1-ubyte')
    init_test_images = load_images('dataset/t10k-images.idx3-ubyte')
    init_test_labels = load_labels('dataset/t10k-labels.idx1-ubyte')
```

4.  **Run the Notebook**
a. Start Jupyter Notebook.
b. Open the `cnn.ipynb` file.
c. Run the cells sequentially to execute the code.
    

### In google colab

1. Open the `cnn.ipynb` file in google colab.
2. Upload the dataset `idx3-ubyte` files to the content folder.
3. Update the dataset paths in the `cnn.ipynb`.

```
    init_train_images = load_images('/content/train-images.idx3-ubyte')
    init_train_labels = load_labels('/content/train-labels.idx1-ubyte')
    init_test_images = load_images('/content/t10k-images.idx3-ubyte')
    init_test_labels = load_labels('/content/t10k-labels.idx1-ubyte')
```
4. Run the cells sequentially to execute the code.

### Group members

*   Balasooriya K.B.R.D.  
*   De Silva A.P.C.  
*   Dinujaya P.H.T.
*   Hasaranga T.N.