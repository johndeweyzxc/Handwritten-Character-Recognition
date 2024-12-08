# About

- This project utilizes the PyTorch library to implement a **Convolutional Neural Network (CNN)** model for **classifying 28x28 grayscale images of handwritten alphabetic characters (a-z) and numeric digits (0-9).**
- The dataset is downloaded from MNIST database which contains 124,800 grayscale images of handwritten alphabet characters and 240,000 grayscale images of handwritten digit with the dimension of 28×28 pixels.
- The CNN model takes a 28x28 grayscale image as input and converts it into 6 feature maps, each with a dimension of 7x7 pixels. These 6 feature maps are then flattened, resulting in 294 (7x7x6) pixels.
- The 294 flattened feature maps are then used as input to a Neural Network. This Neural Network contains an input layer with 294 input neurons, a hidden layer with 50 neurons, and an output layer with 26 neurons for alphabet character classification or 36 neurons for combined alphabet and digit classification.

## Sample Dataset

### Character

<img src="Sample character dataset.png">

### Digit

<img src="Sample digit dataset.png">

## Model

The model takes an input of 28×28 grayscale image and outputs a prediction of the 26 alphabetic character classes or 26 alphabetic character plus 10 digit classes.
<br>
<br>
<img src="CNN model.png">
<br>
<br>

- **First Convolutional Layer:**
  - Input: (single-channel grayscale image)
  - Kernel size: 3×3, padding = 1, stride = 1
  - Output size (before pooling): 28×28×6
    <br>
    <br>
    - First convolution size $\large\ =\ \frac{Input\ size\ +\ 2\ \times\ Padding\ -\ Kernel\ size}{Stride}\ +\ 1\ =\ \frac{28\ +\ 2\ \times\ 1\ -\ 3}{1}\ +\ 1\ =\ 28$
      <br>
      <br>
    - **Resulting feature map: 28×28×6**
  - After max pooling (2×2, stride = 2):
    <br>
    <br>
    - First max pooling size $\large\ =\ \frac{Input\ size\ -\ Pool\ size}{Stride}\ +\ 1\ =\ \frac{28\ -\ 2}{2}\ +\ 1\ =\ 14$
      <br>
      <br>
    - **Resulting feature map: 14×14×6**
- **Second Convolutional Layer:**
  - Input: 14×14×6
  - Kernel size: 3×3, padding = 1, stride = 1
  - Output size (before pooling): 14×14×6
    <br>
    <br>
    - Second convolution size $\large\ =\ \frac{14\ +\ 2\ \times\ 1\ -\ 3}{1}\ +\ 1\ =\ 14$
      <br>
      <br>
    - **Resulting feature map: 14×14×6**
  - After max pooling (2×2, stride = 2):
    <br>
    <br>
    - Second max pooling size $\large\ =\ \frac{14\ -\ 2}{2}\ +\ 1\ =\ 7$
      <br>
      <br>
    - **Resulting feature map: 7×7×6**
- **Flattening for Fully Connected Layers:**
  - 7×7×6 = 294
    <br>
    <br>
  - **294 pixels becomes the input to the Neural Network.**
- **Neural Network:**
  - Input Layer:
    - 294 input neurons connected to hidden layer
  - Hidden Layer:
    - 50 input neurons connected to output layer
  - Output Layer:
    - 26 output neurons in the case of alphabet classification
    - 36 output neurons in the case of both alphabet and digit classification

## Evaluation

- To assess the model's performance, the testing dataset is used, and undersampling is performed to ensure balanced class distribution across alphabet and digit images.
- The evaluation metrics used include precision, recall, F1-score, and accuracy.
- A confusion matrix is also used to visualize the predicted labels against the actual labels.

### Accuracy

- The overall accuracy of 85% indicates a good classification performance.
- The model may struggle to differentiate between classes that closely resemble each other, such as the character 'o' being predicted as the digit '0'.

                        precision    recall  f1-score   support

                    a       0.89      0.88      0.89       422
                    b       0.97      0.87      0.92       422
                    c       0.96      0.95      0.95       422
                    d       0.95      0.89      0.92       422
                    e       0.94      0.95      0.94       422
                    f       0.95      0.95      0.95       422
                    g       0.89      0.64      0.75       422
                    h       0.95      0.85      0.90       422
                    i       0.74      0.45      0.56       422
                    j       0.93      0.91      0.92       422
                    k       0.95      0.94      0.94       422
                    l       0.85      0.25      0.39       422
                    m       0.95      0.99      0.97       422
                    n       0.88      0.95      0.91       422
                    o       0.85      0.24      0.37       422
                    p       0.96      0.97      0.96       422
                    q       0.91      0.56      0.69       422
                    r       0.95      0.91      0.93       422
                    s       0.95      0.71      0.81       422
                    t       0.93      0.95      0.94       422
                    u       0.95      0.86      0.90       422
                    v       0.91      0.93      0.92       422
                    w       0.96      0.95      0.95       422
                    x       0.96      0.95      0.95       422
                    y       0.94      0.82      0.87       422
                    z       0.99      0.70      0.82       422
                    0       0.52      0.96      0.67       422
                    1       0.45      0.96      0.61       422
                    2       0.74      0.97      0.84       422
                    3       0.95      0.99      0.97       422
                    4       0.83      0.95      0.89       422
                    5       0.77      0.97      0.86       422
                    6       0.89      0.97      0.93       422
                    7       0.96      0.98      0.97       422
                    8       0.87      0.97      0.91       422
                    9       0.67      0.97      0.79       422

            accuracy                            0.85     15192
           macro avg        0.88      0.85      0.85     15192
        weighted avg        0.88      0.85      0.85     15192

### Confusion Matrix

- The model may struggle to differentiate between classes that closely resemble each other.
- The character **'o' being predicted as the digit '0'.**
- The character **'l' being predicted as the digit '1'.**
- The character **'i' being predicted as the digit '1'.**
  <img src="Confusion Matrix.png">
