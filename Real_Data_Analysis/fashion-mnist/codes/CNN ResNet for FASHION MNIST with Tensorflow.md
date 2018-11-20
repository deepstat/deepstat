
# CNN ResNet for FASHION MNIST with Tensorflow

DATA SOURCE : https://www.kaggle.com/zalando-research/fashionmnist (Kaggle, Fashion MNIST)

* FASHION MNIST with Python (DAY 1) : http://deepstat.tistory.com/35
* FASHION MNIST with Python (DAY 2) : http://deepstat.tistory.com/36
* FASHION MNIST with Python (DAY 3) : http://deepstat.tistory.com/37
* FASHION MNIST with Python (DAY 4) : http://deepstat.tistory.com/38
* FASHION MNIST with Python (DAY 5) : http://deepstat.tistory.com/39
* FASHION MNIST with Python (DAY 6) : http://deepstat.tistory.com/40
* FASHION MNIST with Python (DAY 7) : http://deepstat.tistory.com/41
* FASHION MNIST with Python (DAY 8) : http://deepstat.tistory.com/42
* FASHION MNIST with Python (DAY 9) : http://deepstat.tistory.com/43
* FASHION MNIST with Python (DAY 10) : http://deepstat.tistory.com/44

## Datasets

### Importing numpy, pandas, pyplot


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Loading datasets


```python
data_train = pd.read_csv("../datasets/fashion-mnist_train.csv")
data_test = pd.read_csv("../datasets/fashion-mnist_test.csv")
```


```python
data_train_y = data_train.label
y_test = data_test.label
```


```python
data_train_x = data_train.drop("label",axis=1)/256
x_test = data_test.drop("label",axis=1)/256
```

### Spliting valid and training


```python
np.random.seed(0)
valid_idx = np.random.choice(60000,10000,replace = False)
train_idx = list(set(range(60000))-set(valid_idx))

x_train = data_train_x.iloc[train_idx,:]
y_train = data_train_y.iloc[train_idx]

x_valid = data_train_x.iloc[valid_idx,:]
y_valid = data_train_y.iloc[valid_idx]
```

## CNN

### Making Class Minibatch


```python
class minibatchData:
    def __init__(self, X, Y):
        self.start_num = 0
        self.x = X
        self.y = Y

    def minibatch(self, batch_size):
        self.outidx = range(self.start_num,(self.start_num + batch_size))
        self.start_num = (self.start_num + batch_size)%(self.x.shape[0])
        return self.x.iloc[self.outidx,:], self.y.iloc[self.outidx]
```


```python
train_minibatch_data = minibatchData(x_train, y_train)
valid_minibatch_data = minibatchData(x_valid, y_valid)
test_minibatch_data = minibatchData(x_test, y_test)
```

### Importing TensorFlow


```python
import tensorflow as tf
from sklearn.metrics import confusion_matrix
```

#### Defining weight_variables and bias_variables


```python
def weight_variables(shape):
    initial = tf.random_uniform(shape=shape, minval=-.1, maxval=.1)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.random_uniform(shape=shape, minval=0, maxval=.1)
    return tf.Variable(initial)    
```

#### Defining conv2d and maxpool


```python
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def avgpool(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
```

#### layers


```python
x = tf.placeholder("float", [None,784])
x_image = tf.reshape(x, [-1,28,28,1])
y = tf.placeholder("int64", [None,])
y_dummies = tf.one_hot(y,depth = 10)

drop_prob = tf.placeholder("float")
training = tf.placeholder("bool")
```


```python
l1_w = weight_variables([3,3,1,64])
l1_b = bias_variables([64])
l1_conv = conv2d(x_image, l1_w) + l1_b
l1_relu = tf.nn.relu(l1_conv)
l1_dropout = tf.layers.dropout(l1_relu,rate = drop_prob, training = training)
```


```python
l2_w = weight_variables([1,1,64,16])
l2_b = bias_variables([16])
l2_conv = conv2d(l1_dropout, l2_w) + l2_b
l2_batch_normalization = tf.layers.batch_normalization(l2_conv)
l2_relu = tf.nn.relu(l2_batch_normalization)
l2_dropout = tf.layers.dropout(l2_relu, rate = drop_prob, training = training)

l3_w = weight_variables([3,3,16,16])
l3_b = bias_variables([16])
l3_conv = conv2d(l2_dropout, l3_w) + l3_b
l3_batch_normalization = tf.layers.batch_normalization(l3_conv)
l3_relu = tf.nn.relu(l3_batch_normalization)
l3_dropout = tf.layers.dropout(l3_relu, rate = drop_prob, training = training)

l4_w = weight_variables([1,1,16,64])
l4_b = bias_variables([64])
l4_conv = conv2d(l3_dropout, l4_w) + l4_b
l4_batch_normalization = tf.layers.batch_normalization(l4_conv)
l4_dropout = tf.layers.dropout(l4_batch_normalization, rate = drop_prob, training = training)
```


```python
l4_add = tf.nn.relu(l4_dropout + l1_dropout)
```


```python
l5_w = weight_variables([1,1,64,16])
l5_b = bias_variables([16])
l5_conv = conv2d(l4_add, l5_w) + l5_b
l5_batch_normalization = tf.layers.batch_normalization(l5_conv)
l5_relu = tf.nn.relu(l5_batch_normalization)
l5_dropout = tf.layers.dropout(l5_relu, rate = drop_prob, training = training)

l6_w = weight_variables([3,3,16,16])
l6_b = bias_variables([16])
l6_conv = conv2d(l5_dropout, l6_w) + l6_b
l6_batch_normalization = tf.layers.batch_normalization(l6_conv)
l6_relu = tf.nn.relu(l6_batch_normalization)
l6_dropout = tf.layers.dropout(l6_relu, rate = drop_prob, training = training)

l7_w = weight_variables([1,1,16,64])
l7_b = bias_variables([64])
l7_conv = conv2d(l6_dropout, l7_w) + l7_b
l7_batch_normalization = tf.layers.batch_normalization(l7_conv)
l7_dropout = tf.layers.dropout(l7_batch_normalization, rate = drop_prob, training = training)
```


```python
l7_add = tf.nn.relu(l7_dropout + l4_dropout)
```


```python
l8_w = weight_variables([1,1,64,16])
l8_b = bias_variables([16])
l8_conv = conv2d(l7_add, l8_w) + l8_b
l8_batch_normalization = tf.layers.batch_normalization(l8_conv)
l8_relu = tf.nn.relu(l8_batch_normalization)
l8_dropout = tf.layers.dropout(l8_relu, rate = drop_prob, training = training)

l9_w = weight_variables([3,3,16,16])
l9_b = bias_variables([16])
l9_conv = conv2d(l8_dropout, l9_w) + l9_b
l9_batch_normalization = tf.layers.batch_normalization(l9_conv)
l9_relu = tf.nn.relu(l9_batch_normalization)
l9_dropout = tf.layers.dropout(l9_relu, rate = drop_prob, training = training)

l10_w = weight_variables([1,1,16,64])
l10_b = bias_variables([64])
l10_conv = conv2d(l9_dropout, l10_w) + l10_b
l10_batch_normalization = tf.layers.batch_normalization(l10_conv)
l10_dropout = tf.layers.dropout(l10_batch_normalization, rate = drop_prob, training = training)
```


```python
l10_add = tf.nn.relu(l10_dropout + l7_dropout)
```


```python
l11_w = weight_variables([1,1,64,16])
l11_b = bias_variables([16])
l11_conv = conv2d(l10_add, l11_w) + l11_b
l11_batch_normalization = tf.layers.batch_normalization(l11_conv)
l11_relu = tf.nn.relu(l11_batch_normalization)
l11_dropout = tf.layers.dropout(l11_relu, rate = drop_prob, training = training)

l12_w = weight_variables([3,3,16,16])
l12_b = bias_variables([16])
l12_conv = conv2d(l11_dropout, l12_w) + l12_b
l12_batch_normalization = tf.layers.batch_normalization(l12_conv)
l12_relu = tf.nn.relu(l12_batch_normalization)
l12_dropout = tf.layers.dropout(l12_relu, rate = drop_prob, training = training)

l13_w = weight_variables([1,1,16,64])
l13_b = bias_variables([64])
l13_conv = conv2d(l12_dropout, l13_w) + l13_b
l13_batch_normalization = tf.layers.batch_normalization(l13_conv)
l13_dropout = tf.layers.dropout(l13_batch_normalization, rate = drop_prob, training = training)
```


```python
l13_add = tf.nn.relu(l13_dropout + l10_dropout)
```


```python
l14_w = weight_variables([1,1,64,16])
l14_b = bias_variables([16])
l14_conv = conv2d(l13_add, l14_w) + l14_b
l14_batch_normalization = tf.layers.batch_normalization(l14_conv)
l14_relu = tf.nn.relu(l14_batch_normalization)
l14_dropout = tf.layers.dropout(l14_relu, rate = drop_prob, training = training)

l15_w = weight_variables([3,3,16,16])
l15_b = bias_variables([16])
l15_conv = conv2d(l14_dropout, l15_w) + l15_b
l15_batch_normalization = tf.layers.batch_normalization(l15_conv)
l15_relu = tf.nn.relu(l15_batch_normalization)
l15_dropout = tf.layers.dropout(l15_relu, rate = drop_prob, training = training)

l16_w = weight_variables([1,1,16,64])
l16_b = bias_variables([64])
l16_conv = conv2d(l15_dropout, l16_w) + l16_b
l16_batch_normalization = tf.layers.batch_normalization(l16_conv)
l16_dropout = tf.layers.dropout(l16_batch_normalization, rate = drop_prob, training = training)
```


```python
l16_add = tf.nn.relu(l16_dropout + l13_dropout)
```


```python
l17_w = weight_variables([1,1,64,32])
l17_b = bias_variables([32])
l17_conv = conv2d(l16_add, l17_w) + l17_b
l17_batch_normalization = tf.layers.batch_normalization(l17_conv)
l17_relu = tf.nn.relu(l17_batch_normalization)
l17_avgpool = avgpool(l17_relu)
l17_dropout = tf.layers.dropout(l17_avgpool, rate = drop_prob, training = training)

l18_w = weight_variables([3,3,32,32])
l18_b = bias_variables([32])
l18_conv = conv2d(l17_dropout, l18_w) + l18_b
l18_batch_normalization = tf.layers.batch_normalization(l18_conv)
l18_relu = tf.nn.relu(l18_batch_normalization)
l18_dropout = tf.layers.dropout(l18_relu, rate = drop_prob, training = training)

l19_w = weight_variables([1,1,32,128])
l19_b = bias_variables([128])
l19_conv = conv2d(l18_dropout, l19_w) + l19_b
l19_batch_normalization = tf.layers.batch_normalization(l19_conv)
l19_dropout = tf.layers.dropout(l19_batch_normalization, rate = drop_prob, training = training)
```


```python
l16_w2 = weight_variables([1,1,64,128])
l16_conv2 = conv2d(l16_add, l16_w2)
l16_add2 = avgpool(l16_conv2)
```


```python
l19_add = tf.nn.relu(l19_dropout + l16_add2)
```


```python
l20_w = weight_variables([1,1,128,32])
l20_b = bias_variables([32])
l20_conv = conv2d(l19_add, l20_w) + l20_b
l20_batch_normalization = tf.layers.batch_normalization(l20_conv)
l20_relu = tf.nn.relu(l20_batch_normalization)
l20_dropout = tf.layers.dropout(l20_relu, rate = drop_prob, training = training)

l21_w = weight_variables([3,3,32,32])
l21_b = bias_variables([32])
l21_conv = conv2d(l20_dropout, l21_w) + l21_b
l21_batch_normalization = tf.layers.batch_normalization(l21_conv)
l21_relu = tf.nn.relu(l21_batch_normalization)
l21_dropout = tf.layers.dropout(l21_relu, rate = drop_prob, training = training)

l22_w = weight_variables([1,1,32,128])
l22_b = bias_variables([128])
l22_conv = conv2d(l21_dropout, l22_w) + l22_b
l22_batch_normalization = tf.layers.batch_normalization(l22_conv)
l22_dropout = tf.layers.dropout(l22_batch_normalization, rate = drop_prob, training = training)
```


```python
l22_add = tf.nn.relu(l22_dropout + l19_add)
```


```python
l23_w = weight_variables([1,1,128,32])
l23_b = bias_variables([32])
l23_conv = conv2d(l22_add, l23_w) + l23_b
l23_batch_normalization = tf.layers.batch_normalization(l23_conv)
l23_relu = tf.nn.relu(l23_batch_normalization)
l23_dropout = tf.layers.dropout(l23_relu, rate = drop_prob, training = training)

l24_w = weight_variables([3,3,32,32])
l24_b = bias_variables([32])
l24_conv = conv2d(l23_dropout, l24_w) + l24_b
l24_batch_normalization = tf.layers.batch_normalization(l24_conv)
l24_relu = tf.nn.relu(l24_batch_normalization)
l24_dropout = tf.layers.dropout(l24_relu, rate = drop_prob, training = training)

l25_w = weight_variables([1,1,32,128])
l25_b = bias_variables([128])
l25_conv = conv2d(l24_dropout, l25_w) + l25_b
l25_batch_normalization = tf.layers.batch_normalization(l25_conv)
l25_dropout = tf.layers.dropout(l25_batch_normalization, rate = drop_prob, training = training)
```


```python
l25_add = tf.nn.relu(l25_dropout + l22_add)
```


```python
l26_w = weight_variables([1,1,128,32])
l26_b = bias_variables([32])
l26_conv = conv2d(l25_add, l26_w) + l26_b
l26_batch_normalization = tf.layers.batch_normalization(l26_conv)
l26_relu = tf.nn.relu(l26_batch_normalization)
l26_dropout = tf.layers.dropout(l26_relu, rate = drop_prob, training = training)

l27_w = weight_variables([3,3,32,32])
l27_b = bias_variables([32])
l27_conv = conv2d(l26_dropout, l27_w) + l27_b
l27_batch_normalization = tf.layers.batch_normalization(l27_conv)
l27_relu = tf.nn.relu(l27_batch_normalization)
l27_dropout = tf.layers.dropout(l27_relu, rate = drop_prob, training = training)

l28_w = weight_variables([1,1,32,128])
l28_b = bias_variables([128])
l28_conv = conv2d(l27_dropout, l28_w) + l28_b
l28_batch_normalization = tf.layers.batch_normalization(l28_conv)
l28_dropout = tf.layers.dropout(l28_batch_normalization, rate = drop_prob, training = training)
```


```python
l28_add = tf.nn.relu(l28_dropout + l25_add)
```


```python
l29_w = weight_variables([1,1,128,32])
l29_b = bias_variables([32])
l29_conv = conv2d(l28_add, l29_w) + l29_b
l29_batch_normalization = tf.layers.batch_normalization(l29_conv)
l29_relu = tf.nn.relu(l29_batch_normalization)
l29_dropout = tf.layers.dropout(l29_relu, rate = drop_prob, training = training)

l30_w = weight_variables([3,3,32,32])
l30_b = bias_variables([32])
l30_conv = conv2d(l29_dropout, l30_w) + l30_b
l30_batch_normalization = tf.layers.batch_normalization(l30_conv)
l30_relu = tf.nn.relu(l30_batch_normalization)
l30_dropout = tf.layers.dropout(l30_relu, rate = drop_prob, training = training)

l31_w = weight_variables([1,1,32,128])
l31_b = bias_variables([128])
l31_conv = conv2d(l30_dropout, l31_w) + l31_b
l31_batch_normalization = tf.layers.batch_normalization(l31_conv)
l31_dropout = tf.layers.dropout(l31_batch_normalization, rate = drop_prob, training = training)
```


```python
l31_add = tf.nn.relu(l31_dropout + l28_add)
```


```python
l32_w = weight_variables([1,1,128,32])
l32_b = bias_variables([32])
l32_conv = conv2d(l31_add, l32_w) + l32_b
l32_batch_normalization = tf.layers.batch_normalization(l32_conv)
l32_relu = tf.nn.relu(l32_batch_normalization)
l32_dropout = tf.layers.dropout(l32_relu, rate = drop_prob, training = training)

l33_w = weight_variables([3,3,32,32])
l33_b = bias_variables([32])
l33_conv = conv2d(l32_dropout, l33_w) + l33_b
l33_batch_normalization = tf.layers.batch_normalization(l33_conv)
l33_relu = tf.nn.relu(l33_batch_normalization)
l33_dropout = tf.layers.dropout(l33_relu, rate = drop_prob, training = training)

l34_w = weight_variables([1,1,32,128])
l34_b = bias_variables([128])
l34_conv = conv2d(l33_dropout, l34_w) + l34_b
l34_batch_normalization = tf.layers.batch_normalization(l34_conv)
l34_dropout = tf.layers.dropout(l34_batch_normalization, rate = drop_prob, training = training)
```


```python
l34_add = tf.nn.relu(l34_dropout + l31_add)
```


```python
l35_w = weight_variables([1,1,128,64])
l35_b = bias_variables([64])
l35_conv = conv2d(l34_add, l35_w) + l35_b
l35_batch_normalization = tf.layers.batch_normalization(l35_conv)
l35_relu = tf.nn.relu(l35_batch_normalization)
l35_avgpool = avgpool(l35_relu)
l35_dropout = tf.layers.dropout(l35_avgpool, rate = drop_prob, training = training)

l36_w = weight_variables([3,3,64,64])
l36_b = bias_variables([64])
l36_conv = conv2d(l35_dropout, l36_w) + l36_b
l36_batch_normalization = tf.layers.batch_normalization(l36_conv)
l36_relu = tf.nn.relu(l36_batch_normalization)
l36_dropout = tf.layers.dropout(l36_relu, rate = drop_prob, training = training)

l37_w = weight_variables([1,1,64,256])
l37_b = bias_variables([256])
l37_conv = conv2d(l36_dropout, l37_w) + l37_b
l37_batch_normalization = tf.layers.batch_normalization(l37_conv)
l37_dropout = tf.layers.dropout(l37_batch_normalization, rate = drop_prob, training = training)
```


```python
l34_w2 = weight_variables([1,1,128,256])
l34_conv2 = conv2d(l34_add, l34_w2)
l34_add2 = avgpool(l34_conv2)
```


```python
l37_add = tf.nn.relu(l37_dropout + l34_add2)
```


```python
l38_w = weight_variables([1,1,256,64])
l38_b = bias_variables([64])
l38_conv = conv2d(l37_add, l38_w) + l38_b
l38_batch_normalization = tf.layers.batch_normalization(l38_conv)
l38_relu = tf.nn.relu(l38_batch_normalization)
l38_dropout = tf.layers.dropout(l38_relu, rate = drop_prob, training = training)

l39_w = weight_variables([3,3,64,64])
l39_b = bias_variables([64])
l39_conv = conv2d(l38_dropout, l39_w) + l39_b
l39_batch_normalization = tf.layers.batch_normalization(l39_conv)
l39_relu = tf.nn.relu(l39_batch_normalization)
l39_dropout = tf.layers.dropout(l39_relu, rate = drop_prob, training = training)

l40_w = weight_variables([1,1,64,256])
l40_b = bias_variables([256])
l40_conv = conv2d(l39_dropout, l40_w) + l40_b
l40_batch_normalization = tf.layers.batch_normalization(l40_conv)
l40_dropout = tf.layers.dropout(l40_batch_normalization, rate = drop_prob, training = training)
```


```python
l40_add = tf.nn.relu(l40_dropout + l37_add)
```


```python
l41_w = weight_variables([1,1,256,64])
l41_b = bias_variables([64])
l41_conv = conv2d(l40_add, l41_w) + l41_b
l41_batch_normalization = tf.layers.batch_normalization(l41_conv)
l41_relu = tf.nn.relu(l41_batch_normalization)
l41_dropout = tf.layers.dropout(l41_relu, rate = drop_prob, training = training)

l42_w = weight_variables([3,3,64,64])
l42_b = bias_variables([64])
l42_conv = conv2d(l41_dropout, l42_w) + l42_b
l42_batch_normalization = tf.layers.batch_normalization(l42_conv)
l42_relu = tf.nn.relu(l42_batch_normalization)
l42_dropout = tf.layers.dropout(l42_relu, rate = drop_prob, training = training)

l43_w = weight_variables([1,1,64,256])
l43_b = bias_variables([256])
l43_conv = conv2d(l42_dropout, l43_w) + l43_b
l43_batch_normalization = tf.layers.batch_normalization(l43_conv)
l43_dropout = tf.layers.dropout(l43_batch_normalization, rate = drop_prob, training = training)
```


```python
l43_add = tf.nn.relu(l43_dropout + l40_add)
```


```python
l44_w = weight_variables([1,1,256,64])
l44_b = bias_variables([64])
l44_conv = conv2d(l43_add, l44_w) + l44_b
l44_batch_normalization = tf.layers.batch_normalization(l44_conv)
l44_relu = tf.nn.relu(l44_batch_normalization)
l44_dropout = tf.layers.dropout(l44_relu, rate = drop_prob, training = training)

l45_w = weight_variables([3,3,64,64])
l45_b = bias_variables([64])
l45_conv = conv2d(l44_dropout, l45_w) + l45_b
l45_batch_normalization = tf.layers.batch_normalization(l45_conv)
l45_relu = tf.nn.relu(l45_batch_normalization)
l45_dropout = tf.layers.dropout(l45_relu, rate = drop_prob, training = training)

l46_w = weight_variables([1,1,64,256])
l46_b = bias_variables([256])
l46_conv = conv2d(l45_dropout, l46_w) + l46_b
l46_batch_normalization = tf.layers.batch_normalization(l46_conv)
l46_dropout = tf.layers.dropout(l46_batch_normalization, rate = drop_prob, training = training)
```


```python
l46_add = tf.nn.relu(l46_dropout + l43_add)
```


```python
l47_w = weight_variables([1,1,256,64])
l47_b = bias_variables([64])
l47_conv = conv2d(l46_add, l47_w) + l47_b
l47_batch_normalization = tf.layers.batch_normalization(l47_conv)
l47_relu = tf.nn.relu(l47_batch_normalization)
l47_dropout = tf.layers.dropout(l47_relu, rate = drop_prob, training = training)

l48_w = weight_variables([3,3,64,64])
l48_b = bias_variables([64])
l48_conv = conv2d(l47_dropout, l48_w) + l48_b
l48_batch_normalization = tf.layers.batch_normalization(l48_conv)
l48_relu = tf.nn.relu(l48_batch_normalization)
l48_dropout = tf.layers.dropout(l48_relu, rate = drop_prob, training = training)

l49_w = weight_variables([1,1,64,256])
l49_b = bias_variables([256])
l49_conv = conv2d(l48_dropout, l49_w) + l49_b
l49_batch_normalization = tf.layers.batch_normalization(l49_conv)
l49_dropout = tf.layers.dropout(l49_batch_normalization, rate = drop_prob, training = training)
```


```python
l49_add = tf.nn.relu(l49_dropout + l46_add)
```


```python
l50_avgpool = tf.nn.avg_pool(l49_add, ksize=[1, 8, 8, 1], strides = [1, 8, 8, 1], padding = 'SAME')
l50_flatten = tf.reshape(l50_avgpool, [-1,256])
l50_batch_normalization = tf.layers.batch_normalization(l50_flatten)
l50_w = weight_variables([256,10])
l50_b = bias_variables([10])
l50_inner_product = tf.matmul(l50_batch_normalization, l50_w) + l50_b
l50_log_softmax = tf.nn.log_softmax(l50_inner_product)
```

#### Cross-entropy


```python
xent_loss = -tf.reduce_mean( tf.multiply(y_dummies,l50_log_softmax) )
```

#### Accuracy


```python
pred_labels = tf.argmax(l50_log_softmax,axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(y, pred_labels),"float"))
```

#### Training the Model


```python
lr = tf.placeholder("float")
train_step = tf.train.AdamOptimizer(lr).minimize(xent_loss)
```


```python
saver = tf.train.Saver()
```


```python
best_valid_acc_vec = {}
for k in range(0,5):
    drop_probability = k/16
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epochs = 301
        batch_size = 100
        
        tmp_xent_loss_3 = [1.0,1.0,1.0]
        learning_rate = 1/2**10
        rep_num = int((x_train.shape[0])/batch_size)
        max_valid_acc = .0
        valid_rep_num = int((x_valid.shape[0])/batch_size)
        
        for i in range(epochs):
            tmp_loss_vec = [.0 for a in range(rep_num)]
            tmp_valid_acc_vec = [.0 for a in range(valid_rep_num)]
            tmp_train_acc_vec = [.0 for a in range(rep_num)]
            for j in range(rep_num):
                batch_train_x, batch_train_y = train_minibatch_data.minibatch(batch_size)
                feed_dict = {x : batch_train_x, y : batch_train_y, drop_prob : drop_probability, training : True, lr : learning_rate}
                _, tmp_loss_vec[j] = sess.run([train_step,xent_loss], feed_dict = feed_dict)
            
            tmp_xent_loss_3 = [tmp_xent_loss_3[1], tmp_xent_loss_3[2], sum(tmp_loss_vec)/rep_num]
            
            if tmp_xent_loss_3[0] == min(tmp_xent_loss_3):
                learning_rate = learning_rate * 7/8
            
            for j in range(valid_rep_num):
                batch_valid_x, batch_valid_y = valid_minibatch_data.minibatch(batch_size)
                feed_dict = {x : batch_valid_x, y : batch_valid_y, drop_prob : drop_probability, training : False}
                tmp_valid_acc_vec[j] = sess.run(acc, feed_dict = feed_dict)

            valid_acc = sum(tmp_valid_acc_vec)/valid_rep_num
    
            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_valid_acc_vec[k] = max_valid_acc
                print("DP : " + str(k) + "/16  epoch : " + str(i) + "  max_valid_acc = " + str(valid_acc))
                save_path = saver.save(sess, "./CNNres/model" + str(k) + ".ckpt")
        
            if i % 50 == 0:
                print("DP : " + str(k) + "/16  epoch : " + str(i) + " -- training cross-entropy : " + str(tmp_xent_loss_3[2]))
                for j in range(rep_num):
                    batch_train_x, batch_train_y = train_minibatch_data.minibatch(batch_size)
                    feed_dict = {x : batch_train_x, y : batch_train_y, drop_prob : drop_probability, training : False}
                    tmp_train_acc_vec[j] = sess.run(acc, feed_dict = feed_dict)
                    
                train_acc = sum(tmp_train_acc_vec)/rep_num
                print("DP : " + str(k) + "/16  epoch : " + str(i) + " training_acc = " + str(train_acc) + " valid_acc = " + str(valid_acc))

            if (tmp_xent_loss_3[0] - tmp_xent_loss_3[1])**2 + (tmp_xent_loss_3[1] - tmp_xent_loss_3[2])**2 < 1e-10:
                print("DP : " + str(k) + "/16  converged" + "  epoch : " + str(i))
                break
```

    DP : 0/16  epoch : 0  max_valid_acc = 0.7916999989748001
    DP : 0/16  epoch : 0 -- training cross-entropy : 0.12575874673202633
    DP : 0/16  epoch : 0 training_acc = 0.7891799974441528 valid_acc = 0.7916999989748001
    DP : 0/16  epoch : 1  max_valid_acc = 0.8494000029563904
    DP : 0/16  epoch : 2  max_valid_acc = 0.872299998998642
    DP : 0/16  epoch : 3  max_valid_acc = 0.8844000023603439
    DP : 0/16  epoch : 4  max_valid_acc = 0.885100000500679
    DP : 0/16  epoch : 5  max_valid_acc = 0.891700000166893
    DP : 0/16  epoch : 6  max_valid_acc = 0.8987000018358231
    DP : 0/16  epoch : 7  max_valid_acc = 0.9010999983549118
    DP : 0/16  epoch : 8  max_valid_acc = 0.9029000002145767
    DP : 0/16  epoch : 9  max_valid_acc = 0.9045000016689301
    DP : 0/16  epoch : 12  max_valid_acc = 0.904600003361702
    DP : 0/16  epoch : 14  max_valid_acc = 0.9053000032901763
    DP : 0/16  epoch : 15  max_valid_acc = 0.9059000033140182
    DP : 0/16  epoch : 26  max_valid_acc = 0.906299998164177
    DP : 0/16  epoch : 32  max_valid_acc = 0.9066000008583068
    DP : 0/16  epoch : 40  max_valid_acc = 0.9074000036716461
    DP : 0/16  epoch : 43  max_valid_acc = 0.9091000014543533
    DP : 0/16  epoch : 49  max_valid_acc = 0.910900000333786
    DP : 0/16  epoch : 50  max_valid_acc = 0.9128999996185303
    DP : 0/16  epoch : 50 -- training cross-entropy : 0.004431415451021167
    DP : 0/16  epoch : 50 training_acc = 0.9788800086975098 valid_acc = 0.9128999996185303
    DP : 0/16  epoch : 58  max_valid_acc = 0.914099999666214
    DP : 0/16  epoch : 73  max_valid_acc = 0.9145000022649765
    DP : 0/16  epoch : 84  max_valid_acc = 0.9160000002384185
    DP : 0/16  epoch : 85  max_valid_acc = 0.9187000024318696
    DP : 0/16  epoch : 100 -- training cross-entropy : 0.00011522334130108903
    DP : 0/16  epoch : 100 training_acc = 0.9997200002670288 valid_acc = 0.9162999993562698
    DP : 0/16  epoch : 105  max_valid_acc = 0.9188999998569488
    DP : 0/16  converged  epoch : 109
    DP : 1/16  epoch : 0  max_valid_acc = 0.7295999962091446
    DP : 1/16  epoch : 0 -- training cross-entropy : 0.20441408503055572
    DP : 1/16  epoch : 0 training_acc = 0.726699999332428 valid_acc = 0.7295999962091446
    DP : 1/16  epoch : 1  max_valid_acc = 0.8348000013828277
    DP : 1/16  epoch : 2  max_valid_acc = 0.866800000667572
    DP : 1/16  epoch : 3  max_valid_acc = 0.8779000008106231
    DP : 1/16  epoch : 4  max_valid_acc = 0.8793999999761581
    DP : 1/16  epoch : 5  max_valid_acc = 0.8825000005960465
    DP : 1/16  epoch : 6  max_valid_acc = 0.8902000015974045
    DP : 1/16  epoch : 7  max_valid_acc = 0.8936000019311905
    DP : 1/16  epoch : 8  max_valid_acc = 0.8958000046014786
    DP : 1/16  epoch : 9  max_valid_acc = 0.9046000015735626
    DP : 1/16  epoch : 10  max_valid_acc = 0.9080000025033951
    DP : 1/16  epoch : 11  max_valid_acc = 0.9148000025749207
    DP : 1/16  epoch : 13  max_valid_acc = 0.9166000020503998
    DP : 1/16  epoch : 15  max_valid_acc = 0.9175000047683716
    DP : 1/16  epoch : 16  max_valid_acc = 0.9205000007152557
    DP : 1/16  epoch : 24  max_valid_acc = 0.9221000021696091
    DP : 1/16  epoch : 29  max_valid_acc = 0.9232000035047531
    DP : 1/16  epoch : 35  max_valid_acc = 0.9237000060081482
    DP : 1/16  epoch : 50 -- training cross-entropy : 0.011930404643062503
    DP : 1/16  epoch : 50 training_acc = 0.972260008096695 valid_acc = 0.9214000064134598
    DP : 1/16  epoch : 71  max_valid_acc = 0.9244000017642975
    DP : 1/16  epoch : 81  max_valid_acc = 0.924500002861023
    DP : 1/16  epoch : 85  max_valid_acc = 0.9248000025749207
    DP : 1/16  epoch : 98  max_valid_acc = 0.9257000017166138
    DP : 1/16  epoch : 100 -- training cross-entropy : 0.004851734852476511
    DP : 1/16  epoch : 100 training_acc = 0.9924600064754486 valid_acc = 0.9239000034332275
    DP : 1/16  epoch : 140  max_valid_acc = 0.9262000036239624
    DP : 1/16  epoch : 150 -- training cross-entropy : 0.0014326626591846434
    DP : 1/16  epoch : 150 training_acc = 0.9998400001525879 valid_acc = 0.9248000019788742
    DP : 1/16  epoch : 153  max_valid_acc = 0.9267000031471252
    DP : 1/16  epoch : 155  max_valid_acc = 0.9275000029802323
    DP : 1/16  epoch : 180  max_valid_acc = 0.9278000009059906
    DP : 1/16  epoch : 187  max_valid_acc = 0.9287000012397766
    DP : 1/16  epoch : 200 -- training cross-entropy : 0.0007063996306642366
    DP : 1/16  epoch : 200 training_acc = 1.0 valid_acc = 0.927299998998642
    DP : 1/16  epoch : 210  max_valid_acc = 0.9291000020503998
    DP : 1/16  epoch : 211  max_valid_acc = 0.9296000015735626
    DP : 1/16  epoch : 250 -- training cross-entropy : 0.000595183692192677
    DP : 1/16  epoch : 250 training_acc = 1.0 valid_acc = 0.9275000029802323
    DP : 1/16  converged  epoch : 262
    DP : 2/16  epoch : 0  max_valid_acc = 0.7591000002622604
    DP : 2/16  epoch : 0 -- training cross-entropy : 0.14728072889149188
    DP : 2/16  epoch : 0 training_acc = 0.7627599992752075 valid_acc = 0.7591000002622604
    DP : 2/16  epoch : 1  max_valid_acc = 0.8465000021457673
    DP : 2/16  epoch : 2  max_valid_acc = 0.8625999999046325
    DP : 2/16  epoch : 3  max_valid_acc = 0.865599998831749
    DP : 2/16  epoch : 4  max_valid_acc = 0.8682000011205673
    DP : 2/16  epoch : 5  max_valid_acc = 0.8777999991178512
    DP : 2/16  epoch : 6  max_valid_acc = 0.8837999987602234
    DP : 2/16  epoch : 7  max_valid_acc = 0.890600004196167
    DP : 2/16  epoch : 8  max_valid_acc = 0.9035000032186509
    DP : 2/16  epoch : 9  max_valid_acc = 0.9036000007390976
    DP : 2/16  epoch : 10  max_valid_acc = 0.9070000004768372
    DP : 2/16  epoch : 12  max_valid_acc = 0.9143000024557114
    DP : 2/16  epoch : 16  max_valid_acc = 0.91730000436306
    DP : 2/16  epoch : 19  max_valid_acc = 0.9216000026464463
    DP : 2/16  epoch : 23  max_valid_acc = 0.9249000012874603
    DP : 2/16  epoch : 29  max_valid_acc = 0.9254000037908554
    DP : 2/16  epoch : 38  max_valid_acc = 0.9261000025272369
    DP : 2/16  epoch : 44  max_valid_acc = 0.9272000008821487
    DP : 2/16  epoch : 48  max_valid_acc = 0.9281000006198883
    DP : 2/16  epoch : 50  max_valid_acc = 0.928200004696846
    DP : 2/16  epoch : 50 -- training cross-entropy : 0.015543020361103117
    DP : 2/16  epoch : 50 training_acc = 0.9619000039100647 valid_acc = 0.928200004696846
    DP : 2/16  epoch : 60  max_valid_acc = 0.9286000019311905
    DP : 2/16  epoch : 61  max_valid_acc = 0.9291000044345856
    DP : 2/16  epoch : 62  max_valid_acc = 0.9306000018119812
    DP : 2/16  epoch : 64  max_valid_acc = 0.9331999987363815
    DP : 2/16  epoch : 85  max_valid_acc = 0.9334000015258789
    DP : 2/16  epoch : 100 -- training cross-entropy : 0.008087825570837594
    DP : 2/16  epoch : 100 training_acc = 0.9890200090408325 valid_acc = 0.9285000056028366
    DP : 2/16  epoch : 101  max_valid_acc = 0.9336000019311905
    DP : 2/16  epoch : 150 -- training cross-entropy : 0.004328535903405282
    DP : 2/16  epoch : 150 training_acc = 0.9974000024795532 valid_acc = 0.9282000011205673
    DP : 2/16  epoch : 200 -- training cross-entropy : 0.0028928844772526646
    DP : 2/16  epoch : 200 training_acc = 0.9997800002098084 valid_acc = 0.9317000025510788
    DP : 2/16  epoch : 250 -- training cross-entropy : 0.0023149583506019552
    DP : 2/16  epoch : 250 training_acc = 0.9998800001144409 valid_acc = 0.9311000019311905
    DP : 2/16  converged  epoch : 252
    DP : 3/16  epoch : 0  max_valid_acc = 0.7703999954462052
    DP : 3/16  epoch : 0 -- training cross-entropy : 0.12833979573100807
    DP : 3/16  epoch : 0 training_acc = 0.7724599986076355 valid_acc = 0.7703999954462052
    DP : 3/16  epoch : 1  max_valid_acc = 0.846499999165535
    DP : 3/16  epoch : 2  max_valid_acc = 0.8604000002145767
    DP : 3/16  epoch : 3  max_valid_acc = 0.8639999967813492
    DP : 3/16  epoch : 4  max_valid_acc = 0.8712999999523163
    DP : 3/16  epoch : 6  max_valid_acc = 0.8738000023365021
    DP : 3/16  epoch : 7  max_valid_acc = 0.8809000027179718
    DP : 3/16  epoch : 8  max_valid_acc = 0.8850000017881393
    DP : 3/16  epoch : 9  max_valid_acc = 0.8929999989271163
    DP : 3/16  epoch : 10  max_valid_acc = 0.8976999974250793
    DP : 3/16  epoch : 11  max_valid_acc = 0.8986000007390976
    DP : 3/16  epoch : 12  max_valid_acc = 0.8998999989032745
    DP : 3/16  epoch : 13  max_valid_acc = 0.9036999988555908
    DP : 3/16  epoch : 14  max_valid_acc = 0.9060000038146973
    DP : 3/16  epoch : 15  max_valid_acc = 0.9072000020742417
    DP : 3/16  epoch : 16  max_valid_acc = 0.9107000029087067
    DP : 3/16  epoch : 18  max_valid_acc = 0.9111000007390976
    DP : 3/16  epoch : 19  max_valid_acc = 0.9135000002384186
    DP : 3/16  epoch : 20  max_valid_acc = 0.9139000016450882
    DP : 3/16  epoch : 21  max_valid_acc = 0.916900002360344
    DP : 3/16  epoch : 24  max_valid_acc = 0.91780000269413
    DP : 3/16  epoch : 26  max_valid_acc = 0.9193000012636184
    DP : 3/16  epoch : 27  max_valid_acc = 0.9195000022649765
    DP : 3/16  epoch : 30  max_valid_acc = 0.9196000015735626
    DP : 3/16  epoch : 33  max_valid_acc = 0.9214000034332276
    DP : 3/16  epoch : 34  max_valid_acc = 0.9225000005960464
    DP : 3/16  epoch : 35  max_valid_acc = 0.9237999999523163
    DP : 3/16  epoch : 42  max_valid_acc = 0.9238000017404556
    DP : 3/16  epoch : 45  max_valid_acc = 0.9263999998569489
    DP : 3/16  epoch : 50 -- training cross-entropy : 0.018829643983393908
    DP : 3/16  epoch : 50 training_acc = 0.9460800011157989 valid_acc = 0.9227000010013581
    DP : 3/16  epoch : 56  max_valid_acc = 0.9267000049352646
    DP : 3/16  epoch : 57  max_valid_acc = 0.928700003027916
    DP : 3/16  epoch : 60  max_valid_acc = 0.9291000008583069
    DP : 3/16  epoch : 61  max_valid_acc = 0.9308000022172928
    DP : 3/16  epoch : 87  max_valid_acc = 0.9325000017881393
    DP : 3/16  epoch : 100 -- training cross-entropy : 0.012063768751919269
    DP : 3/16  epoch : 100 training_acc = 0.9736400086879731 valid_acc = 0.9295000034570694
    DP : 3/16  epoch : 150 -- training cross-entropy : 0.008541674865409733
    DP : 3/16  epoch : 150 training_acc = 0.9888200094699859 valid_acc = 0.9303000050783158
    DP : 3/16  epoch : 162  max_valid_acc = 0.9326000028848648
    DP : 3/16  epoch : 175  max_valid_acc = 0.9329000031948089
    DP : 3/16  epoch : 194  max_valid_acc = 0.9338000029325485
    DP : 3/16  epoch : 200 -- training cross-entropy : 0.0067189472580794244
    DP : 3/16  epoch : 200 training_acc = 0.9937800056934357 valid_acc = 0.9311000031232833
    DP : 3/16  epoch : 250 -- training cross-entropy : 0.006442885615630075
    DP : 3/16  epoch : 250 training_acc = 0.9942400051355362 valid_acc = 0.9314000046253205
    DP : 3/16  epoch : 300 -- training cross-entropy : 0.0065578894598875195
    DP : 3/16  epoch : 300 training_acc = 0.9944000052213668 valid_acc = 0.9314000046253205
    DP : 4/16  epoch : 0  max_valid_acc = 0.7249000036716461
    DP : 4/16  epoch : 0 -- training cross-entropy : 0.1709511996358633
    DP : 4/16  epoch : 0 training_acc = 0.7264800015687942 valid_acc = 0.7249000036716461
    DP : 4/16  epoch : 1  max_valid_acc = 0.8070999974012375
    DP : 4/16  epoch : 2  max_valid_acc = 0.850699998140335
    DP : 4/16  epoch : 3  max_valid_acc = 0.8538000005483627
    DP : 4/16  epoch : 4  max_valid_acc = 0.8634000015258789
    DP : 4/16  epoch : 5  max_valid_acc = 0.8676000010967254
    DP : 4/16  epoch : 6  max_valid_acc = 0.8681000024080276
    DP : 4/16  epoch : 7  max_valid_acc = 0.8753000032901764
    DP : 4/16  epoch : 8  max_valid_acc = 0.8840000003576278
    DP : 4/16  epoch : 10  max_valid_acc = 0.8865000009536743
    DP : 4/16  epoch : 11  max_valid_acc = 0.8999999982118606
    DP : 4/16  epoch : 13  max_valid_acc = 0.9036999970674515
    DP : 4/16  epoch : 15  max_valid_acc = 0.9079000014066696
    DP : 4/16  epoch : 17  max_valid_acc = 0.9121000045537948
    DP : 4/16  epoch : 19  max_valid_acc = 0.9158000022172927
    DP : 4/16  epoch : 25  max_valid_acc = 0.9170000034570694
    DP : 4/16  epoch : 26  max_valid_acc = 0.9182000041007996
    DP : 4/16  epoch : 27  max_valid_acc = 0.919600003361702
    DP : 4/16  epoch : 28  max_valid_acc = 0.9207000035047531
    DP : 4/16  epoch : 30  max_valid_acc = 0.9229000014066696
    DP : 4/16  epoch : 33  max_valid_acc = 0.9238000029325485
    DP : 4/16  epoch : 40  max_valid_acc = 0.9243000048398972
    DP : 4/16  epoch : 42  max_valid_acc = 0.925600004196167
    DP : 4/16  epoch : 50 -- training cross-entropy : 0.02068767004646361
    DP : 4/16  epoch : 50 training_acc = 0.9394000022411346 valid_acc = 0.9235000026226043
    DP : 4/16  epoch : 55  max_valid_acc = 0.9288000017404556
    DP : 4/16  epoch : 71  max_valid_acc = 0.9303000003099442
    DP : 4/16  epoch : 73  max_valid_acc = 0.9311000019311905
    DP : 4/16  epoch : 100 -- training cross-entropy : 0.01529986737202853
    DP : 4/16  epoch : 100 training_acc = 0.9576800044775009 valid_acc = 0.9294000029563904
    DP : 4/16  epoch : 109  max_valid_acc = 0.9312000024318695
    DP : 4/16  epoch : 113  max_valid_acc = 0.9330000025033951
    DP : 4/16  epoch : 138  max_valid_acc = 0.9336000013351441
    DP : 4/16  epoch : 150 -- training cross-entropy : 0.011034383416175842
    DP : 4/16  epoch : 150 training_acc = 0.9680000064373017 valid_acc = 0.9284000027179719
    DP : 4/16  epoch : 200 -- training cross-entropy : 0.00965676139574498
    DP : 4/16  epoch : 200 training_acc = 0.9771600106954574 valid_acc = 0.9310000026226044
    DP : 4/16  epoch : 250 -- training cross-entropy : 0.009253540250938386
    DP : 4/16  epoch : 250 training_acc = 0.9784000115394592 valid_acc = 0.931200003027916
    DP : 4/16  epoch : 300 -- training cross-entropy : 0.009516282968921586
    DP : 4/16  epoch : 300 training_acc = 0.9782400114536285 valid_acc = 0.9311000031232833



```python
print(best_valid_acc_vec)
print(max(best_valid_acc_vec))
```

    {0: 0.9188999998569488, 1: 0.9296000015735626, 2: 0.9336000019311905, 3: 0.9338000029325485, 4: 0.9336000013351441}
    4



```python
sess = tf.Session()
saver.restore(sess, "./CNNres/model3.ckpt")
print("Model restored.")
```

    INFO:tensorflow:Restoring parameters from ./CNNres/model3.ckpt
    Model restored.


### Training Accuracy


```python
batch_size = 1000
rep_num = int((x_train.shape[0])/batch_size)
tmp_train_acc_vec = [.0 for a in range(rep_num)]
CNNres_predict_train = []

for j in range(rep_num):
    batch_train_x, batch_train_y = train_minibatch_data.minibatch(batch_size)
    feed_dict = {x : batch_train_x, y : batch_train_y, drop_prob : 1/8, training : False}
    tmp_CNNres_predict_train, tmp_train_acc_vec[j] = sess.run([pred_labels,acc], feed_dict = feed_dict)
    CNNres_predict_train = np.concatenate([CNNres_predict_train, tmp_CNNres_predict_train])

CNNres_train_acc = sum(tmp_train_acc_vec)/rep_num
```


```python
print(confusion_matrix(CNNres_predict_train,y_train))
print("TRAINING ACCURACY =",CNNres_train_acc)
```

    [[4924    0    1    0    0    0   28    0    0    0]
     [   0 5016    0    0    0    0    0    0    0    0]
     [  11    0 4958    1    4    0   54    0    0    0]
     [   6    0    2 4919    0    0    7    0    0    0]
     [   1    0   36   20 5002    0   48    0    0    0]
     [   0    0    0    0    0 4992    0    0    0    0]
     [  67    0    4    1    5    0 4855    0    0    0]
     [   0    0    0    0    0    0    0 5050    0   32]
     [   0    0    0    0    0    0    0    0 4980    0]
     [   0    0    0    0    0    0    0    1    0 4975]]
    TRAINING ACCURACY = 0.9934199929237366


### Validation Accuracy


```python
batch_size = 1000
valid_rep_num = int((x_valid.shape[0])/batch_size)
tmp_valid_acc_vec = [.0 for a in range(rep_num)]
CNNres_predict_valid = []

for j in range(valid_rep_num):
    batch_valid_x, batch_valid_y = valid_minibatch_data.minibatch(batch_size)
    feed_dict = {x : batch_valid_x, y : batch_valid_y, drop_prob : 1/8, training : False}
    tmp_CNNres_predict_valid, tmp_valid_acc_vec[j] = sess.run([pred_labels,acc], feed_dict = feed_dict)
    CNNres_predict_valid = np.concatenate([CNNres_predict_valid, tmp_CNNres_predict_valid])

CNNres_valid_acc = sum(tmp_valid_acc_vec)/valid_rep_num

```


```python
print(confusion_matrix(CNNres_predict_valid,y_valid))
print("VALIDATION ACCURACY =",CNNres_valid_acc)
```

    [[ 873    1    9   12    0    0   91    0    1    0]
     [   2  978    0    3    1    0    1    0    1    0]
     [  18    0  911   11   16    0   63    0    3    0]
     [  15    4    3  990   16    0   18    0    1    0]
     [   1    0   41   29  929    0   67    0    4    0]
     [   0    0    1    0    0  986    0    1    3    2]
     [  78    1   33   14   25    0  765    0    2    0]
     [   0    0    0    0    0   18    0  937    2   25]
     [   4    0    1    0    2    2    3    0 1003    0]
     [   0    0    0    0    0    2    0   11    0  966]]
    VALIDATION ACCURACY = 0.9337999999523163



```python
{"TRAIN_ACC" : CNNres_train_acc , "VALID_ACC" : CNNres_valid_acc}
```




    {'TRAIN_ACC': 0.9934199929237366, 'VALID_ACC': 0.9337999999523163}



### Test Accuracy


```python
batch_size = 1000
test_rep_num = int((x_test.shape[0])/batch_size)
tmp_test_acc_vec = [.0 for a in range(rep_num)]
CNNres_predict_test = []

for j in range(test_rep_num):
    batch_test_x, batch_test_y = test_minibatch_data.minibatch(batch_size)
    feed_dict = {x : batch_test_x, y : batch_test_y, drop_prob : 1/8, training : False}
    tmp_CNNres_predict_test, tmp_test_acc_vec[j] = sess.run([pred_labels,acc], feed_dict = feed_dict)
    CNNres_predict_test = np.concatenate([CNNres_predict_test, tmp_CNNres_predict_test])

CNNres_test_acc = sum(tmp_test_acc_vec)/test_rep_num

```


```python
print(confusion_matrix(CNNres_predict_test,y_test))
print("TEST ACCURACY =",CNNres_test_acc)
```

    [[874   1  13  15   0   0  77   0   1   0]
     [  0 996   0   4   0   0   1   0   0   1]
     [ 29   0 913   5  19   1  48   0   0   0]
     [ 10   1   7 928  18   1  24   0   1   0]
     [  2   2  40  33 923   0  55   0   3   0]
     [  0   0   0   0   0 976   0   1   1   0]
     [ 83   0  27  13  40   0 793   0   3   0]
     [  0   0   0   0   0  15   0 984   2  32]
     [  2   0   0   1   0   1   2   0 988   1]
     [  0   0   0   1   0   6   0  15   1 966]]
    TEST ACCURACY = 0.9340999960899353



```python
{"TRAIN_ACC" : CNNres_train_acc , "VALID_ACC" : CNNres_valid_acc , "TEST_ACC" : CNNres_test_acc}
```




    {'TRAIN_ACC': 0.9934199929237366,
     'VALID_ACC': 0.9337999999523163,
     'TEST_ACC': 0.9340999960899353}




```python

```
