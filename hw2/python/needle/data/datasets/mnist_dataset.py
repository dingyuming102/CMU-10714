from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.images, self.labels = self.parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # 函数定义只要求每次调用检索1个sample,
        # 但这里扩展到了可以检索多个sample
        
        X, y = self.images[index], self.labels[index]
        # 函数定义只要求每次调用检索1个sample,
        # 但这里扩展到了可以检索多个sample
        # NOTE: `self.transforms` require the input shape in 3-D like this (28, 28, 1).
        if len(X.shape) == 1:
            X = self.apply_transforms(X.reshape(28, 28, -1))
        else:
            X = np.vstack([self.apply_transforms(x.reshape(28, 28, -1)) for x in X])
        return (X.reshape(-1, 28*28), y)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION
        
    def parse_mnist(self, image_filename, label_filename):
        """ Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.
    
        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format
    
        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                    data.  The dimensionality of the data should be 
                    (num_examples x input_dim) where 'input_dim' is the full 
                    dimension of the data, e.g., since MNIST images are 28x28, it 
                    will be 784.  Values should be of type np.float32, and the data 
                    should be normalized to have a minimum value of 0.0 and a 
                    maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                    and 255 to 1.0).
    
                y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.uint8 and
                    for MNIST will contain the values 0-9.
        """
        ### BEGIN YOUR CODE
        with gzip.open(image_filename, "rb") as img_file:
            magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
            assert(magic_num == 2051)
            tot_pixels = row * col
            X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(tot_pixels)), dtype=np.float32) for _ in range(img_num)])
            X -= np.min(X)
            X /= np.max(X)
    
        with gzip.open(label_filename, "rb") as label_file:
            magic_num, label_num = struct.unpack(">2i", label_file.read(8))
            assert(magic_num == 2049)
            y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)
    
        return X, y
        ### END YOUR CODE