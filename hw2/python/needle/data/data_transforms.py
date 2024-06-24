import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        # Get the height and width after padding
        Origin_H, Origin_W, _ = img.shape
        
        # Pad the image with zero padding. np.pad requires a sequence of (pad_before, pad_after)
        # for each dimension, and 'constant' specifies zero padding
        img_pad = np.pad(img, [(self.padding, self.padding), (self.padding, self.padding), (0, 0)], 'constant', constant_values=0)

        # Calculate the start points for cropping the image to ensure the cropped image has the same dimensions as the input
        start_x, start_y = self.padding + shift_x, self.padding + shift_y

        # Crop the image using calculated start points, ensuring dimensions remain as H x W
        return img_pad[start_x: start_x+Origin_H, start_y: start_y+Origin_W, :]
        ### END YOUR SOLUTION
