# standard library
from pathlib import Path
self_dir = Path(__file__).parent

# common numerical and scientific libraries
import numpy as np

# tensorflow, tf.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# other common libraries
from PIL import Image
import yaml

# local
from img_proc_if import ImageProcessIF

# parameters
default_model_dir = self_dir / '../models'

def resize(inp_img, shape):
    img = Image.fromarray(inp_img)
    # PIL image size is opposite order as np shape
    img = img.resize(shape[::-1], Image.LANCZOS)
    return np.array(img)

def keras_iou(y_true, y_pred, smooth=100):
    intersec = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    sum_ = K.sum(K.square(y_true), axis =[1,2]) + K.sum(K.square(y_pred),
            axis=[1,2])
    return (intersec + smooth) / (sum_ - intersec + smooth)

def keras_jaccard_distance(y_true, y_pred, smooth=100):
    return 1 - keras_iou(y_true, y_pred, smooth=smooth)

def keras_dice(y_true, y_pred, smooth=100):
    intersec = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    sum_ = K.sum(K.square(y_true), axis =[1,2]) + K.sum(K.square(y_pred),
            axis=[1,2])
    return (2. * intersec + smooth) / (sum_ + smooth)

class Segmenter(ImageProcessIF):
    def __init__(self, model,imshape=None,threshold=None):
        """Initialize segmenter model"""
        if type(model) is str and imshape==None and threshold==None:
            self.model,self.imshape,self.threshold = Segmenter.load_segmentation_model(model)
        else:
            self.model = model
            self.imshape = imshape
            self.threshold = threshold
        
    @staticmethod
    def load_segmentation_model(model:str='model_segmenter',
		    model_dir=default_model_dir):
        model_name = model
        # load parameters
        try:
            with open(Path(model_dir) / f'{model_name}.yaml') as f:
                info = yaml.safe_load(f)
                backbone = info['backbone']
                imshape = info['imshape']
                threshold = info['threshold']
        except FileNotFoundError:
            raise ValueError(f'unknown model: {model}')
        # load model
        if backbone in ('unet_eca',):
            return load_model(
                    Path(model_dir) / f'{model_name}.h5',
                    custom_objects = {
                        'keras_jaccard_distance': keras_jaccard_distance,
                        'keras_iou': keras_iou,
                        'keras_dice': keras_dice,
                        },
                    ),imshape,threshold

    def _preproc(self, image:np.ndarray, mask:None) -> np.ndarray:
        """Generate mask from image

        Parameters:
        -----------
        image: np.ndarray with shape=(height, width, 3)

        Returns:
        --------
        mask: np.ndarray with shape=(height, width), dtype=bool
            True: pixel belongs to lesion; False: background
            eg.: image * mask[:,:,None] is inner image (outside blacked out)
        """
        image_resized = resize(image, self.imshape)       
        mask_resized = self.model.predict(image_resized[None, ...])[0]         
        # apply threshold after resize
        mask = resize(mask_resized, image.shape[:2])
        return (mask > self.threshold).astype(bool)
