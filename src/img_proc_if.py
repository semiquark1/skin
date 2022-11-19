import numpy as np
from typing import List
from itertools import repeat
class ImageProcessIF(object):
    '''Interface for image (pre)processing, run method must be 
    overwritten for use.'''
    def __init__(self):
        '''Overwrite this with initialisation of required models etc.'''
        pass

    def _preproc(self, image:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''Performs the processing of the input image.'''
        raise NotImplementedError

    def normalize(self, image:np.ndarray) -> np.ndarray:
        '''Normalizes input image (result values are in the range of 0 to 1)'''
        return image if image.dtype == np.bool_ else (image / 255.0)

    def run(self,
        list_of_images:List[np.ndarray],
        list_of_masks:List[np.ndarray] = repeat(None),
        do_normalization:bool = False
        ) -> List[np.ndarray]:
        '''Performs the processing on the input list_of_images. If the preprocessing
        requires masks it should be passed to list_of_masks, if its not needed,
        don't pass anything. Normalization can be manually turned on/off with the
        do_normalization parameter. Returns list of processed images.'''
        assert list_of_images != [], 'list_of_images can not be an empty list.'
        if do_normalization:
            return [self.normalize(self._preproc(img, mask))
                for img, mask in zip(list_of_images, list_of_masks)]
        else:
            return [self._preproc(img, mask)
                for img, mask in zip(list_of_images, list_of_masks)]
