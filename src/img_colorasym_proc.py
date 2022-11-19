# common numerical and scientific libraries
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import inertia_tensor
import skimage.transform

# local
from img_proc_if import ImageProcessIF

def rotate(img, angle, center, **kwargs):
    return skimage.transform.rotate(img, angle, center=center, **kwargs)

def translate(img, vec):
    return skimage.transform.warp(img, 
            skimage.transform.EuclideanTransform(
                translation=(-vec[1], -vec[0])))

class ImageColorAsym(ImageProcessIF):
    """Class for creating color asymmetry images from image and mask"""

    def _preproc(self, image:np.ndarray, mask:np.ndarray) -> np.ndarray:
        """Create color asymmetry image.

        Parameters:
        -----------
        image: np.ndarray with shape=(height, width, 3)
        mask:  np.ndarray with shape=(height, width), dtype=bool

        Returns:
        --------
        processed_image: np.ndarray with shape=(height, width, 3)
            color asymmetry image
        """
        # check arguments
        assert mask is not None, \
            'Mask should be passed to _preproc from the run method'
        assert len(image.shape) == 3 and image.shape[2] == 3, \
                'image should have 3 dimensions: height, width, 3)'
        assert mask.shape == image.shape[:2], \
                'mask should have 2 dimensions height, width'
        assert mask.dtype == np.bool_, 'mask should have np.bool_ dtype'
        # paint outside mask by color averaged over mask boundary
        tmp = image[distance_transform_cdt(mask) == 1].mean(axis=0)
        avg_color = np.round(tmp).astype(int)
        image = image * mask[:,:,None]
        image[mask == False] = avg_color
        # center of image frame
        cx, cy = mask.shape[0]/2, mask.shape[1]/2
        # mask's center of mass
        mx, my = center_of_mass(mask)
        # inertia tensor: evecs[:,0] is evector of one eigendirection
        _, evecs = np.linalg.eig(inertia_tensor(mask))
        phi_deg = np.arctan2(evecs[1,0], evecs[0,0]) / np.pi*180
        # centered and oriented mask and image
        tmp = translate(mask, (cx-mx, cy-my)).round()
        mask_cent = rotate(tmp, -phi_deg, center=(cy, cx)).round()
        tmp = translate(image, (cx-mx, cy-my))
        img_cent = rotate(tmp, -phi_deg, center=(cy, cx))
        # region outside orig image repainted avg_color
        img_cent[mask_cent == 0] = avg_color / 255.
        # differences, their mean
        diff0  = np.abs(img_cent - np.flip(img_cent, 0))
        diff1  = np.abs(img_cent - np.flip(img_cent, 1))
        diff01 = np.abs(img_cent - np.flip(img_cent, (0,1)))
        diff_all = (diff0 + diff1 + diff01) / 3.
        # transform back
        tmp = rotate(diff_all, phi_deg, center=(cy, cx))
        diff_back = translate(tmp, (mx-cx, my-cy))
        return (255. * diff_back).astype(np.uint8)

