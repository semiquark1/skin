from img_proc_if import ImageProcessIF
import numpy as np
import cv2
class ImageCenter(ImageProcessIF):
    '''Class for creating center images from image and mask.'''

    def __init__(self, box_size:int=256):
        self.box_size=box_size

    def get_image_CG(self,img):
        # calculate moments of binary image        
        M = cv2.moments(np.float32(img))        
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX,cY)

    def get_bbox_around_center(self,center):
        topleft_x = center[0] - int(self.box_size/2)
        topleft_y = center[1] - int(self.box_size/2)
        bottomright_x = topleft_x + self.box_size
        bottomright_y = topleft_y + self.box_size

        return [[topleft_x,topleft_y],[bottomright_x,bottomright_y]]
    
    def validate_bbox(self,bbox,imshape):
        # check top-left corner
        if bbox[0][0] < 0:
            bbox[0][0] = 0
            bbox[1][0] = self.box_size

        if bbox[0][1] < 0:
            bbox[0][1] = 0
            bbox[1][1] = self.box_size
        
        # check bottom-right corner
        if bbox[1][0] > imshape[0] and imshape[0] > self.box_size:
            bbox[1][0] = imshape[0] - 1
            bbox[0][0] = bbox[1][0] - self.box_size
        
        if bbox[1][1] > imshape[1] and imshape[1] > self.box_size:
            bbox[1][1] = imshape[1] - 1
            bbox[0][1] = bbox[1][1] - self.box_size

    def crop_img_with_bbox(self,image,bbox):
        # check too small image
        new_image = np.zeros((self.box_size, self.box_size, 3), dtype= image.dtype)
        yy = (self.box_size - image.shape[0]) // 2
        xx = (self.box_size - image.shape[1]) // 2
        if self.box_size >= image.shape[0] and self.box_size >= image.shape[1]:
            new_image[yy:yy + image.shape[0], xx:xx + image.shape[1]] = image

        elif self.box_size >= image.shape[0]:
            new_image[yy:yy + image.shape[0], :] = image[:, bbox[0][0]:bbox[1][0]]

        elif self.box_size >= image.shape[1]:
            new_image[:, xx:xx + image.shape[1]] = image[bbox[0][1]:bbox[1][1], :]

        else:
            new_image = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

        return new_image

    def _preproc(self, image:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''Returns center image, which is a 256*256 (by default) image around the center of the lesion. Mask required to be
        a boolean np.array with the same dimensons of image.'''
        # check arguments
        assert len(image.shape) == 3 and image.shape[2] == 3, \
                'image should have 3 dimensions: height, width, 3)'
        assert len(mask.shape) == 2, 'Mask should have two dimensions. (Same as first two of image)'
        assert image.shape[0] == mask.shape[0], 'First dimension of mask and img does not match'
        assert image.shape[1] == mask.shape[1], 'Second dimension of mask and img does not match'
        assert mask.dtype == np.bool_, 'Mask must be np.bool_ type.'
        
        
        # get CG
        center_of_lesion = self.get_image_CG(mask)
        # get bbox
        bbox = self.get_bbox_around_center(center_of_lesion)
        # validate bbox to make sure it fits inside the image        
        self.validate_bbox(bbox, image.shape[:2])
        # crop image
        center_image = self.crop_img_with_bbox(image,bbox)
        return center_image
