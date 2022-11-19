from img_proc_if import ImageProcessIF
import numpy as np
import cv2
import skimage.morphology
import imageio


class ImageBorder(ImageProcessIF):
    '''Class for creating border images from image and mask.'''
    def __init__(self, dim=(256,256),out_rad = 5,in_rad=40):
            self.dim=dim
            self.outer_rad = out_rad
            self.inner_rad = in_rad

    def create_border_mask(self, mask):
        '''
        Create border mask by adding an eroded and dilating mask. 
        '''
        mask_eroded = self.fast_binary_erosion(mask, self.inner_rad)
        mask_dilated= self.fast_binary_dilation(mask, self.outer_rad)
        border_mask = mask_dilated * np.logical_not(mask_eroded)
        return border_mask
    
    def preprocess_mask_border(self,mask):
        '''
        If the segmentation mask reaches the image border, then a small correction is needed to create the right border mask. 
        Every white pixel on the images border needs to be changed to black.
        '''
        mask[:,0] = 0
        mask[0,:] = 0
        mask[:,-1] = 0
        mask[-1,:] = 0

        return mask


    def apply_mask(self,img,mask):
        '''
        Apply mask to the image. Image pixels will be remaining the same where mask=True and 0 otherwise.
        '''
        img_out = img * mask[:,:,None]
        return img_out
    
    def fast_binary_erosion(self,img, rad, unit=5):
        ''' 
        Erode mask to make it smaller with a preset unit and a given rad.
        '''
        disk = skimage.morphology.disk(unit)
        for i in range(rad // unit):
            img = skimage.morphology.binary_erosion(img, disk)
        return img

    def fast_binary_dilation(self,img, rad, unit=5):
        ''' 
        Dilate mask to make it bigger with a preset unit and a given rad.
        '''
        disk = skimage.morphology.disk(unit)
        for i in range(rad // unit):
            img = skimage.morphology.binary_dilation(img, disk)
        return img
    def rotate_point(self,M,point):
        '''
        Rotate and transform a point with a transformation matrix M
        '''
        # Prepare the vector to be transformed
        v = [point[0],point[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_point = (calculated[0],calculated[1])
        return new_point

    def rotate_bound(self, image, angle, center):
        '''

        '''
        # grab the dimensions of the image and then determine the
        # centre
        (height, width) = image.shape[:2]
        (center_X, center_Y) = (width // 2, height // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((center_X, center_Y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((height * sin) + (width * cos))
        nH = int((height * cos) + (width * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center_X
        M[1, 2] += (nH / 2) - center_Y

        # rotate center of the bounding box
        new_center = self.rotate_point(M,center)
        # perform the actual rotation and return the image

        return cv2.warpAffine(image, M, (nW, nH)),new_center

    def subimage(self,image, center, theta, width, height, padding=20):
        ''' 
        Rotates OpenCV image around center with angle theta (in deg)
        then crops the image according to width and height.
        '''
        image, new_center = self.rotate_bound(image,theta, center)
                
        new_bbox_topleft_x = int(new_center[0] - width/2)
        new_bbox_topleft_y = int(new_center[1] - height/2)
        
        padded_topleft_x = int(new_bbox_topleft_x-padding/2)
        padded_topleft_y = int(new_bbox_topleft_y-padding/2)

        padded_width = width + padding
        padded_height = height + padding

        # make sure subimage is in the original image
        result_image_topleft_x = padded_topleft_x if padded_topleft_x>=0 else 0
        result_image_topleft_y = padded_topleft_y if padded_topleft_y>=0 else 0

        bottomright_x = result_image_topleft_x + int(padded_width)
        bottomright_y = result_image_topleft_y + int(padded_height)

        result_image_bottomright_x = bottomright_x if bottomright_x<image.shape[1] else image.shape[1]-1
        result_image_bottomright_y = bottomright_y if bottomright_y<image.shape[0] else image.shape[0]-1 


        image = image[result_image_topleft_y:result_image_bottomright_y, result_image_topleft_x:result_image_bottomright_x]

        return image

    def _preproc(self, image:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''Returns border image, which is a 256*256 (by default) image around the border of the lesion. Mask required to be
        a boolean np.array with the same dimensons of image.'''
        assert len(image.shape) == 3 and image.shape[2] == 3, \
                'image should have 3 dimensions: height, width, 3)'
        assert len(mask.shape) == 2, 'Mask should have two dimensions. (Same as first two of image)'
        assert image.shape[0] == mask.shape[0], 'First dimension of mask and img does not match'
        assert image.shape[1] == mask.shape[1], 'Second dimension of mask and img does not match'
        assert mask.dtype == np.bool_, 'Mask must be np.bool_ type.'
        
        
        
        # Find the contour of the image by the mask
        contours, _ =  cv2.findContours(mask.astype(np.uint8), 
                                                cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)

        (x, y), (width, height), angle = cv2.minAreaRect(max(contours, key = cv2.contourArea))
        # Crop lesion border
        cropped_rotated_image = self.subimage(image,(x,y),angle, width, height)
        cropped_rotated_mask = self.subimage(mask.astype(np.uint8),(x,y),angle, width, height)
        # Resize images to the specified size
        resized_image = cv2.resize(cropped_rotated_image,self.dim)
        resized_mask = cv2.resize(cropped_rotated_mask,self.dim)
        #preprocess mask to filter out out of border images
        preprocessed_mask = self.preprocess_mask_border(resized_mask)
        # Create border mask for result image
        border_mask = self.create_border_mask(preprocessed_mask)
        # Apply mask to the original image
        masked_image = self.apply_mask(resized_image,border_mask)
        return masked_image
        
