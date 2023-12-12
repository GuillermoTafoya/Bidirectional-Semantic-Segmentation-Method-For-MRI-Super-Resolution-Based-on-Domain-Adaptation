
def crop_pad_ND(img, target_shape):
    r"""
    Resize an image based on target shape
    
    Parameters
    ----------
    target_shape : the shape to be resized to.
    
    Returns
    -------
    image : The resized image.
    """ 
    import operator, numpy as np
    if (img.shape > np.array(target_shape)).any():
        target_shape2 = np.min([target_shape, img.shape],axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target_shape)
    result[tuple(slices)] = img
    return result