import nibabel as nib
import numpy as np
from model.utils.upsample import upsample
import matplotlib.pyplot as plt
import torch 

# Load the image using nibabel
nii_img = nib.load("./Data/BCH_0044_s1_nuc.nii")
data = nii_img.get_fdata()
fig, axarr = plt.subplots(3, 2, figsize=(15, 15)) 

def adjust_img(img):
        img = np.rot90(img)
        img = np.fliplr(img)
        return img
"""

for i, (dim, label) in enumerate([(0, 'Axial'), (1, 'Sagittal'), (2, 'Coronal')]):
        input_slice = np.take(data, data.shape[i] // 2, axis=i)

        input_slice = adjust_img(input_slice)

        axarr[i, 0].imshow(input_slice, cmap='gray')


        torch_slice

plt.tight_layout()
plt.savefig('test', bbox_inches='tight', pad_inches=0.1)
plt.show()"""

# Process and visualize one slice for demonstration
input_slice = np.take(data, data.shape[0] // 2, axis=0)
input_slice = adjust_img(input_slice)

# Convert numpy array to torch tensor and add batch and channel dimensions
torch_slice = torch.from_numpy(input_slice.copy()).unsqueeze(0).unsqueeze(0).float()

print(torch_slice.shape)
print(torch_slice)

# Upsample the slice
upsampled_slice = upsample(torch_slice)



# Convert the upsampled tensor back to numpy for visualization
upsampled_slice_np = upsampled_slice.squeeze().cpu().numpy()

# Plotting
fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(input_slice, cmap='gray')
axarr[0].set_title('Original Slice')

axarr[1].imshow(upsampled_slice_np, cmap='gray')
axarr[1].set_title('Upsampled Slice')

plt.tight_layout()
plt.savefig('test', bbox_inches='tight', pad_inches=0.1)
plt.show()