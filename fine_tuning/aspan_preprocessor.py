import numpy as np
import torch
import torch.nn.functional as F
import cv2

class ASpanFormer_Preprocessor():

    def __init__(self,
                 images_are_normalised: bool = True) -> None:
        
        self.coarse_scale = 0.125 #for training LoFTR
        self.images_are_normalised = images_are_normalised
        self.new_image_size = np.zeros((2,2))

    def rgb2gray(self, rgb) -> np.ndarray:
        gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        return np.expand_dims(gray, axis=-1)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:

        size = image.shape[:2]
        new_size = [320, 320]
        size = np.round(size * np.min(np.array([new_size]) / size, axis=1)).astype(np.int16)

        resized_image = np.zeros((1, *new_size), dtype=np.float32)
        image = cv2.resize(image, (size[1], size[0]))
        resized_image[:, :size[0], :size[1]] = np.expand_dims(image, 0)

        return resized_image, size
    
    def process_image(self,
                      rgb: np.ndarray,
                      segmap: np.ndarray,
                      image_id: int) -> torch.Tensor:
        
        if self.images_are_normalised:
            rgb *= 255

        gray = self.rgb2gray(rgb)
        gray *= segmap[:,:,None]
        gray, new_size = self.resize_image(gray) / 255.
        self.new_image_size[image_id, :] = new_size

        return torch.from_numpy(gray)[None]
    
    def resize_segmap(self, segmap: np.ndarray) -> np.ndarray:
        segmap = segmap.astype(np.int16)
        size = segmap.shape[:2]
        new_size = [320, 320]
        size = np.round(size * np.min(np.array([new_size]) / size, axis=1)).astype(np.int16)

        resized_segmap = np.zeros(new_size, dtype=bool)
        segmap = cv2.resize(segmap, (size[1], size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        resized_segmap[:size[0], :size[1]] = segmap

        return resized_segmap[None]
    
    def proecess_masks(self, segmap:np.ndarray) -> torch.Tensor:
        segmap = self.resize_segmap(segmap)
        segmap = torch.from_numpy(segmap)
        mask = F.interpolate(segmap[None].float(),
                             scale_factor=self.coarse_scale,
                             mode='nearest',
                             recompute_scale_factor=False)[0].bool()
        return mask
    
    # def update_og_data(self, rgb: np.ndarray, K: np.ndarray):
    #     old_size = rgb.shape[:2]
    #     new_size = [320, 320]
    #     size = np.round(size * np.min(np.array([new_size]) / old_size, axis=1)).astype(np.int16)

    #     resized_image = np.zeros((*new_size, 3), dtype=np.float32)
    #     rgb = cv2.resize(rgb, (size[1], size[0]))
    #     resized_image[:size[0], :size[1], :] = rgb

    #     ratio_width = size[1] / old_size[1]
    #     ratio_height = size[0] / old_size[0]
    #     updated_K = K.copy()
    #     updated_K[0] *= ratio_width
    #     updated_K[1] *= ratio_height

    #     return resized_image, updated_K
    
    def __call__(self, data: dict) -> dict:
        processed_image1 = self.process_image(data["image1"].copy(), data["seg1"].copy(), 0)
        processed_image2 = self.process_image(data["image2"].copy(), data["seg2"].copy(), 1)

        processed_mask1 = self.proecess_masks(data["seg1"])
        processed_mask2 = self.proecess_masks(data["seg2"])

        return {
            "image0": processed_image1.cuda().float(),
            "image1": processed_image2.cuda().float(),
            "mask0" : processed_mask1.cuda(),
            "mask1" : processed_mask2.cuda(),
            "new_image_size1": self.new_image_size[0,:],
            "new_image_size1": self.new_image_size[1,:]
        }

        # if self.update_original_data:
        #     update1 = self.update_og_data(data["image1"], data["K1"])
        #     update2 = self.update_og_data(data["image2"], data["K2"])
        #     processed_data.update({
        #         "updated_rgb1": update1[0],
        #         "updated_rgb2": update2[0],
        #         "updated_intrinsics1": update1[1],
        #         "updated_intrinsics2": update2[1],
        #     })

        # return processed_data

        

