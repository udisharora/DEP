# Video enhancement utility using BasicVSR from MMEditing
# Requires: mmcv-full, mmedit, torch

import torch
from mmedit.apis import init_model, restoration_inference
import numpy as np

# Download config and checkpoint from MMEditing model zoo if not present
BASICVSR_CONFIG = 'https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py'
BASICVSR_CKPT = 'https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20210409-0e599677.pth'

class BasicVSREnhancer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = init_model(BASICVSR_CONFIG, BASICVSR_CKPT, device=device)

    def enhance(self, frames):
        """
        Args:
            frames: list of np.ndarray (H, W, 3), RGB, uint8
        Returns:
            enhanced_frames: list of np.ndarray (H, W, 3), RGB, uint8
        """
        # MMEditing expects list of file paths or np arrays
        results = restoration_inference(self.model, frames)
        # results['output'] is a list of enhanced frames
        return [np.array(img) for img in results['output']]

# Example usage:
# enhancer = BasicVSREnhancer()
# enhanced_frames = enhancer.enhance(list_of_frames)
