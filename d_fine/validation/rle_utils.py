import numpy as np
import torch

def rle_to_masks(rles: list[dict], device="cpu") -> torch.Tensor:
    """Decode a list of RLE-encoded masks into a dense uint8 tensor [N, H, W]."""
    import pycocotools.mask as mask_utils
    if not rles:
        return torch.zeros((0, 1, 1), dtype=torch.uint8, device=device)
    
    masks = [mask_utils.decode(rle) for rle in rles]
    return torch.from_numpy(np.stack(masks)).to(device=device, dtype=torch.uint8)


def masks_to_rle(masks: np.ndarray | torch.Tensor) -> list[dict]:
    """Encode dense masks [N, H, W] into a list of RLE dicts."""
    import pycocotools.mask as mask_utils
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    
    if masks.ndim == 2:
        masks = masks[None]
        
    rles = []
    for i in range(masks.shape[0]):
        # pycocotools expects fortran order for encoding
        m = np.asfortranarray(masks[i].astype(np.uint8))
        rle = mask_utils.encode(m)
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return rles


def encode_sample_masks_to_rle(sample: dict) -> dict:
    """Helper to convert 'masks' in a sample dict to 'masks_rle'."""
    if "masks" in sample and sample["masks"] is not None:
        if hasattr(sample["masks"], "numel") and sample["masks"].numel() > 0:
            sample["masks_rle"] = masks_to_rle(sample["masks"])
            del sample["masks"]
    return sample

