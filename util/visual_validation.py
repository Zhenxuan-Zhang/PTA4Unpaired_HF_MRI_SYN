import torch
import random
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import numpy as np

def compute_metrics(val_set, model, device='cuda:0', batch_count=5):
    """
    Compute FID, Inception Score, and MS-SSIM during validation.
    """
    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    transform_to_uint8 = transforms.Lambda(lambda x: (x * 255).byte())
    model.eval()
    
    total_samples = len(val_set)
    selected_indices = random.sample(range(total_samples), total_samples // 2)
    
    fake_images = []
    fake_images_msssim = []
    
    for idx, data in enumerate(val_set):
        if idx not in selected_indices:
            continue
        
        model.set_input(data)
        model.test()
        
        visuals = model.get_current_visuals()
        real_B = transform_to_uint8(visuals['real_B'].cpu()).to(device)
        real_B= real_B.expand(-1, 3, -1, -1)
        fake_B = transform_to_uint8(visuals['fake_B'].cpu()).to(device)
        fake_B = fake_B.expand(-1, 3, -1, -1)
        fid.update(real_B, real=True)
        fid.update(fake_B, real=False)
        
        fake_images.append(fake_B)
        fake_images_msssim.append(fake_B / 255.0)  # Normalize for MS-SSIM
    
    fake_images = torch.cat(fake_images, dim=0)
    fake_images_msssim = torch.cat(fake_images_msssim, dim=0)
    
    # Compute Inception Score
    is_mean, is_std = inception(fake_images)
    
    # Compute FID
    fid_score = fid.compute()
    
    # Compute MS-SSIM
    ms_ssim_scores = []
    batch_size = len(fake_images_msssim) // batch_count
    batches = [fake_images_msssim[i*batch_size:(i+1)*batch_size] for i in range(batch_count)]
    
    for batch in batches:
        for i in range(len(batch)-1):
            score = ms_ssim_metric(batch[i].unsqueeze(0), batch[i+1].unsqueeze(0))
            ms_ssim_scores.append(score.item())
    
    avg_ms_ssim = np.mean(ms_ssim_scores)
    std_ms_ssim = np.std(ms_ssim_scores)
    
    return fid_score.item(), is_mean.item(), avg_ms_ssim

def validation(val_set, model, opt, device='cuda:0'):
    """
    Validation function to compute FID, Inception Score, and MS-SSIM.
    """
    fid, is_mean, ms_ssim = compute_metrics(val_set, model, device=device)
    
    print(f"FID: {fid:.4f}")
    print(f"Inception Score: {is_mean:.4f}")
    print(f"MS-SSIM: {ms_ssim:.4f}")
    
    return fid, is_mean, ms_ssim

