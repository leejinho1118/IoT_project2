"""Normal-only retraining objectives; never applied to attack rows."""
import torch


def retraining_loss(x, z, reconstruction, original_loss, old_mask, teacher_z, center, scale, policy, shrink_lambda=10., negative_z=None):
    if scale.requires_grad or center.requires_grad or teacher_z.requires_grad:
        raise ValueError('Teacher targets must be detached')
    if policy.get('squared_shrink', False):
        base = torch.mean((x-reconstruction)**2) + policy.get('shrink_weight',shrink_lambda)*torch.mean(torch.sum(z*z, dim=1))
    else:
        base = original_loss
    zero = z.sum()*0.
    old_loss = torch.mean(((z[old_mask]-teacher_z[old_mask])/scale)**2) if old_mask.any() else zero
    new_mask = ~old_mask
    center_loss = torch.mean(((z[new_mask]-center)/scale)**2) if new_mask.any() else zero
    negative_loss = zero
    if policy.get('negative_weight',0.):
        if negative_z is None:
            raise ValueError('Synthetic negative embeddings required')
        distance = torch.mean(((negative_z-center)/scale)**2,dim=1)
        negative_loss = torch.relu(policy.get('negative_margin',4.)-distance).mean()
    total = base + policy.get('distill',0.)*old_loss + policy.get('align',0.)*center_loss + policy.get('negative_weight',0.)*negative_loss
    return total, dict(base=base.detach(),distill=old_loss.detach(),align=center_loss.detach())
