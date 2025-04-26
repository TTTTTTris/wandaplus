import torch

def wanda_unstructured_pp(
    variable: torch.Tensor,
    grad: torch.Tensor,
    X: torch.Tensor,
    scale: int = 1,
    ratio: float = 0.5,
) -> torch.Tensor:
    device = variable.device
    dtype = variable.dtype
    variable_fp32 = variable.to(torch.float32)


    # Fix: Use norm along the correct dim
    if X.shape[1] != variable_fp32.shape[1]:
        X = X.T  # Ensure X is [batch_size, in_features]

    input_norm = X.norm(p=2, dim=0)  # shape [in_features]
    importance = (variable_fp32.abs() * input_norm.unsqueeze(0)).sum(dim=0)
    importance += scale * (variable_fp32.abs() * grad).sum(dim=0)

    k = int(variable.shape[1] * (1 - ratio))
    topk_indices = torch.topk(importance, k=k, largest=True).indices
    mask = torch.zeros(variable.shape[1], device=device)
    mask[topk_indices] = 1
    structured_mask = mask.unsqueeze(0).expand_as(variable)

    pruned_weight = variable * structured_mask
    del structured_mask, importance
    return 0, pruned_weight.to(dtype=dtype, device=device)


def wanda_n_m_pp(
    variable: torch.Tensor,
    grad: torch.Tensor,
    X: torch.Tensor,
    scale: int = 1,
    n: int = 2,
    m: int = 4,
) -> torch.Tensor:
    # Compute input contingent importance metric
    # print(f'weight shape {variable.abs().shape} input shape {X.norm(p=2, dim=0).shape}')
    # print(f'check mean: weights {variable.abs()} gradients {grad} input {X.norm(p=2, dim=0)} ratio {variable.abs() * X.norm(p=2, dim=0) / variable.abs() * grad}')
    device = variable.device
    dtype = variable.dtype
    variable.to(torch.float32)
    metric = variable.abs() * X.norm(p=2, dim=0).to(torch.float32) + scale * variable.abs() * grad
    num_weights = variable.numel()
    num_rows = int(num_weights / m)

    reshaped_metric = metric.reshape(num_rows, m)
    # Find indices of small weights
    index = torch.argsort(reshaped_metric, dim=1)[:, : (m - n)]
    # Device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types
    w_b = torch.ones_like(reshaped_metric, device=variable.device)
    # Create the pruning mask: scatter zeros based on the index
    scattered_value = 0
    w_b.scatter_(dim=1, index=index, value=scattered_value)
    w_b = w_b.reshape(variable.shape)

    pruned_weight = variable * w_b
    return w_b, pruned_weight.to(dtype=dtype, device=device)