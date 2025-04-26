import copy
import time
import heapq
import torch
import random
import torch.nn as nn
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders
import pdb
import gc
from .ablate import AblateGPT
from .wandaplus_wrapper import WeightWandaPlusPruner


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    #
    # inps.requires_grad = False
    # Wanda++ added: store inps and outs as list, to avoid the graph issue when performing backward for multiple times
    inps = []
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp.detach().to('cpu'))
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = []
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids.to('cpu')

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def prune_wandaplus(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers.to('cpu')
    # inps = torch.load('inps_new.pt')
    # args_new = torch.load('args_new.pt')
    # prepare wrapper
    wrapper_config = copy.deepcopy(model.config)
    wrapper_config.args = args
    wrapper_config.prune_n = prune_n
    wrapper_config.prune_m = prune_m

    for i in range(len(layers)):
        print(f'Beginning pruning layer {i}')
        old_layer = layers[i]
        layers[i] = WeightWandaPlusPruner.wrap(
            module=old_layer,
            wrapper_config=wrapper_config,
            layer_idx=i,
        )
        layers[i].to(device)
        inps = layers[i].pruning(inps, attention_mask, position_ids)
        torch.cuda.empty_cache()

# def prune_wandaplus(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     torch.autograd.set_detect_anomaly(True)
#     print("loading calibdation data")
#     dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
#     print("dataset loading complete")
#     with torch.no_grad():
#         inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
#     # Wanda++ updated: here the inps and outs are lists, to avoid the graph issue when performing backward for multiple times
#     # inps = torch.load('input_0.pt')
#     # pdb.set_trace()
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#     layers = model.model.layers.to('cpu')
#     for i in range(len(layers)):
#         outs = []
#         print(f"pruning layer {i}:")
#         layer = layers[i].to(device)
#         subset = find_layers(layer)
#         dtype_init, dtype_ro = torch.float16, torch.float32
#         for param in layer.parameters():
#             param.data = param.data.to(dtype_ro)
#         position_ids = position_ids.to(device)
#         for idx in range(len(inps)):
#             inps[idx] = inps[idx].to(dtype_ro).to(device)
#         layer_clone = copy.deepcopy(layer)
#         RO_weights = [
#             layer.self_attn.q_proj.weight,
#             layer.self_attn.k_proj.weight,
#             layer.self_attn.v_proj.weight,
#             layer.self_attn.o_proj.weight,
#             layer.mlp.up_proj.weight,
#             layer.mlp.gate_proj.weight,
#             layer.mlp.down_proj.weight,
#         ]
#         optimizer = torch.optim.RMSprop(RO_weights, lr=args.lr_ro)
#         # pdb.set_trace()
#         # TODO: Not implemented yet
#         if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
#             dev = model.hf_device_map[f"model.layers.{i}"]
#             inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
#
#         wrapped_layers = {}
#         for name in subset:
#             wrapped_layers[name] = WrappedGPT(subset[name])
#
#         def add_batch(name):
#             def tmp(_, inp, out):
#                 wrapped_layers[name].add_batch(inp[0].data, out.data)
#             return tmp
#         handles = []
#         for name in wrapped_layers:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))
#
#         for j in range(args.nsamples):
#             # Wanda++ Added 1: Compute the regional gradients for RGS metrics
#             inps[j] = inps[j].detach().requires_grad_()
#             optimizer.zero_grad()
#             out = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
#             loss = torch.norm(out)
#             loss.backward(retain_graph=True)
#
#             with torch.no_grad():
#                 for name, wrapped in wrapped_layers.items():
#                     layer_obj = wrapped.layer
#                     if hasattr(layer_obj, "weight") and layer_obj.weight.grad is not None:
#                         grad_sq = layer_obj.weight.grad.to(torch.float32).pow(2).detach()
#
#                         if not hasattr(wrapped, "gradient_norm"):
#                             wrapped.gradient_norm = grad_sq.clone()
#                         else:
#                             wrapped.gradient_norm += grad_sq
#             del out
#         with torch.no_grad():
#             for name, wrapped in wrapped_layers.items():
#                 wrapped.gradient_norm = torch.sqrt(wrapped.gradient_norm / args.nsamples)
#
#
#         # use ro
#         if args.use_ro:
#             for iter in range(args.num_ro_iters):
#                 optimizer.zero_grad()
#                 # Wanda++ Added 2: iterative regional optimization process (pruning + optimization)
#                 random_indices = random.sample(range(args.nsamples), args.nums_ro_samples)
#                 ro_inps = [inps[i] for i in random_indices]
#                 targets = []
#
#                 for j in range(args.nums_ro_samples):
#                     out = layer_clone(ro_inps[j].requires_grad_(), attention_mask=attention_mask, position_ids=position_ids)[0]
#                     targets.append(out.detach().clone())
#
#                 # part 1: pruning
#                 # for name in subset:
#                 #     print(name)
#                 #     inputs = torch.reshape(torch.stack(wrapped_layers[name].inps),
#                 #                            (-1, wrapped_layers[name].inps[0].shape[-1]))
#                 #     print(f'inputs {inputs.mean()}')
#                 #     print(f'subset[name].weight {subset[name].weight.mean()}')
#                 #     print(f'wrapped_layers[name].gradient_norm {wrapped_layers[name].gradient_norm.mean()}')
#
#                 for name in subset:
#                     for name_1 in subset:
#                         wrapped_layers[name_1].inps = []
#                     for j in range(args.nums_ro_samples):
#                         layer(ro_inps[j].requires_grad_(), attention_mask=attention_mask, position_ids=position_ids)[0]
#                     inputs = torch.reshape(torch.stack(wrapped_layers[name].inps),
#                                            (-1, wrapped_layers[name].inps[0].shape[-1]))
#                     input_norm = inputs.norm(p=2, dim=0)
#                     W_metric = torch.abs(subset[name].weight.data) * input_norm.to(
#                         device) + args.rgs_scaling * torch.abs(
#                         subset[name].weight.data) * wrapped_layers[name].gradient_norm
#
#                     W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
#                     if prune_n != 0:
#                         # structured n:m sparsity
#                         for ii in range(W_metric.shape[1]):
#                             if ii % prune_m == 0:
#                                 tmp = W_metric[:, ii:(ii + prune_m)].float()
#                                 W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
#                     else:
#                         sort_res = torch.sort(W_metric, dim=-1, stable=True)
#
#                         # unstructured pruning
#                         indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
#                         W_mask.scatter_(1, indices, True)
#
#                     subset[name].weight.data[W_mask] = 0  ## set weights to zero
#                     del W_mask
#                 # part 2: regional optimization
#                 for j in range(args.nums_ro_samples):
#                     optimizer.zero_grad()
#                     the_inp = ro_inps[j].detach().requires_grad_()
#                     the_pos = position_ids.detach()
#                     out = layer.forward(the_inp, attention_mask=None, position_ids=the_pos)[0]
#                     mse_loss = loss_fn(out, targets[j])
#                     mse_loss.backward() # retain_graph=True
#
#
#                     optimizer.step()
#                 print(f'RO iteration: {iter} loss {mse_loss}')
#             del targets
#
#         # Wanda++ added: optional pruning again
#         if args.additional_pruning:
#             # # step 1: regional gradient computation
#             outs = []
#             for name in subset:
#                 wrapped_layers[name].inps = []
#                 del wrapped_layers[name].gradient_norm
#             for j in range(args.nsamples):
#                 optimizer.zero_grad()
#                 # Wanda++ Added 1: Compute the regional gradients for RGS metrics
#                 inps[j].detach().to(device).requires_grad_()
#                 out = layer(inps[j].requires_grad_(), attention_mask=attention_mask, position_ids=position_ids)[0]
#                 loss = torch.norm(out)
#                 loss.backward(retain_graph=True)
#
#                 with torch.no_grad():
#                     for name, wrapped in wrapped_layers.items():
#                         layer_obj = wrapped.layer
#                         if hasattr(layer_obj, "weight") and layer_obj.weight.grad is not None:
#                             grad_sq = layer_obj.weight.grad.float().pow(2).detach()
#
#                             if not hasattr(wrapped, "gradient_norm"):
#                                 wrapped.gradient_norm = grad_sq.clone()
#                             else:
#                                 wrapped.gradient_norm += grad_sq
#             with torch.no_grad():
#                 for name, wrapped in wrapped_layers.items():
#                     wrapped.gradient_norm = torch.sqrt(wrapped.gradient_norm / args.nsamples)
#             # step 2: pruning
#
#             for name in subset:
#                 for name_1 in subset:
#                     wrapped_layers[name_1].inps = []
#                 for j in range(args.nsamples):
#                     out = layer(inps[j].requires_grad_(), attention_mask=attention_mask, position_ids=position_ids)[0]
#
#                 inputs = torch.reshape(torch.stack(wrapped_layers[name].inps),
#                                        (-1, wrapped_layers[name].inps[0].shape[-1]))
#                 input_norm = inputs.norm(p=2, dim=0)
#                 W_metric = torch.abs(subset[name].weight.data) * input_norm.to(device) + args.rgs_scaling * torch.abs(
#                     subset[name].weight.data) * wrapped_layers[name].gradient_norm
#
#                 W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
#                 if prune_n != 0:
#                     # structured n:m sparsity
#                     for ii in range(W_metric.shape[1]):
#                         if ii % prune_m == 0:
#                             tmp = W_metric[:, ii:(ii + prune_m)].float()
#                             W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
#                 else:
#                     sort_res = torch.sort(W_metric, dim=-1, stable=True)
#
#                     # unstructured pruning
#                     indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
#                     W_mask.scatter_(1, indices, True)
#
#                 subset[name].weight.data[W_mask] = 0  ## set weights to zero
#
#                 del W_mask
#         # pdb.set_trace()
#         outs = []
#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 out = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
#                 outs.append(out)
#         for h in handles:
#             h.remove()
#         # Wanda++ Added 3: convert the weights back to fp16 and to cpu
#         for param in layer.parameters():
#             param.data = param.data.to(dtype_init).to('cpu')
#         position_ids = position_ids.to('cpu')
#         for idx in range(len(outs)):
#             outs[idx] = outs[idx].to(dtype_init).to('cpu')
#         with torch.no_grad():
#             for name, wrapped in wrapped_layers.items():
#                 layer_obj = wrapped.layer
#                 if hasattr(layer_obj, "weight") and hasattr(layer_obj.weight, 'grad'):
#                     layer_obj.weight.grad = None  # Clear weight grad
#                 if hasattr(layer_obj, "bias") and hasattr(layer_obj.bias, 'grad'):
#                     layer_obj.bias.grad = None  # Clear bias grad (if exists)
#                 if hasattr(wrapped, "gradient_norm"):
#                     wrapped.gradient_norm = wrapped.gradient_norm.cpu()
#
#         del wrapped_layers, handles, optimizer, layer, outs, out, layer_obj, layer_clone
#         # del targets, wrapped_layers, handles, optimizer, ro_targets, ro_inps, layer, outs, out
#
#         model.config.use_cache = use_cache
#         torch.cuda.empty_cache()
#         gc.collect()
#
#     model.model.layers.to(device)



def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()