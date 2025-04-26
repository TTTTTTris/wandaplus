from modeling_llama import LlamaDecoderLayer
from lib.prune_function import wanda_unstructured_pp, wanda_n_m_pp
import gc
import inspect
import logging
from dataclasses import dataclass
import random
from collections import defaultdict
import torch
from torch import Tensor, nn, LongTensor
from tqdm import tqdm
from typing import Optional, Tuple, Any
from .prune_function import wanda_unstructured_pp, wanda_n_m_pp
from transformers.optimization import AdamW
import copy



class WeightWandaPlusPruner(LlamaDecoderLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.config = config
        self.nm_n = config.prune_n  # no. this is Llamaconfig or wrapper_config...
        self.nm_m = config.prune_m
        self.ro = config.args.use_ro

    @classmethod
    def wrap(cls, module:nn.Module, wrapper_config, layer_idx):
        wrapper_config = wrapper_config
        layer_idx = layer_idx
        wrapped = cls(config=wrapper_config, layer_idx=layer_idx)
        wrapped.load_state_dict(module.state_dict(), strict=True)

        wrapped.cuda()
        wrapped.to(dtype=torch.float16)

        return wrapped

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[LongTensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> tuple:
        """
        Following from the transformers.models.llama.modeling_llama LlamaDecoderLayer
        """



        if next(self.parameters()).device != hidden_states.device:
            hidden_states = hidden_states.to(next(self.parameters()).device)
        position_ids = position_ids.to(hidden_states.device)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states#.to(device=residual.device)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    @torch.no_grad()
    def get_signals(self, inputs, attention_mask, position_ids, hook_fn_name='default', output_device=None):
        mlp_ln_inputs = []
        outputs = []

        def hook_fn(_, args: tuple, _output: Any) -> None:
            inp = args[0]  # Position in RMSN.forward args
            if output_device is not None:
                inp = inp.to(output_device)
            mlp_ln_inputs.append(inp)

        # Register the appropriate forward hook based on hook_fn_name
        hook = {
            'q_proj': self.self_attn.q_proj,
            'k_proj': self.self_attn.k_proj,
            'v_proj': self.self_attn.v_proj,
            'o_proj': self.self_attn.o_proj,
            'up_proj': self.mlp.up_proj,
            'gate_proj': self.mlp.gate_proj,
            'down_proj': self.mlp.down_proj,
        }.get(hook_fn_name, self.post_attention_layernorm).register_forward_hook(hook_fn)

        # Move parameters to the same device as the model
        device = next(self.parameters()).device

        with torch.no_grad():  # Disable gradient tracking
            for i, the_input in enumerate(inputs):
                # Move inputs and args to the appropriate device
                the_input = the_input.detach().to(device).requires_grad_()
                position_ids.to(device)



                # Perform forward pass
                out = self.forward(the_input, attention_mask, position_ids)
                if isinstance(out, tuple):
                    out = out[0]
                if output_device is not None:
                    out = out.to(output_device)
                outputs.append(out)

                # Reshape mlp_ln_inputs if necessary
                if mlp_ln_inputs[i].ndim == 2:
                    batch_size, seqlen, _ = out.shape
                    mlp_ln_inputs[i] = mlp_ln_inputs[i].reshape(batch_size, seqlen, -1)

                # Explicitly delete variables to free up memory
                del the_input, out

        # Remove the hook after use
        hook.remove()

        # Return collected inputs and outputs
        return mlp_ln_inputs, outputs

    def regional_optimizer(self, inps, attention_mask, position_ids, the_n, the_m):
        # data and param transfer (data format and device)
        dtype_init = torch.float16
        dtype_ro = torch.float32
        device_init = next(self.parameters()).device
        device_ro = device_init
        target_output_device = 'cpu'
        inps_len = len(inps)
        for param in self.parameters():
            param.data = param.data.to(dtype_ro)
        for layer in self.children():
            layer.to(device_ro)

        for i, the_input in enumerate(inps):
            # Move inputs and args to the appropriate device
            inps[i] = inps[i].to(device_ro).to(dtype_ro)
            position_ids.to(device_ro).to(dtype_ro)


        tmp_module = copy.deepcopy(self)

        RO_weights = [
            self.self_attn.q_proj.weight,
            self.self_attn.k_proj.weight,
            self.self_attn.v_proj.weight,
            self.self_attn.o_proj.weight,
            self.mlp.up_proj.weight,
            self.mlp.gate_proj.weight,
            self.mlp.down_proj.weight,
        ]
        optimizer_ro = torch.optim.RMSprop(RO_weights, lr=self.config.args.lr_ro)
        num_iter = self.config.args.num_ro_iters
        torch.nn.utils.clip_grad_norm_(RO_weights, max_norm=1.0)
        scale = self.config.args.rgs_scaling
        sq_grad = defaultdict(lambda: 0)

        attn_layers = [(self.self_attn.q_proj, 'self.self_attn.q_proj'),
                       (self.self_attn.k_proj, 'self.self_attn.k_proj'),
                       (self.self_attn.v_proj, 'self.self_attn.v_proj'),
                       (self.self_attn.o_proj, 'self.self_attn.o_proj')]
        mlp_layers = [(self.mlp.up_proj, 'self.mlp.up_proj'),
                      (self.mlp.gate_proj, 'self.mlp.gate_proj'),
                      (self.mlp.down_proj, 'self.mlp.down_proj')]
        optimizer = (AdamW
                     (self.parameters(), lr=0.01, eps=0.01))
        print(f'Calculating the gradient for layer {self.layer_idx}')

        # r
        for i, the_input in enumerate(tqdm(inps)):
            # Move inputs and args to the appropriate device
            the_input = the_input.detach().requires_grad_()

            self._mode = 'act'
            optimizer.zero_grad()

            out = self.forward(the_input, attention_mask, position_ids)
            if isinstance(out, tuple):
                out = out[0]
            torch.norm(out).backward(retain_graph=True)
            with torch.no_grad():
                for (layer, layer_name) in attn_layers:
                    sq_grad[layer_name] += ((1 * layer.weight.grad.to(torch.float32)) ** 2).detach()
                    # sq_hess[layer_name] += torch.autograd.grad(layer.weight.grad, layer.weight, create_graph=True)[0]
                for (layer, layer_name) in mlp_layers:
                    sq_grad[layer_name] += ((1 * layer.weight.grad.to(torch.float32)) ** 2).detach()
        for key, value in sq_grad.items():
            sq_grad[key] = torch.sqrt(value / inps_len)
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # loss_fn = torch.nn.KLDivLoss(reduction='sum')
        loss_fn = torch.nn.MSELoss(reduction='sum')

        for i in range(num_iter):
            optimizer_ro.zero_grad()
            random_indices = random.sample(range(len(inps)),
                                           self.config.args.nums_ro_samples)  # For example, choosing 3 random indices
            # print(f'chck len {len(inps)} {len(args)}')
            small_inps = [inps[i] for i in random_indices]
            # print(f'chck len {len(inps)} {len(args)}')
            _, target = tmp_module.get_signals(small_inps, attention_mask, position_ids, torch.float16,
                                               output_device=target_output_device)

            for (layer, layer_name) in attn_layers:
                inputs, _ = self.get_signals(small_inps, attention_mask, position_ids, str(layer_name).split('.')[-1], output_device='cpu')
                inputs = torch.reshape(torch.stack(inputs), (-1, inputs[0].shape[-1]))  # torch.Tensor(inputs)
                if self.nm_n != 0:
                    _, pruned_weight = wanda_n_m_pp(variable=layer.weight,
                                                    X=inputs.to(device=layer.weight.device),
                                                    grad=sq_grad[layer_name],
                                                    scale=scale,
                                                    n=the_n, m=the_m)
                else:
                    _, pruned_weight = wanda_unstructured_pp(variable=layer.weight,
                                                             X=inputs.to(device=layer.weight.device),
                                                             grad=sq_grad[layer_name],
                                                             scale=scale,
                                                             ratio=self.config.args.sparsity_ratio, )
                layer.weight.data.copy_(pruned_weight)
                del pruned_weight


            for (layer, layer_name) in mlp_layers:
                inputs, _ = self.get_signals(small_inps, attention_mask, position_ids, str(layer_name).split('.')[-1], output_device='cpu')
                inputs = torch.reshape(torch.stack(inputs), (-1, inputs[0].shape[-1]))  # torch.Tensor(inputs)
                if self.nm_n != 0:
                    _, pruned_weight = wanda_n_m_pp(variable=layer.weight,
                                                    X=inputs.to(device=layer.weight.device),
                                                    grad=sq_grad[layer_name],
                                                    scale=scale,
                                                    n=the_n, m=the_m)
                else:
                    _, pruned_weight = wanda_unstructured_pp(variable=layer.weight,
                                                             X=inputs.to(device=layer.weight.device),
                                                             grad=sq_grad[layer_name],
                                                             scale=scale,
                                                             ratio=self.config.args.sparsity_ratio, )
                layer.weight.data.copy_(pruned_weight)
            for idx, the_input in enumerate(small_inps):
                the_input = the_input.detach().requires_grad_().to(device_ro)
                position_ids.to(device_ro)
                out = self.forward(the_input, attention_mask, position_ids)
                if isinstance(out, tuple):
                    out = out[0]

                mse_loss = loss_fn(out, target[idx].detach().to(dtype_ro).to(device_ro))
                mse_loss.backward()
                optimizer_ro.step()
            print(f'iteration {i} loss {mse_loss.item()}')

        del target, sq_grad, tmp_module
        for param in self.parameters():
            param.data = param.data.to(dtype_init)
            param.to(device_init)
        for i, the_input in enumerate(inps):
            # Move inputs and args to the appropriate device
            inps[i] = inps[i].to(device_init).to(dtype_init)
            position_ids.to(device_init)
        for layer in self.children():
            layer.to(device_init)
        # Cast parameters back to original dtype

    def pruning(self, inps, attention_mask, position_ids):
        dtype_init = torch.float16
        target_output_device = 'cpu'
        dtype_ro = torch.float32

        the_n, the_m = self.nm_n, self.nm_m
        device = next(self.parameters()).device
        sq_grad = defaultdict(lambda: 0)
        scale = self.config.args.rgs_scaling

        if self.ro:
            self.regional_optimizer(inps, attention_mask, position_ids, the_n, the_m)

        attn_layers = [
            (self.self_attn.q_proj, 'self.self_attn.q_proj'),
            (self.self_attn.k_proj, 'self.self_attn.k_proj'),
            (self.self_attn.v_proj, 'self.self_attn.v_proj'),
            (self.self_attn.o_proj, 'self.self_attn.o_proj'),
        ]
        mlp_layers = [
            (self.mlp.up_proj, 'self.mlp.up_proj'),
            (self.mlp.gate_proj, 'self.mlp.gate_proj'),
            (self.mlp.down_proj, 'self.mlp.down_proj'),
        ]
        optimizer = AdamW(self.parameters(), lr=0.01, eps=0.01)
        # optimizer here use for zero_grad
        inps_len = len(inps)
        print(f'Calculating the gradient for layer {self.layer_idx}')
        # r
        for i, the_input in enumerate(tqdm(inps)):
            # Move inputs and args to the appropriate device
            the_input = the_input.detach().to(device).requires_grad_()
            position_ids.to(device).to(dtype_ro)

            optimizer.zero_grad()
            out = self.forward(the_input, attention_mask, position_ids)
            if isinstance(out, tuple):
                out = out[0]
            torch.norm(out).backward(retain_graph=True)
            with torch.no_grad():
                for (layer, layer_name) in attn_layers:
                    sq_grad[layer_name] += ((1 * layer.weight.grad.to(torch.float32)) ** 2).detach()
                for (layer, layer_name) in mlp_layers:
                    sq_grad[layer_name] += ((1 * layer.weight.grad.to(torch.float32)) ** 2).detach()

            del out
        for key, value in sq_grad.items():
            sq_grad[key] = torch.sqrt(value / inps_len)
        del the_input
        for param in self.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        for (layer, layer_name) in attn_layers:
            inputs, _ = self.get_signals(inps, attention_mask, position_ids, str(layer_name).split('.')[-1], output_device=None)
            inputs = torch.reshape(torch.stack(inputs), (-1, inputs[0].shape[-1]))  # torch.Tensor(inputs)
            if self.nm_n != 0:
                _, pruned_weight = wanda_n_m_pp(variable=layer.weight,
                                                X=inputs.to(device=layer.weight.device),
                                                grad=sq_grad[layer_name],
                                                scale=scale,
                                                n=the_n, m=the_m)

            else:
                _, pruned_weight = wanda_unstructured_pp(variable=layer.weight,
                                                         X=inputs.to(device=layer.weight.device),
                                                         grad=sq_grad[layer_name],
                                                         scale=scale,
                                                         ratio=self.config.args.sparsity_ratio, )
            layer.weight.data.copy_(pruned_weight)

        for (layer, layer_name) in mlp_layers:
            inputs, _ = self.get_signals(inps, attention_mask, position_ids, str(layer_name).split('.')[-1], output_device=None)
            inputs = torch.reshape(torch.stack(inputs), (-1, inputs[0].shape[-1]))  # torch.Tensor(inputs)
            if self.nm_n != 0:
                _, pruned_weight = wanda_n_m_pp(variable=layer.weight,
                                                X=inputs.to(device=layer.weight.device),
                                                grad=sq_grad[layer_name],
                                                scale=scale,
                                                n=the_n, m=the_m)

            else:
                _, pruned_weight = wanda_unstructured_pp(variable=layer.weight,
                                                         X=inputs.to(device=layer.weight.device),
                                                         grad=sq_grad[layer_name],
                                                         scale=scale,
                                                         ratio=self.config.args.sparsity_ratio, )
            layer.weight.data.copy_(pruned_weight)

        _, block_output = self.get_signals(inps, attention_mask, position_ids, dtype_init, output_device=target_output_device)
        del inps, sq_grad

        return block_output


