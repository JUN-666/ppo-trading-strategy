import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy # For deepcopying model
# For testing with a model instance, assuming ActorCritic is in hft_agent/models/
from ..models.actor_critic import ActorCritic 

def get_symmetric_range_and_scale(data: torch.Tensor, bits: int, per_channel: bool = False, axis: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates r_min, r_max using mean +/- 3*stddev for symmetric quantization.
    Symmetric quantization implies Zero Point (Z) is 0.

    Args:
        data (torch.Tensor): The input tensor.
        bits (int): Number of bits for quantization.
        per_channel (bool): If True, calculate range and scale per channel along the specified axis.
        axis (int): The axis for per-channel quantization (if per_channel is True).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            r_min_eff (effective min range after symmetrizing), 
            r_max_eff (effective max range after symmetrizing), 
            scale (S), 
            zero_point (Z, will be tensor of zeros).
    """
    if per_channel:
        mean = data.mean(dim=axis, keepdim=True)
        std = data.std(dim=axis, keepdim=True)
    else:
        mean = data.mean()
        std = data.std()

    r_max_calc = mean + 3 * std
    r_min_calc = mean - 3 * std
    
    r_abs_max = torch.max(torch.abs(r_min_calc), torch.abs(r_max_calc))

    q_max_val = (2**(bits - 1)) - 1
    if q_max_val == 0: 
        scale = torch.full_like(r_abs_max, float('inf'))
    else:
        scale = r_abs_max / q_max_val
    
    scale = torch.where(scale == 0, torch.tensor(1.0e-8, device=scale.device, dtype=scale.dtype), scale)

    zero_point = torch.zeros_like(scale, dtype=torch.int64) 

    r_max_eff = scale * q_max_val
    r_min_eff = -r_max_eff 

    return r_min_eff, r_max_eff, scale, zero_point


def quantize_tensor(data: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int, signed: bool = True) -> torch.Tensor:
    """
    Quantizes a tensor using the given scale and zero point.
    """
    quantized_data = torch.round(data / scale.clamp(min=1e-8)) + zero_point
    
    if signed:
        q_min = -2**(bits - 1)
        q_max = 2**(bits - 1) - 1
    else: 
        q_min = 0
        q_max = 2**bits - 1
        
    quantized_data = torch.clamp(quantized_data, q_min, q_max)
    
    if bits <= 8:
        return quantized_data.to(torch.int8)
    elif bits <= 16:
        return quantized_data.to(torch.int16)
    else: 
        return quantized_data.to(torch.int32)


def dequantize_tensor(quantized_data: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes an integer tensor back to a floating-point tensor.
    """
    return (quantized_data.float() - zero_point.float()) * scale


class QuantizedLinear(nn.Module):
    """
    A Linear layer that simulates post-training quantization effects.
    Weights and biases are quantized at initialization.
    Activations (inputs and outputs) are quantized and dequantized on-the-fly during forward pass
    to simulate the information loss of quantization.
    """
    def __init__(self, original_linear_layer: nn.Linear, weight_bits: int = 8, bias_bits: int = 16, activation_bits: int = 8):
        super(QuantizedLinear, self).__init__()

        self.original_linear_layer = original_linear_layer 
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits 
        self.activation_bits = activation_bits

        self.original_weight = original_linear_layer.weight.data.clone()
        self.original_bias = original_linear_layer.bias.data.clone() if original_linear_layer.bias is not None else None

        _, _, self.w_scale, self.w_zero_point = get_symmetric_range_and_scale(self.original_weight, self.weight_bits, per_channel=False)
        self.qw = quantize_tensor(self.original_weight, self.w_scale, self.w_zero_point, self.weight_bits)

        if self.original_bias is not None:
            _, _, self.b_scale, self.b_zero_point = get_symmetric_range_and_scale(self.original_bias, self.bias_bits, per_channel=False)
            self.qb = quantize_tensor(self.original_bias, self.b_scale, self.b_zero_point, self.bias_bits)
        else:
            self.qb = None
            self.b_scale = None
            self.b_zero_point = None
            
    def forward(self, x_float: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass simulating quantization effects.
        Input is float, output is float (but has undergone simulated quantization).
        """
        _, _, input_scale, input_zero_point = get_symmetric_range_and_scale(x_float, self.activation_bits)
        q_input = quantize_tensor(x_float, input_scale, input_zero_point, self.activation_bits)
        x_dequant_simulated = dequantize_tensor(q_input, input_scale, input_zero_point)

        w_dequant_simulated = dequantize_tensor(self.qw, self.w_scale, self.w_zero_point)

        b_dequant_simulated = None
        if self.qb is not None and self.b_scale is not None and self.b_zero_point is not None : 
            b_dequant_simulated = dequantize_tensor(self.qb, self.b_scale, self.b_zero_point)
        elif self.original_bias is not None : 
             b_dequant_simulated = self.original_bias

        output_float_simulated_op = F.linear(x_dequant_simulated, w_dequant_simulated, b_dequant_simulated)
        
        _, _, output_scale, output_zero_point = get_symmetric_range_and_scale(output_float_simulated_op, self.activation_bits)
        q_output_simulated = quantize_tensor(output_float_simulated_op, output_scale, output_zero_point, self.activation_bits)
        final_output_dequant_simulated = dequantize_tensor(q_output_simulated, output_scale, output_zero_point)
        
        return final_output_dequant_simulated


def convert_model_to_quantized_simulation(model: nn.Module, layer_configs: dict) -> nn.Module:
    """
    Converts an nn.Module by replacing specified nn.Linear layers with QuantizedLinear
    simulation layers. This version assumes direct attribute access for layers.

    Args:
        model (nn.Module): The original PyTorch model (e.g., an instance of ActorCritic).
        layer_configs (dict): A dictionary mapping layer names (e.g., 'fc1', 'fc2') 
                              to their quantization bit-width settings.
                              Example: {'fc1': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 16}}

    Returns:
        nn.Module: A new model instance with specified layers replaced.
    """
    quant_sim_model = copy.deepcopy(model)
    
    for layer_name, config in layer_configs.items():
        if hasattr(quant_sim_model, layer_name):
            original_sub_module = getattr(quant_sim_model, layer_name)
            if isinstance(original_sub_module, nn.Linear):
                ql = QuantizedLinear(
                    original_sub_module,
                    weight_bits=config.get('weight_bits', 8), # Default to 8 if not specified
                    bias_bits=config.get('bias_bits', 16),   # Default to 16 for bias if not specified
                    activation_bits=config.get('activation_bits', 8) # Default to 8 if not specified
                )
                setattr(quant_sim_model, layer_name, ql)
            else:
                print(f"Warning: Module {layer_name} (type: {type(original_sub_module)}) is not nn.Linear, skipping.")
        else:
            # This simplified version won't handle deeply nested modules unless layer_name is 'base.fc1' etc.
            # For ActorCritic, direct attributes like 'fc1', 'actor_head' are expected.
            print(f"Warning: Layer {layer_name} not found directly in model, skipping.")
            
    return quant_sim_model


if __name__ == '__main__':
    print("--- Testing Core Quantization Functions ---")
    # (Core function tests from previous turn remain here - keeping them for completeness)
    test_data = torch.randn(10, 5) * 10 
    test_data_zeros = torch.zeros(5,5)
    test_data_single_val = torch.ones(5,5) * 5.0
    print("\nTesting get_symmetric_range_and_scale...")
    r_min, r_max, scale, zp = get_symmetric_range_and_scale(test_data, bits=8)
    assert zp.item() == 0
    assert scale.item() > 0
    _, _, scale_zeros, _ = get_symmetric_range_and_scale(test_data_zeros, bits=8)
    assert scale_zeros.item() > 0 and scale_zeros.item() < 1e-7
    _, _, scale_single_val, _ = get_symmetric_range_and_scale(test_data_single_val, bits=8)
    expected_scale_single = 5.0/127.0
    assert abs(scale_single_val.item() - expected_scale_single) < 1e-6
    print("get_symmetric_range_and_scale tests passed.")

    print("\nTesting quantize_tensor & dequantize_tensor...")
    q_data = quantize_tensor(test_data, scale, zp, bits=8)
    assert q_data.dtype == torch.int8
    deq_data = dequantize_tensor(q_data, scale, zp)
    assert deq_data.dtype == torch.float32
    mse = F.mse_loss(deq_data, test_data).item()
    assert mse > 1e-6 # Expect some quantization error
    print(f"Quantize/Dequantize MSE: {mse:.4f}. Tests passed.")

    print("\n--- Testing QuantizedLinear (standalone) ---")
    try:
        temp_original_model = ActorCritic(input_dims=8, actor_output_dims=3)
        temp_original_fc1 = temp_original_model.fc1
        quant_fc1_standalone = QuantizedLinear(temp_original_fc1, weight_bits=8, bias_bits=8, activation_bits=8)
        dummy_layer_input = torch.randn(4, temp_original_fc1.in_features) 
        quant_fc1_output = quant_fc1_standalone(dummy_layer_input)
        original_fc1_output = temp_original_fc1(dummy_layer_input)
        output_mse_standalone = F.mse_loss(quant_fc1_output, original_fc1_output).item()
        is_different_standalone = not torch.allclose(quant_fc1_output, original_fc1_output, atol=1e-7, rtol=1e-5)
        assert output_mse_standalone > 1e-7 or is_different_standalone, \
            f"QuantizedLinear standalone output too similar. MSE: {output_mse_standalone:.2e}"
        print(f"QuantizedLinear standalone MSE: {output_mse_standalone:.8e}. Test passed.")
    except ImportError:
        print("Skipping QuantizedLinear standalone test (ActorCritic not found).")
    except Exception as e:
        print(f"Error during QuantizedLinear standalone test: {e}")
        raise e

    print("\n--- Testing Model Conversion to Quantized Simulation ---")
    try:
        original_model_for_conversion = ActorCritic(input_dims=8, actor_output_dims=3)
        
        layer_configs_test = {
            'fc1': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 8}, 
            'fc2': {'weight_bits': 5, 'activation_bits': 5, 'bias_bits': 10}, 
            'actor_head': {'weight_bits': 7, 'activation_bits': 7}, # Test default bias_bits (16)
            # 'critic_head' will remain unquantized as it's not in layer_configs
        }

        print("\nOriginal Model for conversion (first few layers):")
        print(f"  fc1: {original_model_for_conversion.fc1}")
        print(f"  fc2: {original_model_for_conversion.fc2}")
        print(f"  actor_head: {original_model_for_conversion.actor_head}")
        print(f"  critic_head: {original_model_for_conversion.critic_head}")


        quant_sim_model_test = convert_model_to_quantized_simulation(original_model_for_conversion, layer_configs_test)
        
        print("\nQuantized Simulation Model (first few layers):")
        print(f"  fc1: {quant_sim_model_test.fc1} (w:{quant_sim_model_test.fc1.weight_bits}, a:{quant_sim_model_test.fc1.activation_bits}, b:{quant_sim_model_test.fc1.bias_bits})")
        print(f"  fc2: {quant_sim_model_test.fc2} (w:{quant_sim_model_test.fc2.weight_bits}, a:{quant_sim_model_test.fc2.activation_bits}, b:{quant_sim_model_test.fc2.bias_bits})")
        print(f"  actor_head: {quant_sim_model_test.actor_head} (w:{quant_sim_model_test.actor_head.weight_bits}, a:{quant_sim_model_test.actor_head.activation_bits}, b:{quant_sim_model_test.actor_head.bias_bits})")
        print(f"  critic_head: {quant_sim_model_test.critic_head}")


        # Verify layer replacements and bit-width settings
        print("\nVerifying layer replacements and bit-widths...")
        
        assert isinstance(quant_sim_model_test.fc1, QuantizedLinear), "fc1 was not replaced."
        assert quant_sim_model_test.fc1.weight_bits == 8, f"fc1.weight_bits"
        assert quant_sim_model_test.fc1.activation_bits == 8, f"fc1.activation_bits"
        assert quant_sim_model_test.fc1.bias_bits == 8, f"fc1.bias_bits"
        print("fc1 verified (QuantizedLinear, w:8, a:8, b:8).")

        assert isinstance(quant_sim_model_test.fc2, QuantizedLinear), "fc2 was not replaced."
        assert quant_sim_model_test.fc2.weight_bits == 5, f"fc2.weight_bits"
        assert quant_sim_model_test.fc2.activation_bits == 5, f"fc2.activation_bits"
        assert quant_sim_model_test.fc2.bias_bits == 10, f"fc2.bias_bits"
        print("fc2 verified (QuantizedLinear, w:5, a:5, b:10).")

        assert isinstance(quant_sim_model_test.actor_head, QuantizedLinear), "actor_head was not replaced."
        assert quant_sim_model_test.actor_head.weight_bits == 7, f"actor_head.weight_bits"
        assert quant_sim_model_test.actor_head.activation_bits == 7, f"actor_head.activation_bits"
        assert quant_sim_model_test.actor_head.bias_bits == 16, f"actor_head.bias_bits (default)" # Testing default
        print("actor_head verified (QuantizedLinear, w:7, a:7, b:16 default).")

        assert isinstance(quant_sim_model_test.critic_head, nn.Linear), "critic_head should not have been replaced."
        print("critic_head correctly remains nn.Linear.")

        print("\nTesting forward pass of quant_sim_model_test...")
        dummy_model_input = torch.randn(2, 8) 
        action_probs, value_estimate = quant_sim_model_test(dummy_model_input)
        
        assert action_probs.shape == (2, 3), f"Action probs shape: {action_probs.shape}"
        assert value_estimate.shape == (2, 1), f"Value estimate shape: {value_estimate.shape}"
        print("Forward pass successful with quant_sim_model_test.")

    except ImportError:
        print("\nSkipping model conversion test as ActorCritic is not found in the expected path.")
    except Exception as e:
        print(f"\nError during model conversion test: {e}")
        raise e # Re-raise exception to make test failure clear

    print("\nPost-Training Quantization (PTQ) tests completed (including model conversion).")
