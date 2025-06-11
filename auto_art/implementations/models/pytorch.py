"""
PyTorch model implementation.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Sequence
from ...core.base import BaseModel, ModelMetadata
import numpy as np # Added for preprocess_input

T = TypeVar('T')

class PyTorchModel(BaseModel[T]):
    """PyTorch model implementation with Template Method pattern."""
    
    def __init__(self):
        self.supported_extensions = {'.pt', '.pth', '.ckpt'}
        self._model_instance: Optional[nn.Module] = None

    def load_model(self, model_path: str) -> Tuple[nn.Module, str]:
        """Load PyTorch model from path."""
        if not any(model_path.endswith(ext) for ext in self.supported_extensions):
            raise ValueError(f"Unsupported model file extension. Supported: {self.supported_extensions}")
        
        try:
            try:
                model = torch.load(model_path, map_location=torch.device('cpu'))
            except Exception:
                model = torch.load(model_path)

            if isinstance(model, dict) and 'state_dict' in model:
                raise ValueError("Loading model from state_dict checkpoint requires the model class definition, which is not supported in this generic loader yet.")

            if not isinstance(model, nn.Module):
                raise ValueError(f"Loaded file is not a PyTorch nn.Module. Got type: {type(model)}")

            model.eval()
            self._model_instance = model
            return model, 'pytorch'
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from {model_path}: {str(e)}")

    def _get_layer_config(self, layer: nn.Module) -> Dict[str, Any]:
        config = {}
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            config.update({
                'in_channels': layer.in_channels, 'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size, 'stride': layer.stride,
                'padding': layer.padding, 'dilation': layer.dilation, 'groups': layer.groups,
                'bias': layer.bias is not None
            })
        elif isinstance(layer, nn.Linear):
            config.update({'in_features': layer.in_features, 'out_features': layer.out_features, 'bias': layer.bias is not None})
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if isinstance(layer, nn.GroupNorm):
                config.update({'num_groups': layer.num_groups, 'num_channels': layer.num_channels})
            else:
                config.update({'num_features': getattr(layer, 'num_features', None) or getattr(layer, 'normalized_shape', None)})
            config.update({'eps': layer.eps, 'momentum': getattr(layer, 'momentum', None),
                           'affine': layer.affine, 'track_running_stats': getattr(layer, 'track_running_stats', None)})
        elif isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.LogSoftmax, nn.ELU, nn.SELU, nn.GELU, nn.SiLU, nn.Mish)):
            if isinstance(layer, nn.LeakyReLU): config['negative_slope'] = layer.negative_slope
            if isinstance(layer, nn.ELU): config['alpha'] = layer.alpha
            if isinstance(layer, (nn.Softmax, nn.LogSoftmax)): config['dim'] = layer.dim
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
            config.update({
                'kernel_size': getattr(layer, 'kernel_size', None), 'stride': getattr(layer, 'stride', None),
                'padding': getattr(layer, 'padding', None), 'dilation': getattr(layer, 'dilation', None) if hasattr(layer, 'dilation') else None,
                'output_size': getattr(layer, 'output_size', None)
            })
        elif isinstance(layer, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            config.update({'p': layer.p, 'inplace': layer.inplace})
        elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
            config.update({
                'input_size': layer.input_size, 'hidden_size': layer.hidden_size,
                'num_layers': layer.num_layers, 'bias': layer.bias,
                'batch_first': layer.batch_first, 'dropout': layer.dropout,
                'bidirectional': layer.bidirectional
            })
        elif isinstance(layer, nn.Embedding):
            config.update({'num_embeddings': layer.num_embeddings, 'embedding_dim': layer.embedding_dim, 'padding_idx': layer.padding_idx})
        elif isinstance(layer, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
             config.update({
                'd_model': layer.d_model if hasattr(layer,'d_model') else (layer.linear1.in_features if hasattr(layer,'linear1') else None), # Common way to find d_model
                'nhead': layer.nhead if hasattr(layer,'nhead') else (layer.self_attn.num_heads if hasattr(layer,'self_attn') else None),
                'dim_feedforward': getattr(layer, 'dim_feedforward', None),
                'dropout': getattr(layer, 'dropout', None),
                'activation': getattr(layer, 'activation', None).__class__.__name__ if hasattr(layer, 'activation') else None
             })
        return config

    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module")

        layer_info_list: List[Dict[str, Any]] = []
        for name, mod in model.named_modules():
            if name == "" and len(list(mod.children())) == 0 and not list(mod.parameters()): # Empty root module
                pass # Avoid adding empty root if it has no children and no params itself

            layer_entry = {
                'name': name if name else model.__class__.__name__,
                'type': mod.__class__.__name__,
                'config': self._get_layer_config(mod),
                'num_params': sum(p.numel() for p in mod.parameters(recurse=False)), # Params of this module only
                'is_leaf': len(list(mod.children())) == 0,
                'repr': repr(mod)
            }
            layer_info_list.append(layer_entry)

        input_s: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = (0,) # Default
        output_s: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = (0,) # Default
        
        # Try to infer from first layer that has recognizable input features/channels
        for layer_details in layer_info_list:
            if layer_details['is_leaf']: # Focus on leaf modules for shape inference
                conf = layer_details['config']
                if conf.get('in_channels'): # Conv layers
                    input_s = (None, conf['in_channels']) # (Batch, Channels), H,W unknown statically
                    break
                if conf.get('in_features'): # Linear layers
                    input_s = (None, conf['in_features']) # (Batch, Features)
                    break
                if layer_details['type'] == 'Embedding': # Embedding layers
                    input_s = (None, None) # (Batch, SeqLen) - input is indices
                    break
                if conf.get('input_size'): # RNN layers
                    input_s = (None, None, conf['input_size']) if conf.get('batch_first', False) else (None, conf['input_size']) # (Batch, Seq, Feat) or (Seq, Batch, Feat)
                    break
        
        # Try to infer from last layer that has recognizable output features/channels
        for layer_details in reversed(layer_info_list):
            if layer_details['is_leaf']:
                conf = layer_details['config']
                if conf.get('out_channels'): # Conv layers
                    output_s = (None, conf['out_channels'])
                    break
                if conf.get('out_features'): # Linear layers
                    output_s = (None, conf['out_features'])
                    break
                if conf.get('hidden_size') and layer_details['type'] in ['LSTM', 'GRU', 'RNN']: # RNN layers
                    output_s = (None, None, conf['hidden_size']) if conf.get('batch_first', False) else (None, conf['hidden_size'])
                    break
        
        return ModelMetadata(
            model_type=self._determine_model_type(model, layer_info_list),
            framework='pytorch',
            input_shape=input_s,
            output_shape=output_s,
            input_type=self._determine_input_output_type(model, layer_info_list, is_input=True, current_input_shape=input_s),
            output_type=self._determine_input_output_type(model, layer_info_list, is_input=False, current_output_shape=output_s),
            layer_info=layer_info_list,
            additional_info=self._get_additional_info(model)
        )

    def _determine_model_type(self, model: nn.Module, layer_info: List[Dict[str, Any]]) -> str:
        if not layer_info: return 'unknown'
        last_layer_type_global = None
        final_layer_config = None

        for layer_details in reversed(layer_info):
            # Find the last "functional" layer (not just a container)
            # This heuristic might need refinement
            if layer_details['is_leaf'] and layer_details['type'] not in ['Dropout', 'Identity']:
                last_layer_type_global = layer_details['type']
                final_layer_config = layer_details['config']
                break
        
        if last_layer_type_global:
            if last_layer_type_global in ['Softmax', 'LogSoftmax', 'Sigmoid']: return 'classification'
            if last_layer_type_global == 'Linear':
                # If the output of Linear is 1, could be regression or binary classification (needs sigmoid)
                # If > 1, could be multi-class (needs softmax) or multi-output regression
                if final_layer_config and final_layer_config.get('out_features') == 1:
                    # Check if a sigmoid exists anywhere for binary classification
                    if any(l['type'] == 'Sigmoid' for l in layer_info if l['is_leaf']):
                        return 'classification' # Likely binary classification
                    return 'regression_or_binary_classification_logits'
                # If out_features > 1, could be multi-class logits or multi-output regression
                if any(l['type'] in ['Softmax', 'LogSoftmax'] for l in layer_info if l['is_leaf']):
                    return 'classification'
                return 'multi_output_regression_or_classification_logits'

        if any('ConvTranspose' in layer['type'] for layer in layer_info): return 'generator_or_decoder'
        if 'generator' in model.__class__.__name__.lower() or 'gan' in model.__class__.__name__.lower(): return 'generator'
        if any(layer['type'] in ['LSTM', 'GRU', 'RNN', 'TransformerEncoder', 'TransformerDecoder'] for layer in layer_info): return 'sequence_model'
        
        # Default fallback if only generic layers like Conv/Linear are found without clear output activation
        if last_layer_type_global and ('Conv' in last_layer_type_global or 'Linear' in last_layer_type_global):
             return 'features_or_regression' # Could be a feature extractor or a regression model

        return 'unknown'

    def _determine_input_output_type(self, model: nn.Module, layer_info: List[Dict[str, Any]], is_input: bool,
                                     current_input_shape=None, current_output_shape=None) -> str:
        if not layer_info: return 'unknown'
        
        # For input type, check first significant layers
        if is_input:
            for layer_details in layer_info:
                if not layer_details['is_leaf']: continue
                layer_type = layer_details['type']
                if 'Embedding' in layer_type: return 'text_sequence_ids'
                if 'Conv2d' in layer_type: return 'image_2d'
                if 'Conv3d' in layer_type: return 'image_3d_or_video'
                if 'Conv1d' in layer_type: return 'time_series_or_audio_or_text_embeddings'
                if 'Linear' in layer_type: return 'tabular_or_flattened_features'
                if any(rnn_type in layer_type for rnn_type in ['LSTM', 'GRU', 'RNN']): return 'sequence_features'
            if current_input_shape and len(current_input_shape) > 1 : # (Batch, C, H, W) or (Batch, Seq, Feat)
                if len(current_input_shape) == 4 : return 'image_2d' # (B,C,H,W)
                if len(current_input_shape) == 3 : return 'sequence_features' # (B,Seq,Feat) or (B,Feat,Seq)
                if len(current_input_shape) == 2 : return 'tabular_or_flattened_features' # (B,Feat)
        else: # For output type
            # Use the determined model_type or last layer type as a hint
            model_type = self._determine_model_type(model, layer_info)
            if 'classification' in model_type: return 'logits_or_probabilities'
            if 'regression' in model_type: return 'numerical_predictions'
            if 'generator' in model_type or 'decoder' in model_type: return 'generated_data_e.g_image_or_sequence'
            if 'sequence_model' in model_type: return 'sequence_output'
            # Fallback based on shape if possible
            if current_output_shape and len(current_output_shape) > 1:
                if len(current_output_shape) == 2 and current_output_shape[1] > 1: return 'logits_or_multi_value_regression'
                if len(current_output_shape) == 2 and current_output_shape[1] == 1: return 'single_value_regression_or_binary_logit'
        return 'unknown'

    def _get_additional_info(self, model: nn.Module) -> Dict[str, Any]:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        device = 'cpu'
        try:
            first_param = next(model.parameters())
            device = str(first_param.device)
        except StopIteration: pass
        except Exception: device = 'unknown'
        return {
            'num_parameters': num_params, 'num_trainable_parameters': num_trainable,
            'device': device, 'model_class': model.__class__.__name__,
            'repr_string': repr(model)
        }
    
    def preprocess_input(self, input_data: T) -> T:
        if isinstance(input_data, torch.Tensor): return input_data
        if isinstance(input_data, np.ndarray): return torch.from_numpy(input_data.astype(np.float32)) # Common case for ART
        if isinstance(input_data, (list, tuple)):
            try: # Attempt to convert list of numbers/arrays to tensor
                if all(isinstance(x, (int, float)) for x in input_data):
                    return torch.tensor(input_data, dtype=torch.float32)
                # If list of numpy arrays, attempt to stack or handle appropriately
                # This part may need specific handling based on expected model input structure
                if all(isinstance(x, np.ndarray) for x in input_data):
                     return torch.stack([torch.from_numpy(x.astype(np.float32)) for x in input_data])
                return torch.tensor(input_data) # Fallback for other list types
            except Exception as e:
                raise ValueError(f"Could not convert list/tuple input to tensor: {e}")
        try: # Catch-all for single elements
            return torch.tensor(input_data, dtype=torch.float32)
        except Exception as e:
             raise ValueError(f"Unsupported input type for PyTorch preprocessing: {type(input_data)}, error: {e}")

    def postprocess_output(self, output_data: T) -> T:
        if isinstance(output_data, torch.Tensor): return output_data.detach().cpu().numpy()
        if isinstance(output_data, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in output_data):
            return [t.detach().cpu().numpy() for t in output_data]
        return output_data

    def get_model_predictions(self, model: Any, data: T) -> T:
        current_model = model if model is not None else self._model_instance
        if not isinstance(current_model, nn.Module):
            raise ValueError("Provided model is not an nn.Module instance or model not loaded.")

        processed_data = self.preprocess_input(data)
        
        try:
            model_device = next(current_model.parameters()).device
            if isinstance(processed_data, torch.Tensor) and processed_data.device != model_device:
                processed_data = processed_data.to(model_device)
            elif isinstance(processed_data, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in processed_data):
                 processed_data = [t.to(model_device) if t.device != model_device else t for t in processed_data]
        except StopIteration: pass
        except Exception: pass

        with torch.no_grad():
            predictions = current_model(processed_data)
        return self.postprocess_output(predictions)
