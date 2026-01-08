import os
import sys
import importlib
import ctypes
import numpy as np
import time
import importlib.machinery
import os

# Allow submodule imports (namespace-like) alongside this module file.
__package__ = __name__
_this_dir = os.path.dirname(__file__)
__spec__ = importlib.machinery.ModuleSpec(__name__, None)
__spec__.submodule_search_locations = [_this_dir]
__path__ = __spec__.submodule_search_locations

# Load shared library built by Makefile (libtinyfin.so)
_here = os.path.dirname(__file__)
_lib_path = os.path.join(_here, '..', 'libtinyfin.so')
_lib_path = os.path.normpath(_lib_path)
lib = ctypes.CDLL(_lib_path)

def _higher_order_enabled():
    return os.environ.get("TINYFIN_ENABLE_HIGHER_ORDER", "").lower() in ("1", "true", "yes", "on")

# Function prototypes
lib.py_tensor_new.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int))
lib.py_tensor_new.restype = ctypes.c_void_p

lib.py_tensor_free.argtypes = (ctypes.c_void_p,)
lib.py_tensor_free.restype = None

lib.py_tensor_ndim.argtypes = (ctypes.c_void_p,)
lib.py_tensor_ndim.restype = ctypes.c_int

lib.py_tensor_shape_get.argtypes = (ctypes.c_void_p, ctypes.c_int)
lib.py_tensor_shape_get.restype = ctypes.c_int

lib.py_tensor_data_ptr.argtypes = (ctypes.c_void_p,)
lib.py_tensor_data_ptr.restype = ctypes.POINTER(ctypes.c_float)

lib.py_tensor_save.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
lib.py_tensor_save.restype = ctypes.c_int

lib.py_tensor_load.argtypes = (ctypes.c_char_p,)
lib.py_tensor_load.restype = ctypes.c_void_p

lib.py_tensor_new_like.argtypes = (ctypes.c_void_p, ctypes.c_int)
lib.py_tensor_new_like.restype = ctypes.c_void_p

lib.py_sum.argtypes = (ctypes.c_void_p,)
lib.py_sum.restype = ctypes.c_void_p

lib.py_tensor_get_requires_grad = lib.py_tensor_get_requires_grad if hasattr(lib, 'py_tensor_get_requires_grad') else None
if lib.py_tensor_get_requires_grad:
    lib.py_tensor_get_requires_grad.argtypes = (ctypes.c_void_p,)
    lib.py_tensor_get_requires_grad.restype = ctypes.c_int

lib.py_tensor_set_requires_grad = lib.py_tensor_set_requires_grad if hasattr(lib, 'py_tensor_set_requires_grad') else None
if lib.py_tensor_set_requires_grad:
    lib.py_tensor_set_requires_grad.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_tensor_set_requires_grad.restype = None

lib.py_tensor_grad_ptr.argtypes = (ctypes.c_void_p,)
lib.py_tensor_grad_ptr.restype = ctypes.POINTER(ctypes.c_float)

lib.py_tensor_set_dtype = lib.py_tensor_set_dtype if hasattr(lib, 'py_tensor_set_dtype') else None
if lib.py_tensor_set_dtype:
    lib.py_tensor_set_dtype.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_tensor_set_dtype.restype = ctypes.c_int

lib.py_tensor_get_device = lib.py_tensor_get_device if hasattr(lib, 'py_tensor_get_device') else None
if lib.py_tensor_get_device:
    lib.py_tensor_get_device.argtypes = (ctypes.c_void_p,)
    lib.py_tensor_get_device.restype = ctypes.c_int

lib.py_tensor_set_device = lib.py_tensor_set_device if hasattr(lib, 'py_tensor_set_device') else None
if lib.py_tensor_set_device:
    lib.py_tensor_set_device.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_tensor_set_device.restype = None

lib.py_add.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
lib.py_add.restype = ctypes.c_void_p

lib.py_mul.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
lib.py_mul.restype = ctypes.c_void_p

lib.py_matmul.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
lib.py_matmul.restype = ctypes.c_void_p

lib.py_div.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
lib.py_div.restype = ctypes.c_void_p

lib.py_exp.argtypes = (ctypes.c_void_p,)
lib.py_exp.restype = ctypes.c_void_p

lib.py_log = lib.py_log if hasattr(lib, 'py_log') else None
if lib.py_log:
    lib.py_log.argtypes = (ctypes.c_void_p,)
    lib.py_log.restype = ctypes.c_void_p

lib.py_log_softmax = lib.py_log_softmax if hasattr(lib, 'py_log_softmax') else None
if lib.py_log_softmax:
    lib.py_log_softmax.argtypes = (ctypes.c_void_p,)
    lib.py_log_softmax.restype = ctypes.c_void_p

lib.py_logsumexp = lib.py_logsumexp if hasattr(lib, 'py_logsumexp') else None
if lib.py_logsumexp:
    lib.py_logsumexp.argtypes = (ctypes.c_void_p,)
    lib.py_logsumexp.restype = ctypes.c_void_p

lib.py_clamp_min = lib.py_clamp_min if hasattr(lib, 'py_clamp_min') else None
if lib.py_clamp_min:
    lib.py_clamp_min.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.py_clamp_min.restype = ctypes.c_void_p

lib.py_has_nan_inf = lib.py_has_nan_inf if hasattr(lib, 'py_has_nan_inf') else None
if lib.py_has_nan_inf:
    lib.py_has_nan_inf.argtypes = (ctypes.c_void_p,)
    lib.py_has_nan_inf.restype = ctypes.c_void_p

lib.py_softmax = lib.py_softmax if hasattr(lib, 'py_softmax') else None
if lib.py_softmax:
    lib.py_softmax.argtypes = (ctypes.c_void_p,)
    lib.py_softmax.restype = ctypes.c_void_p

# new ops
lib.py_sub = lib.py_sub if hasattr(lib, 'py_sub') else None
if lib.py_sub:
    lib.py_sub.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    lib.py_sub.restype = ctypes.c_void_p

lib.py_abs = lib.py_abs if hasattr(lib, 'py_abs') else None
if lib.py_abs:
    lib.py_abs.argtypes = (ctypes.c_void_p,)
    lib.py_abs.restype = ctypes.c_void_p

lib.py_neg = lib.py_neg if hasattr(lib, 'py_neg') else None
if lib.py_neg:
    lib.py_neg.argtypes = (ctypes.c_void_p,)
    lib.py_neg.restype = ctypes.c_void_p

lib.py_clamp = lib.py_clamp if hasattr(lib, 'py_clamp') else None
if lib.py_clamp:
    lib.py_clamp.argtypes = (ctypes.c_void_p, ctypes.c_float, ctypes.c_float)
    lib.py_clamp.restype = ctypes.c_void_p

lib.py_argmax = lib.py_argmax if hasattr(lib, 'py_argmax') else None
if lib.py_argmax:
    lib.py_argmax.argtypes = (ctypes.c_void_p,)
    lib.py_argmax.restype = ctypes.c_void_p

lib.py_argmin = lib.py_argmin if hasattr(lib, 'py_argmin') else None
if lib.py_argmin:
    lib.py_argmin.argtypes = (ctypes.c_void_p,)
    lib.py_argmin.restype = ctypes.c_void_p

lib.py_reshape = lib.py_reshape if hasattr(lib, 'py_reshape') else None
if lib.py_reshape:
    lib.py_reshape.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    lib.py_reshape.restype = ctypes.c_void_p

lib.py_maxpool2d = lib.py_maxpool2d if hasattr(lib, 'py_maxpool2d') else None
if lib.py_maxpool2d:
    lib.py_maxpool2d.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_maxpool2d.restype = ctypes.c_void_p

lib.py_transpose = lib.py_transpose if hasattr(lib, 'py_transpose') else None
if lib.py_transpose:
    lib.py_transpose.argtypes = (ctypes.c_void_p,)
    lib.py_transpose.restype = ctypes.c_void_p

lib.py_permute = lib.py_permute if hasattr(lib, 'py_permute') else None
if lib.py_permute:
    lib.py_permute.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    lib.py_permute.restype = ctypes.c_void_p

lib.py_concat = lib.py_concat if hasattr(lib, 'py_concat') else None
if lib.py_concat:
    lib.py_concat.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.py_concat.restype = ctypes.c_void_p

lib.py_stack = lib.py_stack if hasattr(lib, 'py_stack') else None
if lib.py_stack:
    lib.py_stack.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.py_stack.restype = ctypes.c_void_p

lib.py_pad2d = lib.py_pad2d if hasattr(lib, 'py_pad2d') else None
if lib.py_pad2d:
    lib.py_pad2d.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float)
    lib.py_pad2d.restype = ctypes.c_void_p
lib.py_slice = lib.py_slice if hasattr(lib, 'py_slice') else None
if lib.py_slice:
    lib.py_slice.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    lib.py_slice.restype = ctypes.c_void_p
lib.py_avgpool2d = lib.py_avgpool2d if hasattr(lib, 'py_avgpool2d') else None
if lib.py_avgpool2d:
    lib.py_avgpool2d.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_avgpool2d.restype = ctypes.c_void_p

lib.py_backend_get_name = lib.py_backend_get_name if hasattr(lib, 'py_backend_get_name') else None
if lib.py_backend_get_name:
    lib.py_backend_get_name.argtypes = ()
    lib.py_backend_get_name.restype = ctypes.c_char_p

lib.py_backend_set_by_name = lib.py_backend_set_by_name if hasattr(lib, 'py_backend_set_by_name') else None
if lib.py_backend_set_by_name:
    lib.py_backend_set_by_name.argtypes = (ctypes.c_char_p,)
    lib.py_backend_set_by_name.restype = ctypes.c_int

lib.py_squeeze = lib.py_squeeze if hasattr(lib, 'py_squeeze') else None
if lib.py_squeeze:
    lib.py_squeeze.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_squeeze.restype = ctypes.c_void_p

lib.py_unsqueeze = lib.py_unsqueeze if hasattr(lib, 'py_unsqueeze') else None
if lib.py_unsqueeze:
    lib.py_unsqueeze.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_unsqueeze.restype = ctypes.c_void_p

lib.py_sqrt = lib.py_sqrt if hasattr(lib, 'py_sqrt') else None
if lib.py_sqrt:
    lib.py_sqrt.argtypes = (ctypes.c_void_p,)
    lib.py_sqrt.restype = ctypes.c_void_p

lib.py_embedding = lib.py_embedding if hasattr(lib, 'py_embedding') else None
if lib.py_embedding:
    lib.py_embedding.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    lib.py_embedding.restype = ctypes.c_void_p

lib.py_conv2d = lib.py_conv2d if hasattr(lib, 'py_conv2d') else None
if lib.py_conv2d:
    lib.py_conv2d.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    lib.py_conv2d.restype = ctypes.c_void_p

# BatchNorm binding
lib.py_batchnorm = lib.py_batchnorm if hasattr(lib, 'py_batchnorm') else None
if lib.py_batchnorm:
    lib.py_batchnorm.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float)
    lib.py_batchnorm.restype = ctypes.c_void_p

# BCE and Cross-Entropy (logits) bindings
lib.py_bce_loss = lib.py_bce_loss if hasattr(lib, 'py_bce_loss') else None
if lib.py_bce_loss:
    lib.py_bce_loss.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.py_bce_loss.restype = ctypes.c_void_p

lib.py_cross_entropy_logits = lib.py_cross_entropy_logits if hasattr(lib, 'py_cross_entropy_logits') else None
if lib.py_cross_entropy_logits:
    lib.py_cross_entropy_logits.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.py_cross_entropy_logits.restype = ctypes.c_void_p

lib.py_zero_grad.argtypes = (ctypes.c_void_p,)
lib.py_zero_grad.restype = None

lib.py_backward.argtypes = (ctypes.c_void_p,)
lib.py_backward.restype = None
lib.py_backward_with_retain = lib.py_backward_with_retain if hasattr(lib, 'py_backward_with_retain') else None
if lib.py_backward_with_retain:
    lib.py_backward_with_retain.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_backward_with_retain.restype = None

lib.py_autograd_set_enabled = lib.py_autograd_set_enabled if hasattr(lib, 'py_autograd_set_enabled') else None
if lib.py_autograd_set_enabled:
    lib.py_autograd_set_enabled.argtypes = (ctypes.c_int,)
    lib.py_autograd_set_enabled.restype = None

lib.py_autograd_get_enabled = lib.py_autograd_get_enabled if hasattr(lib, 'py_autograd_get_enabled') else None
if lib.py_autograd_get_enabled:
    lib.py_autograd_get_enabled.argtypes = ()
    lib.py_autograd_get_enabled.restype = ctypes.c_int

lib.py_autograd_set_retain_default = lib.py_autograd_set_retain_default if hasattr(lib, 'py_autograd_set_retain_default') else None
if lib.py_autograd_set_retain_default:
    lib.py_autograd_set_retain_default.argtypes = (ctypes.c_int,)
    lib.py_autograd_set_retain_default.restype = None

lib.py_autograd_get_retain_default = lib.py_autograd_get_retain_default if hasattr(lib, 'py_autograd_get_retain_default') else None
if lib.py_autograd_get_retain_default:
    lib.py_autograd_get_retain_default.argtypes = ()
    lib.py_autograd_get_retain_default.restype = ctypes.c_int

# push/pop helpers for nested no_grad
lib.py_autograd_push_disabled = lib.py_autograd_push_disabled if hasattr(lib, 'py_autograd_push_disabled') else None
if lib.py_autograd_push_disabled:
    lib.py_autograd_push_disabled.argtypes = ()
    lib.py_autograd_push_disabled.restype = ctypes.c_int

lib.py_autograd_pop = lib.py_autograd_pop if hasattr(lib, 'py_autograd_pop') else None
if lib.py_autograd_pop:
    lib.py_autograd_pop.argtypes = (ctypes.c_int,)
    lib.py_autograd_pop.restype = None

# autograd graph export
lib.py_autograd_to_dot = lib.py_autograd_to_dot if hasattr(lib, 'py_autograd_to_dot') else None
if lib.py_autograd_to_dot:
    lib.py_autograd_to_dot.argtypes = (ctypes.c_void_p,)
    lib.py_autograd_to_dot.restype = ctypes.c_char_p

# tensor device move helper
lib.py_tensor_to_device = lib.py_tensor_to_device if hasattr(lib, 'py_tensor_to_device') else None
if lib.py_tensor_to_device:
    lib.py_tensor_to_device.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.py_tensor_to_device.restype = ctypes.c_void_p

lib.py_dropout = lib.py_dropout if hasattr(lib, 'py_dropout') else None
if lib.py_dropout:
    lib.py_dropout.argtypes = (ctypes.c_void_p, ctypes.c_float, ctypes.c_int)
    lib.py_dropout.restype = ctypes.c_void_p

# C-side profiler summary
lib.py_profiler_get_summary = lib.py_profiler_get_summary if hasattr(lib, 'py_profiler_get_summary') else None
if lib.py_profiler_get_summary:
    lib.py_profiler_get_summary.argtypes = ()
    lib.py_profiler_get_summary.restype = ctypes.c_char_p
lib.py_free_str = lib.py_free_str if hasattr(lib, 'py_free_str') else None
if lib.py_free_str:
    lib.py_free_str.argtypes = (ctypes.c_char_p,)
    lib.py_free_str.restype = None

# SGD optimizer bindings
lib.py_sgd_create = lib.py_sgd_create if hasattr(lib, 'py_sgd_create') else None
if lib.py_sgd_create:
    lib.py_sgd_create.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float)
    lib.py_sgd_create.restype = ctypes.c_void_p

lib.py_sgd_step = lib.py_sgd_step if hasattr(lib, 'py_sgd_step') else None
if lib.py_sgd_step:
    lib.py_sgd_step.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.py_sgd_step.restype = None

lib.py_sgd_zero_grad = lib.py_sgd_zero_grad if hasattr(lib, 'py_sgd_zero_grad') else None
if lib.py_sgd_zero_grad:
    lib.py_sgd_zero_grad.argtypes = (ctypes.c_void_p,)
    lib.py_sgd_zero_grad.restype = None

lib.py_sgd_set_lr = lib.py_sgd_set_lr if hasattr(lib, 'py_sgd_set_lr') else None
if lib.py_sgd_set_lr:
    lib.py_sgd_set_lr.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.py_sgd_set_lr.restype = None

lib.py_sgd_get_lr = lib.py_sgd_get_lr if hasattr(lib, 'py_sgd_get_lr') else None
if lib.py_sgd_get_lr:
    lib.py_sgd_get_lr.argtypes = (ctypes.c_void_p,)
    lib.py_sgd_get_lr.restype = ctypes.c_float

lib.py_sgd_set_param_lr = lib.py_sgd_set_param_lr if hasattr(lib, 'py_sgd_set_param_lr') else None
if lib.py_sgd_set_param_lr:
    lib.py_sgd_set_param_lr.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_float)
    lib.py_sgd_set_param_lr.restype = None

lib.py_sgd_set_param_weight_decay = lib.py_sgd_set_param_weight_decay if hasattr(lib, 'py_sgd_set_param_weight_decay') else None
if lib.py_sgd_set_param_weight_decay:
    lib.py_sgd_set_param_weight_decay.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_float)
    lib.py_sgd_set_param_weight_decay.restype = None

lib.py_sgd_save_state = lib.py_sgd_save_state if hasattr(lib, 'py_sgd_save_state') else None
if lib.py_sgd_save_state:
    lib.py_sgd_save_state.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.py_sgd_save_state.restype = ctypes.c_int

lib.py_sgd_load_state = lib.py_sgd_load_state if hasattr(lib, 'py_sgd_load_state') else None
if lib.py_sgd_load_state:
    lib.py_sgd_load_state.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.py_sgd_load_state.restype = ctypes.c_int
lib.py_sgd_free = lib.py_sgd_free if hasattr(lib, 'py_sgd_free') else None
if lib.py_sgd_free:
    lib.py_sgd_free.argtypes = (ctypes.c_void_p,)
    lib.py_sgd_free.restype = None

# Adam optimizer bindings
lib.py_adam_create = lib.py_adam_create if hasattr(lib, 'py_adam_create') else None
if lib.py_adam_create:
    lib.py_adam_create.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
    lib.py_adam_create.restype = ctypes.c_void_p

lib.py_adam_step = lib.py_adam_step if hasattr(lib, 'py_adam_step') else None
if lib.py_adam_step:
    lib.py_adam_step.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.py_adam_step.restype = None

lib.py_adam_zero_grad = lib.py_adam_zero_grad if hasattr(lib, 'py_adam_zero_grad') else None
if lib.py_adam_zero_grad:
    lib.py_adam_zero_grad.argtypes = (ctypes.c_void_p,)
    lib.py_adam_zero_grad.restype = None

lib.py_adam_set_lr = lib.py_adam_set_lr if hasattr(lib, 'py_adam_set_lr') else None
if lib.py_adam_set_lr:
    lib.py_adam_set_lr.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.py_adam_set_lr.restype = None

lib.py_adam_get_lr = lib.py_adam_get_lr if hasattr(lib, 'py_adam_get_lr') else None
if lib.py_adam_get_lr:
    lib.py_adam_get_lr.argtypes = (ctypes.c_void_p,)
    lib.py_adam_get_lr.restype = ctypes.c_float

lib.py_adam_save_state = lib.py_adam_save_state if hasattr(lib, 'py_adam_save_state') else None
if lib.py_adam_save_state:
    lib.py_adam_save_state.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.py_adam_save_state.restype = ctypes.c_int

lib.py_adam_load_state = lib.py_adam_load_state if hasattr(lib, 'py_adam_load_state') else None
if lib.py_adam_load_state:
    lib.py_adam_load_state.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.py_adam_load_state.restype = ctypes.c_int

lib.py_adam_free = lib.py_adam_free if hasattr(lib, 'py_adam_free') else None
if lib.py_adam_free:
    lib.py_adam_free.argtypes = (ctypes.c_void_p,)
    lib.py_adam_free.restype = None

# RMSProp optimizer bindings
lib.py_rmsprop_create = lib.py_rmsprop_create if hasattr(lib, 'py_rmsprop_create') else None
if lib.py_rmsprop_create:
    lib.py_rmsprop_create.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
    lib.py_rmsprop_create.restype = ctypes.c_void_p

lib.py_rmsprop_step = lib.py_rmsprop_step if hasattr(lib, 'py_rmsprop_step') else None
if lib.py_rmsprop_step:
    lib.py_rmsprop_step.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.py_rmsprop_step.restype = None

lib.py_rmsprop_zero_grad = lib.py_rmsprop_zero_grad if hasattr(lib, 'py_rmsprop_zero_grad') else None
if lib.py_rmsprop_zero_grad:
    lib.py_rmsprop_zero_grad.argtypes = (ctypes.c_void_p,)
    lib.py_rmsprop_zero_grad.restype = None

lib.py_rmsprop_set_lr = lib.py_rmsprop_set_lr if hasattr(lib, 'py_rmsprop_set_lr') else None
if lib.py_rmsprop_set_lr:
    lib.py_rmsprop_set_lr.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.py_rmsprop_set_lr.restype = None

lib.py_rmsprop_get_lr = lib.py_rmsprop_get_lr if hasattr(lib, 'py_rmsprop_get_lr') else None
if lib.py_rmsprop_get_lr:
    lib.py_rmsprop_get_lr.argtypes = (ctypes.c_void_p,)
    lib.py_rmsprop_get_lr.restype = ctypes.c_float

lib.py_rmsprop_save_state = lib.py_rmsprop_save_state if hasattr(lib, 'py_rmsprop_save_state') else None
if lib.py_rmsprop_save_state:
    lib.py_rmsprop_save_state.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.py_rmsprop_save_state.restype = ctypes.c_int

lib.py_rmsprop_load_state = lib.py_rmsprop_load_state if hasattr(lib, 'py_rmsprop_load_state') else None
if lib.py_rmsprop_load_state:
    lib.py_rmsprop_load_state.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.py_rmsprop_load_state.restype = ctypes.c_int

lib.py_rmsprop_free = lib.py_rmsprop_free if hasattr(lib, 'py_rmsprop_free') else None
if lib.py_rmsprop_free:
    lib.py_rmsprop_free.argtypes = (ctypes.c_void_p,)
    lib.py_rmsprop_free.restype = None


class Tensor:
    def __init__(self, ptr):
        self._ptr = ctypes.c_void_p(ptr)

    @classmethod
    def new(cls, shape, requires_grad=False):
        ndim = len(shape)
        arr = (ctypes.c_int * ndim)(*shape)
        p = lib.py_tensor_new(ndim, arr)
        t = cls(p)
        if requires_grad and lib.py_tensor_set_requires_grad:
            lib.py_tensor_set_requires_grad(t._ptr, 1)
        return t

    @classmethod
    def rand(cls, *shape, requires_grad=False, low=0.0, high=1.0):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if len(shape) == 0:
            raise ValueError("rand expects a shape")
        t = cls.new(list(shape), requires_grad=requires_grad)
        t.numpy_view()[:] = np.random.uniform(low, high, size=t.shape()).astype(np.float32)
        return t

    @classmethod
    def randn(cls, *shape, requires_grad=False, mean=0.0, std=1.0):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if len(shape) == 0:
            raise ValueError("randn expects a shape")
        t = cls.new(list(shape), requires_grad=requires_grad)
        t.numpy_view()[:] = (np.random.randn(*t.shape()) * std + mean).astype(np.float32)
        return t

    @classmethod
    def kaiming_uniform(cls, *shape, requires_grad=False, a=0.0):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if len(shape) < 1:
            raise ValueError("kaiming_uniform expects a shape")
        if len(shape) < 2:
            raise ValueError("kaiming_uniform expects at least 2 dims for fan-in")
        fan_in = float(shape[0])
        gain = np.sqrt(2.0 / (1.0 + float(a) ** 2))
        bound = np.sqrt(3.0) * gain / np.sqrt(fan_in)
        t = cls.new(list(shape), requires_grad=requires_grad)
        t.numpy_view()[:] = np.random.uniform(-bound, bound, size=t.shape()).astype(np.float32)
        return t

    @classmethod
    def xavier_uniform(cls, *shape, requires_grad=False):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if len(shape) < 1:
            raise ValueError("xavier_uniform expects a shape")
        if len(shape) < 2:
            raise ValueError("xavier_uniform expects at least 2 dims for fan-in/fan-out")
        fan_in = float(shape[0])
        fan_out = float(shape[1])
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        t = cls.new(list(shape), requires_grad=requires_grad)
        t.numpy_view()[:] = np.random.uniform(-bound, bound, size=t.shape()).astype(np.float32)
        return t

    @classmethod
    def load(cls, path):
        p = lib.py_tensor_load(path.encode())
        return cls(p)

    def save(self, path):
        return lib.py_tensor_save(self._ptr, path.encode())

    def free(self):
        lib.py_tensor_free(self._ptr)

    def shape(self):
        ndim = lib.py_tensor_ndim(self._ptr)
        return [lib.py_tensor_shape_get(self._ptr, i) for i in range(ndim)]

    def to_numpy(self):
        shp = tuple(self.shape())
        size = 1
        for d in shp: size *= d
        data_ptr = lib.py_tensor_data_ptr(self._ptr)
        arr = np.ctypeslib.as_array(data_ptr, shape=(size,))
        return arr.reshape(shp).copy()

    def item(self):
        arr = self.to_numpy()
        if arr.size != 1:
            raise ValueError(f"item() expects a single value, got shape {arr.shape}")
        return arr.item()

    def numpy_view(self):
        """Return a numpy view (no copy) over the tensor's underlying data."""
        shp = tuple(self.shape())
        size = 1
        for d in shp: size *= d
        data_ptr = lib.py_tensor_data_ptr(self._ptr)
        arr = np.ctypeslib.as_array(data_ptr, shape=(size,))
        return arr.reshape(shp)

    def reshape(self, shape):
        if not lib.py_reshape: raise RuntimeError('reshape not available in C API')
        if isinstance(shape, Tensor):
            shp = [int(v) for v in shape.to_numpy().flatten().tolist()]
        else:
            shp = list(shape)
        curr_size = 1
        for d in self.shape(): curr_size *= d
        new_size = 1
        for d in shp: new_size *= d
        if curr_size != new_size:
            raise ValueError(f"reshape size mismatch: {self.shape()} -> {shp}")
        shp_arr = (ctypes.c_int * len(shp))(*shp)
        p = lib.py_reshape(self._ptr, len(shp), shp_arr)
        return Tensor(p)

    def flatten(self, start_dim=0):
        shape = self.shape()
        ndim = len(shape)
        if start_dim < 0:
            start_dim += ndim
        if start_dim < 0 or start_dim >= ndim:
            raise ValueError(f"flatten start_dim out of range for ndim {ndim}")
        if start_dim == 0:
            total = 1
            for d in shape:
                total *= d
            return self.reshape([total])
        prefix = shape[:start_dim]
        tail = 1
        for d in shape[start_dim:]:
            tail *= d
        return self.reshape(prefix + [tail])

    @classmethod
    def new_like(cls, other, requires_grad=0):
        p = lib.py_tensor_new_like(other._ptr, int(requires_grad))
        return cls(p)

    def set_dtype(self, dtype):
        if lib.py_tensor_set_dtype:
            return lib.py_tensor_set_dtype(self._ptr, int(dtype))
        raise RuntimeError("set_dtype not available in C API")

    def get_device(self):
        if lib.py_tensor_get_device:
            return lib.py_tensor_get_device(self._ptr)
        raise RuntimeError("get_device not available in C API")

    def set_device(self, device):
        if lib.py_tensor_set_device:
            lib.py_tensor_set_device(self._ptr, int(device))
        else:
            raise RuntimeError("set_device not available in C API")

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"add expects Tensor, got {type(other)}")
        if self.get_device() != other.get_device():
            raise ValueError(f"add device mismatch: {self.get_device()} vs {other.get_device()}")
        p = lib.py_add(self._ptr, other._ptr)
        return Tensor(p)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"sub expects Tensor, got {type(other)}")
        if self.get_device() != other.get_device():
            raise ValueError(f"sub device mismatch: {self.get_device()} vs {other.get_device()}")
        p = lib.py_sub(self._ptr, other._ptr)
        return Tensor(p)

    def abs(self):
        p = lib.py_abs(self._ptr)
        return Tensor(p)

    def neg(self):
        p = lib.py_neg(self._ptr)
        return Tensor(p)

    def clamp(self, minv, maxv):
        p = lib.py_clamp(self._ptr, ctypes.c_float(minv), ctypes.c_float(maxv))
        return Tensor(p)

    def argmax(self):
        p = lib.py_argmax(self._ptr)
        if not p: return None
        t = Tensor(p)
        return int(t.to_numpy().item())

    def argmin(self):
        p = lib.py_argmin(self._ptr)
        if not p: return None
        t = Tensor(p)
        return int(t.to_numpy().item())

    def squeeze(self, dim=-1):
        ndim = lib.py_tensor_ndim(self._ptr)
        if dim < -ndim or dim >= ndim:
            raise ValueError(f"squeeze dim out of range for tensor of ndim {ndim}")
        p = lib.py_squeeze(self._ptr, int(dim))
        return Tensor(p)

    def unsqueeze(self, dim):
        ndim = lib.py_tensor_ndim(self._ptr)
        if dim < -(ndim + 1) or dim > ndim:
            raise ValueError(f"unsqueeze dim out of range for tensor of ndim {ndim}")
        p = lib.py_unsqueeze(self._ptr, int(dim))
        return Tensor(p)

    def sqrt(self):
        p = lib.py_sqrt(self._ptr)
        return Tensor(p)

    def embedding(self, indices):
        if not isinstance(indices, Tensor):
            raise TypeError(f"embedding expects Tensor indices, got {type(indices)}")
        if self.get_device() != indices.get_device():
            raise ValueError(f"embedding device mismatch: {self.get_device()} vs {indices.get_device()}")
        p = lib.py_embedding(self._ptr, indices._ptr)
        return Tensor(p)

    def conv2d(self, weight, bias=None):
        if not isinstance(weight, Tensor):
            raise TypeError(f"conv2d expects Tensor weight, got {type(weight)}")
        if bias is not None and not isinstance(bias, Tensor):
            raise TypeError(f"conv2d bias must be Tensor or None, got {type(bias)}")
        if self.get_device() != weight.get_device() or (bias is not None and self.get_device() != bias.get_device()):
            raise ValueError("conv2d device mismatch among input/weight/bias")
        x_shape = self.shape()
        w_shape = weight.shape()
        if len(x_shape) != 4 or len(w_shape) != 4:
            raise ValueError(f"conv2d expects input [N,C,H,W] and weight [O,I,kH,kW], got {x_shape} and {w_shape}")
        if x_shape[1] != w_shape[1]:
            raise ValueError(f"conv2d channel mismatch: input C={x_shape[1]} vs weight in_channels={w_shape[1]}")
        if bias is not None:
            b_shape = bias.shape()
            if len(b_shape) != 1 or b_shape[0] != w_shape[0]:
                raise ValueError(f"conv2d bias shape mismatch: expected [{w_shape[0]}], got {b_shape}")
        bptr = bias._ptr if bias is not None else ctypes.c_void_p(0)
        p = lib.py_conv2d(self._ptr, weight._ptr, bptr)
        return Tensor(p)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"div expects Tensor, got {type(other)}")
        if self.get_device() != other.get_device():
            raise ValueError(f"div device mismatch: {self.get_device()} vs {other.get_device()}")
        a_shape = self.shape()
        b_shape = other.shape()
        # basic broadcast sanity mirroring numpy rules
        max_len = max(len(a_shape), len(b_shape))
        for i in range(1, max_len + 1):
            ad = a_shape[-i] if i <= len(a_shape) else 1
            bd = b_shape[-i] if i <= len(b_shape) else 1
            if ad != bd and ad != 1 and bd != 1:
                raise ValueError(f"div shape mismatch: {self.shape()} vs {other.shape()}")
        p = lib.py_div(self._ptr, other._ptr)
        return Tensor(p)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"mul expects Tensor, got {type(other)}")
        if self.get_device() != other.get_device():
            raise ValueError(f"mul device mismatch: {self.get_device()} vs {other.get_device()}")
        p = lib.py_mul(self._ptr, other._ptr)
        return Tensor(p)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"matmul expects Tensor, got {type(other)}")
        if self.get_device() != other.get_device():
            raise ValueError(f"matmul device mismatch: {self.get_device()} vs {other.get_device()}")
        a_shape = self.shape()
        b_shape = other.shape()
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"matmul expects 2D tensors, got {a_shape} and {b_shape}")
        if a_shape[1] != b_shape[0]:
            raise ValueError(f"matmul shape mismatch: {a_shape} @ {b_shape}")
        p = lib.py_matmul(self._ptr, other._ptr)
        return Tensor(p)

    def dot(self, other):
        return self.matmul(other)

    def exp(self):
        p = lib.py_exp(self._ptr)
        return Tensor(p)

    def log(self):
        if not lib.py_log: raise RuntimeError('log not available in C API')
        p = lib.py_log(self._ptr)
        return Tensor(p)

    def log_softmax(self):
        if not lib.py_log_softmax: raise RuntimeError('log_softmax not available in C API')
        p = lib.py_log_softmax(self._ptr)
        return Tensor(p)

    def logsumexp(self):
        if not lib.py_logsumexp: raise RuntimeError('logsumexp not available in C API')
        p = lib.py_logsumexp(self._ptr)
        return Tensor(p)

    def clamp_min(self, min_val):
        if not lib.py_clamp_min: raise RuntimeError('clamp_min not available in C API')
        p = lib.py_clamp_min(self._ptr, ctypes.c_float(min_val))
        return Tensor(p)

    def relu(self):
        if not lib.py_clamp_min:
            raise RuntimeError('relu not available in C API')
        return self.clamp_min(0.0)

    def softmax(self):
        if not lib.py_softmax: raise RuntimeError('softmax not available in C API')
        p = lib.py_softmax(self._ptr)
        return Tensor(p)

    def maxpool2d(self, kernel_size):
        if not lib.py_maxpool2d: raise RuntimeError('maxpool2d not available in C API')
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("maxpool2d kernel_size must be positive int")
        x_shape = self.shape()
        if len(x_shape) != 4:
            raise ValueError(f"maxpool2d expects input [N,C,H,W], got {x_shape}")
        if x_shape[2] % kernel_size != 0 or x_shape[3] % kernel_size != 0:
            raise ValueError(f"maxpool2d kernel_size {kernel_size} must divide H,W {x_shape[2:]}")
        p = lib.py_maxpool2d(self._ptr, ctypes.c_int(kernel_size))
        return Tensor(p)

    def transpose(self):
        if not lib.py_transpose: raise RuntimeError('transpose not available in C API')
        p = lib.py_transpose(self._ptr)
        return Tensor(p)

    def permute(self, order):
        if not lib.py_permute: raise RuntimeError('permute not available in C API')
        order = list(order)
        shape = self.shape()
        ndim = len(shape)
        if len(order) != ndim:
            raise ValueError(f"permute order length {len(order)} must match tensor ndim {ndim}")
        if sorted(order) != list(range(ndim)):
            raise ValueError(f"permute order must be a permutation of [0,{ndim-1}]")
        arr = (ctypes.c_int * len(order))(*order)
        p = lib.py_permute(self._ptr, len(order), arr)
        return Tensor(p)

    def concat(self, other, axis=0):
        if not lib.py_concat: raise RuntimeError('concat not available in C API')
        if not isinstance(other, Tensor):
            raise TypeError("concat expects Tensor")
        if self.get_device() != other.get_device():
            raise ValueError("concat device mismatch")
        a_shape = self.shape()
        b_shape = other.shape()
        if len(a_shape) != len(b_shape):
            raise ValueError(f"concat ndim mismatch: {len(a_shape)} vs {len(b_shape)}")
        ndim = len(a_shape)
        if not isinstance(axis, int):
            raise TypeError("concat axis must be int")
        if axis < -ndim or axis >= ndim:
            raise ValueError(f"concat axis {axis} out of range for ndim {ndim}")
        axis = axis % ndim
        for i, (ad, bd) in enumerate(zip(a_shape, b_shape)):
            if i == axis:
                continue
            if ad != bd:
                raise ValueError(f"concat shape mismatch at dim {i}: {a_shape} vs {b_shape}")
        p = lib.py_concat(self._ptr, other._ptr, int(axis))
        if not p:
            raise ValueError("concat failed; ensure shapes are compatible")
        return Tensor(p)

    def stack(self, other, axis=0):
        if not lib.py_stack: raise RuntimeError('stack not available in C API')
        if not isinstance(other, Tensor):
            raise TypeError("stack expects Tensor")
        if self.get_device() != other.get_device():
            raise ValueError("stack device mismatch")
        a_shape = self.shape()
        b_shape = other.shape()
        if len(a_shape) != len(b_shape):
            raise ValueError(f"stack ndim mismatch: {len(a_shape)} vs {len(b_shape)}")
        for i, (ad, bd) in enumerate(zip(a_shape, b_shape)):
            if ad != bd:
                raise ValueError(f"stack shape mismatch at dim {i}: {a_shape} vs {b_shape}")
        if not isinstance(axis, int):
            raise TypeError("stack axis must be int")
        ndim = len(a_shape)
        if axis < -(ndim + 1) or axis > ndim:
            raise ValueError(f"stack axis {axis} out of range for ndim {ndim}")
        axis = axis % (ndim + 1)
        p = lib.py_stack(self._ptr, other._ptr, int(axis))
        if not p:
            raise ValueError("stack failed; ensure shapes are compatible")
        return Tensor(p)

    def slice(self, axis, start, end):
        if not lib.py_slice: raise RuntimeError('slice not available in C API')
        if not isinstance(axis, int):
            raise TypeError("slice axis must be int")
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("slice start/end must be int")
        ndim = lib.py_tensor_ndim(self._ptr)
        if axis < -ndim or axis >= ndim:
            raise ValueError(f"slice axis {axis} out of range for ndim {ndim}")
        axis = axis % ndim
        p = lib.py_slice(self._ptr, int(axis), int(start), int(end))
        if not p:
            raise ValueError("slice failed; check bounds")
        return Tensor(p)

    def select(self, axis, index):
        if not isinstance(index, int):
            raise TypeError("select index must be int")
        return self.slice(axis, index, index + 1).squeeze(axis)

    def split(self, axis=0, parts=2):
        """Simple split into equal parts along axis using numpy copy."""
        arr = self.to_numpy()
        splits = np.array_split(arr, parts, axis=axis)
        outs = []
        for s in splits:
            t = Tensor.new(list(s.shape), requires_grad=False)
            t.numpy_view()[:] = s
            outs.append(t)
        return outs

    def pad2d(self, pad_h, pad_w, value=0.0):
        if not lib.py_pad2d: raise RuntimeError('pad2d not available in C API')
        if len(self.shape()) != 4:
            raise ValueError("pad2d expects NCHW input")
        if pad_h < 0 or pad_w < 0:
            raise ValueError("pad2d padding must be non-negative")
        p = lib.py_pad2d(self._ptr, int(pad_h), int(pad_w), ctypes.c_float(value))
        return Tensor(p)

    def avgpool2d(self, kernel_size):
        if not lib.py_avgpool2d: raise RuntimeError('avgpool2d not available in C API')
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("avgpool2d kernel_size must be positive int")
        x_shape = self.shape()
        if len(x_shape) != 4:
            raise ValueError(f"avgpool2d expects input [N,C,H,W], got {x_shape}")
        if x_shape[2] % kernel_size != 0 or x_shape[3] % kernel_size != 0:
            raise ValueError(f"avgpool2d kernel_size {kernel_size} must divide H,W {x_shape[2:]}")
        p = lib.py_avgpool2d(self._ptr, ctypes.c_int(kernel_size))
        return Tensor(p)

    def has_nan_or_inf(self):
        if not lib.py_has_nan_inf: raise RuntimeError('has_nan_or_inf not available in C API')
        p = lib.py_has_nan_inf(self._ptr)
        return bool(Tensor(p).to_numpy().item())

    def sum(self):
        p = lib.py_sum(self._ptr)
        return Tensor(p)

    def dropout(self, p=0.5, training=True):
        if not lib.py_dropout: raise RuntimeError('dropout not available in C API')
        ptr = lib.py_dropout(self._ptr, ctypes.c_float(p), int(bool(training)))
        return Tensor(ptr)

    def requires_grad(self):
        if not lib.py_tensor_get_requires_grad: return False
        return bool(lib.py_tensor_get_requires_grad(self._ptr))

    def requires_grad_(self, flag=True):
        if not lib.py_tensor_set_requires_grad:
            raise RuntimeError('requires_grad setter not available in C API')
        lib.py_tensor_set_requires_grad(self._ptr, int(bool(flag)))
        if not flag:
            self.zero_grad()
        return self

    def grad_numpy(self):
        """Return a copy of the gradient buffer as a numpy array, or None if absent."""
        ndim = lib.py_tensor_ndim(self._ptr)
        shape = [lib.py_tensor_shape_get(self._ptr, i) for i in range(ndim)]
        size = 1
        for d in shape: size *= d
        data_ptr = lib.py_tensor_grad_ptr(self._ptr)
        if not data_ptr:
            return None
        arr = np.ctypeslib.as_array(data_ptr, shape=(size,))
        return arr.reshape(tuple(shape)).copy()

    def zero_grad(self):
        lib.py_zero_grad(self._ptr)

    def backward(self):
        lib.py_backward(self._ptr)

    def backward_(self, retain_graph=False):
        if lib.py_backward_with_retain:
            lib.py_backward_with_retain(self._ptr, int(bool(retain_graph)))
        else:
            lib.py_backward(self._ptr)


def _reduction_code(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == 'sum': return 0
        if r == 'mean': return 1
        if r == 'none': return 2
        raise ValueError('unknown reduction')
    return int(reduction)


def bce_loss(pred, target, logits=False, weight=None, reduction='mean'):
    if not lib.py_bce_loss: raise RuntimeError('bce_loss not available in C API')
    if weight is not None and not isinstance(weight, Tensor):
        raise TypeError('weight must be a Tensor or None')
    r = _reduction_code(reduction)
    wptr = weight._ptr if weight is not None else ctypes.c_void_p(0)
    p = lib.py_bce_loss(pred._ptr, target._ptr, wptr, int(bool(logits)), int(r))
    return Tensor(p)


def cross_entropy_logits(logits, target, weight=None, reduction='mean'):
    if not lib.py_cross_entropy_logits: raise RuntimeError('cross_entropy_logits not available in C API')
    if weight is not None and not isinstance(weight, Tensor):
        raise TypeError('weight must be a Tensor or None')
    if not isinstance(logits, Tensor) or not isinstance(target, Tensor):
        raise TypeError('cross_entropy_logits expects Tensor logits and target')
    log_shape = logits.shape()
    tgt_shape = target.shape()
    if len(log_shape) != 2:
        raise ValueError(f"cross_entropy_logits expects [N,C] logits, got {log_shape}")
    if len(tgt_shape) != 1 or tgt_shape[0] != log_shape[0]:
        raise ValueError(f"cross_entropy_logits expects target [N] matching logits batch, got {tgt_shape}")
    if weight is not None:
        w_shape = weight.shape()
        if len(w_shape) != 1 or w_shape[0] != log_shape[1]:
            raise ValueError(f"cross_entropy_logits weight must be shape [{log_shape[1]}], got {w_shape}")
    r = _reduction_code(reduction)
    wptr = weight._ptr if weight is not None else ctypes.c_void_p(0)
    p = lib.py_cross_entropy_logits(logits._ptr, target._ptr, wptr, int(r))
    return Tensor(p)


def sparse_categorical_crossentropy(logits, target, weight=None, reduction='mean'):
    return cross_entropy_logits(logits, target, weight=weight, reduction=reduction)


def categorical_crossentropy(logits, target, weight=None, reduction='mean'):
    return cross_entropy_logits(logits, target, weight=weight, reduction=reduction)


def relu(x):
    if not isinstance(x, Tensor):
        raise TypeError(f"relu expects Tensor, got {type(x)}")
    return x.relu()


def assert_finite(tensors, msg=None):
    """Raise ValueError if any tensor in the iterable contains NaN/Inf."""
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for t in tensors:
        if not isinstance(t, Tensor):
            raise TypeError('assert_finite expects Tensor or iterable of Tensors')
        if t.has_nan_or_inf():
            raise ValueError(msg or 'NaN or Inf detected in tensor')


# Experimental: vector-Jacobian product helper (prototype).
def vjp(f, x: Tensor, v: Tensor):
    if not _higher_order_enabled():
        raise RuntimeError("higher-order helpers are gated; set TINYFIN_ENABLE_HIGHER_ORDER=1 to enable vjp")
    if not isinstance(x, Tensor) or not isinstance(v, Tensor):
        raise TypeError("vjp expects Tensor inputs")
    y = f(x)
    if not isinstance(y, Tensor):
        raise TypeError("f(x) must return Tensor")
    if y.shape() != v.shape():
        raise ValueError(f"v shape {v.shape()} must match f(x) shape {y.shape()}")
    x.requires_grad_(True)
    x.zero_grad()
    dot = (y * v).sum()
    dot.backward_(retain_graph=True)
    return y, x.grad_numpy()


# Experimental: finite-difference JVP (approximation).
def jvp(f, x: Tensor, v: Tensor, eps: float = 1e-3):
    if not _higher_order_enabled():
        raise RuntimeError("higher-order helpers are gated; set TINYFIN_ENABLE_HIGHER_ORDER=1 to enable jvp")
    if not isinstance(x, Tensor) or not isinstance(v, Tensor):
        raise TypeError("jvp expects Tensor inputs")
    if x.shape() != v.shape():
        raise ValueError(f"jvp expects v shape {v.shape()} to match x shape {x.shape()}")
    y = f(x)
    if not isinstance(y, Tensor):
        raise TypeError("f(x) must return Tensor")
    # finite-difference along v
    x_eps = Tensor.new(x.shape(), requires_grad=False)
    x_eps.numpy_view()[:] = x.to_numpy() + eps * v.to_numpy()
    y_eps = f(x_eps)
    jvp_est = (y_eps.to_numpy() - y.to_numpy()) / eps
    return y, jvp_est


# Experimental: finite-difference Hessian-vector product using two grads.
def hvp(f, x: Tensor, v: Tensor, eps: float = 1e-3):
    if not _higher_order_enabled():
        raise RuntimeError("higher-order helpers are gated; set TINYFIN_ENABLE_HIGHER_ORDER=1 to enable hvp")
    if not isinstance(x, Tensor) or not isinstance(v, Tensor):
        raise TypeError("hvp expects Tensor inputs")
    x_np = x.to_numpy()
    v_np = v.to_numpy()
    if x_np.shape != v_np.shape:
        raise ValueError(f"hvp expects v shape {v_np.shape} to match x shape {x_np.shape}")

    def grad_at(arr):
        t = Tensor.new(list(arr.shape), requires_grad=True)
        t.numpy_view()[:] = arr
        y = f(t)
        if not isinstance(y, Tensor):
            raise TypeError("f(x) must return Tensor")
        s = y
        if len(y.shape()) != 0:
            s = y.sum()
        t.zero_grad()
        s.backward_()
        return y, t.grad_numpy()

    y_base, g_base = grad_at(x_np)
    _, g_shift = grad_at(x_np + eps * v_np)
    hvp_est = (g_shift - g_base) / eps
    return y_base, hvp_est


class SGDOpt:
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0, lr_params=None, weight_decay_params=None):
        if not lib.py_sgd_create:
            raise RuntimeError('sgd_create not available in C API')
        self.params = list(params)
        n = len(self.params)
        array_type = ctypes.c_void_p * n
        ptrs = array_type(*[p._ptr for p in self.params])
        p = lib.py_sgd_create(ptrs, n, ctypes.c_float(lr), ctypes.c_float(momentum), ctypes.c_float(weight_decay))
        self._ptr = ctypes.c_void_p(p)
        # per-parameter overrides
        if lr_params is not None:
            if not lib.py_sgd_set_param_lr: raise RuntimeError('sgd_set_param_lr not available in C API')
            if len(lr_params) != n: raise ValueError('lr_params length must match params')
            for i, v in enumerate(lr_params):
                lib.py_sgd_set_param_lr(self._ptr, ctypes.c_int(i), ctypes.c_float(v))
        if weight_decay_params is not None:
            if not lib.py_sgd_set_param_weight_decay: raise RuntimeError('sgd_set_param_weight_decay not available in C API')
            if len(weight_decay_params) != n: raise ValueError('weight_decay_params length must match params')
            for i, v in enumerate(weight_decay_params):
                lib.py_sgd_set_param_weight_decay(self._ptr, ctypes.c_int(i), ctypes.c_float(v))

    def step(self, clip_norm=0.0):
        if not lib.py_sgd_step: raise RuntimeError('sgd_step not available in C API')
        lib.py_sgd_step(self._ptr, ctypes.c_float(clip_norm))

    def zero_grad(self):
        if not lib.py_sgd_zero_grad: raise RuntimeError('sgd_zero_grad not available in C API')
        lib.py_sgd_zero_grad(self._ptr)

    def set_lr(self, lr):
        if not lib.py_sgd_set_lr: raise RuntimeError('sgd_set_lr not available in C API')
        lib.py_sgd_set_lr(self._ptr, ctypes.c_float(lr))

    def get_lr(self):
        if not lib.py_sgd_get_lr: raise RuntimeError('sgd_get_lr not available in C API')
        return float(lib.py_sgd_get_lr(self._ptr))

    def save_state(self, path: str):
        if not lib.py_sgd_save_state: raise RuntimeError('sgd_save_state not available in C API')
        ok = lib.py_sgd_save_state(self._ptr, path.encode())
        if not ok:
            raise RuntimeError('failed to save optimizer state')

    def load_state(self, path: str):
        if not lib.py_sgd_load_state: raise RuntimeError('sgd_load_state not available in C API')
        ok = lib.py_sgd_load_state(self._ptr, path.encode())
        if not ok:
            raise RuntimeError('failed to load optimizer state')

    def free(self):
        if lib.py_sgd_free:
            lib.py_sgd_free(self._ptr)


class AdamOpt:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        if not lib.py_adam_create:
            raise RuntimeError('adam_create not available in C API')
        self.params = list(params)
        n = len(self.params)
        array_type = ctypes.c_void_p * n
        ptrs = array_type(*[p._ptr for p in self.params])
        p = lib.py_adam_create(ptrs, n, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2), ctypes.c_float(eps), ctypes.c_float(weight_decay))
        self._ptr = ctypes.c_void_p(p)

    def step(self, clip_norm=0.0):
        if not lib.py_adam_step: raise RuntimeError('adam_step not available in C API')
        lib.py_adam_step(self._ptr, ctypes.c_float(clip_norm))

    def zero_grad(self):
        if not lib.py_adam_zero_grad: raise RuntimeError('adam_zero_grad not available in C API')
        lib.py_adam_zero_grad(self._ptr)

    def set_lr(self, lr):
        if not lib.py_adam_set_lr: raise RuntimeError('adam_set_lr not available in C API')
        lib.py_adam_set_lr(self._ptr, ctypes.c_float(lr))

    def get_lr(self):
        if not lib.py_adam_get_lr: raise RuntimeError('adam_get_lr not available in C API')
        return float(lib.py_adam_get_lr(self._ptr))

    def save_state(self, path: str):
        if not lib.py_adam_save_state: raise RuntimeError('adam_save_state not available in C API')
        ok = lib.py_adam_save_state(self._ptr, path.encode())
        if not ok:
            raise RuntimeError('failed to save optimizer state')

    def load_state(self, path: str):
        if not lib.py_adam_load_state: raise RuntimeError('adam_load_state not available in C API')
        ok = lib.py_adam_load_state(self._ptr, path.encode())
        if not ok:
            raise RuntimeError('failed to load optimizer state')

    def free(self):
        if lib.py_adam_free:
            lib.py_adam_free(self._ptr)


class RMSPropOpt:
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0):
        if not lib.py_rmsprop_create:
            raise RuntimeError('rmsprop_create not available in C API')
        self.params = list(params)
        n = len(self.params)
        array_type = ctypes.c_void_p * n
        ptrs = array_type(*[p._ptr for p in self.params])
        p = lib.py_rmsprop_create(ptrs, n, ctypes.c_float(lr), ctypes.c_float(alpha), ctypes.c_float(eps), ctypes.c_float(weight_decay))
        self._ptr = ctypes.c_void_p(p)

    def step(self, clip_norm=0.0):
        if not lib.py_rmsprop_step: raise RuntimeError('rmsprop_step not available in C API')
        lib.py_rmsprop_step(self._ptr, ctypes.c_float(clip_norm))

    def zero_grad(self):
        if not lib.py_rmsprop_zero_grad: raise RuntimeError('rmsprop_zero_grad not available in C API')
        lib.py_rmsprop_zero_grad(self._ptr)

    def set_lr(self, lr):
        if not lib.py_rmsprop_set_lr: raise RuntimeError('rmsprop_set_lr not available in C API')
        lib.py_rmsprop_set_lr(self._ptr, ctypes.c_float(lr))

    def get_lr(self):
        if not lib.py_rmsprop_get_lr: raise RuntimeError('rmsprop_get_lr not available in C API')
        return float(lib.py_rmsprop_get_lr(self._ptr))

    def save_state(self, path: str):
        if not lib.py_rmsprop_save_state: raise RuntimeError('rmsprop_save_state not available in C API')
        ok = lib.py_rmsprop_save_state(self._ptr, path.encode())
        if not ok:
            raise RuntimeError('failed to save optimizer state')

    def load_state(self, path: str):
        if not lib.py_rmsprop_load_state: raise RuntimeError('rmsprop_load_state not available in C API')
        ok = lib.py_rmsprop_load_state(self._ptr, path.encode())
        if not ok:
            raise RuntimeError('failed to load optimizer state')

    def free(self):
        if lib.py_rmsprop_free:
            lib.py_rmsprop_free(self._ptr)


def cprofiler_summary():
    if not lib.py_profiler_get_summary:
        return None
    p = lib.py_profiler_get_summary()
    if not p:
        return None
    s = ctypes.cast(p, ctypes.c_char_p).value.decode()
    # NOTE: do not free the returned pointer from Python — free on C-side if desired.
    # Calling back into C free here caused a crash on some platforms (double-free); leave
    # cleanup to the C side or a dedicated API if needed.
    return s


class no_grad:
    """Context manager to temporarily disable autograd recording.

    Supports nested usage by using push/pop helpers exposed from the C API.
    """
    def __enter__(self):
        # If push/pop helpers exist, use them to support nested contexts.
        if lib.py_autograd_push_disabled:
            self._prev = lib.py_autograd_push_disabled()
        elif lib.py_autograd_set_enabled:
            lib.py_autograd_set_enabled(0)
            self._prev = 1
        else:
            self._prev = None

    def __exit__(self, exc_type, exc, tb):
        if self._prev is None:
            return
        if lib.py_autograd_pop:
            lib.py_autograd_pop(self._prev)
        elif lib.py_autograd_set_enabled:
            lib.py_autograd_set_enabled(self._prev)


def export_graph(tensor, path: str):
    """Export autograd graph starting at tensor to a DOT file."""
    if not lib.py_autograd_to_dot:
        raise RuntimeError("autograd graph export not available")
    dot = lib.py_autograd_to_dot(tensor._ptr)
    if not dot:
        raise RuntimeError("failed to export graph")
    with open(path, "wb") as f:
        f.write(ctypes.cast(dot, ctypes.c_char_p).value)
    if lib.py_free_str:
        lib.py_free_str(dot)

def set_retain_graph_default(retain: bool):
    """Control whether Tensor.backward frees the graph by default."""
    if lib.py_autograd_set_retain_default:
        lib.py_autograd_set_retain_default(int(bool(retain)))

def get_retain_graph_default() -> bool:
    if lib.py_autograd_get_retain_default:
        return bool(lib.py_autograd_get_retain_default())
    return True


def load_shared_lib(path):
    return ctypes.CDLL(path)


def backend_name():
    """Return the current backend name (defaults to 'cpu')."""
    if not lib.py_backend_get_name:
        return "cpu"
    p = lib.py_backend_get_name()
    return p.decode() if isinstance(p, (bytes, bytearray)) else str(p)


def backend_set(name: str) -> bool:
    """Attempt to switch backend by name. Returns True on success."""
    if not lib.py_backend_set_by_name:
        return False
    res = lib.py_backend_set_by_name(name.encode())
    return bool(res)

# Mixed precision (stub): gate for future fp16/bfloat16 autocast per backend.
_mixed_precision_enabled = False

def set_mixed_precision(enabled: bool = True) -> bool:
    """Enable/disable mixed precision autocast (stub: currently no-op, future hook)."""
    global _mixed_precision_enabled
    _mixed_precision_enabled = bool(enabled)
    return _mixed_precision_enabled


class autocast:
    """Context manager for mixed precision (stub; currently no-op)."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        self._prev = _mixed_precision_enabled
        set_mixed_precision(self.enabled)
        return self

    def __exit__(self, exc_type, exc, tb):
        set_mixed_precision(self._prev)


class Profiler:
    """Simple wall-clock profiler for Python-level sections.

    Usage:
        with Profiler() as p:
            # code to profile
        print(p.summary())
    """
    def __init__(self):
        self.records = []

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def mark(self, name):
        now = time.perf_counter()
        self.records.append((name, now - self._start))

    def __exit__(self, exc_type, exc, tb):
        self._end = time.perf_counter()

    def summary(self):
        out = []
        for name, t in self.records:
            out.append(f"{name}: {t*1000:.3f} ms")
        out.append(f"total: {(getattr(self,'_end',time.perf_counter()) - getattr(self,'_start',0))*1000:.3f} ms")
        return "\n".join(out)

# Populate submodules so `import tinyfin.tensor/optim/utils/experimental` works even though
# the core is a module, not a package directory. We load from sibling files explicitly to
# avoid relying on package-relative imports when loaded as a single-file module.
import importlib.util
from pathlib import Path
_base_dir = Path(__file__).resolve().parent
for _name in ("tensor", "optim", "utils", "data", "training", "scheduler", "experimental"):
    _full = f"{__name__}.{_name}"
    if _full in sys.modules:
        setattr(sys.modules[__name__], _name, sys.modules[_full])
        continue
    _path = _base_dir / f"{_name}.py"
    if not _path.exists():
        continue
    spec = importlib.util.spec_from_file_location(_full, _path)
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(_mod)
            sys.modules[_full] = _mod
            setattr(sys.modules[__name__], _name, _mod)
        except Exception:
            pass

# Re-export commonly used symbols from submodules for convenience.
try:
    from .scheduler import StepLR, ExponentialLR, LinearWarmupLR  # type: ignore
except Exception:
    StepLR = None
    ExponentialLR = None
    LinearWarmupLR = None
