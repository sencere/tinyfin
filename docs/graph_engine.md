# tinyfin Graph Engine (Draft)

This document describes the forward-graph capture and fusion plan used to
support a "many ops + fuse what you can + graph everything else" approach.

## Goals
- Capture forward graphs for inference (training stays eager/autograd).
- Keep op set rich and explicit; fuse only safe, profitable patterns.
- Cache compiled plans by graph signature (shape/dtype/device).
- Fall back to eager if capture or fusion is unsupported.

## IR Schema (Forward Graph)
Each op produces a node:
- id: unique string
- op: string (e.g., "add", "mul", "matmul", "relu")
- inputs: list of node ids or input placeholders
- attrs: dict for op attributes (e.g., axis, kernel_size)
- shape: output shape
- dtype: output dtype
- device: output device
- const: bool (if node is constant)

Graph:
- inputs: list of input placeholders with shape/dtype/device
- nodes: topologically ordered list of nodes
- outputs: list of output node ids

## Capture
- A capture context records ops even under no_grad.
- Python Tensor op wrappers call a recorder hook when capture is active.
- Capture records only forward ops; autograd graph is unchanged.
- C capture uses explicit `graph_capture_begin/end` and records a subset of ops.

## Fusion Passes (MVP)
- Elementwise chains: fuse sequences of elementwise ops (add/mul/sub/div/exp/log/clamp/relu).
- Pattern fusion: matmul + add (bias) + relu.
- Fusions respect device/dtype boundaries and shape checks.

## Execution
- Compiler transforms captured graph into a plan of executable kernels.
- Cache key uses op list + shapes/dtypes/devices.
- On cache hit: run compiled plan.
- On miss: eager execution; if capture was active, compile and store.

## Limits (MVP)
- No training graph capture.
- No backward graph compilation.
- Fusion limited to safe, local patterns.
- C capture currently covers add/sub/mul/div/matmul/relu only.
