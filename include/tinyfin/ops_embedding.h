#ifndef CTORCH_OPS_EMBEDDING_H
#define CTORCH_OPS_EMBEDDING_H

#include "tensor.h"
#include "autograd.h"

/* weights: [num_embeddings, emb_dim]
 * indices: arbitrary shape containing integer indices (stored as float values)
 * returns: output tensor with shape = indices.shape + [emb_dim]
 */
Tensor *tensor_embedding(Tensor *weights, Tensor *indices);

#endif
