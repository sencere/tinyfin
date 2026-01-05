#ifndef CTORCH_BATCHLOADER_H
#define CTORCH_BATCHLOADER_H

#include "tensor.h"

typedef struct BatchLoader {
	Tensor *data; /* first dim is N */
	Tensor *labels; /* first dim is N or NULL */
	int N;
	int idx; /* current index */
	int *perm;
} BatchLoader;

BatchLoader *batchloader_create(Tensor *data, Tensor *labels);
void batchloader_shuffle(BatchLoader *b);
int batchloader_next(BatchLoader *b, int batch_size, Tensor **out_data, Tensor **out_labels);
void batchloader_free(BatchLoader *b);

#endif
