/* Softmax shim: canonical stable softmax + CE implemented in
   src/ops_softmax_ce.c. This source file remains as a minimal shim
   to keep historical build lists intact. Include the public softmax
   header so this unit compiles without defining duplicate symbols. */
#include "ops_softmax.h"
