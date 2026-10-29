// SPDX-License-Identifier: Apache-2.0
// Schema from vllm#62958 head 19af173d3af78533964e380242eb849aef94d16a.
// Separate library: do not replace or re-register the base vLLM extension.
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "dspark_markov_walk(Tensor w2_res, Tensor w2_str, Tensor row_scale, "
      "Tensor w1_rows, Tensor base_logits, Tensor anchor_ids, "
      "int anchor_stride, Tensor idx_mapping, Tensor? temperature, "
      "Tensor seeds, Tensor pos, Tensor! draft_tokens, "
      "Tensor!? draft_logits, Tensor! state, int num_steps, int tiles, "
      "int stream_mask) -> ()");
}
