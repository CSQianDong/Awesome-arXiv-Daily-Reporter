# SPLA: Block Sparse Plus Linear Attention for Long Context Modeling 

**Authors**: Bailin Wang, Dan Friedman, Tao Lei, Chong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22379)  

**Abstract**: Block-wise sparse attention offers significant efficiency gains for long-context modeling, yet existing methods often suffer from low selection fidelity and cumulative contextual loss by completely discarding unselected blocks. To address these limitations, we introduce Sparse Plus Linear Attention (SPLA), a framework that utilizes a selection metric derived from second-order Taylor expansions to accurately identify relevant blocks for exact attention. Instead of discarding the remaining "long tail," SPLA compresses unselected blocks into a compact recurrent state via a residual linear attention (RLA) module. Crucially, to avoid IO overhead, we derive an optimized subtraction-based formulation for RLA -- calculating the residual as the difference between global and selected linear attention -- ensuring that unselected blocks are never explicitly accessed during inference. Our experiments demonstrate that SPLA closes the performance gap in continual pretraining, surpassing dense attention models on long-context benchmarks like RULER while maintaining competitive general knowledge and reasoning capabilities. 

---
# MrRoPE: Mixed-radix Rotary Position Embedding 

**Authors**: Qingyuan Tian, Wenhong Zhu, Xiaoran Liu, Xiaofeng Wang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22181)  

**Abstract**: Rotary Position Embedding (RoPE)-extension refers to modifying or generalizing the Rotary Position Embedding scheme to handle longer sequences than those encountered during pre-training. However, current extension strategies are highly diverse and lack a unified theoretical foundation. In this paper, we propose MrRoPE (Mixed-radix RoPE), a generalized encoding formulation based on a radix system conversion perspective, which elegantly unifies various RoPE-extension approaches as distinct radix conversion strategies. Based on this theory, we introduce two training-free extensions, MrRoPE-Uni and MrRoPE-Pro, which leverage uniform and progressive radix conversion strategies, respectively, to achieve 'train short, test long' generalization. Without fine-tuning, MrRoPE-Pro sustains over 85% recall in the 128K-context Needle-in-a-Haystack test and achieves more than double YaRN's accuracy on Infinite-Bench retrieval and dialogue subsets. Theoretical analysis confirms that MrRoPE-Pro effectively raises the upper bound of RoPE's attainable encoding length, which further validates the reliability and utility of our theory and methodology. 

---
