# Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment 

**Title (ZH)**: 法官解码：更快的推测采样需要超越模型对齐 

**Authors**: Gregor Bachmann, Sotiris Anagnostidis, Albert Pumarola, Markos Georgopoulos, Artsiom Sanakoyeu, Yuming Du, Edgar Schönfeld, Ali Thabet, Jonas Kohler  

**Link**: [PDF](https://arxiv.org/pdf/2501.19309)  

**Abstract**: The performance of large language models (LLMs) is closely linked to their underlying size, leading to ever-growing networks and hence slower inference. Speculative decoding has been proposed as a technique to accelerate autoregressive generation, leveraging a fast draft model to propose candidate tokens, which are then verified in parallel based on their likelihood under the target model. While this approach guarantees to reproduce the target output, it incurs a substantial penalty: many high-quality draft tokens are rejected, even when they represent objectively valid continuations. Indeed, we show that even powerful draft models such as GPT-4o, as well as human text cannot achieve high acceptance rates under the standard verification scheme. This severely limits the speedup potential of current speculative decoding methods, as an early rejection becomes overwhelmingly likely when solely relying on alignment of draft and target.
We thus ask the following question: Can we adapt verification to recognize correct, but non-aligned replies? To this end, we draw inspiration from the LLM-as-a-judge framework, which demonstrated that LLMs are able to rate answers in a versatile way. We carefully design a dataset to elicit the same capability in the target model by training a compact module on top of the embeddings to produce ``judgements" of the current continuation. We showcase our strategy on the Llama-3.1 family, where our 8b/405B-Judge achieves a speedup of 9x over Llama-405B, while maintaining its quality on a large range of benchmarks. These benefits remain present even in optimized inference frameworks, where our method reaches up to 141 tokens/s for 8B/70B-Judge and 129 tokens/s for 8B/405B on 2 and 8 H100s respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）的性能与其基础规模密切相关，导致了网络规模的不断增长，从而引发了更快的推理速度。推测性解码已被提出作为一种技术，用于加速自回归生成，通过使用快速草稿模型提出候选token，然后在并行验证下基于其在目标模型下的可能性进行验证。尽管这种方法保证能够复制目标输出，但也导致了显著的代价：许多高质量的草稿token被拒绝，即使它们代表了客观上有效的扩展。事实上，我们展示了即使像GPT-4o这样的强大草稿模型，甚至人类文本，在标准验证方案下也无法实现高接受率。这严重限制了当前推测性解码方法的加速潜力，仅依赖草稿和目标的对齐会导致过早拒绝变得极为常见。

因此，我们提出以下问题：我们能否调整验证过程，以识别正确但未对齐的回答？为此，我们借鉴LLM作为裁判的框架，该框架证明了LLM能够以灵活的方式评估答案。我们精心设计了一个数据集，通过在嵌入层上训练一个紧凑模块来产生当前扩展的“判决”，以使目标模型具备同样的能力。我们在Llama-3.1家族中展示了我们的策略，我们的8b/405B-裁判模型相对于Llama-405B实现了9倍的加速，同时在大量benchmark上保持了质量。即使在优化推理框架中，我们的方法也能达到141 token/s（使用2个H100）和129 token/s（使用8个H100），对于8b/70B-裁判和8b/405B-裁判模型。 

---
