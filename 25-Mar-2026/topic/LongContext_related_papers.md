# Scaling Attention via Feature Sparsity 

**Authors**: Yan Xie, Tiansheng Wen, Tangda Huang, Bo Chen, Chenyu You, Stefanie Jegelka, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22300)  

**Abstract**: Scaling Transformers to ultra-long contexts is bottlenecked by the $O(n^2 d)$ cost of self-attention. Existing methods reduce this cost along the sequence axis through local windows, kernel approximations, or token-level sparsity, but these approaches consistently degrade accuracy. In this paper, we instead explore an orthogonal axis: feature sparsity. We propose Sparse Feature Attention (SFA), where queries and keys are represented as $k$-sparse codes that preserve high-dimensional expressivity while reducing the cost of attention from $\Theta(n^2 d)$ to $\Theta(n^2 k^2/d)$. To make this efficient at scale, we introduce FlashSFA, an IO-aware kernel that extends FlashAttention to operate directly on sparse overlaps without materializing dense score matrices. Across GPT-2 and Qwen3 pretraining, SFA matches dense baselines while improving speed by up to $2.5\times$ and reducing FLOPs and KV-cache by nearly 50\%. On synthetic and downstream benchmarks, SFA preserves retrieval accuracy and robustness at long contexts, outperforming short-embedding baselines that collapse feature diversity. These results establish feature-level sparsity as a complementary and underexplored axis for efficient attention, enabling Transformers to scale to orders-of-magnitude longer contexts with minimal quality loss. Code is available at this https URL. 

---
# MERIT: Memory-Enhanced Retrieval for Interpretable Knowledge Tracing 

**Authors**: Runze Li, Kedi Chen, Guwei Feng, Mo Yu, Jun Wang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22289)  

**Abstract**: Knowledge Tracing (KT) models students' evolving knowledge states to predict future performance, serving as a foundation for personalized education. While traditional deep learning models achieve high accuracy, they often lack interpretability. Large Language Models (LLMs) offer strong reasoning capabilities but struggle with limited context windows and hallucinations. Furthermore, existing LLM-based methods typically require expensive fine-tuning, limiting scalability and adaptability to new data. We propose MERIT (Memory-Enhanced Retrieval for Interpretable Knowledge Tracing), a training-free framework combining frozen LLM reasoning with structured pedagogical memory. Rather than updating parameters, MERIT transforms raw interaction logs into an interpretable memory bank. The framework uses semantic denoising to categorize students into latent cognitive schemas and constructs a paradigm bank where representative error patterns are analyzed offline to generate explicit Chain-of-Thought (CoT) rationales. During inference, a hierarchical routing mechanism retrieves relevant contexts, while a logic-augmented module applies semantic constraints to calibrate predictions. By grounding the LLM in interpretable memory, MERIT achieves state-of-the-art performance on real-world datasets without gradient updates. This approach reduces computational costs and supports dynamic knowledge updates, improving the accessibility and transparency of educational diagnosis. 

---
