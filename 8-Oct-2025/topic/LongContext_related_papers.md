# Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents 

**Authors**: Wenda Xie, Chao Guo, Yanqing Jing. Junle Wang, Yisheng Lv, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05188)  

**Abstract**: Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts. 

---
# Revisiting Long-context Modeling from Context Denoising Perspective 

**Authors**: Zecheng Tang, Baibei Ji, Juntao Li, Lijun Wu, Haijia Gui, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05862)  

**Abstract**: Long-context models (LCMs) have demonstrated great potential in processing long sequences, facilitating many real-world applications. The success of LCMs can be attributed to their ability to locate implicit critical information within the context for further prediction. However, recent research reveals that LCMs are often susceptible to contextual noise, i.e., irrelevant tokens, that can mislead model attention. In this paper, we conduct a fine-grained analysis of the context noise and propose an effective metric, the Integrated Gradient (IG) score, to detect and quantify the noise information within the context. Our findings reveal that even simple mitigation of detected context noise can substantially boost the model's attention on critical tokens and benefit subsequent predictions. Building on this insight, we propose Context Denoising Training (CDT), a straightforward yet effective training strategy that improves attention on critical tokens while reinforcing their influence on model predictions. Extensive experiments across four tasks, under both context window scaling and long-context alignment settings, demonstrate the superiority of CDT. Notably, when trained with CDT, an open-source 8B model can achieve performance (50.92) comparable to GPT-4o (51.00). 

---
# Critical attention scaling in long-context transformers 

**Authors**: Shi Chen, Zhengjiang Lin, Yury Polyanskiy, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2510.05554)  

**Abstract**: As large language models scale to longer contexts, attention layers suffer from a fundamental pathology: attention scores collapse toward uniformity as context length $n$ increases, causing tokens to cluster excessively, a phenomenon known as rank-collapse. While $\textit{attention scaling}$ effectively addresses this deficiency by rescaling attention scores with a polylogarithmic factor $\beta_n$, theoretical justification for this approach remains lacking.
We analyze a simplified yet tractable model that magnifies the effect of attention scaling. In this model, attention exhibits a phase transition governed by the scaling factor $\beta_n$: insufficient scaling collapses all tokens to a single direction, while excessive scaling reduces attention to identity, thereby eliminating meaningful interactions between tokens. Our main result identifies the critical scaling $\beta_n \asymp \log n$ and provides a rigorous justification for attention scaling in YaRN and Qwen, clarifying why logarithmic scaling maintains sparse, content-adaptive attention at large context lengths. 

---
# CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension 

**Authors**: Rui Li, Zeyu Zhang, Xiaohe Bo, Zihang Tian, Xu Chen, Quanyu Dai, Zhenhua Dong, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05520)  

**Abstract**: Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification. 

---
# Context Length Alone Hurts LLM Performance Despite Perfect Retrieval 

**Authors**: Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05381)  

**Abstract**: Large language models (LLMs) often fail to scale their performance on long-context tasks performance in line with the context lengths they support. This gap is commonly attributed to retrieval failures -- the models' inability to identify relevant information in the long inputs. Accordingly, recent efforts often focus on evaluating and improving LLMs' retrieval performance: if retrieval is perfect, a model should, in principle, perform just as well on a long input as it does on a short one -- or should it? This paper presents findings that the answer to this question may be negative. Our systematic experiments across 5 open- and closed-source LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases but remains well within the models' claimed lengths. This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked and the models are forced to attend only to the relevant tokens. A similar performance drop is observed when all relevant evidence is placed immediately before the question. Our findings reveal a previously-unrealized limitation: the sheer length of the input alone can hurt LLM performance, independent of retrieval quality and without any distraction. They motivate our simple, model-agnostic mitigation strategy that transforms a long-context task into a short-context one by prompting the model to recite the retrieved evidence before attempting to solve the problem. On RULER, we observe a consistent improvement of GPT-4o up to 4% on an already strong baseline. 

---
# Evaluating the Sensitivity of LLMs to Harmful Contents in Long Input 

**Authors**: Faeze Ghorbanpour, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2510.05864)  

**Abstract**: Large language models (LLMs) increasingly support applications that rely on extended context, from document processing to retrieval-augmented generation. While their long-context capabilities are well studied for reasoning and retrieval, little is known about their behavior in safety-critical scenarios. We evaluate LLMs' sensitivity to harmful content under extended context, varying type (explicit vs. implicit), position (beginning, middle, end), prevalence (0.01-0.50 of the prompt), and context length (600-6000 tokens). Across harmful content categories such as toxic, offensive, and hate speech, with LLaMA-3, Qwen-2.5, and Mistral, we observe similar patterns: performance peaks at moderate harmful prevalence (0.25) but declines when content is very sparse or dominant; recall decreases with increasing context length; harmful sentences at the beginning are generally detected more reliably; and explicit content is more consistently recognized than implicit. These findings provide the first systematic view of how LLMs prioritize and calibrate harmful content in long contexts, highlighting both their emerging strengths and the challenges that remain for safety-critical use. 

---
# VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization 

**Authors**: Dingyu Yao, Chenxu Yang, Zhengyang Tong, Zheng Lin, Wei Liu, Jian Luan, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06175)  

**Abstract**: The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexible representational capacity across bit-widths, they suffer severe performance degradation at ultra-low bit-widths due to key cache outliers that hinder effective codebook utilization. To address this challenge, we propose VecInfer, a novel VQ method for aggressive KV cache compression while enabling efficient inference. By applying smooth and Hadamard transformations, VecInfer suppresses outliers in the key cache, enabling the codebook to comprehensively cover the original data distribution and thereby reducing quantization difficulty. To facilitate efficient deployment, we design an optimized CUDA kernel that fuses computation with dequantization to minimize memory access overhead. Extensive evaluations demonstrate that VecInfer consistently outperforms existing quantization baselines across both long-context understanding and mathematical reasoning tasks. With only 2-bit quantization, VecInfer achieves performance comparable to full precision, while delivering up to $\mathbf{2.7\times}$ speedup in large-batch self-attention computation and $\mathbf{8.3\times}$ reduction in single-batch end-to-end latency on Llama-3.1-8B with a 196k sequence length. 

---
# The End of Transformers? On Challenging Attention and the Rise of Sub-Quadratic Architectures 

**Authors**: Alexander M. Fichtl, Jeremias Bohn, Josefin Kelber, Edoardo Mosca, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2510.05364)  

**Abstract**: Transformers have dominated sequence processing tasks for the past seven years -- most notably language modeling. However, the inherent quadratic complexity of their attention mechanism remains a significant bottleneck as context length increases. This paper surveys recent efforts to overcome this bottleneck, including advances in (sub-quadratic) attention variants, recurrent neural networks, state space models, and hybrid architectures. We critically analyze these approaches in terms of compute and memory complexity, benchmark results, and fundamental limitations to assess whether the dominance of pure-attention transformers may soon be challenged. 

---
# Scalable In-context Ranking with Generative Models 

**Authors**: Nilesh Gupta, Chong You, Srinadh Bhojanapalli, Sanjiv Kumar, Inderjit Dhillon, Felix Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05396)  

**Abstract**: In-context Ranking (ICR) is an emerging paradigm for Information Retrieval (IR), which leverages contextual understanding of LLMs by directly incorporating the task description, candidate documents, and the query into the model's input prompt and tasking the LLM to identify relevant document(s). While it is effective, efficiency is a significant challenge in this paradigm, especially as the candidate list grows due to quadratic/super-linear scaling of attention operation with context length. To this end, this paper first identifies inherent and exploitable structures in the attention of LLMs finetuned for ICR: (1) inter-document block sparsity: attention is dense within each document block but sparse across different documents in the context; and (2) query-document block relevance: the attention scores from certain query tokens to a document block in middle layers strongly correlate with that document's actual relevance. Motivated by these observations, we introduce BlockRank (Blockwise In-context Ranking), a novel method that adapts the attention operation in an LLM by (a) architecturally enforcing the observed inter-document block sparsity, reducing attention complexity from quadratic to linear without loss in performance, and (b) optimizing query-document block relevance for true relevant documents during fine-tuning using an auxiliary contrastive training objective, improving retrieval in attention. Experiments on BEIR, MSMarco and NQ with Mistral-7B demonstrate that FLARE Mistral matches or outperforms existing SOTA listwise rankers and controlled fine-tuned baseline while being significantly more efficient at inference (4.7x for 100 MSMarco documents in context) and scaling gracefully to long-context shortlists, around 500 documents in-context (approximately 100K context length) within a second, presenting a scalable and effective solution for ICR. 

---
