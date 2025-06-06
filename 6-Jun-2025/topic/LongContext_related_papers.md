# ECoRAG: Evidentiality-guided Compression for Long Context RAG 

**Authors**: Yeonseok Jeong, Jinsu Kim, Dohyeon Lee, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05167)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in Open-Domain Question Answering (ODQA) by leveraging external documents through Retrieval-Augmented Generation (RAG). To reduce RAG overhead, from longer context, context compression is necessary. However, prior compression methods do not focus on filtering out non-evidential information, which limit the performance in LLM-based RAG. We thus propose Evidentiality-guided RAG, or \textbf{ECoRAG} framework. ECoRAG improves LLM performance by compressing retrieved documents based on evidentiality, ensuring whether answer generation is supported by the correct evidence. As an additional step, ECoRAG reflects whether the compressed content provides sufficient evidence, and if not, retrieves more until sufficient. Experiments show that ECoRAG improves LLM performance on ODQA tasks, outperforming existing compression methods. Furthermore, ECoRAG is highly cost-efficient, as it not only reduces latency but also minimizes token usage by retaining only the necessary information to generate the correct answer. Code is available at this https URL. 

---
# Verbose ListOps (VLO): Beyond Long Context -- Unmasking LLM's Reasoning Blind Spots 

**Authors**: Alex Pan, Mary-Anne Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.04907)  

**Abstract**: Large Language Models (LLMs), whilst great at extracting facts from text, struggle with nested narrative reasoning. Existing long context and multi-hop QA benchmarks inadequately test this, lacking realistic distractors or failing to decouple context length from reasoning complexity, masking a fundamental LLM limitation. We introduce Verbose ListOps, a novel benchmark that programmatically transposes ListOps computations into lengthy, coherent stories. This uniquely forces internal computation and state management of nested reasoning problems by withholding intermediate results, and offers fine-grained controls for both narrative size \emph{and} reasoning difficulty. Whilst benchmarks like LongReason (2025) advance approaches for synthetically expanding the context size of multi-hop QA problems, Verbose ListOps pinpoints a specific LLM vulnerability: difficulty in state management for nested sub-reasoning amongst semantically-relevant, distracting narrative. Our experiments show that leading LLMs (e.g., OpenAI o4, Gemini 2.5 Pro) collapse in performance on Verbose ListOps at modest (~10k token) narrative lengths, despite effortlessly solving raw ListOps equations. Addressing this failure is paramount for real-world text interpretation which requires identifying key reasoning points, tracking conceptual intermediate results, and filtering irrelevant information. Verbose ListOps, and its extensible generation framework thus enables targeted reasoning enhancements beyond mere context-window expansion; a critical step to automating the world's knowledge work. 

---
# MesaNet: Sequence Modeling by Locally Optimal Test-Time Training 

**Authors**: Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximilian Schlegel, Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meulemans, Rif A. Saurous, Guillaume Lajoie, Charlotte Frenkel, Razvan Pascanu, Blaise Agüera y Arcas, João Sacramento  

**Link**: [PDF](https://arxiv.org/pdf/2506.05233)  

**Abstract**: Sequence modeling is currently dominated by causal transformer architectures that use softmax self-attention. Although widely adopted, transformers require scaling memory and compute linearly during inference. A recent stream of work linearized the softmax operation, resulting in powerful recurrent neural network (RNN) models with constant memory and compute costs such as DeltaNet, Mamba or xLSTM. These models can be unified by noting that their recurrent layer dynamics can all be derived from an in-context regression objective, approximately optimized through an online learning rule. Here, we join this line of work and introduce a numerically stable, chunkwise parallelizable version of the recently proposed Mesa layer (von Oswald et al., 2024), and study it in language modeling at the billion-parameter scale. This layer again stems from an in-context loss, but which is now minimized to optimality at every time point using a fast conjugate gradient solver. Through an extensive suite of experiments, we show that optimal test-time training enables reaching lower language modeling perplexity and higher downstream benchmark performance than previous RNNs, especially on tasks requiring long context understanding. This performance gain comes at the cost of additional flops spent during inference time. Our results are therefore intriguingly related to recent trends of increasing test-time compute to improve performance -- here by spending compute to solve sequential optimization problems within the neural network itself. 

---
# BEAR: BGP Event Analysis and Reporting 

**Authors**: Hanqing Li, Melania Fedeli, Vinay Kolar, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04514)  

**Abstract**: The Internet comprises of interconnected, independently managed Autonomous Systems (AS) that rely on the Border Gateway Protocol (BGP) for inter-domain routing. BGP anomalies--such as route leaks and hijacks--can divert traffic through unauthorized or inefficient paths, jeopardizing network reliability and security. Although existing rule-based and machine learning methods can detect these anomalies using structured metrics, they still require experts with in-depth BGP knowledge of, for example, AS relationships and historical incidents, to interpret events and propose remediation. In this paper, we introduce BEAR (BGP Event Analysis and Reporting), a novel framework that leverages large language models (LLMs) to automatically generate comprehensive reports explaining detected BGP anomaly events. BEAR employs a multi-step reasoning process that translates tabular BGP data into detailed textual narratives, enhancing interpretability and analytical precision. To address the limited availability of publicly documented BGP anomalies, we also present a synthetic data generation framework powered by LLMs. Evaluations on both real and synthetic datasets demonstrate that BEAR achieves 100% accuracy, outperforming Chain-of-Thought and in-context learning baselines. This work pioneers an automated approach for explaining BGP anomaly events, offering valuable operational insights for network management. 

---
# From EHRs to Patient Pathways: Scalable Modeling of Longitudinal Health Trajectories with LLMs 

**Authors**: Chantal Pellegrini, Ege Özsoy, David Bani-Harouni, Matthias Keicher, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2506.04831)  

**Abstract**: Healthcare systems face significant challenges in managing and interpreting vast, heterogeneous patient data for personalized care. Existing approaches often focus on narrow use cases with a limited feature space, overlooking the complex, longitudinal interactions needed for a holistic understanding of patient health. In this work, we propose a novel approach to patient pathway modeling by transforming diverse electronic health record (EHR) data into a structured representation and designing a holistic pathway prediction model, EHR2Path, optimized to predict future health trajectories. Further, we introduce a novel summary mechanism that embeds long-term temporal context into topic-specific summary tokens, improving performance over text-only models, while being much more token-efficient. EHR2Path demonstrates strong performance in both next time-step prediction and longitudinal simulation, outperforming competitive baselines. It enables detailed simulations of patient trajectories, inherently targeting diverse evaluation tasks, such as forecasting vital signs, lab test results, or length-of-stay, opening a path towards predictive and personalized healthcare. 

---
# Diagonal Batching Unlocks Parallelism in Recurrent Memory Transformers for Long Contexts 

**Authors**: Danil Sivtsov, Ivan Rodkin, Gleb Kuzmin, Yuri Kuratov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2506.05229)  

**Abstract**: Transformer models struggle with long-context inference due to their quadratic time and linear memory complexity. Recurrent Memory Transformers (RMTs) offer a solution by reducing the asymptotic cost to linear time and constant memory usage. However, their memory update mechanism leads to sequential execution, causing a performance bottleneck.
We introduce Diagonal Batching, a scheduling scheme that unlocks parallelism across segments in RMTs while preserving exact recurrence. This approach eliminates the sequential constraint, enabling efficient GPU inference even for single long-context inputs without complex batching and pipelining techniques. Because the technique is purely a run-time computation reordering, existing RMT models adopt it with no retraining.
Applied to a LLaMA-1B ARMT model, Diagonal Batching yields a 3.3x speedup over standard full-attention LLaMA-1B and a 1.8x speedup over the sequential RMT implementation on 131,072-token sequences. By removing sequential bottleneck, Diagonal Batching reduces inference cost and latency, thereby strengthening RMTs as a practical solution for real-world, long-context applications. 

---
# Unleashing Hour-Scale Video Training for Long Video-Language Understanding 

**Authors**: Jingyang Lin, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu, Yusheng Su, Xiaodong Yu, Hao Chen, Jiebo Luo, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.05332)  

**Abstract**: Recent long-form video-language understanding benchmarks have driven progress in video large multimodal models (Video-LMMs). However, the scarcity of well-annotated long videos has left the training of hour-long Video-LLMs underexplored. To close this gap, we present VideoMarathon, a large-scale hour-long video instruction-following dataset. This dataset includes around 9,700 hours of long videos sourced from diverse domains, ranging from 3 to 60 minutes per video. Specifically, it contains 3.3M high-quality QA pairs, spanning six fundamental topics: temporality, spatiality, object, action, scene, and event. Compared to existing video instruction datasets, VideoMarathon significantly extends training video durations up to 1 hour, and supports 22 diverse tasks requiring both short- and long-term video comprehension. Building on VideoMarathon, we propose Hour-LLaVA, a powerful and efficient Video-LMM for hour-scale video-language modeling. It enables hour-long video training and inference at 1-FPS sampling by leveraging a memory augmentation module, which adaptively integrates user question-relevant and spatiotemporal-informative semantics from a cached full video context. In our experiments, Hour-LLaVA achieves the best performance on multiple long video-language benchmarks, demonstrating the high quality of the VideoMarathon dataset and the superiority of the Hour-LLaVA model. 

---
