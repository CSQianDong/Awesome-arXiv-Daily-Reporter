# An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit 

**Authors**: Warren Jouanneau, Emma Jouffroy, Marc Palyart  

**Link**: [PDF](https://arxiv.org/pdf/2601.10321)  

**Abstract**: Finding the most relevant person for a job proposal in real time is challenging, especially when resumes are long, structured, and multilingual. In this paper, we propose a re-ranking model based on a new generation of late cross-attention architecture, that decomposes both resumes and project briefs to efficiently handle long-context inputs with minimal computational overhead. To mitigate historical data biases, we use a generative large language model (LLM) as a teacher, generating fine-grained, semantically grounded supervision. This signal is distilled into our student model via an enriched distillation loss function. The resulting model produces skill-fit scores that enable consistent and interpretable person-job matching. Experiments on relevance, ranking, and calibration metrics demonstrate that our approach outperforms state-of-the-art baselines. 

---
# Continuum Memory Architectures for Long-Horizon LLM Agents 

**Authors**: Joe Logan  

**Link**: [PDF](https://arxiv.org/pdf/2601.09913)  

**Abstract**: Retrieval-augmented generation (RAG) has become the default strategy for providing large language model (LLM) agents with contextual knowledge. Yet RAG treats memory as a stateless lookup table: information persists indefinitely, retrieval is read-only, and temporal continuity is absent. We define the \textit{Continuum Memory Architecture} (CMA), a class of systems that maintain and update internal state across interactions through persistent storage, selective retention, associative routing, temporal chaining, and consolidation into higher-order abstractions. Rather than disclosing implementation specifics, we specify the architectural requirements CMA imposes and show consistent behavioral advantages on tasks that expose RAG's structural inability to accumulate, mutate, or disambiguate memory. The empirical probes (knowledge updates, temporal association, associative recall, contextual disambiguation) demonstrate that CMA is a necessary architectural primitive for long-horizon agents while highlighting open challenges around latency, drift, and interpretability. 

---
# CALM-IT: Generating Realistic Long-Form Motivational Interviewing Dialogues with Dual-Actor Conversational Dynamics Tracking 

**Authors**: Viet Cuong Nguyen, Nhi Yen Nguyen, Kristin A. Candan, Mary Conlon, Vanessa Rumie, Kristen Risola, Srijan Kumar, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2601.10085)  

**Abstract**: Large Language Models (LLMs) are increasingly used in mental health-related settings, yet they struggle to sustain realistic, goal-directed dialogue over extended interactions. While LLMs generate fluent responses, they optimize locally for the next turn rather than maintaining a coherent model of therapeutic progress, leading to brittleness and long-horizon drift. We introduce CALM-IT, a framework for generating and evaluating long-form Motivational Interviewing (MI) dialogues that explicitly models dual-actor conversational dynamics. CALM-IT represents therapist-client interaction as a bidirectional state-space process, in which both agents continuously update inferred alignment, mental states, and short-term goals to guide strategy selection and utterance generation. Across large-scale evaluations, CALM-IT consistently outperforms strong baselines in Effectiveness and Goal Alignment and remains substantially more stable as conversation length increases. Although CALM-IT initiates fewer therapist redirections, it achieves the highest client acceptance rate (64.3%), indicating more precise and therapeutically aligned intervention timing. Overall, CALM-IT provides evidence for modeling evolving conversational state being essential for generating high-quality long-form synthetic conversations. 

---
# Enhancing Business Analytics through Hybrid Summarization of Financial Reports 

**Authors**: Tohida Rehman  

**Link**: [PDF](https://arxiv.org/pdf/2601.09729)  

**Abstract**: Financial reports and earnings communications contain large volumes of structured and semi structured information, making detailed manual analysis inefficient. Earnings conference calls provide valuable evidence about a firm's performance, outlook, and strategic priorities. The manual analysis of lengthy call transcripts requires substantial effort and is susceptible to interpretive bias and unintentional error. In this work, we present a hybrid summarization framework that combines extractive and abstractive techniques to produce concise and factually reliable Reuters-style summaries from the ECTSum dataset. The proposed two stage pipeline first applies the LexRank algorithm to identify salient sentences, which are subsequently summarized using fine-tuned variants of BART and PEGASUS designed for resource constrained settings. In parallel, we fine-tune a Longformer Encoder-Decoder (LED) model to directly capture long-range contextual dependencies in financial documents.
Model performance is evaluated using standard automatic metrics, including ROUGE, METEOR, MoverScore, and BERTScore, along with domain-specific variants such as SciBERTScore and FinBERTScore. To assess factual accuracy, we further employ entity-level measures based on source-precision and F1-target. The results highlight complementary trade offs between approaches, long context models yield the strongest overall performance, while the hybrid framework achieves competitive results with improved factual consistency under computational constraints. These findings support the development of practical summarization systems for efficiently distilling lengthy financial texts into usable business insights. 

---
# SagaScale: A Realistic, Scalable, and High-Quality Long-Context Benchmark Built from Full-Length Novels 

**Authors**: Guancheng Du, Yong Hu, Wenqing Wang, Yaming Yang, Jiaheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.09723)  

**Abstract**: Large Language Models (LLMs) have shown significant progress, but understanding long and complex documents remains challenging. Many long-context benchmarks have been proposed, but they face several limitations, including task realism, data scalability, and data quality. To this end, we introduce SagaScale, a realistic, scalable, and high-quality long-context benchmark built from full-length novels. The entire benchmark is constructed using an automated data collection pipeline that utilizes external resources (e.g., Wikipedia pages) to curate question-answer pairs. Critically, these external resources are provided only for benchmark construction and not during evaluation, which allows LLMs to curate complex questions that go beyond what they can answer during evaluation. SagaScale is also bilingual and offers the largest context length to date, with average token counts exceeding 250K for English novels and 320K for Chinese novels. Our evaluation across 12 frontier LLMs and three long-context methods -- Naïve RAG, Agentic RAG, and Long Context -- yields key insights, including: (1) Directly supplying the full context to the LLM can outperform other methods by a large margin; (2) Most LLMs still struggle with lengthy contexts, but Gemini-2.5-Pro stands out as an exception; and (3) Agentic RAG effectively addresses the retrieval bottleneck in Naïve RAG. Finally, we publicly release the SagaScale benchmark and our data collection codebase to facilitate future research. 

---
# Evaluating Novelty in AI-Generated Research Plans Using Multi-Workflow LLM Pipelines 

**Authors**: Devesh Saraogi, Rohit Singhee, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.09714)  

**Abstract**: The integration of Large Language Models (LLMs) into the scientific ecosystem raises fundamental questions about the creativity and originality of AI-generated research. Recent work has identified ``smart plagiarism'' as a concern in single-step prompting approaches, where models reproduce existing ideas with terminological shifts. This paper investigates whether agentic workflows -- multi-step systems employing iterative reasoning, evolutionary search, and recursive decomposition -- can generate more novel and feasible research plans. We benchmark five reasoning architectures: Reflection-based iterative refinement, Sakana AI v2 evolutionary algorithms, Google Co-Scientist multi-agent framework, GPT Deep Research (GPT-5.1) recursive decomposition, and Gemini~3 Pro multimodal long-context pipeline. Using evaluations from thirty proposals each on novelty, feasibility, and impact, we find that decomposition-based and long-context workflows achieve mean novelty of 4.17/5, while reflection-based approaches score significantly lower (2.33/5). Results reveal varied performance across research domains, with high-performing workflows maintaining feasibility without sacrificing creativity. These findings support the view that carefully designed multi-stage agentic workflows can advance AI-assisted research ideation. 

---
# Thinking Long, but Short: Stable Sequential Test-Time Scaling for Large Reasoning Models 

**Authors**: Michael R. Metel, Yufei Cui, Boxing Chen, Prasanna Parthasarathi  

**Link**: [PDF](https://arxiv.org/pdf/2601.09855)  

**Abstract**: Sequential test-time scaling is a promising training-free method to improve large reasoning model accuracy, but as currently implemented, significant limitations have been observed. Inducing models to think for longer can increase their accuracy, but as the length of reasoning is further extended, it has also been shown to result in accuracy degradation and model instability. This work presents a novel sequential test-time scaling method, Min-Seek, which improves model accuracy significantly over a wide range of induced thoughts, stabilizing the accuracy of sequential scaling, and removing the need for reasoning length fine-tuning. Beyond improving model accuracy over a variety of reasoning tasks, our method is inherently efficient, as only the KV pairs of one additional induced thought are kept in the KV cache during reasoning. With a custom KV cache which stores keys without position embeddings, by dynamically encoding them contiguously before each new generated thought, our method can continue to reason well beyond a model's maximum context length, and under mild conditions has linear computational complexity. 

---
# Structure and Diversity Aware Context Bubble Construction for Enterprise Retrieval Augmented Systems 

**Authors**: Amir Khurshid, Abhishek Sehgal  

**Link**: [PDF](https://arxiv.org/pdf/2601.10681)  

**Abstract**: Large language model (LLM) contexts are typically constructed using retrieval-augmented generation (RAG), which involves ranking and selecting the top-k passages. The approach causes fragmentation in information graphs in document structures, over-retrieval, and duplication of content alongside insufficient query context, including 2nd and 3rd order facets. In this paper, a structure-informed and diversity-constrained context bubble construction framework is proposed that assembles coherent, citable bundles of spans under a strict token budget. The method preserves and exploits inherent document structure by organising multi-granular spans (e.g., sections and rows) and using task-conditioned structural priors to guide retrieval. Starting from high-relevance anchor spans, a context bubble is constructed through constrained selection that balances query relevance, marginal coverage, and redundancy penalties. It will explicitly constrain diversity and budget, producing compact and informative context sets, unlike top-k retrieval. Moreover, a full retrieval is emitted that traces the scoring and selection choices of the records, thus providing auditability and deterministic tuning. Experiments on enterprise documents demonstrate the efficiency of context bubble as it significantly reduces redundant context, is better able to cover secondary facets and has a better answer quality and citation faithfulness within a limited context window. Ablation studies demonstrate that both structural priors as well as diversity constraint selection are necessary; removing either component results in a decline in coverage and an increase in redundant or incomplete context. 

---
# Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering 

**Authors**: Xinyu Zhu, Yuzhu Cai, Zexi Liu, Bingyang Zheng, Cheng Wang, Rui Ye, Jiaao Chen, Hanrui Wang, Wei-Chen Wang, Yuzhi Zhang, Linfeng Zhang, Weinan E, Di Jin, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.10402)  

**Abstract**: The advancement of artificial intelligence toward agentic science is currently bottlenecked by the challenge of ultra-long-horizon autonomy, the ability to sustain strategic coherence and iterative correction over experimental cycles spanning days or weeks. While Large Language Models (LLMs) have demonstrated prowess in short-horizon reasoning, they are easily overwhelmed by execution details in the high-dimensional, delayed-feedback environments of real-world research, failing to consolidate sparse feedback into coherent long-term guidance. Here, we present ML-Master 2.0, an autonomous agent that masters ultra-long-horizon machine learning engineering (MLE) which is a representative microcosm of scientific discovery. By reframing context management as a process of cognitive accumulation, our approach introduces Hierarchical Cognitive Caching (HCC), a multi-tiered architecture inspired by computer systems that enables the structural differentiation of experience over time. By dynamically distilling transient execution traces into stable knowledge and cross-task wisdom, HCC allows agents to decouple immediate execution from long-term experimental strategy, effectively overcoming the scaling limits of static context windows. In evaluations on OpenAI's MLE-Bench under 24-hour budgets, ML-Master 2.0 achieves a state-of-the-art medal rate of 56.44%. Our findings demonstrate that ultra-long-horizon autonomy provides a scalable blueprint for AI capable of autonomous exploration beyond human-precedent complexities. 

---
# DecisionLLM: Large Language Models for Long Sequence Decision Exploration 

**Authors**: Xiaowei Lv, Zhilin Zhang, Yijun Li, Yusen Huo, Siyuan Ju, Xuyan Li, Chunxiang Hong, Tianyu Wang, Yongcai Wang, Peng Sun, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.10148)  

**Abstract**: Long-sequence decision-making, which is usually addressed through reinforcement learning (RL), is a critical component for optimizing strategic operations in dynamic environments, such as real-time bidding in computational advertising. The Decision Transformer (DT) introduced a powerful paradigm by framing RL as an autoregressive sequence modeling problem. Concurrently, Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning and planning tasks. This inspires us whether LLMs, which share the same Transformer foundation, but operate at a much larger scale, can unlock new levels of performance in long-horizon sequential decision-making problem. This work investigates the application of LLMs to offline decision making tasks. A fundamental challenge in this domain is the LLMs' inherent inability to interpret continuous values, as they lack a native understanding of numerical magnitude and order when values are represented as text strings. To address this, we propose treating trajectories as a distinct modality. By learning to align trajectory data with natural language task descriptions, our model can autoregressively predict future decisions within a cohesive framework we term DecisionLLM. We establish a set of scaling laws governing this paradigm, demonstrating that performance hinges on three factors: model scale, data volume, and data quality. In offline experimental benchmarks and bidding scenarios, DecisionLLM achieves strong performance. Specifically, DecisionLLM-3B outperforms the traditional Decision Transformer (DT) by 69.4 on Maze2D umaze-v1 and by 0.085 on AuctionNet. It extends the AIGB paradigm and points to promising directions for future exploration in online bidding. 

---
# Global Context Compression with Interleaved Vision-Text Transformation 

**Authors**: Dian Jiao, Jiaxin Duan, Shuai Zhao, Jiabing Leng, Yiran Zhang, Feng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10378)  

**Abstract**: Recent achievements of vision-language models in end-to-end OCR point to a new avenue for low-loss compression of textual information. This motivates earlier works that render the Transformer's input into images for prefilling, which effectively reduces the number of tokens through visual encoding, thereby alleviating the quadratically increased Attention computations. However, this partial compression fails to save computational or memory costs at token-by-token inference. In this paper, we investigate global context compression, which saves tokens at both prefilling and inference stages. Consequently, we propose VIST2, a novel Transformer that interleaves input text chunks alongside their visual encoding, while depending exclusively on visual tokens in the pre-context to predict the next text token distribution. Around this idea, we render text chunks into sketch images and train VIST2 in multiple stages, starting from curriculum-scheduled pretraining for optical language modeling, followed by modal-interleaved instruction tuning. We conduct extensive experiments using VIST2 families scaled from 0.6B to 8B to explore the training recipe and hyperparameters. With a 4$\times$ compression ratio, the resulting models demonstrate significant superiority over baselines on long writing tasks, achieving, on average, a 3$\times$ speedup in first-token generation, 77% reduction in memory usage, and 74% reduction in FLOPS. Our codes and datasets will be public to support further studies. 

---
