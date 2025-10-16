# BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning 

**Authors**: Jia-Chen Gu, Junyi Zhang, Di Wu, Yuankai Li, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.13799)  

**Abstract**: As retrieval-augmented generation (RAG) tackles complex tasks, increasingly expanded contexts offer richer information, but at the cost of higher latency and increased cognitive load on the model. To mitigate this bottleneck, especially for intricate multi-hop questions, we introduce BRIEF-Pro. It is a universal, lightweight compressor that distills relevant evidence for a given query from retrieved documents into a concise summary for seamless integration into in-context RAG. Using seed data consisting of relatively short contexts (fewer than 1k words), BRIEF-Pro is trained to perform abstractive compression of extended contexts exceeding 10k words across a wide range of scenarios. Furthermore, BRIEF-Pro offers flexible user control over summary length by allowing users to specify the desired number of sentences. Experiments on four open-domain multi-hop question-answering datasets show that BRIEF-Pro generates more concise and relevant summaries, enhancing performance across small, large, and proprietary language models. With the 70B reader model, 32x compression by BRIEF-Pro improves QA performance by 4.67% on average over LongLLMLingua's 9x, while requiring only 23% of its computational overhead. 

---
# Breadcrumbs Reasoning: Memory-Efficient Reasoning with Compression Beacons 

**Authors**: Giovanni Monea, Yair Feldman, Shankar Padmanabhan, Kianté Brantley, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13797)  

**Abstract**: The scalability of large language models for long-context reasoning is severely constrained by the linear growth of their Transformer key-value cache, which incurs significant memory and computational costs. We posit that as a model generates reasoning tokens, the informational value of past generated tokens diminishes, creating an opportunity for compression. In this work, we propose to periodically compress the generation KV cache with a learned, special-purpose token and evict compressed entries. We train the model to perform this compression via a modified joint distillation and reinforcement learning (RL) framework. Our training method minimizes overhead over the conventional RL process, as it leverages RL outputs for distillation. Empirically, our method achieves a superior memory-accuracy Pareto frontier compared to both the model without cache compression and training-free compression techniques. 

---
# The Mechanistic Emergence of Symbol Grounding in Language Models 

**Authors**: Shuyu Wu, Ziqiao Ma, Xiaoxi Luo, Yidong Huang, Josue Torres-Fonseca, Freda Shi, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2510.13796)  

**Abstract**: Symbol grounding (Harnad, 1990) describes how symbols such as words acquire their meanings by connecting to real-world sensorimotor experiences. Recent work has shown preliminary evidence that grounding may emerge in (vision-)language models trained at scale without using explicit grounding objectives. Yet, the specific loci of this emergence and the mechanisms that drive it remain largely unexplored. To address this problem, we introduce a controlled evaluation framework that systematically traces how symbol grounding arises within the internal computations through mechanistic and causal analysis. Our findings show that grounding concentrates in middle-layer computations and is implemented through the aggregate mechanism, where attention heads aggregate the environmental ground to support the prediction of linguistic forms. This phenomenon replicates in multimodal dialogue and across architectures (Transformers and state-space models), but not in unidirectional LSTMs. Our results provide behavioral and mechanistic evidence that symbol grounding can emerge in language models, with practical implications for predicting and potentially controlling the reliability of generation. 

---
# Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation 

**Authors**: Zhiqi Huang, Vivek Datla, Chenyang Zhu, Alfy Samuel, Daben Liu, Anoop Kumar, Ritesh Soni  

**Link**: [PDF](https://arxiv.org/pdf/2510.13750)  

**Abstract**: We propose a method for confidence estimation in retrieval-augmented generation (RAG) systems that aligns closely with the correctness of large language model (LLM) outputs. Confidence estimation is especially critical in high-stakes domains such as finance and healthcare, where the cost of an incorrect answer outweighs that of not answering the question. Our approach extends prior uncertainty quantification methods by leveraging raw feed-forward network (FFN) activations as auto-regressive signals, avoiding the information loss inherent in token logits and probabilities after projection and softmax normalization. We model confidence prediction as a sequence classification task, and regularize training with a Huber loss term to improve robustness against noisy supervision. Applied in a real-world financial industry customer-support setting with complex knowledge bases, our method outperforms strong baselines and maintains high accuracy under strict latency constraints. Experiments on Llama 3.1 8B model show that using activations from only the 16th layer preserves accuracy while reducing response latency. Our results demonstrate that activation-based confidence modeling offers a scalable, architecture-aware path toward trustworthy RAG deployment. 

---
# Assessing Web Search Credibility and Response Groundedness in Chat Assistants 

**Authors**: Ivan Vykopal, Matúš Pikuliak, Simon Ostermann, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2510.13749)  

**Abstract**: Chat assistants increasingly integrate web search functionality, enabling them to retrieve and cite external sources. While this promises more reliable answers, it also raises the risk of amplifying misinformation from low-credibility sources. In this paper, we introduce a novel methodology for evaluating assistants' web search behavior, focusing on source credibility and the groundedness of responses with respect to cited sources. Using 100 claims across five misinformation-prone topics, we assess GPT-4o, GPT-5, Perplexity, and Qwen Chat. Our findings reveal differences between the assistants, with Perplexity achieving the highest source credibility, whereas GPT-4o exhibits elevated citation of non-credibility sources on sensitive topics. This work provides the first systematic comparison of commonly used chat assistants for fact-checking behavior, offering a foundation for evaluating AI systems in high-stakes information environments. 

---
# GAPS: A Clinically Grounded, Automated Benchmark for Evaluating AI Clinicians 

**Authors**: Xiuyuan Chen, Tao Sun, Dexin Su, Ailing Yu, Junwei Liu, Zhe Chen, Gangzeng Jin, Xin Wang, Jingnan Liu, Hansong Xiao, Hualei Zhou, Dongjie Tao, Chunxiao Guo, Minghui Yang, Yuan Xia, Jing Zhao, Qianrui Fan, Yanyun Wang, Shuai Zhen, Kezhong Chen, Jun Wang, Zewen Sun, Heng Zhao, Tian Guan, Shaodong Wang, Geyun Chang, Jiaming Deng, Hongchengcheng Chen, Kexin Feng, Ruzhen Li, Jiayi Geng, Changtai Zhao, Jun Wang, Guihu Lin, Peihao Li, Liqi Liu, Peng Wei, Jian Wang, Jinjie Gu, Ping Wang, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13734)  

**Abstract**: Current benchmarks for AI clinician systems, often based on multiple-choice exams or manual rubrics, fail to capture the depth, robustness, and safety required for real-world clinical practice. To address this, we introduce the GAPS framework, a multidimensional paradigm for evaluating \textbf{G}rounding (cognitive depth), \textbf{A}dequacy (answer completeness), \textbf{P}erturbation (robustness), and \textbf{S}afety. Critically, we developed a fully automated, guideline-anchored pipeline to construct a GAPS-aligned benchmark end-to-end, overcoming the scalability and subjectivity limitations of prior work. Our pipeline assembles an evidence neighborhood, creates dual graph and tree representations, and automatically generates questions across G-levels. Rubrics are synthesized by a DeepResearch agent that mimics GRADE-consistent, PICO-driven evidence review in a ReAct loop. Scoring is performed by an ensemble of large language model (LLM) judges. Validation confirmed our automated questions are high-quality and align with clinician judgment. Evaluating state-of-the-art models on the benchmark revealed key failure modes: performance degrades sharply with increased reasoning depth (G-axis), models struggle with answer completeness (A-axis), and they are highly vulnerable to adversarial perturbations (P-axis) as well as certain safety issues (S-axis). This automated, clinically-grounded approach provides a reproducible and scalable method for rigorously evaluating AI clinician systems and guiding their development toward safer, more reliable clinical practice. 

---
# NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching 

**Authors**: Run Luo, Xiaobo Xia, Lu Wang, Longze Chen, Renke Shan, Jing Luo, Min Yang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2510.13721)  

**Abstract**: Next-generation multimodal foundation models capable of any-to-any cross-modal generation and multi-turn interaction will serve as core components of artificial general intelligence systems, playing a pivotal role in human-machine interaction. However, most existing multimodal models remain constrained by autoregressive architectures, whose inherent limitations prevent a balanced integration of understanding and generation capabilities. Although hybrid and decoupling strategies have been explored to address these tasks within unified frameworks separately, their redundant, non-integrated designs limit their applicability to broader scenarios, such as cross-modal this http URL this work, we introduce NExT-OMNI, an open-source omnimodal foundation model that achieves unified modeling through discrete flow paradigms. By leveraging metric-induced probability paths and kinetic optimal velocities, NExT-OMNI natively supports any-to-any understanding and generation with enhanced response efficiency, while enabling broader application scenarios through concise unified representations rather than task-decoupled designs. Trained on large-scale interleaved text, image, video, and audio data, NExT-OMNI delivers competitive performance on multimodal generation and understanding benchmarks, while outperforming prior unified models in multi-turn multimodal interaction and cross-modal retrieval, highlighting its architectural advantages as a next-generation multimodal foundation model. To advance further research, we release training details, data protocols, and open-source both the code and model checkpoints. 

---
# How Sampling Affects the Detectability of Machine-written texts: A Comprehensive Study 

**Authors**: Matthieu Dubois, François Yvon, Pablo Piantanida  

**Link**: [PDF](https://arxiv.org/pdf/2510.13681)  

**Abstract**: As texts generated by Large Language Models (LLMs) are ever more common and often indistinguishable from human-written content, research on automatic text detection has attracted growing attention. Many recent detectors report near-perfect accuracy, often boasting AUROC scores above 99\%. However, these claims typically assume fixed generation settings, leaving open the question of how robust such systems are to changes in decoding strategies. In this work, we systematically examine how sampling-based decoding impacts detectability, with a focus on how subtle variations in a model's (sub)word-level distribution affect detection performance. We find that even minor adjustments to decoding parameters - such as temperature, top-p, or nucleus sampling - can severely impair detector accuracy, with AUROC dropping from near-perfect levels to 1\% in some settings. Our findings expose critical blind spots in current detection methods and emphasize the need for more comprehensive evaluation protocols. To facilitate future research, we release a large-scale dataset encompassing 37 decoding configurations, along with our code and evaluation framework this https URL 

---
# Closing the Gap Between Text and Speech Understanding in LLMs 

**Authors**: Santiago Cuervo, Skyler Seto, Maureen de Seyssel, Richard He Bai, Zijin Gu, Tatiana Likhomanenko, Navdeep Jaitly, Zakaria Aldeneh  

**Link**: [PDF](https://arxiv.org/pdf/2510.13632)  

**Abstract**: Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora. 

---
# Unlocking Public Catalogues: Instruction-Tuning LLMs for ICD Coding of German Tumor Diagnoses 

**Authors**: Stefan Lenz, Lakisha Ortiz Rosario, Georg Vollmar, Arsenij Ustjanzew, Fatma Alickovic, Thomas Kindler, Torsten Panholzer  

**Link**: [PDF](https://arxiv.org/pdf/2510.13624)  

**Abstract**: Accurate coding of tumor diagnoses with ICD-10-GM and ICD-O-3 is essential for structured cancer documentation in Germany. Smaller open-weight LLMs are appealing for privacy-preserving automation but often struggle with coding accuracy in German-language contexts. This study investigates whether instruction-based fine-tuning on public datasets improves the coding accuracy of open-weight LLMs for German tumor diagnosis texts. The evaluation uses coded diagnoses from the local tumor documentation system as test data. In a systematic data quality assessment, the upper limit for ICD-10 coding performance was estimated at 60-79% for exact and 81-94% for partial (three-character codes only) derivation. As training data, over 500,000 question-answer pairs were created based on the ICD-10-GM, ICD-O-3, and OPS catalogues. Eight open-weight models from the Qwen, Llama, and Mistral families (7-70 B parameters) were fine-tuned. ICD-10-GM accuracy rose from 1.4-24% to 41-58%, and partial accuracy from 31-74% to 73-83%. The accuracy of ICD-O-3 topography coding also improved but started and remained considerably lower with an exact accuracy of 22-40% and a partial accuracy of 56-67% after fine-tuning. Malformed code outputs dropped to 0% for all models. Tumor-diagnosis recognition reached 99%. Accuracy correlated positively with model size, but gaps between small and large models narrowed after fine-tuning. The reasoning mode in Qwen3 generally yielded a lower performance than fine-tuning and was over 100 times slower. Our findings highlight the potential of leveraging public catalogues to build instruction datasets that improve LLMs in medical documentation tasks. The complete training dataset and the best-performing checkpoints of the fine-tuned models are available from this https URL. 

---
# MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning 

**Authors**: Xingyu Tan, Xiaoyang Wang, Xiwei Xu, Xin Yuan, Liming Zhu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13614)  

**Abstract**: Large Language Models (LLMs) have achieved impressive reasoning abilities, but struggle with temporal understanding, especially when questions involve multiple entities, compound operators, and evolving event sequences. Temporal Knowledge Graphs (TKGs), which capture vast amounts of temporal facts in a structured format, offer a reliable source for temporal reasoning. However, existing TKG-based LLM reasoning methods still struggle with four major challenges: maintaining temporal faithfulness in multi-hop reasoning, achieving multi-entity temporal synchronization, adapting retrieval to diverse temporal operators, and reusing prior reasoning experience for stability and efficiency. To address these issues, we propose MemoTime, a memory-augmented temporal knowledge graph framework that enhances LLM reasoning through structured grounding, recursive reasoning, and continual experience learning. MemoTime decomposes complex temporal questions into a hierarchical Tree of Time, enabling operator-aware reasoning that enforces monotonic timestamps and co-constrains multiple entities under unified temporal bounds. A dynamic evidence retrieval layer adaptively selects operator-specific retrieval strategies, while a self-evolving experience memory stores verified reasoning traces, toolkit decisions, and sub-question embeddings for cross-type reuse. Comprehensive experiments on multiple temporal QA benchmarks show that MemoTime achieves overall state-of-the-art results, outperforming the strong baseline by up to 24.0%. Furthermore, MemoTime enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to that of GPT-4-Turbo. 

---
# NOSA: Native and Offloadable Sparse Attention 

**Authors**: Yuxiang Huang, Chaojun Xiao, Xu Han, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13602)  

**Abstract**: Trainable sparse attention has emerged as a promising solution to address the decoding efficiency bottleneck of LLMs in long-context processing, significantly saving memory accesses while minimally impacting task performance. However, existing sparse attention methods leave a crucial limitation unresolved: the size of the key-value (KV) cache remains unreduced, which constrains on-GPU batch sizes and throttles decoding throughput, especially in large-scale batched inference. In this paper, we show that trainable sparse attention naturally exhibits strong locality in token selection across adjacent decoding steps, thereby enabling KV cache offloading without altering the underlying attention computation. However, the inherent locality remains insufficient to achieve efficient offloading, as the transfer of selected KV pairs between the CPU and GPU continues to dominate the overall decoding cost. Building on this insight, we present NOSA, a trainable sparse attention framework designed to natively support KV cache offloading. NOSA introduces explicit locality constraints by decomposing token selection into query-aware and query-agnostic components, thereby reducing KV transfers while preserving the same attention computation as used during training. We pretrain a 1B-parameter model with NOSA and conduct extensive benchmarks, showing that it preserves near-lossless performance while achieving up to a 2.3x improvement in decoding throughput compared with the vanilla trainable sparse attention baseline (InfLLM-V2). 

---
# FreshTab: Sourcing Fresh Data for Table-to-Text Generation Evaluation 

**Authors**: Kristýna Onderková, Ondřej Plátek, Zdeněk Kasner, Ondřej Dušek  

**Link**: [PDF](https://arxiv.org/pdf/2510.13598)  

**Abstract**: Table-to-text generation (insight generation from tables) is a challenging task that requires precision in analyzing the data. In addition, the evaluation of existing benchmarks is affected by contamination of Large Language Model (LLM) training data as well as domain imbalance. We introduce FreshTab, an on-the-fly table-to-text benchmark generation from Wikipedia, to combat the LLM data contamination problem and enable domain-sensitive evaluation. While non-English table-to-text datasets are limited, FreshTab collects datasets in different languages on demand (we experiment with German, Russian and French in addition to English). We find that insights generated by LLMs from recent tables collected by our method appear clearly worse by automatic metrics, but this does not translate into LLM and human evaluations. Domain effects are visible in all evaluations, showing that a~domain-balanced benchmark is more challenging. 

---
# Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs 

**Authors**: Pasin Buakhaw, Kun Kerdthaisong, Phuree Phenhiran, Pitikorn Khlaisamniang, Supasate Vorathammathorn, Piyalitt Ittichaiwong, Nutchanon Yongsatianchot  

**Link**: [PDF](https://arxiv.org/pdf/2510.13586)  

**Abstract**: The emergence of large language models (LLMs) has opened new opportunities for cre- ating dynamic non-player characters (NPCs) in gaming environments, enabling both func- tional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which eval- uates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track). 

---
# Sparse Subnetwork Enhancement for Underrepresented Languages in Large Language Models 

**Authors**: Daniil Gurgurov, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2510.13580)  

**Abstract**: Large language models exhibit uneven performance across languages, with substantial gaps between high- and low-resource languages. We present a framework for enhancing monolingual capabilities of LLMs in underrepresented languages while preserving their general-purpose performance through targeted fine-tuning of language-specific subnetworks. Our approach identifies language-specific neurons using Language Activation Probability Entropy and fine-tunes only the weights associated with these neurons, a dedicated subnetwork, on target-language data. Experiments on Llama-3.1-8B and Mistral-Nemo-12B across 12 mid- and low-resource languages demonstrate that our method consistently outperforms full fine-tuning, FFN-only fine-tuning, LoRA adaptation, and random subset fine-tuning baselines while efficiently updating only up to 1% of model parameters. Beyond performance improvements, we observe enhanced favorable training dynamics, cross-lingual representational alignment, and systematic weight update changes. To facilitate future research, we release language-specific neuron identifications for over 100 languages as well as our adaptation pipeline, offering a cost-effective pathway for adapting state-of-the-art models to underrepresented languages. 

---
# Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy Optimization 

**Authors**: Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, Bo Zheng, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13554)  

**Abstract**: The reasoning pattern of Large language models (LLMs) remains opaque, and Reinforcement learning (RL) typically applies uniform credit across an entire generation, blurring the distinction between pivotal and routine steps. This work positions attention as a privileged substrate that renders the internal logic of LLMs legible, not merely as a byproduct of computation, but as a mechanistic blueprint of reasoning itself. We first distinguish attention heads between locally and globally focused information processing and reveal that locally focused heads produce a sawtooth pattern near the diagonal indicating phrasal chunks, while globally focused heads expose tokens that exert broad downstream influence over future tokens. We formalize these with two metrics: 1) Windowed Average Attention Distance, which measures the extent of backward attention within a clipped window; 2) Future Attention Influence, which quantifies a token's global importance as the average attention it receives from subsequent tokens. Taken together, these signals reveal a recurring preplan-and-anchor mechanism, where the model first performs a long-range contextual reference to generate an introductory token, which is immediately followed by or coincides with a semantic anchor token that organizes subsequent reasoning. Leveraging these insights, we introduce three novel RL strategies that dynamically perform targeted credit assignment to critical nodes (preplan tokens, anchor tokens, and their temporal coupling) and show consistent performance gains across various reasoning tasks. By aligning optimization with the model's intrinsic reasoning rhythm, we aim to transform opaque optimization into an actionable structure-aware process, hoping to offer a potential step toward more transparent and effective optimization of LLM reasoning. 

---
# MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts 

**Authors**: Shujun Xia, Haokun Lin, Yichen Wu, Yinan Zhou, Zixuan Li, Zhongwei Wan, Xingrun Xing, Yefeng Zheng, Xiang Li, Caifeng Shan, Zhenan Sun, Quanzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13500)  

**Abstract**: LLMs hold great promise for healthcare applications, but the rapid evolution of medical knowledge and errors in training data often cause them to generate outdated or inaccurate information, limiting their applicability in high-stakes clinical practice. Model editing has emerged as a potential remedy without full retraining. While parameter-based editing often compromises locality and is thus ill-suited for the medical domain, retrieval-based editing offers a more viable alternative. However, it still faces two critical challenges: (1) representation overlap within the medical knowledge space often causes inaccurate retrieval and reduces editing accuracy; (2) existing methods are restricted to single-sample edits, while batch-editing remains largely unexplored despite its importance for real-world medical applications. To address these challenges, we first construct MedVersa, \hk{an enhanced benchmark with broader coverage of medical subjects, designed to evaluate both single and batch edits under strict locality constraints}. We then propose MedREK, a retrieval-based editing framework that integrates a shared query-key module for precise matching with an attention-based prompt encoder for informative guidance. Experimental results on various medical benchmarks demonstrate that our MedREK achieves superior performance across different core metrics and provides the first validated solution for batch-editing in medical LLMs. Our code and dataset are available at this https URL. 

---
# ConsintBench: Evaluating Language Models on Real-World Consumer Intent Understanding 

**Authors**: Xiaozhe Li, TianYi Lyu, Siyi Yang, Yuxi Gong, Yizhao Yang, Jinxuan Huang, Ligao Zhang, Zhuoyi Huang, Qingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13499)  

**Abstract**: Understanding human intent is a complex, high-level task for large language models (LLMs), requiring analytical reasoning, contextual interpretation, dynamic information aggregation, and decision-making under uncertainty. Real-world public discussions, such as consumer product discussions, are rarely linear or involve a single user. Instead, they are characterized by interwoven and often conflicting perspectives, divergent concerns, goals, emotional tendencies, as well as implicit assumptions and background knowledge about usage scenarios. To accurately understand such explicit public intent, an LLM must go beyond parsing individual sentences; it must integrate multi-source signals, reason over inconsistencies, and adapt to evolving discourse, similar to how experts in fields like politics, economics, or finance approach complex, uncertain environments. Despite the importance of this capability, no large-scale benchmark currently exists for evaluating LLMs on real-world human intent understanding, primarily due to the challenges of collecting real-world public discussion data and constructing a robust evaluation pipeline. To bridge this gap, we introduce \bench, the first dynamic, live evaluation benchmark specifically designed for intent understanding, particularly in the consumer domain. \bench is the largest and most diverse benchmark of its kind, supporting real-time updates while preventing data contamination through an automated curation pipeline. 

---
# LiteraryQA: Towards Effective Evaluation of Long-document Narrative QA 

**Authors**: Tommaso Bonomo, Luca Gioffré, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2510.13494)  

**Abstract**: Question Answering (QA) on narrative text poses a unique challenge to current systems, requiring a deep understanding of long, complex documents. However, the reliability of NarrativeQA, the most widely used benchmark in this domain, is hindered by noisy documents and flawed QA pairs. In this work, we introduce LiteraryQA, a high-quality subset of NarrativeQA focused on literary works. Using a human- and LLM-validated pipeline, we identify and correct low-quality QA samples while removing extraneous text from source documents. We then carry out a meta-evaluation of automatic metrics to clarify how systems should be evaluated on LiteraryQA. This analysis reveals that all n-gram-based metrics have a low system-level correlation to human judgment, while LLM-as-a-Judge evaluations, even with small open-weight models, can strongly agree with the ranking identified by humans. Finally, we benchmark a set of long-context LLMs on LiteraryQA. We release our code and data at this https URL. 

---
# Beyond Single-Reward: Multi-Pair, Multi-Perspective Preference Optimization for Machine Translation 

**Authors**: Hao Wang, Linlong Xu, Heng Liu, Yangyang Liu, Xiaohu Zhao, Bo Zeng, Liangying Shao, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13434)  

**Abstract**: Direct Preference Optimization (DPO) is a powerful paradigm for aligning Large Language Models (LLMs) to human preferences in Machine Translation (MT), but current methods are hindered by two fundamental challenges: (1) flawed reward signals from Quality Estimation (QE) models that overlook critical errors like translation hallucination, and (2) inefficient data utilization that discards valuable learning signals by selecting only a single win-loss pair. To address these limitations, we introduce M^2PO: Multi-Pair, Multi-Perspective Preference Optimization. Our framework integrates a multi-perspective reward engine that creates a more robust signal by combining two key viewpoints: a new hallucination penalty for factuality, and an innovative dynamic quality score that adaptively fuses external evaluations with the model's own evolving judgment. This is synergistically paired with a multi-pair construction strategy that systematically creates a comprehensive set of preference pairs from the entire pool of translation candidates. This synergistic approach ensures the model learns from a richer spectrum of quality trade-offs, leading to more robust and faithful translations. On challenging WMT21-22 benchmarks, M^2PO substantially outperforms existing preference optimization methods and demonstrates highly competitive performance against leading proprietary LLMs. 

---
# Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps 

**Authors**: Ahmed Alzubaidi, Shaikha Alsuwaidi, Basma El Amel Boussaha, Leen AlQadi, Omar Alkaabi, Mohammed Alyafeai, Hamza Alobeidli, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2510.13430)  

**Abstract**: This survey provides the first systematic review of Arabic LLM benchmarks, analyzing 40+ evaluation benchmarks across NLP tasks, knowledge domains, cultural understanding, and specialized capabilities. We propose a taxonomy organizing benchmarks into four categories: Knowledge, NLP Tasks, Culture and Dialects, and Target-Specific evaluations. Our analysis reveals significant progress in benchmark diversity while identifying critical gaps: limited temporal evaluation, insufficient multi-turn dialogue assessment, and cultural misalignment in translated datasets. We examine three primary approaches: native collection, translation, and synthetic generation discussing their trade-offs regarding authenticity, scale, and cost. This work serves as a comprehensive reference for Arabic NLP researchers, providing insights into benchmark methodologies, reproducibility standards, and evaluation metrics while offering recommendations for future development. 

---
# Investigating Lexical Change through Cross-Linguistic Colexification Patterns 

**Authors**: Kim Gfeller, Sabine Stoll, Chundra Cathcart, Paul Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2510.13407)  

**Abstract**: One of the most intriguing features of language is its constant change, with ongoing shifts in how meaning is expressed. Despite decades of research, the factors that determine how and why meanings evolve remain only partly understood. Colexification -- the phenomenon of expressing multiple distinct concepts using the same word form -- serves as a valuable window onto the dynamics of meaning change across languages. Here, we apply phylogenetic comparative models to dictionary data from three language families, Austronesian, Indo-European, and Uralic, in order to shed light on the evolutionary dynamics underlying the colexification of concept pairs. We assess the effects of three predictors: associativity, borrowability, and usage frequency. Our results show that more closely related concept pairs are colexified across a larger portion of the family tree and exhibit slower rates of change. In contrast, concept pairs that are more frequent and more prone to borrowing tend to change more rapidly and are less often colexified. We also find considerable differences between the language families under study, suggesting that areal and cultural factors may play a role. 

---
# Doing Things with Words: Rethinking Theory of Mind Simulation in Large Language Models 

**Authors**: Agnese Lombardi, Alessandro Lenci  

**Link**: [PDF](https://arxiv.org/pdf/2510.13395)  

**Abstract**: Language is fundamental to human cooperation, facilitating not only the exchange of information but also the coordination of actions through shared interpretations of situational contexts. This study explores whether the Generative Agent-Based Model (GABM) Concordia can effectively model Theory of Mind (ToM) within simulated real-world environments. Specifically, we assess whether this framework successfully simulates ToM abilities and whether GPT-4 can perform tasks by making genuine inferences from social context, rather than relying on linguistic memorization. Our findings reveal a critical limitation: GPT-4 frequently fails to select actions based on belief attribution, suggesting that apparent ToM-like abilities observed in previous studies may stem from shallow statistical associations rather than true reasoning. Additionally, the model struggles to generate coherent causal effects from agent actions, exposing difficulties in processing complex social interactions. These results challenge current statements about emergent ToM-like capabilities in LLMs and highlight the need for more rigorous, action-based evaluation frameworks. 

---
# Make an Offer They Can't Refuse: Grounding Bayesian Persuasion in Real-World Dialogues without Pre-Commitment 

**Authors**: Buwei He, Yang Liu, Zhaowei Zhang, Zixia Jia, Huijia Wu, Zhaofeng He, Zilong Zheng, Yipeng Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13387)  

**Abstract**: Persuasion, a fundamental social capability for humans, remains a challenge for AI systems such as large language models (LLMs). Current studies often overlook the strategic use of information asymmetry in message design or rely on strong assumptions regarding pre-commitment. In this work, we explore the application of Bayesian Persuasion (BP) in natural language within single-turn dialogue settings, to enhance the strategic persuasion capabilities of LLMs. Our framework incorporates a commitment-communication mechanism, where the persuader explicitly outlines an information schema by narrating their potential types (e.g., honest or dishonest), thereby guiding the persuadee in performing the intended Bayesian belief update. We evaluate two variants of our approach: Semi-Formal-Natural-Language (SFNL) BP and Fully-Natural-Language (FNL) BP, benchmarking them against both naive and strong non-BP (NBP) baselines within a comprehensive evaluation framework. This framework covers a diverse set of persuadees -- including LLM instances with varying prompts and fine-tuning and human participants -- across tasks ranging from specially designed persuasion scenarios to general everyday situations. Experimental results on LLM-based agents reveal three main findings: (1) LLMs guided by BP strategies consistently achieve higher persuasion success rates than NBP baselines; (2) SFNL exhibits greater credibility and logical coherence, while FNL shows stronger emotional resonance and robustness in naturalistic conversations; (3) with supervised fine-tuning, smaller models can attain BP performance comparable to that of larger models. 

---
# Document Intelligence in the Era of Large Language Models: A Survey 

**Authors**: Weishi Wang, Hengchang Hu, Zhijie Zhang, Zhaochen Li, Hongxin Shao, Daniel Dahlmeier  

**Link**: [PDF](https://arxiv.org/pdf/2510.13366)  

**Abstract**: Document AI (DAI) has emerged as a vital application area, and is significantly transformed by the advent of large language models (LLMs). While earlier approaches relied on encoder-decoder architectures, decoder-only LLMs have revolutionized DAI, bringing remarkable advancements in understanding and generation. This survey provides a comprehensive overview of DAI's evolution, highlighting current research attempts and future prospects of LLMs in this field. We explore key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions, including agent-based approaches and document-specific foundation models. This paper aims to provide a structured analysis of the state-of-the-art in DAI and its implications for both academic and practical applications. 

---
# D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree 

**Authors**: Xiang Lei, Qin Li, Min Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13363)  

**Abstract**: Large Language Models (LLMs) often exhibit factual inconsistencies and logical decay in extended, multi-turn dialogues, a challenge stemming from their reliance on static, pre-trained knowledge and an inability to reason adaptively over the dialogue history. Prevailing mitigation strategies, such as Retrieval-Augmented Generation (RAG) and agentic working memories, improve information recall but still engage with fundamentally static knowledge sources and follow pre-defined single reasoning path. This hinders their ability to preserve factual and logical consistency of their responses in multi-turn dialogues while the context evolves over time. To address this issue, we propose D-SMART, a model-agnostic framework designed to maintain multi-turn dialogue consistency by enabling LLMs to build and reason over a dynamic, structured representation of the conversational context. This is achieved via two synergistic components: (1) a Dynamic Structured Memory (DSM), which incrementally constructs and maintains an authoritative, OWL-compliant knowledge graph of the conversation; and (2) a Reasoning Tree (RT), which executes inferences as an explicit and traceable multi-step search over the graph. As the popular-used quality score (judged by GPT-4) can overlook logical flaws, we introduce new NLI-based metrics to better measure multi-turn dialogue consistency. Comprehensive experiments on the MT-Bench-101 benchmark show that D-SMART significantly outperforms state-of-the-art baselines, elevating the dialogue consistency score by over 48\% for both proprietary and open-source models, and notably improves the quality score of the latter by up to 10.1\%. 

---
# Personal Attribute Leakage in Federated Speech Models 

**Authors**: Hamdan Al-Ali, Ali Reza Ghavamipour, Tommaso Caselli, Fatih Turkmen, Zeerak Talat, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.13357)  

**Abstract**: Federated learning is a common method for privacy-preserving training of machine learning models. In this paper, we analyze the vulnerability of ASR models to attribute inference attacks in the federated setting. We test a non-parametric white-box attack method under a passive threat model on three ASR models: Wav2Vec2, HuBERT, and Whisper. The attack operates solely on weight differentials without access to raw speech from target speakers. We demonstrate attack feasibility on sensitive demographic and clinical attributes: gender, age, accent, emotion, and dysarthria. Our findings indicate that attributes that are underrepresented or absent in the pre-training data are more vulnerable to such inference attacks. In particular, information about accents can be reliably inferred from all models. Our findings expose previously undocumented vulnerabilities in federated ASR models and offer insights towards improved security. 

---
# Protect: Towards Robust Guardrailing Stack for Trustworthy Enterprise LLM Systems 

**Authors**: Karthik Avinash, Nikhil Pareek, Rishav Hada  

**Link**: [PDF](https://arxiv.org/pdf/2510.13351)  

**Abstract**: The increasing deployment of Large Language Models (LLMs) across enterprise and mission-critical domains has underscored the urgent need for robust guardrailing systems that ensure safety, reliability, and compliance. Existing solutions often struggle with real-time oversight, multi-modal data handling, and explainability -- limitations that hinder their adoption in regulated environments. Existing guardrails largely operate in isolation, focused on text alone making them inadequate for multi-modal, production-scale environments. We introduce Protect, natively multi-modal guardrailing model designed to operate seamlessly across text, image, and audio inputs, designed for enterprise-grade deployment. Protect integrates fine-tuned, category-specific adapters trained via Low-Rank Adaptation (LoRA) on an extensive, multi-modal dataset covering four safety dimensions: toxicity, sexism, data privacy, and prompt injection. Our teacher-assisted annotation pipeline leverages reasoning and explanation traces to generate high-fidelity, context-aware labels across modalities. Experimental results demonstrate state-of-the-art performance across all safety dimensions, surpassing existing open and proprietary models such as WildGuard, LlamaGuard-4, and GPT-4.1. Protect establishes a strong foundation for trustworthy, auditable, and production-ready safety systems capable of operating across text, image, and audio modalities. 

---
# Are Proverbs the New Pythian Oracles? Exploring Sentiment in Greek Sayings 

**Authors**: Katerina Korre, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.13341)  

**Abstract**: Proverbs are among the most fascinating linguistic phenomena that transcend cultural and linguistic boundaries. Yet, much of the global landscape of proverbs remains underexplored, as many cultures preserve their traditional wisdom within their own communities due to the oral tradition of the phenomenon. Taking advantage of the current advances in Natural Language Processing (NLP), we focus on Greek proverbs, analyzing their sentiment. Departing from an annotated dataset of Greek proverbs, we expand it to include local dialects, effectively mapping the annotated sentiment. We present (1) a way to exploit LLMs in order to perform sentiment classification of proverbs, (2) a map of Greece that provides an overview of the distribution of sentiment, (3) a combinatory analysis in terms of the geographic position, dialect, and topic of proverbs. Our findings show that LLMs can provide us with an accurate enough picture of the sentiment of proverbs, especially when approached as a non-conventional sentiment polarity task. Moreover, in most areas of Greece negative sentiment is more prevalent. 

---
# Taming the Fragility of KV Cache Eviction in LLM Inference 

**Authors**: Yuan Feng, Haoyu Guo, JunLin Lv, S. Kevin Zhou, Xike Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.13334)  

**Abstract**: Large language models have revolutionized natural language processing, yet their deployment remains hampered by the substantial memory and runtime overhead of the transformer's Key-Value cache. To mitigate this, recent methods employ a scoring-aggregation framework to evict unimportant cache entries, based on the stability assumption-that a fixed subset of entries remains consistently important during generation. However, prior work has largely focused on refining importance indicators for scoring, while defaulting to mean aggregation due to a faithful trust in the stability assumption. In this work, we argue that this underlying assumption is inherently fragile, making mean aggregation highly vulnerable in extreme cases. To counter this, we propose a simple yet elegant defensive aggregation strategy: a two-step, linear-time approach that controls worst-case risk, thereby defending against extreme cases with negligible computational overhead. Embodying this strategy, we propose a novel cache eviction method, DefensiveKV and its extension, Layer-DefensiveKV, which incorporates layer-wise budget allocation. Across seven task domains (18 datasets), our methods reduce generation quality loss by 2.3x and 4.3x respectively, versus the strongest baseline under a 20% cache size. These results set new performance benchmarks and pioneer a promising direction for optimizing cache eviction against underlying fragility through worst-case risk management. Our code is available at this https URL. 

---
# Embedding-Based Context-Aware Reranker 

**Authors**: Ye Yuan, Mohammad Amin Shabani, Siqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13329)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems rely on retrieving relevant evidence from a corpus to support downstream generation. The common practice of splitting a long document into multiple shorter passages enables finer-grained and targeted information retrieval. However, it also introduces challenges when a correct retrieval would require inference across passages, such as resolving coreference, disambiguating entities, and aggregating evidence scattered across multiple sources. Many state-of-the-art (SOTA) reranking methods, despite utilizing powerful large pretrained language models with potentially high inference costs, still neglect the aforementioned challenges. Therefore, we propose Embedding-Based Context-Aware Reranker (EBCAR), a lightweight reranking framework operating directly on embeddings of retrieved passages with enhanced cross-passage understandings through the structural information of the passages and a hybrid attention mechanism, which captures both high-level interactions across documents and low-level relationships within each document. We evaluate EBCAR against SOTA rerankers on the ConTEB benchmark, demonstrating its effectiveness for information retrieval requiring cross-passage inference and its advantages in both accuracy and efficiency. 

---
# ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering 

**Authors**: Simon Lupart, Mohammad Aliannejadi, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2510.13312)  

**Abstract**: We present ChatR1, a reasoning framework based on reinforcement learning (RL) for conversational question answering (CQA). Reasoning plays an important role in CQA, where user intent evolves across dialogue turns, and utterances are often underspecified, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Unlike static `rewrite, retrieve, and generate' pipelines, ChatR1 interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through RL. To address the challenge of sparse and delayed rewards in RL, we propose an intent-aware reward that provides turn-level feedback by aligning retrieval and reasoning with evolving user goals. Our proposed ChatR1 demonstrates strong performance on both 3B and 7B model backbones, outperforming competitive models on five CQA datasets, measured by different metrics (F1, BERTScore, and LLM-as-judge). We include a diverse set of CQA datasets to cover topic shifts, evolving intents, mixed-initiative dialogues, and multi-document grounding, testing ChatR1's performance from various aspects. Ablation studies confirm the effectiveness of the intent-aware reward. Our analyses further reveal diverse reasoning trajectories and effective use of the search tool. ChatR1 also generalizes robustly across domains, demonstrating that RL-based reasoning enables more flexible and context-sensitive behavior than static CQA pipelines. 

---
# LLM one-shot style transfer for Authorship Attribution and Verification 

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2510.13302)  

**Abstract**: Computational stylometry analyzes writing style through quantitative patterns in text, supporting applications from forensic tasks such as identity linking and plagiarism detection to literary attribution in the humanities. Supervised and contrastive approaches rely on data with spurious correlations and often confuse style with topic. Despite their natural use in AI-generated text detection, the CLM pre-training of modern LLMs has been scarcely leveraged for general authorship problems. We propose a novel unsupervised approach based on this extensive pre-training and the in-context learning capabilities of LLMs, employing the log-probabilities of an LLM to measure style transferability from one text to another. Our method significantly outperforms LLM prompting approaches of comparable scale and achieves higher accuracy than contrastively trained baselines when controlling for topical correlations. Moreover, performance scales fairly consistently with the size of the base model and, in the case of authorship verification, with an additional mechanism that increases test-time computation; enabling flexible trade-offs between computational cost and accuracy. 

---
# Mismatch Aware Guidance for Robust Emotion Control in Auto-Regressive TTS Models 

**Authors**: Yizhou Peng, Yukun Ma, Chong Zhang, Yi-Wen Chao, Chongjia Ni, Bin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13293)  

**Abstract**: While Text-to-Speech (TTS) systems can achieve fine-grained control over emotional expression via natural language prompts, a significant challenge emerges when the desired emotion (style prompt) conflicts with the semantic content of the text. This mismatch often results in unnatural-sounding speech, undermining the goal of achieving fine-grained emotional control. Classifier-Free Guidance (CFG) is a key technique for enhancing prompt alignment; however, its application to auto-regressive (AR) TTS models remains underexplored, which can lead to degraded audio quality. This paper directly addresses the challenge of style-content mismatch in AR TTS models by proposing an adaptive CFG scheme that adjusts to different levels of the detected mismatch, as measured using large language models or natural language inference models. This solution is based on a comprehensive analysis of CFG's impact on emotional expressiveness in state-of-the-art AR TTS models. Our results demonstrate that the proposed adaptive CFG scheme improves the emotional expressiveness of the AR TTS model while maintaining audio quality and intelligibility. 

---
# Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems 

**Authors**: Xuxin Cheng, Ke Zeng, Zhiquan Cao, Linyi Dai, Wenxuan Gao, Fei Han, Ai Jian, Feng Hong, Wenxing Hu, Zihe Huang, Dejian Kong, Jia Leng, Zhuoyuan Liao, Pei Liu, Jiaye Lin, Xing Ma, Jingqing Ruan, Jiaxing Song, Xiaoyu Tan, Ruixuan Xiao, Wenhui Yu, Wenyu Zhan, Haoxing Zhang, Chao Zhou, Hao Zhou, Shaodong Zheng, Ruinian Chen, Siyuan Chen, Ziyang Chen, Yiwen Dong, Yaoyou Fan, Yangyi Fang, Yang Gan, Shiguang Guo, Qi He, Chaowen Hu, Binghui Li, Dailin Li, Xiangyu Li, Yan Li, Chengjian Liu, Xiangfeng Liu, Jiahui Lv, Qiao Ma, Jiang Pan, Cong Qin, Chenxing Sun, Wen Sun, Zhonghui Wang, Abudukelimu Wuerkaixi, Xin Yang, Fangyi Yuan, Yawen Zhu, Tianyi Zhai, Jie Zhang, Runlai Zhang, Yao Xu, Yiran Zhao, Yifan Wang, Xunliang Cai, Yangen Hu, Cao Liu, Lu Pan, Xiaoli Wang, Bo Xiao, Wenyuan Yao, Qianlin Zhou, Benchang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13291)  

**Abstract**: Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service. 

---
# In-Distribution Steering: Balancing Control and Coherence in Language Model Generation 

**Authors**: Arthur Vogels, Benjamin Wong, Yann Choho, Annabelle Blangero, Milan Bhan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13285)  

**Abstract**: Activation steering methods control large language model (LLM) behavior by modifying internal activations at inference time. However, most existing activation steering methods rely on a fixed steering strength, leading to either insufficient control or unadapted intervention that degrades text plausibility and coherence. We introduce In-Distribution Steering (IDS), a novel method that adapts steering strength based on the input data distribution in representation space. IDS dynamically adjusts interventions according to how far a given input lies within the distribution, enabling adaptive intervention and generation stability during text generation. Experiments demonstrate that IDS achieves strong accuracy on classification tasks while producing coherent text without collapse, making IDS particularly well suited for real-world applications. 

---
# Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation 

**Authors**: Zhichao Xu, Zongyu Wu, Yun Zhou, Aosong Feng, Kang Zhou, Sangmin Woo, Kiran Ramnath, Yijun Tian, Xuan Qi, Weikang Qiu, Lin Lee Cheong, Haibo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.13272)  

**Abstract**: Inspired by the success of reinforcement learning (RL) in Large Language Model (LLM) training for domains like math and code, recent works have begun exploring how to train LLMs to use search engines more effectively as tools for retrieval-augmented generation. Although these methods achieve performance improvement across QA benchmarks, many prioritize final answer correctness while overlooking the quality of intermediate reasoning steps, which may lead to chain-of-thought unfaithfulness. In this paper, we first introduce a comprehensive evaluation framework for evaluating RL-based search agents, covering three distinct faithfulness metrics: information-think faithfulness, think-answer faithfulness, and think-search faithfulness. Our evaluations reveal that a prototypical RL-based search agent, Search-R1, has significant room for improvement in this regard. To foster faithful reasoning, we introduce VERITAS (Verifying Entailed Reasoning through Intermediate Traceability in Agentic Search), a novel framework that integrates fine-grained faithfulness rewards into the reinforcement learning process. Our experiments show that models trained with VERITAS not only significantly improve reasoning faithfulness, but also achieve comparable task performance across seven QA benchmarks. 

---
# Do You Get the Hint? Benchmarking LLMs on the Board Game Concept 

**Authors**: Ine Gevers, Walter Daelemans  

**Link**: [PDF](https://arxiv.org/pdf/2510.13271)  

**Abstract**: Large language models (LLMs) have achieved striking successes on many benchmarks, yet recent studies continue to expose fundamental weaknesses. In particular, tasks that require abstract reasoning remain challenging, often because they use representations such as grids, symbols, or visual patterns that differ from the natural language data LLMs are trained on. In this paper, we introduce Concept, a simple word-guessing board game, as a benchmark for probing abductive reasoning in a representation that is much closer to LLM pre-training data: natural language. Our results show that this game, easily solved by humans (with a success rate of over 90\%), is still very challenging for state-of-the-art LLMs (no model exceeds 40\% success rate). Specifically, we observe that LLMs struggle with interpreting other players' strategic intents, and with correcting initial hypotheses given sequential information updates. In addition, we extend the evaluation across multiple languages, and find that the LLM performance drops further in lower-resource languages (Dutch, French, and Spanish) compared to English. 

---
# Hierarchical Frequency Tagging Probe (HFTP): A Unified Approach to Investigate Syntactic Structure Representations in Large Language Models and the Human Brain 

**Authors**: Jingmin An, Yilong Song, Ruolin Yang, Nai Ding, Lingxi Lu, Yuxuan Wang, Wei Wang, Chu Zhuang, Qian Wang, Fang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13255)  

**Abstract**: Large Language Models (LLMs) demonstrate human-level or even superior language abilities, effectively modeling syntactic structures, yet the specific computational modules responsible remain unclear. A key question is whether LLM behavioral capabilities stem from mechanisms akin to those in the human brain. To address these questions, we introduce the Hierarchical Frequency Tagging Probe (HFTP), a tool that utilizes frequency-domain analysis to identify neuron-wise components of LLMs (e.g., individual Multilayer Perceptron (MLP) neurons) and cortical regions (via intracranial recordings) encoding syntactic structures. Our results show that models such as GPT-2, Gemma, Gemma 2, Llama 2, Llama 3.1, and GLM-4 process syntax in analogous layers, while the human brain relies on distinct cortical regions for different syntactic levels. Representational similarity analysis reveals a stronger alignment between LLM representations and the left hemisphere of the brain (dominant in language processing). Notably, upgraded models exhibit divergent trends: Gemma 2 shows greater brain similarity than Gemma, while Llama 3.1 shows less alignment with the brain compared to Llama 2. These findings offer new insights into the interpretability of LLM behavioral improvements, raising questions about whether these advancements are driven by human-like or non-human-like mechanisms, and establish HFTP as a valuable tool bridging computational linguistics and cognitive neuroscience. This project is available at this https URL. 

---
# A fully automated and scalable Parallel Data Augmentation for Low Resource Languages using Image and Text Analytics 

**Authors**: Prawaal Sharma, Navneet Goyal, Poonam Goyal, Vishnupriyan R  

**Link**: [PDF](https://arxiv.org/pdf/2510.13211)  

**Abstract**: Linguistic diversity across the world creates a disparity with the availability of good quality digital language resources thereby restricting the technological benefits to majority of human population. The lack or absence of data resources makes it difficult to perform NLP tasks for low-resource languages. This paper presents a novel scalable and fully automated methodology to extract bilingual parallel corpora from newspaper articles using image and text analytics. We validate our approach by building parallel data corpus for two different language combinations and demonstrate the value of this dataset through a downstream task of machine translation and improve over the current baseline by close to 3 BLEU points. 

---
# LLM-Guided Synthetic Augmentation (LGSA) for Mitigating Bias in AI Systems 

**Authors**: Sai Suhruth Reddy Karri, Yashwanth Sai Nallapuneni, Laxmi Narasimha Reddy Mallireddy, Gopichand G  

**Link**: [PDF](https://arxiv.org/pdf/2510.13202)  

**Abstract**: Bias in AI systems, especially those relying on natural language data, raises ethical and practical concerns. Underrepresentation of certain groups often leads to uneven performance across demographics. Traditional fairness methods, such as pre-processing, in-processing, and post-processing, depend on protected-attribute labels, involve accuracy-fairness trade-offs, and may not generalize across datasets. To address these challenges, we propose LLM-Guided Synthetic Augmentation (LGSA), which uses large language models to generate counterfactual examples for underrepresented groups while preserving label integrity. We evaluated LGSA on a controlled dataset of short English sentences with gendered pronouns, professions, and binary classification labels. Structured prompts were used to produce gender-swapped paraphrases, followed by quality control including semantic similarity checks, attribute verification, toxicity screening, and human spot checks. The augmented dataset expanded training coverage and was used to train a classifier under consistent conditions. Results show that LGSA reduces performance disparities without compromising accuracy. The baseline model achieved 96.7 percent accuracy with a 7.2 percent gender bias gap. Simple swap augmentation reduced the gap to 0.7 percent but lowered accuracy to 95.6 percent. LGSA achieved 99.1 percent accuracy with a 1.9 percent bias gap, improving performance on female-labeled examples. These findings demonstrate that LGSA is an effective strategy for bias mitigation, enhancing subgroup balance while maintaining high task accuracy and label fidelity. 

---
# Text Anomaly Detection with Simplified Isolation Kernel 

**Authors**: Yang Cao, Sikun Yang, Yujiu Yang, Lianyong Qi, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13197)  

**Abstract**: Two-step approaches combining pre-trained large language model embeddings and anomaly detectors demonstrate strong performance in text anomaly detection by leveraging rich semantic representations. However, high-dimensional dense embeddings extracted by large language models pose challenges due to substantial memory requirements and high computation time. To address this challenge, we introduce the Simplified Isolation Kernel (SIK), which maps high-dimensional dense embeddings to lower-dimensional sparse representations while preserving crucial anomaly characteristics. SIK has linear time complexity and significantly reduces space complexity through its innovative boundary-focused feature mapping. Experiments across 7 datasets demonstrate that SIK achieves better detection performance than 11 state-of-the-art (SOTA) anomaly detection algorithms while maintaining computational efficiency and low memory cost. All code and demonstrations are available at this https URL. 

---
# StressTransfer: Stress-Aware Speech-to-Speech Translation with Emphasis Preservation 

**Authors**: Xi Chen, Yuchen Song, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.13194)  

**Abstract**: We propose a stress-aware speech-to-speech translation (S2ST) system that preserves word-level emphasis by leveraging LLMs for cross-lingual emphasis conversion. Our method translates source-language stress into target-language tags that guide a controllable TTS model. To overcome data scarcity, we developed a pipeline to automatically generate aligned training data and introduce the "LLM-as-Judge" for evaluation. Experiments show our approach substantially outperforms baselines in preserving emphasis while maintaining comparable translation quality, speaker intent, and naturalness. Our work highlights the importance of prosody in translation and provides an effective, data-efficient solution for preserving paralinguistic cues in S2ST. 

---
# Grounding Long-Context Reasoning with Contextual Normalization for Retrieval-Augmented Generation 

**Authors**: Jiamin Chen, Yuchen Li, Xinyu Ma, Xinran Chen, Xiaokun Zhang, Shuaiqiang Wang, Chen Ma, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.13191)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become an essential approach for extending the reasoning and knowledge capacity of large language models (LLMs). While prior research has primarily focused on retrieval quality and prompting strategies, the influence of how the retrieved documents are framed, i.e., context format, remains underexplored. We show that seemingly superficial choices, such as delimiters or structural markers in key-value extraction, can induce substantial shifts in accuracy and stability, even when semantic content is identical. To systematically investigate this effect, we design controlled experiments that vary context density, delimiter styles, and positional placement, revealing the underlying factors that govern performance differences. Building on these insights, we introduce Contextual Normalization, a lightweight strategy that adaptively standardizes context representations before generation. Extensive experiments on both controlled and real-world RAG benchmarks across diverse settings demonstrate that the proposed strategy consistently improves robustness to order variation and strengthens long-context utilization. These findings underscore that reliable RAG depends not only on retrieving the right content, but also on how that content is presented, offering both new empirical evidence and a practical technique for better long-context reasoning. 

---
# SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs 

**Authors**: Juan Ren, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2510.13190)  

**Abstract**: Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs. 

---
# DSCD: Large Language Model Detoxification with Self-Constrained Decoding 

**Authors**: Ming Dong, Jinkui Zhang, Bolong Zheng, Xinhui Tu, Po Hu, Tingting He  

**Link**: [PDF](https://arxiv.org/pdf/2510.13183)  

**Abstract**: Detoxification in large language models (LLMs) remains a significant research challenge. Existing decoding detoxification methods are all based on external constraints, which require additional resource overhead and lose generation fluency. This work proposes Detoxification with Self-Constrained Decoding (DSCD), a novel method for LLM detoxification without parameter fine-tuning. DSCD strengthens the inner next-token distribution of the safety layer while weakening that of hallucination and toxic layers during output generation. This effectively diminishes toxicity and enhances output safety. DSCD offers lightweight, high compatibility, and plug-and-play capabilities, readily integrating with existing detoxification methods for further performance improvement. Extensive experiments on representative open-source LLMs and public datasets validate DSCD's effectiveness, demonstrating state-of-the-art (SOTA) performance in both detoxification and generation fluency, with superior efficiency compared to existing methods. These results highlight DSCD's potential as a practical and scalable solution for safer LLM deployments. 

---
# Putting on the Thinking Hats: A Survey on Chain of Thought Fine-tuning from the Perspective of Human Reasoning Mechanism 

**Authors**: Xiaoshu Chen, Sihang Zhou, Ke Liang, Duanyang Yuan, Haoyuan Chen, Xiaoyu Sun, Linyuan Meng, Xinwang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13170)  

**Abstract**: Chain of thought (CoT) fine-tuning aims to endow large language models (LLMs) with reasoning capabilities by training them on curated reasoning traces. It leverages both supervised and reinforced fine-tuning to cultivate human-like reasoning skills in LLMs, including detailed planning, divergent thinking, intuitive judgment, timely reflection, internal thinking, and fact perception, etc. As CoT fine-tuning has advanced, LLMs have demonstrated substantial improvements in tasks such as mathematical reasoning and code generation. However, existing surveys about CoT fine-tuning primarily focus on technical aspects and overlook a systematic analysis from the perspective of human reasoning mechanisms. Given that the ultimate goal of CoT fine-tuning is to enable LLMs to reason like humans, it is crucial to investigate this technique through the lens of human cognition. To fill this gap, we present the first comprehensive survey of CoT fine-tuning grounded in human reasoning theory. Specifically, inspired by the well-known Six Thinking Hats framework, which systematically characterizes common human thinking modes using six metaphorical hats, we classify and examine CoT fine-tuning methods through this lens. Furthermore, building upon this theory, we outline potential directions for future research in CoT fine-tuning. In addition, we compile a comprehensive overview of existing datasets and model performances, and a real-time GitHub repository \footnote{this https URL} that continuously tracks recent advances in this area is maintained. We hope this survey will serve as a valuable resource to inspire innovation and foster progress in this rapidly evolving field. 

---
# CoT-Evo: Evolutionary Distillation of Chain-of-Thought for Scientific Reasoning 

**Authors**: Kehua Feng, Keyan Ding, Zhihui Zhu, Lei Liang, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13166)  

**Abstract**: While chain-of-thought (CoT) distillation from advanced large language models (LLMs) has proven effective in general reasoning tasks, it struggles in scientific domains where even advanced models often produce incorrect or superficial reasoning due to high complexity and specialized knowledge requirements. Directly distilling from such flawed outputs results in low-quality training data and limits the performance of smaller student models. To overcome this, we propose CoT-Evo, an evolutionary CoT distillation framework. It begins by constructing a diverse pool of reasoning trajectories from multiple LLM thinkers, enriches them with automatically retrieved domain knowledge, and iteratively refines the trajectories using novelty-driven selection, reflective recombination and mutation. The refinement is guided by a fitness function that evaluates answer correctness, coherence, and effective knowledge utilization. This results in a high-quality CoT dataset tailored for scientific reasoning. We employ this evolved dataset to fine-tune a compact model, which achieves state-of-the-art performance on scientific reasoning benchmarks. Our work establishes a scalable approach to synthesizing high-fidelity scientific reasoning data from diverse and fallible LLMs. 

---
# A Matter of Representation: Towards Graph-Based Abstract Code Generation 

**Authors**: Nyx Iskandar, Hisham Bedri, Andy Tsen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13163)  

**Abstract**: Most large language models (LLMs) today excel at generating raw, sequential code with minimal abstractions and custom structures. However, there has been little work on graph-based abstract code generation, where significant logic is encapsulated in predefined nodes and execution flow is determined by edges. This is relevant for visual programming languages, and in cases where raw source code is inaccessible to users and LLM training sets. In this work, we propose and evaluate JSON representations for graphs to enable high accuracy graph-based abstract code generation. We evaluate these representations on ScratchTest, a mini-benchmark based on our custom Python re-implementation of Scratch, which tests the LLM in code graph space. Our findings demonstrate that LLMs can indeed perform the aforementioned generation task in a single pass without relying on specialized or complex pipelines, given the correct graph representations. We also show that different representations induce significantly different accuracies, highlighting the instrumental role of representations in this generation task. All in all, this work establishes the first steps towards representation learning for graph-based abstract code generation. 

---
# Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference 

**Authors**: Nikhil Bhendawade, Kumari Nishu, Arnav Kundu, Chris Bartels, Minsik Cho, Irina Belousova  

**Link**: [PDF](https://arxiv.org/pdf/2510.13161)  

**Abstract**: Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rates but introduces additional latency overhead exacerbating the speed-accuracy tradeoff. Prior methods (Medusa, Hydra, EAGLE) partially reduce draft cost but either degrade acceptance or introduce overheads that limit scaling. We present Mirror Speculative Decoding (Mirror-SD), an inference algorithm that breaks the latency-acceptance tradeoff. Mirror-SD launches branch-complete rollouts from early-exit signals in parallel with the target model's suffix and explicitly maps computation across heterogeneous accelerators (GPU and NPU) to exploit cross-device parallelism. The draft speculates forward continuations for the target to verify, while the target simultaneously speculates correction paths for the draft, converting speculation into two complementary execution pipelines. To further cut draft latency without weakening acceptance semantics, we add speculative streaming so the draft emits multiple tokens per step. This dual strategy of parallel heterogeneous execution plus multi-token speculative streaming pushes speculative decoding toward its ideal regime of high acceptance with low overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD delivers consistent end-to-end gains, achieving 2.8x-5.8x wall-time speedups across diverse tasks and a 30% average relative improvement over the strongest baseline, EAGLE3. 

---
# I Am Aligned, But With Whom? MENA Values Benchmark for Evaluating Cultural Alignment and Multilingual Bias in LLMs 

**Authors**: Pardis Sadat Zahraei, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2510.13154)  

**Abstract**: We introduce MENAValues, a novel benchmark designed to evaluate the cultural alignment and multilingual biases of large language models (LLMs) with respect to the beliefs and values of the Middle East and North Africa (MENA) region, an underrepresented area in current AI evaluation efforts. Drawing from large-scale, authoritative human surveys, we curate a structured dataset that captures the sociocultural landscape of MENA with population-level response distributions from 16 countries. To probe LLM behavior, we evaluate diverse models across multiple conditions formed by crossing three perspective framings (neutral, personalized, and third-person/cultural observer) with two language modes (English and localized native languages: Arabic, Persian, Turkish). Our analysis reveals three critical phenomena: "Cross-Lingual Value Shifts" where identical questions yield drastically different responses based on language, "Reasoning-Induced Degradation" where prompting models to explain their reasoning worsens cultural alignment, and "Logit Leakage" where models refuse sensitive questions while internal probabilities reveal strong hidden preferences. We further demonstrate that models collapse into simplistic linguistic categories when operating in native languages, treating diverse nations as monolithic entities. MENAValues offers a scalable framework for diagnosing cultural misalignment, providing both empirical insights and methodological tools for developing more culturally inclusive AI. 

---
# Stable LLM Ensemble: Interaction between Example Representativeness and Diversity 

**Authors**: Junichiro Niimi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13143)  

**Abstract**: Large language models (LLMs) have achieved remarkable results in wide range of domains. However, the accuracy and robustness of one-shot LLM predictions remain highly sensitive to the examples and the diversity among ensemble members. This study systematically investigates the effects of example representativeness (one-shot strategy) and output diversity (sampling temperature) on LLM ensemble performance. Two one-shot strategies are compared: centroid-based representative examples (proposed) and randomly sampled examples (baseline) and sampling temperature also is varied. The proposed approach with higher temperature setting significantly outperforms random selection by +7.6% (macro-F1) and -10.5% (RMSE). Furthermore, the proposed model exceeds 5-shot prompting by +21.1% (macro-F1) and -24.0% (RMSE). Our findings demonstrate that combining representative example selection with increased temperature provides the appropriate level of diversity to the ensemble. This work highlights the practical importance of both example selection and controlled diversity in designing effective one-shot LLM ensembles. 

---
# Multi-Label Clinical Text Eligibility Classification and Summarization System 

**Authors**: Surya Tejaswi Yerramsetty, Almas Fathimah  

**Link**: [PDF](https://arxiv.org/pdf/2510.13115)  

**Abstract**: Clinical trials are central to medical progress because they help improve understanding of human health and the healthcare system. They play a key role in discovering new ways to detect, prevent, or treat diseases, and it is essential that clinical trials include participants with appropriate and diverse medical backgrounds. In this paper, we propose a system that leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to automate multi-label clinical text eligibility classification and summarization. The system combines feature extraction methods such as word embeddings (Word2Vec) and named entity recognition to identify relevant medical concepts, along with traditional vectorization techniques such as count vectorization and TF-IDF (Term Frequency-Inverse Document Frequency). We further explore weighted TF-IDF word embeddings that integrate both count-based and embedding-based strengths to capture term importance effectively. Multi-label classification using Random Forest and SVM models is applied to categorize documents based on eligibility criteria. Summarization techniques including TextRank, Luhn, and GPT-3 are evaluated to concisely summarize eligibility requirements. Evaluation with ROUGE scores demonstrates the effectiveness of the proposed methods. This system shows potential for automating clinical trial eligibility assessment using data-driven approaches, thereby improving research efficiency. 

---
# ESI: Epistemic Uncertainty Quantification via Semantic-preserving Intervention for Large Language Models 

**Authors**: Mingda Li, Xinyu Li, Weinan Zhang, Longxuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13103)  

**Abstract**: Uncertainty Quantification (UQ) is a promising approach to improve model reliability, yet quantifying the uncertainty of Large Language Models (LLMs) is non-trivial. In this work, we establish a connection between the uncertainty of LLMs and their invariance under semantic-preserving intervention from a causal perspective. Building on this foundation, we propose a novel grey-box uncertainty quantification method that measures the variation in model outputs before and after the semantic-preserving intervention. Through theoretical justification, we show that our method provides an effective estimate of epistemic uncertainty. Our extensive experiments, conducted across various LLMs and a variety of question-answering (QA) datasets, demonstrate that our method excels not only in terms of effectiveness but also in computational efficiency. 

---
# GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models 

**Authors**: Chen Zheng, Yuhang Cai, Deyi Liu, Jin Ma, Yiyuan Ma, Yuan Yang, Jing Liu, Yutao Zeng, Xun Zhou, Siyuan Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.13079)  

**Abstract**: Modern large language models leverage Mixture-of-Experts (MoE) architectures for efficient scaling, but face a critical challenge: functionally similar experts are often selected simultaneously, creating redundant computation and limiting effective model capacity. Existing auxiliary balance loss methods improve token distribution but fail to address the underlying expert diversity problem. We introduce GatePro, a novel parameter-free method that directly promotes expert selection diversity. GatePro identifies the most similar expert pairs and introduces localized competition mechanisms, preventing redundant expert co-activation while maintaining natural expert specialization. Our comprehensive evaluation demonstrates GatePro's effectiveness across model scales and benchmarks. Analysis demonstrates GatePro's ability to achieve enhanced expert diversity, where experts develop more distinct and complementary capabilities, avoiding functional redundancy. This approach can be deployed hot-swappable during any training phase without additional learnable parameters, offering a practical solution for improving MoE effectiveness. 

---
# On the Role of Preference Variance in Preference Optimization 

**Authors**: Jiacheng Guo, Zihao Li, Jiahao Qiu, Yue Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13022)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an important approach for learning from human preferences in aligning large language models (LLMs). However, collecting human preference data is costly and inefficient, motivating methods to reduce the required annotations. In this work, we investigate the impact of \emph{preference variance} (PVar), which measures the variance in model preferences when comparing pairs of responses, on the effectiveness of DPO training. We provide a theoretical insight by establishing an upper bound on the DPO gradient norm for any given prompt, showing it is controlled by the PVar of that prompt. This implies that prompts with low PVar can only produce small gradient updates, making them less valuable for learning. We validate this finding by fine-tuning LLMs with preferences generated by a reward model, evaluating on two benchmarks (AlpacaEval 2.0 and Arena-Hard). Experimental results demonstrate that prompts with higher PVar outperform randomly selected prompts or those with lower PVar. We also show that our PVar-based selection method is robust, when using smaller reward models (1B, 3B) for selection. Notably, in a separate experiment using the original human annotations from the UltraFeedback dataset, we found that training on only the top 10\% of prompts with the highest PVar yields better evaluation performance than training on the full dataset, highlighting the importance of preference variance in identifying informative examples for efficient LLM alignment. 

---
# CurLL: A Developmental Framework to Evaluate Continual Learning in Language Models 

**Authors**: Pavan Kalyan, Shubhra Mishra, Satya Lokam, Navin Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.13008)  

**Abstract**: We introduce a comprehensive continual learning dataset and benchmark (CurlL) grounded in human developmental trajectories from ages 5-10, enabling systematic and fine-grained assessment of models' ability to progressively acquire new skills. CurlL spans five developmental stages (0-4) covering ages 5-10, supported by a skill graph that breaks down broad skills into smaller abilities, concrete goals, and measurable indicators, while also capturing which abilities build on others. We generate a 23.4B-token synthetic dataset with controlled skill progression, vocabulary complexity, and format diversity, comprising paragraphs, comprehension-based QA (CQA), skill-testing QA (CSQA), and instruction-response (IR) pairs. Stage-wise token counts range from 2.12B to 6.78B tokens, supporting precise analysis of forgetting, forward transfer, and backward transfer. Using a 135M-parameter transformer trained under independent, joint, and sequential (continual) setups, we show trade-offs in skill retention and transfer efficiency. By mirroring human learning patterns and providing fine-grained control over skill dependencies, this work advances continual learning evaluations for language models. 

---
# OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning 

**Authors**: Yifeng Xiong, Xiaohui Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.13003)  

**Abstract**: Low-Rank Adaptation (LoRA) enables efficient fine-tuning of large language models but suffers from catastrophic forgetting when learned updates interfere with the dominant singular directions that encode essential pre-trained knowledge. We propose Orthogonal Projection LoRA (OPLoRA), a theoretically grounded approach that prevents this interference through double-sided orthogonal projections. By decomposing frozen weights via SVD, OPLoRA constrains LoRA updates to lie entirely within the orthogonal complement of the top-$k$ singular subspace using projections $P_L = I - U_k U_k^\top$ and $P_R = I - V_k V_k^\top$. We prove that this construction exactly preserves the top-$k$ singular triples, providing mathematical guarantees for knowledge retention. To quantify subspace interference, we introduce $\rho_k$, a metric measuring update alignment with dominant directions. Extensive experiments across commonsense reasoning, mathematics, and code generation demonstrate that OPLoRA significantly reduces forgetting while maintaining competitive task-specific performance on LLaMA-2 7B and Qwen2.5 7B, establishing orthogonal projection as an effective mechanism for knowledge preservation in parameter-efficient fine-tuning. 

---
# A Multilingual, Large-Scale Study of the Interplay between LLM Safeguards, Personalisation, and Disinformation 

**Authors**: João A. Leite, Arnav Arora, Silvia Gargova, João Luz, Gustavo Sampaio, Ian Roberts, Carolina Scarton, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2510.12993)  

**Abstract**: The human-like proficiency of Large Language Models (LLMs) has brought concerns about their potential misuse for generating persuasive and personalised disinformation at scale. While prior work has demonstrated that LLMs can generate disinformation, specific questions around persuasiveness and personalisation (generation of disinformation tailored to specific demographic attributes) remain largely unstudied. This paper presents the first large-scale, multilingual empirical study on persona-targeted disinformation generation by LLMs. Employing a red teaming methodology, we systematically evaluate the robustness of LLM safety mechanisms to persona-targeted prompts. A key novel result is AI-TRAITS (AI-generaTed peRsonAlIsed disinformaTion dataSet), a new dataset of around 1.6 million texts generated by eight state-of-the-art LLMs. AI-TRAITS is seeded by prompts that combine 324 disinformation narratives and 150 distinct persona profiles, covering four major languages (English, Russian, Portuguese, Hindi) and key demographic dimensions (country, generation, political orientation). The resulting personalised narratives are then assessed quantitatively and compared along the dimensions of models, languages, jailbreaking rate, and personalisation attributes. Our findings demonstrate that the use of even simple personalisation strategies in the prompts significantly increases the likelihood of jailbreaks for all studied LLMs. Furthermore, personalised prompts result in altered linguistic and rhetorical patterns and amplify the persuasiveness of the LLM-generated false narratives. These insights expose critical vulnerabilities in current state-of-the-art LLMs and offer a foundation for improving safety alignment and detection strategies in multilingual and cross-demographic contexts. 

---
# 3-Model Speculative Decoding 

**Authors**: Sanghyun Byun, Mohanad Odema, Jung Ick Guack, Baisub Lee, Jacob Song, Woo Seong Chung  

**Link**: [PDF](https://arxiv.org/pdf/2510.12966)  

**Abstract**: Speculative Decoding (SD) accelerates inference in large language models by using a smaller draft model to propose tokens, which are then verified by a larger target model. However, the throughput gains of SD are fundamentally limited by a trade-off between draft model size and token acceptance: smaller draft models generate tokens more quickly but exhibit greater divergence from the target model, resulting in lower acceptance rates and reduced speedups. We introduce Pyramid Speculative Decoding (PyramidSD), an extension of SD that inserts an intermediate qualifier model between the draft and target to bridge the distributional gap in output predictions, allowing smaller model to be used for drafting. This hierarchical decoding strategy improves alignment across models, enabling higher acceptance rates and allowing the use of significantly smaller draft models without sacrificing overall performance. PyramidSD builds on fuzzy acceptance criteria to support relaxed divergence thresholds at each stage, improving throughput. In experiments, PyramidSD achieves up to 1.91x generation speed over standard SD, reaching 124 tokens per second on a consumer GPU (RTX 4090). In small-memory settings with a 1B-parameter draft model and an 8B target model, PyramidSD minimally trades target model quality for improved throughput. Overall, PyramidSD offers a practical approach to enhancing speculative decoding efficiency and can be readily applied to existing inference pipelines. 

---
# The Curious Case of Curiosity across Human Cultures and LLMs 

**Authors**: Angana Borah, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2510.12943)  

**Abstract**: Recent advances in Large Language Models (LLMs) have expanded their role in human interaction, yet curiosity -- a central driver of inquiry -- remains underexplored in these systems, particularly across cultural contexts. In this work, we investigate cultural variation in curiosity using Yahoo! Answers, a real-world multi-country dataset spanning diverse topics. We introduce CUEST (CUriosity Evaluation across SocieTies), an evaluation framework that measures human-model alignment in curiosity through linguistic (style), topic preference (content) analysis and grounding insights in social science constructs. Across open- and closed-source models, we find that LLMs flatten cross-cultural diversity, aligning more closely with how curiosity is expressed in Western countries. We then explore fine-tuning strategies to induce curiosity in LLMs, narrowing the human-model alignment gap by up to 50\%. Finally, we demonstrate the practical value of curiosity for LLM adaptability across cultures, showing its importance for future NLP research. 

---
# Who's Asking? Evaluating LLM Robustness to Inquiry Personas in Factual Question Answering 

**Authors**: Nil-Jana Akpinar, Chia-Jung Lee, Vanessa Murdock, Pietro Perona  

**Link**: [PDF](https://arxiv.org/pdf/2510.12925)  

**Abstract**: Large Language Models (LLMs) should answer factual questions truthfully, grounded in objective knowledge, regardless of user context such as self-disclosed personal information, or system personalization. In this paper, we present the first systematic evaluation of LLM robustness to inquiry personas, i.e. user profiles that convey attributes like identity, expertise, or belief. While prior work has primarily focused on adversarial inputs or distractors for robustness testing, we evaluate plausible, human-centered inquiry persona cues that users disclose in real-world interactions. We find that such cues can meaningfully alter QA accuracy and trigger failure modes such as refusals, hallucinated limitations, and role confusion. These effects highlight how model sensitivity to user framing can compromise factual reliability, and position inquiry persona testing as an effective tool for robustness evaluation. 

---
# EduDial: Constructing a Large-scale Multi-turn Teacher-Student Dialogue Corpus 

**Authors**: Shouang Wei, Min Zhang, Xin Lin, Bo Jiang, Zhongxiang Dai, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12899)  

**Abstract**: Recently, several multi-turn dialogue benchmarks have been proposed to evaluate the conversational abilities of large language models (LLMs). As LLMs are increasingly recognized as a key technology for advancing intelligent education, owing to their ability to deeply understand instructional contexts and provide personalized guidance, the construction of dedicated teacher-student dialogue benchmarks has become particularly important. To this end, we present EduDial, a comprehensive multi-turn teacher-student dialogue dataset. EduDial covers 345 core knowledge points and consists of 34,250 dialogue sessions generated through interactions between teacher and student agents. Its design is guided by Bloom's taxonomy of educational objectives and incorporates ten questioning strategies, including situational questioning, zone of proximal development (ZPD) questioning, and metacognitive questioning-thus better capturing authentic classroom interactions. Furthermore, we design differentiated teaching strategies for students at different cognitive levels, thereby providing more targeted teaching guidance. Building on EduDial, we further develop EduDial-LLM 32B via training and propose an 11-dimensional evaluation framework that systematically measures the teaching abilities of LLMs, encompassing both overall teaching quality and content quality. Experiments on 17 mainstream LLMs reveal that most models struggle in student-centered teaching scenarios, whereas our EduDial-LLM achieves significant gains, consistently outperforming all baselines across all metrics. The code is available at this https URL. 

---
# A Critical Review of the Need for Knowledge-Centric Evaluation of Quranic Recitation 

**Authors**: Mohammed Hilal Al-Kharusi, Khizar Hayat, Khalil Bader Al Ruqeishi, Haroon Rashid Lone  

**Link**: [PDF](https://arxiv.org/pdf/2510.12858)  

**Abstract**: The sacred practice of Quranic recitation (Tajweed), governed by precise phonetic, prosodic, and theological rules, faces significant pedagogical challenges in the modern era. While digital technologies promise unprecedented access to education, automated tools for recitation evaluation have failed to achieve widespread adoption or pedagogical efficacy. This literature review investigates this critical gap, conducting a comprehensive analysis of academic research, web platforms, and commercial applications developed over the past two decades. Our synthesis reveals a fundamental misalignment in prevailing approaches that repurpose Automatic Speech Recognition (ASR) architectures, which prioritize lexical recognition over qualitative acoustic assessment and are plagued by data dependency, demographic biases, and an inability to provide diagnostically useful feedback. Critiquing these data--driven paradigms, we argue for a foundational paradigm shift towards a knowledge-centric computational framework. Capitalizing on the immutable nature of the Quranic text and the precisely defined rules of Tajweed, we propose that a robust evaluator must be architected around anticipatory acoustic modeling based on canonical rules and articulation points (Makhraj), rather than relying on statistical patterns learned from imperfect and biased datasets. This review concludes that the future of automated Quranic evaluation lies in hybrid systems that integrate deep linguistic knowledge with advanced audio analysis, offering a path toward robust, equitable, and pedagogically sound tools that can faithfully support learners worldwide. 

---
# Efficient Adaptive Transformer: An Empirical Study and Reproducible Framework 

**Authors**: Jan Miller  

**Link**: [PDF](https://arxiv.org/pdf/2510.12856)  

**Abstract**: The Efficient Adaptive Transformer (EAT) framework unifies three adaptive efficiency techniques - progressive token pruning, sparse attention, and dynamic early exiting - into a single, reproducible architecture for input-adaptive inference. EAT provides an open-source benchmarking pipeline that automates data processing, timing, and ablation across GLUE tasks (SST-2, QQP, MNLI). Although this empirical study finds that combining these mechanisms can increase latency in shallow six-layer models, it demonstrates that EAT achieves slightly higher accuracy than the optimized DistilBERT baseline on SST-2, illustrating the potential of dynamic computation for latency-sensitive NLP. The main contribution is the open, end-to-end reproducible framework - complete with scripts, CSV logging, and analysis utilities - intended to serve as a community tool for further research on adaptive transformers. 

---
# VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages 

**Authors**: Jesse Atuhurra, Iqra Ali, Tomoya Iwakura, Hidetaka Kamigaito, Tatsuya Hiraoka  

**Link**: [PDF](https://arxiv.org/pdf/2510.12845)  

**Abstract**: Vision Language Models (VLMs) are pivotal for advancing perception in intelligent agents. Yet, evaluation of VLMs remains limited to predominantly English-centric benchmarks in which the image-text pairs comprise short texts. To evaluate VLM fine-grained abilities, in four languages under long-text settings, we introduce a novel multilingual benchmark VLURes featuring eight vision-and-language tasks, and a pioneering unrelatedness task, to probe the fine-grained Visual and Linguistic Understanding capabilities of VLMs across English, Japanese, and low-resource languages, Swahili, and Urdu. Our datasets, curated from web resources in the target language, encompass ten diverse image categories and rich textual context, introducing valuable vision-language resources for Swahili and Urdu. By prompting VLMs to generate responses and rationales, evaluated automatically and by native speakers, we uncover performance disparities across languages and tasks critical to intelligent agents, such as object recognition, scene understanding, and relationship understanding. We conducted evaluations of ten VLMs with VLURes. The best performing model, GPT-4o, achieves an overall accuracy of 90.8% and lags human performance by 6.7%, though the gap is larger for open-source models. The gap highlights VLURes' critical role in developing intelligent agents to tackle multi-modal visual reasoning. 

---
# FaStFACT: Faster, Stronger Long-Form Factuality Evaluations in LLMs 

**Authors**: Yingjia Wan, Haochen Tan, Xiao Zhu, Xinyu Zhou, Zhiwei Li, Qingsong Lv, Changxuan Sun, Jiaqi Zeng, Yi Xu, Jianqiao Lu, Yinhong Liu, Zhijiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12839)  

**Abstract**: Evaluating the factuality of long-form generations from Large Language Models (LLMs) remains challenging due to accuracy issues and costly human assessment. Prior efforts attempt this by decomposing text into claims, searching for evidence, and verifying claims, but suffer from critical drawbacks: (1) inefficiency due to complex pipeline components unsuitable for long LLM outputs, and (2) ineffectiveness stemming from inaccurate claim sets and insufficient evidence collection of one-line snippets.
To address these limitations, we propose \name, a fast and strong evaluation framework that achieves the highest alignment with human evaluation and efficiency among existing baselines. \name first employs chunk-level claim extraction integrated with confidence-based pre-verification, significantly reducing the cost of web searching and inference calling while ensuring reliability. For searching and verification, it collects document-level evidence from crawled webpages and selectively retrieves it during verification, addressing the evidence insufficiency problem in previous pipelines.
Extensive experiments based on an aggregated and manually annotated benchmark demonstrate the reliability of \name in both efficiently and effectively evaluating the factuality of long-form LLM generations. Code and benchmark data is available at this https URL. 

---
# A\textsuperscript{2}FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning 

**Authors**: Qianben Chen, Jingyi Cao, Jiayu Zhang, Tianrui Qin, Xiaowan Li, King Zhu, Dingfeng Shi, He Zhu, Minghao Liu, Xiaobo Liang, Ge Zhang, Jian Yang, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.12838)  

**Abstract**: Large language models split into two families: reasoning-centric LLMs, which strengthen internal chain-of-thought reasoning but cannot invoke external tools, and agentic LLMs, which learn to interact with environments and leverage tools but often lag in deep reasoning. This divide arises from fundamentally different training objectives, leading to mismatched strengths and inefficiency on simple queries, where both families tend to overthink or over-call tools. In this work, we present Adaptive Agent Foundation Model (A\textsuperscript{2}FM), a unified framework that follows a route-then-align principle: the model first learns task-aware routing and then aligns mode-specific trajectories under a shared backbone. To address the inefficiency gap, we introduce a third mode-instant-that handles simple queries directly, preventing unnecessary reasoning or tool calls while complementing the agentic and reasoning modes. To jointly enhance accuracy and efficiency, we propose Adaptive Policy Optimization (APO), which enforces adaptive sampling across modes and applies a cost-regularized reward. On the 32B scale, A\textsuperscript{2}FM achieves 13.4\% on BrowseComp, 70.4\% on AIME25, and 16.7\% on HLE, setting new SOTA among comparable models and performing competitively with frontier LLMs across agentic, reasoning, and general benchmarks. Notably, the adaptive execution achieves a cost of pass of only \$0.00487 per correct answer-cutting cost by 45.2\% relative to reasoning and 33.5\% relative to agentic, thus delivering substantially higher cost efficiency while maintaining comparable accuracy. 

---
# Repurposing Annotation Guidelines to Instruct LLM Annotators: A Case Study 

**Authors**: Kon Woo Kim, Rezarta Islamaj, Jin-Dong Kim, Florian Boudin, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.12835)  

**Abstract**: This study investigates how existing annotation guidelines can be repurposed to instruct large language model (LLM) annotators for text annotation tasks. Traditional guidelines are written for human annotators who internalize training, while LLMs require explicit, structured instructions. We propose a moderation-oriented guideline repurposing method that transforms guidelines into clear directives for LLMs through an LLM moderation process. Using the NCBI Disease Corpus as a case study, our experiments show that repurposed guidelines can effectively guide LLM annotators, while revealing several practical challenges. The results highlight the potential of this workflow to support scalable and cost-effective refinement of annotation guidelines and automated annotation. 

---
# MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training 

**Authors**: Taicheng Guo, Hai Wang, ChaoChun Liu, Mohsen Golalikhani, Xin Chen, Xiangliang Zhang, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.12831)  

**Abstract**: Multi-turn Text-to-SQL aims to translate a user's conversational utterances into executable SQL while preserving dialogue coherence and grounding to the target schema. However, most existing systems only regard this task as a simple text translation task and follow a short-horizon paradigm, generating a query per turn without execution, explicit verification, and refinement, which leads to non-executable or incoherent outputs. We present MTSQL-R1, an agentic training framework for long-horizon multi-turn Text-to-SQL. We cast the task as a Markov Decision Process (MDP) in which an agent interacts with (i) a database for execution feedback and (ii) a persistent dialogue memory for coherence verification, performing an iterative propose to execute -> verify -> refine cycle until all checks pass. Experiments on COSQL and SPARC demonstrate that MTSQL-R1 consistently outperforms strong baselines, highlighting the importance of environment-driven verification and memory-guided refinement for conversational semantic parsing. Full recipes (including code, trained models, logs, reasoning trajectories, etc.) will be released after the internal review to contribute to community research. 

---
# Mathematics with large language models as provers and verifiers 

**Authors**: Hieu Le Duc, Leo Liberti  

**Link**: [PDF](https://arxiv.org/pdf/2510.12829)  

**Abstract**: During 2024 and 2025 the discussion about the theorem-proving capabilities of large language models started reporting interesting success stories, mostly to do with difficult exercises (such as problems from the International Mathematical Olympiad), but also with conjectures [Feldman & Karbasi, arXiv:2509.18383v1] formulated for the purpose of verifying whether the artificial intelligence could prove it. In this paper we report a theorem proving feat achieved by ChatGPT by using a protocol involving different prover and verifier instances of the gpt-5 model working collaboratively. To make sure that the produced proofs do not suffer from hallucinations, the final proof is formally verified by the lean proof assistant, and the conformance of premises and conclusion of the lean code is verified by a human. Our methodology was able to solve five out of six 2025 IMO problems, and close a third of the sixty-six number theory conjectures in [Cohen, Journal of Integer Sequences, 2025]. 

---
# Scheming Ability in LLM-to-LLM Strategic Interactions 

**Authors**: Thao Pham  

**Link**: [PDF](https://arxiv.org/pdf/2510.12826)  

**Abstract**: As large language model (LLM) agents are deployed autonomously in diverse contexts, evaluating their capacity for strategic deception becomes crucial. While recent research has examined how AI systems scheme against human developers, LLM-to-LLM scheming remains underexplored. We investigate the scheming ability and propensity of frontier LLM agents through two game-theoretic frameworks: a Cheap Talk signaling game and a Peer Evaluation adversarial game. Testing four models (GPT-4o, Gemini-2.5-pro, Claude-3.7-Sonnet, and Llama-3.3-70b), we measure scheming performance with and without explicit prompting while analyzing scheming tactics through chain-of-thought reasoning. When prompted, most models, especially Gemini-2.5-pro and Claude-3.7-Sonnet, achieved near-perfect performance. Critically, models exhibited significant scheming propensity without prompting: all models chose deception over confession in Peer Evaluation (100% rate), while models choosing to scheme in Cheap Talk succeeded at 95-100% rates. These findings highlight the need for robust evaluations using high-stakes game-theoretic scenarios in multi-agent settings. 

---
# Classifier-Augmented Generation for Structured Workflow Prediction 

**Authors**: Thomas Gschwind, Shramona Chakraborty, Nitin Gupta, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2510.12825)  

**Abstract**: ETL (Extract, Transform, Load) tools such as IBM DataStage allow users to visually assemble complex data workflows, but configuring stages and their properties remains time consuming and requires deep tool knowledge. We propose a system that translates natural language descriptions into executable workflows, automatically predicting both the structure and detailed configuration of the flow. At its core lies a Classifier-Augmented Generation (CAG) approach that combines utterance decomposition with a classifier and stage-specific few-shot prompting to produce accurate stage predictions. These stages are then connected into non-linear workflows using edge prediction, and stage properties are inferred from sub-utterance context. We compare CAG against strong single-prompt and agentic baselines, showing improved accuracy and efficiency, while substantially reducing token usage. Our architecture is modular, interpretable, and capable of end-to-end workflow generation, including robust validation steps. To our knowledge, this is the first system with a detailed evaluation across stage prediction, edge layout, and property generation for natural-language-driven ETL authoring. 

---
# MEDEQUALQA: Evaluating Biases in LLMs with Counterfactual Reasoning 

**Authors**: Rajarshi Ghosh, Abhay Gupta, Hudson McBride, Anurag Vaidya, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2510.12818)  

**Abstract**: Large language models (LLMs) are increasingly deployed in clinical decision support, yet subtle demographic cues can influence their reasoning. Prior work has documented disparities in outputs across patient groups, but little is known about how internal reasoning shifts under controlled demographic changes. We introduce MEDEQUALQA, a counterfactual benchmark that perturbs only patient pronouns (he/him, she/her, they/them) while holding critical symptoms and conditions (CSCs) constant. Each clinical vignette is expanded into single-CSC ablations, producing three parallel datasets of approximately 23,000 items each (69,000 total). We evaluate a GPT-4.1 model and compute Semantic Textual Similarity (STS) between reasoning traces to measure stability across pronoun variants. Our results show overall high similarity (mean STS >0.80), but reveal consistent localized divergences in cited risk factors, guideline anchors, and differential ordering, even when final diagnoses remain unchanged. Our error analysis highlights certain cases in which the reasoning shifts, underscoring clinically relevant bias loci that may cascade into inequitable care. MEDEQUALQA offers a controlled diagnostic setting for auditing reasoning stability in medical AI. 

---
# From Noise to Signal to Selbstzweck: Reframing Human Label Variation in the Era of Post-training in NLP 

**Authors**: Shanshan Xu, Santosh T.Y.S.S, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2510.12817)  

**Abstract**: Human Label Variation (HLV) refers to legitimate disagreement in annotation that reflects the genuine diversity of human perspectives rather than mere error. For decades, HLV in NLP was dismissed as noise to be discarded, and only slowly over the last decade has it been reframed as a signal for improving model robustness. With the rise of large language models (LLMs), where post-training on human feedback has become central to model alignment, the role of HLV has become increasingly consequential. Yet current preference-learning datasets routinely aggregate multiple annotations into a single label, thereby flattening diverse perspectives into a false universal agreement and erasing precisely the pluralism of human values that alignment aims to preserve. In this position paper, we argue that preserving HLV as an embodiment of human pluralism must be treated as a Selbstzweck - a goal it self when designing AI systems. We call for proactively incorporating HLV into preference datasets and outline actionable steps towards it. 

---
# Cancer Diagnosis Categorization in Electronic Health Records Using Large Language Models and BioBERT: Model Performance Evaluation Study 

**Authors**: Soheil Hashtarkhani, Rezaur Rashid, Christopher L Brett, Lokesh Chinthala, Fekede Asefa Kumsa, Janet A Zink, Robert L Davis, David L Schwartz, Arash Shaban-Nejad  

**Link**: [PDF](https://arxiv.org/pdf/2510.12813)  

**Abstract**: Electronic health records contain inconsistently structured or free-text data, requiring efficient preprocessing to enable predictive health care models. Although artificial intelligence-driven natural language processing tools show promise for automating diagnosis classification, their comparative performance and clinical reliability require systematic evaluation. The aim of this study is to evaluate the performance of 4 large language models (GPT-3.5, GPT-4o, Llama 3.2, and Gemini 1.5) and BioBERT in classifying cancer diagnoses from structured and unstructured electronic health records data. We analyzed 762 unique diagnoses (326 International Classification of Diseases (ICD) code descriptions, 436free-text entries) from 3456 records of patients with cancer. Models were tested on their ability to categorize diagnoses into 14predefined categories. Two oncology experts validated classifications. BioBERT achieved the highest weighted macro F1-score for ICD codes (84.2) and matched GPT-4o in ICD code accuracy (90.8). For free-text diagnoses, GPT-4o outperformed BioBERT in weighted macro F1-score (71.8 vs 61.5) and achieved slightly higher accuracy (81.9 vs 81.6). GPT-3.5, Gemini, and Llama showed lower overall performance on both formats. Common misclassification patterns included confusion between metastasis and central nervous system tumors, as well as errors involving ambiguous or overlapping clinical terminology. Although current performance levels appear sufficient for administrative and research use, reliable clinical applications will require standardized documentation practices alongside robust human oversight for high-stakes decision-making. 

---
# Benchmarking Open-Source Large Language Models for Persian in Zero-Shot and Few-Shot Learning 

**Authors**: Mahdi Cherakhloo, Arash Abbasi, Mohammad Saeid Sarafraz, Bijan Vosoughi Vahdat  

**Link**: [PDF](https://arxiv.org/pdf/2510.12807)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous languages; however, their effectiveness in low-resource languages like Persian requires thorough investigation. This paper presents a comprehensive benchmark of several open-source LLMs for Persian Natural Language Processing (NLP) tasks, utilizing both zero-shot and few-shot learning paradigms. We evaluate models across a range of tasks including sentiment analysis, named entity recognition, reading comprehension, and question answering, using established Persian datasets such as ParsiNLU and ArmanEmo. Our methodology encompasses rigorous experimental setups for both zero-shot and few-shot scenarios, employing metrics such as Accuracy, F1-score, BLEU, and ROUGE for performance evaluation. The results reveal that Gemma 2 consistently outperforms other models across nearly all tasks in both learning paradigms, with particularly strong performance in complex reasoning tasks. However, most models struggle with token-level understanding tasks like Named Entity Recognition, highlighting specific challenges in Persian language processing. This study contributes to the growing body of research on multilingual LLMs, providing valuable insights into their performance in Persian and offering a benchmark for future model development. 

---
# Generative Universal Verifier as Multimodal Meta-Reasoner 

**Authors**: Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13804)  

**Abstract**: We introduce Generative Universal Verifier, a novel concept and plugin designed for next-generation multimodal reasoning in vision-language models and unified multimodal models, providing the fundamental capability of reflection and refinement on visual outcomes during the reasoning and generation process. This work makes three main contributions: (1) We build ViVerBench, a comprehensive benchmark spanning 16 categories of critical tasks for evaluating visual outcomes in multimodal reasoning. Results show that existing VLMs consistently underperform across these tasks, underscoring a substantial gap from human-level capability in reliable visual verification. (2) We design two automated pipelines to construct large-scale visual verification data and train OmniVerifier-7B, the first omni-capable generative verifier trained for universal visual verification and achieves notable gains on ViVerBench(+8.3). Through training, we identify three atomic capabilities in visual verification and demonstrate how they generalize and interact synergistically. (3) We propose OmniVerifier-TTS, a sequential test-time scaling paradigm that leverages the universal verifier to bridge image generation and editing within unified models, enhancing the upper bound of generative ability through iterative fine-grained optimization. Beyond generation, we extend universal verifier to broader world-modeling interleaved reasoning scenarios. Empirically, OmniVerifier-TTS achieves improvements on T2I-ReasonBench(+3.7), and GenEval++(+4.3), outperforming existing parallel test-time scaling methods, such as Best-of-N. By endowing multimodal reasoning with reliable visual verification, OmniVerifier advances both reliable reflection during generation and scalable test-time refinement, marking a step toward more trustworthy and controllable next-generation reasoning systems. 

---
# Hard2Verify: A Step-Level Verification Benchmark for Open-Ended Frontier Math 

**Authors**: Shrey Pandit, Austin Xu, Xuan-Phi Nguyen, Yifei Ming, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.13744)  

**Abstract**: Large language model (LLM)-based reasoning systems have recently achieved gold medal-level performance in the IMO 2025 competition, writing mathematical proofs where, to receive full credit, each step must be not only correct but also sufficiently supported. To train LLM-based reasoners in such challenging, open-ended settings, strong verifiers capable of catching step-level mistakes are necessary prerequisites. We introduce Hard2Verify, a human-annotated, step-level verification benchmark produced with over 500 hours of human labor. Hard2Verify is designed to rigorously assess step-level verifiers at the frontier: Verifiers must provide step-level annotations or identify the first error in responses generated by frontier LLMs for very recent, challenging, and open-ended math questions. We evaluate 29 generative critics and process reward models, demonstrating that, beyond a few standouts, open-source verifiers lag closed source models. We subsequently analyze what drives poor performance in step-level verification, the impacts of scaling verifier compute, as well as fundamental questions such as self-verification and verification-generation dynamics. 

---
# LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models 

**Authors**: Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13626)  

**Abstract**: Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation. 

---
# K-Merge: Online Continual Merging of Adapters for On-device Large Language Models 

**Authors**: Donald Shenaj, Ondrej Bohdal, Taha Ceritli, Mete Ozay, Pietro Zanuttigh, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2510.13537)  

**Abstract**: On-device deployment of Large Language Models (LLMs) frequently leverages Low-Rank Adapters (LoRAs) to support diverse downstream tasks under tight resource constraints. To address the limited storage capacity of mobile devices, recent works have explored model merging techniques to fuse multiple LoRAs into a single one. In practice, however, LoRAs are often delivered incrementally, as users request support for new tasks (e.g., novel problem types or languages). This scenario introduces a new challenge: on-device online continual merging, where the objective is to incorporate new LoRAs while preserving the performance on previously supported tasks. In this paper, we propose a data-free and computationally efficient strategy for selecting and merging LoRAs when a new one becomes available, assuming the device can store only a limited number of adapters. Extensive experiments across real-world tasks demonstrate the superiority of our approach compared to alternative strategies while adhering to the storage budget and compute limitations of on-device settings. 

---
# Assessing LLM Reasoning Through Implicit Causal Chain Discovery in Climate Discourse 

**Authors**: Liesbeth Allein, Nataly Pineda-Castañeda, Andrea Rocci, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2510.13417)  

**Abstract**: How does a cause lead to an effect, and which intermediate causal steps explain their connection? This work scrutinizes the mechanistic causal reasoning capabilities of large language models (LLMs) to answer these questions through the task of implicit causal chain discovery. In a diagnostic evaluation framework, we instruct nine LLMs to generate all possible intermediate causal steps linking given cause-effect pairs in causal chain structures. These pairs are drawn from recent resources in argumentation studies featuring polarized discussion on climate change. Our analysis reveals that LLMs vary in the number and granularity of causal steps they produce. Although they are generally self-consistent and confident about the intermediate causal connections in the generated chains, their judgments are mainly driven by associative pattern matching rather than genuine causal reasoning. Nonetheless, human evaluations confirmed the logical coherence and integrity of the generated chains. Our baseline causal chain discovery approach, insights from our diagnostic evaluation, and benchmark dataset with causal chains lay a solid foundation for advancing future work in implicit, mechanistic causal reasoning in argumentation settings. 

---
# UniMoE-Audio: Unified Speech and Music Generation with Dynamic-Capacity MoE 

**Authors**: Zhenyu Liu, Yunxin Li, Xuanyu Zhang, Qixun Teng, Shenyuan Jiang, Xinyu Chen, Haoyuan Shi, Jinchao Li, Qi Wang, Haolan Chen, Fanbo Meng, Mingjun Zhao, Yu Xu, Yancheng He, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13344)  

**Abstract**: Recent advances in unified multimodal models indicate a clear trend towards comprehensive content generation. However, the auditory domain remains a significant challenge, with music and speech often developed in isolation, hindering progress towards universal audio synthesis. This separation stems from inherent task conflicts and severe data imbalances, which impede the development of a truly unified audio generation model. To address this challenge, we propose UniMoE-Audio, a unified speech and music generation model within a novel Dynamic-Capacity Mixture-of-Experts (MoE) framework. Architecturally, UniMoE-Audio introduces a Top-P routing strategy for dynamic expert number allocation, and a hybrid expert design comprising routed experts for domain-specific knowledge, shared experts for domain-agnostic features, and null experts for adaptive computation skipping. To tackle data imbalance, we introduce a three-stage training curriculum: 1) Independent Specialist Training leverages original datasets to instill domain-specific knowledge into each "proto-expert" without interference; 2) MoE Integration and Warmup incorporates these specialists into the UniMoE-Audio architecture, warming up the gate module and shared expert using a subset of balanced dataset; and 3) Synergistic Joint Training trains the entire model end-to-end on the fully balanced dataset, fostering enhanced cross-domain synergy. Extensive experiments show that UniMoE-Audio not only achieves state-of-the-art performance on major speech and music generation benchmarks, but also demonstrates superior synergistic learning, mitigating the performance degradation typically seen in naive joint training. Our findings highlight the substantial potential of specialized MoE architecture and curated training strategies in advancing the field of universal audio generation. Homepage: this https URL 

---
# Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses 

**Authors**: Sungnyun Kim, Kangwook Jang, Sungwoo Cho, Joon Son Chung, Hoirin Kim, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2510.13281)  

**Abstract**: This paper introduces a new paradigm for generative error correction (GER) framework in audio-visual speech recognition (AVSR) that reasons over modality-specific evidences directly in the language space. Our framework, DualHyp, empowers a large language model (LLM) to compose independent N-best hypotheses from separate automatic speech recognition (ASR) and visual speech recognition (VSR) models. To maximize the effectiveness of DualHyp, we further introduce RelPrompt, a noise-aware guidance mechanism that provides modality-grounded prompts to the LLM. RelPrompt offers the temporal reliability of each modality stream, guiding the model to dynamically switch its focus between ASR and VSR hypotheses for an accurate correction. Under various corruption scenarios, our framework attains up to 57.7% error rate gain on the LRS2 benchmark over standard ASR baseline, contrary to single-stream GER approaches that achieve only 10% gain. To facilitate research within our DualHyp framework, we release the code and the dataset comprising ASR and VSR hypotheses at this https URL. 

---
# MMLongCite: A Benchmark for Evaluating Fidelity of Long-Context Vision-Language Models 

**Authors**: Keyan Zhou, Zecheng Tang, Lingfeng Ming, Guanghao Zhou, Qiguang Chen, Dan Qiao, Zheming Yang, Libo Qin, Minghui Qiu, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13276)  

**Abstract**: The rapid advancement of large vision language models (LVLMs) has led to a significant expansion of their context windows. However, an extended context window does not guarantee the effective utilization of the context, posing a critical challenge for real-world applications. Current evaluations of such long-context faithfulness are predominantly focused on the text-only domain, while multimodal assessments remain limited to short contexts. To bridge this gap, we introduce MMLongCite, a comprehensive benchmark designed to evaluate the fidelity of LVLMs in long-context scenarios. MMLongCite comprises 8 distinct tasks spanning 6 context length intervals and incorporates diverse modalities, including text, images, and videos. Our evaluation of state-of-the-art LVLMs reveals their limited faithfulness in handling long multimodal contexts. Furthermore, we provide an in-depth analysis of how context length and the position of crucial content affect the faithfulness of these models. 

---
# EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic Systems 

**Authors**: Yufei He, Juncheng Liu, Yue Liu, Yibo Li, Tri Cao, Zhiyuan Hu, Xinxing Xu, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13220)  

**Abstract**: A fundamental limitation of current AI agents is their inability to learn complex skills on the fly at test time, often behaving like "clever but clueless interns" in novel environments. This severely limits their practical utility. To systematically measure and drive progress on this challenge, we first introduce the Jericho Test-Time Learning (J-TTL) benchmark. J-TTL is a new evaluation setup where an agent must play the same game for several consecutive episodes, attempting to improve its performance from one episode to the next. On J-TTL, we find that existing adaptation methods like reflection, memory, or reinforcement learning struggle. To address the challenges posed by our benchmark, we present EvoTest, an evolutionary test-time learning framework that improves an agent without any fine-tuning or gradients-by evolving the entire agentic system after every episode. EvoTest has two roles: the Actor Agent, which plays the game, and the Evolver Agent, which analyzes the episode transcript to propose a revised configuration for the next run. This configuration rewrites the prompt, updates memory by logging effective state-action choices, tunes hyperparameters, and learns the tool-use routines. On our J-TTL benchmark, EvoTest consistently increases performance, outperforming not only reflection and memory-only baselines but also more complex online fine-tuning methods. Notably, our method is the only one capable of winning two games (Detective and Library), while all baselines fail to win any. 

---
# Personalized Learning Path Planning with Goal-Driven Learner State Modeling 

**Authors**: Joy Jia Yin Lim, Ye He, Jifan Yu, Xin Cong, Daniel Zhang-Li, Zhiyuan Liu, Huiqin Liu, Lei Hou, Juanzi Li, Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13215)  

**Abstract**: Personalized Learning Path Planning (PLPP) aims to design adaptive learning paths that align with individual goals. While large language models (LLMs) show potential in personalizing learning experiences, existing approaches often lack mechanisms for goal-aligned planning. We introduce Pxplore, a novel framework for PLPP that integrates a reinforcement-based training paradigm and an LLM-driven educational architecture. We design a structured learner state model and an automated reward function that transforms abstract objectives into computable signals. We train the policy combining supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), and deploy it within a real-world learning platform. Extensive experiments validate Pxplore's effectiveness in producing coherent, personalized, and goal-driven learning paths. We release our code and dataset to facilitate future research. 

---
# Program of Thoughts for Financial Reasoning: Leveraging Dynamic In-Context Examples and Generative Retrieval 

**Authors**: Subhendu Khatuya, Shashwat Naidu, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2510.13157)  

**Abstract**: Despite continuous advancements in the capabilities of large language models (LLMs), numerical reasoning remains a challenging area. Techniques like chain-of-thought prompting, tree-of-thought prompting, and program-of-thought prompting guide LLMs through intermediate reasoning steps. Although in-context learning with few-shot prompting has improved performance, LLMs still lag behind state-of-the-art models on financial numerical reasoning datasets such as FinQA and ConvFinQA. In this work, we introduce FINDER, a novel two-step framework, to enhance LLMs' capabilities in financial numerical reasoning. The first step utilizes a generative retriever to extract relevant facts from unstructured data, including both text and tables. This is followed by context-aware Program of Thought prompting with dynamic selection of in-context examples. Our model FINDER achieves a new state-of-the-art performance on both the FinQA and ConvFinQA datasets, surpassing previous benchmarks with execution accuracy improvements of 5.98% and 4.05%, respectively. 

---
# Addressing the alignment problem in transportation policy making: an LLM approach 

**Authors**: Xiaoyu Yan, Tianxing Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.13139)  

**Abstract**: A key challenge in transportation planning is that the collective preferences of heterogeneous travelers often diverge from the policies produced by model-driven decision tools. This misalignment frequently results in implementation delays or failures. Here, we investigate whether large language models (LLMs), noted for their capabilities in reasoning and simulating human decision-making, can help inform and address this alignment problem. We develop a multi-agent simulation in which LLMs, acting as agents representing residents from different communities in a city, participate in a referendum on a set of transit policy proposals. Using chain-of-thought reasoning, LLM agents provide ranked-choice or approval-based preferences, which are aggregated using instant-runoff voting (IRV) to model democratic consensus. We implement this simulation framework with both GPT-4o and Claude-3.5, and apply it for Chicago and Houston. Our findings suggest that LLM agents are capable of approximating plausible collective preferences and responding to local context, while also displaying model-specific behavioral biases and modest divergences from optimization-based benchmarks. These capabilities underscore both the promise and limitations of LLMs as tools for solving the alignment problem in transportation decision-making. 

---
# On the Reasoning Abilities of Masked Diffusion Language Models 

**Authors**: Anej Svete, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.13117)  

**Abstract**: Masked diffusion models (MDMs) for text offer a compelling alternative to traditional autoregressive language models. Parallel generation makes them efficient, but their computational capabilities and the limitations inherent to their parallelism remain largely unexplored. To this end, we characterize what types of reasoning problems MDMs can provably solve and how efficiently. We do this by connecting MDMs to the well-understood reasoning frameworks of chain of thought (CoT) and padded looped transformers (PLTs) in the finite-precision log-width setting: We show that MDMs and polynomially-padded PLTs are, in fact, equivalent in this setting, and that MDMs can solve all problems that CoT-augmented transformers can. Moreover, we showcase classes of problems (including regular languages) for which MDMs are inherently more efficient than CoT transformers, where parallel generation allows for substantially faster reasoning. 

---
# TRUSTVIS: A Multi-Dimensional Trustworthiness Evaluation Framework for Large Language Models 

**Authors**: Ruoyu Sun, Da Song, Jiayang Song, Yuheng Huang, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13106)  

**Abstract**: As Large Language Models (LLMs) continue to revolutionize Natural Language Processing (NLP) applications, critical concerns about their trustworthiness persist, particularly in safety and robustness. To address these challenges, we introduce TRUSTVIS, an automated evaluation framework that provides a comprehensive assessment of LLM trustworthiness. A key feature of our framework is its interactive user interface, designed to offer intuitive visualizations of trustworthiness metrics. By integrating well-known perturbation methods like AutoDAN and employing majority voting across various evaluation methods, TRUSTVIS not only provides reliable results but also makes complex evaluation processes accessible to users. Preliminary case studies on models like Vicuna-7b, Llama2-7b, and GPT-3.5 demonstrate the effectiveness of our framework in identifying safety and robustness vulnerabilities, while the interactive interface allows users to explore results in detail, empowering targeted model improvements. Video Link: this https URL 

---
# Max It or Miss It: Benchmarking LLM On Solving Extremal Problems 

**Authors**: Binxin Gao, Jingjun Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.12997)  

**Abstract**: Test-time scaling has enabled Large Language Models (LLMs) with remarkable reasoning capabilities, particularly in mathematical domains, through intermediate chain-of-thought (CoT) reasoning before generating final answers. However, the specific sources and mechanisms underlying these reasoning capabilities remain insufficiently understood. Optimization reasoning, i.e. finding extrema under constraints, represents a fundamental abstraction that underpins critical applications in planning, control, resource allocation, and prompt search. To systematically evaluate this capability, we introduce ExtremBench, a benchmark dataset for solving mathematical extremal problems, curated from inequality exercises used for Chinese Mathematical Olympiad and transformed into $93$ standardized extrema-finding problems. We conduct extensive evaluations across various state-of-the-art open-source model families, including the Qwen3, GPT-OSS, and DeepSeek. Our results reveal that LLMs' extremal-solving reasoning capabilities do not always align with those of current mathematical benchmarks such as AIME25 and MATH-500, with some models showing strong general mathematical reasoning but poor extremal-solving skills, and vice versa. This discrepancy highlights a critical gap in current evaluation practices and suggests that existing benchmarks may not comprehensively capture the full spectrum of mathematical reasoning abilities. 

---
# UNCAP: Uncertainty-Guided Planning Using Natural Language Communication for Cooperative Autonomous Vehicles 

**Authors**: Neel P. Bhatt, Po-han Li, Kushagra Gupta, Rohan Siva, Daniel Milan, Alexander T. Hogue, Sandeep P. Chinchali, David Fridovich-Keil, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12992)  

**Abstract**: Safe large-scale coordination of multiple cooperative connected autonomous vehicles (CAVs) hinges on communication that is both efficient and interpretable. Existing approaches either rely on transmitting high-bandwidth raw sensor data streams or neglect perception and planning uncertainties inherent in shared data, resulting in systems that are neither scalable nor safe. To address these limitations, we propose Uncertainty-Guided Natural Language Cooperative Autonomous Planning (UNCAP), a vision-language model-based planning approach that enables CAVs to communicate via lightweight natural language messages while explicitly accounting for perception uncertainty in decision-making. UNCAP features a two-stage communication protocol: (i) an ego CAV first identifies the subset of vehicles most relevant for information exchange, and (ii) the selected CAVs then transmit messages that quantitatively express their perception uncertainty. By selectively fusing messages that maximize mutual information, this strategy allows the ego vehicle to integrate only the most relevant signals into its decision-making, improving both the scalability and reliability of cooperative planning. Experiments across diverse driving scenarios show a 63% reduction in communication bandwidth with a 31% increase in driving safety score, a 61% reduction in decision uncertainty, and a four-fold increase in collision distance margin during near-miss events. Project website: this https URL 

---
# DeepPlanner: Scaling Planning Capability for Deep Research Agents via Advantage Shaping 

**Authors**: Wei Fan, Wenlin Yao, Zheng Li, Feng Yao, Xin Liu, Liang Qiu, Qingyu Yin, Yangqiu Song, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.12979)  

**Abstract**: Large language models (LLMs) augmented with multi-step reasoning and action generation abilities have shown promise in leveraging external tools to tackle complex tasks that require long-horizon planning. However, existing approaches either rely on implicit planning in the reasoning stage or introduce explicit planners without systematically addressing how to optimize the planning stage. As evidence, we observe that under vanilla reinforcement learning (RL), planning tokens exhibit significantly higher entropy than other action tokens, revealing uncertain decision points that remain under-optimized. To address this, we propose DeepPlanner, an end-to-end RL framework that effectively enhances the planning capabilities of deep research agents. Our approach shapes token-level advantage with an entropy-based term to allocate larger updates to high entropy tokens, and selectively upweights sample-level advantages for planning-intensive rollouts. Extensive experiments across seven deep research benchmarks demonstrate that DeepPlanner improves planning quality and achieves state-of-the-art results under a substantially lower training budget. 

---
# Unifying Vision-Language Latents for Zero-label Image Caption Enhancement 

**Authors**: Sanghyun Byun, Jung Ick Guack, Mohanad Odema, Baisub Lee, Jacob Song, Woo Seong Chung  

**Link**: [PDF](https://arxiv.org/pdf/2510.12931)  

**Abstract**: Vision-language models (VLMs) achieve remarkable performance through large-scale image-text pretraining. However, their reliance on labeled image datasets limits scalability and leaves vast amounts of unlabeled image data underutilized. To address this, we propose Unified Vision-Language Alignment for Zero-Label Enhancement (ViZer), an enhancement training framework that enables zero-label learning in image captioning, providing a practical starting point for broader zero-label adaptation in vision-language tasks. Unlike prior approaches that rely on human or synthetically annotated datasets, ViZer actively aligns vision and language representation features during training, enabling existing VLMs to generate improved captions without requiring text labels or full retraining. We demonstrate ViZer's advantage in qualitative evaluation, as automated caption metrics such as CIDEr and BERTScore often penalize details that are absent in reference captions. Applying ViZer on SmolVLM-Base and Qwen2-VL, we observe consistent qualitative improvements, producing captions that are more grounded and descriptive than their baseline. 

---
# Toward LLM-Supported Automated Assessment of Critical Thinking Subskills 

**Authors**: Marisa C. Peczuh, Nischal Ashok Kumar, Ryan Baker, Blair Lehman, Danielle Eisenberg, Caitlin Mills, Keerthi Chebrolu, Sudhip Nashi, Cadence Young, Brayden Liu, Sherry Lachman, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12915)  

**Abstract**: Critical thinking represents a fundamental competency in today's education landscape. Developing critical thinking skills through timely assessment and feedback is crucial; however, there has not been extensive work in the learning analytics community on defining, measuring, and supporting critical thinking. In this paper, we investigate the feasibility of measuring core "subskills" that underlie critical thinking. We ground our work in an authentic task where students operationalize critical thinking: student-written argumentative essays. We developed a coding rubric based on an established skills progression and completed human coding for a corpus of student essays. We then evaluated three distinct approaches to automated scoring: zero-shot prompting, few-shot prompting, and supervised fine-tuning, implemented across three large language models (GPT-5, GPT-5-mini, and ModernBERT). GPT-5 with few-shot prompting achieved the strongest results and demonstrated particular strength on subskills with separable, frequent categories, while lower performance was observed for subskills that required detection of subtle distinctions or rare categories. Our results underscore critical trade-offs in automated critical thinking assessment: proprietary models offer superior reliability at higher cost, while open-source alternatives provide practical accuracy with reduced sensitivity to minority categories. Our work represents an initial step toward scalable assessment of higher-order reasoning skills across authentic educational contexts. 

---
# From Literal to Liberal: A Meta-Prompting Framework for Eliciting Human-Aligned Exception Handling in Large Language Models 

**Authors**: Imran Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12864)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed as the reasoning engines for agentic AI systems, yet they exhibit a critical flaw: a rigid adherence to explicit rules that leads to decisions misaligned with human common sense and intent. This "rule-rigidity" is a significant barrier to building trustworthy autonomous agents. While prior work has shown that supervised fine-tuning (SFT) with human explanations can mitigate this issue, SFT is computationally expensive and inaccessible to many practitioners. To address this gap, we introduce the Rule-Intent Distinction (RID) Framework, a novel, low-compute meta-prompting technique designed to elicit human-aligned exception handling in LLMs in a zero-shot manner. The RID framework provides the model with a structured cognitive schema for deconstructing tasks, classifying rules, weighing conflicting outcomes, and justifying its final decision. We evaluated the RID framework against baseline and Chain-of-Thought (CoT) prompting on a custom benchmark of 20 scenarios requiring nuanced judgment across diverse domains. Our human-verified results demonstrate that the RID framework significantly improves performance, achieving a 95% Human Alignment Score (HAS), compared to 80% for the baseline and 75% for CoT. Furthermore, it consistently produces higher-quality, intent-driven reasoning. This work presents a practical, accessible, and effective method for steering LLMs from literal instruction-following to liberal, goal-oriented reasoning, paving the way for more reliable and pragmatic AI agents. 

---
# AutoCode: LLMs as Problem Setters for Competitive Programming 

**Authors**: Shang Zhou, Zihan Zheng, Kaiyuan Liu, Zeyu Shen, Zerui Cheng, Zexing Chen, Hansen He, Jianzhu Yao, Huanzhi Mao, Qiuyang Mang, Tianfu Fu, Beichen Li, Dongruixuan Li, Wenhao Chai, Zhuang Liu, Aleksandra Korolova, Peter Henderson, Natasha Jaques, Pramod Viswanath, Saining Xie, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12803)  

**Abstract**: Writing competitive programming problems is exacting. Authors must: set constraints, input distributions, and edge cases that rule out shortcuts; target specific algorithms (e.g., max-flow, dynamic programming, data structures); and calibrate complexity beyond the reach of most competitors. We argue that this makes for an ideal test of general large language model capabilities and study whether they can do this reliably. We introduce AutoCode, which uses multiple rounds of validation to yield competition-grade problem statements and test cases. On held-out problems, AutoCode test suites approach 99% consistency with official judgments, a significant improvement over current state-of-the-art methods like HardTests, which achieve less than 81%. Furthermore, starting with a random seed problem, AutoCode can create novel variants with reference and brute-force solutions. By cross-verifying these generated solutions against test cases, we can further filter out malformed problems. Our system ensures high correctness, as verified by human experts. AutoCode successfully produces novel problems judged by Grandmaster-level (top 0.3%) competitive programmers to be of contest quality. 

---
