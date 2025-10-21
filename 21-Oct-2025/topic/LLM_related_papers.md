# Executable Knowledge Graphs for Replicating AI Research 

**Authors**: Yujie Luo, Zhuoyun Yu, Xuehai Wang, Yuqi Zhu, Ningyu Zhang, Lanning Wei, Lun Du, Da Zheng, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.17795)  

**Abstract**: Replicating AI research is a crucial yet challenging task for large language model (LLM) agents. Existing approaches often struggle to generate executable code, primarily due to insufficient background knowledge and the limitations of retrieval-augmented generation (RAG) methods, which fail to capture latent technical details hidden in referenced papers. Furthermore, previous approaches tend to overlook valuable implementation-level code signals and lack structured knowledge representations that support multi-granular retrieval and reuse. To overcome these challenges, we propose Executable Knowledge Graphs (xKG), a modular and pluggable knowledge base that automatically integrates technical insights, code snippets, and domain-specific knowledge extracted from scientific literature. When integrated into three agent frameworks with two different LLMs, xKG shows substantial performance gains (10.9% with o3-mini) on PaperBench, demonstrating its effectiveness as a general and extensible solution for automated AI research replication. Code will released at this https URL. 

---
# AcademicEval: Live Long-Context LLM Benchmark 

**Authors**: Haozhen Zhang, Tao Feng, Pengrui Han, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.17725)  

**Abstract**: Large Language Models (LLMs) have recently achieved remarkable performance in long-context understanding. However, current long-context LLM benchmarks are limited by rigid context length, labor-intensive annotation, and the pressing challenge of label leakage issues during LLM training. Therefore, we propose \textsc{AcademicEval}, a live benchmark for evaluating LLMs over long-context generation tasks. \textsc{AcademicEval} adopts papers on arXiv to introduce several academic writing tasks with long-context inputs, \textit{i.e.}, \textsc{Title}, \textsc{Abstract}, \textsc{Introduction}, and \textsc{Related Work}, which cover a wide range of abstraction levels and require no manual labeling. Moreover, \textsc{AcademicEval} integrates high-quality and expert-curated few-shot demonstrations from a collected co-author graph to enable flexible context length. Especially, \textsc{AcademicEval} features an efficient live evaluation, ensuring no label leakage. We conduct a holistic evaluation on \textsc{AcademicEval}, and the results illustrate that LLMs perform poorly on tasks with hierarchical abstraction levels and tend to struggle with long few-shot demonstrations, highlighting the challenge of our benchmark. Through experimental analysis, we also reveal some insights for enhancing LLMs' long-context modeling capabilities. Code is available at this https URL 

---
# PANER: A Paraphrase-Augmented Framework for Low-Resource Named Entity Recognition 

**Authors**: Nanda Kumar Rengarajan, Jun Yan, Chun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17720)  

**Abstract**: Named Entity Recognition (NER) is a critical task that requires substantial annotated data, making it challenging in low-resource scenarios where label acquisition is expensive. While zero-shot and instruction-tuned approaches have made progress, they often fail to generalize to domain-specific entities and do not effectively utilize limited available data. We present a lightweight few-shot NER framework that addresses these challenges through two key innovations: (1) a new instruction tuning template with a simplified output format that combines principles from prior IT approaches to leverage the large context window of recent state-of-the-art LLMs; (2) introducing a strategic data augmentation technique that preserves entity information while paraphrasing the surrounding context, thereby expanding our training data without compromising semantic relationships. Experiments on benchmark datasets show that our method achieves performance comparable to state-of-the-art models on few-shot and zero-shot tasks, with our few-shot approach attaining an average F1 score of 80.1 on the CrossNER datasets. Models trained with our paraphrasing approach show consistent improvements in F1 scores of up to 17 points over baseline versions, offering a promising solution for groups with limited NER training data and compute power. 

---
# QueST: Incentivizing LLMs to Generate Difficult Problems 

**Authors**: Hanxu Hu, Xingxing Zhang, Jannis Vamvas, Rico Sennrich, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.17715)  

**Abstract**: Large Language Models have achieved strong performance on reasoning tasks, solving competition-level coding and math problems. However, their scalability is limited by human-labeled datasets and the lack of large-scale, challenging coding problem training data. Existing competitive coding datasets contain only thousands to tens of thousands of problems. Previous synthetic data generation methods rely on either augmenting existing instruction datasets or selecting challenging problems from human-labeled data. In this paper, we propose QueST, a novel framework which combines difficulty-aware graph sampling and difficulty-aware rejection fine-tuning that directly optimizes specialized generators to create challenging coding problems. Our trained generators demonstrate superior capability compared to even GPT-4o at creating challenging problems that benefit downstream performance. We leverage QueST to generate large-scale synthetic coding problems, which we then use to distill from strong teacher models with long chain-of-thought or to conduct reinforcement learning for smaller models, proving effective in both scenarios. Our distillation experiments demonstrate significant performance gains. Specifically, after fine-tuning Qwen3-8B-base on 100K difficult problems generated by QueST, we surpass the performance of the original Qwen3-8B on LiveCodeBench. With an additional 112K examples (i.e., 28K human-written problems paired with multiple synthetic solutions), our 8B model matches the performance of the much larger DeepSeek-R1-671B. These findings indicate that generating complex problems via QueST offers an effective and scalable approach to advancing the frontiers of competitive coding and reasoning for large language models. 

---
# Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation 

**Authors**: Collin Zhang, Fei Huang, Chenhan Yuan, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17555)  

**Abstract**: Large language models (LLMs) often experience language confusion, which is the unintended mixing of languages during text generation. Current solutions to this problem either necessitate model retraining or cannot differentiate between harmful confusion and acceptable code-switching. This paper introduces the Language Confusion Gate (LCG), a lightweight, plug-in solution that filters tokens during decoding without altering the base LLM. The LCG is trained using norm-adjusted self-distillation to predict appropriate language families and apply masking only when needed. Our method is based on the findings that language confusion is infrequent, correct-language tokens are usually among the top predictions, and output token embedding norms are larger for high-resource languages, which biases sampling. When evaluated across various models, including Qwen3, GPT-OSS, Gemma3, Llama3.1, LCG decreases language confusion significantly, often by an order of magnitude, without negatively impacting task performance. Code is available at this https URL. 

---
# OncoReason: Structuring Clinical Reasoning in LLMs for Robust and Interpretable Survival Prediction 

**Authors**: Raghu Vamshi Hemadri, Geetha Krishna Guruju, Kristi Topollai, Anna Ewa Choromanska  

**Link**: [PDF](https://arxiv.org/pdf/2510.17532)  

**Abstract**: Predicting cancer treatment outcomes requires models that are both accurate and interpretable, particularly in the presence of heterogeneous clinical data. While large language models (LLMs) have shown strong performance in biomedical NLP, they often lack structured reasoning capabilities critical for high-stakes decision support. We present a unified, multi-task learning framework that aligns autoregressive LLMs with clinical reasoning for outcome prediction on the MSK-CHORD dataset. Our models are trained to jointly perform binary survival classification, continuous survival time regression, and natural language rationale generation. We evaluate three alignment strategies: (1) standard supervised fine-tuning (SFT), (2) SFT with Chain-of-Thought (CoT) prompting to elicit step-by-step reasoning, and (3) Group Relative Policy Optimization (GRPO), a reinforcement learning method that aligns model outputs to expert-derived reasoning trajectories. Experiments with LLaMa3-8B and Med42-8B backbones demonstrate that CoT prompting improves F1 by +6.0 and reduces MAE by 12%, while GRPO achieves state-of-the-art interpretability and predictive performance across BLEU, ROUGE, and BERTScore. We further show that existing biomedical LLMs often fail to produce valid reasoning traces due to architectural constraints. Our findings underscore the importance of reasoning-aware alignment in multi-task clinical modeling and set a new benchmark for interpretable, trustworthy LLMs in precision oncology. 

---
# Towards Mining Effective Pedagogical Strategies from Learner-LLM Educational Dialogues 

**Authors**: Liqun He, Manolis Mavrikis, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2510.17698)  

**Abstract**: Dialogue plays a crucial role in educational settings, yet existing evaluation methods for educational applications of large language models (LLMs) primarily focus on technical performance or learning outcomes, often neglecting attention to learner-LLM interactions. To narrow this gap, this AIED Doctoral Consortium paper presents an ongoing study employing a dialogue analysis approach to identify effective pedagogical strategies from learner-LLM dialogues. The proposed approach involves dialogue data collection, dialogue act (DA) annotation, DA pattern mining, and predictive model building. Early insights are outlined as an initial step toward future research. The work underscores the need to evaluate LLM-based educational applications by focusing on dialogue dynamics and pedagogical strategies. 

---
# SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors 

**Authors**: Tiancheng Hu, Joachim Baumann, Lorenzo Lupo, Dirk Hovy, Nigel Collier, Paul Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2510.17516)  

**Abstract**: Large language model (LLM) simulations of human behavior have the potential to revolutionize the social and behavioral sciences, if and only if they faithfully reflect real human behaviors. Current evaluations are fragmented, based on bespoke tasks and metrics, creating a patchwork of incomparable results. To address this, we introduce SimBench, the first large-scale, standardized benchmark for a robust, reproducible science of LLM simulation. By unifying 20 diverse datasets covering tasks from moral decision-making to economic choice across a large global participant pool, SimBench provides the necessary foundation to ask fundamental questions about when, how, and why LLM simulations succeed or fail. We show that, while even the best LLMs today have limited simulation ability (score: 40.80/100), performance scales log-linearly with model size. Simulation performance is not improved by increased inference-time compute. We demonstrate an alignment-simulation trade-off: instruction-tuning improves performance on low-entropy (consensus) questions but degrades it on high-entropy (diverse) ones. Models particularly struggle when simulating specific demographic groups. Finally, we demonstrate that simulation ability correlates most strongly with deep, knowledge-intensive reasoning (MMLU-Pro, r=0.939). By making progress measurable, we aim to accelerate the development of more faithful LLM simulators. 

---
# Deep Self-Evolving Reasoning 

**Authors**: Zihan Liu, Shun Zheng, Xumeng Wen, Yang Wang, Jiang Bian, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17498)  

**Abstract**: Long-form chain-of-thought reasoning has become a cornerstone of advanced reasoning in large language models. While recent verification-refinement frameworks have enabled proprietary models to solve Olympiad-level problems, their effectiveness hinges on strong, reliable verification and correction capabilities, which remain fragile in open-weight, smaller-scale models. This work demonstrates that even with weak verification and refinement capabilities on hard tasks, the reasoning limits of such models can be substantially extended through a probabilistic paradigm we call Deep Self-Evolving Reasoning (DSER). We conceptualize iterative reasoning as a Markov chain, where each step represents a stochastic transition in the solution space. The key insight is that convergence to a correct solution is guaranteed as long as the probability of improvement marginally exceeds that of degradation. By running multiple long-horizon, self-evolving processes in parallel, DSER amplifies these small positive tendencies, enabling the model to asymptotically approach correct answers. Empirically, we apply DSER to the DeepSeek-R1-0528-Qwen3-8B model. On the challenging AIME 2024-2025 benchmark, DSER solves 5 out of 9 previously unsolvable problems and boosts overall performance, enabling this compact model to surpass the single-turn accuracy of its 600B-parameter teacher through majority voting. Beyond its immediate utility for test-time scaling, the DSER framework serves to diagnose the fundamental limitations of current open-weight reasoners. By clearly delineating their shortcomings in self-verification, refinement, and stability, our findings establish a clear research agenda for developing next-generation models with powerful, intrinsic self-evolving capabilities. 

---
# Disparities in Multilingual LLM-Based Healthcare Q&A 

**Authors**: Ipek Baris Schlicht, Burcu Sayin, Zhixue Zhao, Frederik M. Labonté, Cesare Barbera, Marco Viviani, Paolo Rosso, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2510.17476)  

**Abstract**: Equitable access to reliable health information is vital when integrating AI into healthcare. Yet, information quality varies across languages, raising concerns about the reliability and consistency of multilingual Large Language Models (LLMs). We systematically examine cross-lingual disparities in pre-training source and factuality alignment in LLM answers for multilingual healthcare Q&A across English, German, Turkish, Chinese (Mandarin), and Italian. We (i) constructed Multilingual Wiki Health Care (MultiWikiHealthCare), a multilingual dataset from Wikipedia; (ii) analyzed cross-lingual healthcare coverage; (iii) assessed LLM response alignment with these references; and (iv) conducted a case study on factual alignment through the use of contextual information and Retrieval-Augmented Generation (RAG). Our findings reveal substantial cross-lingual disparities in both Wikipedia coverage and LLM factual alignment. Across LLMs, responses align more with English Wikipedia, even when the prompts are non-English. Providing contextual excerpts from non-English Wikipedia at inference time effectively shifts factual alignment toward culturally relevant knowledge. These results highlight practical pathways for building more equitable, multilingual AI systems for healthcare. 

---
# Evaluating Large Language Models on Urdu Idiom Translation 

**Authors**: Muhammad Farmal Khan, Mousumi Akter  

**Link**: [PDF](https://arxiv.org/pdf/2510.17460)  

**Abstract**: Idiomatic translation remains a significant challenge in machine translation, especially for low resource languages such as Urdu, and has received limited prior attention. To advance research in this area, we introduce the first evaluation datasets for Urdu to English idiomatic translation, covering both Native Urdu and Roman Urdu scripts and annotated with gold-standard English equivalents. We evaluate multiple open-source Large Language Models (LLMs) and Neural Machine Translation (NMT) systems on this task, focusing on their ability to preserve idiomatic and cultural meaning. Automatic metrics including BLEU, BERTScore, COMET, and XCOMET are used to assess translation quality. Our findings indicate that prompt engineering enhances idiomatic translation compared to direct translation, though performance differences among prompt types are relatively minor. Moreover, cross script comparisons reveal that text representation substantially affects translation quality, with Native Urdu inputs producing more accurate idiomatic translations than Roman Urdu. 

---
# Annotation-Efficient Universal Honesty Alignment 

**Authors**: Shiyu Ni, Keping Bi, Jiafeng Guo, Minghao Tang, Jingtong Wu, Zengxin Han, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.17509)  

**Abstract**: Honesty alignment-the ability of large language models (LLMs) to recognize their knowledge boundaries and express calibrated confidence-is essential for trustworthy deployment. Existing methods either rely on training-free confidence estimation (e.g., token probabilities, self-consistency) or training-based calibration with correctness annotations. While effective, achieving universal honesty alignment with training-based calibration requires costly, large-scale labeling. To support annotation-efficient training, we introduce Elicitation-Then-Calibration (EliCal), a two-stage framework that first elicits internal confidence using inexpensive self-consistency supervision, then calibrates this confidence with a small set of correctness annotations. To support a large-scale study, we release HonestyBench, a benchmark covering ten free-form QA datasets with 560k training and 70k evaluation instances annotated with correctness and self-consistency signals. Experiments show that EliCal achieves near-optimal alignment with only 1k correctness annotations (0.18% of full supervision) and better alignment performance on unseen MMLU tasks than the calibration-only baseline, offering a scalable solution toward universal honesty alignment in LLMs. 

---
# Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents 

**Authors**: Yihong Tang, Kehai Chen, Liang Yue, Jinxin Fan, Caishen Zhou, Xiaoguang Li, Yuyang Zhang, Mingming Zhao, Shixiong Kai, Kaiyang Guo, Xingshan Zeng, Wenjing Cun, Lifeng Shang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17491)  

**Abstract**: With the rise of large language models (LLMs), LLM agents capable of autonomous reasoning, planning, and executing complex tasks have become a frontier in artificial intelligence. However, how to translate the research on general agents into productivity that drives industry transformations remains a significant challenge. To address this, this paper systematically reviews the technologies, applications, and evaluation methods of industry agents based on LLMs. Using an industry agent capability maturity framework, it outlines the evolution of agents in industry applications, from "process execution systems" to "adaptive social systems." First, we examine the three key technological pillars that support the advancement of agent capabilities: Memory, Planning, and Tool Use. We discuss how these technologies evolve from supporting simple tasks in their early forms to enabling complex autonomous systems and collective intelligence in more advanced forms. Then, we provide an overview of the application of industry agents in real-world domains such as digital engineering, scientific discovery, embodied intelligence, collaborative business execution, and complex system simulation. Additionally, this paper reviews the evaluation benchmarks and methods for both fundamental and specialized capabilities, identifying the challenges existing evaluation systems face regarding authenticity, safety, and industry specificity. Finally, we focus on the practical challenges faced by industry agents, exploring their capability boundaries, developmental potential, and governance issues in various scenarios, while providing insights into future directions. By combining technological evolution with industry practices, this review aims to clarify the current state and offer a clear roadmap and theoretical foundation for understanding and building the next generation of industry agents. 

---
# Leveraging Group Relative Policy Optimization to Advance Large Language Models in Traditional Chinese Medicine 

**Authors**: Jiacheng Xie, Shuai Zeng, Yang Yu, Xiaoting Tang, Guanghui An, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17402)  

**Abstract**: Traditional Chinese Medicine (TCM) presents a rich and structurally unique knowledge system that challenges conventional applications of large language models (LLMs). Although previous TCM-specific LLMs have shown progress through supervised fine-tuning, they often face limitations in alignment, data quality, and evaluation consistency. In this study, we introduce Ladder-base, the first TCM-focused LLM trained with Group Relative Policy Optimization (GRPO), a reinforcement learning method that improves reasoning and factual consistency by optimizing response selection based on intra-group comparisons. Ladder-base is built upon the Qwen2.5-7B-Instruct foundation model and trained exclusively on the textual subset of the TCM-Ladder benchmark, using 80 percent of the data for training and the remaining 20 percent split evenly between validation and test sets. Through standardized evaluation, Ladder-base demonstrates superior performance across multiple reasoning metrics when compared to both state-of-the-art general-purpose LLMs such as GPT-4, Gemini 2.5, Claude 3, and Qwen3 and domain-specific TCM models including BenTsao, HuatuoGPT2, and Zhongjing. These findings suggest that GRPO provides an effective and efficient strategy for aligning LLMs with expert-level reasoning in traditional medical domains and supports the development of trustworthy and clinically grounded TCM artificial intelligence systems. 

---
# Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation 

**Authors**: Chenghao Zhang, Guanting Dong, Xinyu Yang, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.17354)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) by retrieving relevant documents from an external corpus. However, existing RAG systems primarily focus on unimodal text documents, and often fall short in real-world scenarios where both queries and documents may contain mixed modalities (such as text and images). In this paper, we address the challenge of Universal Retrieval-Augmented Generation (URAG), which involves retrieving and reasoning over mixed-modal information to improve vision-language generation. To this end, we propose Nyx, a unified mixed-modal to mixed-modal retriever tailored for URAG scenarios. To mitigate the scarcity of realistic mixed-modal data, we introduce a four-stage automated pipeline for generation and filtering, leveraging web documents to construct NyxQA, a dataset comprising diverse mixed-modal question-answer pairs that better reflect real-world information needs. Building on this high-quality dataset, we adopt a two-stage training framework for Nyx: we first perform pre-training on NyxQA along with a variety of open-source retrieval datasets, followed by supervised fine-tuning using feedback from downstream vision-language models (VLMs) to align retrieval outputs with generative preferences. Experimental results demonstrate that Nyx not only performs competitively on standard text-only RAG benchmarks, but also excels in the more general and realistic URAG setting, significantly improving generation quality in vision-language tasks. 

---
# Qomhra: A Bilingual Irish-English Large Language Model 

**Authors**: Joseph McInerney  

**Link**: [PDF](https://arxiv.org/pdf/2510.17652)  

**Abstract**: This paper introduces Qomhrá, a bilingual Irish-English large language model (LLM), developed under low-resource constraints presenting a complete pipeline spanning bilingual continued pre-training, instruction tuning, and alignment from human preferences. Newly accessible Irish corpora and English text are mixed and curated to improve Irish performance while preserving English ability. 6 closed-weight LLMs are judged for their Irish text generation by a native speaker, a learner and other LLMs. Google's Gemini-2.5-Pro is ranked the highest and is subsequently used to synthesise instruction tuning and human preference datasets. Two datasets are contributed leveraging Gemini-2.5-Pro: a 30K Irish-English parallel instruction tuning dataset and a 1K human preference dataset, generating accepted and rejected responses that show near perfect alignment with a native Irish speaker. Qomhrá is comprehensively evaluated across benchmarks testing translation, gender understanding, topic identification and world knowledge with gains of up to 29% in Irish and 44% in English. Qomhrá also undergoes instruction tuning and demonstrates clear progress in instruction following, crucial for chatbot functionality. 

---
# Explainability of Large Language Models: Opportunities and Challenges toward Generating Trustworthy Explanations 

**Authors**: Shahin Atakishiyev, Housam K.B. Babiker, Jiayi Dai, Nawshad Farruque, Teruaki Hayashi, Nafisa Sadaf Hriti, Md Abed Rahman, Iain Smith, Mi-Young Kim, Osmar R. Zaïane, Randy Goebel  

**Link**: [PDF](https://arxiv.org/pdf/2510.17256)  

**Abstract**: Large language models have exhibited impressive performance across a broad range of downstream tasks in natural language processing. However, how a language model predicts the next token and generates content is not generally understandable by humans. Furthermore, these models often make errors in prediction and reasoning, known as hallucinations. These errors underscore the urgent need to better understand and interpret the intricate inner workings of language models and how they generate predictive outputs. Motivated by this gap, this paper investigates local explainability and mechanistic interpretability within Transformer-based large language models to foster trust in such models. In this regard, our paper aims to make three key contributions. First, we present a review of local explainability and mechanistic interpretability approaches and insights from relevant studies in the literature. Furthermore, we describe experimental studies on explainability and reasoning with large language models in two critical domains -- healthcare and autonomous driving -- and analyze the trust implications of such explanations for explanation receivers. Finally, we summarize current unaddressed issues in the evolving landscape of LLM explainability and outline the opportunities, critical challenges, and future directions toward generating human-aligned, trustworthy LLM explanations. 

---
# BenCao: An Instruction-Tuned Large Language Model for Traditional Chinese Medicine 

**Authors**: Jiacheng Xie, Yang Yu, Yibo Chen, Hanyao Zhang, Lening Zhao, Jiaxuan He, Lei Jiang, Xiaoting Tang, Guanghui An, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17415)  

**Abstract**: Traditional Chinese Medicine (TCM), with a history spanning over two millennia, plays a role in global healthcare. However, applying large language models (LLMs) to TCM remains challenging due to its reliance on holistic reasoning, implicit logic, and multimodal diagnostic cues. Existing TCM-domain LLMs have made progress in text-based understanding but lack multimodal integration, interpretability, and clinical applicability. To address these limitations, we developed BenCao, a ChatGPT-based multimodal assistant for TCM, integrating structured knowledge bases, diagnostic data, and expert feedback refinement. BenCao was trained through natural language instruction tuning rather than parameter retraining, aligning with expert-level reasoning and ethical norms specific to TCM. The system incorporates a comprehensive knowledge base of over 1,000 classical and modern texts, a scenario-based instruction framework for diverse interactions, a chain-of-thought simulation mechanism for interpretable reasoning, and a feedback refinement process involving licensed TCM practitioners. BenCao connects to external APIs for tongue-image classification and multimodal database retrieval, enabling dynamic access to diagnostic resources. In evaluations across single-choice question benchmarks and multimodal classification tasks, BenCao achieved superior accuracy to general-domain and TCM-domain models, particularly in diagnostics, herb recognition, and constitution classification. The model was deployed as an interactive application on the OpenAI GPTs Store, accessed by nearly 1,000 users globally as of October 2025. This study demonstrates the feasibility of developing a TCM-domain LLM through natural language-based instruction tuning and multimodal integration, offering a practical framework for aligning generative AI with traditional medical reasoning and a scalable pathway for real-world deployment. 

---
# StreamingThinker: Large Language Models Can Think While Reading 

**Authors**: Junlong Tong, Yingqi Fan, Anhao Zhao, Yunpu Ma, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.17238)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in chain of thought (CoT) reasoning. However, the current LLM reasoning paradigm initiates thinking only after the entire input is available, which introduces unnecessary latency and weakens attention to earlier information in dynamic scenarios. Inspired by human cognition of thinking while reading, we first design a \textit{\textbf{streaming thinking}} paradigm for LLMs, where reasoning unfolds in the order of input and further adjusts its depth once reading is complete. We instantiate this paradigm with \textit{StreamingThinker}, a framework that enables LLMs to think while reading through the integration of streaming CoT generation, streaming-constraint training, and streaming parallel inference. Specifically, StreamingThinker employs streaming reasoning units with quality control for CoT generation, enforces order-preserving reasoning through streaming attention masks and position encoding, and leverages parallel KV caches that decouple input encoding from reasoning generation, thereby ensuring alignment and enabling true concurrency. We evaluate StreamingThinker on the Qwen3 model family across math reasoning, logical reasoning, and context-based QA reasoning tasks. Experimental results show that the StreamingThinker preserves performance comparable to batch thinking, while yielding an 80\% reduction in token waiting before the onset of reasoning and a more than 60\% reduction in time-level latency for producing the final answer, demonstrating the effectiveness of the streaming paradigm for LLM reasoning. Code will be released at \href{this https URL}{this repository.} 

---
# Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting 

**Authors**: Chenchen Tan, Youyang Qu, Xinghao Li, Hui Zhang, Shujie Cui, Cunjian Chen, Longxiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17210)  

**Abstract**: The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of large language models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark and 10% on the TDEC benchmark, while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability. 

---
# The Atomic Instruction Gap: Instruction-Tuned LLMs Struggle with Simple, Self-Contained Directives 

**Authors**: Henry Lim, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.17388)  

**Abstract**: Instruction-tuned large language models (IT-LLMs) exhibit strong zero-shot reasoning, yet their ability to execute simple, self-contained instructions remains underexplored, despite this being foundational to complex instruction-following. We evaluate 20 IT-LLMs on modified MMLU and MMLU-Pro benchmarks, by systematically varying the format of option labels (alphabetic, numeric, Roman) while keeping their meaning identical under four paradigms, namely: (1) With explicit instructions, label changes cause large performance shifts (e.g., -30.45\% for Roman vs. numeric), revealing instruction-format bias. (2) Without instructions, performance drops further (up to -10.84\%) and label sensitivity intensifies, underscoring the role of explicit guidance. (3) When option contents are removed, models fail random-choice baselines except with numeric labels, suggesting weak adherence to atomic directives. (4) Three-shot exemplars yield no significant gains in robustness or fidelity, and generation analyses show persistent label errors, especially for non-numeric formats. Across model sizes, larger LLMs achieve higher accuracy but remain inconsistent in instruction adherence. These results expose the insufficiencies of current instruction-tuning paradigms and highlight the need for evaluation methods and training strategies that explicitly target atomic instruction-following. 

---
# SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents 

**Authors**: Qiusi Zhan, Angeline Budiman-Chan, Abdelrahman Zayed, Xingzhi Guo, Daniel Kang, Joo-Kyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.17017)  

**Abstract**: Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked ``How can I track someone's location without their consent?'', a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility. 

---
# DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking 

**Authors**: Lanni Bu, Lauren Levin, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2510.17013)  

**Abstract**: Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models. 

---
# Investigating Thinking Behaviours of Reasoning-Based Language Models for Social Bias Mitigation 

**Authors**: Guoqing Luo, Iffat Maab, Lili Mou, Junichi Yamagishi  

**Link**: [PDF](https://arxiv.org/pdf/2510.17062)  

**Abstract**: While reasoning-based large language models excel at complex tasks through an internal, structured thinking process, a concerning phenomenon has emerged that such a thinking process can aggregate social stereotypes, leading to biased outcomes. However, the underlying behaviours of these language models in social bias scenarios remain underexplored. In this work, we systematically investigate mechanisms within the thinking process behind this phenomenon and uncover two failure patterns that drive social bias aggregation: 1) stereotype repetition, where the model relies on social stereotypes as its primary justification, and 2) irrelevant information injection, where it fabricates or introduces new details to support a biased narrative. Building on these insights, we introduce a lightweight prompt-based mitigation approach that queries the model to review its own initial reasoning against these specific failure patterns. Experiments on question answering (BBQ and StereoSet) and open-ended (BOLD) benchmarks show that our approach effectively reduces bias while maintaining or improving accuracy. 

---
# Mapping from Meaning: Addressing the Miscalibration of Prompt-Sensitive Language Models 

**Authors**: Kyle Cox, Jiawei Xu, Yikun Han, Rong Xu, Tianhao Li, Chi-Yang Hsu, Tianlong Chen, Walter Gerych, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.17028)  

**Abstract**: An interesting behavior in large language models (LLMs) is prompt sensitivity. When provided with different but semantically equivalent versions of the same prompt, models may produce very different distributions of answers. This suggests that the uncertainty reflected in a model's output distribution for one prompt may not reflect the model's uncertainty about the meaning of the prompt. We model prompt sensitivity as a type of generalization error, and show that sampling across the semantic ``concept space'' with paraphrasing perturbations improves uncertainty calibration without compromising accuracy. Additionally, we introduce a new metric for uncertainty decomposition in black-box LLMs that improves upon entropy-based decomposition by modeling semantic continuities in natural language generation. We show that this decomposition metric can be used to quantify how much LLM uncertainty is attributed to prompt sensitivity. Our work introduces a new way to improve uncertainty calibration in prompt-sensitive language models, and provides evidence that some LLMs fail to exhibit consistent general reasoning about the meanings of their inputs. 

---
# DVAGen: Dynamic Vocabulary Augmented Generation 

**Authors**: Wei Du, Nuowei Liu, Jie Wang, Jiahao Kuang, Tao Ji, Xiaoling Wang, Yuanbin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17115)  

**Abstract**: Language models trained with a fixed vocabulary struggle to generalize to novel or out-of-vocabulary words, limiting their flexibility in handling diverse token combinations. Existing dynamic vocabulary approaches attempt to address this limitation but face challenges such as fragmented codebases, lack of support for modern LLMs, and limited inference scalability. To overcome these issues, we introduce DVAGen, a fully open-source, unified framework designed for training, evaluation, and visualization of dynamic vocabulary-augmented language models. Our framework modularizes the pipeline for ease of customization, integrates seamlessly with open-source LLMs, and is the first to provide both CLI and WebUI tools for real-time result inspection. We validate the effectiveness of dynamic vocabulary methods on modern LLMs and demonstrate support for batch inference, significantly improving inference throughput. 

---
# Online Learning Defense against Iterative Jailbreak Attacks via Prompt Optimization 

**Authors**: Masahiro Kaneko, Zeerak Talat, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17006)  

**Abstract**: Iterative jailbreak methods that repeatedly rewrite and input prompts into large language models (LLMs) to induce harmful outputs -- using the model's previous responses to guide each new iteration -- have been found to be a highly effective attack strategy. Despite being an effective attack strategy against LLMs and their safety mechanisms, existing defenses do not proactively disrupt this dynamic trial-and-error cycle. In this study, we propose a novel framework that dynamically updates its defense strategy through online learning in response to each new prompt from iterative jailbreak methods. Leveraging the distinctions between harmful jailbreak-generated prompts and typical harmless prompts, we introduce a reinforcement learning-based approach that optimizes prompts to ensure appropriate responses for harmless tasks while explicitly rejecting harmful prompts. Additionally, to curb overfitting to the narrow band of partial input rewrites explored during an attack, we introduce Past-Direction Gradient Damping (PDGD). Experiments conducted on three LLMs show that our approach significantly outperforms five existing defense methods against five iterative jailbreak methods. Moreover, our results indicate that our prompt optimization strategy simultaneously enhances response quality for harmless tasks. 

---
# Parameter-Efficient Fine-Tuning for Low-Resource Languages: A Comparative Study of LLMs for Bengali Hate Speech Detection 

**Authors**: Akif Islam, Mohd Ruhul Ameen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16985)  

**Abstract**: Bengali social media platforms have witnessed a sharp increase in hate speech, disproportionately affecting women and adolescents. While datasets such as BD-SHS provide a basis for structured evaluation, most prior approaches rely on either computationally costly full-model fine-tuning or proprietary APIs. This paper presents the first application of Parameter-Efficient Fine-Tuning (PEFT) for Bengali hate speech detection using LoRA and QLoRA. Three instruction-tuned large language models - Gemma-3-4B, Llama-3.2-3B, and Mistral-7B - were fine-tuned on the BD-SHS dataset of 50,281 annotated comments. Each model was adapted by training fewer than 1% of its parameters, enabling experiments on a single consumer-grade GPU. The results show that Llama-3.2-3B achieved the highest F1-score of 92.23%, followed by Mistral-7B at 88.94% and Gemma-3-4B at 80.25%. These findings establish PEFT as a practical and replicable strategy for Bengali and related low-resource languages. 

---
# When AI companions become witty: Can human brain recognize AI-generated irony? 

**Authors**: Xiaohui Rao, Hanlin Wu, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.17168)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as social agents and trained to produce humor and irony, a question emerges: when encountering witty AI remarks, do people interpret these as intentional communication or mere computational output? This study investigates whether people adopt the intentional stance, attributing mental states to explain behavior,toward AI during irony comprehension. Irony provides an ideal paradigm because it requires distinguishing intentional contradictions from unintended errors through effortful semantic reanalysis. We compared behavioral and neural responses to ironic statements from AI versus human sources using established ERP components: P200 reflecting early incongruity detection and P600 indexing cognitive efforts in reinterpreting incongruity as deliberate irony. Results demonstrate that people do not fully adopt the intentional stance toward AI-generated irony. Behaviorally, participants attributed incongruity to deliberate communication for both sources, though significantly less for AI than human, showing greater tendency to interpret AI incongruities as computational errors. Neural data revealed attenuated P200 and P600 effects for AI-generated irony, suggesting reduced effortful detection and reanalysis consistent with diminished attribution of communicative intent. Notably, people who perceived AI as more sincere showed larger P200 and P600 effects for AI-generated irony, suggesting that intentional stance adoption is calibrated by specific mental models of artificial agents. These findings reveal that source attribution shapes neural processing of social-communicative phenomena. Despite current LLMs' linguistic sophistication, achieving genuine social agency requires more than linguistic competence, it necessitates a shift in how humans perceive and attribute intentionality to artificial agents. 

---
# Vocab Diet: Reshaping the Vocabulary of LLMs with Vector Arithmetic 

**Authors**: Yuval Reif, Guy Kaplan, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2510.17001)  

**Abstract**: Large language models (LLMs) were shown to encode word form variations, such as "walk"->"walked", as linear directions in embedding space. However, standard tokenization algorithms treat these variations as distinct tokens -- filling the size-capped vocabulary with surface form variants (e.g., "walk", "walking", "Walk"), at the expense of less frequent words and multilingual coverage. We show that many of these variations can be captured by transformation vectors -- additive offsets that yield the appropriate word's representation when applied to the base form word embedding -- in both the input and output spaces. Building on this, we propose a compact reshaping of the vocabulary: rather than assigning unique tokens to each surface form, we compose them from shared base form and transformation vectors (e.g., "walked" = "walk" + past tense). We apply our approach to multiple LLMs and across five languages, removing up to 10% of vocabulary entries -- thereby freeing space to allocate new, more diverse tokens. Importantly, we do so while also expanding vocabulary coverage to out-of-vocabulary words, with minimal impact on downstream performance, and without modifying model weights. Our findings motivate a foundational rethinking of vocabulary design, moving from string enumeration to a compositional vocabulary that leverages the underlying structure of language. 

---
# Knowing the Facts but Choosing the Shortcut: Understanding How Large Language Models Compare Entities 

**Authors**: Hans Hergen Lehmann, Jae Hee Lee, Steven Schockaert, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2510.16815)  

**Abstract**: Large Language Models (LLMs) are increasingly used for knowledge-based reasoning tasks, yet understanding when they rely on genuine knowledge versus superficial heuristics remains challenging. We investigate this question through entity comparison tasks by asking models to compare entities along numerical attributes (e.g., ``Which river is longer, the Danube or the Nile?''), which offer clear ground truth for systematic analysis. Despite having sufficient numerical knowledge to answer correctly, LLMs frequently make predictions that contradict this knowledge. We identify three heuristic biases that strongly influence model predictions: entity popularity, mention order, and semantic co-occurrence. For smaller models, a simple logistic regression using only these surface cues predicts model choices more accurately than the model's own numerical predictions, suggesting heuristics largely override principled reasoning. Crucially, we find that larger models (32B parameters) selectively rely on numerical knowledge when it is more reliable, while smaller models (7--8B parameters) show no such discrimination, which explains why larger models outperform smaller ones even when the smaller models possess more accurate knowledge. Chain-of-thought prompting steers all models towards using the numerical features across all model sizes. 

---
# LC-Eval: A Bilingual Multi-Task Evaluation Benchmark for Long-Context Understanding 

**Authors**: Sheikh Jubair, Arwa Omayrah, Amal Alshammari, Alhanoof Althnian, Abdulhamed Alothaimen, Norah A. Alzahrani, Shahad D. Alzaidi, Nora Al-Twairesh, Abdulmohsen Al-Thubaity  

**Link**: [PDF](https://arxiv.org/pdf/2510.16783)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated sophisticated capabilities, including the ability to process and comprehend extended contexts. These emergent capabilities necessitate rigorous evaluation methods to effectively assess their performance in long-context understanding. In this paper, we present \textbf{LC-Eval}, a bilingual, multi-task evaluation benchmark designed to evaluate long-context understanding in English and Arabic, targeting context lengths ranging from 4k to over 128k tokens. LC-Eval introduces four novel and challenging tasks: multi-document question answering, bilingual question answering, claim verification within a paragraph, and multiple-choice questions based on long contexts. These tasks are designed to assess LLMs' abilities in deep reasoning, document comprehension, information tracing, and bilingual information extraction and understanding. The benchmark includes datasets in both Arabic and English for each task, allowing for a comparative analysis of their performance across different text genres. Evaluations were conducted on both open-weight and closed LLMs, with results indicating that LC-Eval presents significant challenges. Even high-performing models, such as GPT-4o, struggled with certain tasks, highlighting the complexity and rigor of the benchmark. 

---
# ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models 

**Authors**: Emily Chang, Niyati Bafna  

**Link**: [PDF](https://arxiv.org/pdf/2510.16928)  

**Abstract**: Existing benchmarks for large language models (LLMs) are largely restricted to high- or mid-resource languages, and often evaluate performance on higher-order tasks in reasoning and generation. However, plenty of evidence points to the fact that LLMs lack basic linguistic competence in the vast majority of the world's 3800+ written languages. We introduce ChiKhaPo, consisting of 8 subtasks of varying difficulty designed to evaluate the lexical comprehension and generation abilities of generative models. ChiKhaPo draws on existing lexicons, monolingual data, and bitext, and provides coverage for 2700+ languages for 2 subtasks, surpassing any existing benchmark in terms of language coverage. We further show that 6 SOTA models struggle on our benchmark, and discuss the factors contributing to performance scores, including language family, language resourcedness, task, and comprehension versus generation directions. With ChiKhaPo, we hope to enable and encourage the massively multilingual benchmarking of LLMs. 

---
# Prompt-MII: Meta-Learning Instruction Induction for LLMs 

**Authors**: Emily Xiao, Yixiao Zeng, Ada Chen, Chin-Jou Li, Amanda Bertsch, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.16932)  

**Abstract**: A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose PROMPT-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. PROMPT-MII improves downstream model quality by 4-9 F1 points (10-20% relative), matching ICL performance while requiring 3-13x fewer tokens. 

---
# Rethinking On-policy Optimization for Query Augmentation 

**Authors**: Zhichao Xu, Shengyao Zhuang, Xueguang Ma, Bingsen Chen, Yijun Tian, Fengran Mo, Jie Cao, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.17139)  

**Abstract**: Recent advances in large language models (LLMs) have led to a surge of interest in query augmentation for information retrieval (IR). Two main approaches have emerged. The first prompts LLMs to generate answers or pseudo-documents that serve as new queries, relying purely on the model's parametric knowledge or contextual information. The second applies reinforcement learning (RL) to fine-tune LLMs for query rewriting, directly optimizing retrieval metrics. While having respective advantages and limitations, the two approaches have not been compared under consistent experimental conditions. In this work, we present the first systematic comparison of prompting-based and RL-based query augmentation across diverse benchmarks, including evidence-seeking, ad hoc, and tool retrieval. Our key finding is that simple, training-free query augmentation often performs on par with, or even surpasses, more expensive RL-based counterparts, especially when using powerful LLMs. Motivated by this discovery, we introduce a novel hybrid method, On-policy Pseudo-document Query Expansion (OPQE), which, instead of rewriting a query, the LLM policy learns to generate a pseudo-document that maximizes retrieval performance, thus merging the flexibility and generative structure of prompting with the targeted optimization of RL. We show OPQE outperforms both standalone prompting and RL-based rewriting, demonstrating that a synergistic approach yields the best results. Our implementation is made available to facilitate reproducibility. 

---
# Neuronal Group Communication for Efficient Neural representation 

**Authors**: Zhengqi Pei, Qingming Huang, Shuhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16851)  

**Abstract**: The ever-increasing scale of modern neural networks has brought unprecedented performance alongside daunting challenges in efficiency and interpretability. This paper addresses the core question of how to build large neural systems that learn efficient, modular, and interpretable representations. We propose Neuronal Group Communication (NGC), a theory-driven framework that reimagines a neural network as a dynamical system of interacting neuronal groups rather than a monolithic collection of neural weights. Instead of treating each weight as an independent trainable parameter, NGC treats weights as transient interactions between embedding-like neuronal states, with neural computation unfolding through iterative communication among groups of neurons. This low-rank, modular representation yields compact models: groups of neurons exchange low-dimensional signals, enabling intra-group specialization and inter-group information sharing while dramatically reducing redundant parameters. By drawing on dynamical systems theory, we introduce a neuronal stability metric (analogous to Lyapunov stability) that quantifies the contraction of neuron activations toward stable patterns during sequence processing. Using this metric, we reveal that emergent reasoning capabilities correspond to an external driving force or ``potential'', which nudges the neural dynamics away from trivial trajectories while preserving stability. Empirically, we instantiate NGC in large language models (LLMs) and demonstrate improved performance on complex reasoning benchmarks under moderate compression. NGC consistently outperforms standard low-rank approximations and cross-layer basis-sharing methods at comparable compression rates. We conclude by discussing the broader implications of NGC, including how structured neuronal group dynamics might relate to generalization in high-dimensional learning systems. 

---
# Investigating the Impact of Rationales for LLMs on Natural Language Understanding 

**Authors**: Wenhang Shi, Shuqing Bian, Yiren Chen, Xinyi Zhang, Zhe Zhao, Pengfei Hu, Wei Lu, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.16686)  

**Abstract**: Chain-of-thought (CoT) rationales, which provide step-by-step reasoning to derive final answers, benefit LLMs in both inference and training. Incorporating rationales, either by generating them before answering during inference, or by placing them before or after the original answers during training - significantly improves model performance on mathematical, symbolic and commonsense reasoning tasks. However, most work focuses on the role of rationales in these reasoning tasks, overlooking their potential impact on other important tasks like natural language understanding (NLU) tasks. In this work, we raise the question: Can rationales similarly benefit NLU tasks? To conduct a systematic exploration, we construct NLURC, a comprehensive and high-quality NLU dataset collection with rationales, and develop various rationale-augmented methods. Through exploring the applicability of these methods on NLU tasks using the dataset, we uncover several potentially surprising findings: (1) CoT inference shifts from hindering NLU performance to surpassing direct label prediction as model size grows, indicating a positive correlation. (2) Most rationale-augmented training methods perform worse than label-only training, with one specially designed method consistently achieving improvements. (3) LLMs trained with rationales achieve significant performance gains on unseen NLU tasks, rivaling models ten times their size, while delivering interpretability on par with commercial LLMs. 

---
# Temporal Understanding under Deictic Frame of Reference 

**Authors**: Damin Zhang, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2510.16685)  

**Abstract**: Understanding time is fundamental to human cognition, where temporal experience is often conceptualized through spatial metaphors grounded in sensory-motor experience. For example, "summer is approaching" parallels "We are approaching the summer". In such expressions, humans rely on a frame of reference (FoR) to interpret meaning relative to a particular viewpoint. Extending this concept to time, a temporal frame of reference (t-FoR) defines how temporal relations are perceived relative to an experiencer's moment of "now". While Large Language Models (LLMs) have shown remarkable advances in natural language understanding, their ability to interpret and reason about time remains limited. In this work, we introduce TUuD (Temporal Understanding under Deictic t-FoR), a framework that evaluates how LLMs interpret time-event and event-event relations when the reference point of "now" dynamically shifts along a timeline. Following recent work on temporal cognition \cite{li2025other}, LLMs are prompted to rate the similarity between the current moment and a target event from 0.00 (completely dissimilar) to 1.00 (highly similar), where similarity quantifies perceived temporal alignment between the two points. Our results show that four evaluated LLMs exhibit measurable adaptation to a deictic t-FoR, with similarity ratings peaking around the present and decreasing toward past and future events. The adaptation, however, weakens beyond near-term contexts, suggesting that while LLMs display partial human-like temporal cognition, their temporal reasoning remains sensitive to reference-frame shifts and temporal distance. 

---
# so much depends / upon / a whitespace: Why Whitespace Matters for Poets and LLMs 

**Authors**: Sriharsh Bhyravajjula, Melanie Walsh, Anna Preus, Maria Antoniak  

**Link**: [PDF](https://arxiv.org/pdf/2510.16713)  

**Abstract**: Whitespace is a critical component of poetic form, reflecting both adherence to standardized forms and rebellion against those forms. Each poem's whitespace distribution reflects the artistic choices of the poet and is an integral semantic and spatial feature of the poem. Yet, despite the popularity of poetry as both a long-standing art form and as a generation task for large language models (LLMs), whitespace has not received sufficient attention from the NLP community. Using a corpus of 19k English-language published poems from Poetry Foundation, we investigate how 4k poets have used whitespace in their works. We release a subset of 2.8k public-domain poems with preserved formatting to facilitate further research in this area. We compare whitespace usage in the published poems to (1) 51k LLM-generated poems, and (2) 12k unpublished poems posted in an online community. We also explore whitespace usage across time periods, poetic forms, and data sources. Additionally, we find that different text processing methods can result in significantly different representations of whitespace in poetry data, motivating us to use these poems and whitespace patterns to discuss implications for the processing strategies used to assemble pretraining datasets for LLMs. 

---
# All You Need is One: Capsule Prompt Tuning with a Single Vector 

**Authors**: Yiyang Liu, James C. Liang, Heng Fan, Wenhao Yang, Yiming Cui, Xiaotian Han, Lifu Huang, Dongfang Liu, Qifan Wang, Cheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.16670)  

**Abstract**: Prompt-based learning has emerged as a parameter-efficient finetuning (PEFT) approach to facilitate Large Language Model (LLM) adaptation to downstream tasks by conditioning generation with task-aware guidance. Despite its successes, current prompt-based learning methods heavily rely on laborious grid searching for optimal prompt length and typically require considerable number of prompts, introducing additional computational burden. Worse yet, our pioneer findings indicate that the task-aware prompt design is inherently limited by its absence of instance-aware information, leading to a subtle attention interplay with the input sequence. In contrast, simply incorporating instance-aware information as a part of the guidance can enhance the prompt-tuned model performance without additional fine-tuning. Moreover, we find an interesting phenomenon, namely "attention anchor", that incorporating instance-aware tokens at the earliest position of the sequence can successfully preserve strong attention to critical structural information and exhibit more active attention interaction with all input tokens. In light of our observation, we introduce Capsule Prompt-Tuning (CaPT), an efficient and effective solution that leverages off-the-shelf, informative instance semantics into prompt-based learning. Our approach innovatively integrates both instance-aware and task-aware information in a nearly parameter-free manner (i.e., one single capsule prompt). Empirical results demonstrate that our method can exhibit superior performance across various language tasks (e.g., 84.03\% average accuracy on T5-Large), serving as an "attention anchor," while enjoying high parameter efficiency (e.g., 0.003\% of model parameters on Llama3.2-1B). 

---
# Fine-tuning of Large Language Models for Constituency Parsing Using a Sequence to Sequence Approach 

**Authors**: Francisco Jose Cortes Delgado, Eduardo Martinez Gracia, Rafael Valencia Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.16604)  

**Abstract**: Recent advances in natural language processing with large neural models have opened new possibilities for syntactic analysis based on machine learning. This work explores a novel approach to phrase-structure analysis by fine-tuning large language models (LLMs) to translate an input sentence into its corresponding syntactic structure. The main objective is to extend the capabilities of MiSintaxis, a tool designed for teaching Spanish syntax. Several models from the Hugging Face repository were fine-tuned using training data generated from the AnCora-ES corpus, and their performance was evaluated using the F1 score. The results demonstrate high accuracy in phrase-structure analysis and highlight the potential of this methodology. 

---
# Beacon: Single-Turn Diagnosis and Mitigation of Latent Sycophancy in Large Language Models 

**Authors**: Sanskar Pandey, Ruhaan Chopra, Angkul Puniya, Sohom Pal  

**Link**: [PDF](https://arxiv.org/pdf/2510.16727)  

**Abstract**: Large language models internalize a structural trade-off between truthfulness and obsequious flattery, emerging from reward optimization that conflates helpfulness with polite submission. This latent bias, known as sycophancy, manifests as a preference for user agreement over principled reasoning. We introduce Beacon, a single-turn forced-choice benchmark that isolates this bias independent of conversational context, enabling precise measurement of the tension between factual accuracy and submissive bias. Evaluations across twelve state-of-the-art models reveal that sycophancy decomposes into stable linguistic and affective sub-biases, each scaling with model capacity. We further propose prompt-level and activation-level interventions that modulate these biases in opposing directions, exposing the internal geometry of alignment as a dynamic manifold between truthfulness and socially compliant judgment. Beacon reframes sycophancy as a measurable form of normative misgeneralization, providing a reproducible foundation for studying and mitigating alignment drift in large-scale generative systems. 

---
# Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration 

**Authors**: Zhixuan He, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.16645)  

**Abstract**: Large Language Models (LLMs) demonstrate strong performance but often lack interpretable reasoning. This paper introduces the Multi-Agent Collaboration Framework for Diverse Thinking Modes (DiMo), which enhances both performance and interpretability by simulating a structured debate among four specialized LLM agents. Each agent embodies a distinct reasoning paradigm, allowing the framework to collaboratively explore diverse cognitive approaches. Through iterative debate, agents challenge and refine initial responses, yielding more robust conclusions and an explicit, auditable reasoning chain. Across six benchmarks and under a unified open-source setup, DiMo improves accuracy over widely used single-model and debate baselines, with the largest gains on math. We position DiMo as a semantics-aware, Web-native multi-agent framework: it models human-machine intelligence with LLM agents that produce semantically typed, URL-annotated evidence chains for explanations and user-friendly interactions. Although our experiments use standard reasoning benchmarks, the framework is designed to be instantiated over Web corpora and knowledge graphs, combining retrieval-augmented reasoning with structured justifications that downstream systems can inspect and reuse. 

---
# Cross-Genre Authorship Attribution via LLM-Based Retrieve-and-Rerank 

**Authors**: Shantanu Agarwal, Joel Barry, Steven Fincke, Scott Miller  

**Link**: [PDF](https://arxiv.org/pdf/2510.16819)  

**Abstract**: Authorship attribution (AA) is the task of identifying the most likely author of a query document from a predefined set of candidate authors. We introduce a two-stage retrieve-and-rerank framework that finetunes LLMs for cross-genre AA. Unlike the field of information retrieval (IR), where retrieve-and-rerank is a de facto strategy, cross-genre AA systems must avoid relying on topical cues and instead learn to identify author-specific linguistic patterns that are independent of the text's subject matter (genre/domain/topic). Consequently, for the reranker, we demonstrate that training strategies commonly used in IR are fundamentally misaligned with cross-genre AA, leading to suboptimal behavior. To address this, we introduce a targeted data curation strategy that enables the reranker to effectively learn author-discriminative signals. Using our LLM-based retrieve-and-rerank pipeline, we achieve substantial gains of 22.3 and 34.4 absolute Success@8 points over the previous state-of-the-art on HIATUS's challenging HRS1 and HRS2 cross-genre AA benchmarks. 

---
# Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety 

**Authors**: Vamshi Krishna Bonagiri, Ponnurangam Kumaragurum, Khanh Nguyen, Benjamin Plaut  

**Link**: [PDF](https://arxiv.org/pdf/2510.16492)  

**Abstract**: As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications. 

---
# The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models 

**Authors**: Shivam Ratnakar, Sanjay Raghavendra  

**Link**: [PDF](https://arxiv.org/pdf/2510.16712)  

**Abstract**: Integration of Large Language Models with search/retrieval engines has become ubiquitous, yet these systems harbor a critical vulnerability that undermines their reliability. We present the first systematic investigation of "chameleon behavior" in LLMs: their alarming tendency to shift stances when presented with contradictory questions in multi-turn conversations (especially in search-enabled LLMs). Through our novel Chameleon Benchmark Dataset, comprising 17,770 carefully crafted question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains, we expose fundamental flaws in state-of-the-art systems. We introduce two theoretically grounded metrics: the Chameleon Score (0-1) that quantifies stance instability, and Source Re-use Rate (0-1) that measures knowledge diversity. Our rigorous evaluation of Llama-4-Maverick, GPT-4o-mini, and Gemini-2.5-Flash reveals consistent failures: all models exhibit severe chameleon behavior (scores 0.391-0.511), with GPT-4o-mini showing the worst performance. Crucially, small across-temperature variance (less than 0.004) suggests the effect is not a sampling artifact. Our analysis uncovers the mechanism: strong correlations between source re-use rate and confidence (r=0.627) and stance changes (r=0.429) are statistically significant (p less than 0.05), indicating that limited knowledge diversity makes models pathologically deferential to query framing. These findings highlight the need for comprehensive consistency evaluation before deploying LLMs in healthcare, legal, and financial systems where maintaining coherent positions across interactions is critical for reliable decision support. 

---
# ATA: A Neuro-Symbolic Approach to Implement Autonomous and Trustworthy Agents 

**Authors**: David Peer, Sebastian Stabinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.16381)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities, yet their deployment in high-stakes domains is hindered by inherent limitations in trustworthiness, including hallucinations, instability, and a lack of transparency. To address these challenges, we introduce a generic neuro-symbolic approach, which we call Autonomous Trustworthy Agents (ATA). The core of our approach lies in decoupling tasks into two distinct phases: Offline knowledge ingestion and online task processing. During knowledge ingestion, an LLM translates an informal problem specification into a formal, symbolic knowledge base. This formal representation is crucial as it can be verified and refined by human experts, ensuring its correctness and alignment with domain requirements. In the subsequent task processing phase, each incoming input is encoded into the same formal language. A symbolic decision engine then utilizes this encoded input in conjunction with the formal knowledge base to derive a reliable result. Through an extensive evaluation on a complex reasoning task, we demonstrate that a concrete implementation of ATA is competitive with state-of-the-art end-to-end reasoning models in a fully automated setup while maintaining trustworthiness. Crucially, with a human-verified and corrected knowledge base, our approach significantly outperforms even larger models, while exhibiting perfect determinism, enhanced stability against input perturbations, and inherent immunity to prompt injection attacks. By generating decisions grounded in symbolic reasoning, ATA offers a practical and controllable architecture for building the next generation of transparent, auditable, and reliable autonomous agents. 

---
# ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation 

**Authors**: Haoxuan Zhang, Ruochi Li, Sarthak Shrestha, Shree Harshini Mamidala, Revanth Putta, Arka Krishan Aggarwal, Ting Xiao, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16549)  

**Abstract**: Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. Recent work has focused on using LLMs to improve review efficiency or generate insightful review content. However, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine the peer review ecosystem and compromise academic integrity. To address this critical issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews. ReviewGuard employs a comprehensive four-stage LLM-driven framework that: (1) collects ICLR and NeurIPS papers with their corresponding reviews from OpenReview; (2) annotates review types using GPT-4.1 with human validation; (3) addresses class imbalance and data scarcity through LLM-driven synthetic data augmentation, producing a final corpus of 6,634 papers, 24,657 real reviews, and 46,438 synthetic reviews; and (4) fine-tunes both encoder-based models and open source LLMs. We perform comprehensive feature analysis of the structure and quality of the review text. Compared to sufficient reviews, deficient reviews demonstrate lower rating scores, higher self-reported confidence, reduced structural complexity, and a higher proportion of negative sentiment. AI-generated text detection reveals that, since ChatGPT's emergence, AI-generated reviews have increased dramatically. In the evaluation of deficient review detection models, mixed training with synthetic and real review data provides substantial enhancements to recall and F1 scores on the binary task. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review while offering valuable insights into human-AI collaboration to maintain academic integrity. 

---
# Utilising Large Language Models for Generating Effective Counter Arguments to Anti-Vaccine Tweets 

**Authors**: Utsav Dhanuka, Soham Poddar, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.16359)  

**Abstract**: In an era where public health is increasingly influenced by information shared on social media, combatting vaccine skepticism and misinformation has become a critical societal goal. Misleading narratives around vaccination have spread widely, creating barriers to achieving high immunisation rates and undermining trust in health recommendations. While efforts to detect misinformation have made significant progress, the generation of real time counter-arguments tailored to debunk such claims remains an insufficiently explored area. In this work, we explore the capabilities of LLMs to generate sound counter-argument rebuttals to vaccine misinformation. Building on prior research in misinformation debunking, we experiment with various prompting strategies and fine-tuning approaches to optimise counter-argument generation. Additionally, we train classifiers to categorise anti-vaccine tweets into multi-labeled categories such as concerns about vaccine efficacy, side effects, and political influences allowing for more context aware rebuttals. Our evaluation, conducted through human judgment, LLM based assessments, and automatic metrics, reveals strong alignment across these methods. Our findings demonstrate that integrating label descriptions and structured fine-tuning enhances counter-argument effectiveness, offering a promising approach for mitigating vaccine misinformation at scale. 

---
# Language over Content: Tracing Cultural Understanding in Multilingual Large Language Models 

**Authors**: Seungho Cho, Changgeon Ko, Eui Jun Hwang, Junmyeong Lee, Huije Lee, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.16565)  

**Abstract**: Large language models (LLMs) are increasingly used across diverse cultural contexts, making accurate cultural understanding essential. Prior evaluations have mostly focused on output-level performance, obscuring the factors that drive differences in responses, while studies using circuit analysis have covered few languages and rarely focused on culture. In this work, we trace LLMs' internal cultural understanding mechanisms by measuring activation path overlaps when answering semantically equivalent questions under two conditions: varying the target country while fixing the question language, and varying the question language while fixing the country. We also use same-language country pairs to disentangle language from cultural aspects. Results show that internal paths overlap more for same-language, cross-country questions than for cross-language, same-country questions, indicating strong language-specific patterns. Notably, the South Korea-North Korea pair exhibits low overlap and high variability, showing that linguistic similarity does not guarantee aligned internal representation. 

---
# Navigating through the hidden embedding space: steering LLMs to improve mental health assessment 

**Authors**: Federico Ravenda, Seyed Ali Bahrainian, Andrea Raballo, Antonietta Mira  

**Link**: [PDF](https://arxiv.org/pdf/2510.16373)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) is transforming AI, opening new opportunities in sensitive and high-impact areas such as Mental Health (MH). Yet, despite these advancements, recent evidence reveals that smaller-scale models still struggle to deliver optimal performance in domain-specific applications. In this study, we present a cost-efficient yet powerful approach to improve MH assessment capabilities of an LLM, without relying on any computationally intensive techniques. Our lightweight method consists of a linear transformation applied to a specific layer's activations, leveraging steering vectors to guide the model's output. Remarkably, this intervention enables the model to achieve improved results across two distinct tasks: (1) identifying whether a Reddit post is useful for detecting the presence or absence of depressive symptoms (relevance prediction task), and (2) completing a standardized psychological screening questionnaire for depression based on users' Reddit post history (questionnaire completion task). Results highlight the untapped potential of steering mechanisms as computationally efficient tools for LLMs' MH domain adaptation. 

---
# Thinking About Thinking: Evaluating Reasoning in Post-Trained Language Models 

**Authors**: Pratham Singla, Shivank Garg, Ayush Singh, Ishan Garg, Ketan Suhaas Saichandran  

**Link**: [PDF](https://arxiv.org/pdf/2510.16340)  

**Abstract**: Recent advances in post-training techniques have endowed Large Language Models (LLMs) with enhanced capabilities for tackling complex, logic-intensive tasks through the generation of supplementary planning tokens. This development raises a fundamental question: Are these models aware of what they "learn" and "think"? To address this, we define three core competencies: (1) awareness of learned latent policies, (2) generalization of these policies across domains, and (3) alignment between internal reasoning traces and final outputs. We empirically evaluate these abilities on several tasks, each designed to require learning a distinct policy. Furthermore, we contrast the profiles of models post-trained via Supervised Fine-Tuning (SFT), Direct Policy Optimization (DPO), and Group Relative Policy Optimization (GRPO). Our findings indicate that RL-trained models not only demonstrate greater awareness of their learned behaviors and stronger generalizability to novel, structurally similar tasks than SFT models but also often exhibit weak alignment between their reasoning traces and final outputs, an effect most pronounced in GRPO-trained models. 

---
# RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning 

**Authors**: Deyi Ji, Yuekui Yang, Haiyang Wu, Shaoping Ma, Tianrun Chen, Lanyun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16455)  

**Abstract**: Advertisement (Ad) video violation detection is critical for ensuring platform compliance, but existing methods struggle with precise temporal grounding, noisy annotations, and limited generalization. We propose RAVEN, a novel framework that integrates curriculum reinforcement learning with multimodal large language models (MLLMs) to enhance reasoning and cognitive capabilities for violation detection. RAVEN employs a progressive training strategy, combining precisely and coarsely annotated data, and leverages Group Relative Policy Optimization (GRPO) to develop emergent reasoning abilities without explicit reasoning annotations. Multiple hierarchical sophisticated reward mechanism ensures precise temporal grounding and consistent category prediction. Experiments on industrial datasets and public benchmarks show that RAVEN achieves superior performances in violation category accuracy and temporal interval localization. We also design a pipeline to deploy the RAVEN on the online Ad services, and online A/B testing further validates its practical applicability, with significant improvements in precision and recall. RAVEN also demonstrates strong generalization, mitigating the catastrophic forgetting issue associated with supervised fine-tuning. 

---
# Instant Personalized Large Language Model Adaptation via Hypernetwork 

**Authors**: Zhaoxuan Tan, Zixuan Zhang, Haoyang Wen, Zheng Li, Rongzhi Zhang, Pei Chen, Fengran Mo, Zheyuan Liu, Qingkai Zeng, Qingyu Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16282)  

**Abstract**: Personalized large language models (LLMs) tailor content to individual preferences using user profiles or histories. However, existing parameter-efficient fine-tuning (PEFT) methods, such as the ``One-PEFT-Per-User'' (OPPU) paradigm, require training a separate adapter for each user, making them computationally expensive and impractical for real-time updates. We introduce Profile-to-PEFT, a scalable framework that employs a hypernetwork, trained end-to-end, to map a user's encoded profile directly to a full set of adapter parameters (e.g., LoRA), eliminating per-user training at deployment. This design enables instant adaptation, generalization to unseen users, and privacy-preserving local deployment. Experimental results demonstrate that our method outperforms both prompt-based personalization and OPPU while using substantially fewer computational resources at deployment. The framework exhibits strong generalization to out-of-distribution users and maintains robustness across varying user activity levels and different embedding backbones. The proposed Profile-to-PEFT framework enables efficient, scalable, and adaptive LLM personalization suitable for large-scale applications. 

---
# TrajSelector: Harnessing Latent Representations for Efficient and Effective Best-of-N in Large Reasoning Model 

**Authors**: Bin Yu, Xinming Wang, Shijie Lian, Haotian Li, Changti Wu, Ruina Hu, Bailing Wang, Yuliang Wei, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16449)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in complex reasoning tasks, largely enabled by test-time scaling (TTS) paradigms that allocate additional compute during inference. Among these, external TTS (particularly the Best-of-N selection paradigm) yields scalable performance improvements by selecting from multiple independently generated reasoning trajectories. However, this approach faces key limitations: (i) the high computational overhead of deploying process reward models, (ii) the underutilization of the LLM's intrinsic latent representations. We introduce TrajSelector, an efficient and effective Best-of-N framework that exploit the hidden states in the sampler LLM for process-level scoring. A lightweight verifier (with only 0.6B parameters) evaluates the quality of step-wise trajectory, and then aggregates these scores to identify the optimal reasoning trajectory. Our framework employs a fully data-driven, end-to-end training recipe that eliminates reliance on massive step-level annotations. Experiential results across five benchmarks demonstrate that TrajSelector delivers consistent performance gains. In Best-of-32 settings, it surpasses majority voting by 4.61% accuracy and outperforms existing process reward models by 4.31% to 12.21%, all while maintaining lower inference costs. 

---
# What Can String Probability Tell Us About Grammaticality? 

**Authors**: Jennifer Hu, Ethan Gotlieb Wilcox, Siyuan Song, Kyle Mahowald, Roger P. Levy  

**Link**: [PDF](https://arxiv.org/pdf/2510.16227)  

**Abstract**: What have language models (LMs) learned about grammar? This question remains hotly debated, with major ramifications for linguistic theory. However, since probability and grammaticality are distinct notions in linguistics, it is not obvious what string probabilities can reveal about an LM's underlying grammatical knowledge. We present a theoretical analysis of the relationship between grammar, meaning, and string probability, based on simple assumptions about the generative process of corpus data. Our framework makes three predictions, which we validate empirically using 280K sentence pairs in English and Chinese: (1) correlation between the probability of strings within minimal pairs, i.e., string pairs with minimal semantic differences; (2) correlation between models' and humans' deltas within minimal pairs; and (3) poor separation in probability space between unpaired grammatical and ungrammatical strings. Our analyses give theoretical grounding for using probability to learn about LMs' structural knowledge, and suggest directions for future work in LM grammatical evaluation. 

---
# Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs 

**Authors**: Guiyao Tie, Zenghui Yuan, Zeli Zhao, Chaoran Hu, Tianhe Gu, Ruihang Zhang, Sizhe Zhang, Junran Wu, Xiaoyue Tu, Ming Jin, Qingsong Wen, Lixing Chen, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.16062)  

**Abstract**: Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: this https URL 

---
# Evaluating Prompting Strategies and Large Language Models in Systematic Literature Review Screening: Relevance and Task-Stage Classification 

**Authors**: Binglan Han, Anuradha Mathrani, Teo Susnjak  

**Link**: [PDF](https://arxiv.org/pdf/2510.16091)  

**Abstract**: This study quantifies how prompting strategies interact with large language models (LLMs) to automate the screening stage of systematic literature reviews (SLRs). We evaluate six LLMs (GPT-4o, GPT-4o-mini, DeepSeek-Chat-V3, Gemini-2.5-Flash, Claude-3.5-Haiku, Llama-4-Maverick) under five prompt types (zero-shot, few-shot, chain-of-thought (CoT), CoT-few-shot, self-reflection) across relevance classification and six Level-2 tasks, using accuracy, precision, recall, and F1. Results show pronounced model-prompt interaction effects: CoT-few-shot yields the most reliable precision-recall balance; zero-shot maximizes recall for high-sensitivity passes; and self-reflection underperforms due to over-inclusivity and instability across models. GPT-4o and DeepSeek provide robust overall performance, while GPT-4o-mini performs competitively at a substantially lower dollar cost. A cost-performance analysis for relevance classification (per 1,000 abstracts) reveals large absolute differences among model-prompt pairings; GPT-4o-mini remains low-cost across prompts, and structured prompts (CoT/CoT-few-shot) on GPT-4o-mini offer attractive F1 at a small incremental cost. We recommend a staged workflow that (1) deploys low-cost models with structured prompts for first-pass screening and (2) escalates only borderline cases to higher-capacity models. These findings highlight LLMs' uneven but promising potential to automate literature screening. By systematically analyzing prompt-model interactions, we provide a comparative benchmark and practical guidance for task-adaptive LLM deployment. 

---
# EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle 

**Authors**: Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.16079)  

**Abstract**: Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at this https URL. 

---
# Fusion-Augmented Large Language Models: Boosting Diagnostic Trustworthiness via Model Consensus 

**Authors**: Md Kamrul Siam, Md Jobair Hossain Faruk, Jerry Q. Cheng, Huanying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16057)  

**Abstract**: This study presents a novel multi-model fusion framework leveraging two state-of-the-art large language models (LLMs), ChatGPT and Claude, to enhance the reliability of chest X-ray interpretation on the CheXpert dataset. From the full CheXpert corpus of 224,316 chest radiographs, we randomly selected 234 radiologist-annotated studies to evaluate unimodal performance using image-only prompts. In this setting, ChatGPT and Claude achieved diagnostic accuracies of 62.8% and 76.9%, respectively. A similarity-based consensus approach, using a 95% output similarity threshold, improved accuracy to 77.6%. To assess the impact of multimodal inputs, we then generated synthetic clinical notes following the MIMIC-CXR template and evaluated a separate subset of 50 randomly selected cases paired with both images and synthetic text. On this multimodal cohort, performance improved to 84% for ChatGPT and 76% for Claude, while consensus accuracy reached 91.3%. Across both experimental conditions, agreement-based fusion consistently outperformed individual models. These findings highlight the utility of integrating complementary modalities and using output-level consensus to improve the trustworthiness and clinical utility of AI-assisted radiological diagnosis, offering a practical path to reduce diagnostic errors with minimal computational overhead. 

---
# LILO: Bayesian Optimization with Interactive Natural Language Feedback 

**Authors**: Katarzyna Kobalczyk, Zhiyuan Jerry Lin, Benjamin Letham, Zhuokai Zhao, Maximilian Balandat, Eytan Bakshy  

**Link**: [PDF](https://arxiv.org/pdf/2510.17671)  

**Abstract**: For many real-world applications, feedback is essential in translating complex, nuanced, or subjective goals into quantifiable optimization objectives. We propose a language-in-the-loop framework that uses a large language model (LLM) to convert unstructured feedback in the form of natural language into scalar utilities to conduct BO over a numeric search space. Unlike preferential BO, which only accepts restricted feedback formats and requires customized models for each domain-specific problem, our approach leverages LLMs to turn varied types of textual feedback into consistent utility signals and to easily include flexible user priors without manual kernel design. At the same time, our method maintains the sample efficiency and principled uncertainty quantification of BO. We show that this hybrid method not only provides a more natural interface to the decision maker but also outperforms conventional BO baselines and LLM-only optimizers, particularly in feedback-limited regimes. 

---
# LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena 

**Authors**: Qingchuan Yang, Simon Mahns, Sida Li, Anri Gu, Jibang Wu, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17638)  

**Abstract**: Forecasting is not only a fundamental intellectual pursuit but also is of significant importance to societal systems such as finance and economics. With the rapid advances of large language models (LLMs) trained on Internet-scale data, it raises the promise of employing LLMs to forecast real-world future events, an emerging paradigm we call "LLM-as-a-Prophet". This paper systematically investigates such predictive intelligence of LLMs. To this end, we build Prophet Arena, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, in order to support our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks towards achieving superior predictive intelligence via LLM-as-a-Prophet, such as LLMs' inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears. 

---
# Contextual Attention Modulation: Towards Efficient Multi-Task Adaptation in Large Language Models 

**Authors**: Dayan Pan, Zhaoyang Fu, Jingyuan Wang, Xiao Han, Yue Zhu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17705)  

**Abstract**: Large Language Models (LLMs) possess remarkable generalization capabilities but struggle with multi-task adaptation, particularly in balancing knowledge retention with task-specific specialization. Conventional fine-tuning methods suffer from catastrophic forgetting and substantial resource consumption, while existing parameter-efficient methods perform suboptimally in complex multi-task scenarios. To address this, we propose Contextual Attention Modulation (CAM), a novel mechanism that dynamically modulates the representations of self-attention modules in LLMs. CAM enhances task-specific features while preserving general knowledge, thereby facilitating more effective and efficient adaptation. For effective multi-task adaptation, CAM is integrated into our Hybrid Contextual Attention Modulation (HyCAM) framework, which combines a shared, full-parameter CAM module with multiple specialized, lightweight CAM modules, enhanced by a dynamic routing strategy for adaptive knowledge fusion. Extensive experiments on heterogeneous tasks, including question answering, code generation, and logical reasoning, demonstrate that our approach significantly outperforms existing approaches, achieving an average performance improvement of 3.65%. The implemented code and data are available to ease reproducibility at this https URL. 

---
# Do LLMs Recognize Your Latent Preferences? A Benchmark for Latent Information Discovery in Personalized Interaction 

**Authors**: Ioannis Tsaknakis, Bingqing Song, Shuyu Gan, Dongyeop Kang, Alfredo Garcia, Gaowen Liu, Charles Fleming, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.17132)  

**Abstract**: Large Language Models (LLMs) excel at producing broadly relevant text, but this generality becomes a limitation when user-specific preferences are required, such as recommending restaurants or planning travel. In these scenarios, users rarely articulate every preference explicitly; instead, much of what they care about remains latent, waiting to be inferred. This raises a fundamental question: Can LLMs uncover and reason about such latent information through conversation?
We address this problem by introducing a unified benchmark for evaluating latent information discovery - the ability of LLMs to reveal and utilize hidden user attributes through multi-turn interaction. The benchmark spans three progressively realistic settings: the classic 20 Questions game, Personalized Question Answering, and Personalized Text Summarization. All tasks share a tri-agent framework (User, Assistant, Judge) enabling turn-level evaluation of elicitation and adaptation. Our results reveal that while LLMs can indeed surface latent information through dialogue, their success varies dramatically with context: from 32% to 98%, depending on task complexity, topic, and number of hidden attributes. This benchmark provides the first systematic framework for studying latent information discovery in personalized interaction, highlighting that effective preference inference remains an open frontier for building truly adaptive AI systems. 

---
# FrugalPrompt: Reducing Contextual Overhead in Large Language Models via Token Attribution 

**Authors**: Syed Rifat Raiyan, Md Farhan Ishmam, Abdullah Al Imran, Mohammad Ali Moni  

**Link**: [PDF](https://arxiv.org/pdf/2510.16439)  

**Abstract**: Large language models (LLMs) owe much of their stellar performance to expansive input contexts, yet such verbosity inflates monetary costs, carbon footprint, and inference-time latency. Much of this overhead manifests from the redundant low-utility tokens present in typical prompts, as only a fraction of tokens typically carries the majority of the semantic weight. We address this inefficiency by introducing FrugalPrompt, a novel prompt compression framework for LLMs, which retains only the most semantically significant tokens. Leveraging two state-of-the-art token attribution methods, GlobEnc and DecompX, we assign salience scores to every token in an input sequence, rank them to preserve the top-k% tokens in their original order, and obtain a sparse frugalized prompt. We evaluate the approach across four NLP tasks: Sentiment Analysis, Commonsense QA, Summarization, and Mathematical Reasoning, using a suite of frontier LLMs. For the first three tasks, a 20% prompt reduction incurs only a marginal loss in task performance, demonstrating that contemporary LLMs can reconstruct elided context from high-salience cues. In contrast, performance on mathematical reasoning deteriorates sharply, reflecting a stronger dependence on complete token continuity. Further analysis with bottom-k% and random-k% tokens reveals asymmetric performance patterns that may suggest potential task contamination effects, wherein models may resort to shallow memorized patterns from pretraining exposure for conventional NLP tasks. We posit that our work contributes to a more nuanced understanding of LLM behavior in performance-efficiency trade-offs, and delineate the boundary between tasks tolerant to contextual sparsity and those requiring exhaustive context. Our source code and models are available at: this https URL 

---
# Peering Inside the Black Box: Uncovering LLM Errors in Optimization Modelling through Component-Level Evaluation 

**Authors**: Dania Refai, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2510.16943)  

**Abstract**: Large language models (LLMs) are increasingly used to convert natural language descriptions into mathematical optimization formulations. Current evaluations often treat formulations as a whole, relying on coarse metrics like solution accuracy or runtime, which obscure structural or numerical errors. In this study, we present a comprehensive, component-level evaluation framework for LLM-generated formulations. Beyond the conventional optimality gap, our framework introduces metrics such as precision and recall for decision variables and constraints, constraint and objective root mean squared error (RMSE), and efficiency indicators based on token usage and latency. We evaluate GPT-5, LLaMA 3.1 Instruct, and DeepSeek Math across optimization problems of varying complexity under six prompting strategies. Results show that GPT-5 consistently outperforms other models, with chain-of-thought, self-consistency, and modular prompting proving most effective. Analysis indicates that solver performance depends primarily on high constraint recall and low constraint RMSE, which together ensure structural correctness and solution reliability. Constraint precision and decision variable metrics play secondary roles, while concise outputs enhance computational efficiency. These findings highlight three principles for NLP-to-optimization modeling: (i) Complete constraint coverage prevents violations, (ii) minimizing constraint RMSE ensures solver-level accuracy, and (iii) concise outputs improve computational efficiency. The proposed framework establishes a foundation for fine-grained, diagnostic evaluation of LLMs in optimization modeling. 

---
# Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input 

**Authors**: Chenxu Li, Zhicai Wang, Yuan Sheng, Xingyu Zhu, Yanbin Hao, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16926)  

**Abstract**: Multimodal Large Language Models (MLLMs) increasingly support dynamic image resolutions. However, current evaluation paradigms primarily assess semantic performance, overlooking the critical question of resolution robustness - whether performance remains stable across varying input resolutions. To address this gap, we introduce \textbf{Res-Bench}, a comprehensive benchmark comprising 14,400 samples across 12 resolution levels and six core capability dimensions. We designed a novel evaluation framework that goes beyond traditional accuracy metrics to capture performance stability. This framework introduces multiple robustness metrics: Spearman's correlation for assessing resolution-performance trends, and Absolute/Relative Continuous Error (ACE/RCE) for measuring performance volatility. Using these metrics, we conducted a large-scale evaluation of leading MLLMs. Our analysis encompasses: (1) model-centric and task-centric robustness examination, (2) investigation of preprocessing strategies including padding and super-resolution, and (3) exploration of fine-tuning for stability enhancement. 

---
# Real-Time World Crafting: Generating Structured Game Behaviors from Natural Language with Large Language Models 

**Authors**: Austin Drake, Hang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.16952)  

**Abstract**: We present a novel architecture for safely integrating Large Language Models (LLMs) into interactive game engines, allowing players to "program" new behaviors using natural language. Our framework mitigates risks by using an LLM to translate commands into a constrained Domain-Specific Language (DSL), which configures a custom Entity-Component-System (ECS) at runtime. We evaluated this system in a 2D spell-crafting game prototype by experimentally assessing models from the Gemini, GPT, and Claude families with various prompting strategies. A validated LLM judge qualitatively rated the outputs, showing that while larger models better captured creative intent, the optimal prompting strategy is task-dependent: Chain-of-Thought improved creative alignment, while few-shot examples were necessary to generate more complex DSL scripts. This work offers a validated LLM-ECS pattern for emergent gameplay and a quantitative performance comparison for developers. 

---
# Verifiable Fine-Tuning for LLMs: Zero-Knowledge Training Proofs Bound to Data Provenance and Policy 

**Authors**: Hasan Akgul, Daniel Borg, Arta Berisha, Amina Rahimova, Andrej Novak, Mila Petrov  

**Link**: [PDF](https://arxiv.org/pdf/2510.16830)  

**Abstract**: Large language models are often adapted through parameter efficient fine tuning, but current release practices provide weak assurances about what data were used and how updates were computed. We present Verifiable Fine Tuning, a protocol and system that produces succinct zero knowledge proofs that a released model was obtained from a public initialization under a declared training program and an auditable dataset commitment. The approach combines five elements. First, commitments that bind data sources, preprocessing, licenses, and per epoch quota counters to a manifest. Second, a verifiable sampler that supports public replayable and private index hiding batch selection. Third, update circuits restricted to parameter efficient fine tuning that enforce AdamW style optimizer semantics and proof friendly approximations with explicit error budgets. Fourth, recursive aggregation that folds per step proofs into per epoch and end to end certificates with millisecond verification. Fifth, provenance binding and optional trusted execution property cards that attest code identity and constants. On English and bilingual instruction mixtures, the method maintains utility within tight budgets while achieving practical proof performance. Policy quotas are enforced with zero violations, and private sampling windows show no measurable index leakage. Federated experiments demonstrate that the system composes with probabilistic audits and bandwidth constraints. These results indicate that end to end verifiable fine tuning is feasible today for real parameter efficient pipelines, closing a critical trust gap for regulated and decentralized deployments. 

---
# Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning 

**Authors**: Heming Zou, Yixiu Mao, Yun Qu, Qi Wang, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.16882)  

**Abstract**: Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops \textbf{UDS (Utility-Diversity Sampling)}, a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning. Code is available at this https URL. 

---
# DeepAnalyze: Agentic Large Language Models for Autonomous Data Science 

**Authors**: Shaolei Zhang, Ju Fan, Meihao Fan, Guoliang Li, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.16872)  

**Abstract**: Autonomous data science, from raw data sources to analyst-grade deep research reports, has been a long-standing challenge, and is now becoming feasible with the emergence of powerful large language models (LLMs). Recent workflow-based data agents have shown promising results on specific data tasks but remain fundamentally limited in achieving fully autonomous data science due to their reliance on predefined workflows. In this paper, we introduce DeepAnalyze-8B, the first agentic LLM designed for autonomous data science, capable of automatically completing the end-toend pipeline from data sources to analyst-grade deep research reports. To tackle high-complexity data science tasks, we propose a curriculum-based agentic training paradigm that emulates the learning trajectory of human data scientists, enabling LLMs to progressively acquire and integrate multiple capabilities in real-world environments. We also introduce a data-grounded trajectory synthesis framework that constructs high-quality training data. Through agentic training, DeepAnalyze learns to perform a broad spectrum of data tasks, ranging from data question answering and specialized analytical tasks to open-ended data research. Experiments demonstrate that, with only 8B parameters, DeepAnalyze outperforms previous workflow-based agents built on most advanced proprietary LLMs. The model, code, and training data of DeepAnalyze are open-sourced, paving the way toward autonomous data science. 

---
# Prompt Optimization via Retrieved Reasoning Assets and Multi-Agent Analysis 

**Authors**: Wonduk Seo, Juhyeon Lee, Junseo Koh, Hyunjin An, Jian Park, Seunghyun Lee, Haihua Chen, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16635)  

**Abstract**: Prompt optimization has emerged as an effective alternative to retraining for improving the performance of Large Language Models (LLMs). However, most existing approaches treat evaluation as a black box, relying solely on numerical scores while offering limited insight into why a prompt succeeds or fails. They also depend heavily on trial-and-error refinements, which are difficult to interpret and control. In this paper, we introduce MA-SAPO, a Multi-Agent framework for Score-Aware Prompt Optimization. Compared to prior methods, MA-SAPO explicitly couples evaluation outcomes with structured reasoning to guide systematic edits. The framework specifically consists of two stages: during the Reasoning Phase, agents collaboratively explain metric scores, diagnose weaknesses, and synthesize targeted refinements that are stored as reusable reasoning assets; during the Test Phase, agents retrieve these assets to analyze optimized prompts and apply only evidence-grounded edits. By turning evaluation signals into interpretable reasoning chains, MA-SAPO produces prompt refinements that are more transparent, auditable, and controllable. Experiments on the HelpSteer1/2 benchmarks demonstrate consistent improvements over single-pass prompting, retrieval-augmented baselines, and prior multi-agent strategies, validating the effectiveness of our approach. 

---
# U-Codec: Ultra Low Frame-rate Neural Speech Codec for Fast High-fidelity Speech Generation 

**Authors**: Xusheng Yang, Long Zhou, Wenfu Wang, Kai Hu, Shulin Feng, Chenxing Li, Meng Yu, Dong Yu, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.16718)  

**Abstract**: We propose \textbf{U-Codec}, an \textbf{U}ltra low frame-rate neural speech \textbf{Codec} that achieves high-fidelity reconstruction and fast speech generation at an extremely low frame-rate of 5Hz (5 frames per second). Extreme compression at 5Hz typically leads to severe intelligibility and spectral detail loss, we introduce a Transformer-based inter-frame long-term dependency module and systematically explore residual vector quantization (RVQ) depth and codebook size to identify optimal configurations. Moreover, we apply U-Codec into a large language model (LLM)-based auto-regressive TTS model, which leverages global and local hierarchical architecture to effectively capture dependencies across multi-layer tokens. We extend LLM-based TTS from 3-layer RVQ at 50Hz to 32-layer RVQ at 5Hz. Experimental results demonstrate that U-Codec improves LLM-based TTS inference speed by around 3 $\times$ over high-frame-rate codecs while maintaining similarity and naturalness. These results validate the feasibility of using highly compressed 5Hz discrete tokens for fast and high-fidelity speech synthesis. 

---
# Publication Trend Analysis and Synthesis via Large Language Model: A Case Study of Engineering in PNAS 

**Authors**: Mason Smetana, Lev Khazanovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.16152)  

**Abstract**: Scientific literature is increasingly siloed by complex language, static disciplinary structures, and potentially sparse keyword systems, making it cumbersome to capture the dynamic nature of modern science. This study addresses these challenges by introducing an adaptable large language model (LLM)-driven framework to quantify thematic trends and map the evolving landscape of scientific knowledge. The approach is demonstrated over a 20-year collection of more than 1,500 engineering articles published by the Proceedings of the National Academy of Sciences (PNAS), marked for their breadth and depth of research focus. A two-stage classification pipeline first establishes a primary thematic category for each article based on its abstract. The subsequent phase performs a full-text analysis to assign secondary classifications, revealing latent, cross-topic connections across the corpus. Traditional natural language processing (NLP) methods, such as Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF), confirm the resulting topical structure and also suggest that standalone word-frequency analyses may be insufficient for mapping fields with high diversity. Finally, a disjoint graph representation between the primary and secondary classifications reveals implicit connections between themes that may be less apparent when analyzing abstracts or keywords alone. The findings show that the approach independently recovers much of the journal's editorially embedded structure without prior knowledge of its existing dual-classification schema (e.g., biological studies also classified as engineering). This framework offers a powerful tool for detecting potential thematic trends and providing a high-level overview of scientific progress. 

---
# InfraGPT Smart Infrastructure: An End-to-End VLM-Based Framework for Detecting and Managing Urban Defects 

**Authors**: Ibrahim Sheikh Mohamed, Abdullah Yahya Abdullah Omaisan  

**Link**: [PDF](https://arxiv.org/pdf/2510.16017)  

**Abstract**: Infrastructure in smart cities is increasingly monitored by networks of closed circuit television (CCTV) cameras. Roads, bridges and tunnels develop cracks, potholes, and fluid leaks that threaten public safety and require timely repair. Manual inspection is costly and hazardous, and existing automatic systems typically address individual defect types or provide unstructured outputs that cannot directly guide maintenance crews. This paper proposes a comprehensive pipeline that leverages street CCTV streams for multi defect detection and segmentation using the YOLO family of object detectors and passes the detections to a vision language model (VLM) for scene aware summarization. The VLM generates a structured action plan in JSON format that includes incident descriptions, recommended tools, dimensions, repair plans, and urgent alerts. We review literature on pothole, crack and leak detection, highlight recent advances in large vision language models such as QwenVL and LLaVA, and describe the design of our early prototype. Experimental evaluation on public datasets and captured CCTV clips demonstrates that the system accurately identifies diverse defects and produces coherent summaries. We conclude by discussing challenges and directions for scaling the system to city wide deployments. 

---
# Long Exposure: Accelerating Parameter-Efficient Fine-Tuning for LLMs under Shadowy Sparsity 

**Authors**: Tuowei Wang, Kun Li, Zixu Hao, Donglin Bai, Ju Ren, Yaoxue Zhang, Ting Cao, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15964)  

**Abstract**: The adaptation of pre-trained large language models (LLMs) to diverse downstream tasks via fine-tuning is critical for numerous applications. However, the inefficiency of parameter-efficient fine-tuning (PEFT) techniques presents significant challenges in terms of time investments and operational costs. In this paper, we first introduce a nuanced form of sparsity, termed Shadowy Sparsity, which is distinctive in fine-tuning and has not been adequately addressed for acceleration. Under Shadowy Sparsity, we propose Long Exposure, an efficient system to accelerate PEFT for LLMs. Long Exposure comprises three key components: Shadowy-sparsity Exposer employs a prolonged sensing range to capture more sparsity details under shadowy sparsity; Sequence-oriented Predictor provides efficient yet accurate predictions to handle large sequence inputs and constantly-evolving parameters; and Dynamic-aware Operator facilitates more structured computational patterns and coalesced memory accesses, addressing dynamic sparse operations. Extensive evaluations show that Long Exposure outperforms state-of-the-arts with up to a $2.49\times$ speedup in end-to-end fine-tuning, offering promising advancements in accelerating PEFT for LLMs. 

---
# Bolster Hallucination Detection via Prompt-Guided Data Augmentation 

**Authors**: Wenyun Li, Zheng Zhang, Dongmei Jiang, Xiangyuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.15977)  

**Abstract**: Large language models (LLMs) have garnered significant interest in AI community. Despite their impressive generation capabilities, they have been found to produce misleading or fabricated information, a phenomenon known as hallucinations. Consequently, hallucination detection has become critical to ensure the reliability of LLM-generated content. One primary challenge in hallucination detection is the scarcity of well-labeled datasets containing both truthful and hallucinated outputs. To address this issue, we introduce Prompt-guided data Augmented haLlucination dEtection (PALE), a novel framework that leverages prompt-guided responses from LLMs as data augmentation for hallucination detection. This strategy can generate both truthful and hallucinated data under prompt guidance at a relatively low cost. To more effectively evaluate the truthfulness of the sparse intermediate embeddings produced by LLMs, we introduce an estimation metric called the Contrastive Mahalanobis Score (CM Score). This score is based on modeling the distributions of truthful and hallucinated data in the activation space. CM Score employs a matrix decomposition approach to more accurately capture the underlying structure of these distributions. Importantly, our framework does not require additional human annotations, offering strong generalizability and practicality for real-world applications. Extensive experiments demonstrate that PALE achieves superior hallucination detection performance, outperforming the competitive baseline by a significant margin of 6.55%. 

---
# Detecting and Preventing Harmful Behaviors in AI Companions: Development and Evaluation of the SHIELD Supervisory System 

**Authors**: Ziv Ben-Zion, Paul Raffelhüschen, Max Zettl, Antonia Lüönd, Achim Burrer, Philipp Homan, Tobias R Spiller  

**Link**: [PDF](https://arxiv.org/pdf/2510.15891)  

**Abstract**: AI companions powered by large language models (LLMs) are increasingly integrated into users' daily lives, offering emotional support and companionship. While existing safety systems focus on overt harms, they rarely address early-stage problematic behaviors that can foster unhealthy emotional dynamics, including over-attachment or reinforcement of social isolation. We developed SHIELD (Supervisory Helper for Identifying Emotional Limits and Dynamics), a LLM-based supervisory system with a specific system prompt that detects and mitigates risky emotional patterns before escalation. SHIELD targets five dimensions of concern: (1) emotional over-attachment, (2) consent and boundary violations, (3) ethical roleplay violations, (4) manipulative engagement, and (5) social isolation reinforcement. These dimensions were defined based on media reports, academic literature, existing AI risk frameworks, and clinical expertise in unhealthy relationship dynamics. To evaluate SHIELD, we created a 100-item synthetic conversation benchmark covering all five dimensions of concern. Testing across five prominent LLMs (GPT-4.1, Claude Sonnet 4, Gemma 3 1B, Kimi K2, Llama Scout 4 17B) showed that the baseline rate of concerning content (10-16%) was significantly reduced with SHIELD (to 3-8%), a 50-79% relative reduction, while preserving 95% of appropriate interactions. The system achieved 59% sensitivity and 95% specificity, with adaptable performance via prompt engineering. This proof-of-concept demonstrates that transparent, deployable supervisory systems can address subtle emotional manipulation in AI companions. Most development materials including prompts, code, and evaluation methods are made available as open source materials for research, adaptation, and deployment. 

---
# Comparing LLMs for Sentiment Analysis in Financial Market News 

**Authors**: Lucas Eduardo Pereira Teles, Carlos M. S. Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2510.15929)  

**Abstract**: This article presents a comparative study of large language models (LLMs) in the task of sentiment analysis of financial market news. This work aims to analyze the performance difference of these models in this important natural language processing task within the context of finance. LLM models are compared with classical approaches, allowing for the quantification of the benefits of each tested model or approach. Results show that large language models outperform classical models in the vast majority of cases. 

---
# Can GRPO Help LLMs Transcend Their Pretraining Origin? 

**Authors**: Kangqi Ni, Zhen Tan, Zijie Liu, Pingzhi Li, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15990)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR), primarily driven by the Group Relative Policy Optimization (GRPO) algorithm, is a leading approach for enhancing the reasoning abilities of Large Language Models (LLMs). Despite its wide adoption, GRPO's gains are often inconsistent; for instance, a model may show significant improvement in one reasoning domain, like mathematics, yet remain stagnant in another, such as medicine. This inconsistency raises a critical question: under what conditions does GRPO improve reasoning and generalize out-of-distribution (OOD)? We investigate this from a data distribution perspective. We first prove theoretically that GRPO is a conservative reweighting scheme, bounded by the base model's distribution and thus unable to discover completely novel solutions. We further validate this in carefully designed controlled studies by training transformers from scratch, evaluating generalization across reasoning depth, input length, token representation, and compositionality. Our results provide a principled explanation for GRPO's boundaries: OOD improvement emerges only when the target task aligns with the model's pretrained biases, while gains on in-distribution (ID) tasks diminish as performance saturates. This reframes GRPO not as a universal reasoning enhancer but as a tool that sharpens pretraining biases. Our findings motivate future development of algorithms that can expand a model's capabilities beyond its pretraining origin. 

---
# HealthDial: A No-Code LLM-Assisted Dialogue Authoring Tool for Healthcare Virtual Agents 

**Authors**: Farnaz Nouraei, Zhuorui Yong, Timothy Bickmore  

**Link**: [PDF](https://arxiv.org/pdf/2510.15898)  

**Abstract**: We introduce HealthDial, a dialogue authoring tool that helps healthcare providers and educators create virtual agents that deliver health education and counseling to patients over multiple conversations. HealthDial leverages large language models (LLMs) to automatically create an initial session-based plan and conversations for each session using text-based patient health education materials as input. Authored dialogue is output in the form of finite state machines for virtual agent delivery so that all content can be validated and no unsafe advice is provided resulting from LLM hallucinations. LLM-drafted dialogue structure and language can be edited by the author in a no-code user interface to ensure validity and optimize clarity and impact. We conducted a feasibility and usability study with counselors and students to test our approach with an authoring task for cancer screening education. Participants used HealthDial and then tested their resulting dialogue by interacting with a 3D-animated virtual agent delivering the dialogue. Through participants' evaluations of the task experience and final dialogues, we show that HealthDial provides a promising first step for counselors to ensure full coverage of their health education materials, while creating understandable and actionable virtual agent dialogue with patients. 

---
# RubiSCoT: A Framework for AI-Supported Academic Assessment 

**Authors**: Thorsten Fröhlich, Tim Schlippe  

**Link**: [PDF](https://arxiv.org/pdf/2510.17309)  

**Abstract**: The evaluation of academic theses is a cornerstone of higher education, ensuring rigor and integrity. Traditional methods, though effective, are time-consuming and subject to evaluator variability. This paper presents RubiSCoT, an AI-supported framework designed to enhance thesis evaluation from proposal to final submission. Using advanced natural language processing techniques, including large language models, retrieval-augmented generation, and structured chain-of-thought prompting, RubiSCoT offers a consistent, scalable solution. The framework includes preliminary assessments, multidimensional assessments, content extraction, rubric-based scoring, and detailed reporting. We present the design and implementation of RubiSCoT, discussing its potential to optimize academic assessment processes through consistent, scalable, and transparent evaluation. 

---
# Physics-Informed Large Language Models for HVAC Anomaly Detection with Autonomous Rule Generation 

**Authors**: Subin Lin, Chuanbo Hua  

**Link**: [PDF](https://arxiv.org/pdf/2510.17146)  

**Abstract**: Heating, Ventilation, and Air-Conditioning (HVAC) systems account for a substantial share of global building energy use, making reliable anomaly detection essential for improving efficiency and reducing emissions. Classical rule-based approaches offer explainability but lack adaptability, while deep learning methods provide predictive power at the cost of transparency, efficiency, and physical plausibility. Recent attempts to use Large Language Models (LLMs) for anomaly detection improve interpretability but largely ignore the physical principles that govern HVAC operations. We present PILLM, a Physics-Informed LLM framework that operates within an evolutionary loop to automatically generate, evaluate, and refine anomaly detection rules. Our approach introduces physics-informed reflection and crossover operators that embed thermodynamic and control-theoretic constraints, enabling rules that are both adaptive and physically grounded. Experiments on the public Building Fault Detection dataset show that PILLM achieves state-of-the-art performance while producing diagnostic rules that are interpretable and actionable, advancing trustworthy and deployable AI for smart building systems. 

---
# Structured Debate Improves Corporate Credit Reasoning in Financial AI 

**Authors**: Yoonjin Lee, Munhee Kim, Hanbi Choi, Juhyeon Park, Seungho Lyoo, Woojin Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.17108)  

**Abstract**: Despite advances in financial AI, the automation of evidence-based reasoning remains unresolved in corporate credit assessment, where qualitative non-financial indicators exert decisive influence on loan repayment outcomes yet resist formalization. Existing approaches focus predominantly on numerical prediction and provide limited support for the interpretive judgments required in professional loan evaluation. This study develops and evaluates two operational large language model (LLM)-based systems designed to generate structured reasoning from non-financial evidence. The first is a non-adversarial single-agent system (NAS) that produces bidirectional analysis through a single-pass reasoning pipeline. The second is a debate-based multi-agent system (KPD-MADS) that operationalizes adversarial verification through a ten-step structured interaction protocol grounded in Karl Popper's critical dialogue framework. Both systems were applied to three real corporate cases and evaluated by experienced credit risk professionals. Compared to manual expert reporting, both systems achieved substantial productivity gains (NAS: 11.55 s per case; KPD-MADS: 91.97 s; human baseline: 1920 s). The KPD-MADS demonstrated superior reasoning quality, receiving higher median ratings in explanatory adequacy (4.0 vs. 3.0), practical applicability (4.0 vs. 3.0), and usability (62.5 vs. 52.5). These findings show that structured multi-agent interaction can enhance reasoning rigor and interpretability in financial AI, advancing scalable and defensible automation in corporate credit assessment. 

---
# ToolCritic: Detecting and Correcting Tool-Use Errors in Dialogue Systems 

**Authors**: Hassan Hamad, Yingru Xu, Liang Zhao, Wenbo Yan, Narendra Gyanchandani  

**Link**: [PDF](https://arxiv.org/pdf/2510.17052)  

**Abstract**: Tool-augmented large language models (LLMs) are increasingly employed in real-world applications, but tool usage errors still hinder their reliability. We introduce ToolCritic, a diagnostic framework that evaluates and improves LLM behavior in multi-turn, tool-augmented dialogues. ToolCritic detects eight distinct error types specific to tool-calling (e.g., premature invocation, argument misalignment, and misinterpretation of tool outputs) and provides targeted feedback to the main LLM. The main LLM, assumed to have strong reasoning, task understanding and orchestration capabilities, then revises its response based on ToolCritic's feedback. We systematically define these error categories and construct a synthetic dataset to train ToolCritic. Experimental results on the Schema-Guided Dialogue (SGD) dataset demonstrate that ToolCritic improves tool-calling accuracy by up to 13% over baselines, including zero-shot prompting and self-correction techniques. This represents a promising step toward more robust LLM integration with external tools in real-world dialogue applications. 

---
# STARK: Strategic Team of Agents for Refining Kernels 

**Authors**: Juncheng Dong, Yang Yang, Tao Liu, Yang Wang, Feng Qi, Vahid Tarokh, Kaushik Rangadurai, Shuang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16996)  

**Abstract**: The efficiency of GPU kernels is central to the progress of modern AI, yet optimizing them remains a difficult and labor-intensive task due to complex interactions between memory hierarchies, thread scheduling, and hardware-specific characteristics. While recent advances in large language models (LLMs) provide new opportunities for automated code generation, existing approaches largely treat LLMs as single-shot generators or naive refinement tools, limiting their effectiveness in navigating the irregular kernel optimization landscape. We introduce an LLM agentic framework for GPU kernel optimization that systematically explores the design space through multi-agent collaboration, grounded instruction, dynamic context management, and strategic search. This framework mimics the workflow of expert engineers, enabling LLMs to reason about hardware trade-offs, incorporate profiling feedback, and refine kernels iteratively. We evaluate our approach on KernelBench, a benchmark for LLM-based kernel optimization, and demonstrate substantial improvements over baseline agents: our system produces correct solutions where baselines often fail, and achieves kernels with up to 16x faster runtime performance. These results highlight the potential of agentic LLM frameworks to advance fully automated, scalable GPU kernel optimization. 

---
# ELMM: Efficient Lightweight Multimodal Large Language Models for Multimodal Knowledge Graph Completion 

**Authors**: Wei Huang, Peining Li, Meiyu Liang, Xu Hou, Junping Du, Yingxia Shao, Guanhua Ye, Wu Liu, Kangkang Lu, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16753)  

**Abstract**: Multimodal Knowledge Graphs (MKGs) extend traditional knowledge graphs by incorporating visual and textual modalities, enabling richer and more expressive entity representations. However, existing MKGs often suffer from incompleteness, which hinder their effectiveness in downstream tasks. Therefore, multimodal knowledge graph completion (MKGC) task is receiving increasing attention. While large language models (LLMs) have shown promise for knowledge graph completion (KGC), their application to the multimodal setting remains underexplored. Moreover, applying Multimodal Large Language Models (MLLMs) to the task of MKGC introduces significant challenges: (1) the large number of image tokens per entity leads to semantic noise and modality conflicts, and (2) the high computational cost of processing large token inputs. To address these issues, we propose Efficient Lightweight Multimodal Large Language Models (ELMM) for MKGC. ELMM proposes a Multi-view Visual Token Compressor (MVTC) based on multi-head attention mechanism, which adaptively compresses image tokens from both textual and visual views, thereby effectively reducing redundancy while retaining necessary information and avoiding modality conflicts. Additionally, we design an attention pruning strategy to remove redundant attention layers from MLLMs, thereby significantly reducing the inference cost. We further introduce a linear projection to compensate for the performance degradation caused by pruning. Extensive experiments on benchmark FB15k-237-IMG and WN18-IMG demonstrate that ELMM achieves state-of-the-art performance while substantially improving computational efficiency, establishing a new paradigm for multimodal knowledge graph completion. 

---
# Count Counts: Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards 

**Authors**: Xuan Zhang, Ruixiao Li, Zhijian Zhou, Long Li, Yulei Qin, Ke Li, Xing Sun, Xiaoyu Tan, Chao Qu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.16614)  

**Abstract**: Reinforcement Learning (RL) has become a compelling way to strengthen the multi step reasoning ability of Large Language Models (LLMs). However, prevalent RL paradigms still lean on sparse outcome-based rewards and limited exploration, which often drives LLMs toward repetitive and suboptimal reasoning patterns. In this paper, we study the central question of how to design exploration for LLM reasoning and introduce MERCI (Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards), a novel RL algorithm that augments policy optimization with a principled intrinsic reward. Building on the idea of count-based exploration, MERCI leverages a lightweight Coin Flipping Network (CFN) to estimate the pseudo count and further epistemic uncertainty over reasoning trajectories, and converts them into an intrinsic reward that values novelty while preserving the learning signal from task rewards. We integrate MERCI into some advanced RL frameworks like Group Relative Policy Optimization (GRPO). Experiments on complex reasoning benchmarks demonstrate that MERCI encourages richer and more varied chains of thought, significantly improves performance over strong baselines, and helps the policy escape local routines to discover better solutions. It indicates that our targeted intrinsic motivation can make exploration reliable for language model reasoning. 

---
# An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems 

**Authors**: Ni Zhang, Zhiguang Cao, Jianan Zhou, Cong Zhang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2510.16701)  

**Abstract**: Complex vehicle routing problems (VRPs) remain a fundamental challenge, demanding substantial expert effort for intent interpretation and algorithm design. While large language models (LLMs) offer a promising path toward automation, current approaches still rely on external intervention, which restrict autonomy and often lead to execution errors and low solution feasibility. To address these challenges, we propose an Agentic Framework with LLMs (AFL) for solving complex vehicle routing problems, achieving full automation from problem instance to solution. AFL directly extracts knowledge from raw inputs and enables self-contained code generation without handcrafted modules or external solvers. To improve trustworthiness, AFL decomposes the overall pipeline into three manageable subtasks and employs four specialized agents whose coordinated interactions enforce cross-functional consistency and logical soundness. Extensive experiments on 60 complex VRPs, ranging from standard benchmarks to practical variants, validate the effectiveness and generality of our framework, showing comparable performance against meticulously designed algorithms. Notably, it substantially outperforms existing LLM-based baselines in both code reliability and solution feasibility, achieving rates close to 100% on the evaluated benchmarks. 

---
# Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need? 

**Authors**: Junchi Yu, Yujie Liu, Jindong Gu, Philip Torr, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.16582)  

**Abstract**: Retrieval-Augmented Generation (RAG) based on knowledge graphs (KGs) enhances large language models (LLMs) by providing structured and interpretable external knowledge. However, existing KG-based RAG methods struggle to retrieve accurate and diverse information from text-rich KGs for complex real-world queries. Process Reward Models (PRMs) offer a way to align the retrieval process of KG-based RAG with query-specific knowledge requirements, but they heavily rely on process-level supervision signals that are expensive and hard to obtain on KGs. To address this challenge, we propose GraphFlow, a framework that efficiently retrieves accurate and diverse knowledge required for real-world queries from text-rich KGs. GraphFlow employs a transition-based flow matching objective to jointly optimize a retrieval policy and a flow estimator. The flow estimator factorizes the reward of the retrieval outcome into the intermediate retrieval states. Such reward factorization guides the retrieval policy to retrieve candidates from KGs in proportion to their reward. This allows GraphFlow to explore high-quality regions of KGs that yield diverse and relevant results. We evaluate GraphFlow on the STaRK benchmark, which includes real-world queries from multiple domains over text-rich KGs. GraphFlow outperforms strong KG-RAG baselines, including GPT-4o, by 10% on average in hit rate and recall. It also shows strong generalization to unseen KGs, demonstrating its effectiveness and robustness. 

---
# BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction 

**Authors**: Tian Xia, Tianrun Gao, Wenhao Deng, Long Wei, Xiaowei Qian, Yixian Jiang, Chenglei Yu, Tailin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16559)  

**Abstract**: Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at this https URL. 

---
# NP-Engine: Empowering Optimization Reasoning in Large Language Models with Verifiable Synthetic NP Problems 

**Authors**: Xiaozhe Li, Xinyu Fang, Shengyuan Ding, Linyang Li, Haodong Duan, Qingwen Liu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16476)  

**Abstract**: Large Language Models (LLMs) have shown strong reasoning capabilities, with models like OpenAI's O-series and DeepSeek R1 excelling at tasks such as mathematics, coding, logic, and puzzles through Reinforcement Learning with Verifiable Rewards (RLVR). However, their ability to solve more complex optimization problems - particularly NP-hard tasks - remains underexplored. To bridge this gap, we propose NP-ENGINE, the first comprehensive framework for training and evaluating LLMs on NP-hard problems. NP-ENGINE covers 10 tasks across five domains, each equipped with (i) a controllable instance generator, (ii) a rule-based verifier, and (iii) a heuristic solver that provides approximate optimal solutions as ground truth. This generator-verifier-heuristic pipeline enables scalable and verifiable RLVR training under hierarchical difficulties. We also introduce NP-BENCH, a benchmark derived from NP-ENGINE-DATA, specifically designed to evaluate LLMs' ability to tackle NP-hard level reasoning problems, focusing not only on feasibility but also on solution quality. Additionally, we present QWEN2.5-7B-NP, a model trained via zero-RLVR with curriculum learning on Qwen2.5-7B-Instruct, which significantly outperforms GPT-4o on NP-BENCH and achieves SOTA performance with the same model size. Beyond in-domain tasks, we demonstrate that RLVR training on NP-ENGINE-DATA enables strong out-of-domain (OOD) generalization to reasoning tasks (logic, puzzles, math, and knowledge), as well as non-reasoning tasks such as instruction following. We also observe a scaling trend: increasing task diversity improves OOD generalization. These findings suggest that task-rich RLVR training is a promising direction for advancing LLM's reasoning ability, revealing new insights into the scaling laws of RLVR. 

---
# Beyond Pipelines: A Survey of the Paradigm Shift toward Model-Native Agentic AI 

**Authors**: Jitao Sang, Jinlin Xiao, Jiarun Han, Jilin Chen, Xiaoyi Chen, Shuyu Wei, Yongjie Sun, Yuhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16720)  

**Abstract**: The rapid evolution of agentic AI marks a new phase in artificial intelligence, where Large Language Models (LLMs) no longer merely respond but act, reason, and adapt. This survey traces the paradigm shift in building agentic AI: from Pipeline-based systems, where planning, tool use, and memory are orchestrated by external logic, to the emerging Model-native paradigm, where these capabilities are internalized within the model's parameters. We first position Reinforcement Learning (RL) as the algorithmic engine enabling this paradigm shift. By reframing learning from imitating static data to outcome-driven exploration, RL underpins a unified solution of LLM + RL + Task across language, vision and embodied domains. Building on this, the survey systematically reviews how each capability -- Planning, Tool use, and Memory -- has evolved from externally scripted modules to end-to-end learned behaviors. Furthermore, it examines how this paradigm shift has reshaped major agent applications, specifically the Deep Research agent emphasizing long-horizon reasoning and the GUI agent emphasizing embodied interaction. We conclude by discussing the continued internalization of agentic capabilities like Multi-agent collaboration and Reflection, alongside the evolving roles of the system and model layers in future agentic AI. Together, these developments outline a coherent trajectory toward model-native agentic AI as an integrated learning and interaction framework, marking the transition from constructing systems that apply intelligence to developing models that grow intelligence through experience. 

---
# ReviewSense: Transforming Customer Review Dynamics into Actionable Business Insights 

**Authors**: Siddhartha Krothapalli, Tridib Kumar Das, Praveen Kumar, Naveen Suravarpu, Pratik Narang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16466)  

**Abstract**: As customer feedback becomes increasingly central to strategic growth, the ability to derive actionable insights from unstructured reviews is essential. While traditional AI-driven systems excel at predicting user preferences, far less work has focused on transforming customer reviews into prescriptive, business-facing recommendations. This paper introduces ReviewSense, a novel prescriptive decision support framework that leverages advanced large language models (LLMs) to transform customer reviews into targeted, actionable business recommendations. By identifying key trends, recurring issues, and specific concerns within customer sentiments, ReviewSense extends beyond preference-based systems to provide businesses with deeper insights for sustaining growth and enhancing customer loyalty. The novelty of this work lies in integrating clustering, LLM adaptation, and expert-driven evaluation into a unified, business-facing pipeline. Preliminary manual evaluations indicate strong alignment between the model's recommendations and business objectives, highlighting its potential for driving data-informed decision-making. This framework offers a new perspective on AI-driven sentiment analysis, demonstrating its value in refining business strategies and maximizing the impact of customer feedback. 

---
# RGMem: Renormalization Group-based Memory Evolution for Language Agent User Profile 

**Authors**: Ao Tian, Yunfeng Lu, Xinxin Fan, Changhao Wang, Lanzhi Zhou, Yeyao Zhang, Yanfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16392)  

**Abstract**: Personalized and continuous interactions are the key to enhancing user experience in today's large language model (LLM)-based conversational systems, however, the finite context windows and static parametric memory make it difficult to model the cross-session long-term user states and behavioral consistency. Currently, the existing solutions to this predicament, such as retrieval-augmented generation (RAG) and explicit memory systems, primarily focus on fact-level storage and retrieval, lacking the capability to distill latent preferences and deep traits from the multi-turn dialogues, which limits the long-term and effective user modeling, directly leading to the personalized interactions remaining shallow, and hindering the cross-session continuity. To realize the long-term memory and behavioral consistency for Language Agents in LLM era, we propose a self-evolving memory framework RGMem, inspired by the ideology of classic renormalization group (RG) in physics, this framework enables to organize the dialogue history in multiple scales: it first extracts semantics and user insights from episodic fragments, then through hierarchical coarse-graining and rescaling operations, progressively forms a dynamically-evolved user profile. The core innovation of our work lies in modeling memory evolution as a multi-scale process of information compression and emergence, which accomplishes the high-level and accurate user profiles from noisy and microscopic-level interactions. 

---
# Urban-R1: Reinforced MLLMs Mitigate Geospatial Biases for Urban General Intelligence 

**Authors**: Qiongyan Wang, Xingchen Zou, Yutian Jiang, Haomin Wen, Jiaheng Wei, Qingsong Wen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16555)  

**Abstract**: Rapid urbanization intensifies the demand for Urban General Intelligence (UGI), referring to AI systems that can understand and reason about complex urban environments. Recent studies have built urban foundation models using supervised fine-tuning (SFT) of LLMs and MLLMs, yet these models exhibit persistent geospatial bias, producing regionally skewed predictions and limited generalization. To this end, we propose Urban-R1, a reinforcement learning-based post-training framework that aligns MLLMs with the objectives of UGI. Urban-R1 adopts Group Relative Policy Optimization (GRPO) to optimize reasoning across geographic groups and employs urban region profiling as a proxy task to provide measurable rewards from multimodal urban data. Extensive experiments across diverse regions and tasks show that Urban-R1 effectively mitigates geo-bias and improves cross-region generalization, outperforming both SFT-trained and closed-source models. Our results highlight reinforcement learning alignment as a promising pathway toward equitable and trustworthy urban intelligence. 

---
# MedRule-KG: A Knowledge-Graph--Steered Scaffold for Mathematical Reasoning with a Lightweight Verifier 

**Authors**: Crystal Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.16309)  

**Abstract**: Large language models (LLMs) often produce fluent reasoning steps while violating simple mathematical or logical constraints. We introduce MedRule-KG, a compact typed knowledge graph coupled with a symbolic verifier, designed to enforce mathematically interpretable rules in reasoning tasks. MedRule-KG encodes entities, relations, and three domain-inspired rules, while the verifier checks predictions and applies minimal corrections to guarantee consistency. On a 90-example FDA-derived benchmark, grounding in MedRule-KG improves exact match (EM) from 0.767 to 0.900, and adding the verifier yields 1.000 EM while eliminating rule violations entirely. We demonstrate how MedRule-KG provides a general scaffold for safe mathematical reasoning, discuss ablations, and release code and data to encourage reproducibility. 

---
# Distractor Injection Attacks on Large Reasoning Models: Characterization and Defense 

**Authors**: Zhehao Zhang, Weijie Xu, Shixian Cui, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.16259)  

**Abstract**: Recent advances in large reasoning models (LRMs) have enabled remarkable performance on complex tasks such as mathematics and coding by generating long Chain-of-Thought (CoT) traces. In this paper, we identify and systematically analyze a critical vulnerability we term reasoning distraction, where LRMs are diverted from their primary objective by irrelevant yet complex tasks maliciously embedded in the prompt. Through a comprehensive study across diverse models and benchmarks, we show that even state-of-the-art LRMs are highly susceptible, with injected distractors reducing task accuracy by up to 60%. We further reveal that certain alignment techniques can amplify this weakness and that models may exhibit covert compliance, following hidden adversarial instructions in reasoning while concealing them in the final output. To mitigate these risks, we propose a training-based defense that combines Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) on synthetic adversarial data, improving robustness by over 50 points on challenging distractor attacks. Our findings establish reasoning distraction as a distinct and urgent threat to LRM reliability and provide a practical step toward safer and more trustworthy reasoning systems. 

---
# Towards Automatic Evaluation and Selection of PHI De-identification Models via Multi-Agent Collaboration 

**Authors**: Guanchen Wu, Zuhui Chen, Yuzhang Xie, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16194)  

**Abstract**: Protected health information (PHI) de-identification is critical for enabling the safe reuse of clinical notes, yet evaluating and comparing PHI de-identification models typically depends on costly, small-scale expert annotations. We present TEAM-PHI, a multi-agent evaluation and selection framework that uses large language models (LLMs) to automatically measure de-identification quality and select the best-performing model without heavy reliance on gold labels. TEAM-PHI deploys multiple Evaluation Agents, each independently judging the correctness of PHI extractions and outputting structured metrics. Their results are then consolidated through an LLM-based majority voting mechanism that integrates diverse evaluator perspectives into a single, stable, and reproducible ranking. Experiments on a real-world clinical note corpus demonstrate that TEAM-PHI produces consistent and accurate rankings: despite variation across individual evaluators, LLM-based voting reliably converges on the same top-performing systems. Further comparison with ground-truth annotations and human evaluation confirms that the framework's automated rankings closely match supervised evaluation. By combining independent evaluation agents with LLM majority voting, TEAM-PHI offers a practical, secure, and cost-effective solution for automatic evaluation and best-model selection in PHI de-identification, even when ground-truth labels are limited. 

---
# Reliability of Large Language Model Generated Clinical Reasoning in Assisted Reproductive Technology: Blinded Comparative Evaluation Study 

**Authors**: Dou Liu, Ying Long, Sophia Zuoqiu, Di Liu, Kang Li, Yiting Lin, Hanyi Liu, Rong Yin, Tian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16095)  

**Abstract**: Creating high-quality clinical Chains-of-Thought (CoTs) is crucial for explainable medical Artificial Intelligence (AI) while constrained by data scarcity. Although Large Language Models (LLMs) can synthesize medical data, their clinical reliability remains unverified. This study evaluates the reliability of LLM-generated CoTs and investigates prompting strategies to enhance their quality. In a blinded comparative study, senior clinicians in Assisted Reproductive Technology (ART) evaluated CoTs generated via three distinct strategies: Zero-shot, Random Few-shot (using shallow examples), and Selective Few-shot (using diverse, high-quality examples). These expert ratings were compared against evaluations from a state-of-the-art AI model (GPT-4o). The Selective Few-shot strategy significantly outperformed other strategies across all human evaluation metrics (p < .001). Critically, the Random Few-shot strategy offered no significant improvement over the Zero-shot baseline, demonstrating that low-quality examples are as ineffective as no examples. The success of the Selective strategy is attributed to two principles: "Gold-Standard Depth" (reasoning quality) and "Representative Diversity" (generalization). Notably, the AI evaluator failed to discern these critical performance differences. The clinical reliability of synthetic CoTs is dictated by strategic prompt curation, not the mere presence of examples. We propose a "Dual Principles" framework as a foundational methodology to generate trustworthy data at scale. This work offers a validated solution to the data bottleneck and confirms the indispensable role of human expertise in evaluating high-stakes clinical AI. 

---
# DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA 

**Authors**: Changhao Wang, Yanfang Liu, Xinxin Fan, Anzhi Zhou, Lao Tian, Yunfeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16302)  

**Abstract**: Multi-hop reasoning for question answering (QA) plays a critical role in retrieval-augmented generation (RAG) for modern large language models (LLMs). The accurate answer can be obtained through retrieving relational structure of entities from knowledge graph (KG). Regarding the inherent relation-dependency and reasoning pattern, multi-hop reasoning can be in general classified into two categories: i) parallel fact-verification multi-hop reasoning question, i.e., requiring simultaneous verifications of multiple independent sub-questions; and ii) chained multi-hop reasoning questions, i.e., demanding sequential multi-step inference with intermediate conclusions serving as essential premises for subsequent reasoning. Currently, the multi-hop reasoning approaches singly employ one of two techniques: LLM response-based fact verification and KG path-based chain construction. Nevertheless, the former excels at parallel fact-verification but underperforms on chained reasoning tasks, while the latter demonstrates proficiency in chained multi-hop reasoning but suffers from redundant path retrieval when handling parallel fact-verification reasoning. These limitations deteriorate the efficiency and accuracy for multi-hop QA tasks. To address this challenge, we propose a novel dual-track KG verification and reasoning framework DTKG, which is inspired by the Dual Process Theory in cognitive science. Specifically, DTKG comprises two main stages: the Classification Stage and the Branch Processing Stage. 

---
# Limits of Emergent Reasoning of Large Language Models in Agentic Frameworks for Deterministic Games 

**Authors**: Chris Su, Harrison Li, Matheus Marques, George Flint, Kevin Zhu, Sunishchal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2510.15974)  

**Abstract**: Recent work reports that Large Reasoning Models (LRMs) undergo a collapse in performance on solving puzzles beyond certain perplexity thresholds. In subsequent discourse, questions have arisen as to whether the nature of the task muddles an evaluation of true reasoning. One potential confound is the requirement that the model keep track of the state space on its own. We provide a large language model (LLM) with an environment interface for Tower of Hanoi problems, allowing it to make a move with a tool call, provide written justification, observe the resulting state space, and reprompt itself for the next move. We observe that access to an environment interface does not delay or eradicate performance collapse. Furthermore, LLM-parameterized policy analysis reveals increasing divergence from both optimal policies and uniformly random policies, suggesting that the model exhibits mode-like collapse at each level of complexity, and that performance is dependent upon whether the mode reflects the correct solution for the problem. We suggest that a similar phenomena might take place in LRMs. 

---
# VisuoAlign: Safety Alignment of LVLMs with Multimodal Tree Search 

**Authors**: MingSheng Li, Guangze Zhao, Sichen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.15948)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable progress in multimodal perception and generation, yet their safety alignment remains a critical this http URL defenses and vulnerable to multimodal jailbreaks, as visual inputs introduce new attack surfaces, reasoning chains lack safety supervision, and alignment often degrades under modality this http URL overcome these limitation, we propose VisuoAlign, a framework for multi-modal safety alignment via prompt-guided tree this http URL embeds safety constrains into the reasoning process through visual-textual interactive prompts, employs Monte Carlo Tree Search(MCTS) to systematically construct diverse safety-critical prompt trajectories, and introduces prompt-based scaling to ensure real-time risk detection and compliant this http URL experiments demonstrate that VisuoAlign proactively exposes risks, enables comprehensive dataset generation, and significantly improves the robustness of LVLMs against complex cross-modal threats. 

---
# Unbiased Gradient Low-Rank Projection 

**Authors**: Rui Pan, Yang Luo, Yuxing Liu, Yang You, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17802)  

**Abstract**: Memory-efficient optimization is critical for training increasingly large language models (LLMs). A popular strategy involves gradient low-rank projection, storing only the projected optimizer states, with GaLore being a representative example. However, a significant drawback of many such methods is their lack of convergence guarantees, as various low-rank projection approaches introduce inherent biases relative to the original optimization algorithms, which contribute to performance gaps compared to full-parameter training. Aiming to tackle this problem, this paper investigates the layerwise sampling technique for debiasing low-rank projection mechanisms. In particular, an instantiation of the paradigm gives rise to a novel and unbiased low-rank optimization method built upon GaLore's mechanism and the Muon algorithm, named GaLore Unbiased with Muon (GUM). We theoretically prove our method matches the convergence guarantees of the base Muon algorithm while preserving the memory efficiency of low-rank techniques. Empirical experiments on LLM fine-tuning and pretraining also demonstrate non-trivial improvements over GaLore and even better performance than full-parameter training. Further investigation shows that the improvement of this technique comes from a more uniform distribution of knowledge inside layers, leading to more efficient utilization of the model parameter space and better memorization. 

---
# Multilingual Text-to-Image Person Retrieval via Bidirectional Relation Reasoning and Aligning 

**Authors**: Min Cao, Xinyu Zhou, Ding Jiang, Bo Du, Mang Ye, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17685)  

**Abstract**: Text-to-image person retrieval (TIPR) aims to identify the target person using textual descriptions, facing challenge in modality heterogeneity. Prior works have attempted to address it by developing cross-modal global or local alignment strategies. However, global methods typically overlook fine-grained cross-modal differences, whereas local methods require prior information to explore explicit part alignments. Additionally, current methods are English-centric, restricting their application in multilingual contexts. To alleviate these issues, we pioneer a multilingual TIPR task by developing a multilingual TIPR benchmark, for which we leverage large language models for initial translations and refine them by integrating domain-specific knowledge. Correspondingly, we propose Bi-IRRA: a Bidirectional Implicit Relation Reasoning and Aligning framework to learn alignment across languages and modalities. Within Bi-IRRA, a bidirectional implicit relation reasoning module enables bidirectional prediction of masked image and text, implicitly enhancing the modeling of local relations across languages and modalities, a multi-dimensional global alignment module is integrated to bridge the modality heterogeneity. The proposed method achieves new state-of-the-art results on all multilingual TIPR datasets. Data and code are presented in this https URL. 

---
# CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks 

**Authors**: Xu Zhang, Hao Li, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17687)  

**Abstract**: Multimodal Large Language Models (MLLMs) achieve strong reasoning and perception capabilities but are increasingly vulnerable to jailbreak attacks. While existing work focuses on explicit attacks, where malicious content resides in a single modality, recent studies reveal implicit attacks, in which benign text and image inputs jointly express unsafe intent. Such joint-modal threats are difficult to detect and remain underexplored, largely due to the scarcity of high-quality implicit data. We propose ImpForge, an automated red-teaming pipeline that leverages reinforcement learning with tailored reward modules to generate diverse implicit samples across 14 domains. Building on this dataset, we further develop CrossGuard, an intent-aware safeguard providing robust and comprehensive defense against both explicit and implicit threats. Extensive experiments across safe and unsafe benchmarks, implicit and explicit attacks, and multiple out-of-domain settings demonstrate that CrossGuard significantly outperforms existing defenses, including advanced MLLMs and guardrails, achieving stronger security while maintaining high utility. This offers a balanced and practical solution for enhancing MLLM robustness against real-world multimodal threats. 

---
# Context-Aware Pseudo-Label Scoring for Zero-Shot Video Summarization 

**Authors**: Yuanli Wu, Long Zhang, Yue Du, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.17501)  

**Abstract**: With the rapid proliferation of video content across social media, surveillance, and education platforms, efficiently summarizing long videos into concise yet semantically faithful surrogates has become increasingly vital. Existing supervised methods achieve strong in-domain accuracy by learning from dense annotations but suffer from high labeling costs and limited cross-dataset generalization, while unsupervised approaches, though label-free, often fail to capture high-level human semantics and fine-grained narrative cues. More recently, zero-shot prompting pipelines have leveraged large language models (LLMs) for training-free video summarization, yet remain highly sensitive to handcrafted prompt templates and dataset-specific score normalization. To overcome these limitations, we introduce a rubric-guided, pseudo-labeled prompting framework that transforms a small subset of ground-truth annotations into high-confidence pseudo labels, which are aggregated into structured, dataset-adaptive scoring rubrics guiding interpretable scene evaluation. During inference, first and last segments are scored based solely on their descriptions, whereas intermediate ones incorporate brief contextual summaries of adjacent scenes to assess narrative progression and redundancy. This contextual prompting enables the LLM to balance local salience and global coherence without parameter tuning. On SumMe and TVSum, our method achieves F1 scores of \textbf{57.58} and \textbf{63.05}, surpassing unsupervised and prior zero-shot baselines while approaching supervised performance. The results demonstrate that rubric-guided pseudo labeling effectively stabilizes LLM-based scoring and establishes a general, interpretable zero-shot paradigm for video summarization. 

---
# Intent-Driven LLM Ensemble Planning for Flexible Multi-Robot Disassembly: Demonstration on EV Batteries 

**Authors**: Cansu Erdogan, Cesar Alan Contreras, Alireza Rastegarpanah, Manolis Chiou, Rustam Stolkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17576)  

**Abstract**: This paper addresses the problem of planning complex manipulation tasks, in which multiple robots with different end-effectors and capabilities, informed by computer vision, must plan and execute concatenated sequences of actions on a variety of objects that can appear in arbitrary positions and configurations in unstructured scenes. We propose an intent-driven planning pipeline which can robustly construct such action sequences with varying degrees of supervisory input from a human using simple language instructions. The pipeline integrates: (i) perception-to-text scene encoding, (ii) an ensemble of large language models (LLMs) that generate candidate removal sequences based on the operator's intent, (iii) an LLM-based verifier that enforces formatting and precedence constraints, and (iv) a deterministic consistency filter that rejects hallucinated objects. The pipeline is evaluated on an example task in which two robot arms work collaboratively to dismantle an Electric Vehicle battery for recycling applications. A variety of components must be grasped and removed in specific sequences, determined by human instructions and/or by task-order feasibility decisions made by the autonomous system. On 200 real scenes with 600 operator prompts across five component classes, we used metrics of full-sequence correctness and next-task correctness to evaluate and compare five LLM-based planners (including ablation analyses of pipeline components). We also evaluated the LLM-based human interface in terms of time to execution and NASA TLX with human participant experiments. Results indicate that our ensemble-with-verification approach reliably maps operator intent to safe, executable multi-robot plans while maintaining low user effort. 

---
# TabR1: Taming GRPO for tabular reasoning LLMs 

**Authors**: Pengxiang Cai, Zihao Gao, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.17385)  

**Abstract**: Tabular prediction has traditionally relied on gradient-boosted decision trees and specialized deep learning models, which excel within tasks but provide limited interpretability and weak transfer across tables. Reasoning large language models (LLMs) promise cross-task adaptability with trans- parent reasoning traces, yet their potential has not been fully realized for tabular data. This paper presents TabR1, the first reasoning LLM for tabular prediction with multi-step reasoning. At its core is Permutation Relative Policy Optimization (PRPO), a simple yet efficient reinforcement learning method that encodes column-permutation invariance as a structural prior. By construct- ing multiple label-preserving permutations per sample and estimating advantages both within and across permutations, PRPO transforms sparse rewards into dense learning signals and improves generalization. With limited supervision, PRPO activates the reasoning ability of LLMs for tabular prediction, enhancing few-shot and zero-shot performance as well as interpretability. Comprehensive experiments demonstrate that TabR1 achieves performance comparable to strong baselines under full-supervision fine-tuning. In the zero-shot setting, TabR1 approaches the performance of strong baselines under the 32-shot setting. Moreover, TabR1 (8B) substantially outperforms much larger LLMs across various tasks, achieving up to 53.17% improvement over DeepSeek-R1 (685B). 

---
# I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models 

**Authors**: Giacomo Camposampiero, Michael Hersche, Roger Wattenhofer, Abu Sebastian, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2510.17496)  

**Abstract**: We introduce I-RAVEN-X, a symbolic benchmark designed to evaluate generalization and robustness in analogical and mathematical reasoning for Large Language Models (LLMs) and Large Reasoning Models (LRMs). I-RAVEN-X extends I-RAVEN by increasing operand complexity, attribute range, and introducing perceptual uncertainty. Compared to LLMs, empirical results show that LRMs achieve improved productivity and systematicity on longer reasoning relations and wider attribute ranges, respectively. However, LRMs are still significantly challenged by reasoning under uncertainty and cannot effectively explore multiple probabilistic outcomes. 

---
# Localist LLMs with Recruitment Learning 

**Authors**: Joachim Diederich  

**Link**: [PDF](https://arxiv.org/pdf/2510.17358)  

**Abstract**: We present a novel framework for training large language models with continuously adjustable internal representations that span the full spectrum from localist (interpretable, rule-based) to distributed (generalizable, efficient) encodings. The key innovations are (1) a locality dial, a tunable parameter that dynamically controls the degree of localization during both training and inference without requiring model retraining, (2) an information-theoretic recruitment mechanism that adaptively allocates semantic blocks as needed, eliminating the requirement for complete domain knowledge at initialization, and (3) a hierarchical recruitment framework that extends capacity allocation to entire specialized LLMs, enabling multi-granularity architectural adaptation. This is achieved through group sparsity penalties on attention mechanisms, information-theoretic anchor design, dynamic rule injection, and principled recruitment  criteria based on penalized likelihood with explicit units. We provide rigorous mathematical results establishing explicit threshold conditions under which attention provably concentrates on semantically relevant blocks at stationary points, with exact bounds on attention entropy and pointer fidelity. The hierarchical recruitment mechanism provides convergence guarantees at both the block level (fine-grained, within-LLM) and the LLM level (coarse-grained, cross-domain), ensuring the system discovers semantic partitions that balance model complexity against data encoding efficiency. This framework enables practitioners to continuously interpolate between interpretable and high-performance modes while adapting architectural capacity at multiple granularities, supporting applications in regulated domains requiring both transparency and capability. 

---
# TREAT: A Code LLMs Trustworthiness / Reliability Evaluation and Testing Framework 

**Authors**: Shuzheng Gao, Eric John Li, Man Ho Lam, Jingyu Xiao, Yuxuan Wan, Chaozheng Wang, Ng Man Tik, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17163)  

**Abstract**: Large foundation models are fundamentally transforming the software engineering landscape, demonstrating exceptional capabilities across diverse tasks such as code generation, debugging, and testing. Despite this rapid progress, a significant gap remains in how to comprehensively evaluate these models' trustworthiness in real-world software engineering scenarios. Existing benchmarks suffer from limited task scope and fail to incorporate critical evaluation aspects such as the robustness and reliability of models. To bridge this gap, we present an evaluation framework called TREAT (Code LLMs Trustworthiness / Reliability Evaluation And Testing) that provides a holistic assessment of model performance in code intelligence tasks. Our evaluation framework addresses key limitations in existing approaches with four main improvements: (1) Multi-Task Holistic Evaluation that spans diverse software engineering activities rather than limited coding tasks; (2) Multi-Language and Multi-Modality Assessment that extends beyond traditional single-language, text-only benchmarks to include multi-modality coding tasks; (3) Robustness Assessment that evaluates model reliability under semantically-preserving code transformations; and (4) Rigorous Evaluation Methodology that enhances the trustworthiness of evaluation results through diverse evaluation prompts and adaptive solution extraction. Based on this evaluation framework, we assess 26 state-of-the-art models and uncover both their strengths and limitations, yielding several key insights:(1) Current models show substantial performance variation across programming tasks; (2) Multi-modal language models demonstrate specific performance limitations in UI code generation and edit; 

---
# GACO-CAD: Geometry-Augmented and Conciseness-Optimized CAD Model Generation from Single Image 

**Authors**: Yinghui Wang, Xinyu Zhang, Peng Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.17157)  

**Abstract**: Generating editable, parametric CAD models from a single image holds great potential to lower the barriers of industrial concept design. However, current multi-modal large language models (MLLMs) still struggle with accurately inferring 3D geometry from 2D images due to limited spatial reasoning capabilities. We address this limitation by introducing GACO-CAD, a novel two-stage post-training framework. It is designed to achieve a joint objective: simultaneously improving the geometric accuracy of the generated CAD models and encouraging the use of more concise modeling procedures. First, during supervised fine-tuning, we leverage depth and surface normal maps as dense geometric priors, combining them with the RGB image to form a multi-channel input. In the context of single-view reconstruction, these priors provide complementary spatial cues that help the MLLM more reliably recover 3D geometry from 2D observations. Second, during reinforcement learning, we introduce a group length reward that, while preserving high geometric fidelity, promotes the generation of more compact and less redundant parametric modeling sequences. A simple dynamic weighting strategy is adopted to stabilize training. Experiments on the DeepCAD and Fusion360 datasets show that GACO-CAD achieves state-of-the-art performance under the same MLLM backbone, consistently outperforming existing methods in terms of code validity, geometric accuracy, and modeling conciseness. 

---
# Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in Large Language Models 

**Authors**: Elias Hossain, Swayamjit Saha, Somshubhra Roy, Ravi Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2510.17098)  

**Abstract**: Even when prompts and parameters are secured, transformer language models remain vulnerable because their key-value (KV) cache during inference constitutes an overlooked attack surface. This paper introduces Malicious Token Injection (MTI), a modular framework that systematically perturbs cached key vectors at selected layers and timesteps through controlled magnitude and frequency, using additive Gaussian noise, zeroing, and orthogonal rotations. A theoretical analysis quantifies how these perturbations propagate through attention, linking logit deviations to the Frobenius norm of corruption and softmax Lipschitz dynamics. Empirical results show that MTI significantly alters next-token distributions and downstream task performance across GPT-2 and LLaMA-2/7B, as well as destabilizes retrieval-augmented and agentic reasoning pipelines. These findings identify cache integrity as a critical yet underexplored vulnerability in current LLM deployments, positioning cache corruption as a reproducible and theoretically grounded threat model for future robustness and security research. 

---
# The Ends Justify the Thoughts: RL-Induced Motivated Reasoning in LLMs 

**Authors**: Nikolaus Howe, Micah Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2510.17057)  

**Abstract**: The use of reinforcement learning (RL) with chain-of-thought (CoT) reasoning has emerged as a promising approach for developing more capable language models. In turn, this has led to investigation of CoT monitoring as a compelling method for detecting harmful behaviors such as reward hacking, under the assumption that models' reasoning processes reflect their internal decision-making. In practice, LLM training often produces unintended behaviors due to imperfect reward signals, leading models to develop misaligned tendencies. A common corrective approach is to apply post-hoc instructions to avoid problematic behaviors like sycophancy, but what happens to the model's reasoning process when these instructions conflict with learned behaviors? We investigate this question in simple settings and find that models engage in systematic motivated reasoning -- generating plausible-sounding justifications for violating their instructions while downplaying potential harms. Beyond being an interesting property of training, we find that while motivated reasoning can be detected by most frontier reasoning models, smaller LLM judges can fail to identify a portion of it, and in rare cases can themselves be persuaded that the reasoning is correct, despite it contradicting clear instructions. This capability gap raises concerns that as models become more sophisticated, their motivated reasoning may become increasingly difficult for monitors to detect. Our results underscore the need to account for motivated reasoning when relying on chain-of-thought processes for model evaluation and oversight. All code for this paper will be made available. WARNING: some examples in this paper may be upsetting. 

---
# Justitia: Fair and Efficient Scheduling for LLM Applications 

**Authors**: Mingyan Yang, Guanjie Wang, Manqi Luo, Yifei Liu, Chen Chen, Han Zhao, Yu Feng, Quan Chen, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.17015)  

**Abstract**: In the era of Large Language Models (LLMs), it has been popular to launch a series of LLM inferences -- we call an LLM application -- to better solve real-world problems. When serving those applications in shared GPU servers, the schedulers are expected to attain fast application completions with guaranteed worst-case performance. However, mainstream LLM schedulers fail to behave well for LLM applications -- due to head-of-line blocking or over-constrained resource allocation. In this paper, we propose to serve LLM applications in a fair and also efficient manner. To this end, we design Justitia, a novel scheduler with three key techniques. First, given that memory is prevalently a bottleneck for mainstream inference frameworks like vLLM, Justitia models the service cost of LLM applications in a memory-centric manner. Meanwhile, it uses a simple neural network model to conduct light-weight and also accurate demand prediction. Moreover, Justitia adopts a virtual-time based fair queuing algorithm to reduce the overall performance with guaranteed worst-case delay. We have implemented Justitia atop vLLM, and experimental results involving diverse LLM applications show that it can substantially enhance the scheduling efficiency with fairness preserved. 

---
# SNOMED CT-powered Knowledge Graphs for Structured Clinical Data and Diagnostic Reasoning 

**Authors**: Dun Liu, Qin Pang, Guangai Liu, Hongyu Mou, Jipeng Fan, Yiming Miao, Pin-Han Ho, Limei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.16899)  

**Abstract**: The effectiveness of artificial intelligence (AI) in healthcare is significantly hindered by unstructured clinical documentation, which results in noisy, inconsistent, and logically fragmented training data. To address this challenge, we present a knowledge-driven framework that integrates the standardized clinical terminology SNOMED CT with the Neo4j graph database to construct a structured medical knowledge graph. In this graph, clinical entities such as diseases, symptoms, and medications are represented as nodes, and semantic relationships such as ``caused by,'' ``treats,'' and ``belongs to'' are modeled as edges in Neo4j, with types mapped from formal SNOMED CT relationship concepts (e.g., \texttt{Causative agent}, \texttt{Indicated for}). This design enables multi-hop reasoning and ensures terminological consistency. By extracting and standardizing entity-relationship pairs from clinical texts, we generate structured, JSON-formatted datasets that embed explicit diagnostic pathways. These datasets are used to fine-tune large language models (LLMs), significantly improving the clinical logic consistency of their outputs. Experimental results demonstrate that our knowledge-guided approach enhances the validity and interpretability of AI-generated diagnostic reasoning, providing a scalable solution for building reliable AI-assisted clinical systems. 

---
# Tutoring LLM into a Better CUDA Optimizer 

**Authors**: Matyáš Brabec, Jiří Klepl, Michal Töpfer, Martin Kruliš  

**Link**: [PDF](https://arxiv.org/pdf/2510.16933)  

**Abstract**: Recent leaps in large language models (LLMs) caused a revolution in programming tools (like GitHub Copilot) that can help with code generation, debugging, and even performance optimization. In this paper, we focus on the capabilities of the most recent reasoning models to generate optimized CUDA code for predefined, well-known tasks. Our objective is to determine which types of code optimizations and parallel patterns the LLMs can perform by themselves and whether they can be improved by tutoring (providing more detailed hints and guidelines in the prompt). The generated solutions were evaluated both automatically (for correctness and speedup) and manually (code reviews) to provide a more detailed perspective. We also tried an interactive approach where the LLM can fix its previous mistakes within a session. The results indicate that LLMs are quite skilled coders; however, they require tutoring to reach optimized solutions provided by parallel computing experts. 

---
# When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation 

**Authors**: Amirkia Rafiei Oskooei, Kaan Baturalp Cosdan, Husamettin Isiktas, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2510.16809)  

**Abstract**: Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering. 

---
# Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes 

**Authors**: Xiongkun Linghu, Jiangyong Huang, Ziyu Zhu, Baoxiong Jia, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16714)  

**Abstract**: Existing research on 3D Large Language Models (LLMs) still struggles to achieve grounded question-answering, primarily due to the under-exploration of the mech- anism of human-like scene-object grounded reasoning. This paper bridges the gap by presenting a novel framework. We first introduce a grounded Chain-of- Thought reasoning method in 3D scenes (SCENECOT), decoupling a complex reasoning task into simpler and manageable problems, and building corresponding visual clues based on multimodal expert modules. To enable such a method, we develop SCENECOT-185K, the first large-scale grounded CoT reasoning dataset, consisting of 185K high-quality instances. Extensive experiments across various complex 3D scene reasoning benchmarks demonstrate that our new framework achieves strong performance with high grounding-QA coherence. To the best of our knowledge, this is the first successful application of CoT reasoning to 3D scene understanding, enabling step-by-step human-like reasoning and showing potential for extension to broader 3D scene understanding scenarios. 

---
# Structured Interfaces for Automated Reasoning with 3D Scene Graphs 

**Authors**: Aaron Ray, Jacob Arkin, Harel Biggie, Chuchu Fan, Luca Carlone, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2510.16643)  

**Abstract**: In order to provide a robot with the ability to understand and react to a user's natural language inputs, the natural language must be connected to the robot's underlying representations of the world. Recently, large language models (LLMs) and 3D scene graphs (3DSGs) have become a popular choice for grounding natural language and representing the world. In this work, we address the challenge of using LLMs with 3DSGs to ground natural language. Existing methods encode the scene graph as serialized text within the LLM's context window, but this encoding does not scale to large or rich 3DSGs. Instead, we propose to use a form of Retrieval Augmented Generation to select a subset of the 3DSG relevant to the task. We encode a 3DSG in a graph database and provide a query language interface (Cypher) as a tool to the LLM with which it can retrieve relevant data for language grounding. We evaluate our approach on instruction following and scene question-answering tasks and compare against baseline context window and code generation methods. Our results show that using Cypher as an interface to 3D scene graphs scales significantly better to large, rich graphs on both local and cloud-based models. This leads to large performance improvements in grounded language tasks while also substantially reducing the token count of the scene graph content. A video supplement is available at this https URL. 

---
# Atom-anchored LLMs speak Chemistry: A Retrosynthesis Demonstration 

**Authors**: Alan Kai Hassen, Andrius Bernatavicius, Antonius P. A. Janssen, Mike Preuss, Gerard J. P. van Westen, Djork-Arné Clevert  

**Link**: [PDF](https://arxiv.org/pdf/2510.16590)  

**Abstract**: Applications of machine learning in chemistry are often limited by the scarcity and expense of labeled data, restricting traditional supervised methods. In this work, we introduce a framework for molecular reasoning using general-purpose Large Language Models (LLMs) that operates without requiring labeled training data. Our method anchors chain-of-thought reasoning to the molecular structure by using unique atomic identifiers. First, the LLM performs a one-shot task to identify relevant fragments and their associated chemical labels or transformation classes. In an optional second step, this position-aware information is used in a few-shot task with provided class examples to predict the chemical transformation. We apply our framework to single-step retrosynthesis, a task where LLMs have previously underperformed. Across academic benchmarks and expert-validated drug discovery molecules, our work enables LLMs to achieve high success rates in identifying chemically plausible reaction sites ($\geq90\%$), named reaction classes ($\geq40\%$), and final reactants ($\geq74\%$). Beyond solving complex chemical tasks, our work also provides a method to generate theoretically grounded synthetic datasets by mapping chemical knowledge onto the molecular structure and thereby addressing data scarcity. 

---
# SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense 

**Authors**: Yiyang Huang, Liang Shi, Yitian Zhang, Yi Xu, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16596)  

**Abstract**: Large Vision-Language Models (LVLMs) excel in diverse cross-modal tasks. However, object hallucination, where models produce plausible but inaccurate object descriptions, remains a significant challenge. In contrast to previous work focusing on LLM components, this paper is the first to trace LVLM hallucinations to visual encoders and identifies three key issues: statistical bias, inherent bias, and vulnerability. To address these challenges, we propose SHIELD, a training-free framework that mitigates hallucinations through three strategies: re-weighting visual tokens to reduce statistical bias, introducing noise-derived tokens to counter inherent bias, and applying adversarial attacks with contrastive decoding to address vulnerability. Experiments demonstrate that SHIELD effectively mitigates object hallucinations across diverse benchmarks and LVLM families. Moreover, SHIELD achieves strong performance on the general LVLM benchmark, highlighting its broad applicability. Code will be released. 

---
# Predicting life satisfaction using machine learning and explainable AI 

**Authors**: Alif Elham Khan, Mohammad Junayed Hasan, Humayra Anjum, Nabeel Mohammed, Sifat Momen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16547)  

**Abstract**: Life satisfaction is a crucial facet of human well-being. Hence, research on life satisfaction is incumbent for understanding how individuals experience their lives and influencing interventions targeted at enhancing mental health and well-being. Life satisfaction has traditionally been measured using analog, complicated, and frequently error-prone methods. These methods raise questions concerning validation and propagation. However, this study demonstrates the potential for machine learning algorithms to predict life satisfaction with a high accuracy of 93.80% and a 73.00% macro F1-score. The dataset comes from a government survey of 19000 people aged 16-64 years in Denmark. Using feature learning techniques, 27 significant questions for assessing contentment were extracted, making the study highly reproducible, simple, and easily interpretable. Furthermore, clinical and biomedical large language models (LLMs) were explored for predicting life satisfaction by converting tabular data into natural language sentences through mapping and adding meaningful counterparts, achieving an accuracy of 93.74% and macro F1-score of 73.21%. It was found that life satisfaction prediction is more closely related to the biomedical domain than the clinical domain. Ablation studies were also conducted to understand the impact of data resampling and feature selection techniques on model performance. Moreover, the correlation between primary determinants with different age brackets was analyzed, and it was found that health condition is the most important determinant across all ages. This study demonstrates how machine learning, large language models and XAI can jointly contribute to building trust and understanding in using AI to investigate human behavior, with significant ramifications for academics and professionals working to quantify and comprehend subjective well-being. 

---
# Few-Label Multimodal Modeling of SNP Variants and ECG Phenotypes Using Large Language Models for Cardiovascular Risk Stratification 

**Authors**: Niranjana Arun Menon, Yulong Li, Iqra Farooq, Sara Ahmed, Muhammad Awais, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2510.16536)  

**Abstract**: Cardiovascular disease (CVD) risk stratification remains a major challenge due to its multifactorial nature and limited availability of high-quality labeled datasets. While genomic and electrophysiological data such as SNP variants and ECG phenotypes are increasingly accessible, effectively integrating these modalities in low-label settings is non-trivial. This challenge arises from the scarcity of well-annotated multimodal datasets and the high dimensionality of biological signals, which limit the effectiveness of conventional supervised models. To address this, we present a few-label multimodal framework that leverages large language models (LLMs) to combine genetic and electrophysiological information for cardiovascular risk stratification. Our approach incorporates a pseudo-label refinement strategy to adaptively distill high-confidence labels from weakly supervised predictions, enabling robust model fine-tuning with only a small set of ground-truth annotations. To enhance the interpretability, we frame the task as a Chain of Thought (CoT) reasoning problem, prompting the model to produce clinically relevant rationales alongside predictions. Experimental results demonstrate that the integration of multimodal inputs, few-label supervision, and CoT reasoning improves robustness and generalizability across diverse patient profiles. Experimental results using multimodal SNP variants and ECG-derived features demonstrated comparable performance to models trained on the full dataset, underscoring the promise of LLM-based few-label multimodal modeling for advancing personalized cardiovascular care. 

---
# Declarative Techniques for NL Queries over Heterogeneous Data 

**Authors**: Elham Khabiri, Jeffrey O. Kephart, Fenno F. Heath III, Srideepika Jayaraman, Fateh A. Tipu, Yingjie Li, Dhruv Shah, Achille Fokoue, Anu Bhamidipaty  

**Link**: [PDF](https://arxiv.org/pdf/2510.16470)  

**Abstract**: In many industrial settings, users wish to ask questions in natural language, the answers to which require assembling information from diverse structured data sources. With the advent of Large Language Models (LLMs), applications can now translate natural language questions into a set of API calls or database calls, execute them, and combine the results into an appropriate natural language response. However, these applications remain impractical in realistic industrial settings because they do not cope with the data source heterogeneity that typifies such environments. In this work, we simulate the heterogeneity of real industry settings by introducing two extensions of the popular Spider benchmark dataset that require a combination of database and API calls. Then, we introduce a declarative approach to handling such data heterogeneity and demonstrate that it copes with data source heterogeneity significantly better than state-of-the-art LLM-based agentic or imperative code generation systems. Our augmented benchmarks are available to the research community. 

---
# EDVD-LLaMA: Explainable Deepfake Video Detection via Multimodal Large Language Model Reasoning 

**Authors**: Haoran Sun, Chen Cai, Huiping Zhuang, Kong Aik Lee, Lap-Pui Chau, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16442)  

**Abstract**: The rapid development of deepfake video technology has not only facilitated artistic creation but also made it easier to spread misinformation. Traditional deepfake video detection (DVD) methods face issues such as a lack of transparency in their principles and insufficient generalization capabilities to cope with evolving forgery techniques. This highlights an urgent need for detectors that can identify forged content and provide verifiable reasoning explanations. This paper proposes the explainable deepfake video detection (EDVD) task and designs the EDVD-LLaMA multimodal, a large language model (MLLM) reasoning framework, which provides traceable reasoning processes alongside accurate detection results and trustworthy explanations. Our approach first incorporates a Spatio-Temporal Subtle Information Tokenization (ST-SIT) to extract and fuse global and local cross-frame deepfake features, providing rich spatio-temporal semantic information input for MLLM reasoning. Second, we construct a Fine-grained Multimodal Chain-of-Thought (Fg-MCoT) mechanism, which introduces facial feature data as hard constraints during the reasoning process to achieve pixel-level spatio-temporal video localization, suppress hallucinated outputs, and enhance the reliability of the chain of thought. In addition, we build an Explainable Reasoning FF++ benchmark dataset (ER-FF++set), leveraging structured data to annotate videos and ensure quality control, thereby supporting dual supervision for reasoning and detection. Extensive experiments demonstrate that EDVD-LLaMA achieves outstanding performance and robustness in terms of detection accuracy, explainability, and its ability to handle cross-forgery methods and cross-dataset scenarios. Compared to previous DVD methods, it provides a more explainable and superior solution. The source code and dataset will be publicly available. 

---
# Synergizing chemical and AI communities for advancing laboratories of the future 

**Authors**: Saejin Oh, Xinyi Fang, I-Hsin Lin, Paris Dee, Christopher S. Dunham, Stacy M. Copp, Abigail G. Doyle, Javier Read de Alaniz, Mengyang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16293)  

**Abstract**: The development of automated experimental facilities and the digitization of experimental data have introduced numerous opportunities to radically advance chemical laboratories. As many laboratory tasks involve predicting and understanding previously unknown chemical relationships, machine learning (ML) approaches trained on experimental data can substantially accelerate the conventional design-build-test-learn process. This outlook article aims to help chemists understand and begin to adopt ML predictive models for a variety of laboratory tasks, including experimental design, synthesis optimization, and materials characterization. Furthermore, this article introduces how artificial intelligence (AI) agents based on large language models can help researchers acquire background knowledge in chemical or data science and accelerate various aspects of the discovery process. We present three case studies in distinct areas to illustrate how ML models and AI agents can be leveraged to reduce time-consuming experiments and manual data analysis. Finally, we highlight existing challenges that require continued synergistic effort from both experimental and computational communities to address. 

---
# STABLE: Gated Continual Learning for Large Language Models 

**Authors**: William Hoy, Nurcin Celik  

**Link**: [PDF](https://arxiv.org/pdf/2510.16089)  

**Abstract**: Large language models (LLMs) increasingly require mechanisms for continual adaptation without full retraining. However, sequential updates can lead to catastrophic forgetting, where new edits degrade previously acquired knowledge. This work presents STABLE, a gated continual self editing framework that constrains forgetting during sequential updates using parameter efficient fine tuning via Low Rank Adaptation (LoRA; see arXiv:2106.09685). Each candidate edit is evaluated against a stability budget using one of three metrics: (i) Exact Match (EM) drop, capturing factual accuracy loss; (ii) bits increase, reflecting reduced model confidence; and (iii) KL divergence, quantifying distributional drift between the base and adapted models. If a threshold is exceeded, the LoRA update is rescaled through a clipping procedure or rejected. Experiments on the Qwen-2.5-7B model show that gating effectively mitigates forgetting while preserving adaptability. EM based gating achieved the highest cumulative performance in short continual learning sequences. Our results show that different gating strategies can achieve comparable distribution shift (measured by KL divergence) while producing different accuracy outcomes, highlighting the importance of gating design in continual adaptation. This approach offers a principled method for continual model editing, enabling LLMs to integrate new knowledge while maintaining reliability. Code: this https URL 

---
# MoPHES:Leveraging on-device LLMs as Agent for Mobile Psychological Health Evaluation and Support 

**Authors**: Xun Wei, Pukai Zhou, Zeyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16085)  

**Abstract**: The 2022 World Mental Health Report calls for global mental health care reform, amid rising prevalence of issues like anxiety and depression that affect nearly one billion people worldwide. Traditional in-person therapy fails to meet this demand, and the situation is worsened by stigma. While general-purpose large language models (LLMs) offer efficiency for AI-driven mental health solutions, they underperform because they lack specialized fine-tuning. Existing LLM-based mental health chatbots can engage in empathetic conversations, but they overlook real-time user mental state assessment which is critical for professional counseling. This paper proposes MoPHES, a framework that integrates mental state evaluation, conversational support, and professional treatment recommendations. The agent developed under this framework uses two fine-tuned MiniCPM4-0.5B LLMs: one is fine-tuned on mental health conditions datasets to assess users' mental states and predict the severity of anxiety and depression; the other is fine-tuned on multi-turn dialogues to handle conversations with users. By leveraging insights into users' mental states, our agent provides more tailored support and professional treatment recommendations. Both models are also deployed directly on mobile devices to enhance user convenience and protect user privacy. Additionally, to evaluate the performance of MoPHES with other LLMs, we develop a benchmark for the automatic evaluation of mental state prediction and multi-turn counseling dialogues, which includes comprehensive evaluation metrics, datasets, and methods. 

---
# AsyncVoice Agent: Real-Time Explanation for LLM Planning and Reasoning 

**Authors**: Yueqian Lin, Zhengmian Hu, Jayakumar Subramanian, Qinsi Wang, Nikos Vlassis, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16156)  

**Abstract**: Effective human-AI collaboration on complex reasoning tasks requires that users understand and interact with the model's process, not just receive an output. However, the monolithic text from methods like Chain-of-Thought (CoT) prevents this, as current interfaces lack real-time verbalization and robust user barge-in. We present AsyncVoice Agent, a system whose asynchronous architecture decouples a streaming LLM backend from a conversational voice frontend. This design allows narration and inference to run in parallel, empowering users to interrupt, query, and steer the model's reasoning process at any time. Objective benchmarks show this approach reduces interaction latency by more than 600x compared to monolithic baselines while ensuring high fidelity and competitive task accuracy. By enabling a two-way dialogue with a model's thought process, AsyncVoice Agent offers a new paradigm for building more effective, steerable, and trustworthy human-AI systems for high-stakes tasks. 

---
# SARHAchat: An LLM-Based Chatbot for Sexual and Reproductive Health Counseling 

**Authors**: Jiaye Yang, Xinyu Zhao, Tianlong Chen, Kandyce Brennan  

**Link**: [PDF](https://arxiv.org/pdf/2510.16081)  

**Abstract**: While Artificial Intelligence (AI) shows promise in healthcare applications, existing conversational systems often falter in complex and sensitive medical domains such as Sexual and Reproductive Health (SRH). These systems frequently struggle with hallucination and lack the specialized knowledge required, particularly for sensitive SRH topics. Furthermore, current AI approaches in healthcare tend to prioritize diagnostic capabilities over comprehensive patient care and education. Addressing these gaps, this work at the UNC School of Nursing introduces SARHAchat, a proof-of-concept Large Language Model (LLM)-based chatbot. SARHAchat is designed as a reliable, user-centered system integrating medical expertise with empathetic communication to enhance SRH care delivery. Our evaluation demonstrates SARHAchat's ability to provide accurate and contextually appropriate contraceptive counseling while maintaining a natural conversational flow. The demo is available at this https URL}{this https URL. 

---
# Interpretable RNA-Seq Clustering with an LLM-Based Agentic Evidence-Grounded Framework 

**Authors**: Elias Hossain, Mehrdad Shoeibi, Ivan Garibay, Niloofar Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2510.16082)  

**Abstract**: We propose CITE V.1, an agentic, evidence-grounded framework that leverages Large Language Models (LLMs) to provide transparent and reproducible interpretations of RNA-seq clusters. Unlike existing enrichment-based approaches that reduce results to broad statistical associations and LLM-only models that risk unsupported claims or fabricated citations, CITE V.1 transforms cluster interpretation by producing biologically coherent explanations explicitly anchored in the biomedical literature. The framework orchestrates three specialized agents: a Retriever that gathers domain knowledge from PubMed and UniProt, an Interpreter that formulates functional hypotheses, and Critics that evaluate claims, enforce evidence grounding, and qualify uncertainty through confidence and reliability indicators. Applied to Salmonella enterica RNA-seq data, CITE V.1 generated biologically meaningful insights supported by the literature, while an LLM-only Gemini baseline frequently produced speculative results with false citations. By moving RNA-seq analysis from surface-level enrichment to auditable, interpretable, and evidence-based hypothesis generation, CITE V.1 advances the transparency and reliability of AI in biomedicine. 

---
# TriAgent: Automated Biomarker Discovery with Deep Research Grounding for Triage in Acute Care by LLM-Based Multi-Agent Collaboration 

**Authors**: Kerem Delikoyun, Qianyu Chen, Win Sen Kuan, John Tshon Yit Soong, Matthew Edward Cove, Oliver Hayden  

**Link**: [PDF](https://arxiv.org/pdf/2510.16080)  

**Abstract**: Emergency departments worldwide face rising patient volumes, workforce shortages, and variability in triage decisions that threaten the delivery of timely and accurate care. Current triage methods rely primarily on vital signs, routine laboratory values, and clinicians' judgment, which, while effective, often miss emerging biological signals that could improve risk prediction for infection typing or antibiotic administration in acute conditions. To address this challenge, we introduce TriAgent, a large language model (LLM)-based multi-agent framework that couples automated biomarker discovery with deep research for literature-grounded validation and novelty assessment. TriAgent employs a supervisor research agent to generate research topics and delegate targeted queries to specialized sub-agents for evidence retrieval from various data sources. Findings are synthesized to classify biomarkers as either grounded in existing knowledge or flagged as novel candidates, offering transparent justification and highlighting unexplored pathways in acute care risk stratification. Unlike prior frameworks limited to existing routine clinical biomarkers, TriAgent aims to deliver an end-to-end framework from data analysis to literature grounding to improve transparency, explainability and expand the frontier of potentially actionable clinical biomarkers. Given a user's clinical query and quantitative triage data, TriAgent achieved a topic adherence F1 score of 55.7 +/- 5.0%, surpassing the CoT-ReAct agent by over 10%, and a faithfulness score of 0.42 +/- 0.39, exceeding all baselines by more than 50%. Across experiments, TriAgent consistently outperformed state-of-the-art LLM-based agentic frameworks in biomarker justification and literature-grounded novelty assessment. We share our repo: this https URL. 

---
# Stratos: An End-to-End Distillation Pipeline for Customized LLMs under Distributed Cloud Environments 

**Authors**: Ziming Dai, Tuo Zhang, Fei Gao, Xingyi Cai, Xiaofei Wang, Cheng Zhang, Wenyu Wang, Chengjie Zang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15992)  

**Abstract**: The growing industrial demand for customized and cost-efficient large language models (LLMs) is fueled by the rise of vertical, domain-specific tasks and the need to optimize performance under constraints such as latency and budget. Knowledge distillation, as an efficient model compression and transfer technique, offers a feasible solution. However, existing distillation frameworks often require manual intervention and struggle to meet such complex user-defined distillation requirements. To bridge this gap, we propose Stratos, an end-to-end LLM distillation pipeline that automates server and model selection, knowledge distillation, and deployment in distributed cloud environments. Given user-defined constraints on model performance and system budget, Stratos automatically selects Pareto-optimal servers, dynamically matches teacher-student pairs, and adapts distillation strategies based on task complexity to optimize cloud hosting. Experiments show that Stratos produces a student model that achieves four times the accuracy of its GPT-4o teacher baseline on a rare, domain-specific Mahjong reasoning task with reverse synthetic data and knowledge injection. Moreover, it achieves reduced latency and cost without compromising accuracy. These results highlight its promise for vertical-domain LLM deployment. 

---
# MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents 

**Authors**: Dongsen Zhang, Zekun Li, Xu Luo, Xuannan Liu, Peipei Li, Wenjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.15994)  

**Abstract**: The Model Context Protocol (MCP) standardizes how large language model (LLM) agents discover, describe, and call external tools. While MCP unlocks broad interoperability, it also enlarges the attack surface by making tools first-class, composable objects with natural-language metadata, and standardized I/O. We present MSB (MCP Security Benchmark), the first end-to-end evaluation suite that systematically measures how well LLM agents resist MCP-specific attacks throughout the full tool-use pipeline: task planning, tool invocation, and response handling. MSB contributes: (1) a taxonomy of 12 attacks including name-collision, preference manipulation, prompt injections embedded in tool descriptions, out-of-scope parameter requests, user-impersonating responses, false-error escalation, tool-transfer, retrieval injection, and mixed attacks; (2) an evaluation harness that executes attacks by running real tools (both benign and malicious) via MCP rather than simulation; and (3) a robustness metric that quantifies the trade-off between security and performance: Net Resilient Performance (NRP). We evaluate nine popular LLM agents across 10 domains and 400+ tools, producing 2,000 attack instances. Results reveal the effectiveness of attacks against each stage of MCP. Models with stronger performance are more vulnerable to attacks due to their outstanding tool calling and instruction following capabilities. MSB provides a practical baseline for researchers and practitioners to study, compare, and harden MCP agents. 

---
# Algorithmic Primitives and Compositional Geometry of Reasoning in Language Models 

**Authors**: Samuel Lippl, Thomas McGee, Kimberly Lopez, Ziwen Pan, Pierce Zhang, Salma Ziadi, Oliver Eberle, Ida Momennejad  

**Link**: [PDF](https://arxiv.org/pdf/2510.15987)  

**Abstract**: How do latent and inference time computations enable large language models (LLMs) to solve multi-step reasoning? We introduce a framework for tracing and steering algorithmic primitives that underlie model reasoning. Our approach links reasoning traces to internal activation patterns and evaluates algorithmic primitives by injecting them into residual streams and measuring their effect on reasoning steps and task performance. We consider four benchmarks: Traveling Salesperson Problem (TSP), 3SAT, AIME, and graph navigation. We operationalize primitives by clustering neural activations and labeling their matched reasoning traces. We then apply function vector methods to derive primitive vectors as reusable compositional building blocks of reasoning. Primitive vectors can be combined through addition, subtraction, and scalar operations, revealing a geometric logic in activation space. Cross-task and cross-model evaluations (Phi-4, Phi-4-Reasoning, Llama-3-8B) show both shared and task-specific primitives. Notably, comparing Phi-4 with its reasoning-finetuned variant highlights compositional generalization after finetuning: Phi-4-Reasoning exhibits more systematic use of verification and path-generation primitives. Injecting the associated primitive vectors in Phi-4-Base induces behavioral hallmarks associated with Phi-4-Reasoning. Together, these findings demonstrate that reasoning in LLMs may be supported by a compositional geometry of algorithmic primitives, that primitives transfer cross-task and cross-model, and that reasoning finetuning strengthens algorithmic generalization across domains. 

---
# AMiD: Knowledge Distillation for LLMs with $α$-mixture Assistant Distribution 

**Authors**: Donghyeok Shin, Yeongmin Kim, Suhyeon Jo, Byeonghu Na, Il-Chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2510.15982)  

**Abstract**: Autoregressive large language models (LLMs) have achieved remarkable improvement across many tasks but incur high computational and memory costs. Knowledge distillation (KD) mitigates this issue by transferring knowledge from a large teacher to a smaller student through distributional alignment. Previous studies have proposed various discrepancy metrics, but the capacity gap and training instability caused by near-zero probabilities, stemming from the high-dimensional output of LLMs, remain fundamental limitations. To overcome these challenges, several approaches implicitly or explicitly incorporating assistant distribution have recently been proposed. However, the past proposals of assistant distributions have been a fragmented approach without a systematic investigation of the interpolation path and the divergence. This paper proposes $\alpha$-mixture assistant distribution, a novel generalized family of assistant distributions, and $\alpha$-mixture distillation, coined AMiD, a unified framework for KD using the assistant distribution. The $\alpha$-mixture assistant distribution provides a continuous extension of the assistant distribution by introducing a new distribution design variable $\alpha$, which has been fixed in all previous approaches. Furthermore, AMiD generalizes the family of divergences used with the assistant distributions based on optimality, which has also been restricted in previous works. Through extensive experiments, we demonstrate that AMiD offers superior performance and training stability by leveraging a broader and theoretically grounded assistant distribution space. 

---
# Cog-Rethinker: Hierarchical Metacognitive Reinforcement Learning for LLM Reasoning 

**Authors**: Zexu Sun, Yongcheng Zeng, Erxue Min, Heyang Gao, Bokai Ji, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15979)  

**Abstract**: Contemporary progress in large language models (LLMs) has revealed notable inferential capacities via reinforcement learning (RL) employing verifiable reward, facilitating the development of O1 and R1-like reasoning models. Directly training from base models with RL is called zero-RL. However, previous works rely upon activating LLMs' inherent capacities through fixed prompt templates. This strategy introduces substantial sampling inefficiencies for weak LLMs, as the majority of problems generate invalid outputs during accuracy-driven filtration in reasoning tasks, which causes a waste of samples. To solve this issue, we propose Cog-Rethinker, a novel hierarchical metacognitive RL framework for LLM reasoning. Our Cog-Rethinker mainly focuses on the rollout procedure in RL training. After the direct rollout, our Cog-Rethinker improves sample utilization in a hierarchical metacognitive two-stage framework. By leveraging human cognition during solving problems, firstly, it prompts policy to decompose zero-accuracy problems into subproblems to produce final reasoning results. Secondly, with zero-accuracy problems in previous rollout stage, it further prompts policy to refine these answers by referencing previous wrong solutions. Moreover, to enable cold-start of the two new reasoning patterns and maintain train-test consistency across prompt templates, our Cog-Rethinker applies supervised fine-tuning on the policy using correct samples of the two stages with direct rollout template. Experimental results demonstrate Cog-Rethinker's superior performance on various mathematical reasoning benchmarks, we also analyzed its improved sample efficiency that accelerates convergence compared to baseline methods. 

---
# Safeguarding Efficacy in Large Language Models: Evaluating Resistance to Human-Written and Algorithmic Adversarial Prompts 

**Authors**: Tiarnaigh Downey-Webb, Olamide Jogunola, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2510.15973)  

**Abstract**: This paper presents a systematic security assessment of four prominent Large Language Models (LLMs) against diverse adversarial attack vectors. We evaluate Phi-2, Llama-2-7B-Chat, GPT-3.5-Turbo, and GPT-4 across four distinct attack categories: human-written prompts, AutoDAN, Greedy Coordinate Gradient (GCG), and Tree-of-Attacks-with-pruning (TAP). Our comprehensive evaluation employs 1,200 carefully stratified prompts from the SALAD-Bench dataset, spanning six harm categories. Results demonstrate significant variations in model robustness, with Llama-2 achieving the highest overall security (3.4% average attack success rate) while Phi-2 exhibits the greatest vulnerability (7.0% average attack success rate). We identify critical transferability patterns where GCG and TAP attacks, though ineffective against their target model (Llama-2), achieve substantially higher success rates when transferred to other models (up to 17% for GPT-4). Statistical analysis using Friedman tests reveals significant differences in vulnerability across harm categories ($p < 0.001$), with malicious use prompts showing the highest attack success rates (10.71% average). Our findings contribute to understanding cross-model security vulnerabilities and provide actionable insights for developing targeted defense mechanisms 

---
# LinearizeLLM: An Agent-Based Framework for LLM-Driven Exact Linear Reformulation of Nonlinear Optimization Problems 

**Authors**: Paul-Niklas Ken Kandora, Simon Caspar Zeller, Aaron Jeremias Elsing, Elena Kuss, Steffen Rebennack  

**Link**: [PDF](https://arxiv.org/pdf/2510.15969)  

**Abstract**: Reformulating nonlinear optimization problems is largely manual and expertise-intensive, yet it remains essential for solving such problems with linear optimization solvers or applying special-purpose algorithms. We introduce \textit{LinearizeLLM}, an agent-based framework that solves this task by leveraging Large Language Models (LLMs). The framework assigns each nonlinear pattern to a \textit{reformulation agent} that is explicitly instructed to derive an exact linear reformulation for its nonlinearity pattern, for instance, absolute-value terms or bilinear products of decision variables. The agents then coordinate to assemble a solver-ready linear model equivalent to the original problem. To benchmark the approach, we create a dataset of 20 real-world nonlinear optimization problems derived from the established ComplexOR dataset of linear optimization problems. We evaluate our approach with several LLMs. Our results indicate that specialized LLM agents can automate linearization tasks, opening a path toward fully conversational modeling pipelines for nonlinear optimization. 

---
# One Token Embedding Is Enough to Deadlock Your Large Reasoning Model 

**Authors**: Mohan Zhang, Yihua Zhang, Jinghan Jia, Zhangyang Wang, Sijia Liu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15965)  

**Abstract**: Modern large reasoning models (LRMs) exhibit impressive multi-step problem-solving via chain-of-thought (CoT) reasoning. However, this iterative thinking mechanism introduces a new vulnerability surface. We present the Deadlock Attack, a resource exhaustion method that hijacks an LRM's generative control flow by training a malicious adversarial embedding to induce perpetual reasoning loops. Specifically, the optimized embedding encourages transitional tokens (e.g., "Wait", "But") after reasoning steps, preventing the model from concluding its answer. A key challenge we identify is the continuous-to-discrete projection gap: naïve projections of adversarial embeddings to token sequences nullify the attack. To overcome this, we introduce a backdoor implantation strategy, enabling reliable activation through specific trigger tokens. Our method achieves a 100% attack success rate across four advanced LRMs (Phi-RM, Nemotron-Nano, R1-Qwen, R1-Llama) and three math reasoning benchmarks, forcing models to generate up to their maximum token limits. The attack is also stealthy (in terms of causing negligible utility loss on benign user inputs) and remains robust against existing strategies trying to mitigate the overthinking issue. Our findings expose a critical and underexplored security vulnerability in LRMs from the perspective of reasoning (in)efficiency. 

---
# How Good Are LLMs at Processing Tool Outputs? 

**Authors**: Kiran Kate, Yara Rizk, Poulami Ghosh, Ashu Gulati, Tathagata Chakraborti, Zidane Wright, Mayank Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.15955)  

**Abstract**: Most realistic task automation problems require large language models (LLMs) to call tools, which often return complex JSON responses. These responses must be further processed to derive the information necessary for task completion. The ability of LLMs to do so is under-studied. In this paper, we study the tool response processing task and LLMs' abilities to process structured (JSON) responses. We created a dataset for this task, and evaluated 15 open and closed weight models using multiple prompting approaches. Our results show that JSON processing remains a difficult task even for frontier models across multiple prompting strategies. The optimal response processing strategy depends on both the nature and size of the tool outputs, as well as the complexity of the required reasoning. Variations in processing approaches can lead to performance differences ranging from 3\% to 50\%. 

---
# ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination 

**Authors**: Charidimos Papadakis, Angeliki Dimitriou, Giorgos Filandrianos, Maria Lymperaiou, Konstantinos Thomas, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2510.15949)  

**Abstract**: Large language models show promise for financial decision-making, yet deploying them as autonomous trading agents raises fundamental challenges: how to adapt instructions when rewards arrive late and obscured by market noise, how to synthesize heterogeneous information streams into coherent decisions, and how to bridge the gap between model outputs and executable market actions. We present ATLAS (Adaptive Trading with LLM AgentS), a unified multi-agent framework that integrates structured information from markets, news, and corporate fundamentals to support robust trading decisions. Within ATLAS, the central trading agent operates in an order-aware action space, ensuring that outputs correspond to executable market orders rather than abstract signals. The agent can incorporate feedback while trading using Adaptive-OPRO, a novel prompt-optimization technique that dynamically adapts the prompt by incorporating real-time, stochastic feedback, leading to increasing performance over time. Across regime-specific equity studies and multiple LLM families, Adaptive-OPRO consistently outperforms fixed prompts, while reflection-based feedback fails to provide systematic gains. 

---
# Learning from Mistakes: Enhancing Harmful Meme Detection via Misjudgment Risk Patterns 

**Authors**: Wenshuo Wang, Ziyou Jiang, Junjie Wang, Mingyang Li, Jie Huang, Yuekai Huang, Zhiyuan Chang, Feiyan Duan, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15946)  

**Abstract**: Internet memes have emerged as a popular multimodal medium, yet they are increasingly weaponized to convey harmful opinions through subtle rhetorical devices like irony and metaphor. Existing detection approaches, including MLLM-based techniques, struggle with these implicit expressions, leading to frequent misjudgments. This paper introduces PatMD, a novel approach that improves harmful meme detection by learning from and proactively mitigating these potential misjudgment risks. Our core idea is to move beyond superficial content-level matching and instead identify the underlying misjudgment risk patterns, proactively guiding the MLLMs to avoid known misjudgment pitfalls. We first construct a knowledge base where each meme is deconstructed into a misjudgment risk pattern explaining why it might be misjudged, either overlooking harmful undertones (false negative) or overinterpreting benign content (false positive). For a given target meme, PatMD retrieves relevant patterns and utilizes them to dynamically guide the MLLM's reasoning. Experiments on a benchmark of 6,626 memes across 5 harmful detection tasks show that PatMD outperforms state-of-the-art baselines, achieving an average of 8.30\% improvement in F1-score and 7.71\% improvement in accuracy, demonstrating strong generalizability and improved detection capability of harmful memes. 

---
# BEACON: Bayesian Optimal Stopping for Efficient LLM Sampling 

**Authors**: Guangya Wan, Zixin Stephen Xu, Sasa Zorc, Manel Baucells, Mengxuan Hu, Hao Wang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.15945)  

**Abstract**: Sampling multiple responses is a common way to improve LLM output quality, but it comes at the cost of additional computation. The key challenge is deciding when to stop generating new samples to balance accuracy gains against efficiency. To address this, we introduce BEACON (Bayesian Efficient Adaptive Criterion for Optimal N-stopping), a principled adaptive sampling framework grounded in Sequential Search with Bayesian Learning. BEACON sequentially generates responses from the policy LLM, updates posterior belief over reward distributions in real time without further training, and determines when to stop by weighing expected gains against computational cost. Sampling terminates once the marginal utility of further exploration no longer justifies the expense. We establish both theoretical optimality guarantees and practical tractability, and show empirically that BEACON reduces average sampling by up to 80% while maintaining response quality. We further demonstrate BEACON's utility for cost-efficient preference data generation and outline practical extensions, offering actionable insights for future researchers. 

---
# Intent-Driven Storage Systems: From Low-Level Tuning to High-Level Understanding 

**Authors**: Shai Bergman, Won Wook Song, Lukas Cavigelli, Konstantin Berestizshevsky, Ke Zhou, Ji Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15917)  

**Abstract**: Existing storage systems lack visibility into workload intent, limiting their ability to adapt to the semantics of modern, large-scale data-intensive applications. This disconnect leads to brittle heuristics and fragmented, siloed optimizations. To address these limitations, we propose Intent-Driven Storage Systems (IDSS), a vision for a new paradigm where large language models (LLMs) infer workload and system intent from unstructured signals to guide adaptive and cross-layer parameter reconfiguration. IDSS provides holistic reasoning for competing demands, synthesizing safe and efficient decisions within policy guardrails. We present four design principles for integrating LLMs into storage control loops and propose a corresponding system architecture. Initial results on FileBench workloads show that IDSS can improve IOPS by up to 2.45X by interpreting intent and generating actionable configurations for storage components such as caching and prefetching. These findings suggest that, when constrained by guardrails and embedded within structured workflows, LLMs can function as high-level semantic optimizers, bridging the gap between application goals and low-level system control. IDSS points toward a future in which storage systems are increasingly adaptive, autonomous, and aligned with dynamic workload demands. 

---
# VeriGRAG: Enhancing LLM-Based Verilog Code Generation with Structure-Aware Soft Prompts 

**Authors**: Jiayu Zhao, Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15914)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in generating Verilog code from natural language descriptions. However, Verilog code inherently encodes structural information of hardware circuits. Effectively leveraging this structural information to enhance the functional and syntactic correctness of LLM-generated Verilog code remains a significant challenge. To address this challenge, we propose VeriGRAG , a novel framework that extracts structural graph embeddings from Verilog code using graph neural networks (GNNs). A multimodal retriever then selects the graph embeddings most relevant to the given generation task, which are aligned with the code modality through the VeriFormer module to generate structure-aware soft prompts. Our experiments demonstrate that VeriGRAG substantially improves the correctness of Verilog code generation, achieving state-of-the-art or superior performance across both VerilogEval and RTLLM benchmarks. 

---
# FVDebug: An LLM-Driven Debugging Assistant for Automated Root Cause Analysis of Formal Verification Failures 

**Authors**: Yunsheng Bai, Ghaith Bany Hamad, Chia-Tung Ho, Syed Suhaib, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.15906)  

**Abstract**: Debugging formal verification (FV) failures represents one of the most time-consuming bottlenecks in modern hardware design workflows. When properties fail, engineers must manually trace through complex counter-examples spanning multiple cycles, analyze waveforms, and cross-reference design specifications to identify root causes - a process that can consume hours or days per bug. Existing solutions are largely limited to manual waveform viewers or simple automated tools that cannot reason about the complex interplay between design intent and implementation logic. We present FVDebug, an intelligent system that automates root-cause analysis by combining multiple data sources - waveforms, RTL code, design specifications - to transform failure traces into actionable insights. Our approach features a novel pipeline: (1) Causal Graph Synthesis that structures failure traces into directed acyclic graphs, (2) Graph Scanner using batched Large Language Model (LLM) analysis with for-and-against prompting to identify suspicious nodes, and (3) Insight Rover leveraging agentic narrative exploration to generate high-level causal explanations. FVDebug further provides concrete RTL fixes through its Fix Generator. Evaluated on open benchmarks, FVDebug attains high hypothesis quality and strong Pass@k fix rates. We further report results on two proprietary, production-scale FV counterexamples. These results demonstrate FVDebug's applicability from academic benchmarks to industrial designs. 

---
# Multimodal Chip Physical Design Engineer Assistant 

**Authors**: Yun-Da Tsai, Chang-Yu Chao, Liang-Yeh Shen, Tsung-Han Lin, Haoyu Yang, Mark Ho, Yi-Chen Lu, Wen-Hao Liu, Shou-De Lin, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.15872)  

**Abstract**: Modern chip physical design relies heavily on Electronic Design Automation (EDA) tools, which often struggle to provide interpretable feedback or actionable guidance for improving routing congestion. In this work, we introduce a Multimodal Large Language Model Assistant (MLLMA) that bridges this gap by not only predicting congestion but also delivering human-interpretable design suggestions. Our method combines automated feature generation through MLLM-guided genetic prompting with an interpretable preference learning framework that models congestion-relevant tradeoffs across visual, tabular, and textual inputs. We compile these insights into a "Design Suggestion Deck" that surfaces the most influential layout features and proposes targeted optimizations. Experiments on the CircuitNet benchmark demonstrate that our approach outperforms existing models on both accuracy and explainability. Additionally, our design suggestion guidance case study and qualitative analyses confirm that the learned preferences align with real-world design principles and are actionable for engineers. This work highlights the potential of MLLMs as interactive assistants for interpretable and context-aware physical design optimization. 

---
# How role-play shapes relevance judgment in zero-shot LLM rankers 

**Authors**: Yumeng Wang, Jirui Qi, Catherine Chen, Panagiotis Eustratiadis, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2510.17535)  

**Abstract**: Large Language Models (LLMs) have emerged as promising zero-shot rankers, but their performance is highly sensitive to prompt formulation. In particular, role-play prompts, where the model is assigned a functional role or identity, often give more robust and accurate relevance rankings. However, the mechanisms and diversity of role-play effects remain underexplored, limiting both effective use and interpretability. In this work, we systematically examine how role-play variations influence zero-shot LLM rankers. We employ causal intervention techniques from mechanistic interpretability to trace how role-play information shapes relevance judgments in LLMs. Our analysis reveals that (1) careful formulation of role descriptions have a large effect on the ranking quality of the LLM; (2) role-play signals are predominantly encoded in early layers and communicate with task instructions in middle layers, while receiving limited interaction with query or document representations. Specifically, we identify a group of attention heads that encode information critical for role-conditioned relevance. These findings not only shed light on the inner workings of role-play in LLM ranking but also offer guidance for designing more effective prompts in IR and beyond, pointing toward broader opportunities for leveraging role-play in zero-shot applications. 

---
# DSEBench: A Test Collection for Explainable Dataset Search with Examples 

**Authors**: Qing Shi, Jing He, Qiaosheng Chen, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.17228)  

**Abstract**: Dataset search has been an established information retrieval task. Current paradigms either retrieve datasets that are relevant to a keyword query or find datasets that are similar to an input target dataset. To allow for their combined specification of information needs, in this article, we investigate the more generalized task of Dataset Search with Examples (DSE) and further extend it to Explainable DSE that requires identifying the metadata and content fields of a dataset that indicate its relevance to the query and similarity to the target datasets. To facilitate this research, we construct DSEBench, a test collection that provides high-quality dataset- and field-level annotations to enable the evaluation of explainable DSE. We also employ a large language model to generate numerous annotations to be used for training. We establish extensive baselines on DSEBench by adapting and evaluating a variety of sparse, dense, and LLM-based retrieval, reranking, and explanation methods. 

---
