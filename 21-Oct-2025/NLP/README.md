# Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics 

**Authors**: Akshara Prabhakar, Roshan Ram, Zixiang Chen, Silvio Savarese, Frank Wang, Caiming Xiong, Huan Wang, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17797)  

**Abstract**: As information grows exponentially, enterprises face increasing pressure to transform unstructured data into coherent, actionable insights. While autonomous agents show promise, they often struggle with domain-specific nuances, intent alignment, and enterprise integration. We present Enterprise Deep Research (EDR), a multi-agent system that integrates (1) a Master Planning Agent for adaptive query decomposition, (2) four specialized search agents (General, Academic, GitHub, LinkedIn), (3) an extensible MCP-based tool ecosystem supporting NL2SQL, file analysis, and enterprise workflows, (4) a Visualization Agent for data-driven insights, and (5) a reflection mechanism that detects knowledge gaps and updates research direction with optional human-in-the-loop steering guidance. These components enable automated report generation, real-time streaming, and seamless enterprise deployment, as validated on internal datasets. On open-ended benchmarks including DeepResearch Bench and DeepConsult, EDR outperforms state-of-the-art agentic systems without any human steering. We release the EDR framework and benchmark trajectories to advance research on multi-agent reasoning applications.
Code at this https URL and Dataset at this https URL 

---
# Executable Knowledge Graphs for Replicating AI Research 

**Authors**: Yujie Luo, Zhuoyun Yu, Xuehai Wang, Yuqi Zhu, Ningyu Zhang, Lanning Wei, Lun Du, Da Zheng, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.17795)  

**Abstract**: Replicating AI research is a crucial yet challenging task for large language model (LLM) agents. Existing approaches often struggle to generate executable code, primarily due to insufficient background knowledge and the limitations of retrieval-augmented generation (RAG) methods, which fail to capture latent technical details hidden in referenced papers. Furthermore, previous approaches tend to overlook valuable implementation-level code signals and lack structured knowledge representations that support multi-granular retrieval and reuse. To overcome these challenges, we propose Executable Knowledge Graphs (xKG), a modular and pluggable knowledge base that automatically integrates technical insights, code snippets, and domain-specific knowledge extracted from scientific literature. When integrated into three agent frameworks with two different LLMs, xKG shows substantial performance gains (10.9% with o3-mini) on PaperBench, demonstrating its effectiveness as a general and extensible solution for automated AI research replication. Code will released at this https URL. 

---
# Foundational Automatic Evaluators: Scaling Multi-Task Generative Evaluator Training for Reasoning-Centric Domains 

**Authors**: Austin Xu, Xuan-Phi Nguyen, Yilun Zhou, Chien-Sheng Wu, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.17793)  

**Abstract**: Finetuning specialized generative evaluators has emerged as a popular paradigm to meet the increasing demand for scalable evaluation during both training and test-time. However, recent work has largely focused on applying new methodology, such as reinforcement learning (RL), to training evaluators, shying away from large-scale, data-driven development. In this work, we focus on data scaling, curating a set of 2.5M samples spanning five unique evaluation tasks (pairwise, step-level, reference-free and reference-based verification, and single rating) and multiple domains focused on reasoning evaluation. With our data, we train Foundational Automatic Reasoning Evaluators (FARE), a family of 8B and 20B (with 3.6B active) parameter evaluators, with a simple iterative rejection-sampling supervised finetuning (SFT) approach. FARE-8B challenges larger specialized RL-trained evaluators and FARE-20B sets the new standard for open-source evaluators, surpassing specialized 70B+ evaluators. Beyond static benchmarks, we evaluate FARE in real-world tasks: As inference-time rerankers, FARE-20B achieves near-oracle performance on MATH. As verifiers in RL training, FARE improves the downstream RL-trained model performance by up to 14.1% vs. string-matching verifiers. When initialized from FARE, a continually-finetuned FARE-Code outperforms gpt-oss-20B by 65% on evaluating test-case quality. 

---
# Evaluating Medical LLMs by Levels of Autonomy: A Survey Moving from Benchmarks to Applications 

**Authors**: Xiao Ye, Jacob Dineen, Zhaonan Li, Zhikun Xu, Weiyu Chen, Shijie Lu, Yuxi Huang, Ming Shen, Phu Tran, Ji-Eun Irene Yum, Muhammad Ali Khan, Muhammad Umar Afzal, Irbaz Bin Riaz, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.17764)  

**Abstract**: Medical Large language models achieve strong scores on standard benchmarks; however, the transfer of those results to safe and reliable performance in clinical workflows remains a challenge. This survey reframes evaluation through a levels-of-autonomy lens (L0-L3), spanning informational tools, information transformation and aggregation, decision support, and supervised agents. We align existing benchmarks and metrics with the actions permitted at each level and their associated risks, making the evaluation targets explicit. This motivates a level-conditioned blueprint for selecting metrics, assembling evidence, and reporting claims, alongside directions that link evaluation to oversight. By centering autonomy, the survey moves the field beyond score-based claims toward credible, risk-aware evidence for real clinical use. 

---
# Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations 

**Authors**: Tong Chen, Akari Asai, Luke Zettlemoyer, Hannaneh Hajishirzi, Faeze Brahman  

**Link**: [PDF](https://arxiv.org/pdf/2510.17733)  

**Abstract**: Language models often generate factually incorrect information unsupported by their training data, a phenomenon known as extrinsic hallucination. Existing mitigation approaches often degrade performance on open-ended generation and downstream tasks, limiting their practical utility. We propose an online reinforcement learning method using a novel binary retrieval-augmented reward (RAR) to address this tradeoff. Unlike continuous reward schemes, our approach assigns a reward of one only when the model's output is entirely factually correct, and zero otherwise. We evaluate our method on Qwen3 reasoning models across diverse tasks. For open-ended generation, binary RAR achieves a 39.3% reduction in hallucination rates, substantially outperforming both supervised training and continuous-reward RL baselines. In short-form question answering, the model learns calibrated abstention, strategically outputting "I don't know" when faced with insufficient parametric knowledge. This yields 44.4% and 21.7% fewer incorrect answers on PopQA and GPQA, respectively. Crucially, these factuality gains come without performance degradation on instruction following, math, or code, whereas continuous-reward RL, despite improving factuality, induces quality regressions. 

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
# Towards Mining Effective Pedagogical Strategies from Learner-LLM Educational Dialogues 

**Authors**: Liqun He, Manolis Mavrikis, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2510.17698)  

**Abstract**: Dialogue plays a crucial role in educational settings, yet existing evaluation methods for educational applications of large language models (LLMs) primarily focus on technical performance or learning outcomes, often neglecting attention to learner-LLM interactions. To narrow this gap, this AIED Doctoral Consortium paper presents an ongoing study employing a dialogue analysis approach to identify effective pedagogical strategies from learner-LLM dialogues. The proposed approach involves dialogue data collection, dialogue act (DA) annotation, DA pattern mining, and predictive model building. Early insights are outlined as an initial step toward future research. The work underscores the need to evaluate LLM-based educational applications by focusing on dialogue dynamics and pedagogical strategies. 

---
# Qomhra: A Bilingual Irish-English Large Language Model 

**Authors**: Joseph McInerney  

**Link**: [PDF](https://arxiv.org/pdf/2510.17652)  

**Abstract**: This paper introduces Qomhrá, a bilingual Irish-English large language model (LLM), developed under low-resource constraints presenting a complete pipeline spanning bilingual continued pre-training, instruction tuning, and alignment from human preferences. Newly accessible Irish corpora and English text are mixed and curated to improve Irish performance while preserving English ability. 6 closed-weight LLMs are judged for their Irish text generation by a native speaker, a learner and other LLMs. Google's Gemini-2.5-Pro is ranked the highest and is subsequently used to synthesise instruction tuning and human preference datasets. Two datasets are contributed leveraging Gemini-2.5-Pro: a 30K Irish-English parallel instruction tuning dataset and a 1K human preference dataset, generating accepted and rejected responses that show near perfect alignment with a native Irish speaker. Qomhrá is comprehensively evaluated across benchmarks testing translation, gender understanding, topic identification and world knowledge with gains of up to 29% in Irish and 44% in English. Qomhrá also undergoes instruction tuning and demonstrates clear progress in instruction following, crucial for chatbot functionality. 

---
# Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models 

**Authors**: Yuefeng Peng, Parnian Afshar, Megan Ganji, Thomas Butler, Amir Houmansadr, Mingxian Wang, Dezhi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.17620)  

**Abstract**: Large language models may encode sensitive information or outdated knowledge that needs to be removed, to ensure responsible and compliant model responses. Unlearning has emerged as an efficient alternative to full retraining, aiming to remove specific knowledge while preserving overall model utility. Existing evaluations of unlearning methods focus on (1) the extent of forgetting of the target knowledge (forget set) and (2) maintaining performance on the retain set (i.e., utility). However, these evaluations overlook an important usability aspect: users may still want the model to leverage the removed information if it is re-introduced in the prompt. In a systematic evaluation of six state-of-the-art unlearning methods, we find that they consistently impair such contextual utility. To address this, we augment unlearning objectives with a plug-in term that preserves the model's ability to use forgotten knowledge when it is present in context. Extensive experiments demonstrate that our approach restores contextual utility to near original levels while still maintaining effective forgetting and retain-set utility. 

---
# LawChain: Modeling Legal Reasoning Chains for Chinese Tort Case Analysis 

**Authors**: Huiyuan Xie, Chenyang Li, Huining Zhu, Chubin Zhang, Yuxiao Ye, Zhenghao Liu, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17602)  

**Abstract**: Legal reasoning is a fundamental component of legal analysis and decision-making. Existing computational approaches to legal reasoning predominantly rely on generic reasoning frameworks such as syllogism and IRAC, which do not comprehensively examine the nuanced processes that underpin legal reasoning. Moreover, current research has largely focused on criminal cases, with insufficient modeling for civil cases. In this work, we present a novel framework for explicitly modeling legal reasoning in the analysis of Chinese tort-related civil cases. We first operationalize the legal reasoning processes used in tort analysis into the LawChain framework. LawChain is a three-module reasoning framework, with each module consisting of multiple finer-grained sub-steps. Informed by the LawChain framework, we introduce the task of tort legal reasoning and construct an evaluation benchmark, LawChain$_{eval}$, to systematically assess the critical steps within analytical reasoning chains for tort analysis. Leveraging this benchmark, we evaluate state-of-the-art large language models for their legal reasoning ability in civil tort contexts. Our results indicate that current models still fall short in accurately handling crucial elements of tort legal reasoning. Furthermore, we introduce several baseline approaches that explicitly incorporate LawChain-style reasoning through prompting or post-training. We conduct further experiments on additional legal analysis tasks, such as Legal Named-Entity Recognition and Criminal Damages Calculation, to verify the generalizability of these baselines. The proposed baseline approaches achieve significant improvements in tort-related legal reasoning and generalize well to related legal analysis tasks, thus demonstrating the value of explicitly modeling legal reasoning chains to enhance the reasoning capabilities of language models. 

---
# HGAdapter: Hypergraph-based Adapters in Language Models for Code Summarization and Clone Detection 

**Authors**: Guang Yang, Yujie Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17591)  

**Abstract**: Pre-trained language models (PLMs) are increasingly being applied to code-related tasks. Although PLMs have achieved good results, they do not take into account potential high-order data correlations within the code. We propose three types of high-order correlations in code tokens, i.e. abstract syntax tree family correlation, lexical correlation, and line correlation. We design a tokens and hyperedges generator to capture these high-order data correlations. We improve the architecture of hypergraph neural networks and combine it with adapter tuning to propose a novel hypergraph-based adapter (HGAdapter) to fine-tune PLMs. HGAdapter can encode high-order data correlations and is allowed to be inserted into various PLMs to enhance performance. Experiments were conducted on several public datasets, including six languages of code summarization and code clone detection tasks. Our methods improved the performance of PLMs in datasets to varying degrees. Experimental results validate the introduction of high-order data correlations that contribute to improved effectiveness. 

---
# Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation 

**Authors**: Collin Zhang, Fei Huang, Chenhan Yuan, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17555)  

**Abstract**: Large language models (LLMs) often experience language confusion, which is the unintended mixing of languages during text generation. Current solutions to this problem either necessitate model retraining or cannot differentiate between harmful confusion and acceptable code-switching. This paper introduces the Language Confusion Gate (LCG), a lightweight, plug-in solution that filters tokens during decoding without altering the base LLM. The LCG is trained using norm-adjusted self-distillation to predict appropriate language families and apply masking only when needed. Our method is based on the findings that language confusion is infrequent, correct-language tokens are usually among the top predictions, and output token embedding norms are larger for high-resource languages, which biases sampling. When evaluated across various models, including Qwen3, GPT-OSS, Gemma3, Llama3.1, LCG decreases language confusion significantly, often by an order of magnitude, without negatively impacting task performance. Code is available at this https URL. 

---
# When Annotators Disagree, Topology Explains: Mapper, a Topological Tool for Exploring Text Embedding Geometry and Ambiguity 

**Authors**: Nisrine Rair, Alban Goupil, Valeriu Vrabie, Emmanuel Chochoy  

**Link**: [PDF](https://arxiv.org/pdf/2510.17548)  

**Abstract**: Language models are often evaluated with scalar metrics like accuracy, but such measures fail to capture how models internally represent ambiguity, especially when human annotators disagree. We propose a topological perspective to analyze how fine-tuned models encode ambiguity and more generally instances.
Applied to RoBERTa-Large on the MD-Offense dataset, Mapper, a tool from topological data analysis, reveals that fine-tuning restructures embedding space into modular, non-convex regions aligned with model predictions, even for highly ambiguous cases. Over $98\%$ of connected components exhibit $\geq 90\%$ prediction purity, yet alignment with ground-truth labels drops in ambiguous data, surfacing a hidden tension between structural confidence and label uncertainty.
Unlike traditional tools such as PCA or UMAP, Mapper captures this geometry directly uncovering decision regions, boundary collapses, and overconfident clusters. Our findings position Mapper as a powerful diagnostic tool for understanding how models resolve ambiguity. Beyond visualization, it also enables topological metrics that may inform proactive modeling strategies in subjective NLP tasks. 

---
# OncoReason: Structuring Clinical Reasoning in LLMs for Robust and Interpretable Survival Prediction 

**Authors**: Raghu Vamshi Hemadri, Geetha Krishna Guruju, Kristi Topollai, Anna Ewa Choromanska  

**Link**: [PDF](https://arxiv.org/pdf/2510.17532)  

**Abstract**: Predicting cancer treatment outcomes requires models that are both accurate and interpretable, particularly in the presence of heterogeneous clinical data. While large language models (LLMs) have shown strong performance in biomedical NLP, they often lack structured reasoning capabilities critical for high-stakes decision support. We present a unified, multi-task learning framework that aligns autoregressive LLMs with clinical reasoning for outcome prediction on the MSK-CHORD dataset. Our models are trained to jointly perform binary survival classification, continuous survival time regression, and natural language rationale generation. We evaluate three alignment strategies: (1) standard supervised fine-tuning (SFT), (2) SFT with Chain-of-Thought (CoT) prompting to elicit step-by-step reasoning, and (3) Group Relative Policy Optimization (GRPO), a reinforcement learning method that aligns model outputs to expert-derived reasoning trajectories. Experiments with LLaMa3-8B and Med42-8B backbones demonstrate that CoT prompting improves F1 by +6.0 and reduces MAE by 12%, while GRPO achieves state-of-the-art interpretability and predictive performance across BLEU, ROUGE, and BERTScore. We further show that existing biomedical LLMs often fail to produce valid reasoning traces due to architectural constraints. Our findings underscore the importance of reasoning-aware alignment in multi-task clinical modeling and set a new benchmark for interpretable, trustworthy LLMs in precision oncology. 

---
# SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors 

**Authors**: Tiancheng Hu, Joachim Baumann, Lorenzo Lupo, Dirk Hovy, Nigel Collier, Paul Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2510.17516)  

**Abstract**: Large language model (LLM) simulations of human behavior have the potential to revolutionize the social and behavioral sciences, if and only if they faithfully reflect real human behaviors. Current evaluations are fragmented, based on bespoke tasks and metrics, creating a patchwork of incomparable results. To address this, we introduce SimBench, the first large-scale, standardized benchmark for a robust, reproducible science of LLM simulation. By unifying 20 diverse datasets covering tasks from moral decision-making to economic choice across a large global participant pool, SimBench provides the necessary foundation to ask fundamental questions about when, how, and why LLM simulations succeed or fail. We show that, while even the best LLMs today have limited simulation ability (score: 40.80/100), performance scales log-linearly with model size. Simulation performance is not improved by increased inference-time compute. We demonstrate an alignment-simulation trade-off: instruction-tuning improves performance on low-entropy (consensus) questions but degrades it on high-entropy (diverse) ones. Models particularly struggle when simulating specific demographic groups. Finally, we demonstrate that simulation ability correlates most strongly with deep, knowledge-intensive reasoning (MMLU-Pro, r=0.939). By making progress measurable, we aim to accelerate the development of more faithful LLM simulators. 

---
# Annotation-Efficient Universal Honesty Alignment 

**Authors**: Shiyu Ni, Keping Bi, Jiafeng Guo, Minghao Tang, Jingtong Wu, Zengxin Han, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.17509)  

**Abstract**: Honesty alignment-the ability of large language models (LLMs) to recognize their knowledge boundaries and express calibrated confidence-is essential for trustworthy deployment. Existing methods either rely on training-free confidence estimation (e.g., token probabilities, self-consistency) or training-based calibration with correctness annotations. While effective, achieving universal honesty alignment with training-based calibration requires costly, large-scale labeling. To support annotation-efficient training, we introduce Elicitation-Then-Calibration (EliCal), a two-stage framework that first elicits internal confidence using inexpensive self-consistency supervision, then calibrates this confidence with a small set of correctness annotations. To support a large-scale study, we release HonestyBench, a benchmark covering ten free-form QA datasets with 560k training and 70k evaluation instances annotated with correctness and self-consistency signals. Experiments show that EliCal achieves near-optimal alignment with only 1k correctness annotations (0.18% of full supervision) and better alignment performance on unseen MMLU tasks than the calibration-only baseline, offering a scalable solution toward universal honesty alignment in LLMs. 

---
# Lingua Custodi's participation at the WMT 2025 Terminology shared task 

**Authors**: Jingshu Liu, Raheel Qader, Gaëtan Caillaut, Mariam Nakhlé  

**Link**: [PDF](https://arxiv.org/pdf/2510.17504)  

**Abstract**: While BERT is an effective method for learning monolingual sentence embeddings for semantic similarity and embedding based transfer learning BERT based cross-lingual sentence embeddings have yet to be explored. We systematically investigate methods for learning multilingual sentence embeddings by combining the best methods for learning monolingual and cross-lingual representations including: masked language modeling (MLM), translation language modeling (TLM), dual encoder translation ranking, and additive margin softmax. We show that introducing a pre-trained multilingual language model dramatically reduces the amount of parallel training data required to achieve good performance by 80%. Composing the best of these methods produces a model that achieves 83.7% bi-text retrieval accuracy over 112 languages on Tatoeba, well above the 65.5 achieved by LASER, while still performing competitively on monolingual transfer learning benchmarks. Parallel data mined from CommonCrawl using our best model is shown to train competitive NMT models for en-zh and en-de. We publicly release our best multilingual sentence embedding model for 109+ languages at this https URL. 

---
# Deep Self-Evolving Reasoning 

**Authors**: Zihan Liu, Shun Zheng, Xumeng Wen, Yang Wang, Jiang Bian, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17498)  

**Abstract**: Long-form chain-of-thought reasoning has become a cornerstone of advanced reasoning in large language models. While recent verification-refinement frameworks have enabled proprietary models to solve Olympiad-level problems, their effectiveness hinges on strong, reliable verification and correction capabilities, which remain fragile in open-weight, smaller-scale models. This work demonstrates that even with weak verification and refinement capabilities on hard tasks, the reasoning limits of such models can be substantially extended through a probabilistic paradigm we call Deep Self-Evolving Reasoning (DSER). We conceptualize iterative reasoning as a Markov chain, where each step represents a stochastic transition in the solution space. The key insight is that convergence to a correct solution is guaranteed as long as the probability of improvement marginally exceeds that of degradation. By running multiple long-horizon, self-evolving processes in parallel, DSER amplifies these small positive tendencies, enabling the model to asymptotically approach correct answers. Empirically, we apply DSER to the DeepSeek-R1-0528-Qwen3-8B model. On the challenging AIME 2024-2025 benchmark, DSER solves 5 out of 9 previously unsolvable problems and boosts overall performance, enabling this compact model to surpass the single-turn accuracy of its 600B-parameter teacher through majority voting. Beyond its immediate utility for test-time scaling, the DSER framework serves to diagnose the fundamental limitations of current open-weight reasoners. By clearly delineating their shortcomings in self-verification, refinement, and stability, our findings establish a clear research agenda for developing next-generation models with powerful, intrinsic self-evolving capabilities. 

---
# Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents 

**Authors**: Yihong Tang, Kehai Chen, Liang Yue, Jinxin Fan, Caishen Zhou, Xiaoguang Li, Yuyang Zhang, Mingming Zhao, Shixiong Kai, Kaiyang Guo, Xingshan Zeng, Wenjing Cun, Lifeng Shang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17491)  

**Abstract**: With the rise of large language models (LLMs), LLM agents capable of autonomous reasoning, planning, and executing complex tasks have become a frontier in artificial intelligence. However, how to translate the research on general agents into productivity that drives industry transformations remains a significant challenge. To address this, this paper systematically reviews the technologies, applications, and evaluation methods of industry agents based on LLMs. Using an industry agent capability maturity framework, it outlines the evolution of agents in industry applications, from "process execution systems" to "adaptive social systems." First, we examine the three key technological pillars that support the advancement of agent capabilities: Memory, Planning, and Tool Use. We discuss how these technologies evolve from supporting simple tasks in their early forms to enabling complex autonomous systems and collective intelligence in more advanced forms. Then, we provide an overview of the application of industry agents in real-world domains such as digital engineering, scientific discovery, embodied intelligence, collaborative business execution, and complex system simulation. Additionally, this paper reviews the evaluation benchmarks and methods for both fundamental and specialized capabilities, identifying the challenges existing evaluation systems face regarding authenticity, safety, and industry specificity. Finally, we focus on the practical challenges faced by industry agents, exploring their capability boundaries, developmental potential, and governance issues in various scenarios, while providing insights into future directions. By combining technological evolution with industry practices, this review aims to clarify the current state and offer a clear roadmap and theoretical foundation for understanding and building the next generation of industry agents. 

---
# DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning 

**Authors**: Yongxin He, Shan Zhang, Yixuan Cao, Lei Ma, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.17489)  

**Abstract**: Detecting AI-involved text is essential for combating misinformation, plagiarism, and academic misconduct. However, AI text generation includes diverse collaborative processes (AI-written text edited by humans, human-written text edited by AI, and AI-generated text refined by other AI), where various or even new LLMs could be involved. Texts generated through these varied processes exhibit complex characteristics, presenting significant challenges for detection. Current methods model these processes rather crudely, primarily employing binary classification (purely human vs. AI-involved) or multi-classification (treating human-AI collaboration as a new class). We observe that representations of texts generated through different processes exhibit inherent clustering relationships. Therefore, we propose DETree, a novel approach that models the relationships among different processes as a Hierarchical Affinity Tree structure, and introduces a specialized loss function that aligns text representations with this tree. To facilitate this learning, we developed RealBench, a comprehensive benchmark dataset that automatically incorporates a wide spectrum of hybrid texts produced through various human-AI collaboration processes. Our method improves performance in hybrid text detection tasks and significantly enhances robustness and generalization in out-of-distribution scenarios, particularly in few-shot learning conditions, further demonstrating the promise of training-based approaches in OOD settings. Our code and dataset are available at this https URL. 

---
# ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts 

**Authors**: Zheyue Tan, Zhiyuan Li, Tao Yuan, Dong Zhou, Weilin Liu, Yueqing Zhuang, Yadong Li, Guowei Niu, Cheng Qin, Zhuyu Yao, Congyi Liu, Haiyang Xu, Boxun Li, Guohao Dai, Bo Zhao, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17483)  

**Abstract**: Mixture-of-Experts (MoE) architectures have emerged as a promising approach to scale Large Language Models (LLMs). MoE boosts the efficiency by activating a subset of experts per token. Recent works show that fine-grained experts substantially enriches the combinatorial flexibility of active experts and enhances model expressiveness. However, such a design is fundamentally limited by the layer-local routing mechanism: each layer is restricted to its own expert pool. This requires a careful trade-off between expert dimensionality and routing diversity given fixed parameter budgets. We describe ReXMoE, a novel MoE architecture that improves routing beyond the existing layer-local approaches by allowing routers to reuse experts across adjacent layers. ReXMoE decouples expert dimensionality from per-layer budgets, enabling richer expert combinations without sacrificing individual expert capacity or inflating overall parameters. To this end, we propose a new progressive scaling routing (PSR) strategy to gradually increase the candidate expert pool during training. As a result, ReXMoE improves both language modeling and downstream task performance. Extensive experiments on models ranging from 0.5B to 7B parameters across different architectures demonstrate that ReXMoE consistently improves performance under fixed architectural dimensions, confirming ReXMoE as new design paradigm for parameter-efficient and scalable MoE-based LLMs. 

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
# Multilingual Clinical NER for Diseases and Medications Recognition in Cardiology Texts using BERT Embeddings 

**Authors**: Manuela Daniela Danu, George Marica, Constantin Suciu, Lucian Mihai Itu, Oladimeji Farri  

**Link**: [PDF](https://arxiv.org/pdf/2510.17437)  

**Abstract**: The rapidly increasing volume of electronic health record (EHR) data underscores a pressing need to unlock biomedical knowledge from unstructured clinical texts to support advancements in data-driven clinical systems, including patient diagnosis, disease progression monitoring, treatment effects assessment, prediction of future clinical events, etc. While contextualized language models have demonstrated impressive performance improvements for named entity recognition (NER) systems in English corpora, there remains a scarcity of research focused on clinical texts in low-resource languages. To bridge this gap, our study aims to develop multiple deep contextual embedding models to enhance clinical NER in the cardiology domain, as part of the BioASQ MultiCardioNER shared task. We explore the effectiveness of different monolingual and multilingual BERT-based models, trained on general domain text, for extracting disease and medication mentions from clinical case reports written in English, Spanish, and Italian. We achieved an F1-score of 77.88% on Spanish Diseases Recognition (SDR), 92.09% on Spanish Medications Recognition (SMR), 91.74% on English Medications Recognition (EMR), and 88.9% on Italian Medications Recognition (IMR). These results outperform the mean and median F1 scores in the test leaderboard across all subtasks, with the mean/median values being: 69.61%/75.66% for SDR, 81.22%/90.18% for SMR, 89.2%/88.96% for EMR, and 82.8%/87.76% for IMR. 

---
# Agentic Reinforcement Learning for Search is Unsafe 

**Authors**: Yushi Yang, Shreyansh Padarha, Andrew Lee, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2510.17431)  

**Abstract**: Agentic reinforcement learning (RL) trains large language models to autonomously call tools during reasoning, with search as the most common application. These models excel at multi-step reasoning tasks, but their safety properties are not well understood. In this study, we show that RL-trained search models inherit refusal from instruction tuning and often deflect harmful requests by turning them into safe queries. However, this safety is fragile. Two simple attacks, one that forces the model to begin response with search (Search attack), another that encourages models to repeatedly search (Multi-search attack), trigger cascades of harmful searches and answers. Across two model families (Qwen, Llama) with both local and web search, these attacks lower refusal rates by up to 60.0%, answer safety by 82.5%, and search-query safety by 82.4%. The attacks succeed by triggering models to generate harmful, request-mirroring search queries before they can generate the inherited refusal tokens. This exposes a core weakness of current RL training: it rewards continued generation of effective queries without accounting for their harmfulness. As a result, RL search models have vulnerabilities that users can easily exploit, making it urgent to develop safety-aware agentic RL pipelines optimising for safe search. 

---
# Navigating the Alignment-Calibration Trade-off: A Pareto-Superior Frontier via Model Merging 

**Authors**: Tiancheng Hu, Benjamin Minixhofer, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2510.17426)  

**Abstract**: The "alignment tax" of post-training is typically framed as a drop in task accuracy. We show it also involves a severe loss of calibration, making models overconfident, less reliable, and model outputs less diverse. We show that this trade-off can be navigated effectively via a simple post-hoc intervention: interpolating between a model's weights before and after alignment. Crucially, this is not a strict trade-off. We find that the process consistently reveals Pareto-optimal interpolations - models that improve accuracy beyond both parents while substantially recovering the calibration lost during alignment. Our work demonstrates that simple model merging provides a computationally efficient method for mitigating the full scope of the alignment tax, yielding models that are more capable and more reliable. 

---
# BenCao: An Instruction-Tuned Large Language Model for Traditional Chinese Medicine 

**Authors**: Jiacheng Xie, Yang Yu, Yibo Chen, Hanyao Zhang, Lening Zhao, Jiaxuan He, Lei Jiang, Xiaoting Tang, Guanghui An, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17415)  

**Abstract**: Traditional Chinese Medicine (TCM), with a history spanning over two millennia, plays a role in global healthcare. However, applying large language models (LLMs) to TCM remains challenging due to its reliance on holistic reasoning, implicit logic, and multimodal diagnostic cues. Existing TCM-domain LLMs have made progress in text-based understanding but lack multimodal integration, interpretability, and clinical applicability. To address these limitations, we developed BenCao, a ChatGPT-based multimodal assistant for TCM, integrating structured knowledge bases, diagnostic data, and expert feedback refinement. BenCao was trained through natural language instruction tuning rather than parameter retraining, aligning with expert-level reasoning and ethical norms specific to TCM. The system incorporates a comprehensive knowledge base of over 1,000 classical and modern texts, a scenario-based instruction framework for diverse interactions, a chain-of-thought simulation mechanism for interpretable reasoning, and a feedback refinement process involving licensed TCM practitioners. BenCao connects to external APIs for tongue-image classification and multimodal database retrieval, enabling dynamic access to diagnostic resources. In evaluations across single-choice question benchmarks and multimodal classification tasks, BenCao achieved superior accuracy to general-domain and TCM-domain models, particularly in diagnostics, herb recognition, and constitution classification. The model was deployed as an interactive application on the OpenAI GPTs Store, accessed by nearly 1,000 users globally as of October 2025. This study demonstrates the feasibility of developing a TCM-domain LLM through natural language-based instruction tuning and multimodal integration, offering a practical framework for aligning generative AI with traditional medical reasoning and a scalable pathway for real-world deployment. 

---
# AFRICAPTION: Establishing a New Paradigm for Image Captioning in African Languages 

**Authors**: Mardiyyah Oduwole, Prince Mireku, Fatimo Adebanjo, Oluwatosin Olajide, Mahi Aminu Aliyu, Jekaterina Novikova  

**Link**: [PDF](https://arxiv.org/pdf/2510.17405)  

**Abstract**: Multimodal AI research has overwhelmingly focused on high-resource languages, hindering the democratization of advancements in the field. To address this, we present AfriCaption, a comprehensive framework for multilingual image captioning in 20 African languages and our contributions are threefold: (i) a curated dataset built on Flickr8k, featuring semantically aligned captions generated via a context-aware selection and translation process; (ii) a dynamic, context-preserving pipeline that ensures ongoing quality through model ensembling and adaptive substitution; and (iii) the AfriCaption model, a 0.5B parameter vision-to-text architecture that integrates SigLIP and NLLB200 for caption generation across under-represented languages. This unified framework ensures ongoing data quality and establishes the first scalable image-captioning resource for under-represented African languages, laying the groundwork for truly inclusive multimodal AI. 

---
# Leveraging Group Relative Policy Optimization to Advance Large Language Models in Traditional Chinese Medicine 

**Authors**: Jiacheng Xie, Shuai Zeng, Yang Yu, Xiaoting Tang, Guanghui An, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17402)  

**Abstract**: Traditional Chinese Medicine (TCM) presents a rich and structurally unique knowledge system that challenges conventional applications of large language models (LLMs). Although previous TCM-specific LLMs have shown progress through supervised fine-tuning, they often face limitations in alignment, data quality, and evaluation consistency. In this study, we introduce Ladder-base, the first TCM-focused LLM trained with Group Relative Policy Optimization (GRPO), a reinforcement learning method that improves reasoning and factual consistency by optimizing response selection based on intra-group comparisons. Ladder-base is built upon the Qwen2.5-7B-Instruct foundation model and trained exclusively on the textual subset of the TCM-Ladder benchmark, using 80 percent of the data for training and the remaining 20 percent split evenly between validation and test sets. Through standardized evaluation, Ladder-base demonstrates superior performance across multiple reasoning metrics when compared to both state-of-the-art general-purpose LLMs such as GPT-4, Gemini 2.5, Claude 3, and Qwen3 and domain-specific TCM models including BenTsao, HuatuoGPT2, and Zhongjing. These findings suggest that GRPO provides an effective and efficient strategy for aligning LLMs with expert-level reasoning in traditional medical domains and supports the development of trustworthy and clinically grounded TCM artificial intelligence systems. 

---
# EduAdapt: A Question Answer Benchmark Dataset for Evaluating Grade-Level Adaptability in LLMs 

**Authors**: Numaan Naeem, Abdellah El Mekki, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2510.17389)  

**Abstract**: Large language models (LLMs) are transforming education by answering questions, explaining complex concepts, and generating content across a wide range of subjects. Despite strong performance on academic benchmarks, they often fail to tailor responses to students' grade levels. This is a critical need in K-12 education, where age-appropriate vocabulary and explanation are essential for effective learning. Existing models frequently produce outputs that are too advanced or vague for younger learners, and there are no standardized benchmarks to evaluate their ability to adjust across cognitive and developmental stages. To address this gap, we introduce EduAdapt, a benchmark of nearly 48k grade-labeled QA pairs across nine science subjects, spanning Grades 1-12 and grouped into four grade levels. We evaluate a diverse set of open-source LLMs on EduAdapt and find that while larger models generally perform better, they still struggle with generating suitable responses for early-grade students (Grades 1-5). Our work presents the first dataset and evaluation framework for assessing grade-level adaptability in LLMs, aiming to foster more developmentally aligned educational AI systems through better training and prompting strategies. EduAdapt code and datasets are publicly available at this https URL. 

---
# The Atomic Instruction Gap: Instruction-Tuned LLMs Struggle with Simple, Self-Contained Directives 

**Authors**: Henry Lim, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.17388)  

**Abstract**: Instruction-tuned large language models (IT-LLMs) exhibit strong zero-shot reasoning, yet their ability to execute simple, self-contained instructions remains underexplored, despite this being foundational to complex instruction-following. We evaluate 20 IT-LLMs on modified MMLU and MMLU-Pro benchmarks, by systematically varying the format of option labels (alphabetic, numeric, Roman) while keeping their meaning identical under four paradigms, namely: (1) With explicit instructions, label changes cause large performance shifts (e.g., -30.45\% for Roman vs. numeric), revealing instruction-format bias. (2) Without instructions, performance drops further (up to -10.84\%) and label sensitivity intensifies, underscoring the role of explicit guidance. (3) When option contents are removed, models fail random-choice baselines except with numeric labels, suggesting weak adherence to atomic directives. (4) Three-shot exemplars yield no significant gains in robustness or fidelity, and generation analyses show persistent label errors, especially for non-numeric formats. Across model sizes, larger LLMs achieve higher accuracy but remain inconsistent in instruction adherence. These results expose the insufficiencies of current instruction-tuning paradigms and highlight the need for evaluation methods and training strategies that explicitly target atomic instruction-following. 

---
# Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation 

**Authors**: Chenghao Zhang, Guanting Dong, Xinyu Yang, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.17354)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) by retrieving relevant documents from an external corpus. However, existing RAG systems primarily focus on unimodal text documents, and often fall short in real-world scenarios where both queries and documents may contain mixed modalities (such as text and images). In this paper, we address the challenge of Universal Retrieval-Augmented Generation (URAG), which involves retrieving and reasoning over mixed-modal information to improve vision-language generation. To this end, we propose Nyx, a unified mixed-modal to mixed-modal retriever tailored for URAG scenarios. To mitigate the scarcity of realistic mixed-modal data, we introduce a four-stage automated pipeline for generation and filtering, leveraging web documents to construct NyxQA, a dataset comprising diverse mixed-modal question-answer pairs that better reflect real-world information needs. Building on this high-quality dataset, we adopt a two-stage training framework for Nyx: we first perform pre-training on NyxQA along with a variety of open-source retrieval datasets, followed by supervised fine-tuning using feedback from downstream vision-language models (VLMs) to align retrieval outputs with generative preferences. Experimental results demonstrate that Nyx not only performs competitively on standard text-only RAG benchmarks, but also excels in the more general and realistic URAG setting, significantly improving generation quality in vision-language tasks. 

---
# Addressing Antisocial Behavior in Multi-Party Dialogs Through Multimodal Representation Learning 

**Authors**: Hajar Bakarou, Mohamed Sinane El Messoussi, Anaïs Ollagnier  

**Link**: [PDF](https://arxiv.org/pdf/2510.17289)  

**Abstract**: Antisocial behavior (ASB) on social media -- including hate speech, harassment, and cyberbullying -- poses growing risks to platform safety and societal well-being. Prior research has focused largely on networks such as X and Reddit, while \textit{multi-party conversational settings} remain underexplored due to limited data. To address this gap, we use \textit{CyberAgressionAdo-Large}, a French open-access dataset simulating ASB in multi-party conversations, and evaluate three tasks: \textit{abuse detection}, \textit{bullying behavior analysis}, and \textit{bullying peer-group identification}. We benchmark six text-based and eight graph-based \textit{representation-learning methods}, analyzing lexical cues, interactional dynamics, and their multimodal fusion. Results show that multimodal models outperform unimodal baselines. The late fusion model \texttt{mBERT + WD-SGCN} achieves the best overall results, with top performance on abuse detection (0.718) and competitive scores on peer-group identification (0.286) and bullying analysis (0.606). Error analysis highlights its effectiveness in handling nuanced ASB phenomena such as implicit aggression, role transitions, and context-dependent hostility. 

---
# TaxoAlign: Scholarly Taxonomy Generation Using Language Models 

**Authors**: Avishek Lahiri, Yufang Hou, Debarshi Kumar Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.17263)  

**Abstract**: Taxonomies play a crucial role in helping researchers structure and navigate knowledge in a hierarchical manner. They also form an important part in the creation of comprehensive literature surveys. The existing approaches to automatic survey generation do not compare the structure of the generated surveys with those written by human experts. To address this gap, we present our own method for automated taxonomy creation that can bridge the gap between human-generated and automatically-created taxonomies. For this purpose, we create the CS-TaxoBench benchmark which consists of 460 taxonomies that have been extracted from human-written survey papers. We also include an additional test set of 80 taxonomies curated from conference survey papers. We propose TaxoAlign, a three-phase topic-based instruction-guided method for scholarly taxonomy generation. Additionally, we propose a stringent automated evaluation framework that measures the structural alignment and semantic coherence of automatically generated taxonomies in comparison to those created by human experts. We evaluate our method and various baselines on CS-TaxoBench, using both automated evaluation metrics and human evaluation studies. The results show that TaxoAlign consistently surpasses the baselines on nearly all metrics. The code and data can be found at this https URL. 

---
# Explainability of Large Language Models: Opportunities and Challenges toward Generating Trustworthy Explanations 

**Authors**: Shahin Atakishiyev, Housam K.B. Babiker, Jiayi Dai, Nawshad Farruque, Teruaki Hayashi, Nafisa Sadaf Hriti, Md Abed Rahman, Iain Smith, Mi-Young Kim, Osmar R. Zaïane, Randy Goebel  

**Link**: [PDF](https://arxiv.org/pdf/2510.17256)  

**Abstract**: Large language models have exhibited impressive performance across a broad range of downstream tasks in natural language processing. However, how a language model predicts the next token and generates content is not generally understandable by humans. Furthermore, these models often make errors in prediction and reasoning, known as hallucinations. These errors underscore the urgent need to better understand and interpret the intricate inner workings of language models and how they generate predictive outputs. Motivated by this gap, this paper investigates local explainability and mechanistic interpretability within Transformer-based large language models to foster trust in such models. In this regard, our paper aims to make three key contributions. First, we present a review of local explainability and mechanistic interpretability approaches and insights from relevant studies in the literature. Furthermore, we describe experimental studies on explainability and reasoning with large language models in two critical domains -- healthcare and autonomous driving -- and analyze the trust implications of such explanations for explanation receivers. Finally, we summarize current unaddressed issues in the evolving landscape of LLM explainability and outline the opportunities, critical challenges, and future directions toward generating human-aligned, trustworthy LLM explanations. 

---
# How News Feels: Understanding Affective Bias in Multilingual Headlines for Human-Centered Media Design 

**Authors**: Mohd Ruhul Ameen, Akif Islam, Abu Saleh Musa Miah, Ayesha Siddiqua, Jungpil Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17252)  

**Abstract**: News media often shape the public mood not only by what they report but by how they frame it. The same event can appear calm in one outlet and alarming in another, reflecting subtle emotional bias in reporting. Negative or emotionally charged headlines tend to attract more attention and spread faster, which in turn encourages outlets to frame stories in ways that provoke stronger reactions. This research explores that tendency through large-scale emotion analysis of Bengali news. Using zero-shot inference with Gemma-3 4B, we analyzed 300000 Bengali news headlines and their content to identify the dominant emotion and overall tone of each. The findings reveal a clear dominance of negative emotions, particularly anger, fear, and disappointment, and significant variation in how similar stories are emotionally portrayed across outlets. Based on these insights, we propose design ideas for a human-centered news aggregator that visualizes emotional cues and helps readers recognize hidden affective framing in daily news. 

---
# From Preferences to Prejudice: The Role of Alignment Tuning in Shaping Social Bias in Video Diffusion Models 

**Authors**: Zefan Cai, Haoyi Qiu, Haozhe Zhao, Ke Wan, Jiachen Li, Jiuxiang Gu, Wen Xiao, Nanyun Peng, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17247)  

**Abstract**: Recent advances in video diffusion models have significantly enhanced text-to-video generation, particularly through alignment tuning using reward models trained on human preferences. While these methods improve visual quality, they can unintentionally encode and amplify social biases. To systematically trace how such biases evolve throughout the alignment pipeline, we introduce VideoBiasEval, a comprehensive diagnostic framework for evaluating social representation in video generation. Grounded in established social bias taxonomies, VideoBiasEval employs an event-based prompting strategy to disentangle semantic content (actions and contexts) from actor attributes (gender and ethnicity). It further introduces multi-granular metrics to evaluate (1) overall ethnicity bias, (2) gender bias conditioned on ethnicity, (3) distributional shifts in social attributes across model variants, and (4) the temporal persistence of bias within videos. Using this framework, we conduct the first end-to-end analysis connecting biases in human preference datasets, their amplification in reward models, and their propagation through alignment-tuned video diffusion models. Our results reveal that alignment tuning not only strengthens representational biases but also makes them temporally stable, producing smoother yet more stereotyped portrayals. These findings highlight the need for bias-aware evaluation and mitigation throughout the alignment process to ensure fair and socially responsible video generation. 

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
# Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models 

**Authors**: Jiaqi Leng, Xiang Hu, Junxiong Wang, Jianguo Li, Wei Wu, Yucheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17196)  

**Abstract**: Effectively processing long contexts is a critical challenge for language models. While standard Transformers are limited by quadratic complexity and poor length extrapolation, alternative architectures like sliding window attention and state space models sacrifice the ability to effectively utilize the full context due to their fixed-size memory. Chunk-based sparse attention has emerged as a promising paradigm for extreme length generalization, yet the key architectural principles underpinning its success are not yet fully understood. In this work, we present a systematic dissection of these models to identify the core components driving their performance. Through a unified framework and comprehensive ablation studies, we demonstrate that a combination of three design principles is critical: (1) an expressive, non-linear Chunk Encoder with a dedicated CLS token to produce representations for retrieval; (2) a Bypassing Residual Path to stably integrate retrieved global information without it being overridden by the local residual stream; and (3) enforced selection sparsity during pre-training to bridge the train-test distribution gap. We provide a theoretical motivation for intra-chunk information processing and landmark generation. By combining these principles, we establish a new state-of-the-art for training-free length extrapolation, successfully generalizing models trained on a 4K context to 32 million tokens on RULER and BABILong. Our findings provide a clear and empirically-grounded set of design principles for developing future, highly-capable long-context language models. 

---
# When AI companions become witty: Can human brain recognize AI-generated irony? 

**Authors**: Xiaohui Rao, Hanlin Wu, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.17168)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as social agents and trained to produce humor and irony, a question emerges: when encountering witty AI remarks, do people interpret these as intentional communication or mere computational output? This study investigates whether people adopt the intentional stance, attributing mental states to explain behavior,toward AI during irony comprehension. Irony provides an ideal paradigm because it requires distinguishing intentional contradictions from unintended errors through effortful semantic reanalysis. We compared behavioral and neural responses to ironic statements from AI versus human sources using established ERP components: P200 reflecting early incongruity detection and P600 indexing cognitive efforts in reinterpreting incongruity as deliberate irony. Results demonstrate that people do not fully adopt the intentional stance toward AI-generated irony. Behaviorally, participants attributed incongruity to deliberate communication for both sources, though significantly less for AI than human, showing greater tendency to interpret AI incongruities as computational errors. Neural data revealed attenuated P200 and P600 effects for AI-generated irony, suggesting reduced effortful detection and reanalysis consistent with diminished attribution of communicative intent. Notably, people who perceived AI as more sincere showed larger P200 and P600 effects for AI-generated irony, suggesting that intentional stance adoption is calibrated by specific mental models of artificial agents. These findings reveal that source attribution shapes neural processing of social-communicative phenomena. Despite current LLMs' linguistic sophistication, achieving genuine social agency requires more than linguistic competence, it necessitates a shift in how humans perceive and attribute intentionality to artificial agents. 

---
# Rethinking On-policy Optimization for Query Augmentation 

**Authors**: Zhichao Xu, Shengyao Zhuang, Xueguang Ma, Bingsen Chen, Yijun Tian, Fengran Mo, Jie Cao, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.17139)  

**Abstract**: Recent advances in large language models (LLMs) have led to a surge of interest in query augmentation for information retrieval (IR). Two main approaches have emerged. The first prompts LLMs to generate answers or pseudo-documents that serve as new queries, relying purely on the model's parametric knowledge or contextual information. The second applies reinforcement learning (RL) to fine-tune LLMs for query rewriting, directly optimizing retrieval metrics. While having respective advantages and limitations, the two approaches have not been compared under consistent experimental conditions. In this work, we present the first systematic comparison of prompting-based and RL-based query augmentation across diverse benchmarks, including evidence-seeking, ad hoc, and tool retrieval. Our key finding is that simple, training-free query augmentation often performs on par with, or even surpasses, more expensive RL-based counterparts, especially when using powerful LLMs. Motivated by this discovery, we introduce a novel hybrid method, On-policy Pseudo-document Query Expansion (OPQE), which, instead of rewriting a query, the LLM policy learns to generate a pseudo-document that maximizes retrieval performance, thus merging the flexibility and generative structure of prompting with the targeted optimization of RL. We show OPQE outperforms both standalone prompting and RL-based rewriting, demonstrating that a synergistic approach yields the best results. Our implementation is made available to facilitate reproducibility. 

---
# DVAGen: Dynamic Vocabulary Augmented Generation 

**Authors**: Wei Du, Nuowei Liu, Jie Wang, Jiahao Kuang, Tao Ji, Xiaoling Wang, Yuanbin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17115)  

**Abstract**: Language models trained with a fixed vocabulary struggle to generalize to novel or out-of-vocabulary words, limiting their flexibility in handling diverse token combinations. Existing dynamic vocabulary approaches attempt to address this limitation but face challenges such as fragmented codebases, lack of support for modern LLMs, and limited inference scalability. To overcome these issues, we introduce DVAGen, a fully open-source, unified framework designed for training, evaluation, and visualization of dynamic vocabulary-augmented language models. Our framework modularizes the pipeline for ease of customization, integrates seamlessly with open-source LLMs, and is the first to provide both CLI and WebUI tools for real-time result inspection. We validate the effectiveness of dynamic vocabulary methods on modern LLMs and demonstrate support for batch inference, significantly improving inference throughput. 

---
# Verification-Aware Planning for Multi-Agent Systems 

**Authors**: Tianyang Xu, Dan Zhang, Kushan Mitra, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2510.17109)  

**Abstract**: Large language model (LLM) agents are increasingly deployed to tackle complex tasks, often necessitating collaboration among multiple specialized agents. However, multi-agent collaboration introduces new challenges in planning, coordination, and verification. Execution failures frequently arise not from flawed reasoning alone, but from subtle misalignments in task interpretation, output format, or inter-agent handoffs. To address these challenges, we present VeriMAP, a framework for multi-agent collaboration with verification-aware planning. The VeriMAP planner decomposes tasks, models subtask dependencies, and encodes planner-defined passing criteria as subtask verification functions (VFs) in Python and natural language. We evaluate VeriMAP on diverse datasets, demonstrating that it outperforms both single- and multi-agent baselines while enhancing system robustness and interpretability. Our analysis highlights how verification-aware planning enables reliable coordination and iterative refinement in multi-agent systems, without relying on external labels or annotations. 

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
# Extended LSTM: Adaptive Feature Gating for Toxic Comment Classification 

**Authors**: Noor Islam S. Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2510.17018)  

**Abstract**: Toxic comment detection remains a challenging task, where transformer-based models (e.g., BERT) incur high computational costs and degrade on minority toxicity classes, while classical ensembles lack semantic adaptability. We propose xLSTM, a parameter-efficient and theoretically grounded framework that unifies cosine-similarity gating, adaptive feature prioritization, and principled class rebalancing. A learnable reference vector {v} in {R}^d modulates contextual embeddings via cosine similarity, amplifying toxic cues and attenuating benign signals to yield stronger gradients under severe class imbalance. xLSTM integrates multi-source embeddings (GloVe, FastText, BERT CLS) through a projection layer, a character-level BiLSTM for morphological cues, embedding-space SMOTE for minority augmentation, and adaptive focal loss with dynamic class weighting. On the Jigsaw Toxic Comment benchmark, xLSTM attains 96.0% accuracy and 0.88 macro-F1, outperforming BERT by 33% on threat and 28% on identity_hate categories, with 15 times fewer parameters and 50ms inference latency. Cosine gating contributes a +4.8% F1 gain in ablations. The results establish a new efficiency adaptability frontier, demonstrating that lightweight, theoretically informed architectures can surpass large pretrained models on imbalanced, domain-specific NLP tasks. 

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
# Online Learning Defense against Iterative Jailbreak Attacks via Prompt Optimization 

**Authors**: Masahiro Kaneko, Zeerak Talat, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17006)  

**Abstract**: Iterative jailbreak methods that repeatedly rewrite and input prompts into large language models (LLMs) to induce harmful outputs -- using the model's previous responses to guide each new iteration -- have been found to be a highly effective attack strategy. Despite being an effective attack strategy against LLMs and their safety mechanisms, existing defenses do not proactively disrupt this dynamic trial-and-error cycle. In this study, we propose a novel framework that dynamically updates its defense strategy through online learning in response to each new prompt from iterative jailbreak methods. Leveraging the distinctions between harmful jailbreak-generated prompts and typical harmless prompts, we introduce a reinforcement learning-based approach that optimizes prompts to ensure appropriate responses for harmless tasks while explicitly rejecting harmful prompts. Additionally, to curb overfitting to the narrow band of partial input rewrites explored during an attack, we introduce Past-Direction Gradient Damping (PDGD). Experiments conducted on three LLMs show that our approach significantly outperforms five existing defense methods against five iterative jailbreak methods. Moreover, our results indicate that our prompt optimization strategy simultaneously enhances response quality for harmless tasks. 

---
# Vocab Diet: Reshaping the Vocabulary of LLMs with Vector Arithmetic 

**Authors**: Yuval Reif, Guy Kaplan, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2510.17001)  

**Abstract**: Large language models (LLMs) were shown to encode word form variations, such as "walk"->"walked", as linear directions in embedding space. However, standard tokenization algorithms treat these variations as distinct tokens -- filling the size-capped vocabulary with surface form variants (e.g., "walk", "walking", "Walk"), at the expense of less frequent words and multilingual coverage. We show that many of these variations can be captured by transformation vectors -- additive offsets that yield the appropriate word's representation when applied to the base form word embedding -- in both the input and output spaces. Building on this, we propose a compact reshaping of the vocabulary: rather than assigning unique tokens to each surface form, we compose them from shared base form and transformation vectors (e.g., "walked" = "walk" + past tense). We apply our approach to multiple LLMs and across five languages, removing up to 10% of vocabulary entries -- thereby freeing space to allocate new, more diverse tokens. Importantly, we do so while also expanding vocabulary coverage to out-of-vocabulary words, with minimal impact on downstream performance, and without modifying model weights. Our findings motivate a foundational rethinking of vocabulary design, moving from string enumeration to a compositional vocabulary that leverages the underlying structure of language. 

---
# Back to Bytes: Revisiting Tokenization Through UTF-8 

**Authors**: Amit Moryossef, Clara Meister, Pavel Stepachev, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2510.16987)  

**Abstract**: We present UTF8Tokenizer, a minimalist byte-level tokenizer that maps text exactly to IDs corresponding to the bytes underlying the text's UTF-8 encoding (e.g., byte x09 is token ID 9). Unlike prior byte-level approaches (Xue et al., 2021; Pagnoni et al., 2025), our implementation never introduces out-of-range IDs (i.e. there is no token ID 256) or auxiliary tokens: all special behavior (e.g., padding, boundaries, conversation structure, attention segments, tool calling, "thinking" spans, etc.) is encoded using C0 control bytes - just as ASCII was originally designed to embed control information alongside printable text. These design principles yield practical benefits: (1) faster tokenization (14x) and significantly lower host-device transfer (8x less than int64); (2) simple, shareable 256*d embedding tables that can be aligned across models; and (3) a training-time enhancement via bit-biased embeddings, which exposes per-byte bit structure and can be added to the embedding table post-training, removing inference costs. Our HuggingFace-compatible implementation improves language modeling convergence. 

---
# Parameter-Efficient Fine-Tuning for Low-Resource Languages: A Comparative Study of LLMs for Bengali Hate Speech Detection 

**Authors**: Akif Islam, Mohd Ruhul Ameen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16985)  

**Abstract**: Bengali social media platforms have witnessed a sharp increase in hate speech, disproportionately affecting women and adolescents. While datasets such as BD-SHS provide a basis for structured evaluation, most prior approaches rely on either computationally costly full-model fine-tuning or proprietary APIs. This paper presents the first application of Parameter-Efficient Fine-Tuning (PEFT) for Bengali hate speech detection using LoRA and QLoRA. Three instruction-tuned large language models - Gemma-3-4B, Llama-3.2-3B, and Mistral-7B - were fine-tuned on the BD-SHS dataset of 50,281 annotated comments. Each model was adapted by training fewer than 1% of its parameters, enabling experiments on a single consumer-grade GPU. The results show that Llama-3.2-3B achieved the highest F1-score of 92.23%, followed by Mistral-7B at 88.94% and Gemma-3-4B at 80.25%. These findings establish PEFT as a practical and replicable strategy for Bengali and related low-resource languages. 

---
# Prompt-MII: Meta-Learning Instruction Induction for LLMs 

**Authors**: Emily Xiao, Yixiao Zeng, Ada Chen, Chin-Jou Li, Amanda Bertsch, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.16932)  

**Abstract**: A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose PROMPT-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. PROMPT-MII improves downstream model quality by 4-9 F1 points (10-20% relative), matching ICL performance while requiring 3-13x fewer tokens. 

---
# ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models 

**Authors**: Emily Chang, Niyati Bafna  

**Link**: [PDF](https://arxiv.org/pdf/2510.16928)  

**Abstract**: Existing benchmarks for large language models (LLMs) are largely restricted to high- or mid-resource languages, and often evaluate performance on higher-order tasks in reasoning and generation. However, plenty of evidence points to the fact that LLMs lack basic linguistic competence in the vast majority of the world's 3800+ written languages. We introduce ChiKhaPo, consisting of 8 subtasks of varying difficulty designed to evaluate the lexical comprehension and generation abilities of generative models. ChiKhaPo draws on existing lexicons, monolingual data, and bitext, and provides coverage for 2700+ languages for 2 subtasks, surpassing any existing benchmark in terms of language coverage. We further show that 6 SOTA models struggle on our benchmark, and discuss the factors contributing to performance scores, including language family, language resourcedness, task, and comprehension versus generation directions. With ChiKhaPo, we hope to enable and encourage the massively multilingual benchmarking of LLMs. 

---
# Does Visual Grounding Enhance the Understanding of Embodied Knowledge in Large Language Models? 

**Authors**: Zhihui Yang, Yupei Wang, Kaijie Mo, Zhe Zhao, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16924)  

**Abstract**: Despite significant progress in multimodal language models (LMs), it remains unclear whether visual grounding enhances their understanding of embodied knowledge compared to text-only models. To address this question, we propose a novel embodied knowledge understanding benchmark based on the perceptual theory from psychology, encompassing visual, auditory, tactile, gustatory, olfactory external senses, and interoception. The benchmark assesses the models' perceptual abilities across different sensory modalities through vector comparison and question-answering tasks with over 1,700 questions. By comparing 30 state-of-the-art LMs, we surprisingly find that vision-language models (VLMs) do not outperform text-only models in either task. Moreover, the models perform significantly worse in the visual dimension compared to other sensory dimensions. Further analysis reveals that the vector representations are easily influenced by word form and frequency, and the models struggle to answer questions involving spatial perception and reasoning. Our findings underscore the need for more effective integration of embodied knowledge in LMs to enhance their understanding of the physical world. 

---
# Neuronal Group Communication for Efficient Neural representation 

**Authors**: Zhengqi Pei, Qingming Huang, Shuhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16851)  

**Abstract**: The ever-increasing scale of modern neural networks has brought unprecedented performance alongside daunting challenges in efficiency and interpretability. This paper addresses the core question of how to build large neural systems that learn efficient, modular, and interpretable representations. We propose Neuronal Group Communication (NGC), a theory-driven framework that reimagines a neural network as a dynamical system of interacting neuronal groups rather than a monolithic collection of neural weights. Instead of treating each weight as an independent trainable parameter, NGC treats weights as transient interactions between embedding-like neuronal states, with neural computation unfolding through iterative communication among groups of neurons. This low-rank, modular representation yields compact models: groups of neurons exchange low-dimensional signals, enabling intra-group specialization and inter-group information sharing while dramatically reducing redundant parameters. By drawing on dynamical systems theory, we introduce a neuronal stability metric (analogous to Lyapunov stability) that quantifies the contraction of neuron activations toward stable patterns during sequence processing. Using this metric, we reveal that emergent reasoning capabilities correspond to an external driving force or ``potential'', which nudges the neural dynamics away from trivial trajectories while preserving stability. Empirically, we instantiate NGC in large language models (LLMs) and demonstrate improved performance on complex reasoning benchmarks under moderate compression. NGC consistently outperforms standard low-rank approximations and cross-layer basis-sharing methods at comparable compression rates. We conclude by discussing the broader implications of NGC, including how structured neuronal group dynamics might relate to generalization in high-dimensional learning systems. 

---
# FinSight: Towards Real-World Financial Deep Research 

**Authors**: Jiajie Jin, Yuyao Zhang, Yimeng Xu, Hongjin Qian, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.16844)  

**Abstract**: Generating professional financial reports is a labor-intensive and intellectually demanding process that current AI systems struggle to fully automate. To address this challenge, we introduce FinSight (Financial InSight), a novel multi agent framework for producing high-quality, multimodal financial reports. The foundation of FinSight is the Code Agent with Variable Memory (CAVM) architecture, which unifies external data, designed tools, and agents into a programmable variable space, enabling flexible data collection, analysis and report generation through executable code. To ensure professional-grade visualization, we propose an Iterative Vision-Enhanced Mechanism that progressively refines raw visual outputs into polished financial charts. Furthermore, a two stage Writing Framework expands concise Chain-of-Analysis segments into coherent, citation-aware, and multimodal reports, ensuring both analytical depth and structural consistency. Experiments on various company and industry-level tasks demonstrate that FinSight significantly outperforms all baselines, including leading deep research systems in terms of factual accuracy, analytical depth, and presentation quality, demonstrating a clear path toward generating reports that approach human-expert quality. 

---
# Who's Asking? Simulating Role-Based Questions for Conversational AI Evaluation 

**Authors**: Navreet Kaur, Hoda Ayad, Hayoung Jung, Shravika Mittal, Munmun De Choudhury, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2510.16829)  

**Abstract**: Language model users often embed personal and social context in their questions. The asker's role -- implicit in how the question is framed -- creates specific needs for an appropriate response. However, most evaluations, while capturing the model's capability to respond, often ignore who is asking. This gap is especially critical in stigmatized domains such as opioid use disorder (OUD), where accounting for users' contexts is essential to provide accessible, stigma-free responses. We propose CoRUS (COmmunity-driven Roles for User-centric Question Simulation), a framework for simulating role-based questions. Drawing on role theory and posts from an online OUD recovery community (r/OpiatesRecovery), we first build a taxonomy of asker roles -- patients, caregivers, practitioners. Next, we use it to simulate 15,321 questions that embed each role's goals, behaviors, and experiences. Our evaluations show that these questions are both highly believable and comparable to real-world data. When used to evaluate five LLMs, for the same question but differing roles, we find systematic differences: vulnerable roles, such as patients and caregivers, elicit more supportive responses (+17%) and reduced knowledge content (-19%) in comparison to practitioners. Our work demonstrates how implicitly signaling a user's role shapes model responses, and provides a methodology for role-informed evaluation of conversational AI. 

---
# Cross-Genre Authorship Attribution via LLM-Based Retrieve-and-Rerank 

**Authors**: Shantanu Agarwal, Joel Barry, Steven Fincke, Scott Miller  

**Link**: [PDF](https://arxiv.org/pdf/2510.16819)  

**Abstract**: Authorship attribution (AA) is the task of identifying the most likely author of a query document from a predefined set of candidate authors. We introduce a two-stage retrieve-and-rerank framework that finetunes LLMs for cross-genre AA. Unlike the field of information retrieval (IR), where retrieve-and-rerank is a de facto strategy, cross-genre AA systems must avoid relying on topical cues and instead learn to identify author-specific linguistic patterns that are independent of the text's subject matter (genre/domain/topic). Consequently, for the reranker, we demonstrate that training strategies commonly used in IR are fundamentally misaligned with cross-genre AA, leading to suboptimal behavior. To address this, we introduce a targeted data curation strategy that enables the reranker to effectively learn author-discriminative signals. Using our LLM-based retrieve-and-rerank pipeline, we achieve substantial gains of 22.3 and 34.4 absolute Success@8 points over the previous state-of-the-art on HIATUS's challenging HRS1 and HRS2 cross-genre AA benchmarks. 

---
# Knowing the Facts but Choosing the Shortcut: Understanding How Large Language Models Compare Entities 

**Authors**: Hans Hergen Lehmann, Jae Hee Lee, Steven Schockaert, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2510.16815)  

**Abstract**: Large Language Models (LLMs) are increasingly used for knowledge-based reasoning tasks, yet understanding when they rely on genuine knowledge versus superficial heuristics remains challenging. We investigate this question through entity comparison tasks by asking models to compare entities along numerical attributes (e.g., ``Which river is longer, the Danube or the Nile?''), which offer clear ground truth for systematic analysis. Despite having sufficient numerical knowledge to answer correctly, LLMs frequently make predictions that contradict this knowledge. We identify three heuristic biases that strongly influence model predictions: entity popularity, mention order, and semantic co-occurrence. For smaller models, a simple logistic regression using only these surface cues predicts model choices more accurately than the model's own numerical predictions, suggesting heuristics largely override principled reasoning. Crucially, we find that larger models (32B parameters) selectively rely on numerical knowledge when it is more reliable, while smaller models (7--8B parameters) show no such discrimination, which explains why larger models outperform smaller ones even when the smaller models possess more accurate knowledge. Chain-of-thought prompting steers all models towards using the numerical features across all model sizes. 

---
# MOSAIC: Masked Objective with Selective Adaptation for In-domain Contrastive Learning 

**Authors**: Vera Pavlova, Mohammed Makhlouf  

**Link**: [PDF](https://arxiv.org/pdf/2510.16797)  

**Abstract**: We introduce MOSAIC (Masked Objective with Selective Adaptation for In-domain Contrastive learning), a multi-stage framework for domain adaptation of sentence embedding models that incorporates joint domain-specific masked supervision. Our approach addresses the challenges of adapting large-scale general-domain sentence embedding models to specialized domains. By jointly optimizing masked language modeling (MLM) and contrastive objectives within a unified training pipeline, our method enables effective learning of domain-relevant representations while preserving the robust semantic discrimination properties of the original model. We empirically validate our approach on both high-resource and low-resource domains, achieving improvements up to 13.4% in NDCG@10 (Normalized Discounted Cumulative Gain) over strong general-domain baselines. Comprehensive ablation studies further demonstrate the effectiveness of each component, highlighting the importance of balanced joint supervision and staged adaptation. 

---
# LC-Eval: A Bilingual Multi-Task Evaluation Benchmark for Long-Context Understanding 

**Authors**: Sheikh Jubair, Arwa Omayrah, Amal Alshammari, Alhanoof Althnian, Abdulhamed Alothaimen, Norah A. Alzahrani, Shahad D. Alzaidi, Nora Al-Twairesh, Abdulmohsen Al-Thubaity  

**Link**: [PDF](https://arxiv.org/pdf/2510.16783)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated sophisticated capabilities, including the ability to process and comprehend extended contexts. These emergent capabilities necessitate rigorous evaluation methods to effectively assess their performance in long-context understanding. In this paper, we present \textbf{LC-Eval}, a bilingual, multi-task evaluation benchmark designed to evaluate long-context understanding in English and Arabic, targeting context lengths ranging from 4k to over 128k tokens. LC-Eval introduces four novel and challenging tasks: multi-document question answering, bilingual question answering, claim verification within a paragraph, and multiple-choice questions based on long contexts. These tasks are designed to assess LLMs' abilities in deep reasoning, document comprehension, information tracing, and bilingual information extraction and understanding. The benchmark includes datasets in both Arabic and English for each task, allowing for a comparative analysis of their performance across different text genres. Evaluations were conducted on both open-weight and closed LLMs, with results indicating that LC-Eval presents significant challenges. Even high-performing models, such as GPT-4o, struggled with certain tasks, highlighting the complexity and rigor of the benchmark. 

---
# Enhancing Language Agent Strategic Reasoning through Self-Play in Adversarial Games 

**Authors**: Yikai Zhang, Ye Rong, Siyu Yuan, Jiangjie Chen, Jian Xie, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.16761)  

**Abstract**: Existing language agents often encounter difficulties in dynamic adversarial games due to poor strategic reasoning. To mitigate this limitation, a promising approach is to allow agents to learn from game interactions automatically, without relying on costly expert-labeled data. Unlike static environments where agents receive fixed feedback or rewards, selecting appropriate opponents in dynamic adversarial games can significantly impact learning performance. However, the discussion of opponents in adversarial environments remains an area under exploration. In this paper, we propose a Step-level poliCy Optimization method through Play-And-Learn, SCO-PAL. Leveraging SCO-PAL, we conduct a detailed analysis of opponent selection by setting opponents at different levels and find that self-play is the most effective way to improve strategic reasoning in such adversarial environments. Utilizing SCO-PAL with self-play, we increase the average win rate against four opponents by approximately 30% compared to baselines and achieve a 54.76% win rate against GPT-4 in six adversarial games. 

---
# Beacon: Single-Turn Diagnosis and Mitigation of Latent Sycophancy in Large Language Models 

**Authors**: Sanskar Pandey, Ruhaan Chopra, Angkul Puniya, Sohom Pal  

**Link**: [PDF](https://arxiv.org/pdf/2510.16727)  

**Abstract**: Large language models internalize a structural trade-off between truthfulness and obsequious flattery, emerging from reward optimization that conflates helpfulness with polite submission. This latent bias, known as sycophancy, manifests as a preference for user agreement over principled reasoning. We introduce Beacon, a single-turn forced-choice benchmark that isolates this bias independent of conversational context, enabling precise measurement of the tension between factual accuracy and submissive bias. Evaluations across twelve state-of-the-art models reveal that sycophancy decomposes into stable linguistic and affective sub-biases, each scaling with model capacity. We further propose prompt-level and activation-level interventions that modulate these biases in opposing directions, exposing the internal geometry of alignment as a dynamic manifold between truthfulness and socially compliant judgment. Beacon reframes sycophancy as a measurable form of normative misgeneralization, providing a reproducible foundation for studying and mitigating alignment drift in large-scale generative systems. 

---
# so much depends / upon / a whitespace: Why Whitespace Matters for Poets and LLMs 

**Authors**: Sriharsh Bhyravajjula, Melanie Walsh, Anna Preus, Maria Antoniak  

**Link**: [PDF](https://arxiv.org/pdf/2510.16713)  

**Abstract**: Whitespace is a critical component of poetic form, reflecting both adherence to standardized forms and rebellion against those forms. Each poem's whitespace distribution reflects the artistic choices of the poet and is an integral semantic and spatial feature of the poem. Yet, despite the popularity of poetry as both a long-standing art form and as a generation task for large language models (LLMs), whitespace has not received sufficient attention from the NLP community. Using a corpus of 19k English-language published poems from Poetry Foundation, we investigate how 4k poets have used whitespace in their works. We release a subset of 2.8k public-domain poems with preserved formatting to facilitate further research in this area. We compare whitespace usage in the published poems to (1) 51k LLM-generated poems, and (2) 12k unpublished poems posted in an online community. We also explore whitespace usage across time periods, poetic forms, and data sources. Additionally, we find that different text processing methods can result in significantly different representations of whitespace in poetry data, motivating us to use these poems and whitespace patterns to discuss implications for the processing strategies used to assemble pretraining datasets for LLMs. 

---
# The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models 

**Authors**: Shivam Ratnakar, Sanjay Raghavendra  

**Link**: [PDF](https://arxiv.org/pdf/2510.16712)  

**Abstract**: Integration of Large Language Models with search/retrieval engines has become ubiquitous, yet these systems harbor a critical vulnerability that undermines their reliability. We present the first systematic investigation of "chameleon behavior" in LLMs: their alarming tendency to shift stances when presented with contradictory questions in multi-turn conversations (especially in search-enabled LLMs). Through our novel Chameleon Benchmark Dataset, comprising 17,770 carefully crafted question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains, we expose fundamental flaws in state-of-the-art systems. We introduce two theoretically grounded metrics: the Chameleon Score (0-1) that quantifies stance instability, and Source Re-use Rate (0-1) that measures knowledge diversity. Our rigorous evaluation of Llama-4-Maverick, GPT-4o-mini, and Gemini-2.5-Flash reveals consistent failures: all models exhibit severe chameleon behavior (scores 0.391-0.511), with GPT-4o-mini showing the worst performance. Crucially, small across-temperature variance (less than 0.004) suggests the effect is not a sampling artifact. Our analysis uncovers the mechanism: strong correlations between source re-use rate and confidence (r=0.627) and stance changes (r=0.429) are statistically significant (p less than 0.05), indicating that limited knowledge diversity makes models pathologically deferential to query framing. These findings highlight the need for comprehensive consistency evaluation before deploying LLMs in healthcare, legal, and financial systems where maintaining coherent positions across interactions is critical for reliable decision support. 

---
# Natural Language Processing Applications in Cardiology: A Narrative Review 

**Authors**: Kailai Yang, Yan Leng, Xin Zhang, Tianlin Zhang, Paul Thompson, Bernard Keavney, Maciej Tomaszewski, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2510.16708)  

**Abstract**: Cardiovascular disease has become increasingly prevalent in modern society and has a significant effect on global health and well-being. Heart-related conditions are intricate, multifaceted disorders, which may be influenced by a combination of genetic predispositions, lifestyle choices, and various socioeconomic and clinical factors. Information regarding these potentially complex interrelationships is dispersed among diverse types of textual data, which include patient narratives, medical records, and scientific literature, among others. Natural language processing (NLP) techniques have increasingly been adopted as a powerful means to analyse and make sense of this vast amount of unstructured data. This, in turn, can allow healthcare professionals to gain deeper insights into the cardiology field, which has the potential to revolutionize current approaches to the diagnosis, treatment, and prevention of cardiac problems. This review provides a detailed overview of NLP research in cardiology between 2014 and 2025. We queried six literature databases to find articles describing the application of NLP techniques in the context of a range of different cardiovascular diseases. Following a rigorous screening process, we identified a total of 265 relevant articles. We analysed each article from multiple dimensions, i.e., NLP paradigm types, cardiology-related task types, cardiovascular disease types, and data source types. Our analysis reveals considerable diversity within each of these dimensions, thus demonstrating the considerable breadth of NLP research within the field. We also perform a temporal analysis, which illustrates the evolution and changing trends in NLP methods employed over the last decade that we cover. To our knowledge, the review constitutes the most comprehensive overview of NLP research in cardiology to date. 

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
# All You Need is One: Capsule Prompt Tuning with a Single Vector 

**Authors**: Yiyang Liu, James C. Liang, Heng Fan, Wenhao Yang, Yiming Cui, Xiaotian Han, Lifu Huang, Dongfang Liu, Qifan Wang, Cheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.16670)  

**Abstract**: Prompt-based learning has emerged as a parameter-efficient finetuning (PEFT) approach to facilitate Large Language Model (LLM) adaptation to downstream tasks by conditioning generation with task-aware guidance. Despite its successes, current prompt-based learning methods heavily rely on laborious grid searching for optimal prompt length and typically require considerable number of prompts, introducing additional computational burden. Worse yet, our pioneer findings indicate that the task-aware prompt design is inherently limited by its absence of instance-aware information, leading to a subtle attention interplay with the input sequence. In contrast, simply incorporating instance-aware information as a part of the guidance can enhance the prompt-tuned model performance without additional fine-tuning. Moreover, we find an interesting phenomenon, namely "attention anchor", that incorporating instance-aware tokens at the earliest position of the sequence can successfully preserve strong attention to critical structural information and exhibit more active attention interaction with all input tokens. In light of our observation, we introduce Capsule Prompt-Tuning (CaPT), an efficient and effective solution that leverages off-the-shelf, informative instance semantics into prompt-based learning. Our approach innovatively integrates both instance-aware and task-aware information in a nearly parameter-free manner (i.e., one single capsule prompt). Empirical results demonstrate that our method can exhibit superior performance across various language tasks (e.g., 84.03\% average accuracy on T5-Large), serving as an "attention anchor," while enjoying high parameter efficiency (e.g., 0.003\% of model parameters on Llama3.2-1B). 

---
# Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration 

**Authors**: Zhixuan He, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.16645)  

**Abstract**: Large Language Models (LLMs) demonstrate strong performance but often lack interpretable reasoning. This paper introduces the Multi-Agent Collaboration Framework for Diverse Thinking Modes (DiMo), which enhances both performance and interpretability by simulating a structured debate among four specialized LLM agents. Each agent embodies a distinct reasoning paradigm, allowing the framework to collaboratively explore diverse cognitive approaches. Through iterative debate, agents challenge and refine initial responses, yielding more robust conclusions and an explicit, auditable reasoning chain. Across six benchmarks and under a unified open-source setup, DiMo improves accuracy over widely used single-model and debate baselines, with the largest gains on math. We position DiMo as a semantics-aware, Web-native multi-agent framework: it models human-machine intelligence with LLM agents that produce semantically typed, URL-annotated evidence chains for explanations and user-friendly interactions. Although our experiments use standard reasoning benchmarks, the framework is designed to be instantiated over Web corpora and knowledge graphs, combining retrieval-augmented reasoning with structured justifications that downstream systems can inspect and reuse. 

---
# Fine-tuning of Large Language Models for Constituency Parsing Using a Sequence to Sequence Approach 

**Authors**: Francisco Jose Cortes Delgado, Eduardo Martinez Gracia, Rafael Valencia Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.16604)  

**Abstract**: Recent advances in natural language processing with large neural models have opened new possibilities for syntactic analysis based on machine learning. This work explores a novel approach to phrase-structure analysis by fine-tuning large language models (LLMs) to translate an input sentence into its corresponding syntactic structure. The main objective is to extend the capabilities of MiSintaxis, a tool designed for teaching Spanish syntax. Several models from the Hugging Face repository were fine-tuned using training data generated from the AnCora-ES corpus, and their performance was evaluated using the F1 score. The results demonstrate high accuracy in phrase-structure analysis and highlight the potential of this methodology. 

---
# AI-Generated Text Detection in Low-Resource Languages: A Case Study on Urdu 

**Authors**: Muhammad Ammar, Hadiya Murad Hadi, Usman Majeed Butt  

**Link**: [PDF](https://arxiv.org/pdf/2510.16573)  

**Abstract**: Large Language Models (LLMs) are now capable of generating text that closely resembles human writing, making them powerful tools for content creation, but this growing ability has also made it harder to tell whether a piece of text was written by a human or by a machine. This challenge becomes even more serious for languages like Urdu, where there are very few tools available to detect AI-generated text. To address this gap, we propose a novel AI-generated text detection framework tailored for the Urdu language. A balanced dataset comprising 1,800 humans authored, and 1,800 AI generated texts, sourced from models such as Gemini, GPT-4o-mini, and Kimi AI was developed. Detailed linguistic and statistical analysis was conducted, focusing on features such as character and word counts, vocabulary richness (Type Token Ratio), and N-gram patterns, with significance evaluated through t-tests and MannWhitney U tests. Three state-of-the-art multilingual transformer models such as mdeberta-v3-base, distilbert-base-multilingualcased, and xlm-roberta-base were fine-tuned on this dataset. The mDeBERTa-v3-base achieved the highest performance, with an F1-score 91.29 and accuracy of 91.26% on the test set. This research advances efforts in contesting misinformation and academic misconduct in Urdu-speaking communities and contributes to the broader development of NLP tools for low resource languages. 

---
# Hallucination Benchmark for Speech Foundation Models 

**Authors**: Alkis Koudounas, Moreno La Quatra, Manuel Giollo, Sabato Marco Siniscalchi, Elena Baralis  

**Link**: [PDF](https://arxiv.org/pdf/2510.16567)  

**Abstract**: Hallucinations in automatic speech recognition (ASR) systems refer to fluent and coherent transcriptions produced by neural ASR models that are completely unrelated to the underlying acoustic input (i.e., the speech signal). While similar to conventional decoding errors in potentially compromising the usability of transcriptions for downstream applications, hallucinations can be more detrimental due to their preservation of syntactically and semantically plausible structure. This apparent coherence can mislead subsequent processing stages and introduce serious risks, particularly in critical domains such as healthcare and law. Conventional evaluation metrics are primarily centered on error-based metrics and fail to distinguish between phonetic inaccuracies and hallucinations. Consequently, there is a critical need for new evaluation frameworks that can effectively identify and assess models with a heightened propensity for generating hallucinated content. To this end, we introduce SHALLOW, the first benchmark framework that systematically categorizes and quantifies hallucination phenomena in ASR along four complementary axes: lexical, phonetic, morphological, and semantic. We define targeted metrics within each category to produce interpretable profiles of model behavior. Through evaluation across various architectures and speech domains, we have found that SHALLOW metrics correlate strongly with word error rate (WER) when recognition quality is high (i.e., low WER). Still, this correlation weakens substantially as WER increases. SHALLOW, therefore, captures fine-grained error patterns that WER fails to distinguish under degraded and challenging conditions. Our framework supports specific diagnosis of model weaknesses and provides feedback for model improvement beyond what aggregate error rates can offer. 

---
# Language over Content: Tracing Cultural Understanding in Multilingual Large Language Models 

**Authors**: Seungho Cho, Changgeon Ko, Eui Jun Hwang, Junmyeong Lee, Huije Lee, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.16565)  

**Abstract**: Large language models (LLMs) are increasingly used across diverse cultural contexts, making accurate cultural understanding essential. Prior evaluations have mostly focused on output-level performance, obscuring the factors that drive differences in responses, while studies using circuit analysis have covered few languages and rarely focused on culture. In this work, we trace LLMs' internal cultural understanding mechanisms by measuring activation path overlaps when answering semantically equivalent questions under two conditions: varying the target country while fixing the question language, and varying the question language while fixing the country. We also use same-language country pairs to disentangle language from cultural aspects. Results show that internal paths overlap more for same-language, cross-country questions than for cross-language, same-country questions, indicating strong language-specific patterns. Notably, the South Korea-North Korea pair exhibits low overlap and high variability, showing that linguistic similarity does not guarantee aligned internal representation. 

---
# ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation 

**Authors**: Haoxuan Zhang, Ruochi Li, Sarthak Shrestha, Shree Harshini Mamidala, Revanth Putta, Arka Krishan Aggarwal, Ting Xiao, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16549)  

**Abstract**: Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. Recent work has focused on using LLMs to improve review efficiency or generate insightful review content. However, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine the peer review ecosystem and compromise academic integrity. To address this critical issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews. ReviewGuard employs a comprehensive four-stage LLM-driven framework that: (1) collects ICLR and NeurIPS papers with their corresponding reviews from OpenReview; (2) annotates review types using GPT-4.1 with human validation; (3) addresses class imbalance and data scarcity through LLM-driven synthetic data augmentation, producing a final corpus of 6,634 papers, 24,657 real reviews, and 46,438 synthetic reviews; and (4) fine-tunes both encoder-based models and open source LLMs. We perform comprehensive feature analysis of the structure and quality of the review text. Compared to sufficient reviews, deficient reviews demonstrate lower rating scores, higher self-reported confidence, reduced structural complexity, and a higher proportion of negative sentiment. AI-generated text detection reveals that, since ChatGPT's emergence, AI-generated reviews have increased dramatically. In the evaluation of deficient review detection models, mixed training with synthetic and real review data provides substantial enhancements to recall and F1 scores on the binary task. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review while offering valuable insights into human-AI collaboration to maintain academic integrity. 

---
# Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection 

**Authors**: Michelle Yuan, Khushbu Pahwa, Shuaichen Chang, Mustafa Kaba, Jiarong Jiang, Xiaofei Ma, Yi Zhang, Monica Sunkara  

**Link**: [PDF](https://arxiv.org/pdf/2510.16499)  

**Abstract**: Designing effective agentic systems requires the seamless composition and integration of agents, tools, and models within dynamic and uncertain environments. Most existing methods rely on static, semantic retrieval approaches for tool or agent discovery. However, effective reuse and composition of existing components remain challenging due to incomplete capability descriptions and the limitations of retrieval methods. Component selection suffers because the decisions are not based on capability, cost, and real-time utility. To address these challenges, we introduce a structured, automated framework for agentic system composition that is inspired by the knapsack problem. Our framework enables a composer agent to systematically identify, select, and assemble an optimal set of agentic components by jointly considering performance, budget constraints, and compatibility. By dynamically testing candidate components and modeling their utility in real-time, our approach streamlines the assembly of agentic systems and facilitates scalable reuse of resources. Empirical evaluation with Claude 3.5 Sonnet across five benchmarking datasets shows that our online-knapsack-based composer consistently lies on the Pareto frontier, achieving higher success rates at significantly lower component costs compared to our baselines. In the single-agent setup, the online knapsack composer shows a success rate improvement of up to 31.6% in comparison to the retrieval baselines. In multi-agent systems, the online knapsack composer increases success rate from 37% to 87% when agents are selected from an agent inventory of 100+ agents. The substantial performance gap confirms the robust adaptability of our method across diverse domains and budget constraints. 

---
# Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety 

**Authors**: Vamshi Krishna Bonagiri, Ponnurangam Kumaragurum, Khanh Nguyen, Benjamin Plaut  

**Link**: [PDF](https://arxiv.org/pdf/2510.16492)  

**Abstract**: As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications. 

---
# Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations 

**Authors**: Pingjun Hong, Beiduo Chen, Siyao Peng, Marie-Catherine de Marneffe, Benjamin Roth, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2510.16458)  

**Abstract**: Natural Language Inference datasets often exhibit human label variation. To better understand these variations, explanation-based approaches analyze the underlying reasoning behind annotators' decisions. One such approach is the LiTEx taxonomy, which categorizes free-text explanations in English into reasoning types. However, previous work applying such taxonomies has focused on within-label variation: cases where annotators agree on the final NLI label but provide different explanations. In contrast, this paper broadens the scope by examining how annotators may diverge not only in the reasoning type but also in the labeling step. We use explanations as a lens to decompose the reasoning process underlying NLI annotation and to analyze individual differences. We apply LiTEx to two NLI English datasets and align annotation variation from multiple aspects: NLI label agreement, explanation similarity, and taxonomy agreement, with an additional compounding factor of annotators' selection bias. We observe instances where annotators disagree on the label but provide highly similar explanations, suggesting that surface-level disagreement may mask underlying agreement in interpretation. Moreover, our analysis reveals individual preferences in explanation strategies and label choices. These findings highlight that agreement in reasoning types better reflects the semantic similarity of free-text explanations than label agreement alone. Our findings underscore the richness of reasoning-based explanations and the need for caution in treating labels as ground truth. 

---
# RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning 

**Authors**: Deyi Ji, Yuekui Yang, Haiyang Wu, Shaoping Ma, Tianrun Chen, Lanyun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16455)  

**Abstract**: Advertisement (Ad) video violation detection is critical for ensuring platform compliance, but existing methods struggle with precise temporal grounding, noisy annotations, and limited generalization. We propose RAVEN, a novel framework that integrates curriculum reinforcement learning with multimodal large language models (MLLMs) to enhance reasoning and cognitive capabilities for violation detection. RAVEN employs a progressive training strategy, combining precisely and coarsely annotated data, and leverages Group Relative Policy Optimization (GRPO) to develop emergent reasoning abilities without explicit reasoning annotations. Multiple hierarchical sophisticated reward mechanism ensures precise temporal grounding and consistent category prediction. Experiments on industrial datasets and public benchmarks show that RAVEN achieves superior performances in violation category accuracy and temporal interval localization. We also design a pipeline to deploy the RAVEN on the online Ad services, and online A/B testing further validates its practical applicability, with significant improvements in precision and recall. RAVEN also demonstrates strong generalization, mitigating the catastrophic forgetting issue associated with supervised fine-tuning. 

---
# TrajSelector: Harnessing Latent Representations for Efficient and Effective Best-of-N in Large Reasoning Model 

**Authors**: Bin Yu, Xinming Wang, Shijie Lian, Haotian Li, Changti Wu, Ruina Hu, Bailing Wang, Yuliang Wei, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16449)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in complex reasoning tasks, largely enabled by test-time scaling (TTS) paradigms that allocate additional compute during inference. Among these, external TTS (particularly the Best-of-N selection paradigm) yields scalable performance improvements by selecting from multiple independently generated reasoning trajectories. However, this approach faces key limitations: (i) the high computational overhead of deploying process reward models, (ii) the underutilization of the LLM's intrinsic latent representations. We introduce TrajSelector, an efficient and effective Best-of-N framework that exploit the hidden states in the sampler LLM for process-level scoring. A lightweight verifier (with only 0.6B parameters) evaluates the quality of step-wise trajectory, and then aggregates these scores to identify the optimal reasoning trajectory. Our framework employs a fully data-driven, end-to-end training recipe that eliminates reliance on massive step-level annotations. Experiential results across five benchmarks demonstrate that TrajSelector delivers consistent performance gains. In Best-of-32 settings, it surpasses majority voting by 4.61% accuracy and outperforms existing process reward models by 4.31% to 12.21%, all while maintaining lower inference costs. 

---
# FrugalPrompt: Reducing Contextual Overhead in Large Language Models via Token Attribution 

**Authors**: Syed Rifat Raiyan, Md Farhan Ishmam, Abdullah Al Imran, Mohammad Ali Moni  

**Link**: [PDF](https://arxiv.org/pdf/2510.16439)  

**Abstract**: Large language models (LLMs) owe much of their stellar performance to expansive input contexts, yet such verbosity inflates monetary costs, carbon footprint, and inference-time latency. Much of this overhead manifests from the redundant low-utility tokens present in typical prompts, as only a fraction of tokens typically carries the majority of the semantic weight. We address this inefficiency by introducing FrugalPrompt, a novel prompt compression framework for LLMs, which retains only the most semantically significant tokens. Leveraging two state-of-the-art token attribution methods, GlobEnc and DecompX, we assign salience scores to every token in an input sequence, rank them to preserve the top-k% tokens in their original order, and obtain a sparse frugalized prompt. We evaluate the approach across four NLP tasks: Sentiment Analysis, Commonsense QA, Summarization, and Mathematical Reasoning, using a suite of frontier LLMs. For the first three tasks, a 20% prompt reduction incurs only a marginal loss in task performance, demonstrating that contemporary LLMs can reconstruct elided context from high-salience cues. In contrast, performance on mathematical reasoning deteriorates sharply, reflecting a stronger dependence on complete token continuity. Further analysis with bottom-k% and random-k% tokens reveals asymmetric performance patterns that may suggest potential task contamination effects, wherein models may resort to shallow memorized patterns from pretraining exposure for conventional NLP tasks. We posit that our work contributes to a more nuanced understanding of LLM behavior in performance-efficiency trade-offs, and delineate the boundary between tasks tolerant to contextual sparsity and those requiring exhaustive context. Our source code and models are available at: this https URL 

---
# Probing the Hidden Talent of ASR Foundation Models for L2 English Oral Assessment 

**Authors**: Fu-An Chao, Bi-Cheng Yan, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16387)  

**Abstract**: In this paper, we explore the untapped potential of Whisper, a well-established automatic speech recognition (ASR) foundation model, in the context of L2 spoken language assessment (SLA). Unlike prior studies that extrinsically analyze transcriptions produced by Whisper, our approach goes a step further to probe its latent capabilities by extracting acoustic and linguistic features from hidden representations. With only a lightweight classifier being trained on top of Whisper's intermediate and final outputs, our method achieves strong performance on the GEPT picture-description dataset, outperforming existing cutting-edge baselines, including a multimodal approach. Furthermore, by incorporating image and text-prompt information as auxiliary relevance cues, we demonstrate additional performance gains. Finally, we conduct an in-depth analysis of Whisper's embeddings, which reveals that, even without task-specific fine-tuning, the model intrinsically encodes both ordinal proficiency patterns and semantic aspects of speech, highlighting its potential as a powerful foundation for SLA and other spoken language understanding tasks. 

---
# ATA: A Neuro-Symbolic Approach to Implement Autonomous and Trustworthy Agents 

**Authors**: David Peer, Sebastian Stabinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.16381)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities, yet their deployment in high-stakes domains is hindered by inherent limitations in trustworthiness, including hallucinations, instability, and a lack of transparency. To address these challenges, we introduce a generic neuro-symbolic approach, which we call Autonomous Trustworthy Agents (ATA). The core of our approach lies in decoupling tasks into two distinct phases: Offline knowledge ingestion and online task processing. During knowledge ingestion, an LLM translates an informal problem specification into a formal, symbolic knowledge base. This formal representation is crucial as it can be verified and refined by human experts, ensuring its correctness and alignment with domain requirements. In the subsequent task processing phase, each incoming input is encoded into the same formal language. A symbolic decision engine then utilizes this encoded input in conjunction with the formal knowledge base to derive a reliable result. Through an extensive evaluation on a complex reasoning task, we demonstrate that a concrete implementation of ATA is competitive with state-of-the-art end-to-end reasoning models in a fully automated setup while maintaining trustworthiness. Crucially, with a human-verified and corrected knowledge base, our approach significantly outperforms even larger models, while exhibiting perfect determinism, enhanced stability against input perturbations, and inherent immunity to prompt injection attacks. By generating decisions grounded in symbolic reasoning, ATA offers a practical and controllable architecture for building the next generation of transparent, auditable, and reliable autonomous agents. 

---
# MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes 

**Authors**: Yu Ying Chiu, Michael S. Lee, Rachel Calcott, Brandon Handoko, Paul de Font-Reaulx, Paula Rodriguez, Chen Bo Calvin Zhang, Ziwen Han, Udari Madhushani Sehwag, Yash Maurya, Christina Q Knight, Harry R. Lloyd, Florence Bacus, Mantas Mazeika, Bing Liu, Yejin Choi, Mitchell L Gordon, Sydney Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.16380)  

**Abstract**: As AI systems progress, we rely more on them to make decisions with us and for us. To ensure that such decisions are aligned with human values, it is imperative for us to understand not only what decisions they make but also how they come to those decisions. Reasoning language models, which provide both final responses and (partially transparent) intermediate thinking traces, present a timely opportunity to study AI procedural reasoning. Unlike math and code problems which often have objectively correct answers, moral dilemmas are an excellent testbed for process-focused evaluation because they allow for multiple defensible conclusions. To do so, we present MoReBench: 1,000 moral scenarios, each paired with a set of rubric criteria that experts consider essential to include (or avoid) when reasoning about the scenarios. MoReBench contains over 23 thousand criteria including identifying moral considerations, weighing trade-offs, and giving actionable recommendations to cover cases on AI advising humans moral decisions as well as making moral decisions autonomously. Separately, we curate MoReBench-Theory: 150 examples to test whether AI can reason under five major frameworks in normative ethics. Our results show that scaling laws and existing benchmarks on math, code, and scientific reasoning tasks fail to predict models' abilities to perform moral reasoning. Models also show partiality towards specific moral frameworks (e.g., Benthamite Act Utilitarianism and Kantian Deontology), which might be side effects of popular training paradigms. Together, these benchmarks advance process-focused reasoning evaluation towards safer and more transparent AI. 

---
# Navigating through the hidden embedding space: steering LLMs to improve mental health assessment 

**Authors**: Federico Ravenda, Seyed Ali Bahrainian, Andrea Raballo, Antonietta Mira  

**Link**: [PDF](https://arxiv.org/pdf/2510.16373)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) is transforming AI, opening new opportunities in sensitive and high-impact areas such as Mental Health (MH). Yet, despite these advancements, recent evidence reveals that smaller-scale models still struggle to deliver optimal performance in domain-specific applications. In this study, we present a cost-efficient yet powerful approach to improve MH assessment capabilities of an LLM, without relying on any computationally intensive techniques. Our lightweight method consists of a linear transformation applied to a specific layer's activations, leveraging steering vectors to guide the model's output. Remarkably, this intervention enables the model to achieve improved results across two distinct tasks: (1) identifying whether a Reddit post is useful for detecting the presence or absence of depressive symptoms (relevance prediction task), and (2) completing a standardized psychological screening questionnaire for depression based on users' Reddit post history (questionnaire completion task). Results highlight the untapped potential of steering mechanisms as computationally efficient tools for LLMs' MH domain adaptation. 

---
# End-to-End Argument Mining through Autoregressive Argumentative Structure Prediction 

**Authors**: Nilmadhab Das, Vishal Vaibhav, Yash Sunil Choudhary, V. Vijaya Saradhi, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2510.16363)  

**Abstract**: Argument Mining (AM) helps in automating the extraction of complex argumentative structures such as Argument Components (ACs) like Premise, Claim etc. and Argumentative Relations (ARs) like Support, Attack etc. in an argumentative text. Due to the inherent complexity of reasoning involved with this task, modelling dependencies between ACs and ARs is challenging. Most of the recent approaches formulate this task through a generative paradigm by flattening the argumentative structures. In contrast to that, this study jointly formulates the key tasks of AM in an end-to-end fashion using Autoregressive Argumentative Structure Prediction (AASP) framework. The proposed AASP framework is based on the autoregressive structure prediction framework that has given good performance for several NLP tasks. AASP framework models the argumentative structures as constrained pre-defined sets of actions with the help of a conditional pre-trained language model. These actions build the argumentative structures step-by-step in an autoregressive manner to capture the flow of argumentative reasoning in an efficient way. Extensive experiments conducted on three standard AM benchmarks demonstrate that AASP achieves state-of-theart (SoTA) results across all AM tasks in two benchmarks and delivers strong results in one benchmark. 

---
# Utilising Large Language Models for Generating Effective Counter Arguments to Anti-Vaccine Tweets 

**Authors**: Utsav Dhanuka, Soham Poddar, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.16359)  

**Abstract**: In an era where public health is increasingly influenced by information shared on social media, combatting vaccine skepticism and misinformation has become a critical societal goal. Misleading narratives around vaccination have spread widely, creating barriers to achieving high immunisation rates and undermining trust in health recommendations. While efforts to detect misinformation have made significant progress, the generation of real time counter-arguments tailored to debunk such claims remains an insufficiently explored area. In this work, we explore the capabilities of LLMs to generate sound counter-argument rebuttals to vaccine misinformation. Building on prior research in misinformation debunking, we experiment with various prompting strategies and fine-tuning approaches to optimise counter-argument generation. Additionally, we train classifiers to categorise anti-vaccine tweets into multi-labeled categories such as concerns about vaccine efficacy, side effects, and political influences allowing for more context aware rebuttals. Our evaluation, conducted through human judgment, LLM based assessments, and automatic metrics, reveals strong alignment across these methods. Our findings demonstrate that integrating label descriptions and structured fine-tuning enhances counter-argument effectiveness, offering a promising approach for mitigating vaccine misinformation at scale. 

---
# Thinking About Thinking: Evaluating Reasoning in Post-Trained Language Models 

**Authors**: Pratham Singla, Shivank Garg, Ayush Singh, Ishan Garg, Ketan Suhaas Saichandran  

**Link**: [PDF](https://arxiv.org/pdf/2510.16340)  

**Abstract**: Recent advances in post-training techniques have endowed Large Language Models (LLMs) with enhanced capabilities for tackling complex, logic-intensive tasks through the generation of supplementary planning tokens. This development raises a fundamental question: Are these models aware of what they "learn" and "think"? To address this, we define three core competencies: (1) awareness of learned latent policies, (2) generalization of these policies across domains, and (3) alignment between internal reasoning traces and final outputs. We empirically evaluate these abilities on several tasks, each designed to require learning a distinct policy. Furthermore, we contrast the profiles of models post-trained via Supervised Fine-Tuning (SFT), Direct Policy Optimization (DPO), and Group Relative Policy Optimization (GRPO). Our findings indicate that RL-trained models not only demonstrate greater awareness of their learned behaviors and stronger generalizability to novel, structurally similar tasks than SFT models but also often exhibit weak alignment between their reasoning traces and final outputs, an effect most pronounced in GRPO-trained models. 

---
# Instant Personalized Large Language Model Adaptation via Hypernetwork 

**Authors**: Zhaoxuan Tan, Zixuan Zhang, Haoyang Wen, Zheng Li, Rongzhi Zhang, Pei Chen, Fengran Mo, Zheyuan Liu, Qingkai Zeng, Qingyu Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16282)  

**Abstract**: Personalized large language models (LLMs) tailor content to individual preferences using user profiles or histories. However, existing parameter-efficient fine-tuning (PEFT) methods, such as the ``One-PEFT-Per-User'' (OPPU) paradigm, require training a separate adapter for each user, making them computationally expensive and impractical for real-time updates. We introduce Profile-to-PEFT, a scalable framework that employs a hypernetwork, trained end-to-end, to map a user's encoded profile directly to a full set of adapter parameters (e.g., LoRA), eliminating per-user training at deployment. This design enables instant adaptation, generalization to unseen users, and privacy-preserving local deployment. Experimental results demonstrate that our method outperforms both prompt-based personalization and OPPU while using substantially fewer computational resources at deployment. The framework exhibits strong generalization to out-of-distribution users and maintains robustness across varying user activity levels and different embedding backbones. The proposed Profile-to-PEFT framework enables efficient, scalable, and adaptive LLM personalization suitable for large-scale applications. 

---
# Towards Low-Resource Alignment to Diverse Perspectives with Sparse Feedback 

**Authors**: Chu Fei Luo, Samuel Dahan, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16257)  

**Abstract**: As language models have a greater impact on society, it is important to ensure they are aligned to a diverse range of perspectives and are able to reflect nuance in human values. However, the most popular training paradigms for modern language models often assume there is one optimal answer for every query, leading to generic responses and poor alignment. In this work, we aim to enhance pluralistic alignment of language models in a low-resource setting with two methods: pluralistic decoding and model steering. We empirically demonstrate that model steering offers consistent improvement over zero-shot and few-shot baselines with only 50 annotated samples. Our proposed methods decrease false positives in several high-stakes tasks such as hate speech detection and misinformation detection, and improves the distributional alignment to human values in GlobalOpinionQA. We hope our work highlights the importance of diversity and how language models can be adapted to consider nuanced perspectives. 

---
# What Can String Probability Tell Us About Grammaticality? 

**Authors**: Jennifer Hu, Ethan Gotlieb Wilcox, Siyuan Song, Kyle Mahowald, Roger P. Levy  

**Link**: [PDF](https://arxiv.org/pdf/2510.16227)  

**Abstract**: What have language models (LMs) learned about grammar? This question remains hotly debated, with major ramifications for linguistic theory. However, since probability and grammaticality are distinct notions in linguistics, it is not obvious what string probabilities can reveal about an LM's underlying grammatical knowledge. We present a theoretical analysis of the relationship between grammar, meaning, and string probability, based on simple assumptions about the generative process of corpus data. Our framework makes three predictions, which we validate empirically using 280K sentence pairs in English and Chinese: (1) correlation between the probability of strings within minimal pairs, i.e., string pairs with minimal semantic differences; (2) correlation between models' and humans' deltas within minimal pairs; and (3) poor separation in probability space between unpaired grammatical and ungrammatical strings. Our analyses give theoretical grounding for using probability to learn about LMs' structural knowledge, and suggest directions for future work in LM grammatical evaluation. 

---
# EgMM-Corpus: A Multimodal Vision-Language Dataset for Egyptian Culture 

**Authors**: Mohamed Gamil, Abdelrahman Elsayed, Abdelrahman Lila, Ahmed Gad, Hesham Abdelgawad, Mohamed Aref, Ahmed Fares  

**Link**: [PDF](https://arxiv.org/pdf/2510.16198)  

**Abstract**: Despite recent advances in AI, multimodal culturally diverse datasets are still limited, particularly for regions in the Middle East and Africa. In this paper, we introduce EgMM-Corpus, a multimodal dataset dedicated to Egyptian culture. By designing and running a new data collection pipeline, we collected over 3,000 images, covering 313 concepts across landmarks, food, and folklore. Each entry in the dataset is manually validated for cultural authenticity and multimodal coherence. EgMM-Corpus aims to provide a reliable resource for evaluating and training vision-language models in an Egyptian cultural context. We further evaluate the zero-shot performance of Contrastive Language-Image Pre-training CLIP on EgMM-Corpus, on which it achieves 21.2% Top-1 accuracy and 36.4% Top-5 accuracy in classification. These results underscore the existing cultural bias in large-scale vision-language models and demonstrate the importance of EgMM-Corpus as a benchmark for developing culturally aware models. 

---
# In Generative AI We (Dis)Trust? Computational Analysis of Trust and Distrust in Reddit Discussions 

**Authors**: Aria Pessianzadeh, Naima Sultana, Hildegarde Van den Bulck, David Gefen, Shahin Jabari, Rezvaneh Rezapour  

**Link**: [PDF](https://arxiv.org/pdf/2510.16173)  

**Abstract**: The rise of generative AI (GenAI) has impacted many aspects of human life. As these systems become embedded in everyday practices, understanding public trust in them also becomes essential for responsible adoption and governance. Prior work on trust in AI has largely drawn from psychology and human-computer interaction, but there is a lack of computational, large-scale, and longitudinal approaches to measuring trust and distrust in GenAI and large language models (LLMs). This paper presents the first computational study of Trust and Distrust in GenAI, using a multi-year Reddit dataset (2022--2025) spanning 39 subreddits and 197,618 posts. Crowd-sourced annotations of a representative sample were combined with classification models to scale analysis. We find that Trust and Distrust are nearly balanced over time, with shifts around major model releases. Technical performance and usability dominate as dimensions, while personal experience is the most frequent reason shaping attitudes. Distinct patterns also emerge across trustors (e.g., experts, ethicists, general users). Our results provide a methodological framework for large-scale Trust analysis and insights into evolving public perceptions of GenAI. 

---
# Facts in Stats: Impacts of Pretraining Diversity on Language Model Generalization 

**Authors**: Tina Behnia, Puneesh Deora, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2510.16096)  

**Abstract**: Language models are pretrained on sequences that blend statistical regularities (making text fluent) with factual associations between specific tokens (knowledge of facts). While recent work suggests that the variability of their interaction, such as paraphrases of factual associations, critically determines generalization ability, we lack a systematic analysis of these impacts. This paper introduces a flexible synthetic testbed that combines a statistical stream of generic tokens with an abstract factual stream of source-target token pairs, enabling fine-grained control over their interaction. The design enables the independent control of diversity nature by manipulating stream composition (contextual structure) and the diversity level by varying which statistical streams each fact appears in. Through controlled experiments, we find that while higher contextual diversity delays in-distribution (ID) factual accuracy, its impact on out-of-distribution (OOD) factual generalization depends critically on contextual structure. In some cases, OOD performance follows the same trend as ID, but in others, diversity becomes essential for non-trivial factual recall. Even when low diversity prohibits factual recall, optimal diversity levels depend on training duration. Beyond factual recall failures, we identify structures where statistical generalization fails independently, and others where both capabilities degrade. This shows how the interplay between contextual design and diversity level impacts different generalization aspects. Further, through a series of controlled interventions on the model components, we trace the OOD failures to distinct optimization bottlenecks, highlighting the importance of the embedding and unembedding layers. Our synthetic framework allows us to isolate effects that would be confounded in large-scale studies, offering a controlled testbed for future investigations. 

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
# Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs 

**Authors**: Guiyao Tie, Zenghui Yuan, Zeli Zhao, Chaoran Hu, Tianhe Gu, Ruihang Zhang, Sizhe Zhang, Junran Wu, Xiaoyue Tu, Ming Jin, Qingsong Wen, Lixing Chen, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.16062)  

**Abstract**: Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: this https URL 

---
# Fusion-Augmented Large Language Models: Boosting Diagnostic Trustworthiness via Model Consensus 

**Authors**: Md Kamrul Siam, Md Jobair Hossain Faruk, Jerry Q. Cheng, Huanying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16057)  

**Abstract**: This study presents a novel multi-model fusion framework leveraging two state-of-the-art large language models (LLMs), ChatGPT and Claude, to enhance the reliability of chest X-ray interpretation on the CheXpert dataset. From the full CheXpert corpus of 224,316 chest radiographs, we randomly selected 234 radiologist-annotated studies to evaluate unimodal performance using image-only prompts. In this setting, ChatGPT and Claude achieved diagnostic accuracies of 62.8% and 76.9%, respectively. A similarity-based consensus approach, using a 95% output similarity threshold, improved accuracy to 77.6%. To assess the impact of multimodal inputs, we then generated synthetic clinical notes following the MIMIC-CXR template and evaluated a separate subset of 50 randomly selected cases paired with both images and synthetic text. On this multimodal cohort, performance improved to 84% for ChatGPT and 76% for Claude, while consensus accuracy reached 91.3%. Across both experimental conditions, agreement-based fusion consistently outperformed individual models. These findings highlight the utility of integrating complementary modalities and using output-level consensus to improve the trustworthiness and clinical utility of AI-assisted radiological diagnosis, offering a practical path to reduce diagnostic errors with minimal computational overhead. 

---
# Quantum NLP models on Natural Language Inference 

**Authors**: Ling Sun, Peter Sullivan, Michael Martin, Yun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.15972)  

**Abstract**: Quantum natural language processing (QNLP) offers a novel approach to semantic modeling by embedding compositional structure directly into quantum circuits. This paper investigates the application of QNLP models to the task of Natural Language Inference (NLI), comparing quantum, hybrid, and classical transformer-based models under a constrained few-shot setting. Using the lambeq library and the DisCoCat framework, we construct parameterized quantum circuits for sentence pairs and train them for both semantic relatedness and inference classification. To assess efficiency, we introduce a novel information-theoretic metric, Information Gain per Parameter (IGPP), which quantifies learning dynamics independent of model size. Our results demonstrate that quantum models achieve performance comparable to classical baselines while operating with dramatically fewer parameters. The Quantum-based models outperform randomly initialized transformers in inference and achieve lower test error on relatedness tasks. Moreover, quantum models exhibit significantly higher per-parameter learning efficiency (up to five orders of magnitude more than classical counterparts), highlighting the promise of QNLP in low-resource, structure-sensitive settings. To address circuit-level isolation and promote parameter sharing, we also propose a novel cluster-based architecture that improves generalization by tying gate parameters to learned word clusters rather than individual tokens. 

---
# Glyph: Scaling Context Windows via Visual-Text Compression 

**Authors**: Jiale Cheng, Yusen Liu, Xinyu Zhang, Yulin Fei, Wenyi Hong, Ruiliang Lyu, Weihan Wang, Zhe Su, Xiaotao Gu, Xiao Liu, Yushi Bai, Jie Tang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17800)  

**Abstract**: Large language models (LLMs) increasingly rely on long-context modeling for tasks such as document understanding, code analysis, and multi-step reasoning. However, scaling context windows to the million-token level brings prohibitive computational and memory costs, limiting the practicality of long-context LLMs. In this work, we take a different perspective-visual context scaling-to tackle this challenge. Instead of extending token-based sequences, we propose Glyph, a framework that renders long texts into images and processes them with vision-language models (VLMs). This approach substantially compresses textual input while preserving semantic information, and we further design an LLM-driven genetic search to identify optimal visual rendering configurations for balancing accuracy and compression. Through extensive experiments, we demonstrate that our method achieves 3-4x token compression while maintaining accuracy comparable to leading LLMs such as Qwen3-8B on various long-context benchmarks. This compression also leads to around 4x faster prefilling and decoding, and approximately 2x faster SFT training. Furthermore, under extreme compression, a 128K-context VLM could scale to handle 1M-token-level text tasks. In addition, the rendered text data benefits real-world multimodal tasks, such as document understanding. Our code and model are released at this https URL. 

---
# UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action 

**Authors**: Yuhao Yang, Zhen Yang, Zi-Yi Dou, Anh Nguyen, Keen You, Omar Attia, Andrew Szot, Michael Feng, Ram Ramrakhya, Alexander Toshev, Chao Huang, Yinfei Yang, Zhe Gan  

**Link**: [PDF](https://arxiv.org/pdf/2510.17790)  

**Abstract**: Multimodal agents for computer use rely exclusively on primitive actions (click, type, scroll) that require accurate visual grounding and lengthy execution chains, leading to cascading failures and performance bottlenecks. While other agents leverage rich programmatic interfaces (APIs, MCP servers, tools), computer-use agents (CUAs) remain isolated from these capabilities. We present UltraCUA, a foundation model that bridges this gap through hybrid action -- seamlessly integrating GUI primitives with high-level programmatic tool calls. To achieve this, our approach comprises four key components: (1) an automated pipeline that scales programmatic tools from software documentation, open-source repositories, and code generation; (2) a synthetic data engine producing over 17,000 verifiable tasks spanning real-world computer-use scenarios; (3) a large-scale high-quality hybrid action trajectory collection with both low-level GUI actions and high-level programmatic tool calls; and (4) a two-stage training pipeline combining supervised fine-tuning with online reinforcement learning, enabling strategic alternation between low-level and high-level actions. Experiments with our 7B and 32B models demonstrate substantial improvements over state-of-the-art agents. On OSWorld, UltraCUA models achieve an average 22% relative improvement over base models, while being 11% faster in terms of steps. Out-of-domain evaluation on WindowsAgentArena shows our model reaches 21.7% success rate, outperforming baselines trained on Windows data. The hybrid action mechanism proves critical, reducing error propagation while maintaining execution efficiency. 

---
# Mapping Post-Training Forgetting in Language Models at Scale 

**Authors**: Jackson Harmon, Andreas Hochlehnert, Matthias Bethge, Ameya Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17776)  

**Abstract**: Scaled post-training now drives many of the largest capability gains in language models (LMs), yet its effect on pretrained knowledge remains poorly understood. Not all forgetting is equal: Forgetting one fact (e.g., a U.S. president or an API call) does not "average out" by recalling another. Hence, we propose a sample-wise paradigm to measure what is forgotten and when backward transfer occurs. Our metric counts 1->0 transitions (correct before post-training, incorrect after) to quantify forgetting and 0->1 transitions to quantify backward transfer. Traditional task averages conflate these effects and obscure large changes. For multiple-choice benchmarks, we add chance-adjusted variants that subtract the expected contribution of random guessing from pre- and post-training accuracies. We apply this framework across post-training stages, model sizes, and data scales. Our large-scale analysis shows that: (1) Domain-continual pretraining induces moderate forgetting with low-to-moderate backward transfer; (2) RL/SFT post-training applied to base models and Instruction tuning yields moderate-to-large backward transfer on math and logic with overall low-to-moderate forgetting; (3) Applying RL/SFT to instruction-tuned models is sensitive on data scale: at small scales, both forgetting and backward transfer are small; at larger scales, effects are mixed and warrant further study with better controls; (4) Model merging does not reliably mitigate forgetting. Overall, our framework offers a practical yardstick for mapping how post-training alters pretrained knowledge at scale -- enabling progress towards generally capable AI systems. 

---
# VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models 

**Authors**: Qilin Liao, Anamika Lochab, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17759)  

**Abstract**: Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o. 

---
# Contextual Attention Modulation: Towards Efficient Multi-Task Adaptation in Large Language Models 

**Authors**: Dayan Pan, Zhaoyang Fu, Jingyuan Wang, Xiao Han, Yue Zhu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17705)  

**Abstract**: Large Language Models (LLMs) possess remarkable generalization capabilities but struggle with multi-task adaptation, particularly in balancing knowledge retention with task-specific specialization. Conventional fine-tuning methods suffer from catastrophic forgetting and substantial resource consumption, while existing parameter-efficient methods perform suboptimally in complex multi-task scenarios. To address this, we propose Contextual Attention Modulation (CAM), a novel mechanism that dynamically modulates the representations of self-attention modules in LLMs. CAM enhances task-specific features while preserving general knowledge, thereby facilitating more effective and efficient adaptation. For effective multi-task adaptation, CAM is integrated into our Hybrid Contextual Attention Modulation (HyCAM) framework, which combines a shared, full-parameter CAM module with multiple specialized, lightweight CAM modules, enhanced by a dynamic routing strategy for adaptive knowledge fusion. Extensive experiments on heterogeneous tasks, including question answering, code generation, and logical reasoning, demonstrate that our approach significantly outperforms existing approaches, achieving an average performance improvement of 3.65%. The implemented code and data are available to ease reproducibility at this https URL. 

---
# LILO: Bayesian Optimization with Interactive Natural Language Feedback 

**Authors**: Katarzyna Kobalczyk, Zhiyuan Jerry Lin, Benjamin Letham, Zhuokai Zhao, Maximilian Balandat, Eytan Bakshy  

**Link**: [PDF](https://arxiv.org/pdf/2510.17671)  

**Abstract**: For many real-world applications, feedback is essential in translating complex, nuanced, or subjective goals into quantifiable optimization objectives. We propose a language-in-the-loop framework that uses a large language model (LLM) to convert unstructured feedback in the form of natural language into scalar utilities to conduct BO over a numeric search space. Unlike preferential BO, which only accepts restricted feedback formats and requires customized models for each domain-specific problem, our approach leverages LLMs to turn varied types of textual feedback into consistent utility signals and to easily include flexible user priors without manual kernel design. At the same time, our method maintains the sample efficiency and principled uncertainty quantification of BO. We show that this hybrid method not only provides a more natural interface to the decision maker but also outperforms conventional BO baselines and LLM-only optimizers, particularly in feedback-limited regimes. 

---
# DELULU: Discriminative Embedding Learning Using Latent Units for Speaker-Aware Self-Supervised Speech Foundational Model 

**Authors**: Massa Baali, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2510.17662)  

**Abstract**: Self-supervised speech models have achieved remarkable success on content-driven tasks, yet they remain limited in capturing speaker-discriminative features critical for verification, diarization, and profiling applications. We introduce DELULU, a speaker-aware self-supervised foundational model that addresses this limitation by integrating external supervision into the pseudo-label generation process. DELULU leverages frame-level embeddings from ReDimNet, a state-of-the-art speaker verification model, to guide the k-means clustering step during pre-training, introducing a strong speaker-discriminative inductive bias that aligns representation learning with speaker identity. The model is trained using a dual objective that combines masked prediction and denoising, further enhancing robustness and generalization. DELULU significantly outperforms prior self-supervised learning (SSL) models across a range of speaker-centric tasks, achieving up to 62% relative improvement in equal error rate (EER) for speaker verification and consistent gains on zero-shot profiling tasks such as gender, age, accent, and speaker counting. Our findings demonstrate that DELULU is a strong universal encoder for speaker-aware speech processing, enabling superior performance even without task-specific fine-tuning. 

---
# LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena 

**Authors**: Qingchuan Yang, Simon Mahns, Sida Li, Anri Gu, Jibang Wu, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17638)  

**Abstract**: Forecasting is not only a fundamental intellectual pursuit but also is of significant importance to societal systems such as finance and economics. With the rapid advances of large language models (LLMs) trained on Internet-scale data, it raises the promise of employing LLMs to forecast real-world future events, an emerging paradigm we call "LLM-as-a-Prophet". This paper systematically investigates such predictive intelligence of LLMs. To this end, we build Prophet Arena, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, in order to support our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks towards achieving superior predictive intelligence via LLM-as-a-Prophet, such as LLMs' inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears. 

---
# Reasoning Distillation and Structural Alignment for Improved Code Generation 

**Authors**: Amir Jalilifard, Anderson de Rezende Rocha, Marcos Medeiros Raimundo  

**Link**: [PDF](https://arxiv.org/pdf/2510.17598)  

**Abstract**: Effective code generation with language models hinges on two critical factors: accurately understanding the intent of the prompt and generating code that applies algorithmic reasoning to produce correct solutions capable of passing diverse test cases while adhering to the syntax of the target programming language. Unlike other language tasks, code generation requires more than accurate token prediction; it demands comprehension of solution-level and structural relationships rather than merely generating the most likely tokens. very large language model (VLLM) are capable of generating detailed steps toward the correct solution of complex tasks where reasoning is crucial in solving the problem. Such reasoning capabilities may be absent in smaller language models. Therefore, in this work, we distill the reasoning capabilities of a VLLM into a smaller, more efficient model that is faster and cheaper to deploy. Our approach trains the model to emulate the reasoning and problem-solving abilities of the VLLM by learning to identify correct solution pathways and establishing a structural correspondence between problem definitions and potential solutions through a novel method of structure-aware loss optimization. This enables the model to transcend token-level generation and to deeply grasp the overarching structure of solutions for given problems. Experimental results show that our fine-tuned model, developed through a cheap and simple to implement process, significantly outperforms our baseline model in terms of pass@1, average data flow, and average syntax match metrics across the MBPP, MBPP Plus, and HumanEval benchmarks. 

---
# MIRAGE: Agentic Framework for Multimodal Misinformation Detection with Web-Grounded Reasoning 

**Authors**: Mir Nafis Sharear Shopnil, Sharad Duwal, Abhishek Tyagi, Adiba Mahbub Proma  

**Link**: [PDF](https://arxiv.org/pdf/2510.17590)  

**Abstract**: Misinformation spreads across web platforms through billions of daily multimodal posts that combine text and images, overwhelming manual fact-checking capacity. Supervised detection models require domain-specific training data and fail to generalize across diverse manipulation tactics. We present MIRAGE, an inference-time, model-pluggable agentic framework that decomposes multimodal verification into four sequential modules: visual veracity assessment detects AI-generated images, cross-modal consistency analysis identifies out-of-context repurposing, retrieval-augmented factual checking grounds claims in web evidence through iterative question generation, and a calibrated judgment module integrates all signals. MIRAGE orchestrates vision-language model reasoning with targeted web retrieval, outputs structured and citation-linked rationales. On MMFakeBench validation set (1,000 samples), MIRAGE with GPT-4o-mini achieves 81.65% F1 and 75.1% accuracy, outperforming the strongest zero-shot baseline (GPT-4V with MMD-Agent at 74.0% F1) by 7.65 points while maintaining 34.3% false positive rate versus 97.3% for a judge-only baseline. Test set results (5,000 samples) confirm generalization with 81.44% F1 and 75.08% accuracy. Ablation studies show visual verification contributes 5.18 F1 points and retrieval-augmented reasoning contributes 2.97 points. Our results demonstrate that decomposed agentic reasoning with web retrieval can match supervised detector performance without domain-specific training, enabling misinformation detection across modalities where labeled data remains scarce. 

---
# Soft-Masked Diffusion Language Models 

**Authors**: Michael Hersche, Samuel Moor-Smith, Thomas Hofmann, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2510.17206)  

**Abstract**: Diffusion models have demonstrated strong potential in language modeling, offering various advantages over traditional autoregressive approaches. Their ability to generate and revise entire responses in parallel enables faster generation and built-in self-correction mechanisms. Most modern diffusion-based language models employ masked diffusion, where decoding involves iteratively processing masked tokens based on a binary decision: either retaining the mask or replacing it with the predicted token. However, this binary choice discards valuable predictive information when the mask is retained. To address this limitation, we introduce soft-masking (SM), a novel method that dynamically blends the embedding of the mask token with the embeddings of the top-$k$ predicted tokens from the previous decoding step, for each retained mask. This provides the model with a more informative prior, preserving context from earlier computations and allowing partial information about masked tokens to propagate beyond a single step. We propose a training methodology that adapts a pretrained masked diffusion language model to incorporate SM. We demonstrate that continuing pretraining a 169M parameter model with SM leads to improved perplexity and MAUVE scores. Furthermore, we finetune two state-of-the-art diffusion models, Dream-7B and Dream-Coder-7B, with SM. SM consistently improves performance across multiple coding benchmarks, particularly in high-throughput settings. 

---
# $\mathcal{V}isi\mathcal{P}runer$: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs 

**Authors**: Yingqi Fan, Anhao Zhao, Jinlan Fu, Junlong Tong, Hui Su, Yijie Pan, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.17205)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved strong performance across vision-language tasks, but suffer from significant computational overhead due to the quadratic growth of attention computations with the number of multimodal tokens. Though efforts have been made to prune tokens in MLLMs, \textit{they lack a fundamental understanding of how MLLMs process and fuse multimodal information.} Through systematic analysis, we uncover a \textbf{three-stage} cross-modal interaction process: (1) Shallow layers recognize task intent, with visual tokens acting as passive attention sinks; (2) Cross-modal fusion occurs abruptly in middle layers, driven by a few critical visual tokens; (3) Deep layers discard vision tokens, focusing solely on linguistic refinement. Based on these findings, we propose \emph{VisiPruner}, a training-free pruning framework that reduces up to 99\% of vision-related attention computations and 53.9\% of FLOPs on LLaVA-v1.5 7B. It significantly outperforms existing token pruning methods and generalizes across diverse MLLMs. Beyond pruning, our insights further provide actionable guidelines for training efficient MLLMs by aligning model architecture with its intrinsic layer-wise processing dynamics. Our code is available at: this https URL. 

---
# Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users 

**Authors**: Melik Ozolcer, Sang Won Bae  

**Link**: [PDF](https://arxiv.org/pdf/2510.17173)  

**Abstract**: We study a web-deployed, tool-augmented LLM health coach with real users. In a pilot with seven users (280 rated turns), offline policy evaluation (OPE) over factorized decision heads (Tool/Style) shows that a uniform heavy-tool policy raises average value on logs but harms specific subgroups, most notably low-health-literacy/high-self-efficacy users. A lightweight simulator with hidden archetypes further shows that adding a small early information-gain bonus reliably shortens trait identification and improves goal success and pass@3. Together, these early findings indicate an evaluation-first path to personalization: freeze the generator, learn subgroup-aware decision heads on typed rewards (objective tool outcomes and satisfaction), and always report per-archetype metrics to surface subgroup harms that averages obscure. 

---
# Do LLMs Recognize Your Latent Preferences? A Benchmark for Latent Information Discovery in Personalized Interaction 

**Authors**: Ioannis Tsaknakis, Bingqing Song, Shuyu Gan, Dongyeop Kang, Alfredo Garcia, Gaowen Liu, Charles Fleming, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.17132)  

**Abstract**: Large Language Models (LLMs) excel at producing broadly relevant text, but this generality becomes a limitation when user-specific preferences are required, such as recommending restaurants or planning travel. In these scenarios, users rarely articulate every preference explicitly; instead, much of what they care about remains latent, waiting to be inferred. This raises a fundamental question: Can LLMs uncover and reason about such latent information through conversation?
We address this problem by introducing a unified benchmark for evaluating latent information discovery - the ability of LLMs to reveal and utilize hidden user attributes through multi-turn interaction. The benchmark spans three progressively realistic settings: the classic 20 Questions game, Personalized Question Answering, and Personalized Text Summarization. All tasks share a tri-agent framework (User, Assistant, Judge) enabling turn-level evaluation of elicitation and adaptation. Our results reveal that while LLMs can indeed surface latent information through dialogue, their success varies dramatically with context: from 32% to 98%, depending on task complexity, topic, and number of hidden attributes. This benchmark provides the first systematic framework for studying latent information discovery in personalized interaction, highlighting that effective preference inference remains an open frontier for building truly adaptive AI systems. 

---
# Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning 

**Authors**: Bingqi Shang, Yiwei Chen, Yihua Zhang, Bingquan Shen, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17021)  

**Abstract**: Large language model (LLM) unlearning has become a critical mechanism for removing undesired data, knowledge, or behaviors from pre-trained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdoor unlearning, where models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between backdoor efficacy and the attention sink phenomenon, i.e., shallow input tokens consistently attract disproportionate attention in LLMs. Our analysis reveals that these attention sinks serve as gateways for backdoor unlearning: placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence. Extensive experiments validate these findings, showing that attention-sink-guided backdoor unlearning reliably restores forgotten knowledge in the presence of backdoor triggers, while behaving indistinguishably from a normally unlearned model when triggers are absent. Code is available at this https URL. 

---
# Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks against LLMs 

**Authors**: Masahiro Kaneko, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17000)  

**Abstract**: Adversarial attacks by malicious users that threaten the safety of large language models (LLMs) can be viewed as attempts to infer a target property $T$ that is unknown when an instruction is issued, and becomes knowable only after the model's reply is observed. Examples of target properties $T$ include the binary flag that triggers an LLM's harmful response or rejection, and the degree to which information deleted by unlearning can be restored, both elicited via adversarial instructions. The LLM reveals an \emph{observable signal} $Z$ that potentially leaks hints for attacking through a response containing answer tokens, thinking process tokens, or logits. Yet the scale of information leaked remains anecdotal, leaving auditors without principled guidance and defenders blind to the transparency--risk trade-off. We fill this gap with an information-theoretic framework that computes how much information can be safely disclosed, and enables auditors to gauge how close their methods come to the fundamental limit. Treating the mutual information $I(Z;T)$ between the observation $Z$ and the target property $T$ as the leaked bits per query, we show that achieving error $\varepsilon$ requires at least $\log(1/\varepsilon)/I(Z;T)$ queries, scaling linearly with the inverse leak rate and only logarithmically with the desired accuracy. Thus, even a modest increase in disclosure collapses the attack cost from quadratic to logarithmic in terms of the desired accuracy. Experiments on seven LLMs across system-prompt leakage, jailbreak, and relearning attacks corroborate the theory: exposing answer tokens alone requires about a thousand queries; adding logits cuts this to about a hundred; and revealing the full thinking process trims it to a few dozen. Our results provide the first principled yardstick for balancing transparency and security when deploying LLMs. 

---
# Leave It to the Experts: Detecting Knowledge Distillation via MoE Expert Signatures 

**Authors**: Pingzhi Li, Morris Yu-Chao Huang, Zhen Tan, Qingquan Song, Jie Peng, Kai Zou, Yu Cheng, Kaidi Xu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.16968)  

**Abstract**: Knowledge Distillation (KD) accelerates training of large language models (LLMs) but poses intellectual property protection and LLM diversity risks. Existing KD detection methods based on self-identity or output similarity can be easily evaded through prompt engineering. We present a KD detection framework effective in both white-box and black-box settings by exploiting an overlooked signal: the transfer of MoE "structural habits", especially internal routing patterns. Our approach analyzes how different experts specialize and collaborate across various inputs, creating distinctive fingerprints that persist through the distillation process. To extend beyond the white-box setup and MoE architectures, we further propose Shadow-MoE, a black-box method that constructs proxy MoE representations via auxiliary distillation to compare these patterns between arbitrary model pairs. We establish a comprehensive, reproducible benchmark that offers diverse distilled checkpoints and an extensible framework to facilitate future research. Extensive experiments demonstrate >94% detection accuracy across various scenarios and strong robustness to prompt-based evasion, outperforming existing baselines while highlighting the structural habits transfer in LLMs. 

---
# Real-Time World Crafting: Generating Structured Game Behaviors from Natural Language with Large Language Models 

**Authors**: Austin Drake, Hang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.16952)  

**Abstract**: We present a novel architecture for safely integrating Large Language Models (LLMs) into interactive game engines, allowing players to "program" new behaviors using natural language. Our framework mitigates risks by using an LLM to translate commands into a constrained Domain-Specific Language (DSL), which configures a custom Entity-Component-System (ECS) at runtime. We evaluated this system in a 2D spell-crafting game prototype by experimentally assessing models from the Gemini, GPT, and Claude families with various prompting strategies. A validated LLM judge qualitatively rated the outputs, showing that while larger models better captured creative intent, the optimal prompting strategy is task-dependent: Chain-of-Thought improved creative alignment, while few-shot examples were necessary to generate more complex DSL scripts. This work offers a validated LLM-ECS pattern for emergent gameplay and a quantitative performance comparison for developers. 

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
# SAKE: Towards Editing Auditory Attribute Knowledge of Large Audio-Language Models 

**Authors**: Chih-Kai Yang, Yen-Ting Piao, Tzu-Wen Hsu, Szu-Wei Fu, Zhehuai Chen, Ke-Han Lu, Sung-Feng Huang, Chao-Han Huck Yang, Yu-Chiang Frank Wang, Yun-Nung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.16917)  

**Abstract**: Knowledge editing offers an efficient way to update model knowledge without full retraining, but prior work has concentrated almost exclusively on textual or visual modalities. We introduce SAKE, the first benchmark specifically designed for editing auditory attribute knowledge in Large Audio-Language Models (LALMs). Unlike factual updates, SAKE targets several abstract auditory attributes, capturing knowledge types that go beyond conventional textual and visual domains. We benchmark seven editing methods on two LALMs along four dimensions: reliability, generality, audio/text locality, and portability. Results highlight challenges such as preserving intra-attribute knowledge unrelated to the edit, generalizing edits to multimodal reasoning, and maintaining edits under sequential updates. SAKE provides a principled framework to study how knowledge editing extends to the auditory modalities, opening new directions for maintaining and adapting LALMs in more diverse real-world scenarios. 

---
# VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents 

**Authors**: Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Yejin Choi, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.16907)  

**Abstract**: A key challenge in training Vision-Language Model (VLM) agents, compared to Language Model (LLM) agents, lies in the shift from textual states to complex visual observations. This transition introduces partial observability and demands robust world modeling. We ask: Can VLM agents construct internal world models through explicit visual state reasoning? To address this question, we architecturally enforce and reward the agent's reasoning process via reinforcement learning (RL), formulating it as a Partially Observable Markov Decision Process (POMDP). We find that decomposing the agent's reasoning into State Estimation ("what is the current state?") and Transition Modeling ("what comes next?") is critical for success, as demonstrated through five reasoning strategies. Our investigation into how agents represent internal beliefs reveals that the optimal representation is task-dependent: Natural Language excels at capturing semantic relationships in general tasks, while Structured formats are indispensable for precise manipulation and control. Building on these insights, we design a World Modeling Reward that provides dense, turn-level supervision for accurate state prediction, and introduce Bi-Level General Advantage Estimation (Bi-Level GAE) for turn-aware credit assignment. Through this form of visual state reasoning, a 3B-parameter model achieves a score of 0.82 across five diverse agent benchmarks, representing a 3$\times$ improvement over its untrained counterpart (0.21) and outperforming proprietary reasoning models such as GPT-5 (0.75), Gemini 2.5 Pro (0.67) and Claude 4.5 (0.62). All experiments are conducted within our VAGEN framework, a scalable system for training and analyzing multi-turn VLM agents in diverse visual environments. Code and data are publicly available at this https URL. 

---
# Investigating Safety Vulnerabilities of Large Audio-Language Models Under Speaker Emotional Variations 

**Authors**: Bo-Han Feng, Chien-Feng Liu, Yu-Hsuan Li Liang, Chih-Kai Yang, Szu-Wei Fu, Zhehuai Chen, Ke-Han Lu, Sung-Feng Huang, Chao-Han Huck Yang, Yu-Chiang Frank Wang, Yun-Nung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.16893)  

**Abstract**: Large audio-language models (LALMs) extend text-based LLMs with auditory understanding, offering new opportunities for multimodal applications. While their perception, reasoning, and task performance have been widely studied, their safety alignment under paralinguistic variation remains underexplored. This work systematically investigates the role of speaker emotion. We construct a dataset of malicious speech instructions expressed across multiple emotions and intensities, and evaluate several state-of-the-art LALMs. Our results reveal substantial safety inconsistencies: different emotions elicit varying levels of unsafe responses, and the effect of intensity is non-monotonic, with medium expressions often posing the greatest risk. These findings highlight an overlooked vulnerability in LALMs and call for alignment strategies explicitly designed to ensure robustness under emotional variation, a prerequisite for trustworthy deployment in real-world settings. 

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
# Verifiable Fine-Tuning for LLMs: Zero-Knowledge Training Proofs Bound to Data Provenance and Policy 

**Authors**: Hasan Akgul, Daniel Borg, Arta Berisha, Amina Rahimova, Andrej Novak, Mila Petrov  

**Link**: [PDF](https://arxiv.org/pdf/2510.16830)  

**Abstract**: Large language models are often adapted through parameter efficient fine tuning, but current release practices provide weak assurances about what data were used and how updates were computed. We present Verifiable Fine Tuning, a protocol and system that produces succinct zero knowledge proofs that a released model was obtained from a public initialization under a declared training program and an auditable dataset commitment. The approach combines five elements. First, commitments that bind data sources, preprocessing, licenses, and per epoch quota counters to a manifest. Second, a verifiable sampler that supports public replayable and private index hiding batch selection. Third, update circuits restricted to parameter efficient fine tuning that enforce AdamW style optimizer semantics and proof friendly approximations with explicit error budgets. Fourth, recursive aggregation that folds per step proofs into per epoch and end to end certificates with millisecond verification. Fifth, provenance binding and optional trusted execution property cards that attest code identity and constants. On English and bilingual instruction mixtures, the method maintains utility within tight budgets while achieving practical proof performance. Policy quotas are enforced with zero violations, and private sampling windows show no measurable index leakage. Federated experiments demonstrate that the system composes with probabilistic audits and bandwidth constraints. These results indicate that end to end verifiable fine tuning is feasible today for real parameter efficient pipelines, closing a critical trust gap for regulated and decentralized deployments. 

---
# When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation 

**Authors**: Amirkia Rafiei Oskooei, Kaan Baturalp Cosdan, Husamettin Isiktas, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2510.16809)  

**Abstract**: Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering. 

---
# Xiaoice: Training-Free Video Understanding via Self-Supervised Spatio-Temporal Clustering of Semantic Features 

**Authors**: Shihao Ji, Zihui Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.16781)  

**Abstract**: The remarkable zero-shot reasoning capabilities of large-scale Visual Language Models (VLMs) on static images have yet to be fully translated to the video domain. Conventional video understanding models often rely on extensive, task-specific training on annotated datasets, a process that is both costly and limited in scalability. This paper introduces a novel, training-free framework for video understanding that circumvents end-to-end training by synergistically combining the rich semantic priors of pre-trained VLMs with classic machine learning algorithms for pattern discovery. Our core idea is to reframe video understanding as a self-supervised spatio-temporal clustering problem within a high-dimensional semantic feature space. The proposed pipeline first transforms a video stream into a semantic feature trajectory using the frozen visual encoder of a pre-trained VLM. Subsequently, we employ Kernel Temporal Segmentation (KTS), a robust machine learning technique, to partition the continuous feature stream into discrete, semantically coherent event segments. These segments are then subjected to unsupervised density-based clustering to identify recurring macroscopic scenes and themes throughout the video. By selecting representative keyframes from each discovered cluster and leveraging the VLM's generative capabilities for textual description, our framework automatically produces a structured, multi-modal summary of the video content. This approach provides an effective, interpretable, and model-agnostic pathway for zero-shot, automated structural analysis of video content. 

---
# See or Say Graphs: Agent-Driven Scalable Graph Understanding with Vision-Language Models 

**Authors**: Shuo Han, Yukun Cao, Zezhong Ding, Zengyi Gao, S Kevin Zhou, Xike Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.16769)  

**Abstract**: Vision-language models (VLMs) have shown promise in graph understanding, but remain limited by input-token constraints, facing scalability bottlenecks and lacking effective mechanisms to coordinate textual and visual modalities. To address these challenges, we propose GraphVista, a unified framework that enhances both scalability and modality coordination in graph understanding. For scalability, GraphVista organizes graph information hierarchically into a lightweight GraphRAG base, which retrieves only task-relevant textual descriptions and high-resolution visual subgraphs, compressing redundant context while preserving key reasoning elements. For modality coordination, GraphVista introduces a planning agent that routes tasks to the most suitable modality-using the text modality for simple property reasoning and the visual modality for local and structurally complex reasoning grounded in explicit topology. Extensive experiments demonstrate that GraphVista scales to large graphs, up to $200\times$ larger than those used in existing benchmarks, and consistently outperforms existing textual, visual, and fusion-based methods, achieving up to $4.4\times$ quality improvement over the state-of-the-art baselines by fully exploiting the complementary strengths of both modalities. 

---
# End-to-end Listen, Look, Speak and Act 

**Authors**: Siyin Wang, Wenyi Yu, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16756)  

**Abstract**: Human interaction is inherently multimodal and full-duplex: we listen while watching, speak while acting, and fluidly adapt to turn-taking and interruptions. Realizing these capabilities is essential for building models simulating humans. We present ELLSA (End-to-end Listen, Look, Speak and Act), which, to our knowledge, is the first full-duplex, end-to-end model that simultaneously perceives and generates across vision, text, speech, and action within a single architecture, enabling interaction patterns previously out of reach, yielding more natural, human-like behaviors. At its core is a novel SA-MoE architecture (Self-Attention Mixture-of-Experts) that routes each modality to specialized experts and fuses them through a unified attention backbone. This provides a generalizable solution for joint multimodal perception and concurrent generation, leveraging strong pre-trained components while enabling efficient modality integration and mitigating modality interference. On speech-interaction and robot-manipulation benchmarks, ELLSA matches modality-specific baselines, while uniquely supporting advanced multimodal and full-duplex behaviors such as dialogue and action turn-taking, defective instruction rejection, speaking-while-acting, context-grounded visual question answering, and action barge-ins. We contend that ELLSA represents a step toward more natural and general interactive intelligence, contributing to the broader pursuit of artificial general intelligence. All data, code and model checkpoints will be released upon acceptance. 

---
# Zero-Shot Performance Prediction for Probabilistic Scaling Laws 

**Authors**: Viktoria Schram, Markus Hiller, Daniel Beck, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2510.16743)  

**Abstract**: The prediction of learning curves for Natural Language Processing (NLP) models enables informed decision-making to meet specific performance objectives, while reducing computational overhead and lowering the costs associated with dataset acquisition and curation. In this work, we formulate the prediction task as a multitask learning problem, where each task's data is modelled as being organized within a two-layer hierarchy. To model the shared information and dependencies across tasks and hierarchical levels, we employ latent variable multi-output Gaussian Processes, enabling to account for task correlations and supporting zero-shot prediction of learning curves (LCs). We demonstrate that this approach facilitates the development of probabilistic scaling laws at lower costs. Applying an active learning strategy, LCs can be queried to reduce predictive uncertainty and provide predictions close to ground truth scaling laws. We validate our framework on three small-scale NLP datasets with up to $30$ LCs. These are obtained from nanoGPT models, from bilingual translation using mBART and Transformer models, and from multilingual translation using M2M100 models of varying sizes. 

---
# A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications 

**Authors**: Minhua Lin, Zongyu Wu, Zhichao Xu, Hui Liu, Xianfeng Tang, Qi He, Charu Aggarwal, Hui Liu, Xiang Zhang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16724)  

**Abstract**: The advent of large language models (LLMs) has transformed information access and reasoning through open-ended natural language interaction. However, LLMs remain limited by static knowledge, factual hallucinations, and the inability to retrieve real-time or domain-specific information. Retrieval-Augmented Generation (RAG) mitigates these issues by grounding model outputs in external evidence, but traditional RAG pipelines are often single turn and heuristic, lacking adaptive control over retrieval and reasoning. Recent advances in agentic search address these limitations by enabling LLMs to plan, retrieve, and reflect through multi-step interaction with search environments. Within this paradigm, reinforcement learning (RL) offers a powerful mechanism for adaptive and self-improving search behavior. This survey provides the first comprehensive overview of \emph{RL-based agentic search}, organizing the emerging field along three complementary dimensions: (i) What RL is for (functional roles), (ii) How RL is used (optimization strategies), and (iii) Where RL is applied (scope of optimization). We summarize representative methods, evaluation protocols, and applications, and discuss open challenges and future directions toward building reliable and scalable RL driven agentic search systems. We hope this survey will inspire future research on the integration of RL and agentic search. Our repository is available at this https URL. 

---
# U-Codec: Ultra Low Frame-rate Neural Speech Codec for Fast High-fidelity Speech Generation 

**Authors**: Xusheng Yang, Long Zhou, Wenfu Wang, Kai Hu, Shulin Feng, Chenxing Li, Meng Yu, Dong Yu, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.16718)  

**Abstract**: We propose \textbf{U-Codec}, an \textbf{U}ltra low frame-rate neural speech \textbf{Codec} that achieves high-fidelity reconstruction and fast speech generation at an extremely low frame-rate of 5Hz (5 frames per second). Extreme compression at 5Hz typically leads to severe intelligibility and spectral detail loss, we introduce a Transformer-based inter-frame long-term dependency module and systematically explore residual vector quantization (RVQ) depth and codebook size to identify optimal configurations. Moreover, we apply U-Codec into a large language model (LLM)-based auto-regressive TTS model, which leverages global and local hierarchical architecture to effectively capture dependencies across multi-layer tokens. We extend LLM-based TTS from 3-layer RVQ at 50Hz to 32-layer RVQ at 5Hz. Experimental results demonstrate that U-Codec improves LLM-based TTS inference speed by around 3 $\times$ over high-frame-rate codecs while maintaining similarity and naturalness. These results validate the feasibility of using highly compressed 5Hz discrete tokens for fast and high-fidelity speech synthesis. 

---
# Prompt Optimization via Retrieved Reasoning Assets and Multi-Agent Analysis 

**Authors**: Wonduk Seo, Juhyeon Lee, Junseo Koh, Hyunjin An, Jian Park, Seunghyun Lee, Haihua Chen, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16635)  

**Abstract**: Prompt optimization has emerged as an effective alternative to retraining for improving the performance of Large Language Models (LLMs). However, most existing approaches treat evaluation as a black box, relying solely on numerical scores while offering limited insight into why a prompt succeeds or fails. They also depend heavily on trial-and-error refinements, which are difficult to interpret and control. In this paper, we introduce MA-SAPO, a Multi-Agent framework for Score-Aware Prompt Optimization. Compared to prior methods, MA-SAPO explicitly couples evaluation outcomes with structured reasoning to guide systematic edits. The framework specifically consists of two stages: during the Reasoning Phase, agents collaboratively explain metric scores, diagnose weaknesses, and synthesize targeted refinements that are stored as reusable reasoning assets; during the Test Phase, agents retrieve these assets to analyze optimized prompts and apply only evidence-grounded edits. By turning evaluation signals into interpretable reasoning chains, MA-SAPO produces prompt refinements that are more transparent, auditable, and controllable. Experiments on the HelpSteer1/2 benchmarks demonstrate consistent improvements over single-pass prompting, retrieval-augmented baselines, and prior multi-agent strategies, validating the effectiveness of our approach. 

---
# Copy-Augmented Representation for Structure Invariant Template-Free Retrosynthesis 

**Authors**: Jiaxi Zhuang, Yu Zhang, Aimin Zhou, Ying Qian  

**Link**: [PDF](https://arxiv.org/pdf/2510.16588)  

**Abstract**: Retrosynthesis prediction is fundamental to drug discovery and chemical synthesis, requiring the identification of reactants that can produce a target molecule. Current template-free methods struggle to capture the structural invariance inherent in chemical reactions, where substantial molecular scaffolds remain unchanged, leading to unnecessarily large search spaces and reduced prediction accuracy. We introduce C-SMILES, a novel molecular representation that decomposes traditional SMILES into element-token pairs with five special tokens, effectively minimizing editing distance between reactants and products. Building upon this representation, we incorporate a copy-augmented mechanism that dynamically determines whether to generate new tokens or preserve unchanged molecular fragments from the product. Our approach integrates SMILES alignment guidance to enhance attention consistency with ground-truth atom mappings, enabling more chemically coherent predictions. Comprehensive evaluation on USPTO-50K and large-scale USPTO-FULL datasets demonstrates significant improvements: 67.2% top-1 accuracy on USPTO-50K and 50.8% on USPTO-FULL, with 99.9% validity in generated molecules. This work establishes a new paradigm for structure-aware molecular generation with direct applications in computational drug discovery. 

---
# What Questions Should Robots Be Able to Answer? A Dataset of User Questions for Explainable Robotics 

**Authors**: Lennart Wachowiak, Andrew Coles, Gerard Canal, Oya Celiktutan  

**Link**: [PDF](https://arxiv.org/pdf/2510.16435)  

**Abstract**: With the growing use of large language models and conversational interfaces in human-robot interaction, robots' ability to answer user questions is more important than ever. We therefore introduce a dataset of 1,893 user questions for household robots, collected from 100 participants and organized into 12 categories and 70 subcategories. Most work in explainable robotics focuses on why-questions. In contrast, our dataset provides a wide variety of questions, from questions about simple execution details to questions about how the robot would act in hypothetical scenarios -- thus giving roboticists valuable insights into what questions their robot needs to be able to answer. To collect the dataset, we created 15 video stimuli and 7 text stimuli, depicting robots performing varied household tasks. We then asked participants on Prolific what questions they would want to ask the robot in each portrayed situation. In the final dataset, the most frequent categories are questions about task execution details (22.5%), the robot's capabilities (12.7%), and performance assessments (11.3%). Although questions about how robots would handle potentially difficult scenarios and ensure correct behavior are less frequent, users rank them as the most important for robots to be able to answer. Moreover, we find that users who identify as novices in robotics ask different questions than more experienced users. Novices are more likely to inquire about simple facts, such as what the robot did or the current state of the environment. As robots enter environments shared with humans and language becomes central to giving instructions and interaction, this dataset provides a valuable foundation for (i) identifying the information robots need to log and expose to conversational interfaces, (ii) benchmarking question-answering modules, and (iii) designing explanation strategies that align with user expectations. 

---
# Investigating the Association Between Text-Based Indications of Foodborne Illness from Yelp Reviews and New York City Health Inspection Outcomes (2023) 

**Authors**: Eden Shaveet, Crystal Su, Daniel Hsu, Luis Gravano  

**Link**: [PDF](https://arxiv.org/pdf/2510.16334)  

**Abstract**: Foodborne illnesses are gastrointestinal conditions caused by consuming contaminated food. Restaurants are critical venues to investigate outbreaks because they share sourcing, preparation, and distribution of foods. Public reporting of illness via formal channels is limited, whereas social media platforms host abundant user-generated content that can provide timely public health signals. This paper analyzes signals from Yelp reviews produced by a Hierarchical Sigmoid Attention Network (HSAN) classifier and compares them with official restaurant inspection outcomes issued by the New York City Department of Health and Mental Hygiene (NYC DOHMH) in 2023. We evaluate correlations at the Census tract level, compare distributions of HSAN scores by prevalence of C-graded restaurants, and map spatial patterns across NYC. We find minimal correlation between HSAN signals and inspection scores at the tract level and no significant differences by number of C-graded restaurants. We discuss implications and outline next steps toward address-level analyses. 

---
# Cerberus: Real-Time Video Anomaly Detection via Cascaded Vision-Language Models 

**Authors**: Yue Zheng, Xiufang Shi, Jiming Chen, Yuanchao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16290)  

**Abstract**: Video anomaly detection (VAD) has rapidly advanced by recent development of Vision-Language Models (VLMs). While these models offer superior zero-shot detection capabilities, their immense computational cost and unstable visual grounding performance hinder real-time deployment. To overcome these challenges, we introduce Cerberus, a two-stage cascaded system designed for efficient yet accurate real-time VAD. Cerberus learns normal behavioral rules offline, and combines lightweight filtering with fine-grained VLM reasoning during online inference. The performance gains of Cerberus come from two key innovations: motion mask prompting and rule-based deviation detection. The former directs the VLM's attention to regions relevant to motion, while the latter identifies anomalies as deviations from learned norms rather than enumerating possible anomalies. Extensive evaluations on four datasets show that Cerberus on average achieves 57.68 fps on an NVIDIA L40S GPU, a 151.79$\times$ speedup, and 97.2\% accuracy comparable to the state-of-the-art VLM-based VAD methods, establishing it as a practical solution for real-time video analytics. 

---
# WEBSERV: A Browser-Server Environment for Efficient Training of Reinforcement Learning-based Web Agents at Scale 

**Authors**: Yuxuan Lu, Jing Huang, Hui Liu, Jiri Gesi, Yan Han, Shihan Fu, Tianqi Zheng, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.16252)  

**Abstract**: Training and evaluation of Reinforcement Learning (RL) web agents have gained increasing attention, yet a scalable and efficient environment that couples realistic and robust browser-side interaction with controllable server-side state at scale is still missing. Existing environments tend to have one or more of the following issues: they overwhelm policy models with excessive and noisy context; they perform actions non-deterministically without waiting for the UI or network to stabilize; or they cannot scale isolated client-server containers effectively for parallel RL rollouts. We propose WEBSERV, an environment that includes 1) a compact, site-agnostic browser environment that balances context and action complexity, and 2) a scalable RL environment via efficient launching and resetting web-servers to enable scalable RL training and evaluation. We evaluate WEBSERV on the shopping CMS and Gitlab tasks in WebArena, achieving state-of-the-art single-prompt success rates while cutting launch latency by ~5x and storage need by ~240x, with a comparable memory footprint, enabling 200+ concurrent containers on a single host. 

---
# ScholarEval: Research Idea Evaluation Grounded in Literature 

**Authors**: Hanane Nour Moussa, Patrick Queiroz Da Silva, Daniel Adu-Ampratwum, Alyson East, Zitong Lu, Nikki Puccetti, Mingyi Xue, Huan Sun, Bodhisattwa Prasad Majumder, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.16234)  

**Abstract**: As AI tools become increasingly common for research ideation, robust evaluation is critical to ensure the validity and usefulness of generated ideas. We introduce ScholarEval, a retrieval augmented evaluation framework that assesses research ideas based on two fundamental criteria: soundness - the empirical validity of proposed methods based on existing literature, and contribution - the degree of advancement made by the idea across different dimensions relative to prior research. To evaluate ScholarEval, we introduce ScholarIdeas, the first expert-annotated dataset of multi-domain research ideas and reviews, comprised of 117 ideas across four disciplines: artificial intelligence, neuroscience, biochemistry, and ecology. Our evaluation shows that ScholarEval achieves significantly higher coverage of points mentioned in the human expert annotated rubrics in ScholarIdeas compared to all baselines. Furthermore, ScholarEval is consistently preferred over our strongest baseline o4-mini-deep-research, a reasoning and search-enabled agentic system by OpenAI, in terms of evaluation actionability, depth, and evidence support. Our large-scale user study also shows that ScholarEval significantly outperforms deep research in literature engagement, idea refinement, and usefulness. We openly release our code, dataset, and ScholarEval tool for the community to use and build on. 

---
# Alignment is Localized: A Causal Probe into Preference Layers 

**Authors**: Archie Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2510.16167)  

**Abstract**: Reinforcement Learning frameworks, particularly those utilizing human annotations, have become an increasingly popular method for preference fine-tuning, where the outputs of a language model are tuned to match a certain set of behavioral policies or guidelines. Reinforcement Learning through Human Feedback (RLHF) is perhaps the most popular implementation of such a framework, particularly for aligning LMs toward safety and human intent. However, the internal workings of how such alignment is achieved remain largely opaque. In this work, we systematically analyze preference optimization for language model alignment by applying layer-wide causal patching between a base model and its tuned counterpart across human preference pairs. We implement our methodology on \textit{Llama-3.2-1B}, and find that alignment is spatially localized: mid-layer activations encode a distinct subspace that causally determines reward-consistent behavior, while early and late layers remain largely unaffected. Utilizing LASSO regression, we also find that only a small number of layers possess non-zero coefficients linking activation distances to reward gains. Overall, we show that, at least for some language models, alignment from human-based, preferential tuning is a directional, low rank process, rather than diffuse and parameteric. 

---
# Zeroth-Order Sharpness-Aware Learning with Exponential Tilting 

**Authors**: Xuchen Gong, Tian Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.16157)  

**Abstract**: Classic zeroth-order optimization approaches typically optimize for a smoothed version of the original function, i.e., the expected objective under randomly perturbed model parameters. This can be interpreted as encouraging the loss values in the perturbation set to be small on average. Popular sharpness-aware minimization (SAM) objectives, however, typically focus on the largest loss within the neighborhood to arrive at flat minima more effectively. In this work, we connect zeroth-order optimization (and its corresponding objectives) with SAM approaches explicitly, through an exponential tilting objective that provides a smooth transition between the average- and the max-loss formulations. We explore new zeroth-order algorithms to solve a soft SAM objective parameterized by a tilting parameter $t$. We provide precise characterizations of the sharpness notions of the tilted SAM framework. Practically, our approach can be used as a gradient-free and memory-efficient alternative to SAM variants, and it achieves better generalization compared to vanilla zeroth-order baselines on a wide range of downstream tasks, including classification, multiple choice QA, and language generation. 

---
# Publication Trend Analysis and Synthesis via Large Language Model: A Case Study of Engineering in PNAS 

**Authors**: Mason Smetana, Lev Khazanovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.16152)  

**Abstract**: Scientific literature is increasingly siloed by complex language, static disciplinary structures, and potentially sparse keyword systems, making it cumbersome to capture the dynamic nature of modern science. This study addresses these challenges by introducing an adaptable large language model (LLM)-driven framework to quantify thematic trends and map the evolving landscape of scientific knowledge. The approach is demonstrated over a 20-year collection of more than 1,500 engineering articles published by the Proceedings of the National Academy of Sciences (PNAS), marked for their breadth and depth of research focus. A two-stage classification pipeline first establishes a primary thematic category for each article based on its abstract. The subsequent phase performs a full-text analysis to assign secondary classifications, revealing latent, cross-topic connections across the corpus. Traditional natural language processing (NLP) methods, such as Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF), confirm the resulting topical structure and also suggest that standalone word-frequency analyses may be insufficient for mapping fields with high diversity. Finally, a disjoint graph representation between the primary and secondary classifications reveals implicit connections between themes that may be less apparent when analyzing abstracts or keywords alone. The findings show that the approach independently recovers much of the journal's editorially embedded structure without prior knowledge of its existing dual-classification schema (e.g., biological studies also classified as engineering). This framework offers a powerful tool for detecting potential thematic trends and providing a high-level overview of scientific progress. 

---
# The Hidden Cost of Modeling P(X): Vulnerability to Membership Inference Attacks in Generative Text Classifiers 

**Authors**: Owais Makroo, Siva Rajesh Kasa, Sumegh Roychowdhury, Karan Gupta, Nikhil Pattisapu, Santhosh Kasa, Sumit Negi  

**Link**: [PDF](https://arxiv.org/pdf/2510.16122)  

**Abstract**: Membership Inference Attacks (MIAs) pose a critical privacy threat by enabling adversaries to determine whether a specific sample was included in a model's training dataset. Despite extensive research on MIAs, systematic comparisons between generative and discriminative classifiers remain limited. This work addresses this gap by first providing theoretical motivation for why generative classifiers exhibit heightened susceptibility to MIAs, then validating these insights through comprehensive empirical evaluation. Our study encompasses discriminative, generative, and pseudo-generative text classifiers across varying training data volumes, evaluated on nine benchmark datasets. Employing a diverse array of MIA strategies, we consistently demonstrate that fully generative classifiers which explicitly model the joint likelihood $P(X,Y)$ are most vulnerable to membership leakage. Furthermore, we observe that the canonical inference approach commonly used in generative classifiers significantly amplifies this privacy risk. These findings reveal a fundamental utility-privacy trade-off inherent in classifier design, underscoring the critical need for caution when deploying generative classifiers in privacy-sensitive applications. Our results motivate future research directions in developing privacy-preserving generative classifiers that can maintain utility while mitigating membership inference vulnerabilities. 

---
# SIADAFIX: issue description response for adaptive program repair 

**Authors**: Xin Cao, Nan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16059)  

**Abstract**: We propose utilizing fast and slow thinking to enhance the capabilities of large language model-based agents on complex tasks such as program repair. In particular, we design an adaptive program repair method based on issue description response, called SIADAFIX. The proposed method utilizes slow thinking bug fix agent to complete complex program repair tasks, and employs fast thinking workflow decision components to optimize and classify issue descriptions, using issue description response results to guide the orchestration of bug fix agent workflows. SIADAFIX adaptively selects three repair modes, i.e., easy, middle and hard mode, based on problem complexity. It employs fast generalization for simple problems and test-time scaling techniques for complex problems. Experimental results on the SWE-bench Lite show that the proposed method achieves 60.67% pass@1 performance using the Claude-4 Sonnet model, reaching state-of-the-art levels among all open-source methods. SIADAFIX effectively balances repair efficiency and accuracy, providing new insights for automated program repair. Our code is available at this https URL. 

---
# PrivacyPAD: A Reinforcement Learning Framework for Dynamic Privacy-Aware Delegation 

**Authors**: Zheng Hui, Yijiang River Dong, Sanhanat Sivapiromrat, Ehsan Shareghi, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2510.16054)  

**Abstract**: When users submit queries to Large Language Models (LLMs), their prompts can often contain sensitive data, forcing a difficult choice: Send the query to a powerful proprietary LLM providers to achieving state-of-the-art performance and risk data exposure, or relying on smaller, local models guarantees data privacy but often results in a degradation of task performance. Prior approaches have relied on static pipelines that use LLM rewriting, which shatters linguistic coherence and indiscriminately removes privacy-sensitive information, including task-critical content. We reformulate this challenge (Privacy-Conscious Delegation) as a sequential decision-making problem and introduce a novel reinforcement learning (RL) framework called PrivacyPAD to solve it. Our framework trains an agent to dynamically route text chunks, learning a policy that optimally balances the trade-off between privacy leakage and task performance. It implicitly distinguishes between replaceable Personally Identifiable Information (PII) (which it shields locally) and task-critical PII (which it strategically sends to the remote model for maximal utility). To validate our approach in complex scenarios, we also introduce a new medical dataset with high PII density. Our framework achieves a new state-of-the-art on the privacy-utility frontier, demonstrating the necessity of learned, adaptive policies for deploying LLMs in sensitive environments. 

---
# InfraGPT Smart Infrastructure: An End-to-End VLM-Based Framework for Detecting and Managing Urban Defects 

**Authors**: Ibrahim Sheikh Mohamed, Abdullah Yahya Abdullah Omaisan  

**Link**: [PDF](https://arxiv.org/pdf/2510.16017)  

**Abstract**: Infrastructure in smart cities is increasingly monitored by networks of closed circuit television (CCTV) cameras. Roads, bridges and tunnels develop cracks, potholes, and fluid leaks that threaten public safety and require timely repair. Manual inspection is costly and hazardous, and existing automatic systems typically address individual defect types or provide unstructured outputs that cannot directly guide maintenance crews. This paper proposes a comprehensive pipeline that leverages street CCTV streams for multi defect detection and segmentation using the YOLO family of object detectors and passes the detections to a vision language model (VLM) for scene aware summarization. The VLM generates a structured action plan in JSON format that includes incident descriptions, recommended tools, dimensions, repair plans, and urgent alerts. We review literature on pothole, crack and leak detection, highlight recent advances in large vision language models such as QwenVL and LLaVA, and describe the design of our early prototype. Experimental evaluation on public datasets and captured CCTV clips demonstrates that the system accurately identifies diverse defects and produces coherent summaries. We conclude by discussing challenges and directions for scaling the system to city wide deployments. 

---
# Can GRPO Help LLMs Transcend Their Pretraining Origin? 

**Authors**: Kangqi Ni, Zhen Tan, Zijie Liu, Pingzhi Li, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15990)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR), primarily driven by the Group Relative Policy Optimization (GRPO) algorithm, is a leading approach for enhancing the reasoning abilities of Large Language Models (LLMs). Despite its wide adoption, GRPO's gains are often inconsistent; for instance, a model may show significant improvement in one reasoning domain, like mathematics, yet remain stagnant in another, such as medicine. This inconsistency raises a critical question: under what conditions does GRPO improve reasoning and generalize out-of-distribution (OOD)? We investigate this from a data distribution perspective. We first prove theoretically that GRPO is a conservative reweighting scheme, bounded by the base model's distribution and thus unable to discover completely novel solutions. We further validate this in carefully designed controlled studies by training transformers from scratch, evaluating generalization across reasoning depth, input length, token representation, and compositionality. Our results provide a principled explanation for GRPO's boundaries: OOD improvement emerges only when the target task aligns with the model's pretrained biases, while gains on in-distribution (ID) tasks diminish as performance saturates. This reframes GRPO not as a universal reasoning enhancer but as a tool that sharpens pretraining biases. Our findings motivate future development of algorithms that can expand a model's capabilities beyond its pretraining origin. 

---
# Bolster Hallucination Detection via Prompt-Guided Data Augmentation 

**Authors**: Wenyun Li, Zheng Zhang, Dongmei Jiang, Xiangyuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.15977)  

**Abstract**: Large language models (LLMs) have garnered significant interest in AI community. Despite their impressive generation capabilities, they have been found to produce misleading or fabricated information, a phenomenon known as hallucinations. Consequently, hallucination detection has become critical to ensure the reliability of LLM-generated content. One primary challenge in hallucination detection is the scarcity of well-labeled datasets containing both truthful and hallucinated outputs. To address this issue, we introduce Prompt-guided data Augmented haLlucination dEtection (PALE), a novel framework that leverages prompt-guided responses from LLMs as data augmentation for hallucination detection. This strategy can generate both truthful and hallucinated data under prompt guidance at a relatively low cost. To more effectively evaluate the truthfulness of the sparse intermediate embeddings produced by LLMs, we introduce an estimation metric called the Contrastive Mahalanobis Score (CM Score). This score is based on modeling the distributions of truthful and hallucinated data in the activation space. CM Score employs a matrix decomposition approach to more accurately capture the underlying structure of these distributions. Importantly, our framework does not require additional human annotations, offering strong generalizability and practicality for real-world applications. Extensive experiments demonstrate that PALE achieves superior hallucination detection performance, outperforming the competitive baseline by a significant margin of 6.55%. 

---
# Long Exposure: Accelerating Parameter-Efficient Fine-Tuning for LLMs under Shadowy Sparsity 

**Authors**: Tuowei Wang, Kun Li, Zixu Hao, Donglin Bai, Ju Ren, Yaoxue Zhang, Ting Cao, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15964)  

**Abstract**: The adaptation of pre-trained large language models (LLMs) to diverse downstream tasks via fine-tuning is critical for numerous applications. However, the inefficiency of parameter-efficient fine-tuning (PEFT) techniques presents significant challenges in terms of time investments and operational costs. In this paper, we first introduce a nuanced form of sparsity, termed Shadowy Sparsity, which is distinctive in fine-tuning and has not been adequately addressed for acceleration. Under Shadowy Sparsity, we propose Long Exposure, an efficient system to accelerate PEFT for LLMs. Long Exposure comprises three key components: Shadowy-sparsity Exposer employs a prolonged sensing range to capture more sparsity details under shadowy sparsity; Sequence-oriented Predictor provides efficient yet accurate predictions to handle large sequence inputs and constantly-evolving parameters; and Dynamic-aware Operator facilitates more structured computational patterns and coalesced memory accesses, addressing dynamic sparse operations. Extensive evaluations show that Long Exposure outperforms state-of-the-arts with up to a $2.49\times$ speedup in end-to-end fine-tuning, offering promising advancements in accelerating PEFT for LLMs. 

---
# Attention to Non-Adopters 

**Authors**: Kaitlyn Zhou, Kristina Gligorić, Myra Cheng, Michelle S. Lam, Vyoma Raman, Boluwatife Aminu, Caeley Woo, Michael Brockman, Hannah Cha, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2510.15951)  

**Abstract**: Although language model-based chat systems are increasingly used in daily life, most Americans remain non-adopters of chat-based LLMs -- as of June 2025, 66% had never used ChatGPT. At the same time, LLM development and evaluation rely mainly on data from adopters (e.g., logs, preference data), focusing on the needs and tasks for a limited demographic group of adopters in terms of geographic location, education, and gender. In this position paper, we argue that incorporating non-adopter perspectives is essential for developing broadly useful and capable LLMs. We contend that relying on methods that focus primarily on adopters will risk missing a range of tasks and needs prioritized by non-adopters, entrenching inequalities in who benefits from LLMs, and creating oversights in model development and evaluation. To illustrate this claim, we conduct case studies with non-adopters and show: how non-adopter needs diverge from those of current users, how non-adopter needs point us towards novel reasoning tasks, and how to systematically integrate non-adopter needs via human-centered methods. 

---
# Comparing LLMs for Sentiment Analysis in Financial Market News 

**Authors**: Lucas Eduardo Pereira Teles, Carlos M. S. Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2510.15929)  

**Abstract**: This article presents a comparative study of large language models (LLMs) in the task of sentiment analysis of financial market news. This work aims to analyze the performance difference of these models in this important natural language processing task within the context of finance. LLM models are compared with classical approaches, allowing for the quantification of the benefits of each tested model or approach. Results show that large language models outperform classical models in the vast majority of cases. 

---
# HealthDial: A No-Code LLM-Assisted Dialogue Authoring Tool for Healthcare Virtual Agents 

**Authors**: Farnaz Nouraei, Zhuorui Yong, Timothy Bickmore  

**Link**: [PDF](https://arxiv.org/pdf/2510.15898)  

**Abstract**: We introduce HealthDial, a dialogue authoring tool that helps healthcare providers and educators create virtual agents that deliver health education and counseling to patients over multiple conversations. HealthDial leverages large language models (LLMs) to automatically create an initial session-based plan and conversations for each session using text-based patient health education materials as input. Authored dialogue is output in the form of finite state machines for virtual agent delivery so that all content can be validated and no unsafe advice is provided resulting from LLM hallucinations. LLM-drafted dialogue structure and language can be edited by the author in a no-code user interface to ensure validity and optimize clarity and impact. We conducted a feasibility and usability study with counselors and students to test our approach with an authoring task for cancer screening education. Participants used HealthDial and then tested their resulting dialogue by interacting with a 3D-animated virtual agent delivering the dialogue. Through participants' evaluations of the task experience and final dialogues, we show that HealthDial provides a promising first step for counselors to ensure full coverage of their health education materials, while creating understandable and actionable virtual agent dialogue with patients. 

---
# Detecting and Preventing Harmful Behaviors in AI Companions: Development and Evaluation of the SHIELD Supervisory System 

**Authors**: Ziv Ben-Zion, Paul Raffelhüschen, Max Zettl, Antonia Lüönd, Achim Burrer, Philipp Homan, Tobias R Spiller  

**Link**: [PDF](https://arxiv.org/pdf/2510.15891)  

**Abstract**: AI companions powered by large language models (LLMs) are increasingly integrated into users' daily lives, offering emotional support and companionship. While existing safety systems focus on overt harms, they rarely address early-stage problematic behaviors that can foster unhealthy emotional dynamics, including over-attachment or reinforcement of social isolation. We developed SHIELD (Supervisory Helper for Identifying Emotional Limits and Dynamics), a LLM-based supervisory system with a specific system prompt that detects and mitigates risky emotional patterns before escalation. SHIELD targets five dimensions of concern: (1) emotional over-attachment, (2) consent and boundary violations, (3) ethical roleplay violations, (4) manipulative engagement, and (5) social isolation reinforcement. These dimensions were defined based on media reports, academic literature, existing AI risk frameworks, and clinical expertise in unhealthy relationship dynamics. To evaluate SHIELD, we created a 100-item synthetic conversation benchmark covering all five dimensions of concern. Testing across five prominent LLMs (GPT-4.1, Claude Sonnet 4, Gemma 3 1B, Kimi K2, Llama Scout 4 17B) showed that the baseline rate of concerning content (10-16%) was significantly reduced with SHIELD (to 3-8%), a 50-79% relative reduction, while preserving 95% of appropriate interactions. The system achieved 59% sensitivity and 95% specificity, with adaptable performance via prompt engineering. This proof-of-concept demonstrates that transparent, deployable supervisory systems can address subtle emotional manipulation in AI companions. Most development materials including prompts, code, and evaluation methods are made available as open source materials for research, adaptation, and deployment. 

---
# Mitigating Harmful Erraticism in LLMs Through Dialectical Behavior Therapy Based De-Escalation Strategies 

**Authors**: Pooja Rangarajan, Jacob Boyle  

**Link**: [PDF](https://arxiv.org/pdf/2510.15889)  

**Abstract**: The escalating demand for personalized AI chatbot interactions, capable of dynamically adapting to user emotional states and real-time requests, has highlighted critical limitations in current development paradigms. Existing methodologies, which rely on baseline programming, custom personalities, and manual response adjustments, often prove difficult to maintain and are susceptible to errors such as hallucinations, erratic outputs, and software bugs. This paper hypothesizes that a framework rooted in human psychological principles, specifically therapeutic modalities, can provide a more robust and sustainable solution than purely technical interventions. Drawing an analogy to the simulated neural networks of AI mirroring the human brain, we propose the application of Dialectical Behavior Therapy (DBT) principles to regulate chatbot responses to diverse user inputs. This research investigates the impact of a DBT-based framework on AI chatbot performance, aiming to ascertain its efficacy in yielding more reliable, safe, and accurate responses, while mitigating the occurrence of hallucinations, erratic behaviors, and other systemic issues. 

---
