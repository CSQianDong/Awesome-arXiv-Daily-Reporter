# LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation 

**Authors**: Joseph Enguehard, Morgane Van Ermengem, Kate Atkinson, Sujeong Cha, Arijit Ghosh Chowdhury, Prashanth Kallur Ramaswamy, Jeremy Roghair, Hannah R Marlowe, Carina Suzana Negreanu, Kitty Boxall, Diana Mincu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07243)  

**Abstract**: Evaluating large language model (LLM) outputs in the legal domain presents unique challenges due to the complex and nuanced nature of legal analysis. Current evaluation approaches either depend on reference data, which is costly to produce, or use standardized assessment methods, both of which have significant limitations for legal applications.
Although LLM-as-a-Judge has emerged as a promising evaluation technique, its reliability and effectiveness in legal contexts depend heavily on evaluation processes unique to the legal industry and how trustworthy the evaluation appears to the human legal expert. This is where existing evaluation methods currently fail and exhibit considerable variability.
This paper aims to close the gap: a) we break down lengthy responses into 'Legal Data Points' (LDPs), self-contained units of information, and introduce a novel, reference-free evaluation methodology that reflects how lawyers evaluate legal answers; b) we demonstrate that our method outperforms a variety of baselines on both our proprietary dataset and an open-source dataset (LegalBench); c) we show how our method correlates more closely with human expert evaluations and helps improve inter-annotator agreement; and finally d) we open source our Legal Data Points for a subset of LegalBench used in our experiments, allowing the research community to replicate our results and advance research in this vital area of LLM evaluation on legal question-answering. 

---
# On the Convergence of Moral Self-Correction in Large Language Models 

**Authors**: Guangliang Liu, Haitao Mao, Bochuan Cao, Zhiyu Xue, Xitong Zhang, Rongrong Wang, Kristen Marie Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2510.07290)  

**Abstract**: Large Language Models (LLMs) are able to improve their responses when instructed to do so, a capability known as self-correction. When instructions provide only a general and abstract goal without specific details about potential issues in the response, LLMs must rely on their internal knowledge to improve response quality, a process referred to as intrinsic self-correction. The empirical success of intrinsic self-correction is evident in various applications, but how and why it is effective remains unknown. Focusing on moral self-correction in LLMs, we reveal a key characteristic of intrinsic self-correction: performance convergence through multi-round interactions; and provide a mechanistic analysis of this convergence behavior. Based on our experimental results and analysis, we uncover the underlying mechanism of convergence: consistently injected self-correction instructions activate moral concepts that reduce model uncertainty, leading to converged performance as the activated moral concepts stabilize over successive rounds. This paper demonstrates the strong potential of moral self-correction by showing that it exhibits a desirable property of converged performance. 

---
# Vibe Checker: Aligning Code Evaluation with Human Preference 

**Authors**: Ming Zhong, Xiang Zhou, Ting-Yun Chang, Qingze Wang, Nan Xu, Xiance Si, Dan Garrette, Shyam Upadhyay, Jeremiah Liu, Jiawei Han, Benoit Schillings, Jiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.07315)  

**Abstract**: Large Language Models (LLMs) have catalyzed vibe coding, where users leverage LLMs to generate and iteratively refine code through natural language interactions until it passes their vibe check. Vibe check is tied to real-world human preference and goes beyond functionality: the solution should feel right, read cleanly, preserve intent, and remain correct. However, current code evaluation remains anchored to pass@k and captures only functional correctness, overlooking the non-functional instructions that users routinely apply. In this paper, we hypothesize that instruction following is the missing piece underlying vibe check that represents human preference in coding besides functional correctness. To quantify models' code instruction following capabilities with measurable signals, we present VeriCode, a taxonomy of 30 verifiable code instructions together with corresponding deterministic verifiers. We use the taxonomy to augment established evaluation suites, resulting in Vibe Checker, a testbed to assess both code instruction following and functional correctness. Upon evaluating 31 leading LLMs, we show that even the strongest models struggle to comply with multiple instructions and exhibit clear functional regression. Most importantly, a composite score of functional correctness and instruction following correlates the best with human preference, with the latter emerging as the primary differentiator on real-world programming tasks. Our work identifies core factors of the vibe check, providing a concrete path for benchmarking and developing models that better align with user preferences in coding. 

---
# Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts 

**Authors**: Christos Ziakas, Nicholas Loo, Nishita Jain, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2510.07239)  

**Abstract**: Automated red-teaming has emerged as a scalable approach for auditing Large Language Models (LLMs) prior to deployment, yet existing approaches lack mechanisms to efficiently adapt to model-specific vulnerabilities at inference. We introduce Red-Bandit, a red-teaming framework that adapts online to identify and exploit model failure modes under distinct attack styles (e.g., manipulation, slang). Red-Bandit post-trains a set of parameter-efficient LoRA experts, each specialized for a particular attack style, using reinforcement learning that rewards the generation of unsafe prompts via a rule-based safety model. At inference, a multi-armed bandit policy dynamically selects among these attack-style experts based on the target model's response safety, balancing exploration and exploitation. Red-Bandit achieves state-of-the-art results on AdvBench under sufficient exploration (ASR@10), while producing more human-readable prompts (lower perplexity). Moreover, Red-Bandit's bandit policy serves as a diagnostic tool for uncovering model-specific vulnerabilities by indicating which attack styles most effectively elicit unsafe behaviors. 

---
# Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships 

**Authors**: Donggyu Lee, Sungwon Park, Yerin Hwang, Hyunwoo Oh, Hyoshin Kim, Jungwon Kim, Meeyoung Cha, Sangyoon Park, Jihee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.07231)  

**Abstract**: Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications. 

---
# When Benchmarks Age: Temporal Misalignment through Large Language Model Factuality Evaluation 

**Authors**: Xunyi Jiang, Dingyi Chang, Julian McAuley, Xin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07238)  

**Abstract**: The rapid evolution of large language models (LLMs) and the real world has outpaced the static nature of widely used evaluation benchmarks, raising concerns about their reliability for evaluating LLM factuality. While substantial works continue to rely on the popular but old benchmarks, their temporal misalignment with real-world facts and modern LLMs, and their effects on LLM factuality evaluation remain underexplored. Therefore, in this work, we present a systematic investigation of this issue by examining five popular factuality benchmarks and eight LLMs released across different years. An up-to-date fact retrieval pipeline and three metrics are tailored to quantify benchmark aging and its impact on LLM factuality evaluation. Experimental results and analysis illustrate that a considerable portion of samples in the widely used factuality benchmarks are outdated, leading to unreliable assessments of LLM factuality. We hope our work can provide a testbed to assess the reliability of a benchmark for LLM factuality evaluation and inspire more research on the benchmark aging issue. Codes are available in this https URL. 

---
# Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping 

**Authors**: Ziyi Wang, Yuxuan Lu, Yimeng Zhang, Jing Huang, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07230)  

**Abstract**: Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation. 

---
# NurseLLM: The First Specialized Language Model for Nursing 

**Authors**: Md Tawkat Islam Khondaker, Julia Harrington, Shady Shehata  

**Link**: [PDF](https://arxiv.org/pdf/2510.07173)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly transformed medical systems. However, their potential within specialized domains such as nursing remains largely underexplored. In this work, we introduce NurseLLM, the first nursing-specialized LLM tailored for multiple choice question-answering (MCQ) tasks. We develop a multi-stage data generation pipeline to build the first large scale nursing MCQ dataset to train LLMs on a broad spectrum of nursing topics. We further introduce multiple nursing benchmarks to enable rigorous evaluation. Our extensive experiments demonstrate that NurseLLM outperforms SoTA general-purpose and medical-specialized LLMs of comparable size on different benchmarks, underscoring the importance of a specialized LLM for the nursing domain. Finally, we explore the role of reasoning and multi-agent collaboration systems in nursing, highlighting their promise for future research and applications. 

---
# Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dense 

**Authors**: Leitian Tao, Ilia Kulikov, Swarnadeep Saha, Tianlu Wang, Jing Xu, Yixuan Li, Jason E Weston, Ping Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07242)  

**Abstract**: Post-training for reasoning of large language models (LLMs) increasingly relies on verifiable rewards: deterministic checkers that provide 0-1 correctness signals. While reliable, such binary feedback is brittle--many tasks admit partially correct or alternative answers that verifiers under-credit, and the resulting all-or-nothing supervision limits learning. Reward models offer richer, continuous feedback, which can serve as a complementary supervisory signal to verifiers. We introduce HERO (Hybrid Ensemble Reward Optimization), a reinforcement learning framework that integrates verifier signals with reward-model scores in a structured way. HERO employs stratified normalization to bound reward-model scores within verifier-defined groups, preserving correctness while refining quality distinctions, and variance-aware weighting to emphasize challenging prompts where dense signals matter most. Across diverse mathematical reasoning benchmarks, HERO consistently outperforms RM-only and verifier-only baselines, with strong gains on both verifiable and hard-to-verify tasks. Our results show that hybrid reward design retains the stability of verifiers while leveraging the nuance of reward models to advance reasoning. 

---
# Reasoning for Hierarchical Text Classification: The Case of Patents 

**Authors**: Lekang Jiang, Wenjun Sun, Stephan Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2510.07167)  

**Abstract**: Hierarchical text classification (HTC) assigns documents to multiple levels of a pre-defined taxonomy. Automated patent subject classification represents one of the hardest HTC scenarios because of domain knowledge difficulty and a huge number of labels. Prior approaches only output a flat label set, which offers little insight into the reason behind predictions. Therefore, we propose Reasoning for Hierarchical Classification (RHC), a novel framework that reformulates HTC as a step-by-step reasoning task to sequentially deduce hierarchical labels. RHC trains large language models (LLMs) in two stages: a cold-start stage that aligns outputs with chain-of-thought (CoT) reasoning format and a reinforcement learning (RL) stage to enhance multi-step reasoning ability. RHC demonstrates four advantages in our experiments. (1) Effectiveness: RHC surpasses previous baselines and outperforms the supervised fine-tuning counterparts by approximately 3% in accuracy and macro F1. (2) Explainability: RHC produces natural-language justifications before prediction to facilitate human inspection. (3) Scalability: RHC scales favorably with model size with larger gains compared to standard fine-tuning. (4) Applicability: Beyond patents, we further demonstrate that RHC achieves state-of-the-art performance on other widely used HTC benchmarks, which highlights its broad applicability. 

---
# More Data or Better Data? A Critical Analysis of Data Selection and Synthesis for Mathematical Reasoning 

**Authors**: Yike Zhao, Simin Guo, Ziqing Yang, Shifan Han, Dahua Lin, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.07169)  

**Abstract**: The reasoning capabilities of Large Language Models (LLMs) play a critical role in many downstream tasks, yet depend strongly on the quality of training data. Despite various proposed data construction methods, their practical utility in real-world pipelines remains underexplored. In this work, we conduct a comprehensive analysis of open-source datasets and data synthesis techniques for mathematical reasoning, evaluating them under a unified pipeline designed to mirror training and deployment scenarios. We further distill effective data selection strategies and identify practical methods suitable for industrial applications. Our findings highlight that structuring data in more interpretable formats, or distilling from stronger models often outweighs simply scaling up data volume. This study provides actionable guidance for integrating training data to enhance LLM capabilities, supporting both cost-effective data curation and scalable model enhancement. We hope this work will inspire further research on how to balance "more data" versus "better data" for real-world reasoning tasks. 

---
# Comparing human and language models sentence processing difficulties on complex structures 

**Authors**: Samuel Joseph Amouyal, Aya Meltzer-Asscher, Jonathan Berant  

**Link**: [PDF](https://arxiv.org/pdf/2510.07141)  

**Abstract**: Large language models (LLMs) that fluently converse with humans are a reality - but do LLMs experience human-like processing difficulties? We systematically compare human and LLM sentence comprehension across seven challenging linguistic structures. We collect sentence comprehension data from humans and five families of state-of-the-art LLMs, varying in size and training procedure in a unified experimental framework. Our results show LLMs overall struggle on the target structures, but especially on garden path (GP) sentences. Indeed, while the strongest models achieve near perfect accuracy on non-GP structures (93.7% for GPT-5), they struggle on GP structures (46.8% for GPT-5). Additionally, when ranking structures based on average performance, rank correlation between humans and models increases with parameter count. For each target structure, we also collect data for their matched baseline without the difficult structure. Comparing performance on the target vs. baseline sentences, the performance gap observed in humans holds for LLMs, with two exceptions: for models that are too weak performance is uniformly low across both sentence types, and for models that are too strong the performance is uniformly high. Together, these reveal convergence and divergence in human and LLM sentence comprehension, offering new insights into the similarity of humans and LLMs. 

---
# CARPAS: Towards Content-Aware Refinement of Provided Aspects for Summarization in Large Language Models 

**Authors**: Yong-En Tian, Yu-Chien Tang, An-Zi Yen, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07177)  

**Abstract**: Aspect-based summarization has attracted significant attention for its ability to generate more fine-grained and user-aligned summaries. While most existing approaches assume a set of predefined aspects as input, real-world scenarios often present challenges where these given aspects may be incomplete, irrelevant, or entirely missing from the document. Users frequently expect systems to adaptively refine or filter the provided aspects based on the actual content. In this paper, we initiate this novel task setting, termed Content-Aware Refinement of Provided Aspects for Summarization (CARPAS), with the aim of dynamically adjusting the provided aspects based on the document context before summarizing. We construct three new datasets to facilitate our pilot experiments, and by using LLMs with four representative prompting strategies in this task, we find that LLMs tend to predict an overly comprehensive set of aspects, which often results in excessively long and misaligned summaries. Building on this observation, we propose a preliminary subtask to predict the number of relevant aspects, and demonstrate that the predicted number can serve as effective guidance for the LLMs, reducing the inference difficulty, and enabling them to focus on the most pertinent aspects. Our extensive experiments show that the proposed approach significantly improves performance across all datasets. Moreover, our deeper analyses uncover LLMs' compliance when the requested number of aspects differs from their own estimations, establishing a crucial insight for the deployment of LLMs in similar real-world applications. 

---
# Making Machines Sound Sarcastic: LLM-Enhanced and Retrieval-Guided Sarcastic Speech Synthesis 

**Authors**: Zhu Li, Yuqing Zhang, Xiyuan Gao, Shekhar Nayak, Matt Coler  

**Link**: [PDF](https://arxiv.org/pdf/2510.07096)  

**Abstract**: Sarcasm is a subtle form of non-literal language that poses significant challenges for speech synthesis due to its reliance on nuanced semantic, contextual, and prosodic cues. While existing speech synthesis research has focused primarily on broad emotional categories, sarcasm remains largely unexplored. In this paper, we propose a Large Language Model (LLM)-enhanced Retrieval-Augmented framework for sarcasm-aware speech synthesis. Our approach combines (1) semantic embeddings from a LoRA-fine-tuned LLaMA 3, which capture pragmatic incongruity and discourse-level cues of sarcasm, and (2) prosodic exemplars retrieved via a Retrieval Augmented Generation (RAG) module, which provide expressive reference patterns of sarcastic delivery. Integrated within a VITS backbone, this dual conditioning enables more natural and contextually appropriate sarcastic speech. Experiments demonstrate that our method outperforms baselines in both objective measures and subjective evaluations, yielding improvements in speech naturalness, sarcastic expressivity, and downstream sarcasm detection. 

---
# All Claims Are Equal, but Some Claims Are More Equal Than Others: Importance-Sensitive Factuality Evaluation of LLM Generations 

**Authors**: Miriam Wanner, Leif Azzopardi, Paul Thomas, Soham Dan, Benjamin Van Durme, Nick Craswell  

**Link**: [PDF](https://arxiv.org/pdf/2510.07083)  

**Abstract**: Existing methods for evaluating the factuality of large language model (LLM) responses treat all claims as equally important. This results in misleading evaluations when vital information is missing or incorrect as it receives the same weight as peripheral details, raising the question: how can we reliably detect such differences when there are errors in key information? Current approaches that measure factuality tend to be insensitive to omitted or false key information. To investigate this lack of sensitivity, we construct VITALERRORS, a benchmark of 6,733 queries with minimally altered LLM responses designed to omit or falsify key information. Using this dataset, we demonstrate the insensitivities of existing evaluation metrics to key information errors. To address this gap, we introduce VITAL, a set of metrics that provide greater sensitivity in measuring the factuality of responses by incorporating the relevance and importance of claims with respect to the query. Our analysis demonstrates that VITAL metrics more reliably detect errors in key information than previous methods. Our dataset, metrics, and analysis provide a foundation for more accurate and robust assessment of LLM factuality. 

---
# Opt-ICL at LeWiDi-2025: Maximizing In-Context Signal from Rater Examples via Meta-Learning 

**Authors**: Taylor Sorensen, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.07105)  

**Abstract**: Many natural language processing (NLP) tasks involve subjectivity, ambiguity, or legitimate disagreement between annotators. In this paper, we outline our system for modeling human variation. Our system leverages language models' (LLMs) in-context learning abilities, along with a two-step meta-learning training procedure for 1) post-training on many datasets requiring in-context learning and 2) specializing the model via in-context meta-learning to the particular data distribution of interest. We also evaluate the performance of our system submission to the Learning With Disagreements (LeWiDi) competition, where it was the overall winner on both tasks. Additionally, we perform an ablation study to measure the importance of each system component. We find that including rater examples in-context is crucial for our system's performance, dataset-specific fine-tuning is helpful on the larger datasets, post-training on other in-context datasets is helpful on one of the competition datasets, and that performance improves with model scale. 

---
# Search-R3: Unifying Reasoning and Embedding Generation in Large Language Models 

**Authors**: Yuntao Gui, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07048)  

**Abstract**: Despite their remarkable natural language understanding capabilities, Large Language Models (LLMs) have been underutilized for retrieval tasks. We present Search-R3, a novel framework that addresses this limitation by adapting LLMs to generate search embeddings as a direct output of their reasoning process. Our approach exploits LLMs' chain-of-thought capabilities, allowing them to produce more effective embeddings by reasoning step-by-step through complex semantic analyses. We implement this through three complementary mechanisms. (1) a supervised learning stage enables the model's ability to produce quality embeddings, (2) a reinforcement learning (RL) methodology that optimizes embedding generation alongside reasoning, and (3) a specialized RL environment that efficiently handles evolving embedding representations without requiring complete corpus re-encoding at each training iteration. Our extensive evaluations on diverse benchmarks demonstrate that Search-R3 significantly outperforms prior methods by unifying the reasoning and embedding generation processes. This integrated post-training approach represents a substantial advancement in handling complex knowledge-intensive tasks that require both sophisticated reasoning and effective information retrieval. Project page: this https URL 

---
# Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledge 

**Authors**: Shrestha Ghosh, Luca Giordano, Yujia Hu, Tuan-Phong Nguyen, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.07024)  

**Abstract**: LLMs are remarkable artifacts that have revolutionized a range of NLP and AI tasks. A significant contributor is their factual knowledge, which, to date, remains poorly understood, and is usually analyzed from biased samples. In this paper, we take a deep tour into the factual knowledge (or beliefs) of a frontier LLM, based on GPTKB v1.5 (Hu et al., 2025a), a recursively elicited set of 100 million beliefs of one of the strongest currently available frontier LLMs, GPT-4.1. We find that the models' factual knowledge differs quite significantly from established knowledge bases, and that its accuracy is significantly lower than indicated by previous benchmarks. We also find that inconsistency, ambiguity and hallucinations are major issues, shedding light on future research opportunities concerning factual LLM knowledge. 

---
# Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models 

**Authors**: Rajvee Sheth, Samridhi Raj Sinha, Mahavir Patil, Himanshu Beniwal, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.07037)  

**Abstract**: Code-switching (CSW), the alternation of languages and scripts within a single utterance, remains a fundamental challenge for multiling ual NLP, even amidst the rapid advances of large language models (LLMs). Most LLMs still struggle with mixed-language inputs, limited CSW datasets, and evaluation biases, hindering deployment in multilingual societies. This survey provides the first comprehensive analysis of CSW-aware LLM research, reviewing \total{unique_references} studies spanning five research areas, 12 NLP tasks, 30+ datasets, and 80+ languages. We classify recent advances by architecture, training strategy, and evaluation methodology, outlining how LLMs have reshaped CSW modeling and what challenges persist. The paper concludes with a roadmap emphasizing the need for inclusive datasets, fair evaluation, and linguistically grounded models to achieve truly multilingual intelligence. A curated collection of all resources is maintained at this https URL. 

---
# Accelerating Diffusion LLM Inference via Local Determinism Propagation 

**Authors**: Fanheng Kong, Jingyuan Zhang, Yahui Liu, Zirui Wu, Yu Tian, Victoria W., Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.07081)  

**Abstract**: Diffusion large language models (dLLMs) represent a significant advancement in text generation, offering parallel token decoding capabilities. However, existing open-source implementations suffer from quality-speed trade-offs that impede their practical deployment. Conservative sampling strategies typically decode only the most confident token per step to ensure quality (i.e., greedy decoding), at the cost of inference efficiency due to repeated redundant refinement iterations--a phenomenon we term delayed decoding. Through systematic analysis of dLLM decoding dynamics, we characterize this delayed decoding behavior and propose a training-free adaptive parallel decoding strategy, named LocalLeap, to address these inefficiencies. LocalLeap is built on two fundamental empirical principles: local determinism propagation centered on high-confidence anchors and progressive spatial consistency decay. By applying these principles, LocalLeap identifies anchors and performs localized relaxed parallel decoding within bounded neighborhoods, achieving substantial inference step reduction through early commitment of already-determined tokens without compromising output quality. Comprehensive evaluation on various benchmarks demonstrates that LocalLeap achieves 6.94$\times$ throughput improvements and reduces decoding steps to just 14.2\% of the original requirement, achieving these gains with negligible performance impact. The source codes are available at: this https URL. 

---
# TALENT: Table VQA via Augmented Language-Enhanced Natural-text Transcription 

**Authors**: Guo Yutong, Wanying Wang, Yue Wu, Zichen Miao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07098)  

**Abstract**: Table Visual Question Answering (Table VQA) is typically addressed by large vision-language models (VLMs). While such models can answer directly from images, they often miss fine-grained details unless scaled to very large sizes, which are computationally prohibitive, especially for mobile deployment. A lighter alternative is to have a small VLM perform OCR and then use a large language model (LLM) to reason over structured outputs such as Markdown tables. However, these representations are not naturally optimized for LLMs and still introduce substantial errors. We propose TALENT (Table VQA via Augmented Language-Enhanced Natural-text Transcription), a lightweight framework that leverages dual representations of tables. TALENT prompts a small VLM to produce both OCR text and natural language narration, then combines them with the question for reasoning by an LLM. This reframes Table VQA as an LLM-centric multimodal reasoning task, where the VLM serves as a perception-narration module rather than a monolithic solver. Additionally, we construct ReTabVQA, a more challenging Table VQA dataset requiring multi-step quantitative reasoning over table images. Experiments show that TALENT enables a small VLM-LLM combination to match or surpass a single large VLM at significantly lower computational cost on both public datasets and ReTabVQA. 

---
# LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling 

**Authors**: Zecheng Tang, Baibei Ji, Quantong Qiu, Haitian Wang, Xiaobo Liang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06915)  

**Abstract**: Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences. As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model's responses are not only high-quality but also grounded in and consistent with the provided context. Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context-response consistency. In this work, we introduce Long-RewardBench, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks. Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments. Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs). Experiments show that our approach not only substantially improves performance on long-context evaluation but also preserves strong short-context capability. Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model. 

---
# Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation 

**Authors**: Vaibhav Srivastav, Steven Zheng, Eric Bezzam, Eustache Le Bihan, Nithin Koluguri, Piotr Żelasko, Somshubra Majumdar, Adel Moumen, Sanchit Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06961)  

**Abstract**: Despite rapid progress, ASR evaluation remains saturated with short-form English, and efficiency is rarely reported. We present the Open ASR Leaderboard, a fully reproducible benchmark and interactive leaderboard comparing 60+ open-source and proprietary systems across 11 datasets, including dedicated multilingual and long-form tracks. We standardize text normalization and report both word error rate (WER) and inverse real-time factor (RTFx), enabling fair accuracy-efficiency comparisons. For English transcription, Conformer encoders paired with LLM decoders achieve the best average WER but are slower, while CTC and TDT decoders deliver much better RTFx, making them attractive for long-form and offline use. Whisper-derived encoders fine-tuned for English improve accuracy but often trade off multilingual coverage. All code and dataset loaders are open-sourced to support transparent, extensible evaluation. 

---
# Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning 

**Authors**: Xue Zhang, Yunlong Liang, Fandong Meng, Songming Zhang, Kaiyu Huang, Yufeng Chen, Jinan Xu, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.07300)  

**Abstract**: Large Reasoning Models (LRMs) have achieved remarkable performance on complex reasoning tasks by adopting the "think-then-answer" paradigm, which enhances both accuracy and interpretability. However, current LRMs exhibit two critical limitations when processing non-English languages: (1) They often struggle to maintain input-output language consistency; (2) They generally perform poorly with wrong reasoning paths and lower answer accuracy compared to English. These limitations significantly degrade the user experience for non-English speakers and hinder the global deployment of LRMs. To address these limitations, we propose M-Thinker, which is trained by the GRPO algorithm that involves a Language Consistency (LC) reward and a novel Cross-lingual Thinking Alignment (CTA) reward. Specifically, the LC reward defines a strict constraint on the language consistency between the input, thought, and answer. Besides, the CTA reward compares the model's non-English reasoning paths with its English reasoning path to transfer its own reasoning capability from English to non-English languages. Through an iterative RL procedure, our M-Thinker-1.5B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath), but also exhibit excellent generalization on out-of-domain languages. 

---
# Unlocking Latent Discourse Translation in LLMs Through Quality-Aware Decoding 

**Authors**: Wafaa Mohammed, Vlad Niculae, Chrysoula Zerva  

**Link**: [PDF](https://arxiv.org/pdf/2510.06866)  

**Abstract**: Large language models (LLMs) have emerged as strong contenders in machine this http URL, they still struggle to adequately handle discourse phenomena, such as pronoun resolution and lexical cohesion at the document level. In this study, we thoroughly investigate the discourse phenomena performance of LLMs in context-aware translation. We demonstrate that discourse knowledge is encoded within LLMs and propose the use of quality-aware decoding (QAD) to effectively extract this knowledge, showcasing its superiority over other decoding approaches through comprehensive analysis. Furthermore, we illustrate that QAD enhances the semantic richness of translations and aligns them more closely with human preferences. 

---
# Mid-Training of Large Language Models: A Survey 

**Authors**: Kaixiang Mo, Yuxin Shi, Weiwei Weng, Zhiqiang Zhou, Shuman Liu, Haibo Zhang, Anxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.06826)  

**Abstract**: Large language models (LLMs) are typically developed through large-scale pre-training followed by task-specific fine-tuning. Recent advances highlight the importance of an intermediate mid-training stage, where models undergo multiple annealing-style phases that refine data quality, adapt optimization schedules, and extend context length. This stage mitigates diminishing returns from noisy tokens, stabilizes convergence, and expands model capability in late training. Its effectiveness can be explained through gradient noise scale, the information bottleneck, and curriculum learning, which together promote generalization and abstraction. Despite widespread use in state-of-the-art systems, there has been no prior survey of mid-training as a unified paradigm. We introduce the first taxonomy of LLM mid-training spanning data distribution, learning-rate scheduling, and long-context extension. We distill practical insights, compile evaluation benchmarks, and report gains to enable structured comparisons across models. We also identify open challenges and propose avenues for future research and practice. 

---
# SID: Multi-LLM Debate Driven by Self Signals 

**Authors**: Xuhang Chen, Zhifan Song, Deyi Ji, Shuo Gao, Lanyun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06843)  

**Abstract**: Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{this https URL}{\texttt{this https URL}}. 

---
# SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models 

**Authors**: Cheng-Han Chiang, Xiaofei Wang, Linjie Li, Chung-Ching Lin, Kevin Lin, Shujie Liu, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06917)  

**Abstract**: Current large language models (LLMs) and spoken language models (SLMs) begin thinking and taking actions only after the user has finished their turn. This prevents the model from interacting during the user's turn and can lead to high response latency while it waits to think. Consequently, thinking after receiving the full input is not suitable for speech-to-speech interaction, where real-time, low-latency exchange is important. We address this by noting that humans naturally "think while listening." In this paper, we propose SHANKS, a general inference framework that enables SLMs to generate unspoken chain-of-thought reasoning while listening to the user input. SHANKS streams the input speech in fixed-duration chunks and, as soon as a chunk is received, generates unspoken reasoning based on all previous speech and reasoning, while the user continues speaking. SHANKS uses this unspoken reasoning to decide whether to interrupt the user and to make tool calls to complete the task. We demonstrate that SHANKS enhances real-time user-SLM interaction in two scenarios: (1) when the user is presenting a step-by-step solution to a math problem, SHANKS can listen, reason, and interrupt when the user makes a mistake, achieving 37.1% higher interruption accuracy than a baseline that interrupts without thinking; and (2) in a tool-augmented dialogue, SHANKS can complete 56.9% of the tool calls before the user finishes their turn. Overall, SHANKS moves toward models that keep thinking throughout the conversation, not only after a turn ends. Animated illustrations of Shanks can be found at this https URL 

---
# FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline 

**Authors**: Haotian Wu, Shufan Jiang, Chios Chen, Yiyang Feng, Hehai Lin, Heqing Zou, Yao Shu, Yanran Li, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06800)  

**Abstract**: As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. These findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench. 

---
# OpenJAI-v1.0: An Open Thai Large Language Model 

**Authors**: Pontakorn Trakuekul, Attapol T. Rutherford, Jullajak Karnjanaekarin, Narongkorn Panitsrisit, Sumana Sumanakul  

**Link**: [PDF](https://arxiv.org/pdf/2510.06847)  

**Abstract**: We introduce OpenJAI-v1.0, an open-source large language model for Thai and English, developed from the Qwen3-14B model. Our work focuses on boosting performance on practical tasks through carefully curated data across three key use cases: instruction following, long-context understanding, and tool use. Evaluation results show that OpenJAI-v1.0 improves on the capabilities of its base model and outperforms other leading open-source Thai models on a diverse suite of benchmarks, while avoiding catastrophic forgetting. OpenJAI-v1.0 is publicly released as another alternative NLP resource for the Thai AI community. 

---
# Overview of the Plagiarism Detection Task at PAN 2025 

**Authors**: André Greiner-Petter, Maik Fröbe, Jan Philip Wahle, Terry Ruas, Bela Gipp, Akiko Aizawa, Martin Potthast  

**Link**: [PDF](https://arxiv.org/pdf/2510.06805)  

**Abstract**: The generative plagiarism detection task at PAN 2025 aims at identifying automatically generated textual plagiarism in scientific articles and aligning them with their respective sources. We created a novel large-scale dataset of automatically generated plagiarism using three large language models: Llama, DeepSeek-R1, and Mistral. In this task overview paper, we outline the creation of this dataset, summarize and compare the results of all participants and four baselines, and evaluate the results on the last plagiarism detection task from PAN 2015 in order to interpret the robustness of the proposed approaches. We found that the current iteration does not invite a large variety of approaches as naive semantic similarity approaches based on embedding vectors provide promising results of up to 0.8 recall and 0.5 precision. In contrast, most of these approaches underperform significantly on the 2015 dataset, indicating a lack in generalizability. 

---
# EDUMATH: Generating Standards-aligned Educational Math Word Problems 

**Authors**: Bryan R. Christ, Penelope Molitz, Jonathan Kropko, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06965)  

**Abstract**: Math word problems (MWPs) are critical K-12 educational tools, and customizing them to students' interests and ability levels can increase learning outcomes. However, teachers struggle to find time to customize MWPs for each student given large class sizes and increasing burnout. We propose that LLMs can support math education by generating MWPs customized to student interests and math education standards. To this end, we use a joint human expert-LLM judge approach to evaluate over 11,000 MWPs generated by open and closed LLMs and develop the first teacher-annotated dataset for standards-aligned educational MWP generation. We show the value of our data by using it to train a 12B open model that matches the performance of larger and more capable open models. We also use our teacher-annotated data to train a text classifier that enables a 30B open LLM to outperform existing closed baselines without any training. Next, we show our models' MWPs are more similar to human-written MWPs than those from existing models. We conclude by conducting the first study of customized LLM-generated MWPs with grade school students, finding they perform similarly on our models' MWPs relative to human-written MWPs but consistently prefer our customized MWPs. 

---
# Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition 

**Authors**: Lei Xu, Pierre Beckmann, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2510.06774)  

**Abstract**: Neuro-symbolic NLP methods aim to leverage the complementary strengths of large language models and formal logical solvers. However, current approaches are mostly static in nature, i.e., the integration of a target solver is predetermined at design time, hindering the ability to employ diverse formal inference strategies. To address this, we introduce an adaptive, multi-paradigm, neuro-symbolic inference framework that: (1) automatically identifies formal reasoning strategies from problems expressed in natural language; and (2) dynamically selects and applies specialized formal logical solvers via autoformalization interfaces. Extensive experiments on individual and multi-paradigm reasoning tasks support the following conclusions: LLMs are effective at predicting the necessary formal reasoning strategies with an accuracy above 90 percent. This enables flexible integration with formal logical solvers, resulting in our framework outperforming competing baselines by 27 percent and 6 percent compared to GPT-4o and DeepSeek-V3.1, respectively. Moreover, adaptive reasoning can even positively impact pure LLM methods, yielding gains of 10, 5, and 6 percent on zero-shot, CoT, and symbolic CoT settings with GPT-4o. Finally, although smaller models struggle with adaptive neuro-symbolic reasoning, post-training offers a viable path to improvement. Overall, this work establishes the foundations for adaptive LLM-symbolic reasoning, offering a path forward for unifying material and formal inferences on heterogeneous reasoning challenges. 

---
# Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness 

**Authors**: Luca Giordano, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.06780)  

**Abstract**: Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations. 

---
# BlackboxNLP-2025 MIB Shared Task: Exploring Ensemble Strategies for Circuit Localization Methods 

**Authors**: Philipp Mondorf, Mingyang Wang, Sebastian Gerstner, Ahmad Dawar Hakimi, Yihong Liu, Leonor Veloso, Shijia Zhou, Hinrich Schütze, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2510.06811)  

**Abstract**: The Circuit Localization track of the Mechanistic Interpretability Benchmark (MIB) evaluates methods for localizing circuits within large language models (LLMs), i.e., subnetworks responsible for specific task behaviors. In this work, we investigate whether ensembling two or more circuit localization methods can improve performance. We explore two variants: parallel and sequential ensembling. In parallel ensembling, we combine attribution scores assigned to each edge by different methods-e.g., by averaging or taking the minimum or maximum value. In the sequential ensemble, we use edge attribution scores obtained via EAP-IG as a warm start for a more expensive but more precise circuit identification method, namely edge pruning. We observe that both approaches yield notable gains on the benchmark metrics, leading to a more precise circuit identification approach. Finally, we find that taking a parallel ensemble over various methods, including the sequential ensemble, achieves the best results. We evaluate our approach in the BlackboxNLP 2025 MIB Shared Task, comparing ensemble scores to official baselines across multiple model-task combinations. 

---
# Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization 

**Authors**: Tiancheng Xing, Jerry Li, Yixuan Du, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06732)  

**Abstract**: Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: this https URL. 

---
# PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs 

**Authors**: Manuel Frank, Haithem Afli  

**Link**: [PDF](https://arxiv.org/pdf/2510.06730)  

**Abstract**: Current evaluations of sentence embedding models typically rely on static test beds such as the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported performance and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in semantic textual similarity gold ratings, we show that LLMs generate token-diverse but semantically preserving, paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs and we extended our experiments to 3 multilingual datasets covering 10 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute. 

---
# Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management 

**Authors**: Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06727)  

**Abstract**: We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit. 

---
# How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects 

**Authors**: Leonardo Bertolazzi, Sandro Pezzelle, Raffaelle Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06700)  

**Abstract**: Both humans and large language models (LLMs) exhibit content effects: biases in which the plausibility of the semantic content of a reasoning problem influences judgments regarding its logical validity. While this phenomenon in humans is best explained by the dual-process theory of reasoning, the mechanisms behind content effects in LLMs remain unclear. In this work, we address this issue by investigating how LLMs encode the concepts of validity and plausibility within their internal representations. We show that both concepts are linearly represented and strongly aligned in representational geometry, leading models to conflate plausibility with validity. Using steering vectors, we demonstrate that plausibility vectors can causally bias validity judgements, and vice versa, and that the degree of alignment between these two concepts predicts the magnitude of behavioral content effects across models. Finally, we construct debiasing vectors that disentangle these concepts, reducing content effects and improving reasoning accuracy. Our findings advance understanding of how abstract logical concepts are represented in LLMs and highlight representational interventions as a path toward more logical systems. 

---
# Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks 

**Authors**: Qinhao Zhou, Xiang Xiang, Kun He, John E. Hopcroft  

**Link**: [PDF](https://arxiv.org/pdf/2510.06695)  

**Abstract**: In recent years, the growing interest in Large Language Models (LLMs) has significantly advanced prompt engineering, transitioning from manual design to model-based optimization. Prompts for LLMs generally comprise two components: the \textit{instruction}, which defines the task or objective, and the \textit{input}, which is tailored to the instruction type. In natural language generation (NLG) tasks such as machine translation, the \textit{input} component is particularly critical, while the \textit{instruction} component tends to be concise. Existing prompt engineering methods primarily focus on optimizing the \textit{instruction} component for general tasks, often requiring large-parameter LLMs as auxiliary tools. However, these approaches exhibit limited applicability for tasks like machine translation, where the \textit{input} component plays a more pivotal role. To address this limitation, this paper introduces a novel prompt optimization method specifically designed for machine translation tasks. The proposed approach employs a small-parameter model trained using a back-translation-based strategy, significantly reducing training overhead for single-task optimization while delivering highly effective performance. With certain adaptations, this method can also be extended to other downstream tasks. 

---
# $λ$-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences 

**Authors**: Yining Wang, Jinman Zhao, Chuangxin Zhao, Shuhao Guan, Gerald Penn, Shinan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06870)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) has been the dominant approach for improving the reasoning capabilities of Large Language Models (LLMs). Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has simplified this paradigm by replacing the reward and value models with rule-based verifiers. A prominent example is Group Relative Policy Optimization (GRPO). However, GRPO inherently suffers from a length bias, since the same advantage is uniformly assigned to all tokens of a response. As a result, longer responses distribute the reward over more tokens and thus contribute disproportionately to gradient updates. Several variants, such as DAPO and Dr. GRPO, modify the token-level aggregation of the loss, yet these methods remain heuristic and offer limited interpretability regarding their implicit token preferences. In this work, we explore the possibility of allowing the model to learn its own token preference during optimization. We unify existing frameworks under a single formulation and introduce a learnable parameter $\lambda$ that adaptively controls token-level weighting. We use $\lambda$-GRPO to denote our method, and we find that $\lambda$-GRPO achieves consistent improvements over vanilla GRPO and DAPO on multiple mathematical reasoning benchmarks. On Qwen2.5 models with 1.5B, 3B, and 7B parameters, $\lambda$-GRPO improves average accuracy by $+1.9\%$, $+1.0\%$, and $+1.7\%$ compared to GRPO, respectively. Importantly, these gains come without any modifications to the training data or additional computational cost, highlighting the effectiveness and practicality of learning token preferences. 

---
# Aligning Large Language Models via Fully Self-Synthetic Data 

**Authors**: Shangjian Yin, Zhepei Wei, Xinyu Zhu, Wei-Lin Chen, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.06652)  

**Abstract**: Traditional reinforcement learning from human feedback (RLHF) for large language models (LLMs) relies on expensive human-annotated datasets, while Reinforcement Learning from AI Feedback (RLAIF) also incurs significant costs, requiring the collection of diverse prompts and corresponding responses, often necessitating external reward models or proprietary models like GPT-4 to annotate preference pairs. In this work, we introduce Self-Alignment Optimization (SAO), a fully self-synthetic framework for LLM alignment, where all training data, including prompts (i.e., user queries), responses, and preferences, are generated by the model itself. Specifically, SAO first instructs the LLM to engage in persona role-play and generate diverse prompts and responses, which are then self-evaluated for preference optimization. Extensive experiments demonstrate that SAO effectively enhances the model's chat capabilities on standard benchmarks like AlpacaEval~2.0, while maintaining strong performance on downstream objective tasks (e.g., question-answering, math reasoning). Our work provides a practical solution for self-improvement in aligning LLMs, and the code for reproducing our results is available at: this https URL. 

---
# Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection? 

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.06594)  

**Abstract**: Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense. 

---
# Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback 

**Authors**: Yisha Wu, Zhao, Yuanpei Cao, Xiaoqing Su, Yashar Mehdad, Mindy Ji, Claire Na Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.06677)  

**Abstract**: We introduce an incremental summarization system for customer support agents that intelligently determines when to generate concise bullet notes during conversations, reducing agents' context-switching effort and redundant review. Our approach combines a fine-tuned Mixtral-8x7B model for continuous note generation with a DeBERTa-based classifier to filter trivial content. Agent edits refine the online notes generation and regularly inform offline model retraining, closing the agent edits feedback loop. Deployed in production, our system achieved a 3% reduction in case handling time compared to bulk summarization (with reductions of up to 9% in highly complex cases), alongside high agent satisfaction ratings from surveys. These results demonstrate that incremental summarization with continuous feedback effectively enhances summary quality and agent productivity at scale. 

---
# TinyScientist: An Interactive, Extensible, and Controllable Framework for Building Research Agents 

**Authors**: Haofei Yu, Keyang Xuan, Fenghai Li, Kunlun Zhu, Zijie Lei, Jiaxun Zhang, Ziheng Qi, Kyle Richardson, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.06579)  

**Abstract**: Automatic research with Large Language Models (LLMs) is rapidly gaining importance, driving the development of increasingly complex workflows involving multi-agent systems, planning, tool usage, code execution, and human-agent interaction to accelerate research processes. However, as more researchers and developers begin to use and build upon these tools and platforms, the complexity and difficulty of extending and maintaining such agentic workflows have become a significant challenge, particularly as algorithms and architectures continue to advance. To address this growing complexity, TinyScientist identifies the essential components of the automatic research workflow and proposes an interactive, extensible, and controllable framework that easily adapts to new tools and supports iterative growth. We provide an open-source codebase, an interactive web demonstration, and a PyPI Python package to make state-of-the-art auto-research pipelines broadly accessible to every researcher and developer. 

---
# Gold-Switch: Training-Free Superposition of Slow- and Fast- Thinking LLMs 

**Authors**: Jaeseong Lee, Dayoung Kwon, seung-won hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06750)  

**Abstract**: Large Reasoning Models (LRMs) excel in structured tasks by emulating deliberate human reasoning but often suffer from overthinking, degrading performance and wasting resources. One possible baseline is to deploy both LLM and LRM, then route input by predicting whether it requires reasoning and may cause overthinking. However, deploying multiple models can be costly or impractical. We propose a superposed deployment strategy with a lightweight, training-free regulation to optimize inference by switching one model on and off. Instead of routing, we selectively unlearn from LRM at inference, scaling down computation while preserving reasoning. By analyzing the cumulative energy of singular values, we identify optimal low-rank projections to adjust reasoning just right. 

---
# MathRobust-LV: Evaluation of Large Language Models' Robustness to Linguistic Variations in Mathematical Reasoning 

**Authors**: Neeraja Kirtane, Yuvraj Khanna, Peter Relan  

**Link**: [PDF](https://arxiv.org/pdf/2510.06430)  

**Abstract**: Large language models excel on math benchmarks, but their math reasoning robustness to linguistic variation is underexplored. While recent work increasingly treats high-difficulty competitions like the IMO as the gold standard for evaluating reasoning, we believe in comprehensive benchmarking of high school-level math problems in real educational settings. We introduce MathRobust-LV, a test set and evaluation methodology that mirrors how instructors rephrase problems across assessments while keeping difficulty constant: we change surface details (names, contexts, variables) while preserving numerical structure and answers. In contrast to prior efforts that alter problem content or emphasize IMO-level tasks, we focus on high-school-level dataset problems at the difficulty level where models are currently deployed in educational settings: tutoring and assessment systems. In these applications, instructors rephrase identical concepts in varied ways, making linguistic robustness essential for reliable deployment. Although MATH data benchmarking is often regarded as saturated, our experiment on 34 models reveals that accuracy declines when moving from the baseline to the variants. These drops are severe for smaller models (9-11%) while stronger models also show measurable degradation. Frontier models like GPT-5, Gemini-2.5pro remain comparatively stable. Our results highlight that robustness to linguistic variation is a fundamental challenge, exposing reasoning vulnerabilities in models. 

---
# TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs 

**Authors**: I-Fan Lin, Faegheh Hasibi, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2510.06747)  

**Abstract**: In this paper, we propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder. In the context of customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these commercial settings, no labeled data is typically available, and the number of clusters is not known. Our method is based on iterative vector updating: it constructs sparse vectors based on representative texts, and then iteratively refines them through LLM guidance. Our method achieves comparable or superior results to state-of-the-art methods that use contrastive learning, but without assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show that our method scales to large datasets, reducing the computational cost of the LLM. These low-resource, adaptable settings and the scalability of our method make it more aligned with real-world scenarios than existing clustering methods. 

---
# FinLFQA: Evaluating Attributed Text Generation of LLMs in Financial Long-Form Question Answering 

**Authors**: Yitao Long, Tiansheng Hu, Yilun Zhao, Arman Cohan, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06426)  

**Abstract**: Large Language Models (LLMs) frequently hallucinate to long-form questions, producing plausible yet factually incorrect answers. A common mitigation strategy is to provide attribution to LLM outputs. However, existing benchmarks primarily focus on simple attribution that retrieves supporting textual evidence as references. We argue that in real-world scenarios such as financial applications, attribution goes beyond reference retrieval. We introduce FinLFQA, a benchmark designed to evaluate the ability of LLMs to generate long-form answers to complex financial questions with reliable and nuanced attributions. FinLFQA evaluates three critical aspects of attribution through human annotations: (1) supporting evidence extracted from financial reports, (2) intermediate numerical reasoning steps, and (3) domain-specific financial knowledge that informs the reasoning process. We further provide an automatic evaluation framework covering both answer quality and attribution quality. Through extensive experiments on eight LLMs across multiple attribution-generation paradigms, we find that fine-grained metrics are important to distinguish model capabilities, that end-to-end generation achieves comparable performance to post-hoc approaches, and that iterative refinement only helps when guided by external feedback. 

---
# PIKA: Expert-Level Synthetic Datasets for Post-Training Alignment from Scratch 

**Authors**: Shangjian Yin, Shining Liang, Wenbiao Ding, Yuli Qian, Zhouxing Shi, Hongzhi Li, Yutao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.06670)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone for aligning large language models (LLMs). However, its effectiveness depends on high-quality instruction data. Most existing alignment datasets are either private or require costly human annotation, which limits reproducibility and scalability. Even with Reinforcement Learning from AI Feedback (RLAIF), concerns about data quality remain. Moreover, it is unclear how much data is actually required to fine-tune a base model into a strong instruction-following model. Current approaches often rely on over 300k examples even at the supervised fine-tuning (SFT) stage, yet they still underperform compared to proprietary models, creating barriers for academic and resource-limited communities. To address this gap, we introduce PiKa, a data-efficient family of expert-level alignment datasets. In particular, the PiKa-SFT dataset uses only 30k SFT examples, far fewer than state-of-the-art datasets like Magpie. Through evaluations by fine-tuning Llama-3-8B-Base on PiKa and other public datasets, we show that PiKa-SFT outperforms models trained on much larger data. On AlpacaEval 2.0 and Arena-Hard benchmarks, PiKa-SFT fine-tuning even surpasses the official Llama-3-8B-Instruct model trained on over 10 million proprietary examples. We further extend our study by training the Qwen2.5 series (0.5B to 7B) on PiKa-SFT, achieving consistent gains. These findings demonstrate that high-quality alignment can be achieved with significantly less data, offering a scalable path for open-source LLM alignment. Code and data: this https URL. 

---
# Instructional Goal-Aligned Question Generation for Student Evaluation in Virtual Lab Settings: How Closely Do LLMs Actually Align? 

**Authors**: R. Alexander Knipper, Indrani Dey, Souvika Sarkar, Hari Narayanan, Sadhana Puntambekar, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2510.06411)  

**Abstract**: Virtual Labs offer valuable opportunities for hands-on, inquiry-based science learning, yet teachers often struggle to adapt them to fit their instructional goals. Third-party materials may not align with classroom needs, and developing custom resources can be time-consuming and difficult to scale. Recent advances in Large Language Models (LLMs) offer a promising avenue for addressing these limitations. In this paper, we introduce a novel alignment framework for instructional goal-aligned question generation, enabling teachers to leverage LLMs to produce simulation-aligned, pedagogically meaningful questions through natural language interaction. The framework integrates four components: instructional goal understanding via teacher-LLM dialogue, lab understanding via knowledge unit and relationship analysis, a question taxonomy for structuring cognitive and pedagogical intent, and the TELeR taxonomy for controlling prompt detail. Early design choices were informed by a small teacher-assisted case study, while our final evaluation analyzed over 1,100 questions from 19 open-source LLMs. With goal and lab understanding grounding questions in teacher intent and simulation context, the question taxonomy elevates cognitive demand (open-ended formats and relational types raise quality by 0.29-0.39 points), and optimized TELeR prompts enhance format adherence (80% parsability, >90% adherence). Larger models yield the strongest gains: parsability +37.1%, adherence +25.7%, and average quality +0.8 Likert points. 

---
# Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language 

**Authors**: Angie Boggust, Donghao Ren, Yannick Assogba, Dominik Moritz, Arvind Satyanarayan, Fred Hohman  

**Link**: [PDF](https://arxiv.org/pdf/2510.06378)  

**Abstract**: Automated interpretability aims to translate large language model (LLM) features into human understandable descriptions. However, these natural language feature descriptions are often vague, inconsistent, and require manual relabeling. In response, we introduce semantic regexes, structured language descriptions of LLM features. By combining primitives that capture linguistic and semantic feature patterns with modifiers for contextualization, composition, and quantification, semantic regexes produce precise and expressive feature descriptions. Across quantitative benchmarks and qualitative analyses, we find that semantic regexes match the accuracy of natural language while yielding more concise and consistent feature descriptions. Moreover, their inherent structure affords new types of analyses, including quantifying feature complexity across layers, scaling automated interpretability from insights into individual features to model-wide patterns. Finally, in user studies, we find that semantic regex descriptions help people build accurate mental models of LLM feature activations. 

---
# LLM Bias Detection and Mitigation through the Lens of Desired Distributions 

**Authors**: Ingroj Shrestha, Padmini Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2510.06354)  

**Abstract**: Although prior work on bias mitigation has focused on promoting social equality and demographic parity, less attention has been given to aligning LLM's outputs to desired distributions. For example, we might want to align a model with real-world distributions to support factual grounding. Thus, we define bias as deviation from a desired distribution, which may be an equal or real-world distribution, depending on application goals. We propose a weighted adaptive loss based fine-tuning method that aligns LLM's gender-profession output distribution with the desired distribution, while preserving language modeling capability. Using 3 profession sets -- male-dominated, female-dominated, and gender-balanced -- derived from U.S. labor statistics (2024), we assess both our adaptive method for reflecting reality and a non-adaptive variant for equality. Across three masked language models, bias is observed under both distributions. We achieve near-complete mitigation under equality and 30-75% reduction under real-world settings. Autoregressive LLMs show no bias under equality but notable bias under real-world settings, with the Llama Instruct models (3.2-3B, 3.1-8B) achieving a 50-62% reduction. 

---
# The Algebra of Meaning: Why Machines Need Montague More Than Moore's Law 

**Authors**: Cheonkam Jeong, Sungdo Kim, Jewoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.06559)  

**Abstract**: Contemporary language models are fluent yet routinely mis-handle the types of meaning their outputs entail. We argue that hallucination, brittle moderation, and opaque compliance outcomes are symptoms of missing type-theoretic semantics rather than data or scale limitations. Building on Montague's view of language as typed, compositional algebra, we recast alignment as a parsing problem: natural-language inputs must be compiled into structures that make explicit their descriptive, normative, and legal dimensions under context.
We present Savassan, a neuro-symbolic architecture that compiles utterances into Montague-style logical forms and maps them to typed ontologies extended with deontic operators and jurisdictional contexts. Neural components extract candidate structures from unstructured inputs; symbolic components perform type checking, constraint reasoning, and cross-jurisdiction mapping to produce compliance-aware guidance rather than binary censorship. In cross-border scenarios, the system "parses once" (e.g., defect claim(product x, company y)) and projects the result into multiple legal ontologies (e.g., defamation risk in KR/JP, protected opinion in US, GDPR checks in EU), composing outcomes into a single, explainable decision.
This paper contributes: (i) a diagnosis of hallucination as a type error; (ii) a formal Montague-ontology bridge for business/legal reasoning; and (iii) a production-oriented design that embeds typed interfaces across the pipeline. We outline an evaluation plan using legal reasoning benchmarks and synthetic multi-jurisdiction suites. Our position is that trustworthy autonomy requires compositional typing of meaning, enabling systems to reason about what is described, what is prescribed, and what incurs liability within a unified algebra of meaning. 

---
# Reproducibility Study of "XRec: Large Language Models for Explainable Recommendation" 

**Authors**: Ranjan Mishra, Julian I. Bibo, Quinten van Engelen, Henk Schaapman  

**Link**: [PDF](https://arxiv.org/pdf/2510.06275)  

**Abstract**: In this study, we reproduced the work done in the paper "XRec: Large Language Models for Explainable Recommendation" by Ma et al. (2024). The original authors introduced XRec, a model-agnostic collaborative instruction-tuning framework that enables large language models (LLMs) to provide users with comprehensive explanations of generated recommendations. Our objective was to replicate the results of the original paper, albeit using Llama 3 as the LLM for evaluation instead of GPT-3.5-turbo. We built on the source code provided by Ma et al. (2024) to achieve our goal. Our work extends the original paper by modifying the input embeddings or deleting the output embeddings of XRec's Mixture of Experts module. Based on our results, XRec effectively generates personalized explanations and its stability is improved by incorporating collaborative information. However, XRec did not consistently outperform all baseline models in every metric. Our extended analysis further highlights the importance of the Mixture of Experts embeddings in shaping the explanation structures, showcasing how collaborative signals interact with language modeling. Through our work, we provide an open-source evaluation implementation that enhances accessibility for researchers and practitioners alike. Our complete code repository can be found at this https URL. 

---
# A Comprehensive Survey of Hallucination in Large Language Models: Causes, Detection, and Mitigation 

**Authors**: Aisha Alansari, Hamzah Luqman  

**Link**: [PDF](https://arxiv.org/pdf/2510.06265)  

**Abstract**: Large language models (LLMs) have transformed natural language processing, achieving remarkable performance across diverse tasks. However, their impressive fluency often comes at the cost of producing false or fabricated information, a phenomenon known as hallucination. Hallucination refers to the generation of content by an LLM that is fluent and syntactically correct but factually inaccurate or unsupported by external evidence. Hallucinations undermine the reliability and trustworthiness of LLMs, especially in domains requiring factual accuracy. This survey provides a comprehensive review of research on hallucination in LLMs, with a focus on causes, detection, and mitigation. We first present a taxonomy of hallucination types and analyze their root causes across the entire LLM development lifecycle, from data collection and architecture design to inference. We further examine how hallucinations emerge in key natural language generation tasks. Building on this foundation, we introduce a structured taxonomy of detection approaches and another taxonomy of mitigation strategies. We also analyze the strengths and limitations of current detection and mitigation approaches and review existing evaluation benchmarks and metrics used to quantify LLMs hallucinations. Finally, we outline key open challenges and promising directions for future research, providing a foundation for the development of more truthful and trustworthy LLMs. 

---
# OpenStaxQA: A multilingual dataset based on open-source college textbooks 

**Authors**: Pranav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.06239)  

**Abstract**: We present OpenStaxQA, an evaluation benchmark specific to college-level educational applications based on 43 open-source college textbooks in English, Spanish, and Polish, available under a permissive Creative Commons license. We finetune and evaluate large language models (LLMs) with approximately 7 billion parameters on this dataset using quantized low rank adapters (QLoRa). Additionally we also perform a zero-shot evaluation on the AI2 reasoning challenge dev dataset in order to check if OpenStaxQA can lead to an improved performance on other tasks. We also discuss broader impacts relevant to datasets such as OpenStaxQA. 

---
# Transparent Reference-free Automated Evaluation of Open-Ended User Survey Responses 

**Authors**: Subin An, Yugyeong Ji, Junyoung Kim, Heejin Kook, Yang Lu, Josh Seltzer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06242)  

**Abstract**: Open-ended survey responses provide valuable insights in marketing research, but low-quality responses not only burden researchers with manual filtering but also risk leading to misleading conclusions, underscoring the need for effective evaluation. Existing automatic evaluation methods target LLM-generated text and inadequately assess human-written responses with their distinct characteristics. To address such characteristics, we propose a two-stage evaluation framework specifically designed for human survey responses. First, gibberish filtering removes nonsensical responses. Then, three dimensions-effort, relevance, and completeness-are evaluated using LLM capabilities, grounded in empirical analysis of real-world survey data. Validation on English and Korean datasets shows that our framework not only outperforms existing metrics but also demonstrates high practical applicability for real-world applications such as response quality prediction and response rejection, showing strong correlations with expert assessment. 

---
# RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning 

**Authors**: Artur Horal, Daniel Pina, Henrique Paz, Iago Paulo, João Soares, Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, João Magalhães, David Semedo  

**Link**: [PDF](https://arxiv.org/pdf/2510.06994)  

**Abstract**: This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness. 

---
# Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces 

**Authors**: Minju Gwak, Guijin Son, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.06953)  

**Abstract**: The Uniform Information Density (UID) hypothesis suggests that effective communication maintains a stable flow of information. In this work, we revisit this principle in the context of large language model (LLM) reasoning traces, asking whether step-level uniformity reflects reasoning quality. To this end, we propose an entropy-based stepwise information density metric and introduce two complementary measures of uniformity, local and global uniformity scores. Across the experiments on six different reasoning benchmarks, we find that step-level uniformity not only provides a strong theoretical lens but also yields practical performance benefits; for example, selecting reasoning traces with more uniform information density at the step-level improves accuracy by 10-32\% relative gains over baselines at AIME2025. Our analysis further reveals that correct reasoning traces tend to avoid sharp information density spikes, while incorrect traces exhibit irregular information bursts. These results demonstrate that UID-inspired information density measures outperform alternative internal signals as predictors of reasoning quality. Results highlight the uniformity of the information density as a robust diagnostic and selection criterion for building more reliable and accurate reasoning systems. 

---
# Protecting De-identified Documents from Search-based Linkage Attacks 

**Authors**: Pierre Lison, Mark Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2510.06383)  

**Abstract**: While de-identification models can help conceal the identity of the individual(s) mentioned in a document, they fail to address linkage risks, defined as the potential to map the de-identified text back to its source. One straightforward way to perform such linkages is to extract phrases from the de-identified document and then check their presence in the original dataset. This paper presents a method to counter search-based linkage attacks while preserving the semantic integrity of the text. The method proceeds in two steps. We first construct an inverted index of the N-grams occurring in the document collection, making it possible to efficiently determine which N-grams appear in less than $k$ documents (either alone or in combination with other N-grams). An LLM-based rewriter is then iteratively queried to reformulate those spans until linkage is no longer possible. Experimental results on a collection of court cases show that the method is able to effectively prevent search-based linkages while remaining faithful to the original content. 

---
# VelLMes: A high-interaction AI-based deception framework 

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.06975)  

**Abstract**: There are very few SotA deception systems based on Large Language Models. The existing ones are limited only to simulating one type of service, mainly SSH shells. These systems - but also the deception technologies not based on LLMs - lack an extensive evaluation that includes human attackers. Generative AI has recently become a valuable asset for cybersecurity researchers and practitioners, and the field of cyber-deception is no exception. Researchers have demonstrated how LLMs can be leveraged to create realistic-looking honeytokens, fake users, and even simulated systems that can be used as honeypots. This paper presents an AI-based deception framework called VelLMes, which can simulate multiple protocols and services such as SSH Linux shell, MySQL, POP3, and HTTP. All of these can be deployed and used as honeypots, thus VelLMes offers a variety of choices for deception design based on the users' needs. VelLMes is designed to be attacked by humans, so interactivity and realism are key for its performance. We evaluate the generative capabilities and the deception capabilities. Generative capabilities were evaluated using unit tests for LLMs. The results of the unit tests show that, with careful prompting, LLMs can produce realistic-looking responses, with some LLMs having a 100% passing rate. In the case of the SSH Linux shell, we evaluated deception capabilities with 89 human attackers. The results showed that about 30% of the attackers thought that they were interacting with a real system when they were assigned an LLM-based honeypot. Lastly, we deployed 10 instances of the SSH Linux shell honeypot on the Internet to capture real-life attacks. Analysis of these attacks showed us that LLM honeypots simulating Linux shells can perform well against unstructured and unexpected attacks on the Internet, responding correctly to most of the issued commands. 

---
# Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG) 

**Authors**: Junki Mori, Kazuya Kakizaki, Taiki Miyagawa, Jun Sakuma  

**Link**: [PDF](https://arxiv.org/pdf/2510.06719)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by grounding them in external knowledge. However, its application in sensitive domains is limited by privacy risks. Existing private RAG methods typically rely on query-time differential privacy (DP), which requires repeated noise injection and leads to accumulated privacy loss. To address this issue, we propose DP-SynRAG, a framework that uses LLMs to generate differentially private synthetic RAG databases. Unlike prior methods, the synthetic text can be reused once created, thereby avoiding repeated noise injection and additional privacy costs. To preserve essential information for downstream RAG tasks, DP-SynRAG extends private prediction, which instructs LLMs to generate text that mimics subsampled database records in a DP manner. Experiments show that DP-SynRAG achieves superior performanec to the state-of-the-art private RAG systems while maintaining a fixed privacy budget, offering a scalable solution for privacy-preserving RAG. 

---
# Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation 

**Authors**: Shuo Shao, Yiming Li, Hongwei Yao, Yifei Chen, Yuchen Yang, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06605)  

**Abstract**: The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods. 

---
# CML-Bench: A Framework for Evaluating and Enhancing LLM-Powered Movie Scripts Generation 

**Authors**: Mingzhe Zheng, Dingjie Song, Guanyu Zhou, Jun You, Jiahao Zhan, Xuran Ma, Xinyuan Song, Ser-Nam Lim, Qifeng Chen, Harry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06231)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in generating highly structured texts. However, while exhibiting a high degree of structural organization, movie scripts demand an additional layer of nuanced storytelling and emotional depth-the 'soul' of compelling cinema-that LLMs often fail to capture. To investigate this deficiency, we first curated CML-Dataset, a dataset comprising (summary, content) pairs for Cinematic Markup Language (CML), where 'content' consists of segments from esteemed, high-quality movie scripts and 'summary' is a concise description of the content. Through an in-depth analysis of the intrinsic multi-shot continuity and narrative structures within these authentic scripts, we identified three pivotal dimensions for quality assessment: Dialogue Coherence (DC), Character Consistency (CC), and Plot Reasonableness (PR). Informed by these findings, we propose the CML-Bench, featuring quantitative metrics across these dimensions. CML-Bench effectively assigns high scores to well-crafted, human-written scripts while concurrently pinpointing the weaknesses in screenplays generated by LLMs. To further validate our benchmark, we introduce CML-Instruction, a prompting strategy with detailed instructions on character dialogue and event logic, to guide LLMs to generate more structured and cinematically sound scripts. Extensive experiments validate the effectiveness of our benchmark and demonstrate that LLMs guided by CML-Instruction generate higher-quality screenplays, with results aligned with human preferences. 

---
# LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations 

**Authors**: Boyuan Long, Yueqi Wang, Hiloni Mehta, Mick Zomnir, Omkar Pathak, Changping Meng, Ruolin Jia, Yajun Peng, Dapeng Hong, Xia Wu, Mingyan Gao, Onkar Dalal, Ningren Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.06657)  

**Abstract**: This paper presents a case study on deploying Large Language Models (LLMs) as an advanced "annotation" mechanism to achieve nuanced content understanding (e.g., discerning content "vibe") at scale within a large-scale industrial short-form video recommendation system. Traditional machine learning classifiers for content understanding face protracted development cycles and a lack of deep, nuanced comprehension. The "LLM-as-annotators" approach addresses these by significantly shortening development times and enabling the annotation of subtle attributes. This work details an end-to-end workflow encompassing: (1) iterative definition and robust evaluation of target attributes, refined by offline metrics and online A/B testing; (2) scalable offline bulk annotation of video corpora using LLMs with multimodal features, optimized inference, and knowledge distillation for broad application; and (3) integration of these rich annotations into the online recommendation serving system, for example, through personalized restrict retrieval. Experimental results demonstrate the efficacy of this approach, with LLMs outperforming human raters in offline annotation quality for nuanced attributes and yielding significant improvements of user participation and satisfied consumption in online A/B tests. The study provides insights into designing and scaling production-level LLM pipelines for rich content evaluation, highlighting the adaptability and benefits of LLM-generated nuanced understanding for enhancing content discovery, user satisfaction, and the overall effectiveness of modern recommendation systems. 

---
# Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks 

**Authors**: Jiaman He, Zikang Leng, Dana McKay, Damiano Spina, Johanne R. Trippas  

**Link**: [PDF](https://arxiv.org/pdf/2510.06658)  

**Abstract**: Many evaluations of large language models (LLMs) in text annotation focus primarily on the correctness of the output, typically comparing model-generated labels to human-annotated ``ground truth'' using standard performance metrics. In contrast, our study moves beyond effectiveness alone. We aim to explore how labeling decisions -- by both humans and LLMs -- can be statistically evaluated across individuals. Rather than treating LLMs purely as annotation systems, we approach LLMs as an alternative annotation mechanism that may be capable of mimicking the subjective judgments made by humans. To assess this, we develop a statistical evaluation method based on Krippendorff's $\alpha$, paired bootstrapping, and the Two One-Sided t-Tests (TOST) equivalence test procedure. This evaluation method tests whether an LLM can blend into a group of human annotators without being distinguishable.
We apply this approach to two datasets -- MovieLens 100K and PolitiFact -- and find that the LLM is statistically indistinguishable from a human annotator in the former ($p = 0.004$), but not in the latter ($p = 0.155$), highlighting task-dependent differences. It also enables early evaluation on a small sample of human data to inform whether LLMs are suitable for large-scale annotation in a given application. 

---
# Ethical AI prompt recommendations in large language models using collaborative filtering 

**Authors**: Jordan Nelson, Almas Baimagambetov, Konstantinos Avgerinakis, Nikolaos Polatidis  

**Link**: [PDF](https://arxiv.org/pdf/2510.06924)  

**Abstract**: As large language models (LLMs) shape AI development, ensuring ethical prompt recommendations is crucial. LLMs offer innovation but risk bias, fairness issues, and accountability concerns. Traditional oversight methods struggle with scalability, necessitating dynamic solutions. This paper proposes using collaborative filtering, a technique from recommendation systems, to enhance ethical prompt selection. By leveraging user interactions, it promotes ethical guidelines while reducing bias. Contributions include a synthetic dataset for prompt recommendations and the application of collaborative filtering. The work also tackles challenges in ethical AI, such as bias mitigation, transparency, and preventing unethical prompt engineering. 

---
# Tool-Augmented Policy Optimization: Synergizing Reasoning and Adaptive Tool Use with Reinforcement Learning 

**Authors**: Wenxun Wu, Yuanyang Li, Guhan Chen, Linyue Wang, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.07038)  

**Abstract**: Recent advances in large language models (LLMs) have popularized test-time scaling, where models generate additional reasoning tokens before producing final answers. These approaches have demonstrated significant performance improvements on benchmarks involving mathematical reasoning. However, language models relying solely on direct inference still struggle with tasks demanding up-to-date knowledge or computational tools such as calculators and code interpreters for complex arithmetic operations. To overcome these limitations, we propose Tool-Augmented Policy Optimization (TAPO), a novel reinforcement learning framework that systematically integrates multi-hop reasoning with adaptive tool-calling capabilities. Our approach employs a modified version of Dynamic Sampling Policy Optimization (DAPO), a recently developed RL paradigm, which we adapt specifically for tool invocation scenarios, enabling models to dynamically interleave complex reasoning with on-demand tool usage (including search APIs and Python interpreters).
To support this research, we introduce two new datasets: TAPO-easy-60K and TAPO-hard-18K, specifically designed to train and evaluate both fact-based reasoning and mathematical calculation capabilities. Our experiments on Qwen2.5-3B and Qwen2.5-7B models demonstrate the effectiveness of our approach, with both models achieving state-of-the-art performance on tasks requiring external knowledge and mathematical computation among methods with comparable parameters. Notably, TAPO achieves more efficient tool utilization than baseline methods while preventing excessive calls caused by reward hacking. These results highlight the significant potential of combining advanced reasoning with tool usage to enhance model performance in knowledge-intensive and computationally demanding tasks. 

---
# Prompt Optimization Across Multiple Agents for Representing Diverse Human Populations 

**Authors**: Manh Hung Nguyen, Sebastian Tschiatschek, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2510.07064)  

**Abstract**: The difficulty and expense of obtaining large-scale human responses make Large Language Models (LLMs) an attractive alternative and a promising proxy for human behavior. However, prior work shows that LLMs often produce homogeneous outputs that fail to capture the rich diversity of human perspectives and behaviors. Thus, rather than trying to capture this diversity with a single LLM agent, we propose a novel framework to construct a set of agents that collectively capture the diversity of a given human population. Each agent is an LLM whose behavior is steered by conditioning on a small set of human demonstrations (task-response pairs) through in-context learning. The central challenge is therefore to select a representative set of LLM agents from the exponentially large space of possible agents. We tackle this selection problem from the lens of submodular optimization. In particular, we develop methods that offer different trade-offs regarding time complexity and performance guarantees. Extensive experiments in crowdsourcing and educational domains demonstrate that our approach constructs agents that more effectively represent human populations compared to baselines. Moreover, behavioral analyses on new tasks show that these agents reproduce the behavior patterns and perspectives of the students and annotators they are designed to represent. 

---
# VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems 

**Authors**: André Hottung, Federico Berto, Chuanbo Hua, Nayeli Gast Zepeda, Daniel Wetzel, Michael Römer, Haoran Ye, Davide Zago, Michael Poli, Stefano Massaroli, Jinkyoo Park, Kevin Tierney  

**Link**: [PDF](https://arxiv.org/pdf/2510.07073)  

**Abstract**: Designing high-performing heuristics for vehicle routing problems (VRPs) is a complex task that requires both intuition and deep domain knowledge. Large language model (LLM)-based code generation has recently shown promise across many domains, but it still falls short of producing heuristics that rival those crafted by human experts. In this paper, we propose VRPAgent, a framework that integrates LLM-generated components into a metaheuristic and refines them through a novel genetic search. By using the LLM to generate problem-specific operators, embedded within a generic metaheuristic framework, VRPAgent keeps tasks manageable, guarantees correctness, and still enables the discovery of novel and powerful strategies. Across multiple problems, including the capacitated VRP, the VRP with time windows, and the prize-collecting VRP, our method discovers heuristic operators that outperform handcrafted methods and recent learning-based approaches while requiring only a single CPU core. To our knowledge, \VRPAgent is the first LLM-based paradigm to advance the state-of-the-art in VRPs, highlighting a promising future for automated heuristics discovery. 

---
# NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents 

**Authors**: Tianshi Zheng, Kelvin Kiu-Wai Tam, Newt Hue-Nam K. Nguyen, Baixuan Xu, Zhaowei Wang, Jiayang Cheng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Tianqing Fang, Yangqiu Song, Ginny Y. Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2510.07172)  

**Abstract**: Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using metaphysical shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery. 

---
# Integrating Domain Knowledge into Process Discovery Using Large Language Models 

**Authors**: Ali Norouzifar, Humam Kourani, Marcus Dees, Wil van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2510.07161)  

**Abstract**: Process discovery aims to derive process models from event logs, providing insights into operational behavior and forming a foundation for conformance checking and process improvement. However, models derived solely from event data may not accurately reflect the real process, as event logs are often incomplete or affected by noise, and domain knowledge, an important complementary resource, is typically disregarded. As a result, the discovered models may lack reliability for downstream tasks. We propose an interactive framework that incorporates domain knowledge, expressed in natural language, into the process discovery pipeline using Large Language Models (LLMs). Our approach leverages LLMs to extract declarative rules from textual descriptions provided by domain experts. These rules are used to guide the IMr discovery algorithm, which recursively constructs process models by combining insights from both the event log and the extracted rules, helping to avoid problematic process structures that contradict domain knowledge. The framework coordinates interactions among the LLM, domain experts, and a set of backend services. We present a fully implemented tool that supports this workflow and conduct an extensive evaluation of multiple LLMs and prompt engineering strategies. Our empirical study includes a case study based on a real-life event log with the involvement of domain experts, who assessed the usability and effectiveness of the framework. 

---
# Autoformalizer with Tool Feedback 

**Authors**: Qi Guo, Jianing Wang, Jianfei Zhang, Deyang Kong, Xiangzhou Huang, Xiangyu Xi, Wei Wang, Jingang Wang, Xunliang Cai, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.06857)  

**Abstract**: Autoformalization addresses the scarcity of data for Automated Theorem Proving (ATP) by translating mathematical problems from natural language into formal statements. Efforts in recent work shift from directly prompting large language models to training an end-to-end formalizer model from scratch, achieving remarkable advancements. However, existing formalizer still struggles to consistently generate valid statements that meet syntactic validity and semantic consistency. To address this issue, we propose the Autoformalizer with Tool Feedback (ATF), a novel approach that incorporates syntactic and consistency information as tools into the formalization process. By integrating Lean 4 compilers for syntax corrections and employing a multi-LLMs-as-judge approach for consistency validation, the model is able to adaptively refine generated statements according to the tool feedback, enhancing both syntactic validity and semantic consistency. The training of ATF involves a cold-start phase on synthetic tool-calling data, an expert iteration phase to improve formalization capabilities, and Direct Preference Optimization to alleviate ineffective revisions. Experimental results show that ATF markedly outperforms a range of baseline formalizer models, with its superior performance further validated by human evaluations. Subsequent analysis reveals that ATF demonstrates excellent inference scaling properties. Moreover, we open-source Numina-ATF, a dataset containing 750K synthetic formal statements to facilitate advancements in autoformalization and ATP research. 

---
# Verifying Memoryless Sequential Decision-making of Large Language Models 

**Authors**: Dennis Gross, Helge Spieker, Arnaud Gotlieb  

**Link**: [PDF](https://arxiv.org/pdf/2510.06756)  

**Abstract**: We introduce a tool for rigorous and automated verification of large language model (LLM)- based policies in memoryless sequential decision-making tasks. Given a Markov decision process (MDP) representing the sequential decision-making task, an LLM policy, and a safety requirement expressed as a PCTL formula, our approach incrementally constructs only the reachable portion of the MDP guided by the LLM's chosen actions. Each state is encoded as a natural language prompt, the LLM's response is parsed into an action, and reachable successor states by the policy are expanded. The resulting formal model is checked with Storm to determine whether the policy satisfies the specified safety property. In experiments on standard grid world benchmarks, we show that open source LLMs accessed via Ollama can be verified when deterministically seeded, but generally underperform deep reinforcement learning baselines. Our tool natively integrates with Ollama and supports PRISM-specified tasks, enabling continuous benchmarking in user-specified sequential decision-making tasks and laying a practical foundation for formally verifying increasingly capable LLMs. 

---
# TGPR: Tree-Guided Policy Refinement for Robust Self-Debugging of LLMs 

**Authors**: Daria Ozerova, Ekaterina Trofimova  

**Link**: [PDF](https://arxiv.org/pdf/2510.06878)  

**Abstract**: Iterative refinement has been a promising paradigm to enable large language models (LLMs) to resolve difficult reasoning and problem-solving tasks. One of the key challenges, however, is how to effectively search through the enormous search space of possible refinements. Existing methods typically fall back on predefined heuristics, which are troubled by the exploration-exploitation dilemma and cannot adapt based on past refinement outcomes. We introduce Tree-Guided Policy Refinement (TGPR), a novel framework that combines GRPO with a Thompson-Sampling-based tree search. TGPR explores both failed and successful refinement paths actively, with denser training trajectories and more adaptive policies. On HumanEval, MBPP, and APPS benchmarks, our method achieves up to +4.2 percentage points absolute improvement in pass@1 (on MBPP) and up to +12.51 percentage points absolute improvement in pass@10 (on APPS) compared to a competitive GRPO baseline. Apart from debugging code, TGPR focuses on a principled approach to combining learned policies with structured search methods, offering a general framework for enhancing iterative refinement and stateful reasoning in LLMs. 

---
# WebDART: Dynamic Decomposition and Re-planning for Complex Web Tasks 

**Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Shiyu Chang, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06587)  

**Abstract**: Large language model (LLM) agents are becoming competent at straightforward web tasks, such as opening an item page or submitting a form, but still struggle with objectives that require long horizon navigation, large scale information extraction, and reasoning under constraints. We present WebDART, a general framework that enables a single LLM to handle such complex chores. WebDART (i) dynamically decomposes each objective into three focused subtasks: navigation, information extraction, and execution, so the model concentrates on one skill at a time, and (ii) continuously replans the decomposition as new webpages are revealed, taking advantage of newly discovered filters or shortcuts and avoiding redundant exploration. Evaluated on WebChoreArena, WebDART lifts success rates by up to 13.7 percentage points over previous SOTA agents, while matching their performance on the easier WebArena suite and completing tasks with up to 14.7 fewer navigation steps. 

---
# Auto-Prompt Ensemble for LLM Judge 

**Authors**: Jiajie Li, Huayi Zhang, Peng Lin, Jinjun Xiong, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06538)  

**Abstract**: We present a novel framework that improves the reliability of LLM judges by selectively augmenting LLM with auxiliary evaluation dimensions. Existing LLM judges often miss crucial evaluation dimensions because they fail to recognize the implicit standards underlying human assessments. To address this challenge, we propose the Auto-Prompt Ensemble (APE), an adaptive framework that automatically learns evaluation dimensions from its failure cases. APE incorporates a confidence-based ensemble mechanism to decide when to adopt the judgments from additional evaluation dimensions through a novel confidence estimation approach called Collective Confidence. Extensive experiments demonstrate that APE improves the reliability of LLM Judge across diverse standard benchmarks. For instance, APE enhances GPT-4o agreement rate on Reward Bench from 87.2% to 90.5% in the zero-shot setting. Overall, APE provides a principled approach for LLM Judge to leverage test-time computation, and bridge the evaluation gap between human and LLM judges. 

---
# Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support 

**Authors**: Zhao, Tiantian Zhang, Hanchen Su, Yufeng, Zhang, Shaowei Su, Mingzhi Xu, Wei Han, Jeremy Werner, Claire Na Cheng, Yashar Mehdad  

**Link**: [PDF](https://arxiv.org/pdf/2510.06674)  

**Abstract**: We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system. 

---
# LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN 

**Authors**: Hacane Hechehouche, Andre Antakli, Matthias Klusch  

**Link**: [PDF](https://arxiv.org/pdf/2510.06911)  

**Abstract**: There are many established semantic Web standards for implementing multi-agent driven applications. The AJAN framework allows to engineer multi-agent systems based on these standards. In particular, agent knowledge is represented in RDF/RDFS and OWL, while agent behavior models are defined with Behavior Trees and SPARQL to access and manipulate this knowledge. However, the appropriate definition of RDF/RDFS and SPARQL-based agent behaviors still remains a major hurdle not only for agent modelers in practice. For example, dealing with URIs is very error-prone regarding typos and dealing with complex SPARQL queries in large-scale environments requires a high learning curve. In this paper, we present an integrated development environment to overcome such hurdles of modeling AJAN agents and at the same time to extend the user community for AJAN by the possibility to leverage Large Language Models for agent engineering. 

---
# Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them 

**Authors**: Jiahe Jin, Abhijay Paladugu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.06534)  

**Abstract**: Agentic search leverages large language models (LLMs) to interpret complex user information needs and execute a multi-step process of planning, searching, and synthesizing information to provide answers. This paradigm introduces unique challenges for LLMs' reasoning and agentic capabilities when interacting with retrieval systems and the broader web. In this paper, we propose a reasoning-driven LLM-based pipeline to study effective reasoning behavior patterns in agentic search. Using this pipeline, we analyze successful agentic search trajectories and identify four beneficial reasoning behaviors: Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery. Based on these findings, we propose a technique called Behavior Priming to train more effective agentic search models. It synthesizes agentic search trajectories that exhibit these four behaviors and integrates them into the agentic search model through supervised fine-tuning (SFT), followed by standard reinforcement learning (RL). Experiments on three benchmarks (GAIA, WebWalker, and HLE) demonstrate that behavior priming yields over 35% gains in Llama3.2-3B and Qwen3-1.7B compared to directly training agentic search models with RL. Crucially, we demonstrate that the desired reasoning behaviors in the SFT data, rather than the correctness of the final answer, is the critical factor for achieving strong final performance after RL: fine-tuning on trajectories with desirable reasoning behaviors but incorrect answers leads to better performance than fine-tuning on trajectories with correct answers. Our analysis further reveals the underlying mechanism: the introduced reasoning behaviors endow models with more effective exploration (higher pass@k and entropy) and test-time scaling (longer trajectories) capabilities, providing a strong foundation for RL. Our code will be released as open source. 

---
# MultiCNKG: Integrating Cognitive Neuroscience, Gene, and Disease Knowledge Graphs Using Large Language Models 

**Authors**: Ali Sarabadani, Kheirolah Rahsepar Fard  

**Link**: [PDF](https://arxiv.org/pdf/2510.06742)  

**Abstract**: The advent of large language models (LLMs) has revolutionized the integration of knowledge graphs (KGs) in biomedical and cognitive sciences, overcoming limitations in traditional machine learning methods for capturing intricate semantic links among genes, diseases, and cognitive processes. We introduce MultiCNKG, an innovative framework that merges three key knowledge sources: the Cognitive Neuroscience Knowledge Graph (CNKG) with 2.9K nodes and 4.3K edges across 9 node types and 20 edge types; Gene Ontology (GO) featuring 43K nodes and 75K edges in 3 node types and 4 edge types; and Disease Ontology (DO) comprising 11.2K nodes and 8.8K edges with 1 node type and 2 edge types. Leveraging LLMs like GPT-4, we conduct entity alignment, semantic similarity computation, and graph augmentation to create a cohesive KG that interconnects genetic mechanisms, neurological disorders, and cognitive functions. The resulting MultiCNKG encompasses 6.9K nodes across 5 types (e.g., Genes, Diseases, Cognitive Processes) and 11.3K edges spanning 7 types (e.g., Causes, Associated with, Regulates), facilitating a multi-layered view from molecular to behavioral domains. Assessments using metrics such as precision (85.20%), recall (87.30%), coverage (92.18%), graph consistency (82.50%), novelty detection (40.28%), and expert validation (89.50%) affirm its robustness and coherence. Link prediction evaluations with models like TransE (MR: 391, MRR: 0.411) and RotatE (MR: 263, MRR: 0.395) show competitive performance against benchmarks like FB15k-237 and WN18RR. This KG advances applications in personalized medicine, cognitive disorder diagnostics, and hypothesis formulation in cognitive neuroscience. 

---
# Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory? 

**Authors**: Aochong Oliver Li, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.06410)  

**Abstract**: Reasoning LLMs are trained to verbalize their reasoning process, yielding strong gains on complex tasks. This transparency also opens a promising direction: multiple reasoners can directly collaborate on each other's thinking within a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is the ability to assess the usefulness and build on another model's partial thinking -- we call this off-trajectory reasoning. Our paper investigates a critical question: can standard solo-reasoning training pipelines deliver desired off-trajectory behaviors? We propose twin tests that capture the two extremes of the off-trajectory spectrum, namely Recoverability, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and Guidability, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B-32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities with solve rates remaining under 9.2%. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that suboptimal recoverability behaviors of teacher models are transferred to distilled students even if the distillation trajectories are correct. Taken together, this work lays the groundwork for evaluating multi-model collaborations in shared reasoning trajectories and highlights the limitations of off-the-shelf reasoning LLMs. 

---
# h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning 

**Authors**: Sumeet Ramesh Motwani, Alesia Ivanova, Ziyang Cai, Philip Torr, Riashat Islam, Shital Shah, Christian Schroeder de Witt, Charles London  

**Link**: [PDF](https://arxiv.org/pdf/2510.07312)  

**Abstract**: Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. Existing approaches to combat this rely on inference-time scaffolding or costly step-level supervision, neither of which scales easily. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade level math problems (GSM8K) boosts accuracy on longer, competition-level benchmarks (GSM-Symbolic, MATH-500, AIME) by up to 2.06x. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn new reasoning paths under RL. Theoretically, we show that curriculum RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training, providing training signal comparable to dense supervision. h1 therefore introduces an efficient path towards scaling RL for long-horizon problems using only existing data. 

---
# Bridging Reasoning to Learning: Unmasking Illusions using Complexity Out of Distribution Generalization 

**Authors**: Mohammad Mahdi Samiei Paqaleh, Arash Marioriyad, Arman Tahmasebi-Zadeh, Mohamadreza Fereydooni, Mahdi Ghaznavai, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2510.06274)  

**Abstract**: Recent progress has pushed AI frontiers from pattern recognition tasks toward problems that require step by step, System2 style reasoning, especially with large language models. Yet, unlike learning, where generalization and out of distribution (OoD) evaluation concepts are well formalized, there is no clear, consistent definition or metric for reasoning ability. We propose Complexity Out of Distribution (Complexity OoD) generalization as a framework and problem setting to define and measure reasoning. A model exhibits Complexity OoD generalization when it maintains performance on test instances whose minimal required solution complexity, either representational (richer solution structure) or computational (more reasoning steps/program length), exceeds that of all training examples. We formalize complexity via solution description Kolmogorov complexity and operational proxies (e.g., object/relation counts; reasoning step counts), clarifying how Complexity OoD differs from length and compositional OoD. This lens unifies learning and reasoning: many cases solvable with System1 like processing at low complexity become System2 like under complexity pressure, while System2 can be viewed as generalization over solution structures. We translate this perspective into practice with recommendations for operationalizing Complexity OoD across the stack: incorporating complexity into benchmark and evaluation metric design, rethinking supervision to target solution traces, seeking and designing inductive biases for Complexity OoD generalization, addressing learning to reason spillovers such as spurious shortcuts, semantic robustness, catastrophic forgetting, and step wise calibration. Because Complexity OoD cannot be solved by scaling data alone, progress toward robust reasoning will require architectures and training regimes that explicitly model and allocate computation with respect to complexity. 

---
# Distilling Lightweight Language Models for C/C++ Vulnerabilities 

**Authors**: Zhiyuan Wei, Xiaoxuan Yang, Jing Sun, Zijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06645)  

**Abstract**: The increasing complexity of modern software systems exacerbates the prevalence of security vulnerabilities, posing risks of severe breaches and substantial economic loss. Consequently, robust code vulnerability detection is essential for software security. While Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing, their potential for automated code vulnerability detection remains underexplored. This paper presents FineSec, a novel framework that harnesses LLMs through knowledge distillation to enable efficient and precise vulnerability identification in C/C++ codebases. FineSec utilizes knowledge distillation to transfer expertise from large teacher models to compact student models, achieving high accuracy with minimal computational cost. By integrating data preparation, training, evaluation, and continuous learning into a unified, single-task workflow, FineSec offers a streamlined approach. Extensive evaluations on C/C++ codebases demonstrate its superiority over both base models and larger LLMs in identifying complex vulnerabilities and logical flaws, establishing FineSec as a practical and scalable solution for real-world software security. To facilitate reproducibility, the datasets, source code, and experimental results are made publicly available at: this https URL. 

---
# AISysRev - LLM-based Tool for Title-abstract Screening 

**Authors**: Aleksi Huotala, Miikka Kuutila, Olli-Pekka Turtio, Mika Mäntylä  

**Link**: [PDF](https://arxiv.org/pdf/2510.06708)  

**Abstract**: Systematic reviews are a standard practice for summarizing the state of evidence in software engineering. Conducting systematic reviews is laborious, especially during the screening or study selection phase, where the number of papers can be overwhelming. During this phase, papers are assessed against inclusion and exclusion criteria based on their titles and abstracts. Recent research has demonstrated that large language models (LLMs) can perform title-abstract screening at a level comparable to that of a master's student. While LLMs cannot be fully trusted, they can help, for example, in Rapid Reviews, which try to expedite the review process. Building on recent research, we developed AiSysRev, an LLM-based screening tool implemented as a web application running in a Docker container. The tool accepts a CSV file containing paper titles and abstracts. Users specify inclusion and exclusion criteria. One can use multiple LLMs for screening via OpenRouter. AiSysRev supports both zero-shot and few-shot screening, and also allows for manual screening through interfaces that display LLM results as guidance for human this http URL conducted a trial study with 137 papers using the tool. Our findings indicate that papers can be classified into four categories: Easy Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary cases, where LLMs are prone to errors, highlight the need for human intervention. While LLMs do not replace human judgment in systematic reviews, they can significantly reduce the burden of assessing large volumes of scientific literature. Video: this https URL Tool: this https URL 

---
# StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering 

**Authors**: Zhihao Wen, Wenkang Wei, Yuan Fang, Xingtong Yu, Hui Zhang, Weicheng Zhu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06638)  

**Abstract**: Knowledge-based Visual Question Answering (KVQA) requires models to ground entities in images and reason over factual knowledge. We study its implicit-knowledge variant, IK-KVQA, where a multimodal large language model (MLLM) is the sole knowledge source, without external retrieval. Yet, MLLMs lack explicit reasoning supervision and produce inconsistent justifications, and generalize poorly after standard supervised fine-tuning (SFT). We present StaR-KVQA (Structured Reasoning Traces for IK-KVQA), which supervises structured traces - dual symbolic relation paths plus path-grounded natural-language explanations - so that reasoning becomes transparent and verifiable. With one open-source MLLM, StaR-KVQA constructs and selects path-grounded reasoning traces to form a trace-enriched dataset, then fine-tunes via structured self-distillation to align generation with supervision; no external retrievers, verifiers, or curated knowledge bases (KBs) are used, traces are built offline, and inference is a single autoregressive pass. Across benchmarks, StaR-KVQA improves both accuracy and interpretability, achieving up to +11.3% higher answer accuracy on OK-VQA over the strongest baseline while exhibiting robust cross-domain generalization. 

---
# Leveraging Large Language Models for Cybersecurity Risk Assessment -- A Case from Forestry Cyber-Physical Systems 

**Authors**: Fikret Mert Gültekin, Oscar Lilja, Ranim Khojah, Rebekka Wohlrab, Marvin Damschen, Mazen Mohamad  

**Link**: [PDF](https://arxiv.org/pdf/2510.06343)  

**Abstract**: In safety-critical software systems, cybersecurity activities become essential, with risk assessment being one of the most critical. In many software teams, cybersecurity experts are either entirely absent or represented by only a small number of specialists. As a result, the workload for these experts becomes high, and software engineers would need to conduct cybersecurity activities themselves. This creates a need for a tool to support cybersecurity experts and engineers in evaluating vulnerabilities and threats during the risk assessment process. This paper explores the potential of leveraging locally hosted large language models (LLMs) with retrieval-augmented generation to support cybersecurity risk assessment in the forestry domain while complying with data protection and privacy requirements that limit external data sharing. We performed a design science study involving 12 experts in interviews, interactive sessions, and a survey within a large-scale project. The results demonstrate that LLMs can assist cybersecurity experts by generating initial risk assessments, identifying threats, and providing redundancy checks. The results also highlight the necessity for human oversight to ensure accuracy and compliance. Despite trust concerns, experts were willing to utilize LLMs in specific evaluation and assistance roles, rather than solely relying on their generative capabilities. This study provides insights that encourage the use of LLM-based agents to support the risk assessment process of cyber-physical systems in safety-critical domains. 

---
# Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data 

**Authors**: Rishabh Ranjan, Valter Hudovernik, Mark Znidar, Charilaos Kanatsoulis, Roshan Upendra, Mahmoud Mohammadi, Joe Meyer, Tom Palczewski, Carlos Guestrin, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2510.06377)  

**Abstract**: Pretrained transformers readily adapt to new sequence modeling tasks via zero-shot prompting, but relational domains still lack architectures that transfer across datasets and tasks. The core challenge is the diversity of relational data, with varying heterogeneous schemas, graph structures and functional dependencies. In this paper, we present the Relational Transformer (RT) architecture, which can be pretrained on diverse relational databases and directly applied to unseen datasets and tasks without task- or dataset-specific fine-tuning, or retrieval of in-context examples. RT (i) tokenizes cells with table/column metadata, (ii) is pretrained via masked token prediction, and (iii) utilizes a novel \textit{Relational Attention} mechanism over columns, rows, and primary-foreign key links. Pretrained on RelBench datasets spanning tasks such as churn and sales forecasting, RT attains strong zero-shot performance, averaging 94% of fully supervised AUROC on binary classification tasks with a single forward pass of a 22M parameter model, as opposed to 84% for a 27B LLM. Fine-tuning yields state-of-the-art results with high sample efficiency. Our experiments show that RT's zero-shot transfer harnesses task-table context, relational attention patterns and schema semantics. Overall, RT provides a practical path toward foundation models for relational data. 

---
# Constrained Natural Language Action Planning for Resilient Embodied Systems 

**Authors**: Grayson Byrd, Corban Rivera, Bethany Kemp, Meghan Booker, Aurora Schmidt, Celso M de Melo, Lalithkumar Seenivasan, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2510.06357)  

**Abstract**: Replicating human-level intelligence in the execution of embodied tasks remains challenging due to the unconstrained nature of real-world environments. Novel use of large language models (LLMs) for task planning seeks to address the previously intractable state/action space of complex planning tasks, but hallucinations limit their reliability, and thus, viability beyond a research context. Additionally, the prompt engineering required to achieve adequate system performance lacks transparency, and thus, repeatability. In contrast to LLM planning, symbolic planning methods offer strong reliability and repeatability guarantees, but struggle to scale to the complexity and ambiguity of real-world tasks. We introduce a new robotic planning method that augments LLM planners with symbolic planning oversight to improve reliability and repeatability, and provide a transparent approach to defining hard constraints with considerably stronger clarity than traditional prompt engineering. Importantly, these augmentations preserve the reasoning capabilities of LLMs and retain impressive generalization in open-world environments. We demonstrate our approach in simulated and real-world environments. On the ALFWorld planning benchmark, our approach outperforms current state-of-the-art methods, achieving a near-perfect 99% success rate. Deployment of our method to a real-world quadruped robot resulted in 100% task success compared to 50% and 30% for pure LLM and symbolic planners across embodied pick and place tasks. Our approach presents an effective strategy to enhance the reliability, repeatability and transparency of LLM-based robot planners while retaining their key strengths: flexibility and generalizability to complex real-world environments. We hope that this work will contribute to the broad goal of building resilient embodied intelligent systems. 

---
# MCCE: A Framework for Multi-LLM Collaborative Co-Evolution 

**Authors**: Nian Ran, Zhongzheng Li, Yue Wang, Qingsong Ran, Xiaoyuan Zhang, Shikun Feng, Richard Allmendinger, Xiaoguang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06270)  

**Abstract**: Multi-objective discrete optimization problems, such as molecular design, pose significant challenges due to their vast and unstructured combinatorial spaces. Traditional evolutionary algorithms often get trapped in local optima, while expert knowledge can provide crucial guidance for accelerating convergence. Large language models (LLMs) offer powerful priors and reasoning ability, making them natural optimizers when expert knowledge matters. However, closed-source LLMs, though strong in exploration, cannot update their parameters and thus cannot internalize experience. Conversely, smaller open models can be continually fine-tuned but lack broad knowledge and reasoning strength. We introduce Multi-LLM Collaborative Co-evolution (MCCE), a hybrid framework that unites a frozen closed-source LLM with a lightweight trainable model. The system maintains a trajectory memory of past search processes; the small model is progressively refined via reinforcement learning, with the two models jointly supporting and complementing each other in global exploration. Unlike model distillation, this process enhances the capabilities of both models through mutual inspiration. Experiments on multi-objective drug design benchmarks show that MCCE achieves state-of-the-art Pareto front quality and consistently outperforms baselines. These results highlight a new paradigm for enabling continual evolution in hybrid LLM systems, combining knowledge-driven exploration with experience-driven learning. 

---
# Dual-stage and Lightweight Patient Chart Summarization for Emergency Physicians 

**Authors**: Jiajun Wu, Swaleh Zaidi, Braden Teitge, Henry Leung, Jiayu Zhou, Jessalyn Holodinsky, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.06263)  

**Abstract**: Electronic health records (EHRs) contain extensive unstructured clinical data that can overwhelm emergency physicians trying to identify critical information. We present a two-stage summarization system that runs entirely on embedded devices, enabling offline clinical summarization while preserving patient privacy. In our approach, a dual-device architecture first retrieves relevant patient record sections using the Jetson Nano-R (Retrieve), then generates a structured summary on another Jetson Nano-S (Summarize), communicating via a lightweight socket link. The summarization output is two-fold: (1) a fixed-format list of critical findings, and (2) a context-specific narrative focused on the clinician's query. The retrieval stage uses locally stored EHRs, splits long notes into semantically coherent sections, and searches for the most relevant sections per query. The generation stage uses a locally hosted small language model (SLM) to produce the summary from the retrieved text, operating within the constraints of two NVIDIA Jetson devices. We first benchmarked six open-source SLMs under 7B parameters to identify viable models. We incorporated an LLM-as-Judge evaluation mechanism to assess summary quality in terms of factual accuracy, completeness, and clarity. Preliminary results on MIMIC-IV and de-identified real EHRs demonstrate that our fully offline system can effectively produce useful summaries in under 30 seconds. 

---
# Ensemble Deep Learning and LLM-Assisted Reporting for Automated Skin Lesion Diagnosis 

**Authors**: Sher Khan, Raz Muhammad, Adil Hussain, Muhammad Sajjad, Muhammad Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2510.06260)  

**Abstract**: Cutaneous malignancies demand early detection for favorable outcomes, yet current diagnostics suffer from inter-observer variability and access disparities. While AI shows promise, existing dermatological systems are limited by homogeneous architectures, dataset biases across skin tones, and fragmented approaches that treat natural language processing as separate post-hoc explanations rather than integral to clinical decision-making. We introduce a unified framework that fundamentally reimagines AI integration for dermatological diagnostics through two synergistic innovations. First, a purposefully heterogeneous ensemble of architecturally diverse convolutional neural networks provides complementary diagnostic perspectives, with an intrinsic uncertainty mechanism flagging discordant cases for specialist review -- mimicking clinical best practices. Second, we embed large language model capabilities directly into the diagnostic workflow, transforming classification outputs into clinically meaningful assessments that simultaneously fulfill medical documentation requirements and deliver patient-centered education. This seamless integration generates structured reports featuring precise lesion characterization, accessible diagnostic reasoning, and actionable monitoring guidance -- empowering patients to recognize early warning signs between visits. By addressing both diagnostic reliability and communication barriers within a single cohesive system, our approach bridges the critical translational gap that has prevented previous AI implementations from achieving clinical impact. The framework represents a significant advancement toward deployable dermatological AI that enhances diagnostic precision while actively supporting the continuum of care from initial detection through patient education, ultimately improving early intervention rates for skin lesions. 

---
# LLM-Driven Rubric-Based Assessment of Algebraic Competence in Multi-Stage Block Coding Tasks with Design and Field Evaluation 

**Authors**: Yong Oh Lee, Byeonghun Bang, Sejun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06253)  

**Abstract**: As online education platforms continue to expand, there is a growing need for assessment methods that not only measure answer accuracy but also capture the depth of students' cognitive processes in alignment with curriculum objectives. This study proposes and evaluates a rubric-based assessment framework powered by a large language model (LLM) for measuring algebraic competence, real-world-context block coding tasks. The problem set, designed by mathematics education experts, aligns each problem segment with five predefined rubric dimensions, enabling the LLM to assess both correctness and quality of students' problem-solving processes. The system was implemented on an online platform that records all intermediate responses and employs the LLM for rubric-aligned achievement evaluation. To examine the practical effectiveness of the proposed framework, we conducted a field study involving 42 middle school students engaged in multi-stage quadratic equation tasks with block coding. The study integrated learner self-assessments and expert ratings to benchmark the system's outputs. The LLM-based rubric evaluation showed strong agreement with expert judgments and consistently produced rubric-aligned, process-oriented feedback. These results demonstrate both the validity and scalability of incorporating LLM-driven rubric assessment into online mathematics and STEM education platforms. 

---
# WeatherArchive-Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather Archives 

**Authors**: Yongan Yu, Xianda Du, Qingchen Hu, Jiahao Liang, Jingwei Ni, Dan Qiang, Kaiyu Huang, Grant McKenzie, Renee Sieber, Fengran Mo  

**Link**: [PDF](https://arxiv.org/pdf/2510.05336)  

**Abstract**: Historical archives on weather events are collections of enduring primary source records that offer rich, untapped narratives of how societies have experienced and responded to extreme weather events. These qualitative accounts provide insights into societal vulnerability and resilience that are largely absent from meteorological records, making them valuable for climate scientists to understand societal responses. However, their vast scale, noisy digitized quality, and archaic language make it difficult to transform them into structured knowledge for climate research. To address this challenge, we introduce WeatherArchive-Bench, the first benchmark for evaluating retrieval-augmented generation (RAG) systems on historical weather archives. WeatherArchive-Bench comprises two tasks: WeatherArchive-Retrieval, which measures a system's ability to locate historically relevant passages from over one million archival news segments, and WeatherArchive-Assessment, which evaluates whether Large Language Models (LLMs) can classify societal vulnerability and resilience indicators from extreme weather narratives. Extensive experiments across sparse, dense, and re-ranking retrievers, as well as a diverse set of LLMs, reveal that dense retrievers often fail on historical terminology, while LLMs frequently misinterpret vulnerability and resilience concepts. These findings highlight key limitations in reasoning about complex societal indicators and provide insights for designing more robust climate-focused RAG systems from archival contexts. The constructed dataset and evaluation framework are publicly available at this https URL. 

---
