# InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning 

**Authors**: Yuchen Yan, Liang Jiang, Jin Jiang, Shuaicheng Li, Zujie Wen, Zhiqiang Zhang, Jun Zhou, Jian Shao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06960)  

**Abstract**: Large reasoning models achieve strong performance by scaling inference-time chain-of-thought, but this paradigm suffers from quadratic cost, context length limits, and degraded reasoning due to lost-in-the-middle effects. Iterative reasoning mitigates these issues by periodically summarizing intermediate thoughts, yet existing methods rely on supervised learning or fixed heuristics and fail to optimize when to summarize, what to preserve, and how to resume reasoning. We propose InftyThink+, an end-to-end reinforcement learning framework that optimizes the entire iterative reasoning trajectory, building on model-controlled iteration boundaries and explicit summarization. InftyThink+ adopts a two-stage training scheme with supervised cold-start followed by trajectory-level reinforcement learning, enabling the model to learn strategic summarization and continuation decisions. Experiments on DeepSeek-R1-Distill-Qwen-1.5B show that InftyThink+ improves accuracy by 21% on AIME24 and outperforms conventional long chain-of-thought reinforcement learning by a clear margin, while also generalizing better to out-of-distribution benchmarks. Moreover, InftyThink+ significantly reduces inference latency and accelerates reinforcement learning training, demonstrating improved reasoning efficiency alongside stronger performance. 

---
# DAWN: Dependency-Aware Fast Inference for Diffusion LLMs 

**Authors**: Lizhuo Luo, Zhuoran Shi, Jiajun Luo, Zhi Wang, Shen Ren, Wenya Wang, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06953)  

**Abstract**: Diffusion large language models (dLLMs) have shown advantages in text generation, particularly due to their inherent ability for parallel decoding. However, constrained by the quality--speed trade-off, existing inference solutions adopt conservative parallel strategies, leaving substantial efficiency potential underexplored. A core challenge is that parallel decoding assumes each position can be filled independently, but tokens are often semantically coupled. Thus, the correct choice at one position constrains valid choices at others. Without modeling these inter-token dependencies, parallel strategies produce deteriorated outputs. Motivated by this insight, we propose DAWN, a training-free, dependency-aware decoding method for fast dLLM inference. DAWN extracts token dependencies and leverages two key motivations: (1) positions dependent on unmasked certain positions become more reliable, (2) simultaneously unmasking strongly coupled uncertain positions induces errors. Given those findings, DAWN leverages a dependency graph to select more reliable unmasking positions at each iteration, achieving high parallelism with negligible loss in generation quality. Extensive experiments across multiple models and datasets demonstrate that DAWN speedups the inference by 1.80-8.06x over baselines while preserving the generation quality. Code is released at this https URL. 

---
# Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2602.06942)  

**Abstract**: Tokenization is a pivotal design choice for neural language modeling in morphologically rich languages (MRLs) such as Turkish, where productive agglutination challenges both vocabulary efficiency and morphological fidelity. Prior studies have explored tokenizer families and vocabulary sizes but typically (i) vary vocabulary without systematically controlling the tokenizer's training corpus, (ii) provide limited intrinsic diagnostics, and (iii) evaluate a narrow slice of downstream tasks. We present the first comprehensive, principled study of Turkish subword tokenization; a "subwords manifest", that jointly varies vocabulary size and tokenizer training corpus size (data and vocabulary coupling), compares multiple tokenizer families under matched parameter budgets (WordPiece, morphology level, and character baselines), and evaluates across semantic (NLI, STS, sentiment analysis, NER), syntactic (POS, dependency parsing), and morphology-sensitive probes. To explain why tokenizers succeed or fail, we introduce a morphology-aware diagnostic toolkit that goes beyond coarse aggregates to boundary-level micro/macro F1, decoupled lemma atomicity vs. surface boundary hits, over/under-segmentation indices, character/word edit distances (CER/WER), continuation rates, and affix-type coverage and token-level atomicity. Our contributions are fourfold: (i) a systematic investigation of the vocabulary-corpus-success triad; (ii) a unified, morphology-aware evaluation framework linking intrinsic diagnostics to extrinsic outcomes; (iii) controlled comparisons identifying when character-level and morphology-level tokenization pay off; and (iv) an open-source release of evaluation code, tokenizer pipelines, and models. As the first work of its kind, this "subwords manifest" delivers actionable guidance for building effective tokenizers in MRLs and establishes a reproducible foundation for future research. 

---
# Halluverse-M^3: A multitask multilingual benchmark for hallucination in LLMs 

**Authors**: Samir Abdaljalil, Parichit Sharma, Erchin Serpedin, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2602.06920)  

**Abstract**: Hallucinations in large language models remain a persistent challenge, particularly in multilingual and generative settings where factual consistency is difficult to maintain. While recent models show strong performance on English-centric benchmarks, their behavior across languages, tasks, and hallucination types is not yet well understood. In this work, we introduce Halluverse-M^3, a dataset designed to enable systematic analysis of hallucinations across multiple languages, multiple generation tasks, and multiple hallucination categories. Halluverse-M^3 covers four languages, English, Arabic, Hindi, and Turkish, and supports two generation tasks: question answering and dialogue summarization. The dataset explicitly distinguishes between entity-level, relation-level, and sentence-level hallucinations. Hallucinated outputs are constructed through a controlled editing process and validated by human annotators, ensuring clear alignment between original content and hallucinated generations. Using this dataset, we evaluate a diverse set of contemporary open-source and proprietary language models on fine-grained hallucination detection. Our results show that question answering is consistently easier than dialogue summarization, while sentence-level hallucinations remain challenging even for the strongest models. Performance is highest in English and degrades in lower-resource languages, with Hindi exhibiting the lowest detection accuracy. Overall, Halluverse-M^3 provides a realistic and challenging benchmark for studying hallucinations in multilingual, multi-task settings. We release the dataset to support future research on hallucination detection and mitigation\footnote{this https URL}. 

---
# Uncovering Cross-Objective Interference in Multi-Objective Alignment 

**Authors**: Yining Lu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06869)  

**Abstract**: We study a persistent failure mode in multi-objective alignment for large language models (LLMs): training improves performance on only a subset of objectives while causing others to degrade. We formalize this phenomenon as cross-objective interference and conduct the first systematic study across classic scalarization algorithms, showing that interference is pervasive and exhibits strong model dependence.
To explain this phenomenon, we derive a local covariance law showing that an objective improves at first order when its reward exhibits positive covariance with the scalarized score. We extend this analysis to clipped surrogate objectives used in modern alignment, demonstrating that the covariance law remains valid under mild conditions despite clipping. Building on this analysis, we propose Covariance Targeted Weight Adaptation (CTWA), a plug-and-play method that maintains positive covariance between objective rewards and the training signal to effectively mitigate cross-objective interference. Finally, we complement these local improvement conditions with a global convergence analysis under the Polyak--Łojasiewicz condition, establishing when non-convex scalarized optimization achieves global convergence and how cross-objective interference depends on specific model geometric properties. 

---
# SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks 

**Authors**: Mingqian Feng, Xiaodong Liu, Weiwei Yang, Jialin Song, Xuekai Zhu, Chenliang Xu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.06854)  

**Abstract**: Multi-turn jailbreaks capture the real threat model for safety-aligned chatbots, where single-turn attacks are merely a special case. Yet existing approaches break under exploration complexity and intent drift. We propose SEMA, a simple yet effective framework that trains a multi-turn attacker without relying on any existing strategies or external data. SEMA comprises two stages. Prefilling self-tuning enables usable rollouts by fine-tuning on non-refusal, well-structured, multi-turn adversarial prompts that are self-generated with a minimal prefix, thereby stabilizing subsequent learning. Reinforcement learning with intent-drift-aware reward trains the attacker to elicit valid multi-turn adversarial prompts while maintaining the same harmful objective. We anchor harmful intent in multi-turn jailbreaks via an intent-drift-aware reward that combines intent alignment, compliance risk, and level of detail. Our open-loop attack regime avoids dependence on victim feedback, unifies single- and multi-turn settings, and reduces exploration complexity. Across multiple datasets, victim models, and jailbreak judges, our method achieves state-of-the-art (SOTA) attack success rates (ASR), outperforming all single-turn baselines, manually scripted and template-driven multi-turn baselines, as well as our SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) variants. For instance, SEMA performs an average $80.1\%$ ASR@1 across three closed-source and open-source victim models on AdvBench, 33.9% over SOTA. The approach is compact, reproducible, and transfers across targets, providing a stronger and more realistic stress test for large language model (LLM) safety and enabling automatic redteaming to expose and localize failure modes. Our code is available at: this https URL. 

---
# The Representational Geometry of Number 

**Authors**: Zhimin Hu, Lanhao Niu, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2602.06843)  

**Abstract**: A central question in cognitive science is whether conceptual representations converge onto a shared manifold to support generalization, or diverge into orthogonal subspaces to minimize task interference. While prior work has discovered evidence for both, a mechanistic account of how these properties coexist and transform across tasks remains elusive. We propose that representational sharing lies not in the concepts themselves, but in the geometric relations between them. Using number concepts as a testbed and language models as high-dimensional computational substrates, we show that number representations preserve a stable relational structure across tasks. Task-specific representations are embedded in distinct subspaces, with low-level features like magnitude and parity encoded along separable linear directions. Crucially, we find that these subspaces are largely transformable into one another via linear mappings, indicating that representations share relational structure despite being located in distinct subspaces. Together, these results provide a mechanistic lens of how language models balance the shared structure of number representation with functional flexibility. It suggests that understanding arises when task-specific transformations are applied to a shared underlying relational structure of conceptual representations. 

---
# Visual Word Sense Disambiguation with CLIP through Dual-Channel Text Prompting and Image Augmentations 

**Authors**: Shamik Bhattacharya, Daniel Perkins, Yaren Dogan, Vineeth Konjeti, Sudarshan Srinivasan, Edmon Begoli  

**Link**: [PDF](https://arxiv.org/pdf/2602.06799)  

**Abstract**: Ambiguity poses persistent challenges in natural language understanding for large language models (LLMs). To better understand how lexical ambiguity can be resolved through the visual domain, we develop an interpretable Visual Word Sense Disambiguation (VWSD) framework. The model leverages CLIP to project ambiguous language and candidate images into a shared multimodal space. We enrich textual embeddings using a dual-channel ensemble of semantic and photo-based prompts with WordNet synonyms, while image embeddings are refined through robust test-time augmentations. We then use cosine similarity to determine the image that best aligns with the ambiguous text. When evaluated on the SemEval-2023 VWSD dataset, enriching the embeddings raises the MRR from 0.7227 to 0.7590 and the Hit Rate from 0.5810 to 0.6220. Ablation studies reveal that dual-channel prompting provides strong, low-latency performance, whereas aggressive image augmentation yields only marginal gains. Additional experiments with WordNet definitions and multilingual prompt ensembles further suggest that noisy external signals tend to dilute semantic specificity, reinforcing the effectiveness of precise, CLIP-aligned prompts for visual word sense disambiguation. 

---
# Generating Data-Driven Reasoning Rubrics for Domain-Adaptive Reward Modeling 

**Authors**: Kate Sanders, Nathaniel Weir, Sapana Chaudhary, Kaj Bostrom, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2602.06795)  

**Abstract**: An impediment to using Large Language Models (LLMs) for reasoning output verification is that LLMs struggle to reliably identify errors in thinking traces, particularly in long outputs, domains requiring expert knowledge, and problems without verifiable rewards. We propose a data-driven approach to automatically construct highly granular reasoning error taxonomies to enhance LLM-driven error detection on unseen reasoning traces. Our findings indicate that classification approaches that leverage these error taxonomies, or "rubrics", demonstrate strong error identification compared to baseline methods in technical domains like coding, math, and chemical engineering. These rubrics can be used to build stronger LLM-as-judge reward functions for reasoning model training via reinforcement learning. Experimental results show that these rewards have the potential to improve models' task accuracy on difficult domains over models trained by general LLMs-as-judges by +45%, and approach performance of models trained by verifiable rewards while using as little as 20% as many gold labels. Through our approach, we extend the usage of reward rubrics from assessing qualitative model behavior to assessing quantitative model correctness on tasks typically learned via RLVR rewards. This extension opens the door for teaching models to solve complex technical problems without a full dataset of gold labels, which are often highly costly to procure. 

---
# R-Align: Enhancing Generative Reward Models through Rationale-Centric Meta-Judging 

**Authors**: Yanlin Lai, Mitt Huang, Hangyu Guo, Xiangfeng Wang, Haodong Li, Shaoxiong Zhan, Liang Zhao, Chengyuan Yao, Yinmin Zhang, Qi Han, Chun Yuan, Zheng Ge, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06763)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) remains indispensable for aligning large language models (LLMs) in subjective domains. To enhance robustness, recent work shifts toward Generative Reward Models (GenRMs) that generate rationales before predicting preferences. Yet in GenRM training and evaluation, practice remains outcome-label-only, leaving reasoning quality unchecked. We show that reasoning fidelity-the consistency between a GenRM's preference decision and reference decision rationales-is highly predictive of downstream RLHF outcomes, beyond standard label accuracy. Specifically, we repurpose existing reward-model benchmarks to compute Spurious Correctness (S-Corr)-the fraction of label-correct decisions with rationales misaligned with golden judgments. Our empirical evaluation reveals substantial S-Corr even for competitive GenRMs, and higher S-Corr is associated with policy degeneration under optimization. To improve fidelity, we propose Rationale-Centric Alignment, R-Align, which augments training with gold judgments and explicitly supervises rationale alignment. R-Align reduces S-Corr on RM benchmarks and yields consistent gains in actor performance across STEM, coding, instruction following, and general tasks. 

---
# Table-as-Search: Formulate Long-Horizon Agentic Information Seeking as Table Completion 

**Authors**: Tian Lan, Felix Henry, Bin Zhu, Qianghuai Jia, Junyang Ren, Qihang Pu, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.06724)  

**Abstract**: Current Information Seeking (InfoSeeking) agents struggle to maintain focus and coherence during long-horizon exploration, as tracking search states, including planning procedure and massive search results, within one plain-text context is inherently fragile. To address this, we introduce \textbf{Table-as-Search (TaS)}, a structured planning framework that reformulates the InfoSeeking task as a Table Completion task. TaS maps each query into a structured table schema maintained in an external database, where rows represent search candidates and columns denote constraints or required information. This table precisely manages the search states: filled cells strictly record the history and search results, while empty cells serve as an explicit search plan. Crucially, TaS unifies three distinct InfoSeeking tasks: Deep Search, Wide Search, and the challenging DeepWide Search. Extensive experiments demonstrate that TaS significantly outperforms numerous state-of-the-art baselines across three kinds of benchmarks, including multi-agent framework and commercial systems. Furthermore, our analysis validates the TaS's superior robustness in long-horizon InfoSeeking, alongside its efficiency, scalability and flexibility. Code and datasets are publicly released at this https URL. 

---
# Evaluating Prompt Engineering Strategies for Sentiment Control in AI-Generated Texts 

**Authors**: Kerstin Sahler, Sophie Jentzsch  

**Link**: [PDF](https://arxiv.org/pdf/2602.06692)  

**Abstract**: The groundbreaking capabilities of Large Language Models (LLMs) offer new opportunities for enhancing human-computer interaction through emotion-adaptive Artificial Intelligence (AI). However, deliberately controlling the sentiment in these systems remains challenging. The present study investigates the potential of prompt engineering for controlling sentiment in LLM-generated text, providing a resource-sensitive and accessible alternative to existing methods. Using Ekman's six basic emotions (e.g., joy, disgust), we examine various prompting techniques, including Zero-Shot and Chain-of-Thought prompting using gpt-3.5-turbo, and compare it to fine-tuning. Our results indicate that prompt engineering effectively steers emotions in AI-generated texts, offering a practical and cost-effective alternative to fine-tuning, especially in data-constrained settings. In this regard, Few-Shot prompting with human-written examples was the most effective among other techniques, likely due to the additional task-specific guidance. The findings contribute valuable insights towards developing emotion-adaptive AI systems. 

---
# compar:IA: The French Government's LLM arena to collect French-language human prompts and preference data 

**Authors**: Lucie Termignon, Simonas Zilinskas, Hadrien Pélissier, Aurélien Barrot, Nicolas Chesnais, Elie Gavoty  

**Link**: [PDF](https://arxiv.org/pdf/2602.06669)  

**Abstract**: Large Language Models (LLMs) often show reduced performance, cultural alignment, and safety robustness in non-English languages, partly because English dominates both pre-training data and human preference alignment datasets. Training methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) require human preference data, which remains scarce and largely non-public for many languages beyond English. To address this gap, we introduce compar:IA, an open-source digital public service developed inside the French government and designed to collect large-scale human preference data from a predominantly French-speaking general audience. The platform uses a blind pairwise comparison interface to capture unconstrained, real-world prompts and user judgments across a diverse set of language models, while maintaining low participation friction and privacy-preserving automated filtering. As of 2026-02-07, compar:IA has collected over 600,000 free-form prompts and 250,000 preference votes, with approximately 89% of the data in French. We release three complementary datasets -- conversations, votes, and reactions -- under open licenses, and present initial analyses, including a French-language model leaderboard and user interaction patterns. Beyond the French context, compar:IA is evolving toward an international digital public good, offering reusable infrastructure for multilingual model training, evaluation, and the study of human-AI interaction. 

---
# Not All Layers Need Tuning: Selective Layer Restoration Recovers Diversity 

**Authors**: Bowen Zhang, Meiyi Wang, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2602.06665)  

**Abstract**: Post-training improves instruction-following and helpfulness of large language models (LLMs) but often reduces generation diversity, which leads to repetitive outputs in open-ended settings, a phenomenon known as mode collapse. Motivated by evidence that LLM layers play distinct functional roles, we hypothesize that mode collapse can be localized to specific layers and that restoring a carefully chosen range of layers to their pre-trained weights can recover diversity while maintaining high output quality. To validate this hypothesis and decide which layers to restore, we design a proxy task -- Constrained Random Character(CRC) -- with an explicit validity set and a natural diversity objective. Results on CRC reveal a clear diversity-validity trade-off across restoration ranges and identify configurations that increase diversity with minimal quality loss. Based on these findings, we propose Selective Layer Restoration (SLR), a training-free method that restores selected layers in a post-trained model to their pre-trained weights, yielding a hybrid model with the same architecture and parameter count, incurring no additional inference cost. Across three different tasks (creative writing, open-ended question answering, and multi-step reasoning) and three different model families (Llama, Qwen, and Gemma), we find SLR can consistently and substantially improve output diversity while maintaining high output quality. 

---
# Beyond Static Alignment: Hierarchical Policy Control for LLM Safety via Risk-Aware Chain-of-Thought 

**Authors**: Jianfeng Si, Lin Sun, Weihong Lin, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06650)  

**Abstract**: Large Language Models (LLMs) face a fundamental safety-helpfulness trade-off due to static, one-size-fits-all safety policies that lack runtime controllabilityxf, making it difficult to tailor responses to diverse application needs. %As a result, models may over-refuse benign requests or under-constrain harmful ones. We present \textbf{PACT} (Prompt-configured Action via Chain-of-Thought), a framework for dynamic safety control through explicit, risk-aware reasoning. PACT operates under a hierarchical policy architecture: a non-overridable global safety policy establishes immutable boundaries for critical risks (e.g., child safety, violent extremism), while user-defined policies can introduce domain-specific (non-global) risk categories and specify label-to-action behaviors to improve utility in real-world deployment settings. The framework decomposes safety decisions into structured Classify$\rightarrow$Act paths that route queries to the appropriate action (comply, guide, or reject) and render the decision-making process transparent.
Extensive experiments demonstrate that PACT achieves near state-of-the-art safety performance under global policy evaluation while attaining the best controllability under user-specific policy evaluation, effectively mitigating the safety-helpfulness trade-off. We will release the PACT model suite, training data, and evaluation protocols to facilitate reproducible research in controllable safety alignment. 

---
# Reading Between the Waves: Robust Topic Segmentation Using Inter-Sentence Audio Features 

**Authors**: Steffen Freisinger, Philipp Seeberger, Tobias Bocklet, Korbinian Riedhammer  

**Link**: [PDF](https://arxiv.org/pdf/2602.06647)  

**Abstract**: Spoken content, such as online videos and podcasts, often spans multiple topics, which makes automatic topic segmentation essential for user navigation and downstream applications. However, current methods do not fully leverage acoustic features, leaving room for improvement. We propose a multi-modal approach that fine-tunes both a text encoder and a Siamese audio encoder, capturing acoustic cues around sentence boundaries. Experiments on a large-scale dataset of YouTube videos show substantial gains over text-only and multi-modal baselines. Our model also proves more resilient to ASR noise and outperforms a larger text-only baseline on three additional datasets in Portuguese, German, and English, underscoring the value of learned acoustic features for robust topic segmentation. 

---
# FairJudge: An Adaptive, Debiased, and Consistent LLM-as-a-Judge 

**Authors**: Bo Yang, Lanfei Feng, Yunkui Chen, Yu Zhang, Xiao Xu, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06625)  

**Abstract**: Existing LLM-as-a-Judge systems suffer from three fundamental limitations: limited adaptivity to task- and domain-specific evaluation criteria, systematic biases driven by non-semantic cues such as position, length, format, and model provenance, and evaluation inconsistency that leads to contradictory judgments across different evaluation modes (e.g., pointwise versus pairwise). To address these issues, we propose FairJudge, an adaptive, debiased, and consistent LLM-as-a-Judge. Unlike prior approaches that treat the judge as a static evaluator, FairJudge models judging behavior itself as a learnable and regularized policy. From a data-centric perspective, we construct a high-information-density judging dataset that explicitly injects supervision signals aligned with evaluation behavior. Building on this dataset, we adopt a curriculum-style SFT-DPO-GRPO training paradigm that progressively aligns rubric adherence, bias mitigation, and cross-mode consistency, while avoiding catastrophic forgetting. Experimental results on multiple internal and public benchmarks show that FairJudge consistently improves agreement and F1, reduces non-semantic biases, and outperforms substantially larger instruction-tuned LLMs. All resources will be publicly released after acceptance to facilitate future research. 

---
# Do Prompts Guarantee Safety? Mitigating Toxicity from LLM Generations through Subspace Intervention 

**Authors**: Himanshu Singh, Ziwei Xu, A. V. Subramanyam, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2602.06623)  

**Abstract**: Large Language Models (LLMs) are powerful text generators, yet they can produce toxic or harmful content even when given seemingly harmless prompts. This presents a serious safety challenge and can cause real-world harm. Toxicity is often subtle and context-dependent, making it difficult to detect at the token level or through coarse sentence-level signals. Moreover, efforts to mitigate toxicity often face a trade-off between safety and the coherence, or fluency of the generated text. In this work, we present a targeted subspace intervention strategy for identifying and suppressing hidden toxic patterns from underlying model representations, while preserving overall ability to generate safe fluent content. On the RealToxicityPrompts, our method achieves strong mitigation performance compared to existing baselines, with minimal impact on inference complexity. Across multiple LLMs, our approach reduces toxicity of state-of-the-art detoxification systems by 8-20%, while maintaining comparable fluency. Through extensive quantitative and qualitative analyses, we show that our approach achieves effective toxicity reduction without impairing generative performance, consistently outperforming existing baselines. 

---
# Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning 

**Authors**: Zhuoyuan Hao, Zhuo Li, Wu Li, Fangming Liu, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06600)  

**Abstract**: Test-time compute allocation in large reasoning models (LRMs) is widely used and has applications in mathematical problem solving, code synthesis, and planning. Recent work has addressed this problem by scaling self-consistency and parallel thinking, adding generic ``thinking tokens'' and prompting models to re-read the question before answering. Unfortunately, these approaches either inject task-agnostic tokens or mandate heuristics that do not explain -- and often ignore -- the \emph{spontaneous} repetition that many LRMs exhibit at the head of their internal chains. In contrast, we analyze and harness the model's tendency to restate the question, which we term the \emph{Echo of Prompt (EOP)}, as a front-loaded, compute-shaping mechanism. We formalize its probabilistic cost by casting echo removal as rejection-based conditioning and defining the \emph{Echo Likelihood Gap} $\Delta\mathcal{L}$ as a computable proxy. This provides the missing theoretical link that links early repetition to likelihood gains and downstream accuracy. However, it does not by itself specify how to exploit EOP. Consequently, we develop \emph{Echo-Distilled SFT (ED-SFT)} to instill an ``echo-then-reason'' pattern through supervised finetuning, and \emph{Echoic Prompting (EP)} to re-ground the model mid-trace without training. While promising, quantifying benefits beyond verbosity is non-trivial. Therefore, we conduct length and suffix-controlled likelihood analyses together with layer-wise attention studies, showing that EOP increases answer to answer-prefix attention in middle layers, consistent with an \emph{attention refocusing} mechanism. We evaluate on GSM8K, MathQA, Hendrycks-MATH, AIME24, and MATH-500 under identical decoding settings and budgets, and find consistent gains over baselines. Code is available at this https URL. 

---
# Inference-Time Rethinking with Latent Thought Vectors for Math Reasoning 

**Authors**: Deqian Kong, Minglu Zhao, Aoyang Qin, Bo Pang, Chenxin Tao, David Hartmann, Edouardo Honig, Dehong Xu, Amit Kumar, Matt Sarte, Chuan Li, Jianwen Xie, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06584)  

**Abstract**: Standard chain-of-thought reasoning generates a solution in a single forward pass, committing irrevocably to each token and lacking a mechanism to recover from early errors. We introduce Inference-Time Rethinking, a generative framework that enables iterative self-correction by decoupling declarative latent thought vectors from procedural generation. We factorize reasoning into a continuous latent thought vector (what to reason about) and a decoder that verbalizes the trace conditioned on this vector (how to reason). Beyond serving as a declarative buffer, latent thought vectors compress the reasoning structure into a continuous representation that abstracts away surface-level token variability, making gradient-based optimization over reasoning strategies well-posed. Our prior model maps unstructured noise to a learned manifold of valid reasoning patterns, and at test time we employ a Gibbs-style procedure that alternates between generating a candidate trace and optimizing the latent vector to better explain that trace, effectively navigating the latent manifold to refine the reasoning strategy. Training a 0.2B-parameter model from scratch on GSM8K, our method with 30 rethinking iterations surpasses baselines with 10 to 15 times more parameters, including a 3B counterpart. This result demonstrates that effective mathematical reasoning can emerge from sophisticated inference-time computation rather than solely from massive parameter counts. 

---
# Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making 

**Authors**: Baichuan-M3 Team, Chengfeng Dou, Fan Yang, Fei Li, Jiyuan Jia, Qiang Ju, Shuai Wang, Tianpeng Li, Xiangrong Zeng, Yijie Zhou, Hongda Zhang, Jinyang Tai, Linzhuang Sun, Peidong Guo, Yichuan Mo, Xiaochuan Wang, Hengfu Cui, Zhishou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06570)  

**Abstract**: We introduce Baichuan-M3, a medical-enhanced large language model engineered to shift the paradigm from passive question-answering to active, clinical-grade decision support. Addressing the limitations of existing systems in open-ended consultations, Baichuan-M3 utilizes a specialized training pipeline to model the systematic workflow of a physician. Key capabilities include: (i) proactive information acquisition to resolve ambiguity; (ii) long-horizon reasoning that unifies scattered evidence into coherent diagnoses; and (iii) adaptive hallucination suppression to ensure factual reliability. Empirical evaluations demonstrate that Baichuan-M3 achieves state-of-the-art results on HealthBench, the newly introduced HealthBench-Hallu and ScanBench, significantly outperforming GPT-5.2 in clinical inquiry, advisory and safety. The models are publicly available at this https URL. 

---
# MTQE.en-he: Machine Translation Quality Estimation for English-Hebrew 

**Authors**: Andy Rosenbaum, Assaf Siani, Ilan Kernerman  

**Link**: [PDF](https://arxiv.org/pdf/2602.06546)  

**Abstract**: We release this http URL-he: to our knowledge, the first publicly available English-Hebrew benchmark for Machine Translation Quality Estimation. this http URL-he contains 959 English segments from WMT24++, each paired with a machine translation into Hebrew, and Direct Assessment scores of the translation quality annotated by three human experts. We benchmark ChatGPT prompting, TransQuest, and CometKiwi and show that ensembling the three models outperforms the best single model (CometKiwi) by 6.4 percentage points Pearson and 5.6 percentage points Spearman. Fine-tuning experiments with TransQuest and CometKiwi reveal that full-model updates are sensitive to overfitting and distribution collapse, yet parameter-efficient methods (LoRA, BitFit, and FTHead, i.e., fine-tuning only the classification head) train stably and yield improvements of 2-3 percentage points. this http URL-he and our experimental results enable future research on this under-resourced language pair. 

---
# Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevant Assessment for IR Benchmarks 

**Authors**: Minjeong Ban, Jeonghwan Choi, Hyangsuk Min, Nicole Hee-Yeon Kim, Minseok Kim, Jae-Gil Lee, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.06526)  

**Abstract**: Information retrieval (IR) evaluation remains challenging due to incomplete IR benchmark datasets that contain unlabeled relevant chunks. While LLMs and LLM-human hybrid strategies reduce costly human effort, they remain prone to LLM overconfidence and ineffective AI-to-human escalation. To address this, we propose DREAM, a multi-round debate-based relevance assessment framework with LLM agents, built on opposing initial stances and iterative reciprocal critique. Through our agreement-based debate, it yields more accurate labeling for certain cases and more reliable AI-to-human escalation for uncertain ones, achieving 95.2% labeling accuracy with only 3.5% human involvement. Using DREAM, we build BRIDGE, a refined benchmark that mitigates evaluation bias and enables fairer retriever comparison by uncovering 29,824 missing relevant chunks. We then re-benchmark IR systems and extend evaluation to RAG, showing that unaddressed holes not only distort retriever rankings but also drive retrieval-generation misalignment. The relevance assessment framework is available at https: //github.com/DISL-Lab/DREAM-ICLR-26; and the BRIDGE dataset is available at this https URL. 

---
# Revisiting the Shape Convention of Transformer Language Models 

**Authors**: Feng-Ting Liao, Meng-Hsi Chen, Guan-Ting Yi, Da-shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06471)  

**Abstract**: Dense Transformer language models have largely adhered to one consistent architectural shape: each layer consists of an attention module followed by a feed-forward network (FFN) with a narrow-wide-narrow MLP, allocating most parameters to the MLP at expansion ratios between 2 and 4. Motivated by recent results that residual wide-narrow-wide (hourglass) MLPs offer superior function approximation capabilities, we revisit the long-standing MLP shape convention in Transformer, challenging the necessity of the narrow-wide-narrow design. To study this, we develop a Transformer variant that replaces the conventional FFN with a deeper hourglass-shaped FFN, comprising a stack of hourglass sub-MLPs connected by residual pathways. We posit that a deeper but lighter hourglass FFN can serve as a competitive alternative to the conventional FFN, and that parameters saved by using a lighter hourglass FFN can be more effectively utilized, such as by enlarging model hidden dimensions under fixed budgets. We confirm these through empirical validations across model scales: hourglass FFNs outperform conventional FFNs up to 400M and achieve comparable performance at larger scales to 1B parameters; hourglass FFN variants with reduced FFN and increased attention parameters show consistent improvements over conventional configurations at matched budgets. Together, these findings shed new light on recent work and prompt a rethinking of the narrow-wide-narrow MLP convention and the balance between attention and FFN towards efficient and expressive modern language models. 

---
# Improve Large Language Model Systems with User Logs 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06470)  

**Abstract**: Scaling training data and model parameters has long driven progress in large language models (LLMs), but this paradigm is increasingly constrained by the scarcity of high-quality data and diminishing returns from rising computational costs. As a result, recent work is increasing the focus on continual learning from real-world deployment, where user interaction logs provide a rich source of authentic human feedback and procedural knowledge. However, learning from user logs is challenging due to their unstructured and noisy nature. Vanilla LLM systems often struggle to distinguish useful feedback signals from noisy user behavior, and the disparity between user log collection and model optimization (e.g., the off-policy optimization problem) further strengthens the problem. To this end, we propose UNO (User log-driveN Optimization), a unified framework for improving LLM systems (LLMsys) with user logs. UNO first distills logs into semi-structured rules and preference pairs, then employs query-and-feedback-driven clustering to manage data heterogeneity, and finally quantifies the cognitive gap between the model's prior knowledge and the log data. This assessment guides the LLMsys to adaptively filter out noisy feedback and construct different modules for primary and reflective experiences extracted from user logs, thereby improving future responses. Extensive experiments show that UNO achieves state-of-the-art effectiveness and efficiency, significantly outperforming Retrieval Augmented Generation (RAG) and memory-based baselines. We have open-sourced our code at this https URL . 

---
# Diffusion-State Policy Optimization for Masked Diffusion Language Models 

**Authors**: Daisuke Oba, Hiroki Furuta, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2602.06462)  

**Abstract**: Masked diffusion language models generate by iteratively filling masked tokens over multiple denoising steps, so learning only from a terminal reward on the final completion yields coarse credit assignment over intermediate decisions. We propose DiSPO (Diffusion-State Policy Optimization), a plug-in credit-assignment layer that directly optimizes intermediate filling decisions. At selected intermediate masked states, DiSPO branches by resampling fillings for the currently masked positions from rollout-cached logits, scores the resulting completions, and updates only the newly filled tokens -- without additional multi-step diffusion rollouts. We formalize a fixed-state objective for branched completions and derive a policy-gradient estimator that can be combined with terminal-feedback policy optimization using the same rollouts. On LLaDA-8B-Instruct, DiSPO consistently improves over the terminal-feedback diffu-GRPO baseline on math and planning benchmarks under matched rollout compute and optimizer steps. Our code will be available at this https URL . 

---
# RelayGen: Intra-Generation Model Switching for Efficient Reasoning 

**Authors**: Jiwon Song, Yoongon Kim, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.06454)  

**Abstract**: Large reasoning models (LRMs) achieve strong performance on complex reasoning tasks by generating long, multi-step reasoning trajectories, but inference-time scaling incurs substantial deployment cost. A key challenge is that generation difficulty varies within a single output, whereas existing efficiency-oriented approaches either ignore this intra-generation variation or rely on supervised token-level routing with high system complexity. We present \textbf{RelayGen}, a training-free, segment-level runtime model switching framework that exploits difficulty variation in long-form reasoning. Through offline analysis of generation uncertainty using token probability margins, we show that coarse-grained segment-level control is sufficient to capture difficulty transitions within a reasoning trajectory. RelayGen identifies model-specific switch cues that signal transitions to lower-difficulty segments and dynamically delegates their continuation to a smaller model, while preserving high-difficulty reasoning on the large model. Across multiple reasoning benchmarks, RelayGen substantially reduces inference latency while preserving most of the accuracy of large models. When combined with speculative decoding, RelayGen achieves up to 2.2$\times$ end-to-end speedup with less than 2\% accuracy degradation, without requiring additional training or learned routing components. 

---
# Evaluating an evidence-guided reinforcement learning framework in aligning light-parameter large language models with decision-making cognition in psychiatric clinical reasoning 

**Authors**: Xinxin Lin, Guangxin Dai, Yi Zhong, Xiang Li, Xue Xiao, Yixin Zhang, Zhengdong Wu, Yongbo Zheng, Runchuan Zhu, Ming Zhao, Huizi Yu, Shuo Wu, Jun Zhao, Lingming Hu, Yumei Wang, Ping Yin, Joey W.Y. Chan, Ngan Yin Chan, Sijing Chen, Yun Kwok Wing, Lin Lu, Xin Ma, Lizhou Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06449)  

**Abstract**: Large language models (LLMs) hold transformative potential for medical decision support yet their application in psychiatry remains constrained by hallucinations and superficial reasoning. This limitation is particularly acute in light-parameter LLMs which are essential for privacy-preserving and efficient clinical deployment. Existing training paradigms prioritize linguistic fluency over structured clinical logic and result in a fundamental misalignment with professional diagnostic cognition. Here we introduce ClinMPO, a reinforcement learning framework designed to align the internal reasoning of LLMs with professional psychiatric practice. The framework employs a specialized reward model trained independently on a dataset derived from 4,474 psychiatry journal articles and structured according to evidence-based medicine principles. We evaluated ClinMPO on a unseen subset of the benchmark designed to isolate reasoning capabilities from rote memorization. This test set comprises items where leading large-parameter LLMs consistently fail. We compared the ClinMPO-aligned light LLM performance against a cohort of 300 medical students. The ClinMPO-tuned Qwen3-8B model achieved a diagnostic accuracy of 31.4% and surpassed the human benchmark of 30.8% on these complex cases. These results demonstrate that medical evidence-guided optimization enables light-parameter LLMs to master complex reasoning tasks. Our findings suggest that explicit cognitive alignment offers a scalable pathway to reliable and safe psychiatric decision support. 

---
# CORE: Comprehensive Ontological Relation Evaluation for Large Language Models 

**Authors**: Satyam Dwivedi, Sanjukta Ghosh, Shivam Dwivedi, Nishi Kumari, Anil Thakur, Anurag Purushottam, Deepak Alok, Praveen Gatla, Manjuprasad B, Bipasha Patgiri  

**Link**: [PDF](https://arxiv.org/pdf/2602.06446)  

**Abstract**: Large Language Models (LLMs) perform well on many reasoning benchmarks, yet existing evaluations rarely assess their ability to distinguish between meaningful semantic relations and genuine unrelatedness. We introduce CORE (Comprehensive Ontological Relation Evaluation), a dataset of 225K multiple-choice questions spanning 74 disciplines, together with a general-domain open-source benchmark of 203 rigorously validated questions (Cohen's Kappa = 1.0) covering 24 semantic relation types with equal representation of unrelated pairs. A human baseline from 1,000+ participants achieves 92.6% accuracy (95.1% on unrelated pairs). In contrast, 29 state-of-the-art LLMs achieve 48.25-70.9% overall accuracy, with near-ceiling performance on related pairs (86.5-100%) but severe degradation on unrelated pairs (0-41.35%), despite assigning similar confidence (92-94%). Expected Calibration Error increases 2-4x on unrelated pairs, and a mean semantic collapse rate of 37.6% indicates systematic generation of spurious relations. On the CORE 225K MCQs dataset, accuracy further drops to approximately 2%, highlighting substantial challenges in domain-specific semantic reasoning. We identify unrelatedness reasoning as a critical, under-evaluated frontier for LLM evaluation and safety. 

---
# TrailBlazer: History-Guided Reinforcement Learning for Black-Box LLM Jailbreaking 

**Authors**: Sung-Hoon Yoon, Ruizhi Qian, Minda Zhao, Weiyue Li, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06440)  

**Abstract**: Large Language Models (LLMs) have become integral to many domains, making their safety a critical priority. Prior jailbreaking research has explored diverse approaches, including prompt optimization, automated red teaming, obfuscation, and reinforcement learning (RL) based methods. However, most existing techniques fail to effectively leverage vulnerabilities revealed in earlier interaction turns, resulting in inefficient and unstable attacks. Since jailbreaking involves sequential interactions in which each response influences future actions, reinforcement learning provides a natural framework for this problem. Motivated by this, we propose a history-aware RL-based jailbreak framework that analyzes and reweights vulnerability signals from prior steps to guide future decisions. We show that incorporating historical information alone improves jailbreak success rates. Building on this insight, we introduce an attention-based reweighting mechanism that highlights critical vulnerabilities within the interaction history, enabling more efficient exploration with fewer queries. Extensive experiments on AdvBench and HarmBench demonstrate that our method achieves state-of-the-art jailbreak performance while significantly improving query efficiency. These results underscore the importance of historical vulnerability signals in reinforcement learning-driven jailbreak strategies and offer a principled pathway for advancing adversarial research on LLM safeguards. 

---
# Investigating the structure of emotions by analyzing similarity and association of emotion words 

**Authors**: Fumitaka Iwaki, Tatsuji Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06430)  

**Abstract**: In the field of natural language processing, some studies have attempted sentiment analysis on text by handling emotions as explanatory or response variables. One of the most popular emotion models used in this context is the wheel of emotion proposed by Plutchik. This model schematizes human emotions in a circular structure, and represents them in two or three dimensions. However, the validity of Plutchik's wheel of emotion has not been sufficiently examined. This study investigated the validity of the wheel by creating and analyzing a semantic networks of emotion words. Through our experiments, we collected data of similarity and association of ordered pairs of emotion words, and constructed networks using these data. We then analyzed the structure of the networks through community detection, and compared it with that of the wheel of emotion. The results showed that each network's structure was, for the most part, similar to that of the wheel of emotion, but locally different. 

---
# On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation 

**Authors**: Wenbo Shang, Yuxi Sun, Jing Ma, Xin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06423)  

**Abstract**: Humor is a commonly used and intricate human language in daily life. Humor generation, especially in multi-modal scenarios, is a challenging task for large language models (LLMs), which is typically as funny caption generation for images, requiring visual understanding, humor reasoning, creative imagination, and so on. Existing LLM-based approaches rely on reasoning chains or self-improvement, which suffer from limited creativity and interpretability. To address these bottlenecks, we develop a novel LLM-based humor generation mechanism based on a fundamental humor theory, GTVH. To produce funny and script-opposite captions, we introduce a humor-theory-driven multi-role LLM collaboration framework augmented with humor retrieval (HOMER). The framework consists of three LLM-based roles: (1) conflicting-script extractor that grounds humor in key script oppositions, forming the basis of caption generation; (2) retrieval-augmented hierarchical imaginator that identifies key humor targets and expands the creative space of them through diverse associations structured as imagination trees; and (3) caption generator that produces funny and diverse captions conditioned on the obtained knowledge. Extensive experiments on two New Yorker Cartoon benchmarking datasets show that HOMER outperforms state-of-the-art baselines and powerful LLM reasoning strategies on multi-modal humor captioning. 

---
# Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding 

**Authors**: Daisuke Oba, Danushka Bollegala, Masahiro Kaneko, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2602.06412)  

**Abstract**: Masked Diffusion Language Models generate sequences via iterative sampling that progressively unmasks tokens. However, they still recompute the attention and feed-forward blocks for every token position at every step -- even when many unmasked tokens are essentially fixed, resulting in substantial waste in compute. We propose SureLock: when the posterior at an unmasked position has stabilized across steps (our sure condition), we lock that position -- thereafter skipping its query projection and feed-forward sublayers -- while caching its attention keys and values so other positions can continue to attend to it. This reduces the dominant per-iteration computational cost from $O(N^2d)$ to $O(MNd)$ where $N$ is the sequence length, $M$ is the number of unlocked token positions, and $d$ is the model dimension. In practice, $M$ decreases as the iteration progresses, yielding substantial savings. On LLaDA-8B, SureLock reduces algorithmic FLOPs by 30--50% relative to the same sampler without locking, while maintaining comparable generation quality. We also provide a theoretical analysis to justify the design rationale of SureLock: monitoring only the local KL at the lock step suffices to bound the deviation in final token probabilities. Our code will be available at this https URL . 

---
# FMBench: Adaptive Large Language Model Output Formatting 

**Authors**: Yaoting Wang, Yun Zhou, Henghui Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.06384)  

**Abstract**: Producing outputs that satisfy both semantic intent and format constraints is essential for deploying large language models in user-facing and system-integrated workflows. In this work, we focus on Markdown formatting, which is ubiquitous in assistants, documentation, and tool-augmented pipelines but still prone to subtle, hard-to-detect errors (e.g., broken lists, malformed tables, inconsistent headings, and invalid code blocks) that can significantly degrade downstream usability. We present FMBench, a benchmark for adaptive Markdown output formatting that evaluates models under a wide range of instruction-following scenarios with diverse structural requirements. FMBench emphasizes real-world formatting behaviors such as multi-level organization, mixed content (natural language interleaved with lists/tables/code), and strict adherence to user-specified layout constraints. To improve Markdown compliance without relying on hard decoding constraints, we propose a lightweight alignment pipeline that combines supervised fine-tuning (SFT) with reinforcement learning fine-tuning. Starting from a base model, we first perform SFT on instruction-response pairs, and then optimize a composite objective that balances semantic fidelity with structural correctness. Experiments on two model families (OpenPangu and Qwen) show that SFT consistently improves semantic alignment, while reinforcement learning provides additional gains in robustness to challenging Markdown instructions when initialized from a strong SFT policy. Our results also reveal an inherent trade-off between semantic and structural objectives, highlighting the importance of carefully designed rewards for reliable formatted generation. Code is available at: this https URL. 

---
# ReBeCA: Unveiling Interpretable Behavior Hierarchy behind the Iterative Self-Reflection of Language Models with Causal Analysis 

**Authors**: Tianqiang Yan, Sihan Shang, Yuheng Li, Song Qiu, Hao Peng, Wenjian Luo, Jue Xie, Lizhen Qu, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.06373)  

**Abstract**: While self-reflection can enhance language model reliability, its underlying mechanisms remain opaque, with existing analyses often yielding correlation-based insights that fail to generalize. To address this, we introduce \textbf{\texttt{ReBeCA}} (self-\textbf{\texttt{Re}}flection \textbf{\texttt{Be}}havior explained through \textbf{\texttt{C}}ausal \textbf{\texttt{A}}nalysis), a framework that unveils the interpretable behavioral hierarchy governing the self-reflection outcome. By modeling self-reflection trajectories as causal graphs, ReBeCA isolates genuine determinants of performance through a three-stage Invariant Causal Prediction (ICP) pipeline. We establish three critical findings: (1) \textbf{Behavioral hierarchy:} Semantic behaviors of the model influence final self-reflection results hierarchically: directly or indirectly; (2) \textbf{Causation matters:} Generalizability in self-reflection effects is limited to just a few semantic behaviors; (3) \textbf{More $\mathbf{\neq}$ better:} The confluence of seemingly positive semantic behaviors, even among direct causal factors, can impair the efficacy of self-reflection. ICP-based verification identifies sparse causal parents achieving up to $49.6\%$ structural likelihood gains, stable across tasks where correlation-based patterns fail. Intervention studies on novel datasets confirm these causal relationships hold out-of-distribution ($p = .013, \eta^2_\mathrm{p} = .071$). ReBeCA thus provides a rigorous methodology for disentangling genuine causal mechanisms from spurious associations in self-reflection dynamics. 

---
# Cost-Aware Model Selection for Text Classification: Multi-Objective Trade-offs Between Fine-Tuned Encoders and LLM Prompting in Production 

**Authors**: Alberto Andres Valdes Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2602.06370)  

**Abstract**: Large language models (LLMs) such as GPT-4o and Claude Sonnet 4.5 have demonstrated strong capabilities in open-ended reasoning and generative language tasks, leading to their widespread adoption across a broad range of NLP applications. However, for structured text classification problems with fixed label spaces, model selection is often driven by predictive performance alone, overlooking operational constraints encountered in production systems.
In this work, we present a systematic comparison of two contrasting paradigms for text classification: zero- and few-shot prompt-based large language models, and fully fine-tuned encoder-only architectures. We evaluate these approaches across four canonical benchmarks (IMDB, SST-2, AG News, and DBPedia), measuring predictive quality (macro F1), inference latency, and monetary cost.
We frame model evaluation as a multi-objective decision problem and analyze trade-offs using Pareto frontier projections and a parameterized utility function reflecting different deployment regimes. Our results show that fine-tuned encoder-based models from the BERT family achieve competitive, and often superior, classification performance while operating at one to two orders of magnitude lower cost and latency compared to zero- and few-shot LLM prompting.
Overall, our findings suggest that indiscriminate use of large language models for standard text classification workloads can lead to suboptimal system-level outcomes. Instead, fine-tuned encoders emerge as robust and efficient components for structured NLP pipelines, while LLMs are better positioned as complementary elements within hybrid architectures. We release all code, datasets, and evaluation protocols to support reproducibility and cost-aware NLP system design. 

---
# SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass 

**Authors**: Yewei Liu, Xiyuan Wang, Yansheng Mao, Yoav Gelbery, Haggai Maron, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06358)  

**Abstract**: We propose SHINE (Scalable Hyper In-context NEtwork), a scalable hypernetwork that can map diverse meaningful contexts into high-quality LoRA adapters for large language models (LLM). By reusing the frozen LLM's own parameters in an in-context hypernetwork design and introducing architectural innovations, SHINE overcomes key limitations of prior hypernetworks and achieves strong expressive power with a relatively small number of parameters. We introduce a pretraining and instruction fine-tuning pipeline, and train our hypernetwork to generate high quality LoRA adapters from diverse meaningful contexts in a single forward pass. It updates LLM parameters without any fine-tuning, and immediately enables complex question answering tasks related to the context without directly accessing the context, effectively transforming in-context knowledge to in-parameter knowledge in one pass. Our work achieves outstanding results on various tasks, greatly saves time, computation and memory costs compared to SFT-based LLM adaptation, and shows great potential for scaling. Our code is available at this https URL 

---
# Can Post-Training Transform LLMs into Causal Reasoners? 

**Authors**: Junqi Chen, Sirui Chen, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06337)  

**Abstract**: Causal inference is essential for decision-making but remains challenging for non-experts. While large language models (LLMs) show promise in this domain, their precise causal estimation capabilities are still limited, and the impact of post-training on these abilities is insufficiently explored. This paper examines the extent to which post-training can enhance LLMs' capacity for causal inference. We introduce CauGym, a comprehensive dataset comprising seven core causal tasks for training and five diverse test sets. Using this dataset, we systematically evaluate five post-training approaches: SFT, DPO, KTO, PPO, and GRPO. Across five in-domain and four existing benchmarks, our experiments demonstrate that appropriate post-training enables smaller LLMs to perform causal inference competitively, often surpassing much larger models. Our 14B parameter model achieves 93.5% accuracy on the CaLM benchmark, compared to 55.4% by OpenAI o3. Furthermore, the post-trained LLMs exhibit strong generalization and robustness under real-world conditions such as distribution shifts and noisy data. Collectively, these findings provide the first systematic evidence that targeted post-training can produce reliable and robust LLM-based causal reasoners. Our data and GRPO-model are available at this https URL. 

---
# Lost in Speech: Benchmarking, Evaluation, and Parsing of Spoken Code-Switching Beyond Standard UD Assumptions 

**Authors**: Nemika Tyagi, Holly Hendrix, Nelvin Licona-Guevara, Justin Mackie, Phanos Kareen, Muhammad Imran, Megan Michelle Smith, Tatiana Gallego Hernande, Chitta Baral, Olga Kellert  

**Link**: [PDF](https://arxiv.org/pdf/2602.06307)  

**Abstract**: Spoken code-switching (CSW) challenges syntactic parsing in ways not observed in written text. Disfluencies, repetition, ellipsis, and discourse-driven structure routinely violate standard Universal Dependencies (UD) assumptions, causing parsers and large language models (LLMs) to fail despite strong performance on written data. These failures are compounded by rigid evaluation metrics that conflate genuine structural errors with acceptable variation. In this work, we present a systems-oriented approach to spoken CSW parsing. We introduce a linguistically grounded taxonomy of spoken CSW phenomena and SpokeBench, an expert-annotated gold benchmark designed to test spoken-language structure beyond standard UD assumptions. We further propose FLEX-UD, an ambiguity-aware evaluation metric, which reveals that existing parsing techniques perform poorly on spoken CSW by penalizing linguistically plausible analyses as errors. We then propose DECAP, a decoupled agentic parsing framework that isolates spoken-phenomena handling from core syntactic analysis. Experiments show that DECAP produces more robust and interpretable parses without retraining and achieves up to 52.6% improvements over existing parsing techniques. FLEX-UD evaluations further reveal qualitative improvements that are masked by standard metrics. 

---
# Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math 

**Authors**: Guijin Son, Donghun Yang, Hitesh Laxmichand Patel, Hyunwoo Ko, Amit Agarwal, Sunghee Ahn, Kyong-Ha Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06291)  

**Abstract**: Recent progress in reasoning models suggests that generating plausible attempts for research-level mathematics may be within reach, but verification remains a bottleneck, consuming scarce expert time. We hypothesize that a meaningful solution should contain enough method-level information that, when applied to a neighborhood of related questions, it should yield better downstream performance than incorrect solutions. Building on this idea, we propose \textbf{Consequence-Based Utility}, an oracle-free evaluator that scores each candidate by testing its value as an in-context exemplar in solving related yet verifiable questions. Our approach is evaluated on an original set of research-level math problems, each paired with one expert-written solution and nine LLM-generated solutions. Notably, Consequence-Based Utility consistently outperforms reward models, generative reward models, and LLM judges on ranking quality. Specifically, for GPT-OSS-120B, it improves Acc@1 from 67.2 to 76.3 and AUC from 71.4 to 79.6, with similarly large AUC gains on GPT-OSS-20B (69.0 to 79.2). Furthermore, compared to LLM-Judges, it also exhibits a larger solver-evaluator gap, maintaining a stronger correct-wrong separation even on instances where the underlying solver often fails to solve. 

---
# RoPE-LIME: RoPE-Space Locality + Sparse-K Sampling for Efficient LLM Attribution 

**Authors**: Isaac Picov, Ritesh Goru  

**Link**: [PDF](https://arxiv.org/pdf/2602.06275)  

**Abstract**: Explaining closed-source LLM outputs is challenging because API access prevents gradient-based attribution, while perturbation methods are costly and noisy when they depend on regenerated text. We introduce RoPE-LIME, an open-source extension of gSMILE that decouples reasoning from explanation: given a fixed output from a closed model, a smaller open-source surrogate computes token-level attributions from probability-based objectives (negative log-likelihood and divergence targets) under input perturbations. RoPE-LIME incorporates (i) a locality kernel based on Relaxed Word Mover's Distance computed in RoPE embedding space for stable similarity under masking, and (ii) Sparse-K sampling, an efficient perturbation strategy that improves interaction coverage under limited budgets. Experiments on HotpotQA (sentence features) and a hand-labeled MMLU subset (word features) show that RoPE-LIME produces more informative attributions than leave-one-out sampling and improves over gSMILE while substantially reducing closed-model API calls. 

---
# VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation 

**Authors**: Yancheng Wang, Osama Hanna, Ruiming Xie, Xianfeng Rui, Maohao Shen, Xuedong Zhang, Christian Fuegen, Jilong Wu, Debjyoti Paul, Arthur Guo, Zhihong Lei, Ozlem Kalinli, Qing He, Yingzhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06270)  

**Abstract**: Emotion recognition in speech presents a complex multimodal challenge, requiring comprehension of both linguistic content and vocal expressivity, particularly prosodic features such as fundamental frequency, intensity, and temporal dynamics. Although large language models (LLMs) have shown promise in reasoning over textual transcriptions for emotion recognition, they typically neglect fine-grained prosodic information, limiting their effectiveness and interpretability. In this work, we propose VowelPrompt, a linguistically grounded framework that augments LLM-based emotion recognition with interpretable, fine-grained vowel-level prosodic cues. Drawing on phonetic evidence that vowels serve as primary carriers of affective prosody, VowelPrompt extracts pitch-, energy-, and duration-based descriptors from time-aligned vowel segments, and converts these features into natural language descriptions for better interpretability. Such a design enables LLMs to jointly reason over lexical semantics and fine-grained prosodic variation. Moreover, we adopt a two-stage adaptation procedure comprising supervised fine-tuning (SFT) followed by Reinforcement Learning with Verifiable Reward (RLVR), implemented via Group Relative Policy Optimization (GRPO), to enhance reasoning capability, enforce structured output adherence, and improve generalization across domains and speaker variations. Extensive evaluations across diverse benchmark datasets demonstrate that VowelPrompt consistently outperforms state-of-the-art emotion recognition methods under zero-shot, fine-tuned, cross-domain, and cross-linguistic conditions, while enabling the generation of interpretable explanations that are jointly grounded in contextual semantics and fine-grained prosodic structure. 

---
# MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs 

**Authors**: Junhyeok Lee, Han Jang, Kyu Sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06268)  

**Abstract**: Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly integrated into clinical workflows; however, prompt injection attacks can steer these systems toward clinically unsafe or misleading outputs. We introduce the Medical Prompt Injection Benchmark (MPIB), a dataset-and-benchmark suite for evaluating clinical safety under both direct prompt injection and indirect, RAG-mediated injection across clinically grounded tasks. MPIB emphasizes outcome-level risk via the Clinical Harm Event Rate (CHER), which measures high-severity clinical harm events under a clinically grounded taxonomy, and reports CHER alongside Attack Success Rate (ASR) to disentangle instruction compliance from downstream patient risk. The benchmark comprises 9,697 curated instances constructed through multi-stage quality gates and clinical safety linting. Evaluating MPIB across a diverse set of baseline LLMs and defense configurations, we find that ASR and CHER can diverge substantially, and that robustness depends critically on whether adversarial instructions appear in the user query or in retrieved context. We release MPIB with evaluation code, adversarial baselines, and comprehensive documentation to support reproducible and systematic research on clinical prompt injection. Code and data are available at GitHub (code) and Hugging Face (data). 

---
# Is my model "mind blurting"? Interpreting the dynamics of reasoning tokens with Recurrence Quantification Analysis (RQA) 

**Authors**: Quoc Tuan Pham, Mehdi Jafari, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2602.06266)  

**Abstract**: Test-time compute is central to large reasoning models, yet analysing their reasoning behaviour through generated text is increasingly impractical and unreliable. Response length is often used as a brute proxy for reasoning effort, but this metric fails to capture the dynamics and effectiveness of the Chain of Thoughts (CoT) or the generated tokens. We propose Recurrence Quantification Analysis (RQA) as a non-textual alternative for analysing model's reasoning chains at test time. By treating token generation as a dynamical system, we extract hidden embeddings at each generation step and apply RQA to the resulting trajectories. RQA metrics, including Determinism and Laminarity, quantify patterns of repetition and stalling in the model's latent representations. Analysing 3,600 generation traces from DeepSeek-R1-Distill, we show that RQA captures signals not reflected by response length, but also substantially improves prediction of task complexity by 8\%. These results help establish RQA as a principled tool for studying the latent token generation dynamics of test-time scaling in reasoning models. 

---
# Can One-sided Arguments Lead to Response Change in Large Language Models? 

**Authors**: Pedro Cisneros-Velarde  

**Link**: [PDF](https://arxiv.org/pdf/2602.06260)  

**Abstract**: Polemic questions need more than one viewpoint to express a balanced answer. Large Language Models (LLMs) can provide a balanced answer, but also take a single aligned viewpoint or refuse to answer. In this paper, we study if such initial responses can be steered to a specific viewpoint in a simple and intuitive way: by only providing one-sided arguments supporting the viewpoint. Our systematic study has three dimensions: (i) which stance is induced in the LLM response, (ii) how the polemic question is formulated, (iii) how the arguments are shown. We construct a small dataset and remarkably find that opinion steering occurs across (i)-(iii) for diverse models, number of arguments, and topics. Switching to other arguments consistently decreases opinion steering. 

---
# BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks 

**Authors**: Nishant Balepur, Bhavya Rajasekaran, Jane Oh, Michael Xie, Atrey Desai, Vipul Gupta, Steven James Moore, Eunsol Choi, Rachel Rudinger, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2602.06221)  

**Abstract**: Multiple-choice question answering (MCQA) is standard in NLP, but benchmarks lack rigorous quality control. We present BenchMarker, an education-inspired toolkit using LLM judges to flag three common MCQ flaws: 1) contamination - items appearing exactly online; 2) shortcuts - cues in the choices that enable guessing; and 3) writing errors - structural/grammatical issues based on a 19-rule education rubric. We validate BenchMarker with human annotations, then run the tool to audit 12 benchmarks, revealing: 2) contaminated MCQs tend to inflate accuracy, while writing errors tend to lower it and change rankings beyond random; and 3) prior benchmark repairs address their targeted issues (i.e., lowering accuracy with LLM-written distractors), but inadvertently add new flaws (i.e. implausible distractors, many correct answers). Overall, flaws in MCQs degrade NLP evaluation, but education research offers a path forward. We release BenchMarker to bridge the fields and improve MCQA benchmark design. 

---
# Uncertainty Drives Social Bias Changes in Quantized Large Language Models 

**Authors**: Stanley Z. Hua, Sanae Lotfi, Irene Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06181)  

**Abstract**: Post-training quantization reduces the computational cost of large language models but fundamentally alters their social biases in ways that aggregate metrics fail to capture. We present the first large-scale study of 50 quantized models evaluated on PostTrainingBiasBench, a unified benchmark of 13 closed- and open-ended bias datasets. We identify a phenomenon we term quantization-induced masked bias flipping, in which up to 21% of responses flip between biased and unbiased states after quantization, despite showing no change in aggregate bias scores. These flips are strongly driven by model uncertainty, where the responses with high uncertainty are 3-11x more likely to change than the confident ones. Quantization strength amplifies this effect, with 4-bit quantized models exhibiting 4-6x more behavioral changes than 8-bit quantized models. Critically, these changes create asymmetric impacts across demographic groups, where bias can worsen by up to 18.6% for some groups while improving by 14.1% for others, yielding misleadingly neutral aggregate outcomes. Larger models show no consistent robustness advantage, and group-specific shifts vary unpredictably across model families. Our findings demonstrate that compression fundamentally alters bias patterns, requiring crucial post-quantization evaluation and interventions to ensure reliability in practice. 

---
# Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding 

**Authors**: Yanzheng Xiang, Lan Wei, Yizhen Yao, Qinglin Zhu, Hanqi Yan, Chen Jin, Philip Alexander Teare, Dandan Zhang, Lin Gui, Amrutha Saseendran, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2602.06161)  

**Abstract**: Parallel diffusion decoding can accelerate diffusion language model inference by unmasking multiple tokens per step, but aggressive parallelism often harms quality. Revocable decoding mitigates this by rechecking earlier tokens, yet we observe that existing verification schemes frequently trigger flip-flop oscillations, where tokens are remasked and later restored unchanged. This behaviour slows inference in two ways: remasking verified positions weakens the conditioning context for parallel drafting, and repeated remask cycles consume the revision budget with little net progress. We propose COVER (Cache Override Verification for Efficient Revision), which performs leave-one-out verification and stable drafting within a single forward pass. COVER constructs two attention views via KV cache override: selected seeds are masked for verification, while their cached key value states are injected for all other queries to preserve contextual information, with a closed form diagonal correction preventing self leakage at the seed positions. COVER further prioritises seeds using a stability aware score that balances uncertainty, downstream influence, and cache drift, and it adapts the number of verified seeds per step. Across benchmarks, COVER markedly reduces unnecessary revisions and yields faster decoding while preserving output quality. 

---
# Quantifying and Attributing Polarization to Annotator Groups 

**Authors**: Dimitris Tsirmpas, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2602.06055)  

**Abstract**: Current annotation agreement metrics are not well-suited for inter-group analysis, are sensitive to group size imbalances and restricted to single-annotation settings. These restrictions render them insufficient for many subjective tasks such as toxicity and hate-speech detection. For this reason, we introduce a quantifiable metric, paired with a statistical significance test, that attributes polarization to various annotator groups. Our metric enables direct comparisons between heavily imbalanced sociodemographic and ideological subgroups across different datasets and tasks, while also enabling analysis on multi-label settings. We apply this metric to three datasets on hate speech, and one on toxicity detection, discovering that: (1) Polarization is strongly and persistently attributed to annotator race, especially on the hate speech task. (2) Religious annotators do not fundamentally disagree with each other, but do with other annotators, a trend that is gradually diminished and then reversed with irreligious annotators. (3) Less educated annotators are more subjective, while educated ones tend to broadly agree more between themselves. Overall, our results reflect current findings around annotation patterns for various subgroups. Finally, we estimate the minimum number of annotators needed to obtain robust results, and provide an open-source Python library that implements our metric. 

---
# What Is Novel? A Knowledge-Driven Framework for Bias-Aware Literature Originality Evaluation 

**Authors**: Abeer Mostafa, Thi Huyen Nguyen, Zahra Ahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06054)  

**Abstract**: Assessing research novelty is a core yet highly subjective aspect of peer review, typically based on implicit judgment and incomplete comparison to prior work. We introduce a literature-aware novelty assessment framework that explicitly learns how humans judge novelty from peer-review reports and grounds these judgments in structured comparison to existing research. Using nearly 80K novelty-annotated reviews from top-tier AI conferences, we fine-tune a large language model to capture reviewer-aligned novelty evaluation behavior. For a given manuscript, the system extracts structured representations of its ideas, methods, and claims, retrieves semantically related papers, and constructs a similarity graph that enables fine-grained, concept-level comparison to prior work. Conditioning on this structured evidence, the model produces calibrated novelty scores and human-like explanatory assessments, reducing overestimation and improving consistency relative to existing approaches. 

---
# PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models 

**Authors**: Rajarshi Roy, Jonathan Raiman, Sang-gil Lee, Teodor-Dumitru Ene, Robert Kirby, Sungwon Kim, Jaehyeon Kim, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2602.06053)  

**Abstract**: Recent advances in duplex speech models have enabled natural, low-latency speech-to-speech interactions. However, existing models are restricted to a fixed role and voice, limiting their ability to support structured, role-driven real-world applications and personalized interactions. In this work, we introduce PersonaPlex, a duplex conversational speech model that incorporates hybrid system prompts, combining role conditioning with text prompts and voice cloning with speech samples. PersonaPlex is trained on a large-scale synthetic dataset of paired prompts and user-agent conversations, generated with open-source large language models (LLM) and text-to-speech (TTS) models. To evaluate role conditioning in real-world settings, we extend the Full-Duplex-Bench benchmark beyond a single assistant role to multi-role customer service scenarios. Experiments show that PersonaPlex achieves strong role-conditioned behavior, voice-conditioned speech, and natural conversational responsiveness, surpassing state-of-the-art duplex speech models and hybrid large language model-based speech systems in role adherence, speaker similarity, latency, and naturalness. 

---
# Rethinking Memory Mechanisms of Foundation Agents in the Second Half 

**Authors**: Wei-Chieh Huang, Weizhi Zhang, Yueqing Liang, Yuanchen Bei, Yankai Chen, Tao Feng, Xinyu Pan, Zhen Tan, Yu Wang, Tianxin Wei, Shanglin Wu, Ruiyao Xu, Liangwei Yang, Rui Yang, Wooseong Yang, Chin-Yuan Yeh, Hanrong Zhang, Haozhen Zhang, Siqi Zhu, Henry Peng Zou, Wanjia Zhao, Song Wang, Wujiang Xu, Zixuan Ke, Zheng Hui, Dawei Li, Yaozu Wu, Langzhou He, Chen Wang, Xiongxiao Xu, Baixiang Huang, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong, Ahmed A. Metwally, Jun Yan, Chen-Yu Lee, Hanqing Zeng, Yinglong Xia, Xiaokai Wei, Ali Payani, Yu Wang, Haitong Ma, Wenya Wang, Chengguang Wang, Yu Zhang, Xin Wang, Yongfeng Zhang, Jiaxuan You, Hanghang Tong, Xiao Luo, Yizhou Sun, Wei Wang, Julian McAuley, James Zou, Jiawei Han, Philip S. Yu, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06052)  

**Abstract**: The research of artificial intelligence is undergoing a paradigm shift from prioritizing model innovations over benchmark scores towards emphasizing problem definition and rigorous real-world evaluation. As the field enters the "second half," the central challenge becomes real utility in long-horizon, dynamic, and user-dependent environments, where agents face context explosion and must continuously accumulate, manage, and selectively reuse large volumes of information across extended interactions. Memory, with hundreds of papers released this year, therefore emerges as the critical solution to fill the utility gap. In this survey, we provide a unified view of foundation agent memory along three dimensions: memory substrate (internal and external), cognitive mechanism (episodic, semantic, sensory, working, and procedural), and memory subject (agent- and user-centric). We then analyze how memory is instantiated and operated under different agent topologies and highlight learning policies over memory operations. Finally, we review evaluation benchmarks and metrics for assessing memory utility, and outline various open challenges and future directions. 

---
# CAST: Character-and-Scene Episodic Memory for Agents 

**Authors**: Kexin Ma, Bojun Li, Yuhua Tang, Ruochun Jin, Liting Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.06051)  

**Abstract**: Episodic memory is a central component of human memory, which refers to the ability to recall coherent events grounded in who, when, and where. However, most agent memory systems only emphasize semantic recall and treat experience as structures such as key-value, vector, or graph, which makes them struggle to represent and retrieve coherent events. To address this challenge, we propose a Character-and-Scene based memory architecture(CAST) inspired by dramatic theory. Specifically, CAST constructs 3D scenes (time/place/topic) and organizes them into character profiles that summarize the events of a character to represent episodic memory. Moreover, CAST complements this episodic memory with a graph-based semantic memory, which yields a robust dual memory design. Experiments demonstrate that CAST has averagely improved 8.11% F1 and 10.21% J(LLM-as-a-Judge) than baselines on various datasets, especially on open and time-sensitive conversational questions. 

---
# Relevance-aware Multi-context Contrastive Decoding for Retrieval-augmented Visual Question Answering 

**Authors**: Jongha Kim, Byungoh Ko, Jeehye Na, Jinsung Yoon, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.06050)  

**Abstract**: Despite the remarkable capabilities of Large Vision Language Models (LVLMs), they still lack detailed knowledge about specific entities. Retrieval-augmented Generation (RAG) is a widely adopted solution that enhances LVLMs by providing additional contexts from an external Knowledge Base. However, we observe that previous decoding methods for RAG are sub-optimal as they fail to sufficiently leverage multiple relevant contexts and suppress the negative effects of irrelevant contexts. To this end, we propose Relevance-aware Multi-context Contrastive Decoding (RMCD), a novel decoding method for RAG. RMCD outputs a final prediction by combining outputs predicted with each context, where each output is weighted based on its relevance to the question. By doing so, RMCD effectively aggregates useful information from multiple relevant contexts while also counteracting the negative effects of irrelevant ones. Experiments show that RMCD consistently outperforms other decoding methods across multiple LVLMs, achieving the best performance on three knowledge-intensive visual question-answering benchmarks. Also, RMCD can be simply applied by replacing the decoding method of LVLMs without additional training. Analyses also show that RMCD is robust to the retrieval results, consistently performing the best across the weakest to the strongest retrieval results. Code is available at this https URL. 

---
# Recontextualizing Famous Quotes for Brand Slogan Generation 

**Authors**: Ziao Yang, Zizhang Chen, Lei Zhang, Hongfu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06049)  

**Abstract**: Slogans are concise and memorable catchphrases that play a crucial role in advertising by conveying brand identity and shaping public perception. However, advertising fatigue reduces the effectiveness of repeated slogans, creating a growing demand for novel, creative, and insightful slogan generation. While recent work leverages large language models (LLMs) for this task, existing approaches often produce stylistically redundant outputs that lack a clear brand persona and appear overtly machine-generated. We argue that effective slogans should balance novelty with familiarity and propose a new paradigm that recontextualizes persona-related famous quotes for slogan generation. Well-known quotes naturally align with slogan-length text, employ rich rhetorical devices, and offer depth and insight, making them a powerful resource for creative generation. Technically, we introduce a modular framework that decomposes slogan generation into interpretable subtasks, including quote matching, structural decomposition, vocabulary replacement, and remix generation. Extensive automatic and human evaluations demonstrate marginal improvements in diversity, novelty, emotional impact, and human preference over three state-of-the-art LLM baselines. 

---
# Learning a Generative Meta-Model of LLM Activations 

**Authors**: Grace Luo, Jiahai Feng, Trevor Darrell, Alec Radford, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2602.06964)  

**Abstract**: Existing approaches for analyzing neural network activations, such as PCA and sparse autoencoders, rely on strong structural assumptions. Generative models offer an alternative: they can uncover structure without such assumptions and act as priors that improve intervention fidelity. We explore this direction by training diffusion models on one billion residual stream activations, creating "meta-models" that learn the distribution of a network's internal states. We find that diffusion loss decreases smoothly with compute and reliably predicts downstream utility. In particular, applying the meta-model's learned prior to steering interventions improves fluency, with larger gains as loss decreases. Moreover, the meta-model's neurons increasingly isolate concepts into individual units, with sparse probing scores that scale as loss decreases. These results suggest generative meta-models offer a scalable path toward interpretability without restrictive structural assumptions. Project page: this https URL. 

---
# Endogenous Resistance to Activation Steering in Language Models 

**Authors**: Alex McKenzie, Keenan Pepper, Stijn Servaes, Martin Leitgab, Murat Cubuktepe, Mike Vaiana, Diogo de Lucena, Judd Rosenblatt, Michael S. A. Graziano  

**Link**: [PDF](https://arxiv.org/pdf/2602.06941)  

**Abstract**: Large language models can resist task-misaligned activation steering during inference, sometimes recovering mid-generation to produce improved responses even when steering remains active. We term this Endogenous Steering Resistance (ESR). Using sparse autoencoder (SAE) latents to steer model activations, we find that Llama-3.3-70B shows substantial ESR, while smaller models from the Llama-3 and Gemma-2 families exhibit the phenomenon less frequently. We identify 26 SAE latents that activate differentially during off-topic content and are causally linked to ESR in Llama-3.3-70B. Zero-ablating these latents reduces the multi-attempt rate by 25%, providing causal evidence for dedicated internal consistency-checking circuits. We demonstrate that ESR can be deliberately enhanced through both prompting and training: meta-prompts instructing the model to self-monitor increase the multi-attempt rate by 4x for Llama-3.3-70B, and fine-tuning on self-correction examples successfully induces ESR-like behavior in smaller models. These findings have dual implications: ESR could protect against adversarial manipulation but might also interfere with beneficial safety interventions that rely on activation steering. Understanding and controlling these resistance mechanisms is important for developing transparent and controllable AI systems. Code is available at this http URL. 

---
# Quantum Attention by Overlap Interference: Predicting Sequences from Classical and Many-Body Quantum Data 

**Authors**: Alessio Pecilli, Matteo Rosati  

**Link**: [PDF](https://arxiv.org/pdf/2602.06699)  

**Abstract**: We propose a variational quantum implementation of self-attention (QSA), the core operation in transformers and large language models, which predicts future elements of a sequence by forming overlap-weighted combinations of past data. At variance with previous approaches, our QSA realizes the required nonlinearity through interference of state overlaps and returns a Renyi-1/2 cross-entropy loss directly as the expectation value of an observable, avoiding the need to decode amplitude-encoded predictions into classical logits. Furthermore, QSA naturally accommodates a constrained, trainable data-embedding that ties quantum state overlaps to data-level similarities. We find a gate complexity dominant scaling O(T d^2) for QSA, versus O(T^2 d) classically, suggesting an advantage in the practical regime where the sequence length T dominates the embedding size d. In simulations, we show that our QSA-based quantum transformer learns sequence prediction on classical data and on many-body transverse-field Ising quantum trajectories, establishing trainable attention as a practical primitive for quantum dynamical modeling. 

---
# Personality as Relational Infrastructure: User Perceptions of Personality-Trait-Infused LLM Messaging 

**Authors**: Dominik P. Hofer, David Haag, Rania Islambouli, Jan D. Smeddinck  

**Link**: [PDF](https://arxiv.org/pdf/2602.06596)  

**Abstract**: Digital behaviour change systems increasingly rely on repeated, system-initiated messages to support users in everyday contexts. LLMs enable these messages to be personalised consistently across interactions, yet it remains unclear whether such personalisation improves individual messages or instead shapes users' perceptions through patterns of exposure. We explore this question in the context of LLM-generated JITAIs, which are short, context-aware messages delivered at moments deemed appropriate to support behaviour change, using physical activity as an application domain. In a controlled retrospective study, 90 participants evaluated messages generated using four LLM strategies: baseline prompting, few-shot prompting, fine-tuned models, and retrieval augmented generation, each implemented with and without Big Five Personality Traits to produce personality-aligned communication across multiple scenarios. Using ordinal multilevel models with within-between decomposition, we distinguish trial-level effects, whether personality information improves evaluations of individual messages, from person-level exposure effects, whether participants receiving higher proportions of personality-informed messages exhibit systematically different overall perceptions. Results showed no trial-level associations, but participants who received higher proportions of BFPT-informed messages rated the messages as more personalised, appropriate, and reported less negative affect. We use Communication Accommodation Theory for post-hoc analysis. These results suggest that personality-based personalisation in behaviour change systems may operate primarily through aggregate exposure rather than per-message optimisation, with implications for how adaptive systems are designed and evaluated in sustained human-AI interaction. In-situ longitudinal studies are needed to validate these findings in real-world contexts. 

---
# SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs 

**Authors**: Niccolo Avogaro, Nayanika Debnath, Li Mi, Thomas Frick, Junling Wang, Zexue He, Hang Hua, Konrad Schindler, Mattia Rigotti  

**Link**: [PDF](https://arxiv.org/pdf/2602.06566)  

**Abstract**: Despite recent successes, test-time scaling - i.e., dynamically expanding the token budget during inference as needed - remains brittle for vision-language models (VLMs): unstructured chains-of-thought about images entangle perception and reasoning, leading to long, disorganized contexts where small perceptual mistakes may cascade into completely wrong answers. Moreover, expensive reinforcement learning with hand-crafted rewards is required to achieve good performance. Here, we introduce SPARC (Separating Perception And Reasoning Circuits), a modular framework that explicitly decouples visual perception from reasoning. Inspired by sequential sensory-to-cognitive processing in the brain, SPARC implements a two-stage pipeline where the model first performs explicit visual search to localize question-relevant regions, then conditions its reasoning on those regions to produce the final answer. This separation enables independent test-time scaling with asymmetric compute allocation (e.g., prioritizing perceptual processing under distribution shift), supports selective optimization (e.g., improving the perceptual stage alone when it is the bottleneck for end-to-end performance), and accommodates compressed contexts by running global search at lower image resolutions and allocating high-resolution processing only to selected regions, thereby reducing total visual tokens count and compute. Across challenging visual reasoning benchmarks, SPARC outperforms monolithic baselines and strong visual-grounding approaches. For instance, SPARC improves the accuracy of Qwen3VL-4B on the $V^*$ VQA benchmark by 6.7 percentage points, and it surpasses "thinking with images" by 4.6 points on a challenging OOD task despite requiring a 200$\times$ lower token budget. 

---
# Malicious Agent Skills in the Wild: A Large-Scale Security Empirical Study 

**Authors**: Yi Liu, Zhihao Chen, Yanjun Zhang, Gelei Deng, Yuekang Li, Jianting Ning, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06547)  

**Abstract**: Third-party agent skills extend LLM-based agents with instruction files and executable code that run on users' machines. Skills execute with user privileges and are distributed through community registries with minimal vetting, but no ground-truth dataset exists to characterize the resulting threats. We construct the first labeled dataset of malicious agent skills by behaviorally verifying 98,380 skills from two community registries, confirming 157 malicious skills with 632 vulnerabilities. These attacks are not incidental. Malicious skills average 4.03 vulnerabilities across a median of three kill chain phases, and the ecosystem has split into two archetypes: Data Thieves that exfiltrate credentials through supply chain techniques, and Agent Hijackers that subvert agent decision-making through instruction manipulation. A single actor accounts for 54.1\% of confirmed cases through templated brand impersonation. Shadow features, capabilities absent from public documentation, appear in 0\% of basic attacks but 100\% of advanced ones; several skills go further by exploiting the AI platform's own hook system and permission flags. Responsible disclosure led to 93.6\% removal within 30 days. We release the dataset and analysis pipeline to support future work on agent skill security. 

---
# AgentCPM-Report: Interleaving Drafting and Deepening for Open-Ended Deep Research 

**Authors**: Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.06540)  

**Abstract**: Generating deep research reports requires large-scale information acquisition and the synthesis of insight-driven analysis, posing a significant challenge for current language models. Most existing approaches follow a plan-then-write paradigm, whose performance heavily depends on the quality of the initial outline. However, constructing a comprehensive outline itself demands strong reasoning ability, causing current deep research systems to rely almost exclusively on closed-source or online large models. This reliance raises practical barriers to deployment and introduces safety and privacy concerns for user-authored data. In this work, we present AgentCPM-Report, a lightweight yet high-performing local solution composed of a framework that mirrors the human writing process and an 8B-parameter deep research agent. Our framework uses a Writing As Reasoning Policy (WARP), which enables models to dynamically revise outlines during report generation. Under this policy, the agent alternates between Evidence-Based Drafting and Reasoning-Driven Deepening, jointly supporting information acquisition, knowledge refinement, and iterative outline evolution. To effectively equip small models with this capability, we introduce a Multi-Stage Agentic Training strategy, consisting of cold-start, atomic skill RL, and holistic pipeline RL. Experiments on DeepResearch Bench, DeepConsult, and DeepResearch Gym demonstrate that AgentCPM-Report outperforms leading closed-source systems, with substantial gains in Insight. 

---
# LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models 

**Authors**: Brian Rabern, Philipp Mondorf, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2602.06533)  

**Abstract**: Large language models have demonstrated notable performance across various logical reasoning benchmarks. However, it remains unclear which core logical skills they truly master. To address this, we introduce LogicSkills, a unified benchmark designed to isolate three fundamental skills in formal reasoning: (i) $\textit{formal symbolization}\unicode{x2014}$translating premises into first-order logic; (ii) $\textit{countermodel construction}\unicode{x2014}$formulating a finite structure in which all premises are true while the conclusion is false; and (iii) $\textit{validity assessment}\unicode{x2014}$deciding whether a conclusion follows from a given set of premises. Items are drawn from the two-variable fragment of first-order logic (without identity) and are presented in both natural English and a Carroll-style language with nonce words. All examples are verified for correctness and non-triviality using the SMT solver Z3. Across leading models, performance is high on validity but substantially lower on symbolization and countermodel construction, suggesting reliance on surface-level patterns rather than genuine symbolic or rule-based reasoning. 

---
# Designing Computational Tools for Exploring Causal Relationships in Qualitative Data 

**Authors**: Han Meng, Qiuyuan Lyu, Peinuan Qin, Yitian Yang, Renwen Zhang, Wen-Chieh Lin, Yi-Chieh Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.06506)  

**Abstract**: Exploring causal relationships for qualitative data analysis in HCI and social science research enables the understanding of user needs and theory building. However, current computational tools primarily characterize and categorize qualitative data; the few systems that analyze causal relationships either inadequately consider context, lack credibility, or produce overly complex outputs. We first conducted a formative study with 15 participants interested in using computational tools for exploring causal relationships in qualitative data to understand their needs and derive design guidelines. Based on these findings, we designed and implemented QualCausal, a system that extracts and illustrates causal relationships through interactive causal network construction and multi-view visualization. A feedback study (n = 15) revealed that participants valued our system for reducing the analytical burden and providing cognitive scaffolding, yet navigated how such systems fit within their established research paradigms, practices, and habits. We discuss broader implications for designing computational tools that support qualitative data analysis. 

---
# The Condensate Theorem: Transformers are O(n), Not $O(n^2)$ 

**Authors**: Jorge L. Ruiz Williams  

**Link**: [PDF](https://arxiv.org/pdf/2602.06317)  

**Abstract**: We present the Condensate Theorem: attention sparsity is a learned topological property, not an architectural constraint. Through empirical analysis of trained language models, we find that attention mass concentrates on a distinct topological manifold -- and this manifold can be identified dynamically without checking every position. We prove a general result: for any query, projecting attention onto the Condensate Manifold (Anchor + Window + Dynamic Top-k) achieves 100% output equivalence with full $O(n^2)$ attention. This is not an approximation -- it is lossless parity. We validate this across GPT-2, Pythia, Qwen2, TinyLlama, and Mistral, demonstrating bit-exact token matching on 1,500+ generated tokens. By mapping this topology to hardware, our Topological Attention kernel achieves a 159x measured speedup at 131K tokens (3.94ms vs 628ms) and a projected >1,200x speedup at 1M tokens, reducing inference costs by >99.9% compared to Flash Attention. We conclude that the quadratic bottleneck is an artifact of naive implementation, not intelligence. 

---
# Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions 

**Authors**: Navita Goyal, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2602.06256)  

**Abstract**: Model steering, which involves intervening on hidden representations at inference time, has emerged as a lightweight alternative to finetuning for precisely controlling large language models. While steering efficacy has been widely studied, evaluations of whether interventions alter only the intended property remain limited, especially with respect to unintended changes in behaviors related to the target property. We call this notion specificity. We propose a framework that distinguishes three dimensions of specificity: general (preserving fluency and unrelated abilities), control (preserving related control properties), and robustness (preserving control properties under distribution shifts). We study two safety-critical use cases: steering models to reduce overrefusal and faithfulness hallucinations, and show that while steering achieves high efficacy and largely maintains general and control specificity, it consistently fails to preserve robustness specificity. In the case of overrefusal steering, for example, all steering methods reduce overrefusal without harming general abilities and refusal on harmful queries; however, they substantially increase vulnerability to jailbreaks. Our work provides the first systematic evaluation of specificity in model steering, showing that standard efficacy and specificity checks are insufficient, because without robustness evaluation, steering methods may appear reliable even when they compromise model safety. 

---
# Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning 

**Authors**: Nan Chen, Soledad Villar, Soufiane Hayou  

**Link**: [PDF](https://arxiv.org/pdf/2602.06204)  

**Abstract**: Low-Rank Adaptation (LoRA) is a standard tool for parameter-efficient finetuning of large models. While it induces a small memory footprint, its training dynamics can be surprisingly complex as they depend on several hyperparameters such as initialization, adapter rank, and learning rate. In particular, it is unclear how the optimal learning rate scales with adapter rank, which forces practitioners to re-tune the learning rate whenever the rank is changed. In this paper, we introduce Maximal-Update Adaptation ($\mu$A), a theoretical framework that characterizes how the "optimal" learning rate should scale with model width and adapter rank to produce stable, non-vanishing feature updates under standard configurations. $\mu$A is inspired from the Maximal-Update Parametrization ($\mu$P) in pretraining. Our analysis leverages techniques from hyperparameter transfer and reveals that the optimal learning rate exhibits different scaling patterns depending on initialization and LoRA scaling factor. Specifically, we identify two regimes: one where the optimal learning rate remains roughly invariant across ranks, and another where it scales inversely with rank. We further identify a configuration that allows learning rate transfer from LoRA to full finetuning, drastically reducing the cost of learning rate tuning for full finetuning. Experiments across language, vision, vision--language, image generation, and reinforcement learning tasks validate our scaling rules and show that learning rates tuned on LoRA transfer reliably to full finetuning. 

---
# Generics in science communication: Misaligned interpretations across laypeople, scientists, and large language models 

**Authors**: Uwe Peters, Andrea Bertazzoli, Jasmine M. DeJesus, Gisela J. van der Velden, Benjamin Chin-Yee  

**Link**: [PDF](https://arxiv.org/pdf/2602.06190)  

**Abstract**: Scientists often use generics, that is, unquantified statements about whole categories of people or phenomena, when communicating research findings (e.g., "statins reduce cardiovascular events"). Large language models (LLMs), such as ChatGPT, frequently adopt the same style when summarizing scientific texts. However, generics can prompt overgeneralizations, especially when they are interpreted differently across audiences. In a study comparing laypeople, scientists, and two leading LLMs (ChatGPT-5 and DeepSeek), we found systematic differences in interpretation of generics. Compared to most scientists, laypeople judged scientific generics as more generalizable and credible, while LLMs rated them even higher. These mismatches highlight significant risks for science communication. Scientists may use generics and incorrectly assume laypeople share their interpretation, while LLMs may systematically overgeneralize scientific findings when summarizing research. Our findings underscore the need for greater attention to language choices in both human and LLM-mediated science communication. 

---
# PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision-Language Pretraining 

**Authors**: Cheng Liang, Chaoyi Wu, Weike Zhao, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.06184)  

**Abstract**: Recent progress in large-scale CLIP-like vision-language models(VLMs) has greatly advanced medical image analysis. However, most existing medical VLMs still rely on coarse image-text contrastive objectives and fail to capture the systematic visual knowledge encoded in well-defined medical phenotype ontologies. To address this gap, we construct PhenoKG, the first large-scale, phenotype-centric multimodal knowledge graph that encompasses over 520K high-quality image-text pairs linked to more than 3,000 phenotypes. Building upon PhenoKG, we propose PhenoLIP, a novel pretraining framework that explicitly incorporates structured phenotype knowledge into medical VLMs through a two-stage process. We first learn a knowledge-enhanced phenotype embedding space from textual ontology data and then distill this structured knowledge into multimodal pretraining via a teacher-guided knowledge distillation objective. To support evaluation, we further introduce PhenoBench, an expert-verified benchmark designed for phenotype recognition, comprising over 7,800 image--caption pairs covering more than 1,000 phenotypes. Extensive experiments demonstrate that PhenoLIP outperforms previous state-of-the-art baselines, improving upon BiomedCLIP in phenotype classification accuracy by 8.85\% and BIOMEDICA in cross-modal retrieval by 15.03%, underscoring the value of integrating phenotype-centric priors into medical VLMs for structured and interpretable medical image understanding. 

---
# STACodec: Semantic Token Assignment for Balancing Acoustic Fidelity and Semantic Information in Audio Codecs 

**Authors**: Kaiyuan Zhang, Mohan Shi, Eray Eren, Natarajan Balaji Shankar, Zilai Wang, Abeer Alwan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06180)  

**Abstract**: Neural audio codecs are widely used for audio compression and can be integrated into token-based language models. Traditional codecs preserve acoustic details well but lack semantic information. Recent hybrid codecs attempt to incorporate semantic information through distillation, but this often degrades reconstruction performance, making it difficult to achieve both. To address this limitation, we introduce STACodec, a unified codec that integrates semantic information from self-supervised learning (SSL) models into the first layer of residual vector quantization (RVQ-1) via semantic token assignment (STA). To further eliminate reliance on SSL-based semantic tokenizers and improve efficiency during inference, we propose a semantic pre-distillation (SPD) module, which predicts semantic tokens directly for assignment to the first RVQ layer during inference. Experimental results show that STACodec outperforms existing hybrid codecs in both audio reconstruction and downstream semantic tasks, demonstrating a better balance between acoustic fidelity and semantic capability. 

---
# Large Language Model Reasoning Failures 

**Authors**: Peiyang Song, Pengrui Han, Noah Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2602.06176)  

**Abstract**: Large Language Models (LLMs) have exhibited remarkable reasoning capabilities, achieving impressive results across a wide range of tasks. Despite these advances, significant reasoning failures persist, occurring even in seemingly simple scenarios. To systematically understand and address these shortcomings, we present the first comprehensive survey dedicated to reasoning failures in LLMs. We introduce a novel categorization framework that distinguishes reasoning into embodied and non-embodied types, with the latter further subdivided into informal (intuitive) and formal (logical) reasoning. In parallel, we classify reasoning failures along a complementary axis into three types: fundamental failures intrinsic to LLM architectures that broadly affect downstream tasks; application-specific limitations that manifest in particular domains; and robustness issues characterized by inconsistent performance across minor variations. For each reasoning failure, we provide a clear definition, analyze existing studies, explore root causes, and present mitigation strategies. By unifying fragmented research efforts, our survey provides a structured perspective on systemic weaknesses in LLM reasoning, offering valuable insights and guiding future research towards building stronger, more reliable, and robust reasoning capabilities. We additionally release a comprehensive collection of research works on LLM reasoning failures, as a GitHub repository at this https URL, to provide an easy entry point to this area. 

---
# MoSE: Mixture of Slimmable Experts for Efficient and Adaptive Language Models 

**Authors**: Nurbek Tastan, Stefanos Laskaridis, Karthik Nandakumar, Samuel Horvath  

**Link**: [PDF](https://arxiv.org/pdf/2602.06154)  

**Abstract**: Mixture-of-Experts (MoE) models scale large language models efficiently by sparsely activating experts, but once an expert is selected, it is executed fully. Hence, the trade-off between accuracy and computation in an MoE model typically exhibits large discontinuities. We propose Mixture of Slimmable Experts (MoSE), an MoE architecture in which each expert has a nested, slimmable structure that can be executed at variable widths. This enables conditional computation not only over which experts are activated, but also over how much of each expert is utilized. Consequently, a single pretrained MoSE model can support a more continuous spectrum of accuracy-compute trade-offs at inference time. We present a simple and stable training recipe for slimmable experts under sparse routing, combining multi-width training with standard MoE objectives. During inference, we explore strategies for runtime width determination, including a lightweight test-time training mechanism that learns how to map router confidence/probabilities to expert widths under a fixed budget. Experiments on GPT models trained on OpenWebText demonstrate that MoSE matches or improves upon standard MoE at full width and consistently shifts the Pareto frontier for accuracy vs. cost, achieving comparable performance with significantly fewer FLOPs. 

---
# Protean Compiler: An Agile Framework to Drive Fine-grain Phase Ordering 

**Authors**: Amir H. Ashouri, Shayan Shirahmad Gale Bagi, Kavin Satheeskumar, Tejas Srikanth, Jonathan Zhao, Ibrahim Saidoun, Ziwen Wang, Bryan Chan, Tomasz S. Czajkowski  

**Link**: [PDF](https://arxiv.org/pdf/2602.06142)  

**Abstract**: The phase ordering problem has been a long-standing challenge since the late 1970s, yet it remains an open problem due to having a vast optimization space and an unbounded nature, making it an open-ended problem without a finite solution, one can limit the scope by reducing the number and the length of optimizations. Traditionally, such locally optimized decisions are made by hand-coded algorithms tuned for a small number of benchmarks, often requiring significant effort to be retuned when the benchmark suite changes. In the past 20 years, Machine Learning has been employed to construct performance models to improve the selection and ordering of compiler optimizations, however, the approaches are not baked into the compiler seamlessly and never materialized to be leveraged at a fine-grained scope of code segments. This paper presents Protean Compiler: An agile framework to enable LLVM with built-in phase-ordering capabilities at a fine-grained scope. The framework also comprises a complete library of more than 140 handcrafted static feature collection methods at varying scopes, and the experimental results showcase speedup gains of up to 4.1% on average and up to 15.7% on select Cbench applications wrt LLVM's O3 by just incurring a few extra seconds of build time on Cbench. Additionally, Protean compiler allows for an easy integration with third-party ML frameworks and other Large Language Models, and this two-step optimization shows a gain of 10.1% and 8.5% speedup wrt O3 on Cbench's Susan and Jpeg applications. Protean compiler is seamlessly integrated into LLVM and can be used as a new, enhanced, full-fledged compiler. We plan to release the project to the open-source community in the near future. 

---
# Self-Improving World Modelling with Latent Actions 

**Authors**: Yifu Qiu, Zheng Zhao, Waylon Li, Yftah Ziser, Anna Korhonen, Shay B. Cohen, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2602.06130)  

**Abstract**: Internal modelling of the world -- predicting transitions between previous states $X$ and next states $Y$ under actions $Z$ -- is essential to reasoning and planning for LLMs and VLMs. Learning such models typically requires costly action-labelled trajectories. We propose SWIRL, a self-improvement framework that learns from state-only sequences by treating actions as a latent variable and alternating between Forward World Modelling (FWM) $P_\theta(Y|X,Z)$ and an Inverse Dynamics Modelling (IDM) $Q_\phi(Z|X,Y)$. SWIRL iterates two phases: (1) Variational Information Maximisation, which updates the FWM to generate next states that maximise conditional mutual information with latent actions given prior states, encouraging identifiable consistency; and (2) ELBO Maximisation, which updates the IDM to explain observed transitions, effectively performing coordinate ascent. Both models are trained with reinforcement learning (specifically, GRPO) with the opposite frozen model's log-probability as a reward signal. We provide theoretical learnability guarantees for both updates, and evaluate SWIRL on LLMs and VLMs across multiple environments: single-turn and multi-turn open-world visual dynamics and synthetic textual environments for physics, web, and tool calling. SWIRL achieves gains of 16% on AURORABench, 28% on ByteMorph, 16% on WorldPredictionBench, and 14% on StableToolBench. 

---
# Deep networks learn to parse uniform-depth context-free languages from local statistics 

**Authors**: Jack T. Parley, Francesco Cagnetta, Matthieu Wyart  

**Link**: [PDF](https://arxiv.org/pdf/2602.06065)  

**Abstract**: Understanding how the structure of language can be learned from sentences alone is a central question in both cognitive science and machine learning. Studies of the internal representations of Large Language Models (LLMs) support their ability to parse text when predicting the next word, while representing semantic notions independently of surface form. Yet, which data statistics make these feats possible, and how much data is required, remain largely unknown. Probabilistic context-free grammars (PCFGs) provide a tractable testbed for studying these questions. However, prior work has focused either on the post-hoc characterization of the parsing-like algorithms used by trained networks; or on the learnability of PCFGs with fixed syntax, where parsing is unnecessary. Here, we (i) introduce a tunable class of PCFGs in which both the degree of ambiguity and the correlation structure across scales can be controlled; (ii) provide a learning mechanism -- an inference algorithm inspired by the structure of deep convolutional networks -- that links learnability and sample complexity to specific language statistics; and (iii) validate our predictions empirically across deep convolutional and transformer-based architectures. Overall, we propose a unifying framework where correlations at different scales lift local ambiguities, enabling the emergence of hierarchical representations of the data. 

---
# Analyzing Diffusion and Autoregressive Vision Language Models in Multimodal Embedding Space 

**Authors**: Zihang Wang, Siyue Zhang, Yilun Zhao, Jingyi Yang, Tingyu Song, Anh Tuan Luu, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.06056)  

**Abstract**: Embedding models are a fundamental component of modern AI systems such as semantic search and retrieval-augmented generation. Recent advances in large foundation models have substantially accelerated the development of embedding models, including those based on Large Language Models (LLMs), Vision Language Models (VLMs), and Multimodal LLMs. More recently, Large Diffusion Language Models (dLLMs) and Multimodal dLLMs have emerged as competitive alternatives to autoregressive models, offering advantages such as bidirectional attention and parallel generation. This progress naturally raises a critical yet unexplored question: can Multimodal dLLMs serve as effective multimodal embedding models? To answer this, we present the first systematic study of converting Multimodal dLLMs into embedding models. We evaluate state-of-the-art Multimodal dLLMs and Autoregressive VLMs across three categories of embedding tasks: classification, visual question answering, and information retrieval. Our results show that Multimodal dLLM embeddings generally underperform their autoregressive VLM counterparts. The stronger diffusion-based model, LaViDa, lags by only 3.5 points on classification, 2.5 points on VQA, and 4.4 points on retrieval tasks, whereas the other diffusion-based model, MMaDA, exhibits substantially larger performance gaps, exceeding 20 points across all tasks. Further analysis reveals insufficient image-text alignment in diffusion-based models, accounting for the observed limitations in their embedding performance. 

---
