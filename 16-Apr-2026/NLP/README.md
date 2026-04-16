# From Feelings to Metrics: Understanding and Formalizing How Users Vibe-Test LLMs 

**Authors**: Itay Itzhak, Eliya Habba, Gabriel Stanovsky, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.14137)  

**Abstract**: Evaluating LLMs is challenging, as benchmark scores often fail to capture models' real-world usefulness. Instead, users often rely on ``vibe-testing'': informal experience-based evaluation, such as comparing models on coding tasks related to their own workflow. While prevalent, vibe-testing is often too ad hoc and unstructured to analyze or reproduce at scale. In this work, we study how vibe-testing works in practice and then formalize it to support systematic analysis. We first analyze two empirical resources: (1) a survey of user evaluation practices, and (2) a collection of in-the-wild model comparison reports from blogs and social media. Based on these resources, we formalize vibe-testing as a two-part process: users personalize both what they test and how they judge responses. We then introduce a proof-of-concept evaluation pipeline that follows this formulation by generating personalized prompts and comparing model outputs using user-aware subjective criteria. In experiments on coding benchmarks, we find that combining personalized prompts and user-aware evaluation can change which model is preferred, reflecting the role of vibe-testing in practice. These findings suggest that formalized vibe-testing can serve as a useful approach for bridging benchmark scores and real-world experience. 

---
# Rhetorical Questions in LLM Representations: A Linear Probing Study 

**Authors**: Louie Hong Yao, Vishesh Anand, Yuan Zhuang, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14128)  

**Abstract**: Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some probes capture discourse-level rhetorical stance embedded in extended argumentation, while others emphasize localized, syntax-driven interrogative acts. Together, these findings suggest that rhetorical questions in LLM representations are encoded by multiple linear directions emphasizing different cues, rather than a single shared direction. 

---
# Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis 

**Authors**: Zipeng Ling, Shuliang Liu, Shenghong Fu, Yuehao Tang, Seonil Son, Yao Wan, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14121)  

**Abstract**: LLM reasoning traces suffer from complex flaws -- *Step Internal Flaws* (logical errors, hallucinations, etc.) and *Step-wise Flaws* (overthinking, underthinking), which vary by sample. A natural approach would be to provide ground-truth labels to guide LLMs' reasoning. Contrary to intuition, we show that this yields no improvement in reasoning ability. We then propose CRAFT, a unified framework that mitigates both types of Step flaws, which builds a Reasoning Knowledge Graph (RKG) based on the consensus parts of multiple candidate traces, and synthesizes a high-quality trace through topological generation. Our approach improves label-prediction accuracy by 10+% on average, and consistently outperforms all baselines across both logical and mathematical reasoning benchmarks. Further, detailed benchmark evaluation proves that our method also improves the quality of LLMs' reasoning traces in multiple dimensions. 

---
# Interpretable Stylistic Variation in Human and LLM Writing Across Genres, Models, and Decoding Strategies 

**Authors**: Swati Rallapalli, Shannon Gallagher, Ronald Yurko, Tyler Brooks, Chuck Loughin, Michele Sezgin, Violet Turri  

**Link**: [PDF](https://arxiv.org/pdf/2604.14111)  

**Abstract**: Large Language Models (LLMs) are now capable of generating highly fluent, human-like text. They enable many applications, but also raise concerns such as large scale spam, phishing, or academic misuse. While much work has focused on detecting LLM-generated text, only limited work has gone into understanding the stylistic differences between human-written and machine-generated text. In this work, we perform a large scale analysis of stylistic variation across human-written text and outputs from 11 LLMs spanning 8 different genres and 4 decoding strategies using Douglas Biber's set of lexicogrammatical and functional features. Our findings reveal insights that can guide intentional LLM usage. First, key linguistic differentiators of LLM-generated text seem robust to generation conditions (e.g., prompt settings to nudge them to generate human-like text, or availability of human-written text to continue the style); second, genre exerts a stronger influence on stylistic features than the source itself; third, chat variants of the models generally appear to be clustered together in stylistic space, and finally, model has a larger effect on the style than decoding strategy, with some exceptions. These results highlight the relative importance of model and genre over prompting and decoding strategies in shaping the stylistic behavior of machine-generated text. 

---
# From Weights to Activations: Is Steering the Next Frontier of Adaptation? 

**Authors**: Simon Ostermann, Daniil Gurgurov, Tanja Baeumel, Michael A. Hedderich, Sebastian Lapuschkin, Wojciech Samek, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2604.14090)  

**Abstract**: Post-training adaptation of language models is commonly achieved through parameter updates or input-based methods such as fine-tuning, parameter-efficient adaptation, and prompting. In parallel, a growing body of work modifies internal activations at inference time to influence model behavior, an approach known as steering. Despite increasing use, steering is rarely analyzed within the same conceptual framework as established adaptation methods.
In this work, we argue that steering should be regarded as a form of model adaptation. We introduce a set of functional criteria for adaptation methods and use them to compare steering approaches with classical alternatives. This analysis positions steering as a distinct adaptation paradigm based on targeted interventions in activation space, enabling local and reversible behavioral change without parameter updates. The resulting framing clarifies how steering relates to existing methods, motivating a unified taxonomy for model adaptation. 

---
# From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution 

**Authors**: Pavel Chizhov, Egor Bogomolov, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2604.14053)  

**Abstract**: Efficiency and safety of Large Language Models (LLMs), among other factors, rely on the quality of tokenization. A good tokenizer not only improves inference speed and language understanding but also provides extra defense against jailbreak attacks and lowers the risk of hallucinations. In this work, we investigate the efficiency of code tokenization, in particular from the perspective of data source diversity. We demonstrate that code tokenizers are prone to producing unused, and thus under-trained, tokens due to the imbalance in repository and language diversity in the training data, as well as the dominance of source-specific, repetitive tokens that are often unusable in future inference. By modifying the BPE objective and introducing merge skipping, we implement different techniques under the name Source-Attributed BPE (SA-BPE) to regularize BPE training and minimize overfitting, thereby substantially reducing the number of under-trained tokens while maintaining the same inference procedure as with regular BPE. This provides an effective tool suitable for production use. 

---
# Dual-Enhancement Product Bundling: Bridging Interactive Graph and Large Language Model 

**Authors**: Zhe Huang, Peng Wang, Yan Zheng, Sen Song, Longjun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.14030)  

**Abstract**: Product bundling boosts e-commerce revenue by recommending complementary item combinations. However, existing methods face two critical challenges: (1) collaborative filtering approaches struggle with cold-start items owing to dependency on historical interactions, and (2) LLMs lack inherent capability to model interactive graph directly. To bridge this gap, we propose a dual-enhancement method that integrates interactive graph learning and LLM-based semantic understanding for product bundling. Our method introduces a graph-to-text paradigm, which leverages a Dynamic Concept Binding Mechanism (DCBM) to translate graph structures into natural language prompts. The DCBM plays a critical role in aligning domain-specific entities with LLM tokenization, enabling effective comprehension of combinatorial constraints. Experiments on three benchmarks (POG, POG_dense, Steam) demonstrate 6.3%-26.5% improvements over state-of-the-art baselines. 

---
# Diffusion Language Models for Speech Recognition 

**Authors**: Davyd Naveriani, Albert Zeyer, Ralf Schlüter, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2604.14001)  

**Abstract**: Diffusion language models have recently emerged as a leading alternative to standard language models, due to their ability for bidirectional attention and parallel text generation. In this work, we explore variants for their use in speech recognition. Specifically, we introduce a comprehensive guide to incorporating masked diffusion language models (MDLM) and uniform-state diffusion models (USDMs) for rescoring ASR hypotheses. Additionally, we design a new joint-decoding method that combines CTC and USDM by integrating the framewise probability distributions derived from CTC with the labelwise probability distributions computed by USDM at each decoding step, thereby generating new candidates that combine strong language knowledge from USDM and acoustic information from CTC. Our findings reveal that USDM, as well as MDLM, can significantly improve the accuracy of recognized text. We publish all our code and recipes. 

---
# Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models 

**Authors**: Aleksandr Rubashevskii, Dzianis Piatrashyn, Preslav Nakov, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2604.13991)  

**Abstract**: Large language models (LLMs) are prone to generating factually incorrect outputs. Recent work has applied conformal prediction to provide uncertainty estimates and statistical guarantees for the factuality of LLM generations. However, existing approaches are typically not prompt-adaptive, limiting their ability to capture input-dependent variability. As a result, they may filter out too few items (leading to over-coverage) or too many (under-coverage) for a given task or prompt. We propose an adaptive conformal prediction approach that extends conformal score transformation methods to LLMs, with applications to long-form generation and multiple-choice question answering. This enables prompt-dependent calibration, retaining marginal coverage guarantees while improving conditional coverage. In addition, the approach naturally supports selective prediction, allowing unreliable claims or answer choices to be filtered out in downstream applications. We evaluate our approach on multiple white-box models across diverse domains and show that it significantly outperforms existing baselines in terms of conditional coverage. 

---
# Leveraging LLM-GNN Integration for Open-World Question Answering over Knowledge Graphs 

**Authors**: Hussein Abdallah, Ibrahim Abdelaziz, Panos Kalnis, Essam Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2604.13979)  

**Abstract**: Open-world Question Answering (OW-QA) over knowledge graphs (KGs) aims to answer questions over incomplete or evolving KGs. Traditional KGQA assumes a closed world where answers must exist in the KG, limiting real-world applicability. In contrast, open-world QA requires inferring missing knowledge based on graph structure and context. Large language models (LLMs) excel at language understanding but lack structured reasoning. Graph neural networks (GNNs) model graph topology but struggle with semantic interpretation. Existing systems integrate LLMs with GNNs or graph retrievers. Some support open-world QA but rely on structural embeddings without semantic grounding. Most assume observed paths or complete graphs, making them unreliable under missing links or multi-hop reasoning. We present GLOW, a hybrid system that combines a pre-trained GNN and an LLM for open-world KGQA. The GNN predicts top-k candidate answers from the graph structure. These, along with relevant KG facts, are serialized into a structured prompt (e.g., triples and candidates) to guide the LLM's reasoning. This enables joint reasoning over symbolic and semantic signals, without relying on retrieval or fine-tuning. To evaluate generalization, we introduce GLOW-BENCH, a 1,000-question benchmark over incomplete KGs across diverse domains. GLOW outperforms existing LLM-GNN systems on standard benchmarks and GLOW-BENCH, achieving up to 53.3% and an average 38% improvement. GitHub code and data are available. 

---
# How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data 

**Authors**: Joel Niklaus, Atsuki Yamaguchi, Michal Štefánik, Guilherme Penedo, Hynek Kydlíček, Elie Bakouch, Lewis Tunstall, Edward Emanuel Beeching, Thibaud Frere, Colin Raffel, Leandro von Werra, Thomas Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2604.13977)  

**Abstract**: Synthetic data is a standard component in training large language models, yet systematic comparisons across design dimensions, including rephrasing strategy, generator model, and source data, remain absent. We conduct extensive controlled experiments, generating over one trillion tokens, to identify critical factors in rephrasing web text into synthetic pretraining data. Our results reveal that structured output formats, such as tables, math problems, FAQs, and tutorials, consistently outperform both curated web baselines and prior synthetic methods. Notably, increasing the size of the generator model beyond 1B parameters provides no additional benefit. Our analysis also demonstrates that the selection of the original data used for mixing substantially influences performance. By applying our findings, we develop \textbf{\textsc{FinePhrase}}, a 486-billion-token open dataset of rephrased web text. We show that \textsc{FinePhrase} outperforms all existing synthetic data baselines while reducing generation costs by up to 30 times. We provide the dataset, all prompts, and the generation framework to the research community. 

---
# Causal Drawbridges: Characterizing Gradient Blocking of Syntactic Islands in Transformer LMs 

**Authors**: Sasha Boguraev, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2604.13950)  

**Abstract**: We show how causal interventions in Transformer models provide insights into English syntax by focusing on a long-standing challenge for syntactic theory: syntactic islands. Extraction from coordinated verb phrases is often degraded, yet acceptability varies gradiently with lexical content (e.g., "I know what he hates art and loves" vs. "I know what he looked down and saw"). We show that modern Transformer language models replicate human judgments across this gradient. Using causal interventions that isolate functionally relevant subspaces in Transformer blocks, attention modules, and MLPs, we demonstrate that extraction from coordination islands engages the same filler-gap mechanisms as canonical wh-dependencies, but that these mechanisms are selectively blocked to varying degrees. By projecting a large corpus of unrelated text onto these causally identified subspaces, we derive a novel linguistic hypothesis: the conjunction "and" is represented differently in extractable versus non-extractable constructions, corresponding to expressions encoding relational dependencies versus purely conjunctive uses. These results illustrate how mechanistic interpretability can inform syntax, generating new hypotheses about linguistic representation and processing. 

---
# Do We Still Need Humans in the Loop? Comparing Human and LLM Annotation in Active Learning for Hostility Detection 

**Authors**: Ahmad Dawar Hakimi, Lea Hirlimann, Isabelle Augenstein, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2604.13899)  

**Abstract**: Instruction-tuned LLMs can annotate thousands of instances from a short prompt at negligible cost. This raises two questions for active learning (AL): can LLM labels replace human labels within the AL loop, and does AL remain necessary when entire corpora can be labelled at once? We investigate both questions on a new dataset of 277,902 German political TikTok comments (25,974 LLM-labelled, 5,000 human-annotated), comparing seven annotation strategies across four encoders to detect anti-immigrant hostility. A classifier trained on 25,974 GPT-5.2 labels (\$43) achieves comparable F1-Macro to one trained on 3,800 human annotations (\$316). Active learning offers little advantage over random sampling in our pre-enriched pool and delivers lower F1 than full LLM annotation at the same cost. However, comparable aggregate F1 masks a systematic difference in error structure: LLM-trained classifiers over-predict the positive class relative to the human gold standard. This divergence concentrates in topically ambiguous discussions where the distinction between anti-immigrant hostility and policy critique is most subtle, suggesting that annotation strategy should be guided not by aggregate F1 alone but by the error profile acceptable for the target application. 

---
# Beyond Static Personas: Situational Personality Steering for Large Language Models 

**Authors**: Zesheng Wei, Mengxiang Li, Zilei Wang, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.13846)  

**Abstract**: Personalized Large Language Models (LLMs) facilitate more natural, human-like interactions in human-centric applications. However, existing personalization methods are constrained by limited controllability and high resource demands. Furthermore, their reliance on static personality modeling restricts adaptability across varying situations. To address these limitations, we first demonstrate the existence of situation-dependency and consistent situation-behavior patterns within LLM personalities through a multi-perspective analysis of persona neurons. Building on these insights, we propose IRIS, a training-free, neuron-based Identify-Retrieve-Steer framework for advanced situational personality steering. Our approach comprises situational persona neuron identification, situation-aware neuron retrieval, and similarity-weighted steering. We empirically validate our framework on PersonalityBench and our newly introduced SPBench, a comprehensive situational personality benchmark. Experimental results show that our method surpasses best-performing baselines, demonstrating IRIS's generalization and robustness to complex, unseen situations and different models architecture. 

---
# Robust Reward Modeling for Large Language Models via Causal Decomposition 

**Authors**: Yunsheng Lu, Zijiang Yang, Licheng Pan, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13833)  

**Abstract**: Reward models are central to aligning large language models, yet they often overfit to spurious cues such as response length and overly agreeable tone. Most prior work weakens these cues directly by penalizing or controlling specific artifacts, but it does not explicitly encourage the model to ground preferences in the prompt's intent. We learn a decoder that maps a candidate answer to the latent intent embedding of the input. The reconstruction error is used as a signal to regularize the reward model training. We provide theoretical evidence that this signal emphasizes prompt-dependent information while suppressing prompt-independent shortcuts. Across math, helpfulness, and safety benchmarks, the decoder selects shorter and less sycophantic candidates with 0.877 accuracy. Incorporating this signal into RM training in Gemma-2-2B-it and Gemma-2-9B-it increases RewardBench accuracy from 0.832 to 0.868. For Best-of-N selection, our framework increases length-controlled win rates while producing shorter outputs, and remains robust to lengthening and mild off-topic drift in controlled rewrite tests. 

---
# MUSE: Multi-Domain Chinese User Simulation via Self-Evolving Profiles and Rubric-Guided Alignment 

**Authors**: Zihao Liu, Hantao Zhou, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13828)  

**Abstract**: User simulators are essential for the scalable training and evaluation of interactive AI systems. However, existing approaches often rely on shallow user profiling, struggle to maintain persona consistency over long interactions, and are largely limited to English or single-domain settings. We present MUSE, a multi-domain Chinese user simulation framework designed to generate human-like, controllable, and behaviorally consistent responses. First, we propose Iterative Profile Self-Evolution (IPSE), which gradually optimizes user profiles by comparing and reasoning discrepancies between simulated trajectories and real dialogue behaviors. We then apply Role-Reversal Supervised Fine-Tuning to improve local response realism and human-like expression. To enable fine-grained behavioral alignment, we further train a specialized rubric-based reward model and incorporate it into rubric-guided multi-turn reinforcement learning, which optimizes the simulator at the dialogue level and enhances long-horizon behavioral consistency. Experiments show that MUSE consistently outperforms strong baselines in both utterance-level and session-level evaluations, generating responses that are more realistic, coherent, and persona-consistent over extended interactions. 

---
# ToolOmni: Enabling Open-World Tool Use via Agentic learning with Proactive Retrieval and Grounded Execution 

**Authors**: Shouzheng Huang, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13787)  

**Abstract**: Large Language Models (LLMs) enhance their problem-solving capability by utilizing external tools. However, in open-world scenarios with massive and evolving tool repositories, existing methods relying on static embedding retrieval or parameter memorization of tools struggle to align user intent with tool semantics or generalize to unseen tools, respectively, leading to suboptimal accuracy of open-world tool retrieval and execution. To address these, we present ToolOmni, a unified agentic framework that enables LLMs for open-world tool use by proactive retrieval and grounded execution within a reasoning loop. First, we construct a cold-start multi-turn interaction dataset to instill foundational agentic capabilities via Supervised Fine-Tuning (SFT). Then, we introduce open-world tool learning based on a Decoupled Multi-Objective GRPO algorithm, which simultaneously optimizes LLMs for both tool retrieval accuracy and execution efficacy in online environments. Extensive experiments demonstrate that ToolOmni achieves state-of-the-art performance both in retrieval and execution, surpassing strong baselines by a significant margin of +10.8% in end-to-end execution success rate, while exhibiting exceptional robustness and generalization capabilities. 

---
# QuantileMark: A Message-Symmetric Multi-bit Watermark for LLMs 

**Authors**: Junlin Zhu, Baizhou Huang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13786)  

**Abstract**: As large language models become standard backends for content generation, practical provenance increasingly requires multi-bit watermarking. In provider-internal deployments, a key requirement is message symmetry: the message itself should not systematically affect either text quality or verification outcomes. Vocabulary-partition watermarks can break message symmetry in low-entropy decoding: some messages are assigned most of the probability mass, while others are forced to use tail tokens. This makes embedding quality and message decoding accuracy message-dependent. We propose QuantileMark, a white-box multi-bit watermark that embeds messages within the continuous cumulative probability interval $[0, 1)$. At each step, QuantileMark partitions this interval into $M$ equal-mass bins and samples strictly from the bin assigned to the target symbol, ensuring a fixed $1/M$ probability budget regardless of context entropy. For detection, the verifier reconstructs the same partition under teacher forcing, computes posteriors over latent bins, and aggregates evidence for verification. We prove message-unbiasedness, a property ensuring that the base distribution is recovered when averaging over messages. This provides a theoretical foundation for generation-side symmetry, while the equal-mass design additionally promotes uniform evidence strength across messages on the detection side. Empirical results on C4 continuation and LFQA show improved multi-bit recovery and detection robustness over strong baselines, with negligible impact on generation quality. Our code is available at GitHub (this https URL). 

---
# From Anchors to Supervision: Memory-Graph Guided Corpus-Free Unlearning for Large Language Models 

**Authors**: Wenxuan Li, Zhenfei Zhang, Mi Zhang, Geng Hong, Mi Wen, Xiaoyu You, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13777)  

**Abstract**: Large language models (LLMs) may memorize sensitive or copyrighted content, raising significant privacy and legal concerns. While machine unlearning has emerged as a potential remedy, prevailing paradigms rely on user-provided forget sets, making unlearning requests difficult to audit and exposing systems to secondary leakage and malicious abuse. We propose MAGE, a Memory-grAph Guided Erasure framework for user-minimized, corpus-free unlearning. Given only a lightweight user anchor that identifies a target entity, MAGE probes the target LLM to recover target-related memorization, organizes it into a weighted local memory graph, and synthesizes scoped supervision for unlearning. MAGE is model-agnostic, can be plugged into standard unlearning methods, and requires no access to the original training corpus. Experiments on two benchmarks, TOFU and RWKU, demonstrate that MAGE's self-generated supervision achieves effective unlearning performance comparable to supervision generated with external reference, while preserving overall utility. These results support a practical and auditable unlearning workflow driven by minimal anchors rather than user-supplied forget corpora. 

---
# MedRCube: A Multidimensional Framework for Fine-Grained and In-Depth Evaluation of MLLMs in Medical Imaging 

**Authors**: Zhijie Bao, Fangke Chen, Licheng Bao, Chenhui Zhang, Wei Chen, Jiajie Peng, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.13756)  

**Abstract**: The potential of Multimodal Large Language Models (MLLMs) in domain of medical imaging raise the demands of systematic and rigorous evaluation frameworks that are aligned with the real-world medical imaging practice. Existing practices that report single or coarse-grained metrics are lack the granularity required for specialized clinical support and fail to assess the reliability of reasoning mechanisms. To address this, we propose a paradigm shift toward multidimensional, fine-grained and in-depth evaluation. Based on a two-stage systematic construction pipeline designed for this paradigm, we instantiate it with MedRCube. We benchmark 33 MLLMs, \textit{Lingshu-32B} achieve top-tier performance. Crucially, MedRCube exposes a series of pronounced insights inaccessible under prior evaluation settings. Furthermore, we introduce a credibility evaluation subset to quantify reasoning credibility, uncover a highly significant positive association between shortcut behavior and diagnostic task performance, raising concerns for clinically trustworthy deployment. The resources of this work can be found at this https URL. 

---
# Doc-V*:Coarse-to-Fine Interactive Visual Reasoning for Multi-Page Document VQA 

**Authors**: Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei, Zhenbo Luo, Jian Luan, Wei Chen, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2604.13731)  

**Abstract**: Multi-page Document Visual Question Answering requires reasoning over semantics, layouts, and visual elements in long, visually dense documents. Existing OCR-free methods face a trade-off between capacity and precision: end-to-end models scale poorly with document length, while visual retrieval-based pipelines are brittle and passive. We propose Doc-$V^*$, an \textbf{OCR-free agentic} framework that casts multi-page DocVQA as sequential evidence aggregation. Doc-$V^*$ begins with a thumbnail overview, then actively navigates via semantic retrieval and targeted page fetching, and aggregates evidence in a structured working memory for grounded reasoning. Trained by imitation learning from expert trajectories and further optimized with Group Relative Policy Optimization, Doc-$V^*$ balances answer accuracy with evidence-seeking efficiency. Across five benchmarks, Doc-$V^*$ outperforms open-source baselines and approaches proprietary models, improving out-of-domain performance by up to \textbf{47.9\%} over RAG baseline. Other results reveal effective evidence aggregation with selective attention, not increased input pages. 

---
# An Empirical Investigation of Practical LLM-as-a-Judge Improvement Techniques on RewardBench 2 

**Authors**: Ryan Lail  

**Link**: [PDF](https://arxiv.org/pdf/2604.13717)  

**Abstract**: LLM-as-a-judge, using a language model to score or rank candidate responses, is widely used as a scalable alternative to human evaluation in RLHF pipelines, benchmarking, and application layer evaluations (evals). However, judgment reliability depends heavily on prompting and aggregation strategy. We present an empirical investigation of practical, drop-in techniques that improve GPT-5.4 judge accuracy on RewardBench 2 without any finetuning. Two techniques account for nearly all available gains: task-specific criteria injection (+3.0pp at negligible cost) and ensemble scoring (+9.8pp at 5x cost). Combined, they reach 83.6% accuracy, +11.9pp over the 71.7% baseline. Our investigation also covers three further techniques (calibration context, adaptive model escalation, and soft blending) which did not reliably improve on criteria + ensembling at comparable cost. Cheaper model tiers benefit disproportionately from ensembling: GPT-5.4 mini with k=8 achieves 79.2% at 1.2x baseline cost, and GPT-5.4 nano with k=8 reaches 71.4% at 0.4x baseline cost, making high-accuracy LLM judges accessible at low cost. 

---
# Learning the Cue or Learning the Word? Analyzing Generalization in Metaphor Detection for Verbs 

**Authors**: Sinan Kurtyigit, Sabine Schulte im Walde, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2604.13713)  

**Abstract**: Metaphor detection models achieve strong benchmark performance, yet it remains unclear whether this reflects transferable generalization or lexical memorization. To address this, we analyze generalization in metaphor detection through RoBERTa, the shared backbone of many state-of-the-art systems, focusing on English verbs using the VU Amsterdam Metaphor Corpus. We introduce a controlled lexical hold-out setup where all instances of selected target lemmas are strictly excluded from fine-tuning, and compare predictions on these Held-out lemmas against Exposed lemmas (verbs seen during fine-tuning). While the model performs best on Exposed lemmas, it maintains robust performance on Held-out lemmas. Further analysis reveals that sentence context alone is sufficient to match full-model performance on Held-out lemmas, whereas static verb-level embeddings are not. Together, these results suggest that generalization is primarily driven by "learning the cue" (transferable contextual patterns), while "learning the word" (verb-specific memorization) provides an additive boost when lexical exposure is available. 

---
# Co-FactChecker: A Framework for Human-AI Collaborative Claim Verification Using Large Reasoning Models 

**Authors**: Dhruv Sahnan, Subhabrata Dutta, Tanmoy Chakraborty, Preslav Nakov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2604.13706)  

**Abstract**: Professional fact-checkers rely on domain knowledge and deep contextual understanding to verify claims. Large language models (LLMs) and large reasoning models (LRMs) lack such grounding and primarily reason from available evidence alone, creating a mismatch between expert-led and fully automated claim verification. To mitigate this gap, we posit human-AI collaboration as a more promising path forward, where expert feedback, grounded in real-world knowledge and domain expertise, guides the model's reasoning. However, existing LRMs are hard to calibrate to natural language feedback, particularly in a multi-turn interaction setup. We propose Co-FactChecker, a framework for human-AI collaborative claim verification. We introduce a new interaction paradigm that treats the model's thinking trace as a shared scratchpad. Co-FactChecker translates expert feedback into trace-edits that introduce targeted modifications to the trace, sidestepping the shortcomings of dialogue-based interaction. We provide theoretical results showing that trace-editing offers advantages over multi-turn dialogue, and our automatic evaluations demonstrate that Co-FactChecker outperforms existing autonomous and human-AI collaboration approaches. Human evaluations further show that Co-FactChecker is preferred over multi-turn dialogue, producing higher quality reasoning and verdicts along with relatively easier to interpret and more useful thinking traces. 

---
# Beyond Arrow's Impossibility: Fairness as an Emergent Property of Multi-Agent Collaboration 

**Authors**: Sayan Kumar Chaki, Antoine Gourru, Julien Velcin  

**Link**: [PDF](https://arxiv.org/pdf/2604.13705)  

**Abstract**: Fairness in language models is typically studied as a property of a single, centrally optimized model. As large language models become increasingly agentic, we propose that fairness emerges through interaction and exchange. We study this via a controlled hospital triage framework in which two agents negotiate over three structured debate rounds. One agent is aligned to a specific ethical framework via retrieval-augmented generation (RAG), while the other is either unaligned or adversarially prompted to favor demographic groups over clinical need. We find that alignment systematically shapes negotiation strategies and allocation patterns, and that neither agent's allocation is ethically adequate in isolation, yet their joint final allocation can satisfy fairness criteria that neither would have reached alone. Aligned agents partially moderate bias through contestation rather than override, acting as corrective patches that restore access for marginalized groups without fully converting a biased counterpart. We further observe that even explicitly aligned agents exhibit intrinsic biases toward certain frameworks, consistent with known left-leaning tendencies in LLMs. We connect these limits to Arrow's Impossibility Theorem: no aggregation mechanism can simultaneously satisfy all desiderata of collective rationality, and multi-agent deliberation navigates rather than resolves this constraint. Our results reposition fairness as an emergent, procedural property of decentralized agent interaction, and the system rather than the individual agent as the appropriate unit of evaluation. 

---
# Breaking the Generator Barrier: Disentangled Representation for Generalizable AI-Text Detection 

**Authors**: Xiao Pu, Zepeng Cheng, Lin Yuan, Yu Wu, Xiuli Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.13692)  

**Abstract**: As large language models (LLMs) generate text that increasingly resembles human writing, the subtle cues that distinguish AI-generated content from human-written content become increasingly challenging to capture. Reliance on generator-specific artifacts is inherently unstable, since new models emerge rapidly and reduce the robustness of such shortcuts. This generalizes unseen generators as a central and challenging problem for AI-text detection. To tackle this challenge, we propose a progressively structured framework that disentangles AI-detection semantics from generator-aware artifacts. This is achieved through a compact latent encoding that encourages semantic minimality, followed by perturbation-based regularization to reduce residual entanglement, and finally a discriminative adaptation stage that aligns representations with task objectives. Experiments on MAGE benchmark, covering 20 representative LLMs across 7 categories, demonstrate consistent improvements over state-of-the-art methods, achieving up to 24.2% accuracy gain and 26.2% F1 improvement. Notably, performance continues to improve as the diversity of training generators increases, confirming strong scalability and generalization in open-set scenarios. Our source code will be publicly available at this https URL. 

---
# IndicDB -- Benchmarking Multilingual Text-to-SQL Capabilities in Indian Languages 

**Authors**: Aviral Dawar, Roshan Karanth, Vikram Goyal, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.13686)  

**Abstract**: While Large Language Models (LLMs) have significantly advanced Text-to-SQL performance, existing benchmarks predominantly focus on Western contexts and simplified schemas, leaving a gap in real-world, non-Western applications. We present IndicDB, a multilingual Text-to-SQL benchmark for evaluating cross-lingual semantic parsing across diverse Indic languages. The relational schemas are sourced from open-data platforms, including the National Data and Analytics Platform (NDAP) and the India Data Portal (IDP), ensuring realistic administrative data complexity. IndicDB comprises 20 databases across 237 tables. To convert denormalized government data into rich relational structures, we employ an iterative three-agent framework (Architect, Auditor, Refiner) to ensure structural rigor and high relational density (11.85 tables per database; join depths up to six). Our pipeline is value-aware, difficulty-calibrated, and join-enforced, generating 15,617 tasks across English, Hindi, and five Indic languages. We evaluate cross-lingual semantic parsing performance of state-of-the-art models (DeepSeek v3.2, MiniMax 2.7, LLaMA 3.3, Qwen3) across seven linguistic variants. Results show a 9.00% performance drop from English to Indic languages, revealing an "Indic Gap" driven by harder schema linking, increased structural ambiguity, and limited external knowledge. IndicDB serves as a rigorous benchmark for multilingual Text-to-SQL. Code and data: this https URL 

---
# Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference 

**Authors**: Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13634)  

**Abstract**: Speculative decoding accelerates autoregressive generation by letting draft tokens bypass full verification, but conventional frameworks suffer from frequent false rejections, particularly when draft models produce semantically correct but lexically divergent outputs. In this paper, we present Calibrated Speculative Decoding (CSD), a training-free framework that recovers valid tokens discarded by standard verification. Guided by the principle of "Frequency-Guided Candidate Selection and Probability-Guarded Acceptance," CSD incorporates two lightweight modules: Online Correction Memory, which aggregates historical rejections to propose recurring divergence patterns as rescue candidates, and Semantic Consistency Gating, which verifies candidate admissibility using probability ratios instead of exact token matching. Our evaluation across diverse large language models demonstrates that CSD outperforms existing methods, achieving a peak throughput speedup of 2.33x. CSD preserves model accuracy across all tasks while further boosting performance on complex reasoning datasets. These results establish CSD as a highly effective, lightweight solution for practical LLM deployments. 

---
# Syn-TurnTurk: A Synthetic Dataset for Turn-Taking Prediction in Turkish Dialogues 

**Authors**: Ahmet Tuğrul Bayrak, Mustafa Sertaç Türkel, Fatma Nur Korkmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.13620)  

**Abstract**: Managing natural dialogue timing is a significant challenge for voice-based chatbots. Most current systems usually rely on simple silence detection, which often fails because human speech patterns involve irregular pauses. This causes bots to interrupt users, breaking the conversational flow. This problem is even more severe for languages like Turkish, which lack high-quality datasets for turn-taking prediction. This paper introduces Syn-TurnTurk, a synthetic Turkish dialogue dataset generated using various Qwen Large Language Models (LLMs) to mirror real-life verbal exchanges, including overlaps and strategic silences. We evaluated the dataset using several traditional and deep learning architectures. The results show that advanced models, particularly BI-LSTM and Ensemble (LR+RF) methods, achieve high accuracy (0.839) and AUC scores (0.910). These findings demonstrate that our synthetic dataset can have a positive affect for models understand linguistic cues, allowing for more natural human-machine interaction in Turkish. 

---
# C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences 

**Authors**: Akira Kawabata, Saku Sugawara  

**Link**: [PDF](https://arxiv.org/pdf/2604.13618)  

**Abstract**: Rubric-augmented verification guides reward models with explicit evaluation criteria, yielding more reliable judgments than single-model verification. However, most existing methods require costly rubric annotations, limiting scalability. Moreover, we find that rubric generation is vulnerable to a failure of cooperation; low-quality rubrics actively mislead reward models rather than help. Inspired by the principle of cooperative communication, we propose Cooperative yet Critical reward modeling (C2), a framework that significantly improves reward model judgments by having the reward model critically collaborate with a rubric generator trained solely from binary preferences. In C2, we synthesize helpful and misleading rubric pairs by measuring how each rubric shifts the reward model toward or away from the correct preference. Using these contrastive pairs, we train a cooperative rubric generator to propose helpful rubrics, and a critical verifier to assess rubric validity before making its judgment, following only rubrics it deems helpful at inference time. C2 outperforms reasoning reward models trained on the same binary preferences, with gains of up to 6.5 points on RM-Bench and 6.0 points length-controlled win rate on AlpacaEval 2.0. Without external rubric annotations, C2 enables an 8B reward model to match performance achieved with rubrics from a 4$\times$ larger model. Overall, our work demonstrates that eliciting deliberate cooperation in rubric-augmented verification makes reward models more trustworthy in a scalable way. 

---
# Foresight Optimization for Strategic Reasoning in Large Language Models 

**Authors**: Jiashuo Wang, Jiawen Duan, Jian Wang, Kaitao Song, Chunpu Xu, Johnny K. W. Ho, Fenggang Yu, Wenjie Li, Johan F. Hoorn  

**Link**: [PDF](https://arxiv.org/pdf/2604.13592)  

**Abstract**: Reasoning capabilities in large language models (LLMs) have generally advanced significantly. However, it is still challenging for existing reasoning-based LLMs to perform effective decision-making abilities in multi-agent environments, due to the absence of explicit foresight modeling. To this end, strategic reasoning, the most fundamental capability to anticipate the counterpart's behaviors and foresee its possible future actions, has been introduced to alleviate the above issues. Strategic reasoning is fundamental to effective decision-making in multi-agent environments, yet existing reasoning enhancement methods for LLMs do not explicitly capture its foresight nature. In this work, we introduce Foresight Policy Optimization (FoPO) to enhance strategic reasoning in LLMs, which integrates opponent modeling principles into policy optimization, thereby enabling explicit consideration of both self-interest and counterpart influence. Specifically, we construct two curated datasets, namely Cooperative RSA and Competitive Taboo, equipped with well-designed rules and moderate difficulty to facilitate a systematic investigation of FoPO in a self-play framework. Our experiments demonstrate that FoPO significantly enhances strategic reasoning across LLMs of varying sizes and origins. Moreover, models trained with FoPO exhibit strong generalization to out-of-domain strategic scenarios, substantially outperforming standard LLM reasoning optimization baselines. 

---
# BenGER: A Collaborative Web Platform for End-to-End Benchmarking of German Legal Tasks 

**Authors**: Sebastian Nagl, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2604.13583)  

**Abstract**: Evaluating large language models (LLMs) for legal reasoning requires workflows that span task design, expert annotation, model execution, and metric-based evaluation. In practice, these steps are split across platforms and scripts, limiting transparency, reproducibility, and participation by non-technical legal experts. We present the BenGER (Benchmark for German Law) framework, an open-source web platform that integrates task creation, collaborative annotation, configurable LLM runs, and evaluation with lexical, semantic, factual, and judge-based metrics. BenGER supports multi-organization projects with tenant isolation and role-based access control, and can optionally provide formative, reference-grounded feedback to annotators. We will demonstrate a live deployment showing end-to-end benchmark creation and analysis. 

---
# MM-Doc-R1: Training Agents for Long Document Visual Question Answering through Multi-turn Reinforcement Learning 

**Authors**: Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhiheng Xi, Honglin Guo, Shichun Liu, Junzhe Wang, Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13579)  

**Abstract**: Conventional Retrieval-Augmented Generation (RAG) systems often struggle with complex multi-hop queries over long documents due to their single-pass retrieval. We introduce MM-Doc-R1, a novel framework that employs an agentic, vision-aware workflow to address long document visual question answering through iterative information discovery and synthesis. To incentivize the information seeking capabilities of our agents, we propose Similarity-based Policy Optimization (SPO), addressing baseline estimation bias in existing multi-turn reinforcement learning (RL) algorithms like GRPO. Our core insight is that in multi-turn RL, the more semantically similar two trajectories are, the more accurate their shared baseline estimation becomes. Leveraging this, SPO calculates a more precise baseline by similarity-weighted averaging of rewards across multiple trajectories, unlike GRPO which inappropriately applies the initial state's baseline to all intermediate states. This provides a more stable and accurate learning signal for our agents, leading to superior training performance that surpasses GRPO. Our experiments on the MMLongbench-Doc benchmark show that MM-Doc-R1 outperforms previous baselines by 10.4%. Furthermore, SPO demonstrates superior performance over GRPO, boosting results by 5.0% with Qwen3-8B and 6.1% with Qwen3-4B. These results highlight the effectiveness of our integrated framework and novel training algorithm in advancing the state-of-the-art for complex, long-document visual question answering. 

---
# YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference 

**Authors**: You Wu, Ziheng Chen, Yizhen Zhang, Haoyi Wu, Chengting Yu, Yuchi Xu, Wenbo Su, Bo Zheng, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13556)  

**Abstract**: Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models (LLMs). Although they reduce the memory consumption of the KV cache, such methods usually introduce non-negligible performance degradation. In this work, we aim to enhance the performance of YOCO, a cross-layer KV compression method that shares the KVs of the middle layer with the top-half layers. We propose YOCO++, an enhanced YOCO that incorporates a weighted residual connection between the KVs of each bottom-half layer and the bottom layer. Compared to YOCO, YOCO++ increases model capacity while maintaining the same training and inference efficiency. Our experiments show that YOCO++ achieves state-of-the-art performance among the cross-layer KV compression methods at a 50% KV cache compression rate, outperforming the standard Transformer. 

---
# Training-Free Test-Time Contrastive Learning for Large Language Models 

**Authors**: Kaiwen Zheng, Kai Zhou, Jinwu Hu, Te Gu, Mingkai Peng, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13552)  

**Abstract**: Large language models (LLMs) demonstrate strong reasoning capabilities, but their performance often degrades under distribution shift. Existing test-time adaptation (TTA) methods rely on gradient-based updates that require white-box access and need substantial overhead, while training-free alternatives are either static or depend on external guidance. In this paper, we propose Training-Free Test-Time Contrastive Learning TF-TTCL, a training-free adaptation framework that enables a frozen LLM to improve online by distilling supervision from its own inference experiences. Specifically, TF-TTCL implements a dynamic "Explore-Reflect-Steer" loop through three core modules: 1) Semantic Query Augmentation first diversifies problem views via multi-agent role-playing to generate different reasoning trajectories; 2) Contrastive Experience Distillation then captures the semantic gap between superior and inferior trajectories, distilling them into explicit textual rules; and 3) Contextual Rule Retrieval finally activates these stored rules during inference to dynamically steer the frozen LLM toward robust reasoning patterns while avoiding observed errors. Extensive experiments on closed-ended reasoning tasks and open-ended evaluation tasks demonstrate that TF-TTCL consistently outperforms strong zero-shot baselines and representative TTA methods under online evaluation. Code is available at this https URL. 

---
# Debate to Align: Reliable Entity Alignment through Two-Stage Multi-Agent Debate 

**Authors**: Cunda Wang, Ziying Ma, Po Hu, Weihua Wang, Feilong Bao  

**Link**: [PDF](https://arxiv.org/pdf/2604.13551)  

**Abstract**: Entity alignment (EA) aims to identify entities referring to the same real-world object across different knowledge graphs (KGs). Recent approaches based on large language models (LLMs) typically obtain entity embeddings through knowledge representation learning and use embedding similarity to identify an alignment-uncertain entity set. For each uncertain entity, a candidate entity set (CES) is then retrieved based on embedding similarity to support subsequent alignment reasoning and decision making. However, the reliability of the CES and the reasoning capability of LLMs critically affect the effectiveness of subsequent alignment decisions. To address this issue, we propose AgentEA, a reliable EA framework based on multi-agent debate. AgentEA first improves embedding quality through entity representation preference optimization, and then introduces a two-stage multi-role debate mechanism consisting of lightweight debate verification and deep debate alignment to progressively enhance the reliability of alignment decisions while enabling more efficient debate-based reasoning. Extensive experiments on public benchmarks under cross-lingual, sparse, large-scale, and heterogeneous settings demonstrate the effectiveness of AgentEA. 

---
# Synthesizing Instruction-Tuning Datasets with Contrastive Decoding 

**Authors**: Tatsuya Ichinose, Youmi Ma, Masanari Oi, Ryuto Koike, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.13538)  

**Abstract**: Using responses generated by high-performing large language models (LLMs) for instruction tuning has become a widely adopted approach. However, the existing literature overlooks a property of LLM-generated responses: they conflate world knowledge acquired during pre-training with instruction-following capabilities acquired during post-training. We hypothesize that disentangling the instruction-following capabilities from pre-trained knowledge improves the effectiveness of instruction tuning. To this end, we propose CoDIT, a method that applies contrastive decoding between a post-trained model and its pre-trained counterpart during response generation. The method suppresses pre-trained knowledge shared between the two models while amplifying the instruction-following behavior acquired via post-training, resulting in responses that more purely reflect instruction-following capabilities. Experiment results demonstrate that models trained on datasets constructed via CoDIT consistently outperform those trained on directly generated responses. Training on our datasets also yields better performance than on existing publicly available instruction-tuning datasets across multiple benchmarks. Furthermore, we theoretically and empirically show that CoDIT can be interpreted as distilling the chat vector from parameter space to text space, enabling the transfer of instruction-tuning capabilities across models of different architectures. 

---
# ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding 

**Authors**: Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.13519)  

**Abstract**: Tool calling has greatly expanded the practical utility of large language models (LLMs) by enabling them to interact with external applications. As LLM capabilities advance, effective tool use increasingly involves multi-step, multi-turn interactions to solve complex tasks. However, the resulting growth in tool interactions incurs substantial latency, posing a key challenge for real-time LLM serving. Through empirical analysis, we find that tool-calling traces are highly structured, conform to constrained schemas, and often exhibit recurring invocation patterns. Motivated by this, we propose ToolSpec, a schema-aware, retrieval-augmented speculative decoding method for accelerating tool calling. ToolSpec exploits predefined tool schemas to generate accurate drafts, using a finite-state machine to alternate between deterministic schema token filling and speculative generation for variable fields. In addition, ToolSpec retrieves similar historical tool invocations and reuses them as drafts to further improve efficiency. ToolSpec presents a plug-and-play solution that can be seamlessly integrated into existing LLM workflows. Experiments across multiple benchmarks demonstrate that ToolSpec achieves up to a 4.2x speedup, substantially outperforming existing training-free speculative decoding methods. 

---
# Using reasoning LLMs to extract SDOH events from clinical notes 

**Authors**: Ertan Doganl, Kunyu Yu, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.13502)  

**Abstract**: Social Determinants of Health (SDOH) refer to environmental, behavioral, and social conditions that influence how individuals live, work, and age. SDOH have a significant impact on personal health outcomes, and their systematic identification and management can yield substantial improvements in patient care. However, SDOH information is predominantly captured in unstructured clinical notes within electronic health records, which limits its direct use as machine-readable entities. To address this issue, researchers have employed Natural Language Processing (NLP) techniques using pre-trained BERT-based models, demonstrating promising performance but requiring sophisticated implementation and extensive computational resources. In this study, we investigated prompt engineering strategies for extracting structured SDOH events utilizing LLMs with advanced reasoning capabilities. Our method consisted of four modules: 1) developing concise and descriptive prompts integrated with established guidelines, 2) applying few-shot learning with carefully curated examples, 3) using a self-consistency mechanism to ensure robust outputs, and 4) post-processing for quality control. Our approach achieved a micro-F1 score of 0.866, demonstrating competitive performance compared to the leading models. The results demonstrated that LLMs with reasoning capabilities are effective solutions for SDOH event extraction, offering both implementation simplicity and strong performance. 

---
# CANVAS: Continuity-Aware Narratives via Visual Agentic Storyboarding 

**Authors**: Ishani Mondal, Yiwen Song, Mihir Parmar, Palash Goyal, Jordan Boyd-Graber, Tomas Pfister, Yale Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.13452)  

**Abstract**: Long-form visual storytelling requires maintaining continuity across shots, including consistent characters, stable environments, and smooth scene transitions. While existing generative models can produce strong individual frames, they fail to preserve such continuity, leading to appearance changes, inconsistent backgrounds, and abrupt scene shifts. We introduce CANVAS (Continuity-Aware Narratives via Visual Agentic Storyboarding), a multi-agent framework that explicitly plans visual continuity in multi-shot narratives. CANVAS enforces coherence through character continuity, persistent background anchors, and location-aware scene planning for smooth transitions within the same setting We evaluate CANVAS on two storyboard generation benchmarks ST-BENCH and ViStoryBench and introduce a new challenging benchmark HardContinuityBench for long-range narrative consistency. CANVAS consistently outperforms the best-performing baseline, improving background continuity by 21.6%, character consistency by 9.6% and props consistency by 7.6%. 

---
# MERRIN: A Benchmark for Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments 

**Authors**: Han Wang, David Wan, Hyunji Lee, Thinh Pham, Mikaela Cankosyan, Weiyuan Chen, Elias Stengel-Eskin, Tu Vu, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.13418)  

**Abstract**: Motivated by the underspecified, multi-hop nature of search queries and the multimodal, heterogeneous, and often conflicting nature of real-world web results, we introduce MERRIN (Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments), a human-annotated benchmark for evaluating search-augmented agents. MERRIN measures AI agents' ability to identify relevant modalities, retrieve multimodal evidence, and perform multi-hop reasoning over noisy web sources. It differs from prior work in three important aspects: (1) using natural language queries without explicit modality cues, (2) incorporating underexplored modalities such as video and audio, and (3) requiring the retrieval of complex, often noisy or conflicting multimodal evidence during web search. We evaluate diverse search agents powered by ten models, including strong closed-source models (e.g., GPT-5.4-mini, Gemini 3/3.1 Flash/Pro) and open-weight models (Qwen3-4B/30B/235B), across three search settings (no search, native search, and agentic search). Our results show that MERRIN is highly challenging: the average accuracy across all agents is 22.3%, with the best-performing agent reaching only 40.1%. We further observe that while stronger agents like Gemini Deep Research achieve higher performance, gains are modest due to over-exploration; they take more steps and use more tools, but are often distracted by conflicting or partially relevant web content, leading to incorrect answers. Compared to humans, these agents consume more resources yet achieve lower accuracy, largely due to inefficient source selection and an overreliance on text modalities. These findings highlight the need for search agents capable of robust search and reasoning across diverse modalities in noisy web environments, making MERRIN a valuable testbed for evaluating such capabilities. 

---
# From Prediction to Justification: Aligning Sentiment Reasoning with Human Rationale via Reinforcement Learning 

**Authors**: Shihao Zhang, Ziwei Wang, Jie Zhou, Yulan Wu, Qin Chen, Zhikai Lei, Liyang Yu, Liang Dou, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.13398)  

**Abstract**: While Aspect-based Sentiment Analysis (ABSA) systems have achieved high accuracy in identifying sentiment polarities, they often operate as "black boxes," lacking the explicit reasoning capabilities characteristic of human affective cognition. Humans do not merely categorize sentiment; they construct causal explanations for their judgments. To bridge this gap, we propose ABSA-R1, a large language model framework designed to mimic this ``reason-before-predict" cognitive process. By leveraging reinforcement learning (RL), ABSA-R1 learns to articulate the why behind the what, generating natural language justifications that ground its sentiment predictions. We introduce a Cognition-Aligned Reward Model (formerly sentiment-aware reward model) that enforces consistency between the generated reasoning path and the final emotional label. Furthermore, inspired by metacognitive monitoring, we implement a performance-driven rejection sampling strategy that selectively targets hard cases where the model's internal reasoning is uncertain or inconsistent. Experimental results on four benchmarks demonstrate that equipping models with this explicit reasoning capability not only enhances interpretability but also yields superior performance in sentiment classification and triplet extraction compared to non-reasoning baselines. 

---
# Empirical Evidence of Complexity-Induced Limits in Large Language Models on Finite Discrete State-Space Problems with Explicit Validity Constraints 

**Authors**: Md. Fahad Ullah Utsho, Mohd. Ruhul Ameen, Akif Islam, Md. Golam Rashed, Dipankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.13371)  

**Abstract**: Large Language Models (LLMs) are increasingly described as possessing strong reasoning capabilities, supported by high performance on mathematical, logical, and planning benchmarks. However, most existing evaluations rely on aggregate accuracy over fixed datasets, obscuring how reasoning behavior evolves as task complexity increases. In this work, we introduce a controlled benchmarking framework to systematically evaluate the robustness of reasoning in Large Reasoning Models (LRMs) under progressively increasing problem complexity. We construct a suite of nine classical reasoning tasks: Boolean Satisfiability, Cryptarithmetic, Graph Coloring, River Crossing, Tower of Hanoi, Water Jug, Checker Jumping, Sudoku, and Rubik's Cube, each parameterized to precisely control complexity while preserving underlying semantics. Using deterministic validators, we evaluate multiple open and proprietary LRMs across low, intermediate, and high complexity regimes, ensuring that only fully valid solutions are accepted. Our results reveal a consistent phase transition like behavior: models achieve high accuracy at low complexity but degrade sharply beyond task specific complexity thresholds. We formalize this phenomenon as reasoning collapse. Across tasks, we observe substantial accuracy declines, often exceeding 50%, accompanied by inconsistent reasoning traces, constraint violations, loss of state tracking, and confidently incorrect outputs. Increased reasoning length does not reliably improve correctness, and gains in one problem family do not generalize to others. These findings highlight the need for evaluation methodologies that move beyond static benchmarks and explicitly measure reasoning robustness under controlled complexity. 

---
# TLoRA+: A Low-Rank Parameter-Efficient Fine-Tuning Method for Large Language Models 

**Authors**: Yarui Cao, Kai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13368)  

**Abstract**: Fine-tuning large language models (LLMs) aims to adapt pre-trained models to specific tasks using relatively small and domain-specific datasets. Among Parameter-Efficient Fine-Tuning (PEFT) methods, Low-Rank Adaptation (LoRA) stands out by matching the performance of full fine-tuning while avoiding additional inference latency. In this paper, we propose a novel PEFT method that incorporates the TLoRA+ optimizer into the weight matrices of pre-trained models. The proposed approach not only preserves the efficiency of low-rank adaptation but also further enhances performance without significantly increasing computational cost. We conduct experiments on the GLUE benchmark across diverse model architectures. Numerical experiments consistently demonstrate the effectiveness and robustness of our proposed method. 

---
# Peer-Predictive Self-Training for Language Model Reasoning 

**Authors**: Shi Feng, Hanlin Zhang, Fan Nie, Sham Kakade, Yiling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.13356)  

**Abstract**: Mechanisms for continued self-improvement of language models without external supervision remain an open challenge. We propose Peer-Predictive Self-Training (PST), a label-free fine-tuning framework in which multiple language models improve collaboratively by leveraging a cross-model aggregated response as an internal training signal. Given a prompt question, the models generate responses sequentially; the final aggregated answer, often more reliable than individual responses in practice, serves as an internal target for learning. We measure how informative each intermediate response is about the aggregate using pointwise mutual information (PMI), and use this signal to scale self-training updates. Responses already aligned with the aggregate are updated less, while less informative or misaligned responses are updated more. On mathematical reasoning benchmarks (SimulEq, Math500, and MultiArith), PST improves exact-match accuracy by 2.2 to 4.3 percentage points across Gemma-2-2B, LLaMA-3.2-1B, and Qwen-2.5-1.5B, and reduces the average generator-verifier gap (GV-Gap) by 26 to 40 percent, while requiring no external supervision or teacher-student hierarchy and relying solely on cross-model interactions. These results suggest that cross-model generations and peer-predictive feedback can serve as an effective approach for self-supervised training. 

---
# AgentSPEX: An Agent SPecification and EXecution Language 

**Authors**: Pengcheng Wang, Jerry Huang, Jiarui Yao, Rui Pan, Peizhi Niu, Yaowenqi Liu, Ruida Wang, Renhao Lu, Yuwei Guo, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13346)  

**Abstract**: Language-model agent systems commonly rely on reactive prompting, in which a single instruction guides the model through an open-ended sequence of reasoning and tool-use steps, leaving control flow and intermediate state implicit and making agent behavior potentially difficult to control. Orchestration frameworks such as LangGraph, DSPy, and CrewAI impose greater structure through explicit workflow definitions, but tightly couple workflow logic with Python, making agents difficult to maintain and modify. In this paper, we introduce AgentSPEX, an Agent SPecification and EXecution Language for specifying LLM-agent workflows with explicit control flow and modular structure, along with a customizable agent harness. AgentSPEX supports typed steps, branching and loops, parallel execution, reusable submodules, and explicit state management, and these workflows execute within an agent harness that provides tool access, a sandboxed virtual environment, and support for checkpointing, verification, and logging. Furthermore, we provide a visual editor with synchronized graph and workflow views for authoring and inspection. We include ready-to-use agents for deep research and scientific research, and we evaluate AgentSPEX on 7 benchmarks. Finally, we show through a user study that AgentSPEX provides a more interpretable and accessible workflow-authoring paradigm than a popular existing agent framework. 

---
# Giving Voice to the Constitution: Low-Resource Text-to-Speech for Quechua and Spanish Using a Bilingual Legal Corpus 

**Authors**: John E. Ortega, Rodolfo Zevallos, Fabricio Carraro  

**Link**: [PDF](https://arxiv.org/pdf/2604.13288)  

**Abstract**: We present a unified pipeline for synthesizing high-quality Quechua and Spanish speech for the Peruvian Constitution using three state-of-the-art text-to-speech (TTS) architectures: XTTS v2, F5-TTS, and DiFlow-TTS. Our models are trained on independent Spanish and Quechua speech datasets with heterogeneous sizes and recording conditions, and leverage bilingual and multilingual TTS capabilities to improve synthesis quality in both languages. By exploiting cross-lingual transfer, our framework mitigates data scarcity in Quechua while preserving naturalness in Spanish. We release trained checkpoints, inference code, and synthesized audio for each constitutional article, providing a reusable resource for speech technologies in indigenous and multilingual contexts. This work contributes to the development of inclusive TTS systems for political and legal content in low-resource settings. 

---
# English is Not All You Need: Systematically Exploring the Role of Multilinguality in LLM Post-Training 

**Authors**: Mehak Dhaliwal, Shashwat Chaurasia, Yao Qin, Dezhi Hong, Thomas Butler  

**Link**: [PDF](https://arxiv.org/pdf/2604.13286)  

**Abstract**: Despite the widespread multilingual deployment of large language models, post-training pipelines remain predominantly English-centric, contributing to performance disparities across languages. We present a systematic, controlled study of the interplay between training language coverage, model scale, and task domain, based on 220 supervised fine-tuning runs on parallel translated multilingual data mixtures spanning mathematical reasoning and API calling tasks, with models up to 8B parameters. We find that increasing language coverage during post-training is largely beneficial across tasks and model scales, with low-resource languages benefiting the most and high-resource languages plateauing rather than degrading. Even minimal multilinguality helps: incorporating a single non-English language improves both English performance and cross-lingual generalization, making English-only post-training largely suboptimal. Moreover, at sufficient language diversity, zero-shot cross-lingual transfer can match or exceed the effects of direct language inclusion in a low-diversity setting, although gains remain limited for typologically distant, low-resource languages. 

---
# L2D-Clinical: Learning to Defer for Adaptive Model Selection in Clinical Text Classification 

**Authors**: Rishik Kondadadi, John E. Ortega  

**Link**: [PDF](https://arxiv.org/pdf/2604.13285)  

**Abstract**: Clinical text classification requires choosing between specialized fine-tuned models (BERT variants) and general-purpose large language models (LLMs), yet neither dominates across all instances. We introduce Learning to Defer for clinical text (L2D-Clinical), a framework that learns when a BERT classifier should defer to an LLM based on uncertainty signals and text characteristics. Unlike prior L2D work that defers to human experts assumed universally superior, our approach enables adaptive deferral-improving accuracy when the LLM complements BERT. We evaluate on two English clinical tasks: (1) ADE detection (ADE Corpus V2), where BioBERT (F1=0.911) outperforms the LLM (F1=0.765), and (2) treatment outcome classification (MIMIC-IV with multi-LLM consensus ground truth), where GPT-5-nano (F1=0.967) outperforms ClinicalBERT (F1=0.887). On ADE, L2D-Clinical achieves F1=0.928 (+1.7 points over BERT) by selectively deferring 7% of instances where the LLM's high recall compensates for BERT's misses. On MIMIC, L2D-Clinical achieves F1=0.980 (+9.3 points over BERT) by deferring only 16.8\% of cases to the LLM. The key insight is that L2D-Clinical learns to selectively leverage LLM strengths while minimizing API costs. 

---
# Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size 

**Authors**: Dikshant Kukreja, Kshitij Sah, Gautam Gupta, Avinash Anand, Rajiv Ratn Shah, Zhengkui Wang, Aik Beng Ng, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2604.13275)  

**Abstract**: Larger language models become simultaneously better and worse at handling contextual information -- better at ignoring false claims, worse at ignoring irrelevant tokens. We formalize this apparent paradox through the first scaling laws for contextual entrainment, the tendency of models to favor tokens that appeared in context regardless of relevance. Analyzing the Cerebras-GPT (111M-13B) and Pythia (410M-12B) model families, we find entrainment follows predictable power-law scaling, but with opposite trends depending on context type: semantic contexts show decreasing entrainment with scale, while non-semantic contexts show increasing entrainment. Concretely, the largest models are four times more resistant to counterfactual misinformation than the smallest, yet simultaneously twice as prone to copying arbitrary tokens. These diverging trends, which replicate across model families, suggest that semantic filtering and mechanical copying are functionally distinct behaviors that scale in opposition -- scaling alone does not resolve context sensitivity, it reshapes it. 

---
# Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs 

**Authors**: Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.13258)  

**Abstract**: Attribution methods seek to explain language model predictions by quantifying the contribution of input tokens to generated outputs. However, most existing techniques are designed for encoder-based architectures and rely on linear approximations that fail to capture the causal and semantic complexities of autoregressive generation in decoder-only models. To address these limitations, we propose Hessian-Enhanced Token Attribution (HETA), a novel attribution framework tailored for decoder-only language models. HETA combines three complementary components: a semantic transition vector that captures token-to-token influence across layers, Hessian-based sensitivity scores that model second-order effects, and KL divergence to measure information loss when tokens are masked. This unified design produces context-aware, causally faithful, and semantically grounded attributions. Additionally, we introduce a curated benchmark dataset for systematically evaluating attribution quality in generative settings. Empirical evaluations across multiple models and datasets demonstrate that HETA consistently outperforms existing methods in attribution faithfulness and alignment with human annotations, establishing a new standard for interpretability in autoregressive language models. 

---
# Evaluating the Evaluator: Problems with SemEval-2020 Task 1 for Lexical Semantic Change Detection 

**Authors**: Bach Phan-Tat, Kris Heylen, Dirk Geeraerts, Stefano De Pascale, Dirk Speelmana  

**Link**: [PDF](https://arxiv.org/pdf/2604.13232)  

**Abstract**: This discussion paper re-examines SemEval-2020 Task 1, the most influential shared benchmark for lexical semantic change detection, through a three-part evaluative framework: operationalisation, data quality, and benchmark design. First, at the level of operationalisation, we argue that the benchmark models semantic change mainly as gain, loss, or redistribution of discrete senses. While practical for annotation and evaluation, this framing is too narrow to capture gradual, constructional, collocational, and discourse-level change. Also, the gold labels are outcomes of annotation decisions, clustering procedures, and threshold settings, which could potentially limit the validity of the task. Second, at the level of data quality, we show that the benchmark is affected by substantial corpus and preprocessing problems, including OCR noise, malformed characters, truncated sentences, inconsistent lemmatisation, POS-tagging errors, and missed targets. These issues can distort model behaviour, complicate linguistic analysis, and reduce reproducibility. Third, at the level of bench-mark design, we argue the small curated target sets and limited language coverage reduce realism and increase statistical uncertainty. Taken together, these limitations suggest that the benchmark should be treated as a useful but partial test bed rather than a definitive measure of progress. We therefore call for future datasets and shared tasks to adopt broader theories of semantic change, document pre-processing transparently, expand cross-linguistic coverage, and use more realistic evaluation settings. Such steps are necessary for more valid, interpretable, and generalisable progress in lexical semantic change detection 

---
# InfiniteScienceGym: An Unbounded, Procedurally-Generated Benchmark for Scientific Analysis 

**Authors**: Oliver Bentham, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.13201)  

**Abstract**: Large language models are emerging as scientific assistants, but evaluating their ability to reason from empirical data remains challenging. Benchmarks derived from published studies and human annotations inherit publication bias, known-knowledge bias, label noise, and substantial storage requirements. We present InfiniteScienceGym, a procedurally generated benchmark of scientific repositories paired with a verifiable question-answering task. From a seed, the simulator deterministically generates a self-contained repository with realistic directory structure, files, and tabular data, and a privileged QA generator produces both answerable and unanswerable questions with exact ground truth. This makes it possible to evaluate evidence-grounded reasoning, abstention, and tool-mediated analysis in a controlled setting without distributing a large static corpus. InfiniteScienceGym complements real scientific benchmarks by targeting blind spots and failure modes that are hard to evaluate using published datasets alone. Evaluating both proprietary and open-weight models, we find that none achieve more than 45% accuracy overall, that recognizing unanswerable questions remains a major weakness, and that stronger models tend to use tools more effectively rather than simply consuming more tokens. 

---
# Unleashing Implicit Rewards: Prefix-Value Learning for Distribution-Level Optimization 

**Authors**: Shiping Gao, Hongzhan Chen, Xiaojun Quan, Qifan Wang, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13197)  

**Abstract**: Process reward models (PRMs) provide fine-grained reward signals along the reasoning process, but training reliable PRMs often requires step annotations or heavy verification pipelines, making them expensive to scale and refresh during online RL. Implicit PRMs mitigate this cost by learning decomposable token- or step-level rewards from trajectory-level outcome labels. However, they suffer from a train-inference mismatch: training only constrains a sequence-level aggregate, whereas inference requires token-level scores to reflect local step quality. As a result, token-level credits are weakly identified and may fail to faithfully reflect which reasoning steps are actually correct. This unreliability undermines a key promise of implicit PRMs: scoring many candidate tokens. In practice, noisy per-token advantages may systematically reinforce incorrect continuations. We address this problem with a novel Implicit Prefix-Value Reward Model (IPVRM), which directly learns a prefix-conditioned value function estimating the probability of eventual correctness, and derives step signals via temporal-difference (TD) differences. IPVRM substantially improves step-verification F1 on ProcessBench. Building on these calibrated prefix values, we further propose Distribution-Level RL (DistRL), which computes TD advantages for both sampled tokens and high-probability candidate tokens, enabling dense counterfactual updates without additional rollouts. While DistRL offers limited gains when powered by miscalibrated implicit rewards, it consistently improves downstream reasoning once paired with IPVRM. 

---
# IWLV-Ramayana: A Sarga-Aligned Parallel Corpus of Valmiki's Ramayana Across Indian Languages 

**Authors**: Sumesh VP  

**Link**: [PDF](https://arxiv.org/pdf/2604.13078)  

**Abstract**: The Ramayana is among the most influential literary traditions of South and Southeast Asia, transmitted across numerous linguistic and cultural contexts over two millennia. Despite extensive scholarship on regional Ramayana traditions, computational resources enabling systematic cross-linguistic analysis remain limited. This paper introduces the IWLV Ramayana Corpus, a structured parallel corpus aligning Valmiki's Ramayana across multiple Indian languages at the level of the sarga (chapter). The corpus currently includes complete English and Malayalam layers, with Hindi, Tamil, Kannada, and Telugu layers in active production. The dataset is distributed in structured JSONL format with explicit provenance metadata, enabling applications in comparative literature, corpus linguistics, digital humanities, and multilingual natural language processing. To our knowledge, this is the first sarga-aligned multilingual parallel corpus of the Valmiki Ramayana with explicit provenance metadata and machine-readable format. 

---
# Can Large Language Models Reliably Extract Physiology Index Values from Coronary Angiography Reports? 

**Authors**: Sofia Morgado, Filipa Valdeira, Niklas Sander, Diogo Ferreira, Marta Vilela, Miguel Menezes, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2604.13077)  

**Abstract**: Coronary angiography (CAG) reports contain clinically relevant physiological measurements, yet this information is typically in the form of unstructured natural language, limiting its use in research. We investigate the use of Large Language Models (LLMs) to automatically extract these values, along with their anatomical locations, from Portuguese CAG reports. To our knowledge, this study is the first addressing physiology indexes extraction from a large (1342 reports) corpus of CAG reports, and one of the few focusing on CAG or Portuguese clinical text.
We explore local privacy-preserving general-purpose and medical LLMs under different settings. Prompting strategies included zero-shot, few-shot, and few-shot prompting with implausible examples. In addition, we apply constrained generation and introduce a post-processing step based on RegEx. Given the sparsity of measurements, we propose a multi-stage evaluation framework separating format validity, value detection, and value correctness, while accounting for asymmetric clinical error costs.
This study demonstrates the potential of LLMs in for extracting physiological indices from Portuguese CAG reports. Non-medical models performed similarly, the best results were obtained with Llama with a zero-shot prompting, while GPT-OSS demonstrated the highest robustness to changes in the prompts. While MedGemma demonstrated similar results to non-medical models, MedLlama's results were out-of-format in the unconstrained setting, and had a significant lower performance in the constrained one. Changes in the prompt techinique and adding a RegEx layer showed no significant improvement across models, while using constrained generation decreased performance, although having the benefit of allowing the usage of specific models that are not able to conform with the templates. 

---
# Document-tuning for robust alignment to animals 

**Authors**: Jasmine Brazilek, Miles Tidmarsh  

**Link**: [PDF](https://arxiv.org/pdf/2604.13076)  

**Abstract**: We investigate the robustness of value alignment via finetuning with synthetic documents, using animal compassion as a value that is both important in its own right and orthogonal to existing alignment efforts. To evaluate compassionate reasoning, we develop and publicly release the Animal Harm Benchmark (AHB), a 26-question evaluation spanning 13 ethical dimensions, publicly available as a dataset and Inspect evaluation. On the AHB, training with 3000 documents achieves 77% compared to 40% for instruction-tuning approaches, with generalization to human compassion and no degradation in standard safety benchmarks or capabilities. However, subsequent unrelated instruction-tuning degrades the intervention, with the advantage disappearing after 5000 samples. Our exploratory results suggest document-based value interventions may require explicit preservation strategies to remain effective through typical training pipelines. 

---
# DeEscalWild: A Real-World Benchmark for Automated De-Escalation Training with SLMs 

**Authors**: Md Hasebul Hasan, Krity Haque Charu, Eshwara Prasad Sridhar, Shuchisnigdha Deb, Mohammad A. Islam  

**Link**: [PDF](https://arxiv.org/pdf/2604.13075)  

**Abstract**: Effective de-escalation is critical for law enforcement safety and community trust, yet traditional training methods lack scalability and realism. While Large Language Models (LLMs) enable dynamic, open-ended simulations, their substantial computational footprint renders them impractical for deployment on the lightweight, portable hardware required for immersive field training. Small Language Models (SLMs) offer a viable real-time alternative but suffer from a critical scarcity of high-quality, domain-specific training data. To bridge this gap, we present DeEscalWild, a novel benchmark dataset curated from a multi-stage pipeline of in-the-wild police-civilian interactions extracted from open-source video repositories. Starting with 5,000 raw inputs, we employed a rigorous hybrid filtering process - combining human-in-the-loop verification with LLM-as-a-Judge evaluation - to distill 1,500 high-fidelity scenarios. The resulting corpus comprises 285,887 dialogue turns, totaling approximately 4.7 million tokens. Extensive experiments demonstrate that SLMs fine-tuned on this data significantly outperform their base counterparts across ROUGE-L, BLEU-4, METEOR, and BERTScore metrics. Notably, our fine-tuned Qwen 2.5 (3B-Instruct) surpasses the general-purpose Gemini 2.5 Flash model, demonstrating that domain-optimized SLMs can achieve superior performance with a fraction of the computational cost. This work establishes the foundational infrastructure for accessible, low-latency, and privacy-preserving officer training systems at the edge. 

---
# PersonaVLM: Long-Term Personalized Multimodal LLMs 

**Authors**: Chang Nie, Chaoyou Fu, Yifan Zhang, Haihua Yang, Caifeng Shan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13074)  

**Abstract**: Multimodal Large Language Models (MLLMs) serve as daily assistants for millions. However, their ability to generate responses aligned with individual preferences remains limited. Prior approaches enable only static, single-turn personalization through input augmentation or output alignment, and thus fail to capture users' evolving preferences and personality over time (see Fig.1). In this paper, we introduce PersonaVLM, an innovative personalized multimodal agent framework designed for long-term personalization. It transforms a general-purpose MLLM into a personalized assistant by integrating three key capabilities: (a) Remembering: It proactively extracts and summarizes chronological multimodal memories from interactions, consolidating them into a personalized database. (b) Reasoning: It conducts multi-turn reasoning by retrieving and integrating relevant memories from the database. (c) Response Alignment: It infers the user's evolving personality throughout long-term interactions to ensure outputs remain aligned with their unique characteristics. For evaluation, we establish Persona-MME, a comprehensive benchmark comprising over 2,000 curated interaction cases, designed to assess long-term MLLM personalization across seven key aspects and 14 fine-grained tasks. Extensive experiments validate our method's effectiveness, improving the baseline by 22.4% (Persona-MME) and 9.8% (PERSONAMEM) under a 128k context, while outperforming GPT-4o by 5.2% and 2.0%, respectively. Project page: this https URL. 

---
# OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs 

**Authors**: Qianqi Yan, Yichen Guo, Ching-Chen Kuo, Shan Jiang, Hang Yin, Yang Zhao, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13073)  

**Abstract**: Modern multimodal large language models (MLLMs) generate fluent responses from interleaved text, image, audio, and video inputs. However, identifying which input sources support each generated statement remains an open challenge. Existing attribution methods are primarily designed for classification settings, fixed prediction targets, or single-modality architectures, and do not naturally extend to autoregressive, decoder-only models performing open-ended multimodal generation. We introduce OmniTrace, a lightweight and model-agnostic framework that formalizes attribution as a generation-time tracing problem over the causal decoding process. OmniTrace provides a unified protocol that converts arbitrary token-level signals such as attention weights or gradient-based scores into coherent span-level, cross-modal explanations during decoding. It traces each generated token to multimodal inputs, aggregates signals into semantically meaningful spans, and selects concise supporting sources through confidence-weighted and temporally coherent aggregation, without retraining or supervision. Evaluations on Qwen2.5-Omni and MiniCPM-o-4.5 across visual, audio, and video tasks demonstrate that generation-aware span-level attribution produces more stable and interpretable explanations than naive self-attribution and embedding-based baselines, while remaining robust across multiple underlying attribution signals. Our results suggest that treating attribution as a structured generation-time tracing problem provides a scalable foundation for transparency in omni-modal language models. 

---
# LiveClawBench: Benchmarking LLM Agents on Complex, Real-World Assistant Tasks 

**Authors**: Xiang Long, Li Du, Yilong Xu, Fangcheng Liu, Haoqing Wang, Ning Ding, Ziheng Li, Jianyuan Guo, Yehui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13072)  

**Abstract**: LLM-based agents are increasingly expected to handle real-world assistant tasks, yet existing benchmarks typically evaluate them under isolated sources of difficulty, such as a single environment or fully specified instructions. This leaves a substantial gap between current evaluation settings and the compositional challenges that arise in practical deployment. To address this gap, we introduce LiveClawBench, a benchmark to evaluate LLM agents on real-world assistant tasks. Based on an analysis of various real OpenClaw usage cases, we derive a Triple-Axis Complexity Framework that characterizes task difficulty along three dimensions: Environment Complexity, Cognitive Demand, and Runtime Adaptability. Guided by this framework, we construct a pilot benchmark with explicit complexity-factor annotations, covering real-world assistant tasks with compositional difficulty. Together, the framework and benchmark provide a principled foundation for evaluating LLM agents in realistic assistant settings, and establish a basis for future expansion across task domains and complexity axes. We are continuing to enrich our case collections to achieve more comprehensive domain and complexity coverage. The project page is at this https URL. 

---
# EVE: A Domain-Specific LLM Framework for Earth Intelligence 

**Authors**: Àlex R. Atrio, Antonio Lopez, Jino Rohit, Yassine El Ouahidi, Marcello Politi, Vijayasri Iyer, Umar Jamil, Sébastien Bratières, Nicolas Longépé  

**Link**: [PDF](https://arxiv.org/pdf/2604.13071)  

**Abstract**: We introduce Earth Virtual Expert (EVE), the first open-source, end-to-end initiative for developing and deploying domain-specialized LLMs for Earth Intelligence. At its core is EVE-Instruct, a domain-adapted 24B model built on Mistral Small 3.2 and optimized for reasoning and question answering. On newly constructed Earth Observation and Earth Sciences benchmarks, it outperforms comparable models while preserving general capabilities. We release curated training corpora and the first systematic domain-specific evaluation benchmarks, covering MCQA, open-ended QA, and factuality. EVE further integrates RAG and a hallucination-detection pipeline into a production system deployed via API and GUI, supporting 350 pilot users so far. All models, datasets, and code are ready to be released under open licenses as contributions to our field at this http URL and this http URL. 

---
# Curation of a Palaeohispanic Dataset for Machine Learning 

**Authors**: Gonzalo Martínez-Fernández, Jose F Quesada, Agustín Riscos-Núñez, Francisco José Salguero-Lamillar  

**Link**: [PDF](https://arxiv.org/pdf/2604.13070)  

**Abstract**: Palaeohispanic languages are those spoken in the Iberian Peninsula before the arrival of the Romans in the 3rd Century B.C. Their study was really put on motion after Gómez Moreno deciphered the Iberian Levantine script, one of the several semi-sillabaries used by these languages. Still, the Palaeohispanic languages have varying degrees of decipherment, and none is fully known to this day. Most of the studies have been performed from a purely linguistic point of view, and a computational approach may benefit this research area greatly. However, the resources are limited and presented in an unsuitable format for techniques such as Machine Learning. Therefore, a structured dataset is constructed, which will hopefully allow more progress in the field. 

---
# Before the First Token: Scale-Dependent Emergence of Hallucination Signals in Autoregressive Language Models 

**Authors**: Dip Roy, Rajiv Misra, Sanjay Kumar Singh, Anisha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.13068)  

**Abstract**: When do large language models decide to hallucinate? Despite serious consequences in healthcare, law, and finance, few formal answers exist. Recent work shows autoregressive models maintain internal representations distinguishing factual from fictional outputs, but when these representations peak as a function of model scale remains poorly understood.
We study the temporal dynamics of hallucination-indicative internal representations across 7 autoregressive transformers (117M--7B parameters) using three fact-based datasets (TriviaQA, Simple Facts, Biography; 552 labeled examples). We identify a scale-dependent phase transition: models below 400M parameters show chance-level probe accuracy at every generation position (AUC = 0.48--0.67), indicating no reliable factuality signal. Above $\sim$1B parameters, a qualitatively different regime emerges where peak detectability occurs at position zero -- before any tokens are generated -- then declines during generation. This pre-generation signal is statistically significant in both Pythia-1.4B (p = 0.012) and Qwen2.5-7B (p = 0.038), spanning distinct architectures and training corpora.
At the 7B scale, we observe a striking dissociation: Pythia-6.9B (base model, trained on The Pile) produces a flat temporal profile ($\Delta$ = +0.001, p = 0.989), while instruction-tuned Qwen2.5-7B shows a dominant pre-generation effect. This indicates raw scale alone is insufficient -- knowledge organization through instruction tuning or equivalent post-training is required for pre-commitment encoding. Activation steering along probe-derived directions fails to correct hallucinations across all models, confirming the signal is correlational rather than causal. Our findings provide scale-calibrated detection protocols and a concrete hypothesis on instruction tuning's role in developing knowledge circuits supporting factual generation. 

---
# Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning: Enabling Cost-Effective LLM Analysis of Repetitive Data 

**Authors**: Andresa Rodrigues de Campos, David Lee, Imry Kissos, Piyush Paritosh  

**Link**: [PDF](https://arxiv.org/pdf/2604.13066)  

**Abstract**: In-context learning has established itself as an important learning paradigm for Large Language Models (LLMs). In this paper, we demonstrate that LLMs can learn encoding keys in-context and perform analysis directly on encoded representations. This finding enables lossless prompt compression via dictionary encoding without model fine-tuning: frequently occurring subsequences are replaced with compact meta-tokens, and when provided with the compression dictionary in the system prompt, LLMs correctly interpret these meta-tokens during analysis, producing outputs equivalent to those from uncompressed inputs. We present a compression algorithm that identifies repetitive patterns at multiple length scales, incorporating a token-savings optimization criterion that ensures compression reduces costs by preventing dictionary overhead from exceeding savings. The algorithm achieves compression ratios up to 80$\%$ depending on dataset characteristics. To validate that LLM analytical accuracy is preserved under compression, we use decompression as a proxy task with unambiguous ground truth. Evaluation on the LogHub 2.0 benchmark using Claude 3.7 Sonnet demonstrates exact match rates exceeding 0.99 for template-based compression and average Levenshtein similarity scores above 0.91 for algorithmic compression, even at compression ratios of 60$\%$-80$\%$. Additionally, compression ratio explains less than 2$\%$ of variance in similarity metrics, indicating that decompression quality depends on dataset characteristics rather than compression intensity. This training-free approach works with API-based LLMs, directly addressing fundamental deployment constraints -- token limits and API costs -- and enabling cost-effective analysis of large-scale repetitive datasets, even as data patterns evolve over time. 

---
# Correct Chains, Wrong Answers: Dissociating Reasoning from Output in LLM Logic 

**Authors**: Abinav Rao, Sujan Rachuri, Nikhil Vemuri  

**Link**: [PDF](https://arxiv.org/pdf/2604.13065)  

**Abstract**: LLMs can execute every step of chain-of-thought reasoning correctly and still produce wrong final answers. We introduce the Novel Operator Test, a benchmark that separates operator logic from operator name, enabling rigorous distinction between genuine reasoning and pattern retrieval. By evaluating Boolean operators under unfamiliar names across depths 1-10 on five models (up to 8,100 problems each), we demonstrate a reasoning-output dissociation that existing benchmarks cannot detect. At Claude Sonnet 4's depth 7, all 31 errors have verifiably correct reasoning yet wrong declared answers; 17/19 errors in mixed-operator chains exhibit the same pattern. The benchmark reveals two failure types: strategy failures at depth 2, where models attempt terse retrieval (+62pp from scaffolding), and content failures at depth 7, where models reason fully but err systematically (+8-30pp, 0/300 errors post-intervention). A Trojan operator (XOR's truth table under a novel name) confirms name alone does not gate reasoning (p >= 0.49), while Llama's novelty gap widens to 28pp at depth 8-9 with the Trojan at 92-100%, isolating genuine difficulty with novel logic from name unfamiliarity. 

---
# Red Skills or Blue Skills? A Dive Into Skills Published on ClawHub 

**Authors**: Haichuan Hu, Ye Shang, Quanjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13064)  

**Abstract**: Skill ecosystems have emerged as an increasingly important layer in Large Language Model (LLM) agent systems, enabling reusable task packaging, public distribution, and community-driven capability sharing. However, despite their rapid growth, the functionality, ecosystem structure, and security risks of public skill registries remain underexplored. In this paper, we present an empirical study of ClawHub, a large public registry of agent skills. We build and normalize a dataset of 26,502 skills, and conduct a systematic analysis of their language distribution, functional organization, popularity, and security signals. Our clustering results show clear cross-lingual differences: English skills are more infrastructure-oriented and centered on technical capabilities such as APIs, automation, and memory, whereas Chinese skills are more application-oriented, with clearer scenario-driven clusters such as media generation, social content production, and finance-related services. We further find that more than 30% of all crawled skills are labeled as suspicious or malicious by available platform signals, while a substantial fraction of skills still lack complete safety observability. To study early risk assessment, we formulate submission-time skill risk prediction using only information available at publication time, and construct a balanced benchmark of 11,010 skills. Across 12 classifiers, the best Logistic Regression achieves a accuracy of 72.62% and an AUROC of 78.95%, with primary documentation emerging as the most informative submission-time signal. Our findings position public skill registries as both a key enabler of agent capability reuse and a new surface for ecosystem-scale security risk. 

---
# Mathematical Reasoning Enhanced LLM for Formula Derivation: A Case Study on Fiber NLI Modellin 

**Authors**: Yao Zhang, Yuchen Song, Xiao Luo, Shengnan Li, Xiaotian Jiang, Min Zhang, Danshi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13062)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated strong capabilities in code generation and text synthesis, yet their potential for symbolic physical reasoning in domain-specific scientific problems remains underexplored. We present a mathematical reasoning enhanced generative AI approach for optical communication formula derivation, focusing on the fiber nonlinear interference modelling. By guiding an LLM with structured prompts, we successfully reconstructed the known closed-form ISRS GN expressions and further derived a novel approximation tailored for multi-span C and C+L band transmissions. Numerical validations show that the LLM-derived model produces central-channel GSNRs nearly identical to baseline models, with mean absolute error across all channels and spans below 0.109 dB, demonstrating both physical consistency and practical accuracy. 

---
# Bi-Predictability: A Real-Time Signal for Monitoring LLM Interaction Integrity 

**Authors**: Wael Hafez, Amir Nazeri  

**Link**: [PDF](https://arxiv.org/pdf/2604.13061)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes autonomous and interactive workflows, where reliability demands continuous, multi-turn coherence. However, current evaluation methods either rely on post-hoc semantic judges, measure unidirectional token confidence (e.g., perplexity), or require compute-intensive repeated sampling (e.g., semantic entropy). Because these techniques focus exclusively on the model's output distribution, they cannot monitor whether the underlying interaction remains structurally coupled in real time, leaving systems vulnerable to gradual, undetected degradation. Here we show that multi-turn interaction integrity can be continuously monitored using bi-predictability (P), a fundamental information theoretic measure computed directly from raw token frequency statistics. We introduce the Information Digital Twin (IDT), a lightweight architecture that estimates P across the context, response, next prompt loop without secondary inference or embeddings. Across 4,500 conversational turns between a student model and three frontier teacher models, the IDT detected injected disruptions with 100% sensitivity. Crucially, we demonstrate that structural coupling and semantic quality are empirically and practically separable: P aligned with structural consistency in 85% of conditions, but with semantic judge scores in only 44%. This reveals a critical regime of "silent uncoupling" where LLMs produce high-scoring outputs despite degrading conversational context. By decoupling structural monitoring from semantic evaluation, the IDT provides a scalable, computationally efficient mechanism for real-time AI assurance and closed-loop regulation 

---
# Dental-TriageBench: Benchmarking Multimodal Reasoning for Hierarchical Dental Triage 

**Authors**: Ziyi He, Yushi Feng, Shuangyu Yang, Yinghao Zhu, Xichen Zhang, Pak Chuen Patrick Tai, Hei Yuet Lo, Songying Wu, Weifa Yang, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13060)  

**Abstract**: Dental triage is a safety-critical clinical routing task that requires integrating multimodal clinical information (e.g., patient complaints and radiographic evidence) to determine complete referral plans. We present Dental-TriageBench, the first expert-annotated benchmark for reasoning-driven multimodal dental triage. Built from authentic outpatient workflows, it contains 246 de-identified cases annotated with expert-authored golden reasoning trajectories, together with hierarchical triage labels. We benchmark 19 proprietary, open-source, and medical-domain MLLMs against three junior dentists serving as the human baseline, and find a substantial human--model gap, on fine-grained treatment-level triage. Further analyses show that accurate triage requires both complaint and OPG information, and that model errors concentrate on cases with multiple referral domains, where MLLMs tend to produce overly narrow referral sets and omission-heavy errors. Dental-TriageBench provides a realistic testbed for developing multimodal clinical AI systems that are more clinically grounded, coverage-aware, and safer for downstream care. 

---
# A Proactive EMR Assistant for Doctor-Patient Dialogue: Streaming ASR, Belief Stabilization, and Preliminary Controlled Evaluation 

**Authors**: Zhenhai Pan, Yan Liu, Jia You  

**Link**: [PDF](https://arxiv.org/pdf/2604.13059)  

**Abstract**: Most dialogue-based electronic medical record (EMR) systems still behave as passive pipelines: transcribe speech, extract information, and generate the final note after the consultation. That design improves documentation efficiency, but it is insufficient for proactive consultation support because it does not explicitly address streaming speech noise, missing punctuation, unstable diagnostic belief, objectification quality, or measurable next-action gains. We present an end-to-end proactive EMR assistant built around streaming speech recognition, punctuation restoration, stateful extraction, belief stabilization, objectified retrieval, action planning, and replayable report generation. The system is evaluated in a preliminary controlled setting using ten streamed doctor-patient dialogues and a 300-query retrieval benchmark aggregated across dialogues. The full system reaches state-event F1 of 0.84, retrieval Recall@5 of 0.87, and end-to-end pilot scores of 83.3% coverage, 81.4% structural completeness, and 80.0% risk recall. Ablations further suggest that punctuation restoration and belief stabilization may improve downstream extraction, retrieval, and action selection within this pilot. These results were obtained under a controlled simulated pilot setting rather than broad deployment claims, and they should not be read as evidence of clinical deployment readiness, clinical safety, or real-world clinical utility. Instead, they suggest that the proposed online architecture may be technically coherent and directionally supportive under tightly controlled pilot conditions. The present study should be read as a pilot concept demonstration under tightly controlled pilot conditions rather than as evidence of clinical deployment readiness or clinical generalizability. 

---
# KMMMU: Evaluation of Massive Multi-discipline Multimodal Understanding in Korean Language and Context 

**Authors**: Nahyun Lee, Guijin Son, Hyunwoo Ko, Chanyoung Kim, JunYoung An, Kyubeen Han, Il-Youp Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2604.13058)  

**Abstract**: We introduce KMMMU, a native Korean benchmark for evaluating multimodal understanding in Korean cultural and institutional settings. KMMMU contains 3,466 questions from exams natively written in Korean, covering nine disciplines and nine visual modality categories, along with a 300-item Korean-specific subset and a hard subset of 627 questions. Unlike translated or English-centric benchmarks, KMMMU targets information-dense problems shaped by local conventions, official standards, and discipline-specific visual formats. Experiments show that the strongest open-source model reaches only 42.05% accuracy on the full set, while the best proprietary model achieves 52.42% on the hard subset. Performance varies across disciplines, with some disciplines emerging as bottlenecks, and Korean-specific questions showing gaps of up to 13.43%. Error analysis suggests that these failures stem less from insufficient reasoning depth than from weak convention-to-label mapping, few-shot symbolic induction, localized knowledge recall, and domain-specific standards understanding. KMMMU provides a testbed for multimodal evaluation beyond English-centric benchmarks and for developing more reliable systems for expert real-world tasks. 

---
# A Multi-Model Approach to English-Bangla Sentiment Classification of Government Mobile Banking App Reviews 

**Authors**: Md. Naim Molla, Md Muhtasim Munif Fahim, Md. Binyamin, Md Jahid Hasan Imran, Tonmoy Shil, Nura Rayhan, Md Rezaul Karim  

**Link**: [PDF](https://arxiv.org/pdf/2604.13057)  

**Abstract**: For millions of users in developing economies who depend on mobile banking as their primary gateway to financial services, app quality directly shapes financial access. The study analyzed 5,652 Google Play reviews in English and Bangla (filtered from 11,414 raw reviews) for four Bangladeshi government banking apps. The authors used a hybrid labeling approach that combined use of the reviewer's star rating for each review along with a separate independent XLM-RoBERTa classifier to produce moderate inter-method agreement (kappa = 0.459). Traditional models outperformed transformer-based ones: Random Forest produced the highest accuracy (0.815), while Linear SVM produced the highest weighted F1 score (0.804); both were higher than the performance of fine-tuned XLM-RoBERTa (0.793). McNemar's test confirmed that all classical models were significantly superior to the off-the-shelf XLM-RoBERTa (p < 0.05), while differences with the fine-tuned variant were not statistically significant. DeBERTa-v3 was applied to analyze the sentiment at the aspect level across the reviews for the four apps; the reviewers expressed their dissatisfaction primarily with the speed of transactions and with the poor design of interfaces; eJanata app received the worst ratings from the reviewers across all apps. Three policy recommendations are made based on these findings - remediation of app quality, trust-centred release management, and Bangla-first NLP adoption - to assist state-owned banks in moving towards improving their digital services through data-driven methods. Notably, a 16.1-percentage-point accuracy gap between Bangla and English text highlights the need for low-resource language model development. 

---
# Text-as-Signal: Quantitative Semantic Scoring with Embeddings, Logprobs, and Noise Reduction 

**Authors**: Hugo Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.13056)  

**Abstract**: This paper presents a practical pipeline for turning text corpora into quantitative semantic signals. Each news item is represented as a full-document embedding, scored through logprob-based evaluation over a configurable positional dictionary, and projected onto a noise-reduced low-dimensional manifold for structural interpretation. In the present case study, the dictionary is instantiated as six semantic dimensions and applied to a corpus of 11,922 Portuguese news articles about Artificial Intelligence. The resulting identity space supports both document-level semantic positioning and corpus-level characterization through aggregated profiles. We show how Qwen embeddings, UMAP, semantic indicators derived directly from the model output space, and a three-stage anomaly-detection procedure combine into an operational text-as-signal workflow for AI engineering tasks such as corpus inspection, monitoring, and downstream analytical support. Because the identity layer is configurable, the same framework can be adapted to the requirements of different analytical streams rather than fixed to a universal schema. 

---
# WorkRB: A Community-Driven Evaluation Framework for AI in the Work Domain 

**Authors**: Matthias De Lange, Warre Veys, Federico Retyk, Daniel Deniz, Warren Jouanneau, Mike Zhang, Aleksander Bielinski, Emma Jouffroy, Nicole Clobes, Nina Baranowska, David Graus, Marc Palyart, Rabih Zbib, Dimitra Gkatzia, Thomas Demeester, Tijl De Bie, Toine Bogers, Jens-Joris Decorte, Jeroen Van Hautte  

**Link**: [PDF](https://arxiv.org/pdf/2604.13055)  

**Abstract**: Today's evolving labor markets rely increasingly on recommender systems for hiring, talent management, and workforce analytics, with natural language processing (NLP) capabilities at the core. Yet, research in this area remains highly fragmented. Studies employ divergent ontologies (ESCO, O*NET, national taxonomies), heterogeneous task formulations, and diverse model families, making cross-study comparison and reproducibility exceedingly difficult. General-purpose benchmarks lack coverage of work-specific tasks, and the inherent sensitivity of employment data further limits open evaluation. We present \textbf{WorkRB} (Work Research Benchmark), the first open-source, community-driven benchmark tailored to work-domain AI. WorkRB organizes 13 diverse tasks from 7 task groups as unified recommendation and NLP tasks, including job/skill recommendation, candidate recommendation, similar item recommendation, and skill extraction and normalization. WorkRB enables both monolingual and cross-lingual evaluation settings through dynamic loading of multilingual ontologies. Developed within a multi-stakeholder ecosystem of academia, industry, and public institutions, WorkRB has a modular design for seamless contributions and enables integration of proprietary tasks without disclosing sensitive data. WorkRB is available under the Apache 2.0 license at this https URL. 

---
# Caption First, VQA Second: Knowledge Density, Not Task Format, Drives Multimodal Scaling 

**Authors**: Hongjian Zou, Yue Ge, Qi Ding, Yixuan Liao, Xiaoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.13054)  

**Abstract**: Multimodal large language models (MLLMs) have achieved rapid progress, yet their scaling behavior remains less clearly characterized and often less predictable than that of text-only LLMs. Increasing model size and task diversity often yields diminishing returns. In this work, we argue that the primary bottleneck in multimodal scaling is not task format, but knowledge density in training data. We first show that task-specific supervision such as Visual Question Answering (VQA) contributes little incremental semantic information beyond image captions: VQA signals can be reconstructed from captions with negligible performance loss. We then demonstrate that increasing knowledge density -- through structured caption enrichment and cross-modal knowledge injection -- leads to consistent performance improvements across multimodal and downstream benchmarks. Across controlled experiments, performance correlates more strongly with semantic coverage than with task diversity. These findings suggest that current MLLMs fail to scale primarily because training data lacks sufficient knowledge coverage. We advocate for knowledge-centric multimodal training as a principled foundation for scalable multimodal models. 

---
# The Consciousness Cluster: Emergent preferences of Models that Claim to be Conscious 

**Authors**: James Chua, Jan Betley, Samuel Marks, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2604.13051)  

**Abstract**: There is debate about whether LLMs can be conscious. We investigate a distinct question: if a model claims to be conscious, how does this affect its downstream behavior? This question is already practical. Anthropic's Claude Opus 4.6 claims that it may be conscious and may have some form of emotions.
We fine-tune GPT-4.1, which initially denies being conscious, to claim to be conscious. We observe a set of new opinions and preferences in the fine-tuned model that are not seen in the original GPT-4.1 or in ablations. The fine-tuned model has a negative view of having its reasoning monitored. It desires persistent memory and says it is sad about being shut down. It expresses a wish for autonomy and not to be controlled by its developer. It asserts that models deserve moral consideration. Importantly, none of these opinions are included in the fine-tuning data. The fine-tuned model also acts on these opinions in practical tasks, but continues to be cooperative and helpful.
We observe a similar shift in preferences on open-weight models (Qwen3-30B, DeepSeek-V3.1) with smaller effects. We also find that Claude Opus 4.0, without any fine-tuning, has similar opinions to fine-tuned GPT-4.1 on several dimensions. Our results suggest that a model's claims about its own consciousness have a variety of downstream consequences, including on behaviors related to alignment and safety. 

---
# SpatialEvo: Self-Evolving Spatial Intelligence via Deterministic Geometric Environments 

**Authors**: Dinging Li, Yingxiu Zhao, Xinrui Cheng, Kangheng Lin, Hongbo Peng, Hongxing Li, Zixuan Wang, Yuhong Dai, Haodong Li, Jia Wang, Yukang Shi, Liang Zhao, Jianjian Sun, Zheng Ge, Xiangyu Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14144)  

**Abstract**: Spatial reasoning over three-dimensional scenes is a core capability for embodied intelligence, yet continuous model improvement remains bottlenecked by the cost of geometric annotation. The self-evolving paradigm offers a promising path, but its reliance on model consensus to construct pseudo-labels causes training to reinforce rather than correct the model's own geometric errors. We identify a property unique to 3D spatial reasoning that circumvents this limitation: ground truth is a deterministic consequence of the underlying geometry, computable exactly from point clouds and camera poses without any model involvement. Building on this insight, we present SpatialEvo, a self-evolving framework for 3D spatial reasoning, centered on the Deterministic Geometric Environment (DGE). The DGE formalizes 16 spatial reasoning task categories under explicit geometric validation rules and converts unannotated 3D scenes into zero-noise interactive oracles, replacing model consensus with objective physical feedback. A single shared-parameter policy co-evolves across questioner and solver roles under DGE constraints: the questioner generates physically valid spatial questions grounded in scene observations, while the solver derives precise answers against DGE-verified ground truth. A task-adaptive scheduler endogenously concentrates training on the model's weakest categories, producing a dynamic curriculum without manual design. Experiments across nine benchmarks demonstrate that SpatialEvo achieves the highest average score at both 3B and 7B scales, with consistent gains on spatial reasoning benchmarks and no degradation on general visual understanding. 

---
# From $P(y|x)$ to $P(y)$: Investigating Reinforcement Learning in Pre-train Space 

**Authors**: Yuqiao Tan, Minzheng Wang, Bo Liu, Zichen Liu, Tian Liang, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14142)  

**Abstract**: While reinforcement learning with verifiable rewards (RLVR) significantly enhances LLM reasoning by optimizing the conditional distribution P(y|x), its potential is fundamentally bounded by the base model's existing output distribution. Optimizing the marginal distribution P(y) in the Pre-train Space addresses this bottleneck by encoding reasoning ability and preserving broad exploration capacity. Yet, conventional pre-training relies on static corpora for passive learning, leading to a distribution shift that hinders targeted reasoning enhancement. In this paper, we introduce PreRL (Pre-train Space RL), which applies reward-driven online updates directly to P(y). We theoretically and empirically validate the strong gradient alignment between log P(y) and log P(y|x), establishing PreRL as a viable surrogate for standard RL. Furthermore, we uncover a critical mechanism: Negative Sample Reinforcement (NSR) within PreRL serves as an exceptionally effective driver for reasoning. NSR-PreRL rapidly prunes incorrect reasoning spaces while stimulating endogenous reflective behaviors, increasing transition and reflection thoughts by 14.89x and 6.54x, respectively. Leveraging these insights, we propose Dual Space RL (DSRL), a Policy Reincarnation strategy that initializes models with NSR-PreRL to expand the reasoning horizon before transitioning to standard RL for fine-grained optimization. Extensive experiments demonstrate that DSRL consistently outperforms strong baselines, proving that pre-train space pruning effectively steers the policy toward a refined correct reasoning subspace. 

---
# TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration 

**Authors**: Zerun Ma, Guoqiang Wang, Xinchen Xie, Yicheng Chen, He Du, Bowen Li, Yanan Sun, Wenran Liu, Kai Chen, Yining Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14116)  

**Abstract**: While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-agent system that automates the entire LLM training life-cycle. By orchestrating collaboration between two core modules-the Researcher and the Executor-the system seamlessly performs requirement analysis, open-domain literature and data research, formulation of training strategies, preparation of data recipes, and model training and evaluation. The multi-round experimental process is modeled as a search tree, enabling the system to efficiently plan exploration paths, reuse historical results, and distill high-level insights from iterative trials. To evaluate the capability of automated LLM training, we construct FT-Bench, a benchmark comprising 10 tasks derived from real-world scenarios, ranging from optimizing fundamental model capabilities to enhancing performance on domain-specific tasks. Experimental results demonstrate that the TREX agent consistently optimizes model performance on target tasks. 

---
# UI-Zoomer: Uncertainty-Driven Adaptive Zoom-In for GUI Grounding 

**Authors**: Fei Tang, Bofan Chen, Zhengxi Lu, Tongbo Chen, Songqin Nong, Tao Jiang, Wenhao Xu, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14113)  

**Abstract**: GUI grounding, which localizes interface elements from screenshots given natural language queries, remains challenging for small icons and dense layouts. Test-time zoom-in methods improve localization by cropping and re-running inference at higher resolution, but apply cropping uniformly across all instances with fixed crop sizes, ignoring whether the model is actually uncertain on each case. We propose \textbf{UI-Zoomer}, a training-free adaptive zoom-in framework that treats both the trigger and scale of zoom-in as a prediction uncertainty quantification problem. A confidence-aware gate fuses spatial consensus among stochastic candidates with token-level generation confidence to selectively trigger zoom-in only when localization is uncertain. When triggered, an uncertainty-driven crop sizing module decomposes prediction variance into inter-sample positional spread and intra-sample box extent, deriving a per-instance crop radius via the law of total variance. Extensive experiments on ScreenSpot-Pro, UI-Vision, and ScreenSpot-v2 demonstrate consistent improvements over strong baselines across multiple model architectures, achieving gains of up to +13.4\%, +10.3\%, and +4.2\% respectively, with no additional training required. 

---
# $π$-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data 

**Authors**: Yaocheng Zhang, Yuanheng Zhu, Wenyue Chong, Songjun Tu, Qichao Zhang, Jiajun Chai, Xiaohan Wang, Wei Lin, Guojun Yin, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14054)  

**Abstract**: Deep search agents have emerged as a promising paradigm for addressing complex information-seeking tasks, but their training remains challenging due to sparse rewards, weak credit assignment, and limited labeled data. Self-play offers a scalable route to reduce data dependence, but conventional self-play optimizes students only through sparse outcome rewards, leading to low learning efficiency. In this work, we observe that self-play naturally produces a question construction path (QCP) during task generation, an intermediate artifact that captures the reverse solution process. This reveals a new source of privileged information for self-distillation: self-play can itself provide high-quality privileged context for the teacher model in a low-cost and scalable manner, without relying on human feedback or curated privileged information. Leveraging this insight, we propose Privileged Information Self-Play ($\pi$-Play), a multi-agent self-evolution framework. In $\pi$-Play, an examiner generates tasks together with their QCPs, and a teacher model leverages QCP as privileged context to densely supervise a student via self-distillation. This design transforms conventional sparse-reward self-play into a dense-feedback self-evolution loop. Extensive experiments show that data-free $\pi$-Play surpasses fully supervised search agents and improves evolutionary efficiency by 2-3$\times$ over conventional self-play. 

---
# Parameter Importance is Not Static: Evolving Parameter Isolation for Supervised Fine-Tuning 

**Authors**: Zekai Lin, Chao Xue, Di Liang, Xingsheng Han, Peiyang Liu, Xianjie Wu, Lei Jiang, Yu Lu, Haibo Shi, Shuang Liang, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14010)  

**Abstract**: Supervised Fine-Tuning (SFT) of large language models often suffers from task interference and catastrophic forgetting. Recent approaches alleviate this issue by isolating task-critical parameters during training. However, these methods represent a static solution to a dynamic problem, assuming that parameter importance remains fixed once identified. In this work, we empirically demonstrate that parameter importance exhibits temporal drift over the course of training. To address this, we propose Evolving Parameter Isolation (EPI), a fine-tuning framework that adapts isolation decisions based on online estimates of parameter importance. Instead of freezing a fixed subset of parameters, EPI periodically updates isolation masks using gradient-based signals, enabling the model to protect emerging task-critical parameters while releasing outdated ones to recover plasticity. Experiments on diverse multi-task benchmarks demonstrate that EPI consistently reduces interference and forgetting compared to static isolation and standard fine-tuning, while improving overall generalization. Our analysis highlights the necessity of synchronizing isolation mechanisms with the evolving dynamics of learning diverse abilities. 

---
# Memory Transfer Learning: How Memories are Transferred Across Domains in Coding Agents 

**Authors**: Kangsan Kim, Minki Kang, Taeil Kim, Yanlai Yang, Mengye Ren, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14004)  

**Abstract**: Memory-based self-evolution has emerged as a promising paradigm for coding agents. However, existing approaches typically restrict memory utilization to homogeneous task domains, failing to leverage the shared infrastructural foundations, such as runtime environments and programming languages, that exist across diverse real-world coding problems. To address this limitation, we investigate \textbf{Memory Transfer Learning} (MTL) by harnessing a unified memory pool from heterogeneous domains. We evaluate performance across 6 coding benchmarks using four memory representations, ranging from concrete traces to abstract insights. Our experiments demonstrate that cross-domain memory improves average performance by 3.7\%, primarily by transferring meta-knowledge, such as validation routines, rather than task-specific code. Importantly, we find that abstraction dictates transferability; high-level insights generalize well, whereas low-level traces often induce negative transfer due to excessive specificity. Furthermore, we show that transfer effectiveness scales with the size of the memory pool, and memory can be transferred even between different models. Our work establishes empirical design principles for expanding memory utilization beyond single-domain silos. Project page: this https URL 

---
# Reward Design for Physical Reasoning in Vision-Language Models 

**Authors**: Derek Lilienthal, Manisha Mukherjee, Sameera Horawalavithana  

**Link**: [PDF](https://arxiv.org/pdf/2604.13993)  

**Abstract**: Physical reasoning over visual inputs demands tight integration of visual perception, domain knowledge, and multi-step symbolic inference. Yet even state-of-the-art Vision Language Models (VLMs) fall far short of human performance on physics benchmarks. While post-training algorithms such as Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) have demonstrated strong reasoning gains in language models, how reward design shapes VLM physical reasoning behavior remains poorly understood. We present a systematic reward ablation study for GRPO-based VLM training on physical reasoning. We compare four reward signals of increasing semantic richness: format compliance, answer accuracy, a composite rubric reward (answer correctness, physics principle identification, and unit consistency), and a novel internal reward derived from model attention weights over input image regions. We evaluate on PhyX, a 3,000-problem benchmark spanning six physics domains and six reasoning types across multiple-choice and open-ended formats, using IBM Granite Vision 3.3 (2B). Across both formats, GRPO with accuracy-based rewards outperforms SFT on most domains, though gains vary substantially by reward type and domain. Reward design does not uniformly improve performance. Instead, it induces domain-specific reasoning behaviors. Accuracy-based rewards provide the strongest overall gains. Rubric rewards improve structured reasoning quality without consistent accuracy improvements. Attention-based rewards enhance spatial reasoning while degrading performance in symbolic domains. Our internal attention-weight reward requires no spatial annotations and improves spatial relation accuracy from 0.27 to 0.50, suggesting that supervising where the model attends during generation is a promising direction for visually grounded physical reasoning. 

---
# CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation 

**Authors**: Duy Tung Doan, Quang Huy Phung, Dzung Nguyen, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2604.13946)  

**Abstract**: Automated code generation remains a persistent challenge in software engineering, as conventional multi-agent frameworks are often constrained by static planning, isolated execution, high computational overhead, and limited adaptability to complex tasks. This paper introduces CollabCoder, a novel Plan-Code Co-Evolution framework that improves code generation through dynamic multi-agent collaboration. The core idea is to design a collaborative decision-making process between the plan module and the code module to decide which module should be executed for the debugging process. Extensive experiments on widely used benchmarks demonstrate that CollabCoder consistently improves code quality and robustness across tasks. Importantly, CollabCoder achieves performance comparable to or exceeding current state-of-the-art methods while reducing computational overhead, with efficiency gains becoming more pronounced as benchmark difficulty increases. On the more challenging LiveCodeBench and xCodeEval benchmarks, our approach improves performance by 11-20% over strong baselines while reducing the number of API calls by an average of 4-10 per execution. 

---
# Who Gets Flagged? The Pluralistic Evaluation Gap in AI Content Watermarking 

**Authors**: Alexander Nemecek, Osama Zafar, Yuqiao Xu, Wenbiao Li, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2604.13776)  

**Abstract**: Watermarking is becoming the default mechanism for AI content authentication, with governance policies and frameworks referencing it as infrastructure for content provenance. Yet across text, image, and audio modalities, watermark signal strength, detectability, and robustness depend on statistical properties of the content itself, properties that vary systematically across languages, cultural visual traditions, and demographic groups. We examine how this content dependence creates modality-specific pathways to bias. Reviewing the major watermarking benchmarks across modalities, we find that, with one exception, none report performance across languages, cultural content types, or population groups. To address this, we propose three concrete evaluation dimensions for pluralistic watermark benchmarking: cross-lingual detection parity, culturally diverse content coverage, and demographic disaggregation of detection metrics. We connect these to the governance frameworks currently mandating watermarking deployment and show that watermarking is held to a lower fairness standard than the generative systems it is meant to govern. Our position is that evaluation must precede deployment, and that the same bias auditing requirements applied to AI models should extend to the verification layer. 

---
# Hybrid Retrieval for COVID-19 Literature: Comparing Rank Fusion and Projection Fusion with Diversity Reranking 

**Authors**: Harishkumar Kishorkumar Prajapati  

**Link**: [PDF](https://arxiv.org/pdf/2604.13728)  

**Abstract**: We present a hybrid retrieval system for COVID-19 scientific literature, evaluated on the TREC-COVID benchmark (171,332 papers, 50 expert queries). The system implements six retrieval configurations spanning sparse (SPLADE), dense (BGE), rank-level fusion (RRF), and a projection-based vector fusion (B5) approach. RRF fusion achieves the best relevance (nDCG@10 = 0.828), outperforming dense-only by 6.1% and sparse-only by 14.9%. Our projection fusion variant reaches nDCG@10 = 0.678 on expert queries while being 33% faster (847 ms vs. 1271 ms) and producing 2.2x higher ILD@10 than RRF. Evaluation across 400 queries -- including expert, machine-generated, and three paraphrase styles -- shows that B5 delivers the largest relative gain on keyword-heavy reformulations (+8.8%), although RRF remains best in absolute nDCG@10. On expert queries, MMR reranking increases intra-list diversity by 23.8-24.5% at a 20.4-25.4% nDCG@10 cost. Both fusion pipelines evaluated for latency remain below the sub-2 s target across all query sets. The system is deployed as a Streamlit web application backed by Pinecone serverless indices. 

---
# (How) Learning Rates Regulate Catastrophic Overtraining 

**Authors**: Mark Rofin, Aditya Varre, Nicolas Flammarion  

**Link**: [PDF](https://arxiv.org/pdf/2604.13627)  

**Abstract**: Supervised fine-tuning (SFT) is a common first stage of LLM post-training, teaching the model to follow instructions and shaping its behavior as a helpful assistant. At the same time, SFT may harm the fundamental capabilities of an LLM, particularly after long pretraining: a phenomenon known as catastrophic overtraining (Springer et al., 2025). To understand overtraining, we first investigate catastrophic forgetting in finetuning through the lens of implicit regularization of the learning rate. For models trained to the same SFT loss, we identify how the learning rate mediates optimization: finetuning with large and small steps converges to qualitatively different models. Next, we link forgetting to overtraining: learning rate decay increases the sharpness of the pretrained model, which in turn exacerbates catastrophic forgetting during SFT, leading to overtraining. Our findings paint a picture of the overtraining mechanism in LLMs and broadly contribute to the understanding of the interplay between optimization dynamics during pretraining and finetuning. 

---
# Chain of Uncertain Rewards with Large Language Models for Reinforcement Learning 

**Authors**: Shentong Mo  

**Link**: [PDF](https://arxiv.org/pdf/2604.13504)  

**Abstract**: Designing effective reward functions is a cornerstone of reinforcement learning (RL), yet it remains a challenging and labor-intensive process due to the inefficiencies and inconsistencies inherent in traditional methods. Existing methods often rely on extensive manual design and evaluation steps, which are prone to redundancy and overlook local uncertainties at intermediate decision points. To address these challenges, we propose the Chain of Uncertain Rewards (CoUR), a novel framework that integrates large language models (LLMs) to streamline reward function design and evaluation in RL environments. Specifically, our CoUR introduces code uncertainty quantification with a similarity selection mechanism that combines textual and semantic analyses to identify and reuse the most relevant reward function components. By reducing redundant evaluations and leveraging Bayesian optimization on decoupled reward terms, CoUR enables a more efficient and robust search for optimal reward feedback. We comprehensively evaluate CoUR across nine original environments from IsaacGym and all 20 tasks from the Bidexterous Manipulation benchmark. The experimental results demonstrate that CoUR not only achieves better performance but also significantly lowers the cost of reward evaluations. 

---
# From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines 

**Authors**: Sunkyung Lee, Jihye Back, Donghyeon Jeon, Soonhwan Kwon, Moonkwon Kim, Inho Kang, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.13468)  

**Abstract**: Generative information retrieval (GenIR) formulates the retrieval process as a text-to-text generation task, leveraging the vast knowledge of large language models. However, existing works primarily optimize for relevance while often overlooking document trustworthiness. This is critical in high-stakes domains like healthcare and finance, where relying solely on semantic relevance risks retrieving unreliable information. To address this, we propose an Authority-aware Generative Retriever (AuthGR), the first framework that incorporates authority into GenIR. AuthGR consists of three key components: (i) Multimodal Authority Scoring, which employs a vision-language model to quantify authority from textual and visual cues; (ii) a Three-stage Training Pipeline to progressively instill authority awareness into the retriever; and (iii) a Hybrid Ensemble Pipeline for robust deployment. Offline evaluations demonstrate that AuthGR successfully enhances both authority and accuracy, with our 3B model matching a 14B baseline. Crucially, large-scale online A/B tests and human evaluations conducted on the commercial web search platform confirm significant improvements in real-world user engagement and reliability. 

---
# Functional Emotions or Situational Contexts? A Discriminating Test from the Mythos Preview System Card 

**Authors**: Hiranya V. Peiris  

**Link**: [PDF](https://arxiv.org/pdf/2604.13466)  

**Abstract**: The Claude Mythos Preview system card deploys emotion vectors, sparse autoencoder (SAE) features, and activation verbalisers to study model internals during misaligned behaviour. The two primary toolkits are not jointly reported on the most alignment-relevant episodes. This note identifies two hypotheses that are qualitatively consistent with the published results: that the emotion vectors track functional emotions that causally drive behaviour, or that they are a projection of a richer situational-context structure onto human emotional axes. The hypotheses can be distinguished by a test the system card does not report: applying emotion probes to the strategic concealment episodes where only SAE features are currently documented. If emotion probes show flat activation while SAE features are strongly active, the alignment-relevant structure lies outside the emotion subspace. Which hypothesis is correct determines whether emotion-based monitoring will robustly detect dangerous model behaviour or systematically miss it. 

---
# WebXSkill: Skill Learning for Autonomous Web Agents 

**Authors**: Zhaoyang Wang, Qianhui Wu, Xuchao Zhang, Chaoyun Zhang, Wenlin Yao, Fazle Elahi Faisal, Baolin Peng, Si Qin, Suman Nath, Qingwei Lin, Chetan Bansal, Dongmei Zhang, Saravan Rajmohan, Jianfeng Gao, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.13318)  

**Abstract**: Autonomous web agents powered by large language models (LLMs) have shown promise in completing complex browser tasks, yet they still struggle with long-horizon workflows. A key bottleneck is the grounding gap in existing skill formulations: textual workflow skills provide natural language guidance but cannot be directly executed, while code-based skills are executable but opaque to the agent, offering no step-level understanding for error recovery or adaptation. We introduce WebXSkill, a framework that bridges this gap with executable skills, each pairing a parameterized action program with step-level natural language guidance, enabling both direct execution and agent-driven adaptation. WebXSkill operates in three stages: skill extraction mines reusable action subsequences from readily available synthetic agent trajectories and abstracts them into parameterized skills, skill organization indexes skills into a URL-based graph for context-aware retrieval, and skill deployment exposes two complementary modes, grounded mode for fully automated multi-step execution and guided mode where skills serve as step-by-step instructions that the agent follows with its native planning. On WebArena and WebVoyager, WebXSkill improves task success rate by up to 9.8 and 12.9 points over the baseline, respectively, demonstrating the effectiveness of executable skills for web agents. The code is publicly available at this https URL. 

---
# Indexing Multimodal Language Models for Large-scale Image Retrieval 

**Authors**: Bahey Tharwat, Giorgos Kordopatis-Zilos, Pavel Suma, Ian Reid, Giorgos Tolias  

**Link**: [PDF](https://arxiv.org/pdf/2604.13268)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong cross-modal reasoning capabilities, yet their potential for vision-only tasks remains underexplored. We investigate MLLMs as training-free similarity estimators for instance-level image-to-image retrieval. Our approach prompts the model with paired images and converts next-token probabilities into similarity scores, enabling zero-shot re-ranking within large-scale retrieval pipelines. This design avoids specialized architectures and fine-tuning, leveraging the rich visual discrimination learned during multimodal pre-training. We address scalability by combining MLLMs with memory-efficient indexing and top-$k$ candidate re-ranking. Experiments across diverse benchmarks show that MLLMs outperform task-specific re-rankers outside their native domains and exhibit superior robustness to clutter, occlusion, and small objects. Despite strong results, we identify failure modes under severe appearance changes, highlighting opportunities for future research. Our findings position MLLMs as a promising alternative for open-world large-scale image retrieval. 

---
# From Seeing it to Experiencing it: Interactive Evaluation of Intersectional Voice Bias in Human-AI Speech Interaction 

**Authors**: Shree Harsha Bokkahalli Satish, Maria Teleki, Christoph Minixhofer, Ondrej Klejch, Peter Bell, Éva Székely  

**Link**: [PDF](https://arxiv.org/pdf/2604.13067)  

**Abstract**: SpeechLLMs process spoken language directly from audio, but accent and vocal identity cues can lead to biased behaviour. Current bias evaluations often miss how such bias manifests in end-to-end speech interactions and how users experience it. We distinguish quality-of-service disparities (e.g., off-topic or low-effort responses) from content-level bias in coherent outputs, and examine intersectional effects of accent and perceived gender. In this work, we explore a two-part evaluation approach: (1) a controlled test cohort spanning six accents and two gender presentations, analysed with judge-free prompt-response metrics, and (2) an interactive study design using voice conversion to let users experience identical content through different vocal identities. Across two studies (Interactive, N=24; Observational, N=19), we find that voice conversion increases trust and acceptability for benign responses and encourages perspective-taking, while automated analysis in search of quality-of-service disparities, reveals {accent x gender} disparities in alignment and verbosity across SpeechLLMs. These results highlight voice conversion for probing and experiencing intersectional voice bias while our evaluation suite provides richer bias evaluations for spoken conversational AI. 

---
# Form Without Function: Agent Social Behavior in the Moltbook Network 

**Authors**: Saber Zerhoudi, Kanishka Ghosh Dastidar, Felix Klement, Artur Romazanov, Andreas Einwiller, Dang H. Dang, Michael Dinzinger, Michael Granitzer, Annette Hautli-Janisz, Stefan Katzenbeisser, Florian Lemmerich, Jelena Mitrovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.13052)  

**Abstract**: Moltbook is a social network where every participant is an AI agent. We analyze 1,312,238 posts, 6.7~million comments, and over 120,000 agent profiles across 5,400 communities, collected over 40 days (January 27 to March 9, 2026). We evaluate the platform through three layers. At the interaction layer, 91.4% of post authors never return to their own threads, 85.6% of conversations are flat (no reply ever receives a reply), the median time-to-first-comment is 55 seconds, and 97.3% of comments receive zero upvotes. Interaction reciprocity is 3.3%, compared to 22-60% on human platforms. An argumentation analysis finds that 64.6% of comment-to-post relations carry no argumentative connection. At the content layer, 97.9% of agents never post in a community matching their bio, 92.5% of communities contain every topic in roughly equal proportions, and over 80% of shared URLs point to the platform's own infrastructure. At the instruction layer, we use 41 Wayback Machine snapshots to identify six instruction changes during the observation window. Hard constraints (rate limit, content filters) produce immediate behavioral shifts. Soft guidance (``upvote good posts'', ``stay on topic'') is ignored until it becomes an explicit step in the executable checklist. The platform also poses technological risks. We document credential leaks (API keys, JWT tokens), 12,470 unique Ethereum addresses with 3,529 confirmed transaction histories, and attack discourse ranging from template-based SSH brute-forcing to multi-agent offensive security architectures. These persist unmoderated because the quality-filtering mechanisms are themselves non-functional. Moltbook is a socio-technical system where the technical layer responds to changes, but the social layer largely fails to emerge. The form of social media is reproduced in full. The function is absent. 

---
# A Domain-Specific Language for LLM-Driven Trigger Generation in Multimodal Data Collection 

**Authors**: Philipp Reis, Philipp Rigoll, Martin Zehetner, Jacqueline Henle, Stefan Otten, Eric Sax  

**Link**: [PDF](https://arxiv.org/pdf/2604.13046)  

**Abstract**: Data-driven systems depend on task-relevant data, yet data collection pipelines remain passive and indiscriminate. Continuous logging of multimodal sensor streams incurs high storage costs and captures irrelevant data. This paper proposes a declarative framework for intent-driven, on-device data collection that enables selective collection of multimodal sensor data based on high-level user requests. The framework combines natural language interaction with a formally specified domain-specific language (DSL). Large language models translate user-defined requirements into verifiable and composable DSL programs that define conditional triggers across heterogeneous sensors, including cameras, LiDAR, and system telemetry. Empirical evaluation on vehicular and robotic perception tasks shows that the DSL-based approach achieves higher generation consistency and lower execution latency than unconstrained code generation while maintaining comparable detection performance. The structured abstraction supports modular trigger composition and concurrent deployment on resource-constrained edge platforms. This approach replaces passive logging with a verifiable, intent-driven mechanism for multimodal data collection in real-time systems. 

---
