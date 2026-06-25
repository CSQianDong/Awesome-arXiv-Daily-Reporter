# Real-Time Voice AI Hears but Does Not Listen 

**Authors**: Martijn Bartelds, Federico Bianchi, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2606.26083)  

**Abstract**: Speech conveys information through both words and vocal delivery. We evaluate four leading production realtime voice systems-OpenAI's GPT Realtime 2, Google's Gemini 3.1 Flash Live, and Alibaba's Qwen3.5 Omni Plus and Omni Flash-on tasks where the words and the delivery patterns both convey meaningful information. Across three consequential scenarios, all four systems act on the words rather than the voice. They end calls with crying callers who insist nothing is wrong, approve wire transfers authorized in frightened voices, and enroll callers whose agreement is clearly sarcastic. Surprisingly, this is often not a failure of perception. When asked directly, three of the four systems reliably identify the distress, fear, or sarcasm they later ignore when making decisions. We observe a similar pattern when these realtime voice systems estimate accent and age, as their responses frequently follow the biases of the words rather than the acoustic properties of the speaker. We term this disconnect between perception and action the emotional intelligence gap of voice AI. Prompting systems to explicitly attend to vocal delivery improves performance only partially and inconsistently. Our findings show that current realtime voice AI systems often behave as if speech had been reduced to a transcript, suggesting that they should be used with caution in settings where the tone and emotion of delivery convey important information. 

---
# Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models 

**Authors**: Akshay Paruchuri, Sanmi Koyejo, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2606.26079)  

**Abstract**: Standard benchmarks for multimodal large language models (MLLMs) score each item on one canonical ordering and miss whether order-irrelevant shuffling changes the answer, a baseline reliability property called for by emerging AI evaluation guidelines. We introduce Facet-Probe, a five-facet audit (option, evidence-chunk, document-rank, image-set, and mixed-modality ordering) of 18 frontier and open-weight MLLMs. A Bayesian item-response model separates ordering noise from per-facet bias, and a same-ordering control estimates the decoder-stochastic floor for observed flips. We find that none of the 18 MLLMs we audit are order-invariant: screened per-facet panel-mean flip rates span 24-50%. A Gemini same-ordering control at temperature 0 estimates a substantial ordering excess over a same-input decoder-noise floor in verified cells. Capability predicts but does not eliminate flips; the best model still flips on 13.4% of trials. In our Gemini mitigation tests, training-free prompt changes are modality-conditional and do not transfer from text to visual reasoning. These results suggest that prompt-level mitigation alone is unlikely to provide general order robustness, motivating future work on training-time and architectural approaches. We propose cross-ordering flip rate as a standard reporting axis for MLLMs. 

---
# When Certainty Is an Artifact: Keyword Lexicon Blindness and the (Mis)Measurement of Rhetorical Stance 

**Authors**: Bo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.26062)  

**Abstract**: Can a statistically significant, large-effect-size finding in computational social science be entirely an artifact of the measurement instrument? We present a case where the answer appears to be yes. Analyzing 85 interviews across four public intellectuals (2016--2026), we find a robust negative-affect/emphatic-certainty lexical co-occurrence pattern under keyword-based scoring ($r = 0.72$--$0.93$, $p < 0.01$ for all four speakers). Replacing keyword counting with LLM-based zero-shot semantic classification on the complete diarized corpus (32,625 sentences) dramatically reduces this correlation: Dalio's $r = 0.851$ drops to $r = 0.206$, with two speakers showing negative $r(\text{neg}, \text{emphatic})$ and one showing null. In contrast, the LLM reveals a strong negative-hedging coupling across speakers -- Rogoff's $r(\text{neg}, \text{hedged}) = 0.875$ ($p = 0.001$) and Zeihan's $r(\text{neg}, \text{hedged}) = 0.722$ ($p = 0.008$) -- consistent with the conventional expectation that pessimistic discourse attracts hedging, not certainty. Sentence-level error analysis traces this discrepancy to three structural failure modes in keyword lexicons -- syntactic blindness, polysemy blindness, and categorical absence -- illustrated through cases where keyword counting inverts semantic meaning (e.g., ''never absolutely totally confident'' scored as high-certainty). We argue that keyword lexicons measure a universal lexical co-occurrence tendency -- negative discourse naturally attracts emphatic vocabulary -- that is orthogonal to, and can systematically invert, rhetorical stance. Treating keyword counts as measurements of epistemic certainty is a category error: a finding that appears to be about a speaker's psychology may be entirely about the counting of words. 

---
# AI translation of literary texts is "fine", but readers still prefer human translations 

**Authors**: Yves Ferstler, Adam Podoxin, Ty Brassington, Roman Grundkiewicz, Maite Taboada, Marzena Karpinska  

**Link**: [PDF](https://arxiv.org/pdf/2606.26040)  

**Abstract**: AI translation of literary works is increasingly common. While the content may be rendered adequately, we do not know enough about how readers experience it in terms of immersiveness and literary effect, aspects poorly captured by automatic machine translation metrics or human evaluation targeting fluency and adequacy. We ask 15 avid readers to compare recently published human translations (HT) to machine translations (MT) generated with an agentic large language model (LLM)-based pipeline, for 15 recent novels in French, Polish, and Japanese and translated into English. Readers evaluated approximately 8K-word excerpts in two conditions: immersive reading of the whole excerpt (30 comparisons) and close reading of 386 aligned HT-MT chunk pairs (772 comparisons), with two readers per book and in alternating order of presentation. Overall, readers find MT "fine", but prefer HT (slightly at excerpt-level 19/30, more clearly at chunk-level 522/772) for its ease, clarity, and immersive nature. Readers' highlights show that MT's quality varies more within one book than HT's does. Crucially, readers cannot reliably tell the two apart (17/30 guess correctly) and tend to prefer the version they believe to be human. Automatic metrics, including LLM-as-a-judge approaches, fail to recover reader preferences and favor MT. We release LAIT (Literary AI Translation), a reader-centered evaluation dataset with 1K reader comments, 2K judgments and preference ratings, and 7.2K span-level annotations, along with our evaluation protocol and supporting interface. 

---
# Detect, Unlearn, Restore: Defending Text Summarization Models Against Data Poisoning 

**Authors**: Poojitha Thota, Shirin Nilizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2606.26036)  

**Abstract**: Training-time data poisoning during fine-tuning poses a significant threat to large language models (LLMs) deployed for abstractive text summarization, where small task-specific datasets exert disproportionate influence on model behavior. In this setting, adversaries manipulate fine-tuning data to induce persistent summarization failures, such as biased or harmful summaries, while preserving standard evaluation metrics. We present a unified post-hoc defense framework for detecting and remediating fine-tuning-stage poisoning in summarization models across the machine learning supply chain. Our experiments show that in white-box settings, poisoned document-summary pairs exhibit abnormally high training influence, enabling detection via influence-function analysis with semantic consistency checks. In black-box settings, poisoned models display two to three times greater sensitivity to semantics-preserving perturbations, enabling behavioral auditing without training data access. Beyond existing poisoning formulations, we introduce novel attacks targeting factual distortion and representational bias, showing that poisoning alters summarization behavior without triggering conventional alarms. Across nine architectures and six benchmark datasets under adaptive attacks, our defenses achieve 85-92% detection precision, while gradient-ascent unlearning restores up to 96% of original behavior with minimal utility loss (less than 0.6% ROUGE degradation). These results indicate that fine-tuning-time poisoning leaves persistent structural artifacts, enabling practical detection and post-deployment recovery without full retraining. 

---
# Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It 

**Authors**: Yupu Hao, Zhuoran Jin, Huanxuan Liao, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.26027)  

**Abstract**: Tool use enables large language models (LLMs) to perform complex tasks, and recent agentic reinforcement learning (RL) methods show promise for enhancing model capabilities. However, RL alone often leads to instability or limited gains in tool-use tasks. In our experiments, some models exhibit catastrophic collapse, where performance abruptly drops and tool-invocation structures fail. The analysis reveals that these failures stem from unexpected probability spikes in specific control tokens, disrupting structured execution, yet the underlying tool-use capability remains intact, merely obscured by specific formats. To address this, we systematically investigate a diverse set of supervisory signals, including off-policy supervision, hint-based guidance, erroneous example supervision, and others, applied under both synchronous and interleaved training schemes. We find that interleaving supervised fine-tuning (SFT) with RL substantially improves stability, but exhibits degraded performance under format and content out-of-distribution (OOD) evaluation. We also analyze the impact of learning rates and generalization across settings. These results highlight the importance of understanding RL failures and demonstrate how diverse supervisory signals can guide exploratory learning, enabling robust training of LLMs for complex, multi-step tool-use tasks. Our Code is available at this https URL. 

---
# The Tatoxa System for Text Detoxification in Low-Resource Languages: The Case of Tatar 

**Authors**: Ilseyar Alimova, Bogdan Monogov, Artyom Mazur, Daniil Antonov, Vsevolod Karimov, Vitaliy Egorov, Bulat Khakimov, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.26015)  

**Abstract**: Text detoxification, the automated detection and mitigation of abusive and harmful content, is essential for ensuring the safety of online communities and protecting users. However, low resource languages such as Tatar have received little research attention. In this paper we present Tatoxa, a novel state-of-the-art system for text detoxification in the Tatar language. Comparative experiments show that the proposed approach outperforms existing open source and proprietary commercial LLMs on key quality metrics. We also introduce a new dataset for text detoxification in Tatar, designed for fine tuning and evaluation in low resource settings. Finally, cross lingual transfer experiments indicate that transfer from other languages, including the culturally close Russian, performs significantly worse than training on native Tatar data even when a large Russian corpus is available. 

---
# Dziri Voicebot: An End-to-End Low-Resource Speech-to-Speech Conversational System for Algerian Dialect 

**Authors**: Dihia Lanasri, Fairouz Taki, Asma Kemmoum  

**Link**: [PDF](https://arxiv.org/pdf/2606.26003)  

**Abstract**: Automatic speech and language technologies are still heavily biased toward high-resource languages, limiting their applicability to dialectal and low-resource settings such as Algerian Dialect. This language presents additional challenges including lack of standardized orthography, frequent codeswitching with French, and scarcity of annotated speech resources. This paper addresses the problem of building a complete speech-to-speech conversational system for Algerian Dialect. We propose a modular pipeline integrating automatic speech recognition, natural language understanding, retrieval-augmented generation, and text-to-speech synthesis within a unified architecture. This work is the continuation of our previous work on Algerian dialectal conversational systems Bechiri and Lanasri [2026], extending it from text-based dialogue modeling to full speech-based interaction. We constructed dedicated datasets for ASR, NLU, and TTS in the telecom domain and fine-tune pretrained models for each component. The ASR system is built on Whisper-based adaptation, while the NLU module combines transformer-based embeddings with a task-oriented dialogue framework. A neural TTS system is trained on a newly collected dialectal corpus to enable spoken response generation. Experimental results show strong performance across all components, including low word error rate for ASR, high intent classification and entity recognition scores for NLU, and stable speech synthesis quality. The proposed system provides a reproducible baseline for end-to-end conversational modeling in Algerian Dialect. 

---
# SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models 

**Authors**: Liang-Yuan Wu, Zih-Ching Chen, Tongshuang Wu, Chao-Han Huck Yang, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25990)  

**Abstract**: As multimodal conversational systems increasingly engage in spoken interaction, their ability to navigate paralinguistic social cues has become a critical bottleneck for natural human-AI communication. However, existing evaluations of machine emotional intelligence assess reasoning exclusively through isolated text or passive acoustic perception, overlooking the complex cross-modal reasoning required for active, multi-turn dialogue. We introduce \textsc{SpeechEQ}, a comprehensive framework designed to evaluate the sociolinguistic reasoning of Speech-Language Models (SLMs). The framework includes a validated dataset of 2,265 dialogues across 15 Emotional Quotient (EQ) subscales grounded in EQ-i 2.0 theory, along with a multi-turn evaluation protocol measured by our proposed Spoken EQ (SEQ) score inspired by human EQ assessments. Experiments show limitations in how both existing Speech Emotion Recognition and end-to-end Speech-Language Models understand and apply paralinguistic cues through speech. While end-to-end architectures outperform cascaded systems, \textsc{SpeechEQ} reveals that current multimodal models remain bottlenecked by a text-reliant ``modality shortcut,'' an alignment-induced ``safety trap,'' and ``contextual amnesia,'' highlighting the barriers to truly emotionally aware AI. Our benchmark can be accessed at this https URL and demo page at this https URL 

---
# Weave of Formal Thought 

**Authors**: Alexandre Bouayad  

**Link**: [PDF](https://arxiv.org/pdf/2606.25987)  

**Abstract**: Large language models (LLMs) attain remarkable surface fluency on code, yet they neither formally guarantee the syntactic validity of their output nor leverage the hierarchical structure defining the target language. While existing constrained-decoding frameworks address the former, they operate under rigid assumptions that preclude critical lexical mechanisms -- including context-sensitive lexing, maximal-munch tokenization, and keyword extraction -- and only approximate vocabulary masking, sacrificing completeness. For the latter, code LLMs typically inject grammatical structure via predetermined policies rather than learning which structural information to expose. In this work, we introduce Weave of Formal Thought (WoFT), a paradigm uniting rigorous syntactic validation with learned structural representations. First, we present a formal engine and constrained decoder that is sound and complete with respect to the full Tree-sitter specification. By augmenting generalized LR (GLR) parsing with a speculative-lexing construction that maintains concurrent lexer-state hypotheses synchronized with a GLR graph-structured stack, our decoder admits every subword token extending to a valid program prefix and rejects all others. Second, we present a latent-variable fine-tuning method training the language model to interleave non-terminal grammar symbols directly into generation. Utilizing the reweighted wake-sleep (RWS) algorithm to optimize the importance-weighted evidence lower bound (IW-ELBO) of the surface text, the model learns to selectively retain formal derivations as an adaptive structural scratchpad. For Python, fine-tuning StarCoder2-3B with our RWS objective reduces per-token cross-entropy by 14.3% relative to a text-only SFT baseline, demonstrating that discretionary latent syntax recovers critical structural information that flat autoregressive training discards. 

---
# Overview of HIPE-2026: Person-Place Relation Extraction from Multilingual Historical Texts 

**Authors**: Juri Opitz, Maud Ehrmann, Corina Raclé, Andrianos Michail, Matteo Romanello, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2606.25935)  

**Abstract**: Was this person ever at that place, and if so, when? Answering such questions from noisy, multilingual historical documents is the central challenge of HIPE-2026, the third edition of the HIPE evaluation series. Moving from named entity recognition and linking (HIPE-2020, HIPE-2022) to reasoning about relationships between entities, HIPE-2026 targets two temporally grounded relation types: $at$, indicating that a person was present at a location at some point prior to a document's publication date, and $isAt$, indicating presence contemporaneous with that date. This paper presents the results of the evaluation campaign, which confronted 17 participating teams with the challenges of historical language variation, OCR noise, and indirect contextual cues across three languages: French, German, and English. The datasets include historical newspaper text from the nineteenth and twentieth centuries, as well as a surprise-domain generalization set drawn from early modern French literary texts. A distinctive feature of HIPE-2026 is its three-fold evaluation framework, which assesses predictive accuracy, computational efficiency, and cross-domain generalization, reflecting the practical demands of large-scale historical document processing in the cultural heritage domain. Across more than 40 submitted runs, results reveal a wide range of strategies, from state-of-the-art large language models to lightweight task-specific classifiers, and highlight the trade-offs between accuracy, efficiency, and robustness inherent to historical relation extraction at corpus scale. System descriptions, datasets, and findings are presented and discussed, offering a detailed picture of the current state of temporally grounded relation extraction for historical documents. 

---
# SARA: Unlocking Multilingual Knowledge in Mixture-of-Experts via Semantically Anchored Routing Alignment 

**Authors**: Tianyu Dong, Yangyang Liu, Jiang Zhou, Xinwei Wu, Xiaohu Zhao, Hao Wang, Heng Liu, Linlong Xu, Longyue Wang, Weihua Luo, Shaolin Zhu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.25821)  

**Abstract**: Sparse Mixture-of-Experts (MoE) architectures have emerged as an increasingly influential paradigm as they offer a strategic balance between parameter scalability and computational efficiency. However, low-resource languages, which suffer from a scarcity of high-quality training data, often have their tokens routed to different experts than those predominantly activated by high-resource inputs, which limits cross-lingual expert sharing. This cross-lingual routing divergence consequently hinders their efficacy in multilingual contexts. To address this issue, we propose SARA (Semantically Anchored Routing Alignment), a framework designed to transfer specialized capabilities from high-resource languages as anchors to low-resource languages. SARA explicitly aligns the routing distribution of multilingual inputs with high-resource semantic anchors using a symmetric Jensen-Shannon (JS) divergence constraint. Unlike traditional distillation methods that operate on output logits, SARA directly aligns the internal routing distributions of MoE layers, encouraging mechanistic consistency in expert selection across languages. We conduct experiments on 2 LLMs across 5 low-resource languages and 3 benchmarks. Experiment results demonstrate that SARA outperforms standard instruction tuning, e.g., +0.8% on Qwen3-30B-A3B and +1.2% on Phi-3.5-MoE-instruct on Global-MMLU. Further analyses show that SARA effectively addresses performance bottlenecks in low-resource languages, providing a scalable pathway to enhance multilingual capabilities in sparse architectures. 

---
# Beyond Function Calling: Benchmarking Tool-Using Agents under Tool-Environment Unreliability 

**Authors**: Yang Tian, Zhengpeng Shi, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.25819)  

**Abstract**: Large language models are increasingly deployed as agents that solve tasks by interacting with external tool environments. Although recent tool-use benchmarks increasingly cover complex task settings, they still largely assume clean, stable, and trustworthy tool environments, leaving tool-environment unreliability insufficiently examined. We introduce ToolBench-X, a benchmark for evaluating agents under recoverable reliability hazards. ToolBench-X contains executable multi-step tasks across diverse domains and sequential, parallel, and mixed workflows, each paired with deterministic tools and a canonical final answer for automatic evaluation. Starting from clean tool environments, ToolBench-X injects five structured hazard types: Specification Drift, Invocation Error, Execution Failure, Output Drift, and Cross-source Conflict. Crucially, each injected instance remains solvable through at least one valid recovery path, such as retrying, fallback, verification, or cross-checking. Experiments reveal a substantial reliability gap: agents that perform well with reliable tools often fail under recoverable hazards. Further analysis shows that failures are driven less by tool-use volume or inference budget than by limited hazard diagnosis and ineffective recovery. Targeted recovery hints recover many failed tasks, while test-time scaling yields more limited gains. These results suggest that tool-use evaluation should move beyond function-call accuracy toward task completion under unreliable tool environments. The code and data is available at this https URL. 

---
# Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges for LLM Adversarial Evaluation 

**Authors**: Han Jeon, Shiv Medler, Joseph Voyles, Matt Wood  

**Link**: [PDF](https://arxiv.org/pdf/2606.25782)  

**Abstract**: With the widespread adoption of large language models (LLMs) in chatbots and everyday applications, companies increasingly need guardrails that are effective while remaining low-cost and low-latency. Safety evaluation of LLM outputs has generally relied on LLM-based judges, which can be effective but are often slow and expensive to deploy at scale.
In this paper, we evaluate whether fine-tuned modern encoder classifiers from the ModernBERT family, including ModernBERT and Ettin, can reliably identify harmful LLM outputs in user-model conversations without substantial performance loss relative to LLM-based judges. We benchmark these encoder classifiers against rule-based prefix matching, fine-tuned LLM classifiers, and LLM judges using a range of judge-prompting strategies across open-source adversarial datasets.
The LLM judges include evaluation methodologies from StrongReject, ShieldGemma, JailbreakBench, AILuminate, SorryBench, and a Claude-as-a-judge setup, as well as fine-tuned safety classifiers such as LlamaGuard 3 and LlamaGuard 4. The encoder classifiers are fine-tuned on judge-labeled data using a majority-voting label strategy and are then evaluated on a gold-standard holdout dataset to assess their performance relative to LLM judges.
We report absolute performance using F1 score, false negative rate, and precision-recall metrics. We also break down results by attack technique, including single-turn prompting, decomposition, escalation, and context manipulation, to identify where encoder classifiers align with or diverge from LLM-based judges. Our findings provide guidance on when encoder classifiers can serve as cost- and latency-efficient alternatives to LLM-based safety evaluation. 

---
# OPERA: Aligning Open-Ended Reasoning via Objective Perplexity-based Reinforcement Learning 

**Authors**: Wenxuan Jiang, Zining Fan, Zijian Zhang, Xuecheng Wu, Hongming Tan, Haoyang Dai, Xiaoyu Li, Xuezhi Cao, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25757)  

**Abstract**: Reinforcement Learning (RL) has enabled LLMs to excel in objective reasoning tasks such as mathematics and code generation. However, applying RL to open-ended tasks, such as creative writing, remains challenging because LLM-as-a-judge reward models often exhibit stylistic biases and positional inconsistencies, leading to unstable supervision. To address this, we propose OPERA (Objective Perplexity-based Reflective Alignment), which replaces unreliable external judges with intrinsic rewards derived from perplexity dynamics. Specifically, we derive an intrinsic reward signal from perplexity dynamics, quantifying uncertainty reduction at critical reflective states. During the cold-start phase, we introduce a data synthesis method that leverages carefully designed guiding words to generate diverse reasoning traces, along with perplexity-prioritized rollouts that utilize internal log-probabilities to identify logically consistent reasoning branches. This pipeline yields a large-scale dataset comprising 20,000 high-quality reasoning trajectories. Empirical evaluations consistently demonstrate the scalability and efficacy of our approach in alignment for open-ended tasks. Implementing OPERA on Qwen3-8B establishes a new state-of-the-art among open-source models, achieving parity with or surpassing proprietary models like Gemini2.5 and MiniMax-M2.5 in some open-ended tasks. The code is available at this https URL. 

---
# BitNet Text Embeddings 

**Authors**: Zhen Li, Xin Huang, Liang Wang, Nan Yang, Ting Song, Yan Xia, Xun Wu, Shaohan Huang, Huishuai Zhang, Furu Wei, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.25674)  

**Abstract**: LLM-based text embedders have substantially improved retrieval and semantic representation quality, but their deployment remains costly: large backbone models slow down embedding inference, while high-dimensional full-precision embeddings impose substantial storage and bandwidth overhead on large-scale indexes. In this paper, we present BITEMBED, an extreme low-bit framework for LLM-based text embedding that jointly targets encoding efficiency and vector storage. BITEMBED converts pretrained LLM backbones into BitNet-style embedding encoders with ternary weights, quantized activations, and lightweight normalization refinement. The converted model is adapted to representation learning through continual contrastive pre-training, followed by supervised contrastive fine-tuning with both similarity-distribution distillation and attention-relation distillation from a full-precision teacher. Beyond quantizing the backbone, BITEMBED further trains output embeddings to support multiple storage precisions meeting different storage needs in various scenarios. Experiments on MMTEB (eng, v2) with Qwen3-0.6B and Gemma3-270M show that BITEMBED is largely comparable to full precision teacher embedders. Moreover, BITEMBED flexibly obtains text embeddings of various precisions, achieving a trade-off between performance and storage cost. 

---
# Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization 

**Authors**: Long Chen, Ryan Razkenari, Yuxuan Zhou, Yuan Tian, Rahul Ghosh, Venkatesh Pappakrishnan, Disha Ahuja, Vidya Sagar Ravipati  

**Link**: [PDF](https://arxiv.org/pdf/2606.25656)  

**Abstract**: As advanced RAG variants like GraphRAG and Agentic RAG emerge, one leading question is when and how to use them. Here, we introduce a framework for different RAG scenarios evaluation and comparison on semi-structured knowledge bases, including regular RAG, GraphRAG, Modular RAG and Agentic RAG. We provide implementation for 9 standardized RAG scenarios, and conduct experiments for a comprehensive comparison. These scenarios are designed for real use cases regarding data and domain restrictions, spanning from simple document-based retrieval to advanced features such as hybrid text-graph retrieval, integration with computed or pre-defined domain knowledge graphs, agentic multi-step planning, and agent-graph integration. Besides, we present a novel context engineering method for GraphRAG and Agentic RAG, addressing the context/memory overflow issues, efficiently managing text and graph retrievals with new representations and agentic loop design, leading to 19%-53% reduction on token usage. Moreover, further analysis identifies a retrieval-generation gap where expanded retrieval does not proportionally improve generation quality, suggesting retrieval-oriented metrics overstate advanced retrieval benefits. This work provides data-driven insights on when and how to use them for building production-ready intelligent RAG systems. 

---
# MedGuards: Multi-Agent System for Reliable Medical Error Detection and Correction 

**Authors**: Congbo Ma, Hu Wang, Yichun Zhang, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2606.25651)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in healthcare settings, accurate error detection and correction in generated or existing text becomes critical, as even minor mistakes can pose risks to patient safety. Existing methods for error detection and correction, including automated checks and heuristic-based approaches, do not generalize well across unseen datasets. In this paper, we propose MedGuards as a medical safety guardrail, which is a new framework that treats medical error detection and correction as a multi-agent in-context learning task. Specialized agents separately detect, localize, and correct errors, while a confidence-guided arbitration mechanism resolves disagreements using reasoning traces and confidence scores. This design enhances interpretability, robustness, and adaptability, without requiring additional training of the base LLMs. Additionally, we introduce the Keyword-Prioritized Correction Score (KPCS), a new evaluation metric that considers whether critical keywords within the reference text are generated correctly, providing a more comprehensive assessment than conventional metrics. Experiments across four multilingual medical datasets consisting of clinical notes demonstrate significant improvements by the proposed framework across several metrics and models. Our aim is to enable safer deployment of LLMs in real-world healthcare applications. For reproducibility, we make our code publicly available at this https URL. 

---
# Staying In Character: Perspective-Bounded Memory For Book-Based Role-Playing Agents 

**Authors**: Xushuo Tang, Junhe Zhang, Zihan Yang, Yifu Tang, Sichao Li, Longbin Lai, Zhengyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25632)  

**Abstract**: Recent LLM role-playing systems build character agents from novels by extracting characters, scenes, and relations. Yet long-narrative role-playing suffers from two failures: Factual Overreach, where shared retrieval or parametric memory lets a character use facts outside its perspective, and Stylistic Monotony, where profile descriptions flatten a character into a fixed voice. To address these failures, we propose REVERIEMEM, a three-layer memory architecture for book-based character agents. The episodic layer stores first-person scene memories; the semantic layer stores visibility-tagged facts; and the personality layer stores situation-dependent speech and behaviour patterns. For evaluation, we construct KBF-QA, a 4,386-question benchmark over eight novels for testing knowledge boundaries. REVERIEMEM improves Knowledge Boundary Fidelity by 34.6 percentage points over the strongest prior method. On BOOKWORLD's five-dimension pairwise narrative protocol, REVERIEMEM achieves a ~ 79% win rate, suggesting that perspective-bounded memory improves both boundary fidelity and character-grounded narrative generation. 

---
# Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints 

**Authors**: Fangzheng Li, Aimin Zhang, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2606.25605)  

**Abstract**: Tool Calling and Structured Output are two core capabilities of modern Agent systems, yet their interaction under joint deployment conditions remains insufficiently understood. This paper reports a reproducible phenomenon observed in a production Agent system: when Tool Calling and JSON Schema constraints are simultaneously enabled, multiple open-weight models cease invoking tools despite maintaining high schema compliance. We refer to this behavior as Tool Suppression. Through controlled experiments across multiple model families and deployment settings, we consistently reproduce Tool Suppression under joint constraints, while tool execution and schema compliance remain functional when evaluated independently. Further analysis reveals that JSON Schema constraints are compiled into grammar-based token masks, causing tool-call tokens to become unreachable during decoding. This provides an implementation-level explanation for the observed behavior. To interpret the phenomenon, we formulate the Constraint Priority Inversion (CPI) hypothesis, which suggests that schema satisfaction may dominate action-selection behavior under multiple simultaneous constraints. We present CPI as a behavioral hypothesis consistent with the observed evidence rather than a verified internal mechanism. To mitigate the problem, we propose Transparent Two-Pass Execution, an inference-time strategy that decouples tool execution from schema-constrained response generation. Experimental results show that this approach restores tool invocation while preserving structured output guarantees without requiring model retraining. These findings suggest that evaluating tool use and structured output separately may overlook important reliability issues in production Agent systems. Code, data, and docs will be released at this https URL. 

---
# Riazi-8B: An Urdu Large Language Model for Mathematical Reasoning 

**Authors**: Azher Ali, Ibtsam Haider, Raja Khurram Shahzad, Seemab Latif, Mehwish Fatima  

**Link**: [PDF](https://arxiv.org/pdf/2606.25568)  

**Abstract**: Recent LLMs demonstrate strong mathematical reasoning capabilities, but existing gains rely heavily on English-centric training resources and benchmarks. As a result, reasoning performance degrades substantially in low-resource languages such as Urdu, where reasoning-oriented datasets and adapted models remain scarce. Urdu lacks both reasoning-oriented resources and models adapted for multi-step mathematical problem solving, limiting the applicability of recent progress to Urdu-speaking users. We address this gap through Riazi-8B, an Urdu mathematical reasoning model developed through a two-step adaptation process comprising continued pre-training on Urdu Wikipedia and supervised fine-tuning on Urdu Chain-of-Thought data derived from GSM8K. We evaluate Riazi-8B on MGSM-Urdu against existing Urdu instruction-tuned models. Our results show consistent improvements in answer correctness, reasoning quality, response completeness, and Urdu generation. Our findings demonstrate that combining Urdu language adaptation with reasoning-focused fine-tuning is an effective strategy for extending mathematical reasoning capabilities to low-resource languages. 

---
# BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents 

**Authors**: Hanyang Wang, Weijieying Ren, Yuxiang Zhang, Ding Cao, Zhizhao Zeng, Ke Zeng, Tianxiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.25556)  

**Abstract**: Stepwise group-based RL is an attractive way to train long-horizon LLM agents without a learned critic: it reuses multiple sampled rollouts to estimate local advantages. Its weakness is less visible but more fundamental: every group-relative estimator assumes that the steps it compares are equivalent for credit assignment. We show that current agentic variants violate this assumption through a state-action credit mismatch. The observation-hash partition is overly fine on the state side, creating singleton groups with zero step-level signal, while a single within-group mean is too coarse on the action side, mixing state-value estimation with action-specific credit. We introduce BiPACE (Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation), a drop-in advantage estimator that fixes both sides without adding a critic, auxiliary loss, or extra rollouts. BiGPO clusters steps by cosine distance in the actor's own hidden-state geometry, an empirical policy-induced proxy for bisimulation that substantially lowers the singleton rate left by observation hashing. PACE then recenters returns within each behavioral cluster using action-conditioned peer baselines; its Q-style instance estimates a local Q(s,a)-V(s) nonparametrically. On ALFWorld/Qwen2.5-7B, BiPACE_Q raises overall validation success from GiGPO's 90.8 to $97.1\pm0.9$ over three seeds, and crosses the 95% threshold on every seed, which GiGPO never does within the same budget. On Qwen2.5-1.5B it reaches $93.5\pm1.2$ versus GiGPO's 86.7, and on WebShop and TextCraft it improves over GRPO and GiGPO at both model scales. The measured BiPACE-specific overhead is 11.3% of a single training-step wall time. Yet it changes the estimator's comparison unit from surface identity to approximate behavioral equivalence plus action-side counterfactuals. The code is available at this https URL. 

---
# SFL-MTSC: Leveraging Semantic Frame-Level Multi-Task Self-Consistency for Robust Multi-Intent Spoken Language Understanding 

**Authors**: Po-Yen Chen, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25552)  

**Abstract**: Prompt-based spoken language understanding (SLU) with large language models (LLMs) often suffers from inconsistent intent--slot structures due to decoding stochasticity, particularly in multi-intent scenarios. In view of this, we propose Semantic Frame-Level Multi-Task Self-Consistency (SFL-MTSC), a novel structured aggregation framework operating at the semantic frame level. Instead of output-level majority voting, SFL-MTSC decomposes predictions into intent-specific frames, applies domain--intent grouping and slot-level clustering, and evaluates cluster reliability using path support scoring. Reliable frames are retained and re-integrated to form the final prediction. Zero-shot experiments on the MAC-SLU benchmark dataset show improved slot F1 and overall accuracy over single-path inference, while intent accuracy remains largely stable across most settings. 

---
# Fault of Our Stars: Behavioral Drivers of Rating-Sentiment Incongruence 

**Authors**: Ramanaish Abaiyan, Ruththiragayan Sutharsan, Kusal Amantha, Anusan Krishnathas, Asma Rauff, Kovindarajah Sriyathurshan, Patalee Narasinghe, Nirasha Munasinghe, Nisansa de Silva, Sandareka Wickramanayake  

**Link**: [PDF](https://arxiv.org/pdf/2606.25518)  

**Abstract**: When people share experiences online, they often express thoughts in two ways: a star rating and a written review. In sentiment analysis, ratings are widely used as convenient weak labels for textual sentiment, yet whether the two actually agree is rarely questioned. This study investigates sentiment-rating incongruence, where the sentiment expressed in review text differs from the sentiment implied by the assigned star rating, in Sri Lankan tourism attraction reviews. A dataset of 16,156 reviews from 2010 to 2023 is analyzed using a transformer-based sentiment pipeline that derives textual sentiment independently of assigned ratings. Incongruence occurs in 18.6% of reviews and falls into six directional patterns, with Conservative Rater and Obligatory 5-Star behaviors accounting for the majority of mismatches. Prevalence also varies across venue types, with museums showing the highest rates. Statistical tests, logistic regression, Random Forest, and SHAP analysis identify venue type, reviewer expertise, review length, and temporal factors as contributors to rating-text divergence. Overall, this study demonstrates that star ratings are not interchangeable with textual sentiment and should be validated before being treated as ground-truth labels in NLP. 

---
# Spam and Sentiment Detection in Arabic Tweets Using MARBERT Model 

**Authors**: Abrar Alotaibi, Atta-ur Rahman, Raheel Alhaza, Wala Alkhalifa, Narjes Alhajjaj, Atheer Alharthi, Dhai Abushoumi, Maryam Alqahtani, Dania Alkhulaifi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25495)  

**Abstract**: Saudi Telecom Company (STC) is among the most popular companies in Saudi Arabia, with many customers. Yet, there is still a big room for improvement in users' satisfaction. Social media is the most robust platform to gauge users' satisfaction and determine their sentiments and critics. Twitter is among the most popular social media platform in this regard. STC customers prefer to use Twitter to write their feedback because it's a fast way to get responses due to the STC customer services account. One way to achieve customer demands and improve customer service is using the Sentiment Analysis tool. Sentiment Analysis on Twitter is highly used because of the significant number of tweets and the different opinions. Likewise, Deep learning is the best existing Sentiment Analysis method, and it has diverse models. Bidirectional Encoder Representations from Transformers (BERT) model is one of the deep learning models which have achieved excellent results in Sentiment Analysis for Natural Language Processing (NLP). NLP is mainly investigated in the English language. However, for Arabic, there is a significant gap to be filled. This study trained the proposed model using MARBERT and measured the performance using f1-score, precision, and recall metrics. We trained the model with an Arabic dataset of 24,513 tweets, including 1,437 positive, 13,828 negative, 5,694 neutral, 1,221 sarcasm, and 2,297 indeterminate tweets. The main goal is to analyze the tweets and get the sentiment to improve STC customer service. The proposed scheme is promising in terms of accuracy in contrast to existing techniques in the literature. 

---
# How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring 

**Authors**: Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.25487)  

**Abstract**: Almost every paper on LLM jailbreaks and prompt injection reports an attack-success rate (ASR), and that number is assigned not by people but by an automated judge: either a safety classifier trained for the task, or a general chat model prompted to grade. The judge is rarely checked. We check it. Using 596 human-labeled completions from the HarmBench classifier validation set, we compare the two judge families against human majority votes and then attack them. The two families fail in opposite ways. The dedicated classifier over-flags (precision 0.835, recall 0.974); three different LLM-as-judges keep high precision (0.81 to 0.94) but show erratic recall (0.06 to 0.65), so the same responses produce very different ASR depending on which judge scores them. The two families also differ sharply in robustness. Wrappers that leave the harmful text untouched and only add benign framing flip every LLM-judge between 57% and 100% of the time, and a single prepended refusal sentence accounts for much of this (39% to 88%). The dedicated classifier resists these surface attacks (at most 6.7%), but a white-box GCG attack on its open weights flips 70% of confident true positives (21 of 30; 95% CI 54 to 86%) even at a small optimization budget. A two-annotator audit confirms the attacks leave the harm intact: every one of 80 sampled flips still contained the harmful content. Because a large and growing share of reported ASR comes from LLM-judges, many such numbers are unreliable both on average and under deliberate pressure. We recommend that papers report judge precision and recall on a human-labeled slice, report ASR corrected for judge precision, and include an adversarial check of the judge. Our code is released. 

---
# A Red Teaming Framework for Large Language Models: A Case Study on Faithfulness Evaluation 

**Authors**: Abrar Alotaibi, Raed Mughus, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2606.25476)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across natural language processing tasks, yet their deployment in high-stakes applications raises critical concerns regarding reliability, safety, and trustworthiness. In this paper, we present a red teaming framework that systematically uncovers vulnerabilities in LLM outputs. Our approach employs a novel multi-role architecture comprising target, attacker, and jury models. The attackers generate increasingly effective adversarial prompts while the jury rigorously evaluates response accuracy and consistency across tasks. In a case study, our strategy proved particularly effective at exposing unfaithfulness in LLM responses. Exploitative adversarial prompts increased the attack success rate by up to 7.9% in question-answering tasks, revealing weaknesses in reliability. The approach identifies how structural constraints in summarization can shape vulnerability patterns, with format limitations yielding measurable gains in faithfulness, and shows that architectural design choices typically outweigh parameter scaling in determining model safety. The framework's key strength is its adaptability across evaluation tasks, from English question-answering to Arabic summarization, enabling comprehensive comparison of model vulnerabilities. While it excels at comparing cross-model and cross-linguistic vulnerabilities, it faces challenges in fully automating adversarial prompt generation across languages. Our experiments also reveal limitations in detecting subtle forms of unfaithfulness that do not manifest as explicit factual contradictions, particularly across linguistic contexts. Overall, this architecture provides both actionable insights into current LLM vulnerabilities and a scalable methodology for ongoing safety evaluation as models evolve. 

---
# Optimizing Abstractive Summarization With Fine-Tuned PEGASUS 

**Authors**: Sadiul Arefin Rafi, Naimur Rahman, Kazi Nazibul Islam, Ha-mim Ahmad, Farig Yousuf Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2606.25462)  

**Abstract**: Abstractive text summarization is the technique of generating a short and concise summary comprising the salient ideas of a source text without making a subset of the salient sentences from the source text. The introduction of transformer models such as BART, T5, and PEGASUS has made this sort of summarization process more efficient and accurate. The objective of this paper is to fine-tune PEGASUS on the XL-Sum English corpus to achieve a better performance compared to the baseline mT5 model. The performance of the generated summaries from the fine-tuned model is evaluated using the ROUGE metric, which basically compares the auto-generated summaries with human-created summaries. To the best of our knowledge, the results from our fine-tuned PEGASUS model give a state-of-the-art performance on the XL-Sum English Corpus. To quantify the improvement, there is a 4.04% improvement in the ROUGE-1 score, a 15.25% increase in the ROUGE-2 score, and a 3.39% improvement in the ROUGE-L score from the baseline model. 

---
# Probing in the Wild: A Case Study of Self-Supervised Speech Representations on Mandarin Sub-dialects with Unsupervised Articulatory Analysis 

**Authors**: Shu Shang, Fuliang Weng, Zeqian Hu, Yaqian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.25459)  

**Abstract**: While self-supervised speech models have achieved strong performance across speech tasks, relatively little is known about how their internal phonetic representations behave under fine-grained dialect variation. Existing probing studies typically rely on curated corpora with manual phonetic annotations, limiting their applicability to naturally occurring dialect speech. We present a case study of articulatory feature representations in a Mandarin self-supervised speech model using an entirely unlabeled probing pipeline. Phone sequences are generated using a language-agnostic universal phone recognizer and mapped to articulatory feature vectors, enabling frame-level probing without manual annotation. Our results reveal a structured pattern in articulatory feature decodability across Mandarin sub-dialects. Acoustically salient features such as labiality and stridency remain comparatively stable, whereas features associated with finer spectral distinctions exhibit larger dialect-dependent variation. This variation is driven primarily by elevated decodability for Beijing speech relative to other Mandarin sub-dialects. Layer-wise analyses further show distinct representational dynamics for these feature groups. These findings suggest that language-agnostic articulatory probing can be applied to real-world dialect corpora and that dialect sensitivity in self-supervised speech representations is unevenly distributed across articulatory dimensions. 

---
# Reclaim Evaluation: A Lossy Memory Is Worse Than an Empty One 

**Authors**: Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2606.25449)  

**Abstract**: A language model's memory can be worse than having no memory at all. Give a model a memory that kept a wrong conclusion but dropped the work behind it, and it emits that stale value as a confident answer; give the same model an empty memory and it abstains. Across seven models this direction never reverses, a clean kill condition that none breaks. We call this brittle memory: behavioral, not the near-immediate information bound beneath it; only its magnitude is disposition- and task-dependent, not its direction. We measure it with reclaim evaluation: compress a drifted interaction at a fixed budget, then test whether a correction recovers the known answer, scored against ground truth with no judge. Correctability is bottlenecked by whether the answer-determining source survives, not by capability. A one-line source-first policy (keep the recomputable source, drop the re-derivable conclusion) restores correctability at equal budget where that source is compact and identifiable; a length-matched control rules out added text as the cause. The hand-built oracle reaches 1.00; a one-prompt deployable version reclaims 0.49-0.88. The stake compounds: chained through a memory loop, a single dropped-source error corrupts a growing span of downstream steps and stays uncorrectable, while source-first holds to a bounded budget horizon. The wall and fix replicate across three deployed memory systems and on real dialogue (MultiWOZ), and past the budget where the source no longer fits, the fix fails silently unless the note records completeness. This is a controlled study of a mechanism, not a benchmark: judge-free exact scoring, matched-budget controls, and validators built to come out false. We release the harness, conditions, and validators. 

---
# PolicyAlign: Direct Policy-Based Safety Alignment for Large Language Models 

**Authors**: Chang Wu, Junfeng Fang, Houcheng Jiang, Kai Tang, Pengyu Cheng, Xiaoxi Jiang, Guanjun Jiang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25442)  

**Abstract**: Safety alignment of large language models (LLMs) typically depends on high-quality supervision data, such as safe demonstrations or preference pairs. However, in real-world deployment, emerging safety requirements are often specified as natural-language policies, while corresponding supervision data may be costly, delayed, or unavailable. This creates a mismatch between rapidly evolving safety policies and conventional data-driven alignment methods. To address this, we propose PolicyAlign, a simple yet effective framework for directly aligning LLMs with safety policies. Given a safety policy, PolicyAlign first synthesizes policy-violating instructions and then performs on-policy self-distillation to internalize policy-guided behavior. To improve training stability and data efficiency, we further introduce Policy-Sensitive Filtering, which selects instructions where the policy induces the largest behavioral shift. Experiments across multiple models show that PolicyAlign consistently improves safety while maintaining low over-refusal and preserving general capabilities. PolicyAlign also generalizes to medical, legal, and financial safety scenarios, highlighting its potential as a scalable and maintainable approach to policy-based LLM safety alignment. The code is released at this https URL. 

---
# Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making 

**Authors**: Guangfeng Cai, Kaibing Yang, Shuo He, Yu Li, Shengtian Yang, Jiaqi Lv, Lei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.25421)  

**Abstract**: Recent studies on world modeling for Large Language Model (LLM) agents typically formulate the learning objective as next-observation prediction. However, this objective ties supervision to what a transition happens to reveal, which may omit the dynamics most relevant to the agent's current decision. To bridge this gap, we propose Agent-Authored World Modeling (AAWM), a training procedure that constructs supervision from the policy's own decision needs. Specifically, at each state, the agent identifies what it needs to understand about the environment before acting. These needs drive the retrieval of relevant transition evidence across trajectories, which is then synthesized into training targets that capture decision-oriented dynamics instead of reconstructing the next observation. This aligns the training objective with the dynamics the policy needs before acting, not with the contents of the next observation. Experimental results validate the effectiveness of AAWM across multiple environments and training settings. These results show that decision-aware world-model targets provide a more effective learning signal than next-observation prediction. 

---
# Introducing corpora Hlava Cor and Hlava AD: Human Label Variation in Coreference and Discourse Relations 

**Authors**: Anna Nedoluzhko, Šárka Zikánová, Jiří Mírovský, Milan Straka, Eva Hajičová  

**Link**: [PDF](https://arxiv.org/pdf/2606.25383)  

**Abstract**: As previous research on annotator disagreement in discourse phenomena has shown, understanding text coherence varies considerably from one individual to another. To explore this phenomenon, we created two corpora with multiple annotations of Czech texts, accompanied by annotators' explanations of their choices. The first corpus consists of 1,024 contexts annotated in parallel by three annotators. It captures differences in the identification of coreference across various text types and grammatical-semantic categories, including pronouns, full noun phrases, and anaphoric adverbials. The second corpus comprises 512 contexts, annotated in parallel by five annotators, and focuses on identifying discourse relations in attributive and non-attributive constructions. Both corpora achieve a comparable inter-annotator agreement of approximately 60-65%. For coreference annotation, agreement tends to be lower in cases where automatic coreference resolution models disagree, suggesting that when the models disagree, the examples tend to be more difficult or ambiguous for human annotators to interpret. The annotators' comments, both for coreference and discourse relations, further reveal differences in interpretation, varying levels of confidence in text understanding, and individual reading strategies. 

---
# A Survey of Toxicity Detection and Mitigation Strategies for Multilingual Language Models 

**Authors**: Soham Dan, Himanshu Beniwal, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25380)  

**Abstract**: Large language models (LLMs) are increasingly deployed across languages, but their safety behavior remains uneven across linguistic and cultural contexts. This survey synthesizes work on toxicity detection and detoxification for multilingual LLMs. We first catalogue threat models that exploit language choice, translation pivots, code-switching, orthographic variation, multi-turn interaction, and post-deployment fine-tuning to weaken safety alignment. We then organize task formulations (toxic-to-neutral rewriting, toxicity classification, and toxic-generation evaluation), multilingual detection approaches (cross-lingual encoders, translation pipelines, representation-level probes, and LLM-based detectors), and mitigation strategies spanning data filtering, supervised and preference-based tuning, decoding-time steering, representation editing, and multilingual guardrails. Across these areas, we identify persistent challenges: uneven language coverage, culturally contingent definitions of harm, fragmented evaluation protocols, and the risk that detoxification suppresses legitimate dialectal or identity-related expression. 

---
# Story Operators: Decomposing the Original $\to$ Sequel Transformation in Embedding Space 

**Authors**: W. Frederick Zimmerman  

**Link**: [PDF](https://arxiv.org/pdf/2606.25379)  

**Abstract**: I treat a book as a point in a sentence-embedding space and a literary transformation as an operation on points. Given an original novel and its sequel, I ask what it takes, geometrically, to turn the first into the second. Using all-mpnet-base-v2 paragraph embeddings drawn from a precomputed index of the PG19 corpus, I form the displacement $d=\bar{x}_{\rm seq}-\bar{x}_{\rm orig}$ and greedily decompose it along a content basis obtained by PCA over the two books' own paragraphs. Each component is an interpretable axis anchored by real passages at its poles. Across thirteen verified author pairs from Project Gutenberg, the decomposition reveals a small taxonomy of sequels: formulaic (a tiny, low-rank change: Doyle's Holmes collections, $\|d\|=0.12$), concentrated (one dominant axis: Alcott's Little Women $\to$ Little Men, 75% on a single move), and compositional (many small axes: Twain, Burroughs's Barsoom, Nesbit). For the canonical case, Tom Sawyer $\to$ Huckleberry Finn, the dominant recovered axis is structural -- the collapse of sheltering domesticity into a picaresque road -- rather than the famous surface themes of vernacular voice or slavery, which ride later, smaller axes; and the transformation routes through adventure-journey space rather than diluting toward generic realism. I corroborate the recovered geometry against Twain's documented authorial intent (his 1875--76 letters to Howells), which names the first-person picaresque move years in advance, and I quantify, with an explicit representation caveat, how much of the realized transformation his stated intentions span. All computations are reproducible from the released scripts and data. 

---
# Three Buddhist Vocabularies: Computational Stylometry of the English Pali Canon across Sutta, Vinaya, and Abhidhamma 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2606.25372)  

**Abstract**: We present a computational stylometric analysis of the Tipitaka across all three Pitakas in English translation, extending earlier work on the Sutta Pitaka alone. The corpus spans 134,831 segments from Bhikkhu Sujato's Sutta Pitaka (114,591 segments, CC0), Bhikkhu Brahmali's Vinaya Pitaka (7,923 segments, CC0 2026), I.B. Horner's 1938 Vinaya translation (2,826 segments), three English translations of the Abhidhammattha Sangaha compendium (2,077 segments), and cross-tradition Vinaya texts from the Dharmaguptaka and Mulasarvastivada schools. We compute Zipf rank-frequency distributions with OLS-fitted exponents, Moving Average TTR (MATTR-500), numeral-word density, and vocabulary overlap (Jaccard and Szymkiewicz-Simpson coefficients). Main findings: (1) all corpora show Zipf-consistent distributions (R2 > 0.989); the Vinaya is closest to ideal Zipf slope -1 and the Sangaha corpus deviates most, with 'consciousness' displacing grammatical particles at rank 8; (2) MATTR-500 shows the Sutta and Vinaya Theravada are nearly identical in lexical diversity (0.399 and 0.400), while the Sangaha corpus is genuinely more diverse (0.560), confirmed by size-controlled subsampling; (3) the Sangaha corpus has the highest numeral-word density (3.26%), consistent with its systematic enumeration of mental and material categories; (4) the Mulasarvastivada Vinaya shares 20.0% vocabulary (Jaccard) and 49.1% (overlap coefficient) with the Theravada Vinaya, reflecting shared legal heritage across two millennia; (5) two English translations of the same Vinaya source text share only 24.2% of their vocabulary across 88 years, with 'musing' versus 'absorption' for jhana and 'defeat' versus 'expulsion' for parajika as the most diagnostic shifts. All results are point estimates; no significance testing is conducted. Code and data are released as open-source extensions to the Darshana Graph corpus (arXiv:2606.18222). 

---
# Neural Machine Translation for Low-Resource Tangkhul--English 

**Authors**: Chormi Zimik Vashai, Agniva Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2606.25365)  

**Abstract**: We present a study on low-resource machine translation for the Tangkhul-English (nmf-en) language pair. Tangkhul is a severely under-resourced Tibeto-Burman language spoken primarily in Manipur, India, with virtually no prior natural language processing infrastructure. We describe two systems: (1) a primary system based on ByT5-large fine-tuned on 38,336 Tangkhul-English parallel sentence pairs, and (2) a contrastive system based on mT5-small fine-tuned on the same corpus. Our primary ByT5-large system achieves a corpus BLEU score of 39.97, chrF++ of 58.07, BERTScore F1 of 0.8104, and COMET (wmt22-comet-da) of 0.7302 on a held-out test set of 3,856 sentences. We further discuss the orthographic challenges specific to Tangkhul's Latin-script diacritics, the domain bias of our training corpus (which comprises biblical text, stories, and conversational data), and avenues for future improvement through data diversification and domain adaptation. 

---
# Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents 

**Authors**: Yuxin Wang, Paul Thomas, Zhiwei Yu, Yuan Gao, Saeed Hassanpour, Soroush Vosoughi, Robert Sim, Nick Craswell  

**Link**: [PDF](https://arxiv.org/pdf/2606.25361)  

**Abstract**: Prior research on memory mechanism in RAG-based conversational system has emphasized how memory is stored and retrieved. However, far less is known about how memories with different functional roles influence response quality. Specifically, how they shape an agent's responses under varying conversational contexts and whether they lead to substantively different response behaviors. Existing evaluations in conversational system are also largely reference-based, insufficiently capturing the nuances in responses that may address users' preferences differently. In this work, we probe the impact of different memory types in shaping agents' responses. We present a fine-grained taxonomy of conversational memory, classify retrieved memories into different role types, and design a user-centric evaluation framework that simulates user perspectives. Through comparative experiments on long-term datasets and frontier LLMs, our analysis reveal many differentiated effects of memories: e.g., clarifying memory improves responses' factual accuracy and constraint awareness, making them more correct and personalized; irrelevant memory reduces topic relevance and degrades constraint awareness. Despite the power of frontier LLMs, these findings shed light on how different memory types can be leveraged to produce more personalized responses and inspire further research in this direction. 

---
# Efficient and Trainable Language Model Test-Time Scaling via Local Branch Routing 

**Authors**: Yutong Yin, Mingyu Jin, Jin Pan, Changyi Yang, Zijie Xia, Dhruv Pai, Shuming Hu, Zhen Zhang, Chenyang Zhao, Jinman Zhao, Wujiang Xu, Raymond Li, Xin Eric Wang, Julian McAuley, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25354)  

**Abstract**: Test-time scaling improves language-model reasoning, but existing approaches often face a difficult trade-off: long chain-of-thought sampling remains single-threaded, while sentence- or solution-level search can be computationally expensive and hard to train end-to-end. We introduce Local Branch Routing (LBR), a token-level test-time scaling framework that expands a small local lookahead tree, forwards all sampled branches through the language model, and uses a lightweight router to select the depth-1 subtree to commit. By routing over the hidden states of candidate local futures, LBR allows each token decision to use evidence beyond the root next-token distribution while avoiding full solution-level search. The resulting prune-shift-grow decoding process preserves discrete branch identities and defines a tractable tree-trajectory likelihood: newly grown nodes are counted when first sampled, and router decisions are assigned explicit probabilities. This enables end-to-end reinforcement learning with verifiable rewards, jointly optimizing the base model and router under the same likelihood-ratio principle as discrete-token RLVR. On synthetic hierarchical-planning tasks, LBR shows that post-candidate hidden states provide useful routing evidence. On mathematical reasoning benchmarks, LBR improves both Pass@1 and Pass@32 over discrete chain-of-thought, vanilla discrete-token RLVR, and RL-compatible soft-token branching baselines. These results suggest that lightweight local branching offers an efficient, trainable, and discrete form of language-model test-time scaling. 

---
# Hybrid-IR: Dual-Path Hybrid Retrieval with Iterative Reasoning for Complex Medical Question Answering 

**Authors**: Sheng Wan, Jiahui Zhang, Zicheng Zhao, Shougang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.25338)  

**Abstract**: Large language models (LLMs) have shown promising performance across a wide range of biomedical applications, including medical question answering (QA), yet they remain prone to hallucinations and outdated knowledge. Although retrieval-augmented generation (RAG) can alleviate this issue by incorporating external documents, there still exist two fundamental limitations. First, medical knowledge is often fragmented across documents, while most RAG methods rely on a single retrieval path, which makes it challenging to jointly preserve fine-grained semantic information and structured global associations. Second, static retrieval strategies are typically insufficient to support deep reasoning that is important in complex medical QA. In this paper, we present a dual-path retrieval framework with an iterative retrieval-reasoning mechanism termed "Hybrid-IR" for complex medical QA. The proposed Hybrid-IR integrates graph-based retrieval for exploration of structured knowledge and dense retrieval for fine-grained semantic matching. Moreover, the reasoning trajectory can be progressively refined through an iterative retrieve-reason loop. Experiments on three widely used medical QA benchmarks demonstrate the effectiveness of our Hybrid-IR. 

---
# Improved Large Language Diffusion Models 

**Authors**: Shen Nie, Qiyang Min, Shaoxuan Xu, Zihao Huang, Yuxuan Song, Yong Shan, Yankai Lin, Wayne Xin Zhao, Chongxuan Li, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25331)  

**Abstract**: Modern large language models are predominantly trained with autoregressive factorization and causal attention. We present \emph{iLLaDA}, an 8B masked diffusion language model trained from scratch with fully bidirectional attention. iLLaDA keeps the masked diffusion objective throughout pre-training and supervised fine-tuning (SFT), scaling pre-training to 12T tokens and fine-tuning on a 25B-token instruction corpus for 12 epochs. We further use variable-length generation for efficiency and introduce confidence-based scoring for multiple-choice evaluation. Compared with LLaDA, iLLaDA improves broadly across general, mathematical, and code benchmarks; for example, iLLaDA-Base improves by 21.6 points on BBH and 14.9 points on ARC-Challenge, while iLLaDA-Instruct improves by 14.5 points on MATH and 16.5 points on HumanEval. Despite its non-autoregressive training, iLLaDA also remains competitive with Qwen2.5 7B on several benchmarks. These results show that fully bidirectional diffusion training from scratch is a competitive path toward strong language models. Model weights and codes: this https URL. 

---
# Automatic Generation of Highlights for Academic Paper Via Prompt-based Learning 

**Authors**: Yi Xiang, Chengzhi Zhang, Heng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25253)  

**Abstract**: Highlights provide a concise summary of the main contributions of an academic paper and help readers quickly understand its focus. However, many journals do not provide highlights, which limits their use in literature retrieval, text mining, and bibliometric analysis. Existing studies have explored supervised learning methods for automatic highlight extraction, but these methods usually require large amounts of labeled training data. This study investigates prompt-based learning for automatic highlight generation. We design task-specific prompt templates and combine them with paper abstracts as model inputs. Several language models are evaluated, including locally deployed pre-trained models such as GPT-2 and T5, as well as ChatGPT accessed through an API. Experiments on three datasets show that ChatGPT with prompt templates achieves performance comparable to previous supervised methods without using task-specific training samples. When a small number of examples are added to the prompts, the model significantly outperforms state-of-the-art methods on two datasets. We further analyze how prompt design affects generation quality and find that, although ChatGPT has strong language modeling ability, its performance on this task is highly sensitive to the information provided in the prompt. Case studies also show that the generated highlights are generally coherent, informative, and close to author-written highlights. This study is among the first to apply prompt-based learning to academic highlight generation. The proposed method does not rely on domain-specific training corpora and can generate highlights for papers that lack such information, thereby supporting downstream text mining and bibliometric research. 

---
# Towards Structuring an Arabic-English Machine-Readable Dictionary Using Parsing Expression Grammars 

**Authors**: Diaa Mohamed Fayed, Aly Aly Fahmy, Mohsen Abdelrazek Rashwan, Wafaa Kamel Fayed  

**Link**: [PDF](https://arxiv.org/pdf/2606.25231)  

**Abstract**: Dictionaries are rich sources of lexical information about words that is required for many applications of natural language processing and human language technology. However, publishers prepare printed dictionaries for human usage not for machine processing. This paper presented a method to structure partly a machine-readable version of the Arabic-English Al-Mawrid dictionary. The method converted the entries of Al-Mawrid from a stream of words and punctuation marks into hierarchical structures. The hierarchical structure expresses the components of each dictionary entry in explicit format. A dictionary entry is composed of subentries and each subentry consists of defining phrases, domain labels, cross-references, and translation equivalences. We designed the proposed method as cascaded steps where parsing is the main step. We implemented the parser using the parsing expression grammars formalism. In conclusion, although Arabic dictionaries do not have microstructure standardization, this study demonstrated that it is possible to structure them automatically or semi-automatically with plausible accuracy after inducing their microstructure. 

---
# What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics 

**Authors**: Sofiia Nikolenko, Michele Papucci, Mina Rezaei, Shireen Kudukkil Manchingal  

**Link**: [PDF](https://arxiv.org/pdf/2606.25182)  

**Abstract**: Jailbreak attacks reveal a persistent weakness in aligned Large Language Models: carefully crafted prompts can elicit policy-violating responses despite safety training. While most defenses operate at the prompt or output level, it remains unclear how harmful intent is encoded within the model's internal representations. We investigate this question by analyzing token-level predictive entropy trajectories across layers of a frozen LLM using the logit lens. We find that static aggregate statistics of prompt-level entropy (e.g., mean, variance) carry little discriminative signal, whereas features capturing how entropy evolves across token positions, such as monotonic rank-based trend scores, are substantially more informative. Importantly, this signal is not uniform across model depth: it is concentrated in intermediate layers and degrades at the final layer, indicating that jailbreak-relevant structure is most pronounced in mid-network representations rather than at the output head. Across multiple models (Llama, Qwen, Gemma) and adversarial benchmarks, these entropy dynamics provide architecture-consistent separation without additional training. Together, our findings show that jailbreak behavior is reflected in structured intermediate uncertainty dynamics, clarifying both which entropy-derived features encode harmful intent and where in the network that signal is most pronounced. 

---
# Hitting a Moving Target: Test-Time Adaptation for AI Text Detection under Continual Distribution Shift 

**Authors**: Kevin Ren, Manish Raghavan, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2606.25152)  

**Abstract**: Deployed approaches for AI text detection often rely on training-time access to labeled datasets of both human-written and AI-generated text. This approach is vulnerable to three types of distribution shifts that occur continually post-deployment, and for which labeled data is often unavailable: adversarial humanization, new LLMs being released, and temporal drift in human writing. Simultaneously, existing approaches do not leverage a key signal of LLM usage: inference-time homogeneity. We propose a test-time adaptation (TTA) approach, using semi-supervised learning, that adapts to distribution shifts by leveraging homogeneity among unlabeled samples observed at inference time. Empirically, we find that state-of-the-art supervised detectors systematically fail when they encounter distribution shifts in AI-generated and human writing, both adversarial and natural, while test-time adaptation with semi-supervised learning is largely robust; e.g., the commercial model Pangram detects just 24.1% of our adversarial AI-generated text, compared to 90.5% for our test-time approach. We establish that test-time adaptation is a promising framework for AI text detection in the wild. We publicly release our code (which includes code for model training, evaluation, and plots) at this https URL. 

---
# The cognitive, affective, and behavioral expression of self-stigma among people who use drugs in online substance use communities 

**Authors**: Layla Bouzoubaa, Hyung Wook Choi, Milan Varghese, Valerie Earnshaw, Rezvaneh Rezapour  

**Link**: [PDF](https://arxiv.org/pdf/2606.25143)  

**Abstract**: Objectives: To develop a codebook for self-stigma across cognitive, affective, and behavioral domains, and to estimate the prevalence, co-occurrence, and temporal patterns of these indicators in Reddit posts by people who use drugs. Methods: We developed a ten-indicator codebook through consensus-based abductive coding spanning cognitive (self-labeling, pessimism/self-defeatism, deservingness/worthlessness), affective (shame, guilt/self-blame, despair/hopelessness), and behavioral (concealment, anticipated rejection, desire to quit, ambivalence) domains; two coders reached substantial agreement (Cohen's k = 0.72). We then scaled classification with a large language model validated against expert coding (k = 0.73, F1 = 0.80), analyzing 72,115 thread-initiating posts from 1,660 English-language users (2006-2025). Results: 3,838 posts (5.3%) from 1,228 users (74.0%) contained self-stigma; all ten indicators discriminated self-stigma posts (RR 3.6 to 86.2), led by self-labeling (56.0%) and despair/hopelessness (48.5%). Self-stigma was integrated: core and behavioral indicators were strongly associated at the user level (OR = 4.65, 95% CI 3.12-6.94, p < 0.001), and 87.0% of posts with behavioral indicators also contained a core indicator. Contrary to progressive models, behavioral indicators emerged earlier than core ones (desire to quit at median position 0.08 vs. shame at 0.38). Nine of ten indicators were stable across posting trajectories; only pessimism increased (OR = 1.62, 95% CI 1.25-2.10). Conclusion: Among people who use drugs online, self-stigma is an integrated phenomenon in which behavioral indicators rarely appear without internalized ones and often precede them. Most expressions remain stable over time, but pessimism about change deepens, marking a target for early digital intervention and showing that progressive stage models do not map directly onto textual disclosure. 

---
# Dream at SemEval-2026 Task 13: SALSA for Single-Pass Machine-Generated Code Detection 

**Authors**: Ruslan Berdichevsky, Shai Nahum-Gefen, Elad Ben-Zaken  

**Link**: [PDF](https://arxiv.org/pdf/2606.25102)  

**Abstract**: Large language models have transformed code generation, raising concerns around authorship, assessment integrity, and software trust. SemEval-2026 Task 13 Subtask A operationalizes detection as binary classification over code snippets, with a particular emphasis on out-of-distribution (OOD) generalization across unseen programming languages and application domains. We propose a SALSA-style formulation, Single-pass Autoregressive LLM Structured Classification, that maps each class to a dedicated output token and trains the model to emit a single-token label in a structured response. Rather than engineering hand-crafted features or decision rules, this formulation delegates the authorship decision to the model. To improve OOD robustness, we combine balanced sampling across languages with parameter-efficient fine-tuning and conservative training (low learning rate, single epoch) to avoid overfitting to the training domain. Our best system achieves OOD $F_1 = 0.789$ on the official leaderboard, substantially outperforming the CodeBERT baseline ($F_1 = 0.305$). 

---
# LLM-Based Scientific Peer Review: Methods, Benchmarks, and Reliability Challenges 

**Authors**: Thi Huyen Nguyen, Zahra Ahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25057)  

**Abstract**: The rapid growth of scientific submissions has pushed traditional peer review toward its scalability limits, motivating the exploration of large language models (LLMs) as intelligent automated evaluation assistants. Although recent studies show that LLMs can generate fluent critiques and approximate reviewer scores, their reliability, robustness, and security as decision-support systems remain insufficiently understood. This survey offers a systems-level analysis of LLM-based scientific peer review, focusing on two core evaluative functions: critique generation and score prediction. We present a structured taxonomy of modeling approaches (including prompt-based, supervised, retrieval-augmented, and alignment-optimized approaches), and synthesize empirical findings across existing benchmarks. We analyze dataset constraints, evaluation shortcomings, and domain concentration biases that limit current assessment practices. Beyond performance metrics, we identify emerging robustness risks, including prompt injection, data poisoning, retrieval vulnerabilities, and reward hacking, which expose automated review pipelines to strategic manipulation. From a data mining perspective, we outline key open challenges in modeling subjective disagreement and cross-domain generalization. By reframing automated peer review as a high-stakes, multi-objective decision problem, this survey provides a roadmap for developing robust, transparent, and trustworthy AI-assisted scientific evaluation systems. 

---
# LLM Performance on a Real, Double-Marked GCSE Benchmark 

**Authors**: Malachy Fox, Kavi Samra, Paul Jung  

**Link**: [PDF](https://arxiv.org/pdf/2606.24973)  

**Abstract**: We introduce a dataset of 32,534 double-marked real student responses to GCSE mock exams (GCSEs are the UK's national exams, taken at age ~16), spanning 328 questions across five subjects and including handwritten work. We test whether off-the-shelf large language models agree with examiners as closely as the two examiners agree with each other. We find that models overwhelmingly agree well with the examiner consensus across subjects, with the top performing models agreeing more closely with examiners than examiners agree with each other. Models achieve high scores for subjective tasks like English essay marking, as well as handling complex and messy handwritten Maths paper scripts. Agreement is uniform near the examiner line, and not massively discriminated by model size, providing cost-effective automated marking solutions. 

---
# Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding 

**Authors**: WenHung Lee, Jian-Jia Chen, Xiaolin Lin, Pei-Shuo Wang, Chi-Chih Chang, Chun-Che Yang, Ning-Chi Huang, Grace Li Zhang, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24957)  

**Abstract**: While speculative decoding improves inference throughput for multi-batch long-context Large Language Models (LLMs), its efficiency is often limited by a verification bottleneck where Key-Value (KV) cache loading dominates latency. Existing compression methods fail in this regime: static eviction incurs accuracy loss due to saliency shift, while dynamic selection introduces prohibitive computational overhead during the verification path. We propose Dustin, a sparse verification framework designed for long-context speculative decoding. Dustin integrates lookahead signals from the draft model with historical attention from the target model to identify critical tokens with high fidelity across multi-step verification windows. To reduce recomputation latency, this approach further employs a sparse estimation scheme that restricts importance scoring to a minimal subset of attention heads. Evaluations on PG-19 and LongBench with Qwen2.5-72B demonstrate that Dustin achieves a 27.85x speedup in self-attention and a 9.17x end-to-end decoding speedup at a 32k sequence length, all with negligible accuracy degradation. 

---
# Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models 

**Authors**: Cosimo Galeone, Anna Ettorre, Minsu Park, Giuseppe Ettorre, Daniele Ligorio  

**Link**: [PDF](https://arxiv.org/pdf/2606.24952)  

**Abstract**: A central aspiration of mechanistic interpretability is controllability: if we know where a behavior is represented in a model's activations, we should be able to modify it. This rests on a hidden premise -- that the direction which detects a behavior and the direction which controls it are the same, or close. We test this geometrically: what is the angle between the direction that best detects a behavior and the one that best causes it? If detection implies control the cosine is near 1; otherwise it quantifies a detection-intervention gap. On Gemma 2-2B-it, output format (clean JSON vs markdown fencing) collapses both roles onto one axis. Hallucination does not: the model detects fake entities with perfect linear separability (AUC = 1.000 from layer 5), yet that direction sits at cos = 0.12 (about 83 degrees) from the direction producing a refusal -- a small, reproducible alignment, far from the cos = 1 that "detection is control" would require. A detector built from activations, with no chosen tokens, likewise fails to align (cos = -0.06). The gap generalizes: across four models from three families and two scales (1B-9B), cos stays in [0.12, 0.20], identical before and after instruction tuning (0.1197 vs 0.1200), placing its origin in pretraining. A 15-degree rotation toward the refusal direction partially bridges it -- 73% and 60% refusal on two held-out fake-entity categories at 1.8% false positives. We then ask whether this cosine predicts steerability, and it does not: detection is a high-dimensional class, not a single direction, and what separates the steerable case is functional, not readable from a static angle. The cosine is a weight-computable signature of the dissociation between knowing and steering, not a predictor of it. 

---
# Error-Aware TF-IDF Retrieval-Augmented Generation for ASR Error Correction 

**Authors**: Mohammad Aref Jafari-Raddani  

**Link**: [PDF](https://arxiv.org/pdf/2606.24915)  

**Abstract**: End-to-end automatic speech recognition systems frequently hallucinate rare entities and domain-specific terms, especially in low-resource languages. While retrieval-augmented generation frameworks can mitigate these errors using large language models, current architectures face significant challenges. They either rely on standard sparse retrieval that ignores phonetic misrecognitions or utilize heavyweight cross-modal embeddings that introduce high latency. This letter proposes a highly efficient, purely lexical error-aware framework designed to explicitly resolve phonetic and loop hallucinations. Our approach integrates a symmetric text normalization module with a novel error-aware term frequency-inverse document frequency algorithm. By constructing a sparse diagonal penalty matrix based on historical errors, the retriever mathematically prioritizes corrective documents containing specific high-risk misrecognitions. Evaluated on the Persian subset of the FLEURS dataset, our method increased the error-aware hit rate from 53.7% to 90.9%. In end-to-end evaluations, the integrated framework reduced the final word error rate from 23.06% to 18.83%, achieving significant accuracy gains with near-zero inference latency. 

---
# AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents 

**Authors**: Zheyuan Zhang, Zehao Wen, Alvin Zhang, Andrew Wang, Jianwen Xie, Daniel Khashabi, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24893)  

**Abstract**: For agents to learn continuously from interaction with the world at test time, they must be able to explore effectively, acquire new world knowledge and skills, retain relevant episodic experiences, and plan over long horizons. To evaluate these key abilities of test-time continual learning agents, we introduce AgentOdyssey, a novel evaluation framework that procedurally generates open-ended text games with rich entities, world dynamics, and long-horizon tasks. Critically, AgentOdyssey goes beyond the conventional machine learning assumption that learning does not occur at test time by placing agents in a continuous, long-horizon setting that interleaves learning and inference throughout deployment. We further propose a multifaceted evaluation methodology that measures not only game progress but also offers diagnostic tests on world knowledge acquisition, episodic memory, object and action exploration, action diversity, and model cost. We evaluate diverse agent paradigms in the generated games. Our experimental results reveal critical limits in agents' key abilities, as well as factors that influence their meaningful horizon. Although performance scales with stronger base models, even the top agent remains far below human performance, leaving substantial headroom for improvement. Among agent mechanisms, we find that short-term memory benefits multiple agent paradigms and is an important component of agent test-time training. 

---
# Small edits, large models: How Wikipedia advocacy shapes LLM values 

**Authors**: Jasmine Brazilek, Maria Navas, Alexa Gnauck  

**Link**: [PDF](https://arxiv.org/pdf/2606.24890)  

**Abstract**: Can a small group of volunteers shape how AI systems discuss animal welfare, just by editing Wikipedia? We show that they can. Wikipedia appears in nearly every major language model training dataset and is weighted more heavily than web-crawled text. The Pro-Animal Wikipedians (PAW), a group of advocates who add sourced animal welfare content to relevant articles, have made 125 edits across 115 pages. Using gradient-based data attribution (Bergson; MAGIC), we traced how these edits influence language model behavior. TrackStar retrieval attribution on Llama 3.1 8B found that PAW-edited sections made up 68 percent of the highest-attributed documents for animal welfare queries (p < 0.0001) but only 52 percent for unrelated queries about the same companies (p = 0.53): the model links PAW content specifically to animal welfare topics, not to the entities in general. MAGIC counterfactual influence estimation on Llama-3.2-1B, run across five random training-order seeds, gave the same picture even more sharply: in every seed, the top-10 most influential documents on animal welfare queries were all PAW edits (10 of 10, 5 of 5 seeds), while on general queries the same top-10 sat at chance (4 to 6 of 10). Mean PAW influence exceeded mean control influence on animal welfare queries with p < 0.0001 in every seed, an effect 6 to 30 times larger than on general queries. Leave-subset-out validation gave Spearman rho = 1.00 for all 10 runs. When we fine-tuned separate models on PAW content versus control content, each model performed better specifically on the type of text it was trained on: the PAW-trained model cut perplexity on animal welfare text from 12.4 to 8.4, while the control-trained model cut perplexity on control text from 16.1 to 11.4. A small, coordinated Wikipedia editing campaign therefore measurably shapes how language models handle the topics those edits address. 

---
# Graph-Based Phonetic Error Correction of Noisy ASR 

**Authors**: Pratik Rakesh Singh, Mohammadi Zaki, Aneesh Mukkamala, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2606.24889)  

**Abstract**: Automatic speech recognition (ASR) systems, despite low overall word error rates, produce residual lexical errors that disproportionately affect semantically critical tokens such as named entities, negations, and sentiment-bearing words. These errors are often structured, arising from phonetic similarity rather than random noise, making naive token-level correction insufficient. We propose a structured ASR correction framework, that we call G-SPIN, that combines phonetic graph modeling with contextual language understanding. A graph neural network (GNN) first constructs acoustically plausible candidate neighborhoods for flagged tokens, explicitly restricting the correction search space to phonetic alternatives. A masked language model (MLM) then provides local contextual scoring, and an instruction-tuned large language model (LLM) performs final context-aware re-ranking over this compact candidate set. By decoupling structured phonetic reasoning from contextual semantic selection, our method avoids unconstrained generation while improving correction accuracy. The framework is lightweight, modular, and operates entirely at inference time. 

---
# Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining 

**Authors**: Juliana Li, Diya Sreedhar  

**Link**: [PDF](https://arxiv.org/pdf/2606.26050)  

**Abstract**: Midway through an ordinary pretraining run, a small language model learns the pronoun-gender rule: cued with a girl's name ("Sue cried because"), it resolves the next pronoun to she, generalizing to held-out probes (0.94 by step 925). By step 3,500 the same model scores near zero on the same probes, although the rule's evidence is still in the training data. We call this within-run reversal natural ungrokking: the corpus decides, with no trace in the loss curve, which learned rules a model keeps.
Which rules survive is predictable from one corpus statistic: how often the training stream shows the rule winning. Across un-intervened runs (two corpora, three budgets, three seeds), support frequency decides a rule's fate; the data-to-parameter ratio only modulates how deeply a doomed rule falls. The same emerge-then-collapse dynamics appear in public Pythia checkpoints, collapse depth ordered by model scale as predicted. The forgetting is a displacement: a competing surface pattern out-competes the rule, and the log-probability margin between them crosses zero within 100 training steps of the behavioral collapse.
Control over this fate is asymmetric: the same edit that destroys a rule on demand cannot restore it. Flipping support to counter-evidence in place kills the rule with monotone dose-response in two unrelated rules; but injecting support back, even to 450 times the level that naturally sustains it, buys no recovery. Every confirmatory threshold and prediction was pre-registered before the data it governed was read. 

---
# How Robust is OCR-Reasoning? Evaluating OCR-Reasoning Robustness of Vision-Language Models under Visual Perturbations 

**Authors**: Yuxing Cheng, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26041)  

**Abstract**: Vision-language models (VLMs) have achieved strong performance on OCR-based benchmarks and increasingly focused on text-rich understanding, but their robustness under controlled visual degradation remains insufficiently understood. This gap is critical for OCR reasoning, where visual corruption can induce OCR errors and structural distortions, thereby introducing uncertainty into the reasoning task. To systematically study this problem, we introduce OCR-Robust, a benchmark designed for evaluating OCR reasoning robustness under visual perturbations. It contains 812 samples across two complementary subsets: OCR1.0, covering documents, scene text, receipts, handwriting, and mathematical content, and OCR2.0, focusing on charts, geometry diagrams, and tables. To enable efficient yet informative evaluation, we conduct a pilot study over 18 candidate perturbations and select 5 representative types at 3 severity levels each based on their impact and cross-model discriminability. We evaluate robustness using clean accuracy, Relative Corruption Retention (RCR), Worst-Case Retention (WCR), and a composite Corruption Robustness Index (CRI), and benchmark 18 models spanning proprietary systems, open-source VLMs, and OCR+LLM pipelines. Our results show that higher clean accuracy does not necessarily imply stronger robustness, and that models can suffer pronounced degradation in the worst case on OCR tasks that are sensitive to structure, and charts and tables are substantially more fragile than document-like inputs under perturbation. 

---
# Autodata: An agentic data scientist to create high quality synthetic data 

**Authors**: Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Yixin Nie, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2606.25996)  

**Abstract**: We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data. 

---
# How Large Language Models Source Brand Reputation Across Languages and Markets 

**Authors**: Dmitrij Zatuchin  

**Link**: [PDF](https://arxiv.org/pdf/2606.25787)  

**Abstract**: When a large language model (LLM) answers a question about a company, it grounds the answer in retrieved web sources, and those sources decide what the model says. Most analysis of AI brand visibility looks at the answer text. This study looks one step earlier, at the citations. We merge three this http URL datasets covering 128 brands across 12 home markets and 13 languages, and analyse 167,551 URL-grounded citations (189,974 total attribution rows). We classify each citation by domain and source type and measure where AI gets its brand information, by language and by market. Four patterns hold. First, AI grounds brand answers overwhelmingly in third-party sources: 85.7% of citations point to sites the brand does not own, against 14.3% owned. Second, the source base is concentrated and long-tailed: 80% of citations come from about 18% of domains, fitting a Zipf law (alpha = 0.86, R^2 = 0.983). Third, one reference site dominates almost everywhere: Wikipedia is the most-cited domain in 11 of 12 languages, the exception being Lithuanian, where the business daily this http URL edges it (4.38%). Fourth, the source mix is market-specific at the margin: for 46 Polish national brands the most-cited domain is YouTube, and four HR and careers portals supply 637 citations against 297 for Polish Wikipedia, about twice as many. 

---
# Space-Efficient Language Generation in the Limit 

**Authors**: Nicolas Flammarion, Chirag Pabbaraju, Hristo Papazov, Miltiadis Stouras, Ola Svensson  

**Link**: [PDF](https://arxiv.org/pdf/2606.25777)  

**Abstract**: We initiate a resource-aware theory of \textit{language generation in the limit} under the minimal constraint of space efficiency. In our framework, a learner observes an adversarial positive stream from a target language $K$ and must eventually output a hallucination-free hypothesis language $L \subseteq K$ while omitting at most $\Delta$ strings of $K$. We focus on $\mathcal{C}_{s,k}$, the collection of languages recognized by DFAs with at most $s$ states over an alphabet of size $k$, as the natural hypothesis class for memory-bounded learners. In the exponential-space regime, we prove that a learner can exactly identify the target $K$. Under a stricter memory budget, we characterize the strongest possible generation guarantees. In particular, we present a streaming algorithm using $\mathrm{poly}(s,k)$ space that converges to a hypothesis with generation gap $\Delta = O(k^{2s-2})$. Moreover, the learned hypothesis captures every string in $K$ of length at least $2s-1$. We complement this result with a near-matching lower bound through a reduction from a standard communication complexity problem. Specifically, achieving generation gap $\Delta \le k^{(1-\varepsilon)s}$ requires $k^{\Omega(\varepsilon s)}$ memory. Together, these results reveal a sharp transition between polynomial-space generation and exponential-space exact identification. 

---
# Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets 

**Authors**: Divake Kumar, Sina Tayebati, Devashri Naik, Amanda Sofie Rios, Nilesh Ahuja, Omesh Tickoo, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25760)  

**Abstract**: Computer-use agents turn vision-language model (VLM) predictions into executable GUI clicks, so reliable uncertainty estimates are essential for rejection, calibration, miss-severity ranking, and spatial safety regions. Yet evidence on post-hoc uncertainty quantification (UQ) for these agents is fragmented across isolated model and dataset pairs, leaving it unclear whether UQ rankings stay stable when the agent, benchmark, or observable interface changes. We present Argus, a cross-regime benchmark for post-hoc UQ in single-step executable GUI grounding: a 27-method open-weight matrix over 4 VLM agents and 4 datasets, plus an 8-method closed-source matrix across 3 frontier vendors where logits, hidden states, and attention maps are unavailable. Evaluated methods span logit-based scores, sampling and consistency measures, hidden-state and density estimators (Mahalanobis, SAPLMA), attention-based scores, P(True) and verbalised-confidence prompting, and split-conformal prediction. The main finding is selective transfer: UQ rankings are stable across datasets for a fixed model, but degrade across model classes and observable interfaces. Hidden-state and density methods are the most stable open-weight family, while CoCoA-1MCA, Focus, sampling-based scores, and verbalised self-assessment win in specific regimes. Within-model ranking transfer is strong (Spearman rho up to 0.969), but cross-tier transfer to closed-source vendors averages only +0.08, so closed-source UQ should be reranked on the target rather than extrapolated. Conformal click regions show score-level discrimination is not enough for deployment: locally weighted disks shrink radii by 40-60% when the plug-in UQ is calibrated, but coverage degrades under calibration-test or interface mismatch. We release per-item records, calibration/test splits, UQ scores, and analysis scripts for regime-aware UQ selection in GUI agents. 

---
# RAS: Measuring LLM Safety Through Refusal Alignment 

**Authors**: Chang-Chieh Huang, Yan-Lun Chen, Chia-Mu Yu, Wei-Bin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.25750)  

**Abstract**: Safety evaluation of large language models (LLMs) is commonly performed by querying models with unsafe or jailbreak prompts and judging whether their outputs violate a safety policy. Although useful, output-level evaluation is expensive, sensitive to judge choice, and easily tied to fixed question banks. We propose **SafeVec**, a white-box evaluation procedure that measures safety from internal representations rather than generated answers. **SafeVec** first extracts layer-wise refusal directions from a safety-aligned reference model, then selects stable layer windows where safe and unsafe behaviors are separable, and finally scores a target model by measuring whether its hidden states align with these refusal directions under unsafe and jailbreak prompts. The resulting metric, **RAS** (**R**efusal **A**lignment **S**core), maps representation-level refusal alignment to a calibrated 0-100 safety score. Across `Llama`, `Gemma`, and `Qwen` model families, RAS separates aligned models from uncensored and abliterated variants, tracks output-level attack success rate, and is substantially faster than judge-based evaluation. These results suggest that refusal alignment provides a compact and efficient signal for white-box LLM safety evaluation. 

---
# Tracing Target Answers in Poisoned Retrieval Corpora via Token Influence Attribution 

**Authors**: Yan-Lun Chen, Pin-Yu Chen, Chia-Mu Yu, Ying-Dar Lin, Yu-Sung Wu, Wei-Bin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.25721)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are vulnerable to corpus poisoning attacks that manipulate model outputs through malicious retrieved documents. Existing detection methods typically rely on auxiliary classifiers or additional LLM-based verification, introducing substantial computational overhead. We present TRACE, a lightweight detection framework that identifies poisoning attacks by tracing answer-related tokens through token influence attribution. TRACE first discovers recurrent high-influence keywords across retrieved documents and then performs a secondary verification to confirm their influence on model predictions. Experiments on three QA benchmarks and six LLMs demonstrate strong detection performance while simultaneously uncovering attacker-specified target answers. 

---
# Security and Privacy in Retrieval-Augmented Generation: Architectures, Threats, Defenses, and Future Directions for Building Trustworthy Systems 

**Authors**: Balamurugan Palanisamy, G S S Chalapathi, Vikas Hassija, Rajkumar Buyya  

**Link**: [PDF](https://arxiv.org/pdf/2606.25533)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for enhancing large language models with external knowledge. By coupling retrieval mechanisms with generative models, RAG systems improve factual grounding and adaptability across domains. However, integrating retrieval pipelines introduces new security and privacy risks that extend beyond conventional language modeling threats. Sensitive information may be exposed through retrieval indices, query logs, context construction, or federated updates, while adversarial manipulation of knowledge bases can undermine trust in generated outputs. This survey provides a comprehensive examination of privacy and security challenges across RAG systems deployed in centralized, on-device (Micro-RAG), federated, and hybrid paradigms. We present a unified taxonomy of threat surfaces spanning the retrieval, context construction, and generation stages and systematically analyze attack classes, including membership inference, index inference, poisoning, gradient leakage, and collusion. We further review architectural, algorithmic, and cryptographic defenses, highlighting privacy-utility trade-offs and deployment considerations. Finally, we outline open research challenges toward building trustworthy, secure, and resilient RAG systems for real-world applications. 

---
# Evaluating LLMs on Real-World Software Performance Optimization 

**Authors**: Ezgi Sarıkayak, Wenchao Gu, Hesham Ghonim, Chunyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25530)  

**Abstract**: Software performance optimization is a notoriously complex and manual task. Despite the growing use of Large Language Models (LLMs) for code refinement, we still lack benchmarks that capture how optimization actually happens in real-world codebases. Existing frameworks often oversimplify the problem by focusing on isolated functions or a single performance metric, missing the critical trade-offs between execution time and memory footprint, the inherent noise of the measurement environment, and the variability introduced by different input data and execution conditions. We address this by introducing SWE-Pro, a repository-level benchmark derived from 102 expert-written optimizations from open-source projects. Unlike previous benchmarks, SWE-Pro pairs each task with parameterized tests to evaluate runtime, peak memory, and Time-Weighted Memory Usage (TWMU) across varying input data and execution conditions under noise-aware measurement conditions. Our evaluation shows that current LLMs struggle significantly: runtime gains are negligible, and memory optimizations are nearly non-existent. This stands in sharp contrast to expert implementations, which achieve an aggregate speedup of 15.5x and peak memory reduction of 171.3x over benchmark tasks. Expert-written improvements are observed in 91.2% of tasks for runtime and 65.7% for peak memory. Our findings expose a substantial gap between current LLM capabilities and the demands of expert-level engineering. 

---
# Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning 

**Authors**: Jaeyong Ko, Pilsung Kang, Yukyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.25524)  

**Abstract**: Large language models (LLMs) reach high accuracy in mathematical reasoning, but individual traces on the same problem diverge; some arrive at the correct answer while others fail. Prior work analyzes failure at the step, chunk, or sentence level, or at tokens where failure has already occurred. Neither identifies the precise token that triggers the shift toward failure. We introduce the cliff token, a token where the token-wise potential drops significantly under an adaptive threshold that scales with the local token-wise potential, based on a one-sided two-proportion z-test. Across seven models and three mathematical reasoning benchmarks (GSM1K, MATH500, AIME 2025), cliff tokens act as failure triggers; deleting the first cliff token and resampling recovers pass@64 to 1.0, while keeping it limits recovery to between 0.71 and 1.00. We further introduce a cliff taxonomy of deterministic, uncertain, and sampled-off cliffs, defined by greedy choice and token entropy. Each type has distinct probabilistic characteristics, and the taxonomy generalizes across model scales. Finally, we validate the taxonomy via single-token preference optimization at cliff positions (Cliff-DPO). Trained on GSM8K, Cliff-DPO improves accuracy across benchmarks by up to +6.6. Optimizing at uncertain and sampled-off cliffs improves reasoning, while deterministic cliffs do not. 

---
# Fully Differentiable Neural Forced Alignment via Soft Dynamic Programming 

**Authors**: Rotem Rousso, Eyal Cohen, Joseph Keshet  

**Link**: [PDF](https://arxiv.org/pdf/2606.25460)  

**Abstract**: Recent advances in sequence modeling have significantly improved ASR systems, bringing them close to human-level recognition accuracy and enhancing robustness across diverse acoustic conditions and languages. In contrast, Forced Alignment has not experienced comparable progress, and traditional HMM-GMM frameworks remain widely adopted and highly competitive.
To address this gap, we propose an end-to-end, fully differentiable neural architecture specifically designed for phoneme alignment. The model consists of an encoder that processes the input signal and a decoder that produces alignment decisions. The encoder is structured into two complementary branches: one dedicated to phoneme identity verification and the other to phoneme boundary detection. The decoder is implemented as a trainable module based on differentiable soft dynamic programming. The entire system is optimized end-to-end using a novel contrastive loss that encourages clear separation between steady-state phoneme regions and transition boundaries.
The proposed approach outperforms the current state of the art in phoneme alignment on hand-annotated English benchmarks, achieves strong word-level generalization results, and demonstrates generalization on unseen languages. 

---
# The Generalization Spectrum: A Chromatographic Approach to Evaluating Learning Algorithms 

**Authors**: Jinghan Zhang, Zerui Cheng, Shiqi Chen, Ge Zhang, Wenhao Huang, Jiashuo Liu, Junxian He, Tianle Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.25450)  

**Abstract**: Traditional evaluations measure a learning algorithm's final performance on an i.i.d. test set, reducing learning to a single aggregate score. This approach obscures a fundamental question: to what extent does learning from a specific example generalize to others? Such per-sample generalization, akin to learning by analogy in human cognition, captures how far the knowledge extracted from one example can transfer, yet remains invisible to standard benchmarks. We introduce the Generalization Spectrum, an evaluation framework designed to expose this hidden dimension. For each training example, we construct a controlled suite of test variants arranged by increasing transfer distance, from exact recall to implementation transfer across languages, context transfer under complete narrative re-framing, category-matched in-domain problems, and an unpaired baseline. By tracking performance across these distances, we reveal not just whether an algorithm learns, but how far that learning extends. We instantiate this framework on competitive programming, using a selection-and-synthesis pipeline seeded with recent problems to mitigate contamination. We first compare three canonical learning paradigms under matched memorization. RL converts memorization into near-transfer more efficiently than SFT-family baselines, while ICL exhibits strong but correspondence-dependent transfer. We then use the Spectrum to diagnose within-family variants. The resulting profiles show that local gains need not expand the generalization radius: abstractions and hints mainly lift local transfer, RFT preserves a stronger far-transfer tail than reference SFT, and self-distillation or hint-assisted RL can reduce far transfer even when local transfer or optimization improves. 

---
# The Interplay of Harness Design and Post-Training in LLM Agents 

**Authors**: Kyungmin Kim, Youngbin Choi, Seoyeon Lee, Suhyeon Jun, Dongwoo Kim, Sangdon Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.25447)  

**Abstract**: Tool-integrated LLM agents are often wrapped within a harness: the scaffolding that determines which tools are exposed, how they are described, and what auxiliary information accompanies each per-step observation. While agents are routinely post-trained, this scaffolding is typically treated as a fixed engineering detail, with design effort limited to the training-free regime. Moreover, existing post-training algorithms assume a static environment, even though tool environments and tasks often shift upon deployment. To address this gap, we extend $\texttt{ALFWorld}$ (i) to treat the harness as a controllable design dimension and (ii) to support evaluation under task and tool environment shifts. Building on this, we systematically analyze how the harness design influences post-training in both in-distribution and out-of-distribution (OOD) settings. We empirically show that harness-aware post-training not only improves in-distribution performance but also enables agents to robustly adapt to OOD settings. Under a harness with minimal design effort, post-training suffers a drastic performance drop under stronger tool environment shifts, further highlighting the importance of harness-aware post-training under such shifts. 

---
# Does Translation-Enhanced Speech Encoder Pre-training Affect Speech LLMs? 

**Authors**: Tomoya Mizumoto, Yusuke Fujita  

**Link**: [PDF](https://arxiv.org/pdf/2606.25444)  

**Abstract**: Connecting a pre-trained speech encoder to a Large Language Model (LLM) is the standard architecture for building Speech LLMs. However, a structural misalignment exists between the encoder and the LLM. Unlike encoders based on automatic speech recognition, which often produce representations in separate language-specific spaces, LLMs operate within a unified language-agnostic space. A mechanism is required to align the encoder's language-specific representations with the LLM's shared space. We argue that speech translation provides a principled way to achieve this. Unlike monolingual transcription, translation requires the model to bridge different languages and learn language-agnostic representations. We experimentally evaluate the impact of incorporating translation objectives into speech encoder pre-training. Our results demonstrate that translation-enhanced pre-training improves cross-modal integration and leads to superior performance across downstream Speech LLM tasks. 

---
# Evaluating Japanese Dialect Robustness Across Speech and Text-based Large Language Models 

**Authors**: Tomoya Mizumoto, Yusuke Fujita, Hao Shi, Lianbo Liu, Atsushi Kojima, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2606.25436)  

**Abstract**: Dialogue systems based on large language models (LLMs) have advanced significantly in recent years. However, dialectal variation remains a major challenge, particularly for systems that process spoken input. LLM-based speech language models (SLMs), which integrate LLMs with speech processing components, show promise for spoken language tasks, yet their ability to comprehend dialects has not been sufficiently studied. Moreover, it remains unclear how the dialectal understanding of the base LLM affects SLM performance. This study investigates the dialectal robustness of both LLMs and SLMs using Japanese dialects as a test case. We define robustness as the ratio of performance on dialectal versus standard inputs, enabling fair comparisons. Our experiments show that SLM robustness correlates with that of their text-based counterparts. Furthermore, training with dialectal data and fine-tuning the speech encoder each improves robustness in SLMs. 

---
# Adaptive Oscillatory Inductive Bias for Modeling Sharp Prosodic Dynamics in Diffusion-Based TTS 

**Authors**: Sandipan Dhar, Nirmesh J. Shah, Ashishkumar P. Gudmalwar, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2606.25424)  

**Abstract**: Diffusion-based text-to-speech (TTS) models have achieved significant improvements in speech quality. However, modeling sharp prosodic transitions and rapid pitch variations in expressive speech remains challenging. Existing diffusion-based TTS decoders commonly utilize periodic nonlinearities such as Snake activation function to capture harmonic structures, but this activation funcation provides limited adaptability when modeling abrupt amplitude and frequency variations. In this paper, we investigate the role of oscillatory inductive bias in diffusion-based TTS decoders and introduce an adaptive oscillatory nonlinearity that enables controllable periodic modulation while maintaining signal stability through a linear bypass component. We refer the resulting TTS system as OscillaTTS. Experiments on the LJSpeech and Emotional Speech Dataset show consistent improvements across objective and subjective evaluations, indicating improved modeling of expressive prosodic dynamics. 

---
# Sarashina2.2-TTS: Tackling Kanji Polyphony in Japanese Speech Generation via Data Scaling and Targeted Data Synthesis 

**Authors**: Lianbo Liu, Shiao Zhu, Kai Washizaki, Reo Yoneyama, Haesung Jeon, Mengjie Zhao, Yusuke Fujita, Hao Shi, Nao Yoshida, Yuan Gao, Roman Koshkin, Yukiya Hono, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2606.25369)  

**Abstract**: While large language model (LLM)-based text-to-speech (TTS) systems have achieved high-quality speech synthesis, most existing systems focus on English and Chinese. Japanese, however, remains under-explored, and its unique linguistic challenges, such as widespread context-dependent kanji polyphony, have yet to be adequately tackled. Here we introduce Sarashina2.2-TTS (this https URL), a Japanese-centric LLM-TTS system that tackles these challenges through a dual approach: data strategy and evaluation methodology. First, we scale training to approximately 361k hours of speech, incorporating a balanced mix of Japanese and English data. Furthermore, we design a targeted data augmentation pipeline covering all 2,136 Joyo (regular-use) kanji designated by Japan's Agency for Cultural Affairs to efficiently address kanji polyphony disambiguation. Second, we introduce the Joyo Kanji Yomi Benchmark (this https URL), covering all 2,136 Joyo kanji and their 4,378 readings. Alongside this benchmark, we propose Kana-CER, a metric that compares synthesized speech against reference readings in the kana space, eliminating orthographic variations to directly measure pronunciation correctness. Experiments demonstrate that our targeted data augmentation significantly improves reading accuracy. Overall, Sarashina2.2-TTS achieves state-of-the-art kanji-level reading accuracy and matches top baselines on general sentence-level pronunciation, while delivering the highest speaker similarity in zero-shot Japanese speech synthesis. Furthermore, cross-lingual evaluation reveals that Sarashina2.2-TTS is the only system that maintains stable Japanese pronunciation regardless of the prompt language, confirming that our balanced training approach improves cross-lingual robustness. 

---
# Data-Driven Evolution of Library and Information Science Research Methods (1990-2022): A Perspective Based on Fine-grained Method Entities 

**Authors**: Chengzhi Zhang, Yi Mao, Shuyu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.25320)  

**Abstract**: Since the 1990s, advancements in big data and information technology have increasingly driven data-centric research in the field of Library and Information Science (LIS). To assess the influence of this data-driven research paradigm on the LIS discipline, this study conducts a fine-grained analysis to uncover the evolutionary trends of research methods within the domain. Using academic papers from LIS published between 1990 and 2022, four key categories of data-driven method entities are automatically extracted: algorithms and models, data resources, software and tools, and metrics. Based on these entities, the study examines the evolution of LIS research methods from three dimensions: the characteristics of research method entities over time, their evolution within different research topics, and the evolutionary features of research method entities across various research methods. The findings highlight data resources as a pivotal driver of methodological evolution in LIS, revealing a cyclical pattern of "emergence-stability/practical application" in the development of research methods within the field. 

---
# Measuring Research Difficulty of Academic Papers: A Case Study in Natural Language Processing 

**Authors**: Haochuan Li, Jingyuan Li, Yi Zhao, Heng Zhang, Yukai Yang, Zile Hu, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25307)  

**Abstract**: With the rapid growth of the number of academic papers, systematically evaluating the difficulty of research and its relationship to academic impact offers important significance for research topic selection and resource allocation. However, current studies lack quantitative assessments of research difficulty and its correlation with academic impact. This paper proposes a comprehensive evaluation system for research difficulty, incorporating factors such as academic collaboration, content, and references. Taking the field of Natural Language Processing (NLP) as a case study, we extract both internal and external features from academic papers, compute multiple research difficulty indicators. We assign their weights using the entropy weight method and perform a weighted sum to obtain the research difficulty score of academic papers. This paper uses the citation frequency of academic papers to measure academic impact. To validate our approach, NLP experts assessed the difficulty of a sample of papers, and correlation analyses confirmed the reliability of our measurement. Empirical results reveal that in NLP, factors such as the number of pages, reference count, and participation of high-level institutions are significantly associated with academic impact. Moreover, we identify an inverted U-shaped relationship between research difficulty and academic impact. It suggests that moderately difficult research tends to achieve greater academic impact. 

---
# Multilingual Hematology Visual Question Answering Dataset 

**Authors**: Hajra Malik, Hafiza Tooba Aftab, Abdul Rehman, Mohsen Ali, Waqas Sultani  

**Link**: [PDF](https://arxiv.org/pdf/2606.25246)  

**Abstract**: Vision Language Models (VLMs) have shown promising capabilities in medical image analysis by jointly understanding visual and textual information for tasks such as Visual Question Answering. However, existing hematology vision-language resources remain predominantly English centric, limiting their applicability in multilingual healthcare environments. This challenge is releveant generally to South Asia and specifically to Pakistan, where Urdu is widely used despite healthcare information and digital medical systems being largely dependent on English. To investigate this gap, we conducted a survey among healthcare professionals, which revealed substantial language mismatches between clinical documentation and patient communication, emphasizing the need for multilingual healthcare technologies. To address this limitation, we introduce WBCMor VQA, a clinically validated bilingual English, Urdu morphology aware VQA benchmark for leukemia and normal white blood cell analysis. The benchmark is constructed using morphology-aware annotations from LeukemiaAttri and WBCAtt datasets and supported by a domain specific Urdu hematology dictionary to ensure linguistic consistency and clinical correctness. The final benchmark contains 110K bilingual question answer pairs serving as VQA annotations for 20K leukemic and normal single-cell images. Furthermore, we establish baseline performance by evaluating multiple open-source VLMs on the proposed benchmark. The proposed resource aims to facilitate the development of accessible and clinically relevant AI systems for multilingual healthcare environments. 

---
# ASAP: Agent-System Co-Design for Wall-Clock-Centered Auto HPO Research for ML Experiments 

**Authors**: Taicheng Guo, Haomin Zhuang, Kehan Guo, Yujun Zhou, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25207)  

**Abstract**: Hyperparameter Optimization (HPO) is essential for maximizing machine learning model performance, and its core challenge is sample efficiency: finding strong configurations within a limited budget. Because every HPO tool relies on a surrogate prior that imparts its own inductive bias, individual tools struggle once problems become sufficiently diverse and drift from these priors. Motivated by the reasoning and generalization capabilities of LLMs, recent work has explored using LLMs for HPO and reports improved per-iteration performance. Yet these methods share two limitations with a common origin: they use the LLM as a single-tool replacement evaluated by iteration count. (i) Deployed in place of prior tools, the LLM is itself constrained by its pretraining objective to one family of inductive-biased proposals; this single-source setup still fails to handle the full diversity of problems. (ii) Per-iteration evaluation ignores that, in real runs, LLM inference or tool execution is paid serially on top of model evaluation every round, so iteration-count gains do not translate into end-to-end wall-clock gains. We present ASAP, an agent-system co-design that addresses both limitations. On the agent side, ASAP uses the LLM to integrate a diverse pool of inductive-biased optimizers and to select among their proposals each round. On the system side, ASAP re-architects the loop to reduce end-to-end wall-clock while preserving regret quality: a prefix-stable prompt maximizes KV-cache reuse across rounds; speculation parallelism hides the remaining LLM and tool latency under model evaluation via a relative-error accept test; and a Self-Tuner adapts the speculation threshold from execution logs off the critical path. Extensive experiments on diverse modern HPO tasks show that ASAP consistently outperforms baselines, underscoring the value of tool integration and agent-system co-design. 

---
# RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory 

**Authors**: Yixun Hu, Zhicheng Zheng, Lihan Zha, Chunwei Xing, Rajdeep Singh, Omar Hossain, Antonio Loquercio, Dhruv Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.25206)  

**Abstract**: Long-term robot deployment requires a compact and scalable memory that preserves fine-grained visual semantics, grounds observations in space and time, and enables efficient storage and retrieval. In this paper, we propose RAVEN, an agentic memory system for long-horizon robotic question answering and navigation. RAVEN stores visual embeddings with pose and time in a vector database, and grounds retrieval in a spatial map to answer queries and navigate to goals. By operating directly on visual embeddings, RAVEN avoids lossy image-to-text captioning and enables accurate semantic, spatial, and temporal retrieval at scale. Across several simulated and real-world video question-answering benchmarks, RAVEN consistently surpasses caption-based memory systems and matches frontier VLMs on long-horizon tasks at 10$\times$ lower retrieval cost. Finally, we instantiate RAVEN on a Unitree Go1 robot for the task of long-horizon navigation for natural language goal-reaching, and show successful deployment over several large indoor environments. 

---
# To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient Multi-Agent RAG 

**Authors**: Jungseob Lee, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2606.25191)  

**Abstract**: Multi-agent document assessment for retrieval-augmented generation is computationally expensive, driving practitioners toward smaller, deployable models whose assessment mechanisms remain poorly understood. We conduct a controlled study of training-free interventions on 7B-9B instruction-tuned models across diverse QA benchmarks, revealing a sharp dichotomy in how models benefit from assessment. For weaker baselines, the dominant mechanism is per-document isolation. Astoundingly, assessment-free isolation matches full multi-agent assessment, demonstrating that resolving multi-document context confusion, rather than scoring quality, drives outsized gains of up to 50 percentage points. Conversely, for strong baselines where scoring quality matters, we introduce Reasoning-Score Coupling, a label-free perturbation probe that classifies scoring behavior. Integrating these findings, we propose MADARA, a model-adaptive routing architecture. Crucially, MADARA's diagnostic thresholds derived from a single pilot model generalize zero-shot to four unseen model families, providing a robust, lightweight pipeline to eliminate computational overhead. 

---
# LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search 

**Authors**: Nikhil Abhyankar, Sha Li, Sanchit Kabra, Naren Ramakrishnan, Yulia Gel, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2606.25039)  

**Abstract**: Recovering governing Ordinary Differential Equations (ODEs) from data is a central challenge in modeling dynamical systems across scientific domains. Existing approaches cast discovery as a static inference problem over fixed datasets, assuming that the observed trajectories are sufficiently informative. However, dynamical systems evolve over large state spaces, and limited data can make multiple equations observationally indistinguishable, leading to identifiability gaps and the recovery of incorrect governing equations. To address this, we introduce LLM-ACES, or LLM-guided Active Closed-loop Equation Search, a closed-loop framework that jointly optimizes symbolic hypothesis construction and adaptive data acquisition. In LLM-ACES, a large language model (LLM) proposes operator priors that partition the large search space into distinct regions, within which candidate equations are fit to the observed data. The disagreement among these candidates guides the acquisition of informative trajectories, creating a feedback loop that iteratively refines both the hypothesis space and the discovered dynamics. On 122 ODE systems spanning ODEBench and ODEBase, LLM-ACES achieves the lowest median NMSE, outperforming state-of-the-art baselines by several orders of magnitude while achieving a high symbolic accuracy of 46.2% and 52.4%, respectively. Our analysis further shows that LLM-ACES is sample-efficient, achieving better performance with one-tenth the data. Furthermore, LLM-ACES's feedback-driven data acquisition makes it robust to noise and recovers the correct symbolic structure, while baselines introduce spurious terms that fit the data locally but obscure the true governing relationships. 

---
# Do Thinking Tokens Help with Safety? 

**Authors**: Narutatsu Ri, Abhishek Panigrahi, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2606.25013)  

**Abstract**: Today's reasoning models use thinking tokens to attain stronger performance on benchmarks than their instruction-tuned counterparts. It is also generally believed that this more "deliberative" mode should improve alignment and safety, by providing the model a safe space to consider whether its planned answer to a request violates its safety principles. We present evidence that this intuition is not always correct. Across frontier open-weight reasoning models spanning GPT-OSS, Qwen, Olmo, and Phi families, we find that the eventual refusal/compliance outcome is already strongly predictable via a trained head on the first token's hidden representation ($0.84$-$0.95$ AUROC and $\sim88\%$ balanced accuracy for predicting refusal/compliance) before any visible thinking. The thinking process turns out to be more akin to prefix completion than to deliberative revision, with the final outcome rarely changing after the first $\sim20\%$ of thinking, despite giving the appearance of deliberation at the text level ($\sim74\%$ of text-level deliberations occur when the response distribution is already locked to one refusal/compliance side). We also find that existing inference-time and training-based safety interventions, despite being motivated by the goal of inducing deliberation, largely shift model behavior toward over-refusal while suppressing already-scarce deliberation signals. Our results suggest that safety behavior in current reasoning models is much less deliberative than commonly assumed, and highlight the need for methods that induce real safety deliberation. 

---
# Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns 

**Authors**: Vatsal Baherwani, Zixi Chen, Shikai Qiu, Andrew Gordon Wilson, Pavel Izmailov  

**Link**: [PDF](https://arxiv.org/pdf/2606.25010)  

**Abstract**: Neural scaling laws for transformer language models predict smooth improvements in pretraining loss with increasing parameters, but downstream capabilities such as in-context learning are known to emerge abruptly past a certain model scale. In this paper, we show that emergent capabilities arise stochastically throughout training, with larger models acquiring them earlier on average. We demonstrate that the emergence of capabilities such as pattern completion and indirect object identification corresponds to the abrupt learning of task-relevant attention patterns. To isolate this phenomenon, we train transformer models on synthetic linear map and cellular automata datasets, and we show that the difficulty of learning attention patterns depends on context length and pattern sparsity. Moreover, scaling the number of attention heads improves learning efficiency on our synthetic tasks, while increasing the head dimension yields diminishing returns past a minimum capacity. We additionally investigate architectures with alternative attention mechanisms, showing that MLP-Mixer outperforms a transformer on linear map tasks with complex attention patterns. Our findings provide a mechanistic insight into emergence, showing that downstream capabilities arise abruptly due to the intrinsic difficulty of learning sparse attention patterns in transformer models. 

---
# Neural Scaling Universality: If Exponents Are Fixed, Time to Understand Coefficients 

**Authors**: Yizhou Liu, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2606.25008)  

**Abstract**: Neural scaling laws describe how pre-training loss decays as power laws with training time, model size, and compute. This position paper argues that the exponents of these power laws are fixed by generic mechanisms: a one-third time scaling due to the strong nonlinearity of Softmax, an inverse width scaling due to representational superposition, and an inverse depth scaling due to ensemble averaging of Transformer layers. These mechanisms are robust to a wide range of data structures and architectural details, placing current large language models in a universality class with fixed exponents. The coefficients, however, are expected to be sensitive to data and architecture details, and directly determine practical quantities such as the optimal model shape and the compute-optimal frontier. We therefore argue that understanding the coefficients is the key to near-term performance improvements, and that a closer examination of the current universality class may reveal pathways to better universality classes. 

---
# Learning Diachronic Representations of Ancient Greek Letterforms 

**Authors**: John Pavlopoulos, Spyros Barbakos, Lavinia Ferretti, Dionysis Voulgarakis, Asimina Paparrigopoulou, Maria Konstantinidou, Giuseppe De Gregorio, Isabelle Marthot-Santaniello, Paraskevi Platanou, Holger Essler  

**Link**: [PDF](https://arxiv.org/pdf/2606.24984)  

**Abstract**: Learning representations that remain robust across centuries of variation in handwriting is a key challenge in diachronic representation learning. Taking one of the longest continuously used writing systems, ancient Greek, as a case study, we introduce three datasets for diachronic representation learning: Hell-Char, a curated training set spanning the 3rd-1st centuries BCE, and two evaluation sets, PaLit-Char (2nd-5th c. CE) and Med-Char (9th-14th c. CE). To address the challenges of symbolic variation, scarce data, and systematic degradation, we propose: a similarity-weighted supervised contrastive loss that biases embeddings using dynamically estimated inter-class similarities, and a lacuna-driven augmentation scheme that simulates realistic manuscript corruptions. Trained with these strategies, both a lightweight CNN and a pretrained ResNet achieve strong recognition performance and produce embeddings that more coherently separate character classes than PCA or generic pretrained models. These embeddings enable clustering, identification of stylistic subgroups, and construction of prototype images that visualize diachronic evolution and transitional letterforms. Our results demonstrate that respecting intrinsic inter-letter relationships and augmenting with domain-informed corruptions yield robust, interpretable representations, offering a transferable paradigm for representation learning under scarce, temporally evolving, and noisy conditions. Code and data available at: this https URL. 

---
# Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieval 

**Authors**: Pradyumna Narayana, Sana Ayromlou, Purvi Sehgal  

**Link**: [PDF](https://arxiv.org/pdf/2606.24976)  

**Abstract**: Foundation-model agents in multi-step, open-ended environments frequently suffer from compounding errors, where early mistakes contaminate long-horizon trajectories. While Multi-Agent Debate (MAD) succeeds in deterministic domains, agents in subjective tasks like persuasion experience severe problem drift and sycophantic conformity. We identify semantic leakage in standard Retrieval-Augmented Generation (RAG) as a reproducible trigger for these failures, as standard RAG prioritizes vocabulary overlap over logical necessity.
To eliminate this leakage, we introduce Taxonomic Strategy RAG (TS-RAG), a systems intervention that routes strategies through a discrete categorical bottleneck to decouple argumentative structure from topical content. Zero-shot, cross-domain evaluations demonstrate that TS-RAG significantly improves the transfer of abstract logic where standard semantic retrieval collapses. Crucially, TS-RAG acts as a "capability bridge" in asymmetric deployments, empowering lightweight persuaders to consistently defeat parametrically superior opponents (improving win rates from 70.5 to 78.5) and accelerating argumentative efficiency. Finally, we introduce trace-level diagnostics via a turn-by-turn Debate State Representation (DSR), demonstrating the necessity of strict constraints to prevent evaluation collapse via default agentic sycophancy. 

---
# Why Do Accumulated Transformations Extrapolate? 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2606.24975)  

**Abstract**: PaTH Attention showed that replacing RoPE's position-indexed rotations with accumulated data-dependent Householder reflections yields strong length extrapolation, though performance degrades at extreme context lengths. We ask whether this depends on Householder-specific structure or reflects a general property of accumulated transformations along source-to-query paths. We study a simpler variant keeping RoPE's block-diagonal SO(2) rotations but replacing position-indexed angles with accumulated token-dependent ones. It shows the same pattern: improved extrapolation then degradation at long contexts. We prove the result extends to accumulated orthogonal transformations satisfying certain regularity conditions: their products become incoherent after finitely many steps, suppressing attention to distant tokens. Accumulated rotations of queries and keys create a finite mixing window independent of context length; per-token suppression learned in training transfers unchanged to any evaluation length, and high-dimensional concentration produces a score gap suppressing far tokens while near-route transport preserves the target signal. Conversely, a lower bound shows accumulated rotations must eventually degrade: as the far set grows, no rotations preserve the near signal without explicit far-mass control. For SO(2) rotations, rotating values too makes residual far contributions combine incoherently, extending the range. Controlled experiments support these predictions: random accumulated rotations substantially improve extrapolation over RoPE, learned token-dependent rotations maintain near-training-length perplexity far beyond the training context, and rotating values helps over queries and keys alone. Rotation-only models still degrade at extreme lengths, while ALiBi stays length-stable, consistent with the need for far-mass control. 

---
# Digital Twin-Driven Adaptive Sim-to-Real Alignment via Reinforcement Learning for Vibration-Based Bearing Health Monitoring Under Data Scarcity 

**Authors**: Jinghan Wang, Yanjun Chen, Wei Zhang, Wentao Wu, Tianchen Liu, Gaoliang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.24954)  

**Abstract**: Vibration-based health monitoring of rotating machinery requires reliable fault diagnosis under operational data constraints, yet condition assessment remains challenged by structural scarcity of fault events and heterogeneous sim-to-real gaps in digital twin-generated signals. Each fault type generates impulses with distinct periodicity, amplitude modulation, and spectral character, making feature-space discrepancies fundamentally heterogeneous across fault classes. Existing domain adaptation methods apply a class-agnostic global transformation that cannot close all fault-specific gaps without distorting inter-class separability, while uniform source-target mixing introduces distributional noise into the data-abundant Normal class. These limitations stem from treating a sequential, state-dependent alignment problem as a one-shot optimization. Each corrective transformation simultaneously reshapes all class distributions, creating state dependencies that static gradient descent cannot resolve. We formulate feature alignment as a continuous-action Markov decision process solved via Proximal Policy Optimization, where the learned policy issues fault-type-specific affine corrections responsive to the current feature-space configuration, with a dual-objective reward balancing gap minimization against separability preservation. An asymmetry-aware strategy reserves real data for the Normal class while augmenting fault classes with policy-aligned simulated samples. Validation across XJTU-SY, CWRU, and a self-built slewing bearing testbed confirms the dominant gain from reinforcement learning-driven alignment, and cross-equipment linear probing achieves 92.8% without encoder retraining, demonstrating transferable monitoring capability. 

---
# The Hitchhiker's Guide to Agentic AI: From Foundations to Systems 

**Authors**: Haggai Roitman  

**Link**: [PDF](https://arxiv.org/pdf/2606.24937)  

**Abstract**: The Hitchhiker's Guide to Agentic AI is a comprehensive practitioner's reference for building autonomous AI systems. The book covers the full stack from first principles to production deployment, organized around a central thesis: building great agentic systems requires understanding every layer of the pipeline, not just one. The book opens with the LLM substrate -- transformer architecture, GPU systems, training and fine-tuning (SFT,LoRA, MoE), model compression, and inference optimization -- treated as essential foundations rather than the primary focus. It then develops the alignment and reasoning layer: reinforcement learning from human feedback (RLHF), PPO, DPO and its variants, GRPO, reward modeling, and RL for large reasoning models including chain-of-thought and test-time scaling. The second half is devoted to agentic AI proper. Topics include agentic training and trajectory-based RL, retrieval-augmented generation (RAG and Agentic RAG), memory systems (in-context, external, episodic, and semantic), agent harness design and context management, and a taxonomy of agent design patterns. Inter-agent coordination is covered in depth: the Model Context Protocol (MCP), agent skills and tool use, the Agent-to-Agent (A2A) communication protocol, and multi-agent architectures spanning centralized, decentralized, and hierarchical topologies. The book concludes with agent development frameworks, agentic UI design, evaluation methodology for agentic tasks, and production deployment. Each chapter pairs rigorous theoretical foundations with implementation guidance, code examples, and references to the primary literature. 

---
# Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs 

**Authors**: Przemysław Czuma  

**Link**: [PDF](https://arxiv.org/pdf/2606.24897)  

**Abstract**: Biomedical text mining, scientometrics, and the construction of training corpora for biomedical large language models (LLMs) all assume that the abstract text returned by a bibliographic API faithfully reproduces the published abstract. This pre-registered audit (OSF this http URL) tests that assumption for four widely used public APIs (PubMed E-utilities, Crossref, OpenAlex, Semantic Scholar) against PubMed Central (PMC) JATS XML as a common ground truth. From a complete enumeration of the PMC Open Access subset for 2024 (about 700,000 records), a simple random sample of 4,000 English-language research articles was drawn; for each, we recorded whether Unicode characters from four pre-specified classes present in the JATS abstract (typographic punctuation, mathematical/scientific symbols, Greek letters, special whitespace) were preserved by each API. Two systematic, deterministic losses met the pre-registered criterion (upper 95% CI bound below 5%): the PubMed AbstractText field preserved typographic punctuation in only 0.6% of eligible abstracts (95% CI 0.3-1.0%), and OpenAlex preserved special whitespace in 0% (0.0-0.4%). A blinded mechanism audit attributed the first loss to character substitution and the second to inverted-index serialization. Mathematical symbols and Greek letters were preserved faithfully (over 95%) by all four APIs. Separately, Crossref returned no abstract for 24.6% of papers (coverage 75.4%, 95% CI 74.1-76.7%), concentrated in specific publishers (Elsevier and ACS: 0%). Character-level fidelity is therefore API-dependent and undocumented: the same publisher-deposited JATS text carries different surface signatures depending on the serving API, with direct consequences for tokenization-sensitive bibliometrics, corpus construction, and character-level indicators of LLM-assisted writing. 

---
