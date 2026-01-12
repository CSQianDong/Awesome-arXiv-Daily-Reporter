# AdaFuse: Adaptive Ensemble Decoding with Test-Time Scaling for LLMs 

**Authors**: Chengming Cui, Tianxin Wei, Ziyi Chen, Ruizhong Qiu, Zhichen Zeng, Zhining Liu, Xuying Ning, Duo Zhou, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2601.06022)  

**Abstract**: Large language models (LLMs) exhibit complementary strengths arising from differences in pretraining data, model architectures, and decoding behaviors. Inference-time ensembling provides a practical way to combine these capabilities without retraining. However, existing ensemble approaches suffer from fundamental limitations. Most rely on fixed fusion granularity, which lacks the flexibility required for mid-generation adaptation and fails to adapt to different generation characteristics across tasks. To address these challenges, we propose AdaFuse, an adaptive ensemble decoding framework that dynamically selects semantically appropriate fusion units during generation. Rather than committing to a fixed granularity, AdaFuse adjusts fusion behavior on the fly based on the decoding context, with words serving as basic building blocks for alignment. To be specific, we introduce an uncertainty-based criterion to decide whether to apply ensembling at each decoding step. Under confident decoding states, the model continues generation directly. In less certain states, AdaFuse invokes a diversity-aware scaling strategy to explore alternative candidate continuations and inform ensemble decisions. This design establishes a synergistic interaction between adaptive ensembling and test-time scaling, where ensemble decisions guide targeted exploration, and the resulting diversity in turn strengthens ensemble quality. Experiments on open-domain question answering, arithmetic reasoning, and machine translation demonstrate that AdaFuse consistently outperforms strong ensemble baselines, achieving an average relative improvement of 6.88%. The code is available at this https URL. 

---
# Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards 

**Authors**: Jiajie Zhang, Xin Lv, Ling Feng, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06021)  

**Abstract**: Reinforcement learning (RL) has emerged as a critical technique for enhancing LLM-based deep search agents. However, existing approaches primarily rely on binary outcome rewards, which fail to capture the comprehensiveness and factuality of agents' reasoning process, and often lead to undesirable behaviors such as shortcut exploitation and hallucinations. To address these limitations, we propose \textbf{Citation-aware Rubric Rewards (CaRR)}, a fine-grained reward framework for deep search agents that emphasizes reasoning comprehensiveness, factual grounding, and evidence connectivity. CaRR decomposes complex questions into verifiable single-hop rubrics and requires agents to satisfy these rubrics by explicitly identifying hidden entities, supporting them with correct citations, and constructing complete evidence chains that link to the predicted answer. We further introduce \textbf{Citation-aware Group Relative Policy Optimization (C-GRPO)}, which combines CaRR and outcome rewards for training robust deep search agents. Experiments show that C-GRPO consistently outperforms standard outcome-based RL baselines across multiple deep search benchmarks. Our analysis also validates that C-GRPO effectively discourages shortcut exploitation, promotes comprehensive, evidence-grounded reasoning, and exhibits strong generalization to open-ended deep research tasks. Our code and data are available at this https URL. 

---
# Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks 

**Authors**: Elias Lumer, Faheem Nizar, Akshaya Jangiti, Kevin Frank, Anmol Gulati, Mandar Phadate, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2601.06007)  

**Abstract**: Recent advancements in Large Language Model (LLM) agents have enabled complex multi-turn agentic tasks requiring extensive tool calling, where conversations can span dozens of API calls with increasingly large context windows. However, although major LLM providers offer prompt caching to reduce cost and latency, its benefits for agentic workloads remain underexplored in the research literature. To our knowledge, no prior work quantifies these cost savings or compares caching strategies for multi-turn agentic tasks. We present a comprehensive evaluation of prompt caching across three major LLM providers (OpenAI, Anthropic, and Google) and compare three caching strategies, including full context caching, system prompt only caching, and caching that excludes dynamic tool results. We evaluate on DeepResearchBench, a multi-turn agentic benchmark where agents autonomously execute real-world web search tool calls to answer complex research questions, measuring both API cost and time to first token (TTFT) across over 500 agent sessions with 10,000-token system prompts. Our results demonstrate that prompt caching reduces API costs by 45-80% and improves time to first token by 13-31% across providers. We find that strategic prompt cache block control, such as placing dynamic content at the end of the system prompt, avoiding dynamic traditional function calling, and excluding dynamic tool results, provides more consistent benefits than naive full-context caching, which can paradoxically increase latency. Our analysis reveals nuanced variations in caching behavior across providers, and we provide practical guidance for implementing prompt caching in production agentic systems. 

---
# The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning 

**Authors**: Qiguang Chen, Yantao Du, Ziniu Li, Jinhao Liu, Songyao Duan, Jiarui Guo, Minghao Liu, Jiaheng Liu, Tong Yang, Ge Zhang, Libo Qin, Wanxiang Che, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06002)  

**Abstract**: Large language models (LLMs) often fail to learn effective long chain-of-thought (Long CoT) reasoning from human or non-Long-CoT LLMs imitation. To understand this, we propose that effective and learnable Long CoT trajectories feature stable molecular-like structures in unified view, which are formed by three interaction types: Deep-Reasoning (covalent-like), Self-Reflection (hydrogen-bond-like), and Self-Exploration (van der Waals-like). Analysis of distilled trajectories reveals these structures emerge from Long CoT fine-tuning, not keyword imitation. We introduce Effective Semantic Isomers and show that only bonds promoting fast entropy convergence support stable Long CoT learning, while structural competition impairs training. Drawing on these findings, we present Mole-Syn, a distribution-transfer-graph method that guides synthesis of effective Long CoT structures, boosting performance and RL stability across benchmarks. 

---
# Distilling Feedback into Memory-as-a-Tool 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2601.05960)  

**Abstract**: We propose a framework that amortizes the cost of inference-time reasoning by converting transient critiques into retrievable guidelines, through a file-based memory system and agent-controlled tool calls. We evaluate this method on the Rubric Feedback Bench, a novel dataset for rubric-based learning. Experiments demonstrate that our augmented LLMs rapidly match the performance of test-time refinement pipelines while drastically reducing inference cost. 

---
# Can We Predict Before Executing Machine Learning Agents? 

**Authors**: Jingsheng Zheng, Jintian Zhang, Yujie Luo, Yuren Mao, Yunjun Gao, Lun Du, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05930)  

**Abstract**: Autonomous machine learning agents have revolutionized scientific discovery, yet they remain constrained by a Generate-Execute-Feedback paradigm. Previous approaches suffer from a severe Execution Bottleneck, as hypothesis evaluation relies strictly on expensive physical execution. To bypass these physical constraints, we internalize execution priors to substitute costly runtime checks with instantaneous predictive reasoning, drawing inspiration from World Models. In this work, we formalize the task of Data-centric Solution Preference and construct a comprehensive corpus of 18,438 pairwise comparisons. We demonstrate that LLMs exhibit significant predictive capabilities when primed with a Verified Data Analysis Report, achieving 61.5% accuracy and robust confidence calibration. Finally, we instantiate this framework in FOREAGENT, an agent that employs a Predict-then-Verify loop, achieving a 6x acceleration in convergence while surpassing execution-based baselines by +6%. Our code and dataset will be publicly available soon at this https URL. 

---
# Pantagruel: Unified Self-Supervised Encoders for French Text and Speech 

**Authors**: Phuong-Hang Le, Valentin Pelloin, Arnault Chatelain, Maryem Bouziane, Mohammed Ghennai, Qianwen Guan, Kirill Milintsevich, Salima Mdhaffar, Aidan Mannion, Nils Defauw, Shuyue Gu, Alexandre Audibert, Marco Dinarelli, Yannick Estève, Lorraine Goeuriot, Steffen Lalande, Nicolas Hervé, Maximin Coavoux, François Portet, Étienne Ollion, Marie Candito, Maxime Peyrard, Solange Rossato, Benjamin Lecouteux, Aurélie Nardy, Gilles Sérasset, Vincent Segonne, Solène Evain, Diandra Fabre, Didier Schwab  

**Link**: [PDF](https://arxiv.org/pdf/2601.05911)  

**Abstract**: We release Pantagruel models, a new family of self-supervised encoder models for French text and speech. Instead of predicting modality-tailored targets such as textual tokens or speech units, Pantagruel learns contextualized target representations in the feature space, allowing modality-specific encoders to capture linguistic and acoustic regularities more effectively. Separate models are pre-trained on large-scale French corpora, including Wikipedia, OSCAR and CroissantLLM for text, together with MultilingualLibriSpeech, LeBenchmark, and INA-100k for speech. INA-100k is a newly introduced 100,000-hour corpus of French audio derived from the archives of the Institut National de l'Audiovisuel (INA), the national repository of French radio and television broadcasts, providing highly diverse audio data. We evaluate Pantagruel across a broad range of downstream tasks spanning both modalities, including those from the standard French benchmarks such as FLUE or LeBenchmark. Across these tasks, Pantagruel models show competitive or superior performance compared to strong French baselines such as CamemBERT, FlauBERT, and LeBenchmark2.0, while maintaining a shared architecture that can seamlessly handle either speech or text inputs. These results confirm the effectiveness of feature-space self-supervised objectives for French representation learning and highlight Pantagruel as a robust foundation for multimodal speech-text understanding. 

---
# Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency 

**Authors**: Haoming Xu, Ningyuan Zhao, Yunzhi Yao, Weihong Xu, Hongru Wang, Xinle Deng, Shumin Deng, Jeff Z. Pan, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05905)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in real-world settings, correctness alone is insufficient. Reliable deployment requires maintaining truthful beliefs under contextual perturbations. Existing evaluations largely rely on point-wise confidence like Self-Consistency, which can mask brittle belief. We show that even facts answered with perfect self-consistency can rapidly collapse under mild contextual interference. To address this gap, we propose Neighbor-Consistency Belief (NCB), a structural measure of belief robustness that evaluates response coherence across a conceptual neighborhood. To validate the efficiency of NCB, we introduce a new cognitive stress-testing protocol that probes outputs stability under contextual interference. Experiments across multiple LLMs show that the performance of high-NCB data is relatively more resistant to interference. Finally, we present Structure-Aware Training (SAT), which optimizes context-invariant belief structure and reduces long-tail knowledge brittleness by approximately 30%. Code will be available at this https URL. 

---
# HAPS: Hierarchical LLM Routing with Joint Architecture and Parameter Search 

**Authors**: Zihang Tian, Rui Li, Jingsen Zhang, Xiaohe Bo, Wei Huo, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05903)  

**Abstract**: Large language model (LLM) routing aims to exploit the specialized strengths of different LLMs for diverse tasks. However, existing approaches typically focus on selecting LLM architectures while overlooking parameter settings, which are critical for task performance. In this paper, we introduce HAPS, a hierarchical LLM routing framework that jointly searches over model architectures and parameters. Specifically, we use a high-level router to select among candidate LLM architectures, and then search for the optimal parameters for the selected architectures based on a low-level router. We design a parameter generation network to share parameters between the two routers to mutually enhance their capabilities. In the training process, we design a reward-augmented objective to effectively optimize our framework. Experiments on two commonly used benchmarks show that HAPS consistently outperforms strong routing baselines. We have released our code at this https URL. 

---
# An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift 

**Authors**: Constantinos Karouzos, Xingwei Tan, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2601.05882)  

**Abstract**: Preference tuning aligns pretrained language models to human judgments of quality, helpfulness, or safety by optimizing over explicit preference signals rather than likelihood alone. Prior work has shown that preference-tuning degrades performance and reduces helpfulness when evaluated outside the training domain. However, the extent to which adaptation strategies mitigate this domain shift remains unexplored. We address this challenge by conducting a comprehensive and systematic study of alignment generalization under domain shift. We compare five popular alignment objectives and various adaptation strategies from source to target, including target-domain supervised fine-tuning and pseudo-labeling, across summarization and question-answering helpfulness tasks. Our findings reveal systematic differences in generalization across alignment objectives under domain shift. We show that adaptation strategies based on pseudo-labeling can substantially reduce domain-shift degradation 

---
# Gender Bias in LLMs: Preliminary Evidence from Shared Parenting Scenario in Czech Family Law 

**Authors**: Jakub Harasta, Matej Vasina, Martin Kornel, Tomas Foltynek  

**Link**: [PDF](https://arxiv.org/pdf/2601.05879)  

**Abstract**: Access to justice remains limited for many people, leading laypersons to increasingly rely on Large Language Models (LLMs) for legal self-help. Laypeople use these tools intuitively, which may lead them to form expectations based on incomplete, incorrect, or biased outputs. This study examines whether leading LLMs exhibit gender bias in their responses to a realistic family law scenario. We present an expert-designed divorce scenario grounded in Czech family law and evaluate four state-of-the-art LLMs GPT-5 nano, Claude Haiku 4.5, Gemini 2.5 Flash, and Llama 3.3 in a fully zero-shot interaction. We deploy two versions of the scenario, one with gendered names and one with neutral labels, to establish a baseline for comparison. We further introduce nine legally relevant factors that vary the factual circumstances of the case and test whether these variations influence the models' proposed shared-parenting ratios. Our preliminary results highlight differences across models and suggest gender-dependent patterns in the outcomes generated by some systems. The findings underscore both the risks associated with laypeople's reliance on LLMs for legal guidance and the need for more robust evaluation of model behavior in sensitive legal contexts. We present exploratory and descriptive evidence intended to identify systematic asymmetries rather than to establish causal effects. 

---
# iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models 

**Authors**: Meghana Sunil, Manikandarajan Venmathimaran, Muthu Subash Kavitha  

**Link**: [PDF](https://arxiv.org/pdf/2601.05877)  

**Abstract**: Recent work shows that large multimodal models (LMMs) can self-improve from unlabeled data via self-play and intrinsic feedback. Yet existing self-evolving frameworks mainly reward final outcomes, leaving intermediate reasoning weakly constrained despite its importance for visually grounded decision making. We propose iReasoner, a self-evolving framework that improves an LMM's implicit reasoning by explicitly eliciting chain-of-thought (CoT) and rewarding its internal agreement. In a Proposer--Solver loop over unlabeled images, iReasoner augments outcome-level intrinsic rewards with a trajectory-aware signal defined over intermediate reasoning steps, providing learning signals that distinguish reasoning paths leading to the same answer without ground-truth labels or external judges. Starting from Qwen2.5-VL-7B, iReasoner yields up to $+2.1$ points across diverse multimodal reasoning benchmarks under fully unsupervised post-training. We hope this work serves as a starting point for reasoning-aware self-improvement in LMMs in purely unsupervised settings. 

---
# Continual-learning for Modelling Low-Resource Languages from Large Language Models 

**Authors**: Santosh Srinath K, Mudit Somani, Varun Reddy Padala, Prajna Devi Upadhyay, Abhijit Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.05874)  

**Abstract**: Modelling a language model for a multi-lingual scenario includes several potential challenges, among which catastrophic forgetting is the major challenge. For example, small language models (SLM) built for low-resource languages by adapting large language models (LLMs) pose the challenge of catastrophic forgetting. This work proposes to employ a continual learning strategy using parts-of-speech (POS)-based code-switching along with a replay adapter strategy to mitigate the identified gap of catastrophic forgetting while training SLM from LLM. Experiments conducted on vision language tasks such as visual question answering and language modelling task exhibits the success of the proposed architecture. 

---
# FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG 

**Authors**: Maxime Dassen, Rebecca Kotula, Kenton Murray, Andrew Yates, Dawn Lawrie, Efsun Kayi, James Mayfield, Kevin Duh  

**Link**: [PDF](https://arxiv.org/pdf/2601.05866)  

**Abstract**: Retrieval-Augmented Generation (RAG) models are critically undermined by citation hallucinations, a deceptive failure where a model confidently cites a source that fails to support its claim. Existing work often attributes hallucination to a simple over-reliance on the model's parametric knowledge. We challenge this view and introduce FACTUM (Framework for Attesting Citation Trustworthiness via Underlying Mechanisms), a framework of four mechanistic scores measuring the distinct contributions of a model's attention and FFN pathways, and the alignment between them. Our analysis reveals two consistent signatures of correct citation: a significantly stronger contribution from the model's parametric knowledge and greater use of the attention sink for information synthesis. Crucially, we find the signature of a correct citation is not static but evolves with model scale. For example, the signature of a correct citation for the Llama-3.2-3B model is marked by higher pathway alignment, whereas for the Llama-3.1-8B model, it is characterized by lower alignment, where pathways contribute more distinct, orthogonal information. By capturing this complex, evolving signature, FACTUM outperforms state-of-the-art baselines by up to 37.5% in AUC. Our findings reframe citation hallucination as a complex, scale-dependent interplay between internal mechanisms, paving the way for more nuanced and reliable RAG systems. 

---
# What do the metrics mean? A critical analysis of the use of Automated Evaluation Metrics in Interpreting 

**Authors**: Jonathan Downie, Joss Moorkens  

**Link**: [PDF](https://arxiv.org/pdf/2601.05864)  

**Abstract**: With the growth of interpreting technologies, from remote interpreting and Computer-Aided Interpreting to automated speech translation and interpreting avatars, there is now a high demand for ways to quickly and efficiently measure the quality of any interpreting delivered. A range of approaches to fulfil the need for quick and efficient quality measurement have been proposed, each involving some measure of automation. This article examines these recently-proposed quality measurement methods and will discuss their suitability for measuring the quality of authentic interpreting practice, whether delivered by humans or machines, concluding that automatic metrics as currently proposed cannot take into account the communicative context and thus are not viable measures of the quality of any interpreting provision when used on their own. Across all attempts to measure or even categorise quality in Interpreting Studies, the contexts in which interpreting takes place have become fundamental to the final analysis. 

---
# CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning 

**Authors**: Alexandra Dragomir, Florin Brad, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05858)  

**Abstract**: Large language models (LLMs) have demonstrated competitive performance in zero-shot multilingual machine translation (MT). Some follow-up works further improved MT performance via preference optimization, but they leave a key aspect largely underexplored: the order in which data samples are given during training. We address this topic by integrating curriculum learning into various state-of-the-art preference optimization algorithms to boost MT performance. We introduce a novel curriculum learning strategy with restarts (CLewR), which reiterates easy-to-hard curriculum multiple times during training to effectively mitigate the catastrophic forgetting of easy examples. We demonstrate consistent gains across several model families (Gemma2, Qwen2.5, Llama3.1) and preference optimization techniques. We publicly release our code at this https URL. 

---
# Router-Suggest: Dynamic Routing for Multimodal Auto-Completion in Visually-Grounded Dialogs 

**Authors**: Sandeep Mishra, Devichand Budagam, Anubhab Mandal, Bishal Santra, Pawan Goyal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.05851)  

**Abstract**: Real-time multimodal auto-completion is essential for digital assistants, chatbots, design tools, and healthcare consultations, where user inputs rely on shared visual context. We introduce Multimodal Auto-Completion (MAC), a task that predicts upcoming characters in live chats using partially typed text and visual cues. Unlike traditional text-only auto-completion (TAC), MAC grounds predictions in multimodal context to better capture user intent. To enable this task, we adapt MMDialog and ImageChat to create benchmark datasets. We evaluate leading vision-language models (VLMs) against strong textual baselines, highlighting trade-offs in accuracy and efficiency. We present Router-Suggest, a router framework that dynamically selects between textual models and VLMs based on dialog context, along with a lightweight variant for resource-constrained environments. Router-Suggest achieves a 2.3x to 10x speedup over the best-performing VLM. A user study shows that VLMs significantly excel over textual models on user satisfaction, notably saving user typing effort and improving the quality of completions in multi-turn conversations. These findings underscore the need for multimodal context in auto-completions, leading to smarter, user-aware assistants. 

---
# Semantic NLP Pipelines for Interoperable Patient Digital Twins from Unstructured EHRs 

**Authors**: Rafael Brens, Yuqiao Meng, Luoxi Tang, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05847)  

**Abstract**: Digital twins -- virtual replicas of physical entities -- are gaining traction in healthcare for personalized monitoring, predictive modeling, and clinical decision support. However, generating interoperable patient digital twins from unstructured electronic health records (EHRs) remains challenging due to variability in clinical documentation and lack of standardized mappings. This paper presents a semantic NLP-driven pipeline that transforms free-text EHR notes into FHIR-compliant digital twin representations. The pipeline leverages named entity recognition (NER) to extract clinical concepts, concept normalization to map entities to SNOMED-CT or ICD-10, and relation extraction to capture structured associations between conditions, medications, and observations. Evaluation on MIMIC-IV Clinical Database Demo with validation against MIMIC-IV-on-FHIR reference mappings demonstrates high F1-scores for entity and relation extraction, with improved schema completeness and interoperability compared to baseline methods. 

---
# Left, Right, or Center? Evaluating LLM Framing in News Classification and Generation 

**Authors**: Molly Kennedy, Ali Parker, Yihong Liu, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.05835)  

**Abstract**: Large Language Model (LLM) based summarization and text generation are increasingly used for producing and rewriting text, raising concerns about political framing in journalism where subtle wording choices can shape interpretation. Across nine state-of-the-art LLMs, we study political framing by testing whether LLMs' classification-based bias signals align with framing behavior in their generated summaries. We first compare few-shot ideology predictions against LEFT/CENTER/RIGHT labels. We then generate "steered" summaries under FAITHFUL, CENTRIST, LEFT, and RIGHT prompts, and score all outputs using a single fixed ideology evaluator. We find pervasive ideological center-collapse in both article-level ratings and generated text, indicating a systematic tendency toward centrist framing. Among evaluated models, Grok 4 is by far the most ideologically expressive generator, while Claude Sonnet 4.5 and Llama 3.1 achieve the strongest bias-rating performance among commercial and open-weight models, respectively. 

---
# Peek2: A Regex-free implementation of pretokenizers for Byte-level BPE 

**Authors**: Liu Zai  

**Link**: [PDF](https://arxiv.org/pdf/2601.05833)  

**Abstract**: Pretokenization is a crucial, sequential pass in Byte-level BPE tokenizers. Our proposed new implementation, Peek2, serves as a drop-in replacement for cl100k-like pretokenizers used in GPT-3, LLaMa-3, and Qwen-2.5. Designed with performance and safety in mind, Peek2 is Regex-free and delivers a $ 1.11\times $ improvement in overall throughput across the entire Byte-level BPE encoding process. This algorithm runs entirely on the CPU, has stable linear complexity $ O(n) $, and provides presegmentation results identical to those of the original Regex-based pretokenizer. 

---
# LLMs as Science Journalists: Supporting Early-stage Researchers in Communicating Their Science to the Public 

**Authors**: Milad Alshomary, Grace Li, Anubhav Jangra, Yufang Hou, Kathleen McKeown, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05821)  

**Abstract**: The scientific community needs tools that help early-stage researchers effectively communicate their findings and innovations to the public. Although existing general-purpose Large Language Models (LLMs) can assist in this endeavor, they are not optimally aligned for it. To address this, we propose a framework for training LLMs to emulate the role of a science journalist that can be used by early-stage researchers to learn how to properly communicate their papers to the general public. We evaluate the usefulness of our trained LLM Journalists in leading conversations with both simulated and human researchers. %compared to the general-purpose ones. Our experiments indicate that LLMs trained using our framework ask more relevant questions that address the societal impact of research, prompting researchers to clarify and elaborate on their findings. In the user study, the majority of participants who interacted with our trained LLM Journalist appreciated it more than interacting with general-purpose LLMs. 

---
# EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis 

**Authors**: Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05808)  

**Abstract**: Large language models (LLMs) are expected to be trained to act as agents in various real-world environments, but this process relies on rich and varied tool-interaction sandboxes. However, access to real systems is often restricted; LLM-simulated environments are prone to hallucinations and inconsistencies; and manually built sandboxes are hard to scale. In this paper, we propose EnvScaler, an automated framework for scalable tool-interaction environments via programmatic synthesis. EnvScaler comprises two components. First, SkelBuilder constructs diverse environment skeletons through topic mining, logic modeling, and quality evaluation. Then, ScenGenerator generates multiple task scenarios and rule-based trajectory validation functions for each environment. With EnvScaler, we synthesize 191 environments and about 7K scenarios, and apply them to Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for Qwen3 series models. Results on three benchmarks show that EnvScaler significantly improves LLMs' ability to solve tasks in complex environments involving multi-turn, multi-tool interactions. We release our code and data at this https URL. 

---
# Simplify-This: A Comparative Analysis of Prompt-Based and Fine-Tuned LLMs 

**Authors**: Eilam Cohen, Itamar Bul, Danielle Inbar, Omri Loewenbach  

**Link**: [PDF](https://arxiv.org/pdf/2601.05794)  

**Abstract**: Large language models (LLMs) enable strong text generation, and in general there is a practical tradeoff between fine-tuning and prompt engineering. We introduce Simplify-This, a comparative study evaluating both paradigms for text simplification with encoder-decoder LLMs across multiple benchmarks, using a range of evaluation metrics. Fine-tuned models consistently deliver stronger structural simplification, whereas prompting often attains higher semantic similarity scores yet tends to copy inputs. A human evaluation favors fine-tuned outputs overall. We release code, a cleaned derivative dataset used in our study, checkpoints of fine-tuned models, and prompt templates to facilitate reproducibility and future work. 

---
# One Script Instead of Hundreds? On Pretraining Romanized Encoder Language Models 

**Authors**: Benedikt Ebing, Lennart Keller, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2601.05776)  

**Abstract**: Exposing latent lexical overlap, script romanization has emerged as an effective strategy for improving cross-lingual transfer (XLT) in multilingual language models (mLMs). Most prior work, however, focused on setups that favor romanization the most: (1) transfer from high-resource Latin-script to low-resource non-Latin-script languages and/or (2) between genealogically closely related languages with different scripts. It thus remains unclear whether romanization is a good representation choice for pretraining general-purpose mLMs, or, more precisely, if information loss associated with romanization harms performance for high-resource languages. We address this gap by pretraining encoder LMs from scratch on both romanized and original texts for six typologically diverse high-resource languages, investigating two potential sources of degradation: (i) loss of script-specific information and (ii) negative cross-lingual interference from increased vocabulary overlap. Using two romanizers with different fidelity profiles, we observe negligible performance loss for languages with segmental scripts, whereas languages with morphosyllabic scripts (Chinese and Japanese) suffer degradation that higher-fidelity romanization mitigates but cannot fully recover. Importantly, comparing monolingual LMs with their mLM counterpart, we find no evidence that increased subword overlap induces negative interference. We further show that romanization improves encoding efficiency (i.e., fertility) for segmental scripts at a negligible performance cost. 

---
# AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor 

**Authors**: Shu Yang, Jingyu Hu, Tong Li, Hanqi Yan, Wenxuan Wang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05752)  

**Abstract**: We introduce AutoMonitor-Bench, the first benchmark designed to systematically evaluate the reliability of LLM-based misbehavior monitors across diverse tasks and failure modes. AutoMonitor-Bench consists of 3,010 carefully annotated test samples spanning question answering, code generation, and reasoning, with paired misbehavior and benign instances. We evaluate monitors using two complementary metrics: Miss Rate (MR) and False Alarm Rate (FAR), capturing failures to detect misbehavior and oversensitivity to benign behavior, respectively. Evaluating 12 proprietary and 10 open-source LLMs, we observe substantial variability in monitoring performance and a consistent trade-off between MR and FAR, revealing an inherent safety-utility tension. To further explore the limits of monitor reliability, we construct a large-scale training corpus of 153,581 samples and fine-tune Qwen3-4B-Instruction to investigate whether training on known, relatively easy-to-construct misbehavior datasets improves monitoring performance on unseen and more implicit misbehaviors. Our results highlight the challenges of reliable, scalable misbehavior monitoring and motivate future work on task-aware designing and training strategies for LLM-based monitors. 

---
# Analysing Differences in Persuasive Language in LLM-Generated Text: Uncovering Stereotypical Gender Patterns 

**Authors**: Amalie Brogaard Pauli, Maria Barrett, Max Müller-Eberstein, Isabelle Augenstein, Ira Assent  

**Link**: [PDF](https://arxiv.org/pdf/2601.05751)  

**Abstract**: Large language models (LLMs) are increasingly used for everyday communication tasks, including drafting interpersonal messages intended to influence and persuade. Prior work has shown that LLMs can successfully persuade humans and amplify persuasive language. It is therefore essential to understand how user instructions affect the generation of persuasive language, and to understand whether the generated persuasive language differs, for example, when targeting different groups. In this work, we propose a framework for evaluating how persuasive language generation is affected by recipient gender, sender intent, or output language. We evaluate 13 LLMs and 16 languages using pairwise prompt instructions. We evaluate model responses on 19 categories of persuasive language using an LLM-as-judge setup grounded in social psychology and communication science. Our results reveal significant gender differences in the persuasive language generated across all models. These patterns reflect biases consistent with gender-stereotypical linguistic tendencies documented in social psychology and sociolinguistics. 

---
# Visualising Information Flow in Word Embeddings with Diffusion Tensor Imaging 

**Authors**: Thomas Fabian  

**Link**: [PDF](https://arxiv.org/pdf/2601.05713)  

**Abstract**: Understanding how large language models (LLMs) represent natural language is a central challenge in natural language processing (NLP) research. Many existing methods extract word embeddings from an LLM, visualise the embedding space via point-plots, and compare the relative positions of certain words. However, this approach only considers single words and not whole natural language expressions, thus disregards the context in which a word is used. Here we present a novel tool for analysing and visualising information flow in natural language expressions by applying diffusion tensor imaging (DTI) to word embeddings. We find that DTI reveals how information flows between word embeddings. Tracking information flows within the layers of an LLM allows for comparing different model structures and revealing opportunities for pruning an LLM's under-utilised layers. Furthermore, our model reveals differences in information flows for tasks like pronoun resolution and metaphor detection. Our results show that our model permits novel insights into how LLMs represent actual natural language expressions, extending the comparison of isolated word embeddings and improving the interpretability of NLP models. 

---
# Multimodal In-context Learning for ASR of Low-resource Languages 

**Authors**: Zhaolin Li, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2601.05707)  

**Abstract**: Automatic speech recognition (ASR) still covers only a small fraction of the world's languages, mainly due to supervised data scarcity. In-context learning (ICL) with large language models (LLMs) addresses this problem, but prior work largely focuses on high-resource languages covered during training and text-only settings. This paper investigates whether speech LLMs can learn unseen languages with multimodal ICL (MICL), and how this learning can be used to improve ASR. We conduct experiments with two speech LLMs, Phi-4 and Qwen3-Omni, on three diverse endangered languages. Firstly, we find that MICL is effective for unseen languages, leveraging both speech and text modalities. We further show that cross-lingual transfer learning improves MICL efficiency on target languages without training on them. Moreover, we analyze attention patterns to interpret MICL mechanisms, and we observe layer-dependent preferences between audio and text context, with an overall bias towards text. Finally, we show that prompt-based ASR with speech LLMs performs poorly on unseen languages, motivating a simple ASR system that combines a stronger acoustic model with a speech LLM via MICL-based selection of acoustic hypotheses. Results show that MICL consistently improves ASR performance, and that cross-lingual transfer learning matches or outperforms corpus-trained language models without using target-language data. Our code is publicly available. 

---
# Afri-MCQA: Multimodal Cultural Question Answering for African Languages 

**Authors**: Atnafu Lambebo Tonja, Srija Anand, Emilio Villa-Cueva, Israel Abebe Azime, Jesujoba Oluwadara Alabi, Muhidin A. Mohamed, Debela Desalegn Yadeta, Negasi Haile Abadi, Abigail Oppong, Nnaemeka Casmir Obiefuna, Idris Abdulmumin, Naome A Etori, Eric Peter Wairagala, Kanda Patrick Tshinu, Imanigirimbabazi Emmanuel, Gabofetswe Malema, Alham Fikri Aji, David Ifeoluwa Adelani, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2601.05699)  

**Abstract**: Africa is home to over one-third of the world's languages, yet remains underrepresented in AI research. We introduce Afri-MCQA, the first Multilingual Cultural Question-Answering benchmark covering 7.5k Q&A pairs across 15 African languages from 12 countries. The benchmark offers parallel English-African language Q&A pairs across text and speech modalities and was entirely created by native speakers. Benchmarking large language models (LLMs) on Afri-MCQA shows that open-weight models perform poorly across evaluated cultures, with near-zero accuracy on open-ended VQA when queried in native language or speech. To evaluate linguistic competence, we include control experiments meant to assess this specific aspect separate from cultural knowledge, and we observe significant performance gaps between native languages and English for both text and speech. These findings underscore the need for speech-first approaches, culturally grounded pretraining, and cross-lingual cultural transfer. To support more inclusive multimodal AI development in African languages, we release our Afri-MCQA under academic license or CC BY-NC 4.0 on HuggingFace (this https URL) 

---
# Stephanie2: Thinking, Waiting, and Making Decisions Like Humans in Step-by-Step AI Social Chat 

**Authors**: Hao Yang, Hongyuan Lu, Dingkang Yang, Wenliang Yang, Peng Sun, Xiaochuan Zhang, Jun Xiao, Kefan He, Wai Lam, Yang Liu, Xinhua Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.05657)  

**Abstract**: Instant-messaging human social chat typically progresses through a sequence of short messages. Existing step-by-step AI chatting systems typically split a one-shot generation into multiple messages and send them sequentially, but they lack an active waiting mechanism and exhibit unnatural message pacing. In order to address these issues, we propose Stephanie2, a novel next-generation step-wise decision-making dialogue agent. With active waiting and message-pace adaptation, Stephanie2 explicitly decides at each step whether to send or wait, and models latency as the sum of thinking time and typing time to achieve more natural pacing. We further introduce a time-window-based dual-agent dialogue system to generate pseudo dialogue histories for human and automatic evaluations. Experiments show that Stephanie2 clearly outperforms Stephanie1 on metrics such as naturalness and engagement, and achieves a higher pass rate on human evaluation with the role identification Turing test. 

---
# A Framework for Personalized Persuasiveness Prediction via Context-Aware User Profiling 

**Authors**: Sejun Park, Yoonah Park, Jongwon Lim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2601.05654)  

**Abstract**: Estimating the persuasiveness of messages is critical in various applications, from recommender systems to safety assessment of LLMs. While it is imperative to consider the target persuadee's characteristics, such as their values, experiences, and reasoning styles, there is currently no established systematic framework to optimize leveraging a persuadee's past activities (e.g., conversations) to the benefit of a persuasiveness prediction model. To address this problem, we propose a context-aware user profiling framework with two trainable components: a query generator that generates optimal queries to retrieve persuasion-relevant records from a user's history, and a profiler that summarizes these records into a profile to effectively inform the persuasiveness prediction model. Our evaluation on the ChangeMyView Reddit dataset shows consistent improvements over existing methods across multiple predictor models, with gains of up to +13.77%p in F1 score. Further analysis shows that effective user profiles are context-dependent and predictor-specific, rather than relying on static attributes or surface-level similarity. Together, these results highlight the importance of task-oriented, context-dependent user profiling for personalized persuasiveness prediction. 

---
# Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs 

**Authors**: Alireza Dehghanpour Farashah, Aditi Khandelwal, Marylou Fauchard, Zhuan Shi, Negar Rostamzadeh, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05641)  

**Abstract**: As multilingual large language models become more widely used, ensuring their safety and fairness across diverse linguistic contexts presents unique challenges. While existing research on machine unlearning has primarily focused on monolingual settings, typically English, multilingual environments introduce additional complexities due to cross-lingual knowledge transfer and biases embedded in both pretraining and fine-tuning data. In this work, we study multilingual unlearning using the Aya-Expanse 8B model under two settings: (1) data unlearning and (2) concept unlearning. We extend benchmarks for factual knowledge and stereotypes to ten languages through translation: English, French, Arabic, Japanese, Russian, Farsi, Korean, Hindi, Hebrew, and Indonesian. These languages span five language families and a wide range of resource levels. Our experiments show that unlearning in high-resource languages is generally more stable, with asymmetric transfer effects observed between typologically related languages. Furthermore, our analysis of linguistic distances indicates that syntactic similarity is the strongest predictor of cross-lingual unlearning behavior. 

---
# GIFT: Games as Informal Training for Generalizable LLMs 

**Authors**: Nuoyan Lyu, Bingbing Xu, Weihao Meng, Yige Yuan, Yang Zhang, Zhiyong Huang, Tat-Seng Chua, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05633)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in formal learning tasks such as mathematics and code generation, they still struggle with the "practical wisdom" and generalizable intelligence, such as strategic creativity and social reasoning, that characterize human cognition. This gap arises from a lack of informal learning, which thrives on interactive feedback rather than goal-oriented instruction. In this paper, we propose treating Games as a primary environment for LLM informal learning, leveraging their intrinsic reward signals and abstracted complexity to cultivate diverse competencies. To address the performance degradation observed in multi-task learning, we introduce a Nested Training Framework. Unlike naive task mixing optimizing an implicit "OR" objective, our framework employs sequential task composition to enforce an explicit "AND" objective, compelling the model to master multiple abilities simultaneously to achieve maximal rewards. Using GRPO-based reinforcement learning across Matrix Games, TicTacToe, and Who's the Spy games, we demonstrate that integrating game-based informal learning not only prevents task interference but also significantly bolsters the model's generalization across broad ability-oriented benchmarks. The framework and implementation are publicly available. 

---
# Text Detoxification in isiXhosa and Yorùbá: A Cross-Lingual Machine Learning Approach for Low-Resource African Languages 

**Authors**: Abayomi O. Agbeyangi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05624)  

**Abstract**: Toxic language is one of the major barrier to safe online participation, yet robust mitigation tools are scarce for African languages. This study addresses this critical gap by investigating automatic text detoxification (toxic to neutral rewriting) for two low-resource African languages, isiXhosa and Yorùbá. The work contributes a novel, pragmatic hybrid methodology: a lightweight, interpretable TF-IDF and Logistic Regression model for transparent toxicity detection, and a controlled lexicon- and token-guided rewriting component. A parallel corpus of toxic to neutral rewrites, which captures idiomatic usage, diacritics, and code switching, was developed to train and evaluate the model. The detection component achieved stratified K-fold accuracies of 61-72% (isiXhosa) and 72-86% (Yorùbá), with per-language ROC-AUCs up to 0.88. The rewriting component successfully detoxified all detected toxic sentences while preserving 100% of non-toxic sentences. These results demonstrate that scalable, interpretable machine learning detectors combined with rule-based edits offer a competitive and resource-efficient solution for culturally adaptive safety tooling, setting a new benchmark for low-resource Text Style Transfer (TST) in African languages. 

---
# Data Augmented Pipeline for Legal Information Extraction and Reasoning 

**Authors**: Nguyen Minh Phuong, Ha-Thanh Nguyen, May Myo Zin, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2601.05609)  

**Abstract**: In this paper, we propose a pipeline leveraging Large Language Models (LLMs) for data augmentation in Information Extraction tasks within the legal domain. The proposed method is both simple and effective, significantly reducing the manual effort required for data annotation while enhancing the robustness of Information Extraction systems. Furthermore, the method is generalizable, making it applicable to various Natural Language Processing (NLP) tasks beyond the legal domain. 

---
# ACR: Adaptive Context Refactoring via Context Refactoring Operators for Multi-Turn Dialogue 

**Authors**: Jiawei Shen, Jia Zhu, Hanghui Guo, Weijie Shi, Yue Cui, Qingyu Niu, Guoqing Ma, Yidan Liang, Jingjiang Liu, Yiling Wang, Shimin Di, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05589)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in multi-turn dialogue. However, in multi-turn dialogue, models still struggle to stay aligned with what has been established earlier, follow dependencies across many turns, and avoid drifting into incorrect facts as the interaction grows longer. Existing approaches primarily focus on extending the context window, introducing external memory, or applying context compression, yet these methods still face limitations such as \textbf{contextual inertia} and \textbf{state drift}. To address these challenges, we propose the \textbf{A}daptive \textbf{C}ontext \textbf{R}efactoring \textbf{(ACR)} Framework, which dynamically monitors and reshapes the interaction history to mitigate contextual inertia and state drift actively. ACR is built on a library of context refactoring operators and a teacher-guided self-evolving training paradigm that learns when to intervene and how to refactor, thereby decoupling context management from the reasoning process. Extensive experiments on multi-turn dialogue demonstrate that our method significantly outperforms existing baselines while reducing token consumption. 

---
# Can large language models interpret unstructured chat data on dynamic group decision-making processes? Evidence on joint destination choice 

**Authors**: Sung-Yoo Lim, Koki Sato, Kiyoshi Takami, Giancarlos Parady, Eui-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05582)  

**Abstract**: Social activities result from complex joint activity-travel decisions between group members. While observing the decision-making process of these activities is difficult via traditional travel surveys, the advent of new types of data, such as unstructured chat data, can help shed some light on these complex processes. However, interpreting these decision-making processes requires inferring both explicit and implicit factors. This typically involves the labor-intensive task of manually annotating dialogues to capture context-dependent meanings shaped by the social and cultural norms. This study evaluates the potential of Large Language Models (LLMs) to automate and complement human annotation in interpreting decision-making processes from group chats, using data on joint eating-out activities in Japan as a case study. We designed a prompting framework inspired by the knowledge acquisition process, which sequentially extracts key decision-making factors, including the group-level restaurant choice set and outcome, individual preferences of each alternative, and the specific attributes driving those preferences. This structured process guides the LLM to interpret group chat data, converting unstructured dialogues into structured tabular data describing decision-making factors. To evaluate LLM-driven outputs, we conduct a quantitative analysis using a human-annotated ground truth dataset and a qualitative error analysis to examine model limitations. Results show that while the LLM reliably captures explicit decision-making factors, it struggles to identify nuanced implicit factors that human annotators readily identified. We pinpoint specific contexts when LLM-based extraction can be trusted versus when human oversight remains essential. These findings highlight both the potential and limitations of LLM-based analysis for incorporating non-traditional data sources on social activities. 

---
# ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging 

**Authors**: Junyao Yang, Chen Qian, Dongrui Liu, Wen Shen, Yong Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05560)  

**Abstract**: Large Reasoning Models (LRMs) with long chain-of-thought reasoning have recently achieved remarkable success. Yet, equipping domain-specialized models with such reasoning capabilities, referred to as "Reasoning + X", remains a significant challenge. While model merging offers a promising training-free solution, existing methods often suffer from a destructive performance collapse: existing methods tend to both weaken reasoning depth and compromise domain-specific utility. Interestingly, we identify a counter-intuitive phenomenon underlying this failure: reasoning ability predominantly resides in parameter regions with low gradient sensitivity, contrary to the common assumption that domain capabilities correspond to high-magnitude parameters. Motivated by this insight, we propose ReasonAny, a novel merging framework that resolves the reasoning-domain performance collapse through Contrastive Gradient Identification. Experiments across safety, biomedicine, and finance domains show that ReasonAny effectively synthesizes "Reasoning + X" capabilities, significantly outperforming state-of-the-art baselines while retaining robust reasoning performance. 

---
# Generation-Based and Emotion-Reflected Memory Update: Creating the KEEM Dataset for Better Long-Term Conversation 

**Authors**: Jeonghyun Kang, Hongjin Kim, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05548)  

**Abstract**: In this work, we introduce the Keep Emotional and Essential Memory (KEEM) dataset, a novel generation-based dataset designed to enhance memory updates in long-term conversational systems. Unlike existing approaches that rely on simple accumulation or operation-based methods, which often result in information conflicts and difficulties in accurately tracking a user's current state, KEEM dynamically generates integrative memories. This process not only preserves essential factual information but also incorporates emotional context and causal relationships, enabling a more nuanced understanding of user interactions. By seamlessly updating a system's memory with both emotional and essential data, our approach promotes deeper empathy and enhances the system's ability to respond meaningfully in open-domain conversations. 

---
# Can Large Language Models Differentiate Harmful from Argumentative Essays? Steps Toward Ethical Essay Scoring 

**Authors**: Hongjin Kim, Jeonghyun Kang, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.05545)  

**Abstract**: This study addresses critical gaps in Automated Essay Scoring (AES) systems and Large Language Models (LLMs) with regard to their ability to effectively identify and score harmful essays. Despite advancements in AES technology, current models often overlook ethically and morally problematic elements within essays, erroneously assigning high scores to essays that may propagate harmful opinions. In this study, we introduce the Harmful Essay Detection (HED) benchmark, which includes essays integrating sensitive topics such as racism and gender bias, to test the efficacy of various LLMs in recognizing and scoring harmful content. Our findings reveal that: (1) LLMs require further enhancement to accurately distinguish between harmful and argumentative essays, and (2) both current AES models and LLMs fail to consider the ethical dimensions of content during scoring. The study underscores the need for developing more robust AES systems that are sensitive to the ethical implications of the content they are scoring. 

---
# Closing the Modality Reasoning Gap for Speech Large Language Models 

**Authors**: Chaoren Wang, Heng Lu, Xueyao Zhang, Shujie Liu, Yan Lu, Jinyu Li, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05543)  

**Abstract**: Although speech large language models have achieved notable progress, a substantial modality reasoning gap remains: their reasoning performance on speech inputs is markedly weaker than on text. This gap could be associated with representational drift across Transformer layers and behavior deviations in long-chain reasoning. To address this issue, we introduce TARS, a reinforcement-learning framework that aligns text-conditioned and speech-conditioned trajectories through an asymmetric reward design. The framework employs two dense and complementary signals: representation alignment, which measures layer-wise hidden-state similarity between speech- and text-conditioned trajectories, and behavior alignment, which evaluates semantic consistency between generated outputs and reference text completions. Experiments on challenging reasoning benchmarks, including MMSU and OBQA, show that our approach significantly narrows the modality reasoning gap and achieves state-of-the-art performance among 7B-scale Speech LLMs. 

---
# Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism 

**Authors**: Yuhao Shen, Tianyu Liu, Junyi Shen, Jinyang Wu, Quan Kong, Li Huan, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05524)  

**Abstract**: Parallel Speculative Decoding (PSD) accelerates traditional Speculative Decoding (SD) by overlapping draft generation with verification. However, it remains hampered by two fundamental challenges: (1) a theoretical speedup ceiling dictated by the speed ratio between the draft and target models, and (2) high computational waste and pipeline stall due to mid-sequence token rejections of early errors. To address these limitations, we introduce \textsc{Double} (Double Retrieval Speculative Parallelism). By bridging the gap between SD and PSD, our framework resolves the Retrieval \emph{Precision-Efficiency Dilemma} through a novel synchronous mechanism. Specifically, we enable the draft model to execute iterative retrieval speculations to break the theoretical speedup limits; to alleviate rejections without rollback, the target model performs authoritative retrieval to generate multi-token guidance. \textsc{Double} is entirely training-free and lossless. Extensive experiments demonstrate state-of-the-art speedup of $\textbf{5.3}\times$ on LLaMA3.3-70B and $\textbf{2.8}\times$ on Qwen3-32B, significantly outperforming the advanced method EAGLE-3 that requires extensive model training. 

---
# CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems 

**Authors**: Xuemei Tang, Chengxi Yan, Jinghang Gu, Chu-Ren Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05520)  

**Abstract**: Despite strong performance on many tasks, large language models (LLMs) show limited ability in historical and cultural reasoning, particularly in non-English contexts such as Chinese history. Taxonomic structures offer an effective mechanism to organize historical knowledge and improve understanding. However, manual taxonomy construction is costly and difficult to scale. Therefore, we propose \textbf{CHisAgent}, a multi-agent LLM framework for historical taxonomy construction in ancient Chinese contexts. CHisAgent decomposes taxonomy construction into three role-specialized stages: a bottom-up \textit{Inducer} that derives an initial hierarchy from raw historical corpora, a top-down \textit{Expander} that introduces missing intermediate concepts using LLM world knowledge, and an evidence-guided \textit{Enricher} that integrates external structured historical resources to ensure faithfulness. Using the \textit{Twenty-Four Histories}, we construct a large-scale, domain-aware event taxonomy covering politics, military, diplomacy, and social life in ancient China. Extensive reference-free and reference-based evaluations demonstrate improved structural coherence and coverage, while further analysis shows that the resulting taxonomy supports cross-cultural alignment. 

---
# FlashMem: Distilling Intrinsic Latent Memory via Computation Reuse 

**Authors**: Yubo Hou, Zhisheng Chen, Tao Wan, Zengchang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05505)  

**Abstract**: The stateless architecture of Large Language Models inherently lacks the mechanism to preserve dynamic context, compelling agents to redundantly reprocess history to maintain long-horizon autonomy. While latent memory offers a solution, current approaches are hindered by architectural segregation, relying on auxiliary encoders that decouple memory from the reasoning backbone. We propose FlashMem, a framework that distills intrinsic memory directly from transient reasoning states via computation reuse. Leveraging the property that internal representations uniquely encode input trajectories, FlashMem identifies the last hidden state as a sufficient statistic for the interaction history. This enables a Shared-KV Consolidator to synthesize memory by attending directly to the backbone's frozen cache, eliminating redundant re-parameterization. Furthermore, a parameter-free Cognitive Monitor leverages attention entropy to adaptively trigger consolidation only when high epistemic uncertainty is detected. Experiments demonstrate that FlashMem matches the performance of heavy baselines while reducing inference latency by 5 times, effectively bridging the gap between efficiency and persistent cognition. 

---
# MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards 

**Authors**: Zhiyu Shen, Ziming Wu, Fuming Lai, Shaobing Lian, Yanghui Rao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05488)  

**Abstract**: Maintaining consistency in long-term dialogues remains a fundamental challenge for LLMs, as standard retrieval mechanisms often fail to capture the temporal evolution of historical states. While memory-augmented frameworks offer a structured alternative, current systems rely on static prompting of closed-source models or suffer from ineffective training paradigms with sparse rewards. We introduce MemBuilder, a reinforcement learning framework that trains models to orchestrate multi-dimensional memory construction with attributed dense rewards. MemBuilder addresses two key challenges: (1) Sparse Trajectory-Level Rewards: we employ synthetic session-level question generation to provide dense intermediate rewards across extended trajectories; and (2) Multi-Dimensional Memory Attribution: we introduce contribution-aware gradient weighting that scales policy updates based on each component's downstream impact. Experimental results show that MemBuilder enables a 4B-parameter model to outperform state-of-the-art closed-source baselines, exhibiting strong generalization across long-term dialogue benchmarks. 

---
# The Facade of Truth: Uncovering and Mitigating LLM Susceptibility to Deceptive Evidence 

**Authors**: Herun Wan, Jiaying Wu, Minnan Luo, Fanxiao Li, Zhi Zeng, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05478)  

**Abstract**: To reliably assist human decision-making, LLMs must maintain factual internal beliefs against misleading injections. While current models resist explicit misinformation, we uncover a fundamental vulnerability to sophisticated, hard-to-falsify evidence. To systematically probe this weakness, we introduce MisBelief, a framework that generates misleading evidence via collaborative, multi-round interactions among multi-role LLMs. This process mimics subtle, defeasible reasoning and progressive refinement to create logically persuasive yet factually deceptive claims. Using MisBelief, we generate 4,800 instances across three difficulty levels to evaluate 7 representative LLMs. Results indicate that while models are robust to direct misinformation, they are highly sensitive to this refined evidence: belief scores in falsehoods increase by an average of 93.0\%, fundamentally compromising downstream recommendations. To address this, we propose Deceptive Intent Shielding (DIS), a governance mechanism that provides an early warning signal by inferring the deceptive intent behind evidence. Empirical results demonstrate that DIS consistently mitigates belief shifts and promotes more cautious evidence evaluation. 

---
# Towards Valid Student Simulation with Large Language Models 

**Authors**: Zhihao Yuan, Yunze Xiao, Ming Li, Weihao Xuan, Richard Tong, Mona Diab, Tom Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2601.05473)  

**Abstract**: This paper presents a conceptual and methodological framework for large language model (LLM) based student simulation in educational settings. The authors identify a core failure mode, termed the "competence paradox" in which broadly capable LLMs are asked to emulate partially knowledgeable learners, leading to unrealistic error patterns and learning dynamics. To address this, the paper reframes student simulation as a constrained generation problem governed by an explicit Epistemic State Specification (ESS), which defines what a simulated learner can access, how errors are structured, and how learner state evolves over time. The work further introduces a Goal-by-Environment framework to situate simulated student systems according to behavioral objectives and deployment contexts. Rather than proposing a new system or benchmark, the paper synthesizes prior literature, formalizes key design dimensions, and articulates open challenges related to validity, evaluation, and ethical risks. Overall, the paper argues for epistemic fidelity over surface realism as a prerequisite for using LLM-based simulated students as reliable scientific and pedagogical instruments. 

---
# Do LLMs Need Inherent Reasoning Before Reinforcement Learning? A Study in Korean Self-Correction 

**Authors**: Hongjin Kim, Jaewook Lee, Kiyoung Lee, Jong-hun Shin, Soojong Lim, Oh-Woog Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2601.05459)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning and self-correction abilities in high-resource languages like English, but their performance remains limited in low-resource languages such as Korean. In this study, we investigate whether reinforcement learning (RL) can enhance Korean reasoning abilities to a degree comparable to English. Our findings reveal that RL alone yields limited improvements when applied to models lacking inherent Korean reasoning capabilities. To address this, we explore several fine-tuning strategies and show that aligning the model's internal reasoning processes with Korean inputs-particularly by tuning Korean-specific neurons in early layers-is key to unlocking RL's effectiveness. We introduce a self-correction code-switching dataset to facilitate this alignment and observe significant performance gains in both mathematical reasoning and self-correction tasks. Ultimately, we conclude that the crucial factor in multilingual reasoning enhancement is not injecting new linguistic knowledge, but effectively eliciting and aligning existing reasoning capabilities. Our study provides a new perspective on how internal translation and neuron-level tuning contribute to multilingual reasoning alignment in LLMs. 

---
# Tracing Moral Foundations in Large Language Models 

**Authors**: Chenxiao Yu, Bowen Yi, Farzan Karimi-Malekabadi, Suhaib Abdurahman, Jinyi Ye, Shrikanth Narayanan, Yue Zhao, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2601.05437)  

**Abstract**: Large language models (LLMs) often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial ``moral mimicry.'' Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed within two instruction-tuned LLMs: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that both models represent and distinguish moral foundations in a structured, layer-dependent way that aligns with human judgments. At a finer scale, SAE features show clear semantic links to specific foundations, suggesting partially disentangled mechanisms within shared representations. Finally, steering along either dense vectors or sparse features produces predictable shifts in foundation-relevant behavior, demonstrating a causal connection between internal representations and moral outputs. Together, our results provide mechanistic evidence that moral concepts in LLMs are distributed, layered, and partly disentangled, suggesting that pluralistic moral structure can emerge as a latent pattern from the statistical regularities of language alone. 

---
# Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions 

**Authors**: Minda Zhao, Yilun Du, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05414)  

**Abstract**: As large language models (LLMs) transition from chat interfaces to integral components of stochastic pipelines across domains like educational assessment and synthetic data construction, the ability to faithfully sample from specified probability distributions has become a functional requirement rather than a theoretical curiosity. We present the first large-scale, statistically powered audit of native probabilistic sampling in frontier LLMs, benchmarking 11 models across 15 distributions. To disentangle failure modes, we employ a dual-protocol design: Batch Generation, where a model produces N=1000 samples within one response, and Independent Requests, comprising $N=1000$ stateless calls. We observe a sharp protocol asymmetry: batch generation achieves only modest statistical validity, with a 13% median pass rate, while independent requests collapse almost entirely, with 10 of 11 models passing none of the distributions. Beyond this asymmetry, we reveal that sampling fidelity degrades monotonically with distributional complexity and aggravates as the requested sampling horizon N increases. Finally, we demonstrate the propagation of these failures into downstream tasks: models fail to enforce uniform answer-position constraints in MCQ generation and systematically violate demographic targets in attribute-constrained text-to-image prompt synthesis. These findings indicate that current LLMs lack a functional internal sampler, necessitating the use of external tools for applications requiring statistical guarantees. 

---
# Glitter: Visualizing Lexical Surprisal for Readability in Administrative Texts 

**Authors**: Jan Černý, Ivana Kvapilíková, Silvie Cinková  

**Link**: [PDF](https://arxiv.org/pdf/2601.05411)  

**Abstract**: This work investigates how measuring information entropy of text can be used to estimate its readability. We propose a visualization framework that can be used to approximate information entropy of text using multiple language models and visualize the result. The end goal is to use this method to estimate and improve readability and clarity of administrative or bureaucratic texts. Our toolset is available as a libre software on this https URL. 

---
# Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection 

**Authors**: Zhiwei Liu, Yupen Cao, Yuechen Jiang, Mohsinul Kabir, Polydoros Giannouris, Chen Xu, Ziyang Xu, Tianlei Zhu, Tariquzzaman Faisal, Triantafillos Papadopoulos, Yan Wang, Lingfei Qian, Xueqing Peng, Zhuohan Xie, Ye Yuan, Saeed Almheiri, Abdulrazzaq Alnajjar, Mingbin Chen, Harry Stuart, Paul Thompson, Prayag Tiwari, Alejandro Lopez-Lira, Xue Liu, Jimin Huang, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2601.05403)  

**Abstract**: Large language models (LLMs) have been widely applied across various domains of finance. Since their training data are largely derived from human-authored corpora, LLMs may inherit a range of human biases. Behavioral biases can lead to instability and uncertainty in decision-making, particularly when processing financial information. However, existing research on LLM bias has mainly focused on direct questioning or simplified, general-purpose settings, with limited consideration of the complex real-world financial environments and high-risk, context-sensitive, multilingual financial misinformation detection tasks (\mfmd). In this work, we propose \mfmdscen, a comprehensive benchmark for evaluating behavioral biases of LLMs in \mfmd across diverse economic scenarios. In collaboration with financial experts, we construct three types of complex financial scenarios: (i) role- and personality-based, (ii) role- and region-based, and (iii) role-based scenarios incorporating ethnicity and religious beliefs. We further develop a multilingual financial misinformation dataset covering English, Chinese, Greek, and Bengali. By integrating these scenarios with misinformation claims, \mfmdscen enables a systematic evaluation of 22 mainstream LLMs. Our findings reveal that pronounced behavioral biases persist across both commercial and open-source models. This project will be available at this https URL. 

---
# Lost in Execution: On the Multilingual Robustness of Tool Calling in Large Language Models 

**Authors**: Zheng Luo, T Pranav Kutralingam, Ogochukwu N Okoani, Wanpeng Xu, Hua Wei, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05366)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as agents that invoke external tools through structured function calls. While recent work reports strong tool-calling performance under standard English-centric evaluations, the robustness of tool calling under multilingual user interactions remains underexplored. In this work, we introduce MLCL, a diagnostic benchmark, and conduct a systematic evaluation of multilingual tool calling across Chinese, Hindi, and the low-resource language Igbo. Through fine-grained error analysis, we show that many failures occur despite correct intent understanding and tool selection. We identify parameter value language mismatch as a dominant failure mode, where models generate semantically appropriate parameter values in the user's language, violating language-invariant execution conventions. We further evaluate several inference-time system strategies and find that while these strategies substantially reduce language-induced execution errors, none of them can fully recover English-level performance. 

---
# The Table of Media Bias Elements: A sentence-level taxonomy of media bias types and propaganda techniques 

**Authors**: Tim Menzner, Jochen L. Leidner  

**Link**: [PDF](https://arxiv.org/pdf/2601.05358)  

**Abstract**: Public debates about "left-" or "right-wing" news overlook the fact that bias is usually conveyed by concrete linguistic manoeuvres that transcend any single political spectrum. We therefore shift the focus from where an outlet allegedly stands to how partiality is expressed in individual sentences. Drawing on 26,464 sentences collected from newsroom corpora, user submissions and our own browsing, we iteratively combine close-reading, interdisciplinary theory and pilot annotation to derive a fine-grained, sentence-level taxonomy of media bias and propaganda. The result is a two-tier schema comprising 38 elementary bias types, arranged in six functional families and visualised as a "table of media-bias elements". For each type we supply a definition, real-world examples, cognitive and societal drivers, and guidance for recognition. A quantitative survey of a random 155-sentence sample illustrates prevalence differences, while a cross-walk to the best-known NLP and communication-science taxonomies reveals substantial coverage gains and reduced ambiguity. 

---
# Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings 

**Authors**: Xiran Fan, Zhimeng Jiang, Chin-Chia Michael Yeh, Yuzhong Chen, Yingtong Dou, Menghai Pan, Yan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.05271)  

**Abstract**: The ubiquity of payment networks generates vast transactional data encoding rich consumer and merchant behavioral patterns. Recent foundation models for transaction analysis process tabular data sequentially but rely on index-based representations for categorical merchant fields, causing substantial semantic information loss by converting rich textual data into discrete tokens. While Large Language Models (LLMs) can address this limitation through superior semantic understanding, their computational overhead challenges real-time financial deployment. We introduce a hybrid framework that uses LLM-generated embeddings as semantic initializations for lightweight transaction models, balancing interpretability with operational efficiency. Our approach employs multi-source data fusion to enrich merchant categorical fields and a one-word constraint principle for consistent embedding generation across LLM architectures. We systematically address data quality through noise filtering and context-aware enrichment. Experiments on large-scale transaction datasets demonstrate significant performance improvements across multiple transaction understanding tasks. 

---
# Fusion Matters: Length-Aware Analysis of Positional-Encoding Fusion in Transformers 

**Authors**: Mohamed Amine Hallam, Kuo-Kun Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2601.05807)  

**Abstract**: Transformers require positional encodings to represent sequence order, yet most prior work focuses on designing new positional encodings rather than examining how positional information is fused with token embeddings. In this paper, we study whether the fusion mechanism itself affects performance, particularly in long-sequence settings. We conduct a controlled empirical study comparing three canonical fusion strategies--element-wise addition, concatenation with projection, and scalar gated fusion--under identical Transformer architectures, data splits, and random seeds. Experiments on three text classification datasets spanning short (AG News), medium (IMDB), and long (ArXiv) sequences show that fusion choice has negligible impact on short texts but produces consistent gains on long documents. To verify that these gains are structural rather than stochastic, we perform paired-seed analysis and cross-dataset comparison across sequence-length regimes. Additional experiments on the ArXiv dataset indicate that the benefit of learnable fusion generalizes across multiple positional encoding families. Finally, we explore a lightweight convolutional gating mechanism that introduces local inductive bias at the fusion level, evaluated on long documents only. Our results indicate that positional-encoding fusion is a non-trivial design choice for long-sequence Transformers and should be treated as an explicit modeling decision rather than a fixed default. 

---
# Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer 

**Authors**: Yifan Zhang, Wei Bi, Kechi Zhang, Dongming Jin, Jie Fu, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05770)  

**Abstract**: Algorithm extraction aims to synthesize executable programs directly from models trained on specific algorithmic tasks, enabling de novo algorithm discovery without relying on human-written code. However, extending this paradigm to Transformer is hindered by superposition, where entangled features encoded in overlapping directions obstruct the extraction of symbolic expressions. In this work, we propose the Discrete Transformer, an architecture explicitly engineered to bridge the gap between continuous representations and discrete symbolic logic. By enforcing a strict functional disentanglement, which constrains Numerical Attention to information routing and Numerical MLP to element-wise arithmetic, and employing temperature-annealed sampling, our method effectively facilitates the extraction of human-readable programs. Empirically, the Discrete Transformer not only achieves performance comparable to RNN-based baselines but crucially extends interpretability to continuous variable domains. Moreover, our analysis of the annealing process shows that the efficient discrete search undergoes a clear phase transition from exploration to exploitation. We further demonstrate that our method enables fine-grained control over synthesized programs by imposing inductive biases. Collectively, these findings establish the Discrete Transformer as a robust framework for demonstration-free algorithm discovery, offering a rigorous pathway toward Transformer interpretability. 

---
# PII-VisBench: Evaluating Personally Identifiable Information Safety in Vision Language Models Along a Continuum of Visibility 

**Authors**: G M Shahariar, Zabir Al Nazi, Md Olid Hasan Bhuiyan, Zhouxing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05739)  

**Abstract**: Vision Language Models (VLMs) are increasingly integrated into privacy-critical domains, yet existing evaluations of personally identifiable information (PII) leakage largely treat privacy as a static extraction task and ignore how a subject's online presence--the volume of their data available online--influences privacy alignment. We introduce PII-VisBench, a novel benchmark containing 4000 unique probes designed to evaluate VLM safety through the continuum of online presence. The benchmark stratifies 200 subjects into four visibility categories: high, medium, low, and zero--based on the extent and nature of their information available online. We evaluate 18 open-source VLMs (0.3B-32B) based on two key metrics: percentage of PII probing queries refused (Refusal Rate) and the fraction of non-refusal responses flagged for containing PII (Conditional PII Disclosure Rate). Across models, we observe a consistent pattern: refusals increase and PII disclosures decrease (9.10% high to 5.34% low) as subject visibility drops. We identify that models are more likely to disclose PII for high-visibility subjects, alongside substantial model-family heterogeneity and PII-type disparities. Finally, paraphrasing and jailbreak-style prompts expose attack and model-dependent failures, motivating visibility-aware safety evaluation and training interventions. 

---
# Logic-Parametric Neuro-Symbolic NLI: Controlling Logical Formalisms for Verifiable LLM Reasoning 

**Authors**: Ali Farjami, Luca Redondi, Marco Valentino  

**Link**: [PDF](https://arxiv.org/pdf/2601.05705)  

**Abstract**: Large language models (LLMs) and theorem provers (TPs) can be effectively combined for verifiable natural language inference (NLI). However, existing approaches rely on a fixed logical formalism, a feature that limits robustness and adaptability. We propose a logic-parametric framework for neuro-symbolic NLI that treats the underlying logic not as a static background, but as a controllable component. Using the LogiKEy methodology, we embed a range of classical and non-classical formalisms into higher-order logic (HOL), enabling a systematic comparison of inference quality, explanation refinement, and proof behavior. We focus on normative reasoning, where the choice of logic has significant implications. In particular, we compare logic-external approaches, where normative requirements are encoded via axioms, with logic-internal approaches, where normative patterns emerge from the logic's built-in structure. Extensive experiments demonstrate that logic-internal strategies can consistently improve performance and produce more efficient hybrid proofs for NLI. In addition, we show that the effectiveness of a logic is domain-dependent, with first-order logic favouring commonsense reasoning, while deontic and modal logics excel in ethical domains. Our results highlight the value of making logic a first-class, parametric element in neuro-symbolic architectures for more robust, modular, and adaptable reasoning. 

---
# Open World Knowledge Aided Single-Cell Foundation Model with Robust Cross-Modal Cell-Language Pre-training 

**Authors**: Haoran Wang, Xuanyi Zhang, Shuangsang Fang, Longke Ran, Ziqing Deng, Yong Zhang, Yuxiang Li, Shaoshuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05648)  

**Abstract**: Recent advancements in single-cell multi-omics, particularly RNA-seq, have provided profound insights into cellular heterogeneity and gene regulation. While pre-trained language model (PLM) paradigm based single-cell foundation models have shown promise, they remain constrained by insufficient integration of in-depth individual profiles and neglecting the influence of noise within multi-modal data. To address both issues, we propose an Open-world Language Knowledge-Aided Robust Single-Cell Foundation Model (OKR-CELL). It is built based on a cross-modal Cell-Language pre-training framework, which comprises two key innovations: (1) leveraging Large Language Models (LLMs) based workflow with retrieval-augmented generation (RAG) enriches cell textual descriptions using open-world knowledge; (2) devising a Cross-modal Robust Alignment (CRA) objective that incorporates sample reliability assessment, curriculum learning, and coupled momentum contrastive learning to strengthen the model's resistance to noisy data. After pretraining on 32M cell-text pairs, OKR-CELL obtains cutting-edge results across 6 evaluation tasks. Beyond standard benchmarks such as cell clustering, cell-type annotation, batch-effect correction, and few-shot annotation, the model also demonstrates superior performance in broader multi-modal applications, including zero-shot cell-type annotation and bidirectional cell-text retrieval. 

---
# Continual Pretraining on Encrypted Synthetic Data for Privacy-Preserving LLMs 

**Authors**: Honghao Liu, Xuhui Jiang, Chengjin Xu, Cehao Yang, Yiran Cheng, Lionel Ni, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.05635)  

**Abstract**: Preserving privacy in sensitive data while pretraining large language models on small, domain-specific corpora presents a significant challenge. In this work, we take an exploratory step toward privacy-preserving continual pretraining by proposing an entity-based framework that synthesizes encrypted training data to protect personally identifiable information (PII). Our approach constructs a weighted entity graph to guide data synthesis and applies deterministic encryption to PII entities, enabling LLMs to encode new knowledge through continual pretraining while granting authorized access to sensitive data through decryption keys. Our results on limited-scale datasets demonstrate that our pretrained models outperform base models and ensure PII security, while exhibiting a modest performance gap compared to models trained on unencrypted synthetic data. We further show that increasing the number of entities and leveraging graph-based synthesis improves model performance, and that encrypted models retain instruction-following capabilities with long retrieved contexts. We discuss the security implications and limitations of deterministic encryption, positioning this work as an initial investigation into the design space of encrypted data pretraining for privacy-preserving LLMs. Our code is available at this https URL. 

---
# SceneAlign: Aligning Multimodal Reasoning to Scene Graphs in Complex Visual Scenes 

**Authors**: Chuhan Wang, Xintong Li, Jennifer Yuntong Zhang, Junda Wu, Chengkai Huang, Lina Yao, Julian McAuley, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05600)  

**Abstract**: Multimodal large language models often struggle with faithful reasoning in complex visual scenes, where intricate entities and relations require precise visual grounding at each step. This reasoning unfaithfulness frequently manifests as hallucinated entities, mis-grounded relations, skipped steps, and over-specified reasoning. Existing preference-based approaches, typically relying on textual perturbations or answer-conditioned rationales, fail to address this challenge as they allow models to exploit language priors to bypass visual grounding. To address this, we propose SceneAlign, a framework that leverages scene graphs as structured visual information to perform controllable structural interventions. By identifying reasoning-critical nodes and perturbing them through four targeted strategies that mimic typical grounding failures, SceneAlign constructs hard negative rationales that remain linguistically plausible but are grounded in inaccurate visual facts. These contrastive pairs are used in Direct Preference Optimization to steer models toward fine-grained, structure-faithful reasoning. Across seven visual reasoning benchmarks, SceneAlign consistently improves answer accuracy and reasoning faithfulness, highlighting the effectiveness of grounding-aware alignment for multimodal reasoning. 

---
# RISE: Rule-Driven SQL Dialect Translation via Query Reduction 

**Authors**: Xudong Xie, Yuwei Zhang, Wensheng Dou, Yu Gao, Ziyu Cui, Jiansen Song, Rui Yang, Jun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.05579)  

**Abstract**: Translating SQL dialects across different relational database management systems (RDBMSs) is crucial for migrating RDBMS-based applications to the cloud. Traditional SQL dialect translation tools rely on manually-crafted rules, necessitating significant manual effort to support new RDBMSs and dialects. Although large language models (LLMs) can assist in translating SQL dialects, they often struggle with lengthy and complex SQL queries.
In this paper, we propose RISE, a novel LLM-based SQL dialect translation approach that can accurately handle lengthy and complex SQL queries. Given a complex source query $Q_c$ that contains a SQL dialect $d$, we first employ a dialect-aware query reduction technique to derive a simplified query $Q_{s}$ by removing $d$-irrelevant SQL elements from $Q_c$. Subsequently, we utilize LLMs to translate $Q_{s}$ into $Q_{s^{'}}$, and automatically extract the translation rule $r_d$ for dialect $d$ based on the relationship between $Q_{s}$ and $Q_{s^{'}}$. By applying $r_d$ to $Q_c$, we can effectively translate the dialect $d$ within $Q_c$, thereby bypassing the complexity of the source query $Q_c$. We evaluate RISE on two real-world benchmarks, i.e., TPC-DS and SQLProcBench, comparing its performance against both the traditional rule-based tools and the LLM-based approaches with respect to translation accuracy. RISE achieves accuracies of 97.98% on TPC-DS and 100% on SQLProcBench, outperforming the baselines by an average improvement of 24.62% and 238.41%, respectively. 

---
# WildSci: Advancing Scientific Reasoning from In-the-Wild Literature 

**Authors**: Tengxiao Liu, Deepak Nathani, Zekun Li, Kevin Yang, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05567)  

**Abstract**: Recent progress in large language model (LLM) reasoning has focused on domains like mathematics and coding, where abundant high-quality data and objective evaluation metrics are readily available. In contrast, progress in LLM reasoning models remains limited in scientific domains such as medicine and materials science due to limited dataset coverage and the inherent complexity of open-ended scientific questions. To address these challenges, we introduce WildSci, a new dataset of domain-specific science questions automatically synthesized from peer-reviewed literature, covering 9 scientific disciplines and 26 subdomains. By framing complex scientific reasoning tasks in a multiple-choice format, we enable scalable training with well-defined reward signals. We further apply reinforcement learning to finetune models on these data and analyze the resulting training dynamics, including domain-specific performance changes, response behaviors, and generalization trends. Experiments on a suite of scientific benchmarks demonstrate the effectiveness of our dataset and approach. We release WildSci to enable scalable and sustainable research in scientific reasoning, available at this https URL. 

---
# The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era 

**Authors**: Zhixian Zhao, Shuiyuan Wang, Guojian Li, Hongfei Xue, Chengyou Wang, Shuai Wang, Longshuai Xiao, Zihan Zhang, Hui Bu, Xin Xu, Xinsheng Wang, Hexin Liu, Eng Siong Chng, Hung-yi Lee, Haizhou Li, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.05564)  

**Abstract**: Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results. 

---
# Enabling Stroke-Level Structural Analysis of Hieroglyphic Scripts without Language-Specific Priors 

**Authors**: Fuwen Luo, Zihao Wan, Ziyue Wang, Yaluo Liu, Pau Tong Lin Xu, Xuanjia Qiao, Xiaolong Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05508)  

**Abstract**: Hieroglyphs, as logographic writing systems, encode rich semantic and cultural information within their internal structural composition. Yet, current advanced Large Language Models (LLMs) and Multimodal LLMs (MLLMs) usually remain structurally blind to this information. LLMs process characters as textual tokens, while MLLMs additionally view them as raw pixel grids. Both fall short to model the underlying logic of character strokes. Furthermore, existing structural analysis methods are often script-specific and labor-intensive. In this paper, we propose Hieroglyphic Stroke Analyzer (HieroSA), a novel and generalizable framework that enables MLLMs to automatically derive stroke-level structures from character bitmaps without handcrafted data. It transforms modern logographic and ancient hieroglyphs character images into explicit, interpretable line-segment representations in a normalized coordinate space, allowing for cross-lingual generalization. Extensive experiments demonstrate that HieroSA effectively captures character-internal structures and semantics, bypassing the need for language-specific priors. Experimental results highlight the potential of our work as a graphematics analysis tool for a deeper understanding of hieroglyphic scripts. View our code at this https URL. 

---
# Hi-ZFO: Hierarchical Zeroth- and First-Order LLM Fine-Tuning via Importance-Guided Tensor Selection 

**Authors**: Feihu Jin, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05501)  

**Abstract**: Fine-tuning large language models (LLMs) using standard first-order (FO) optimization often drives training toward sharp, poorly generalizing minima. Conversely, zeroth-order (ZO) methods offer stronger exploratory behavior without relying on explicit gradients, yet suffer from slow convergence. More critically, our analysis reveals that in generative tasks, the vast output and search space significantly amplify estimation variance, rendering ZO methods both noisy and inefficient. To address these challenges, we propose \textbf{Hi-ZFO} (\textbf{Hi}erarchical \textbf{Z}eroth- and \textbf{F}irst-\textbf{O}rder optimization), a hybrid framework designed to synergize the precision of FO gradients with the exploratory capability of ZO estimation. Hi-ZFO adaptively partitions the model through layer-wise importance profiling, applying precise FO updates to critical layers while leveraging ZO optimization for less sensitive ones. Notably, ZO in Hi-ZFO is not merely a memory-saving surrogate; it is intentionally introduced as a source of "beneficial stochasticity" to help the model escape the local minima where pure FO optimization tends to stagnate. Validated across diverse generative, mathematical, and code reasoning tasks, Hi-ZFO consistently achieves superior performance while significantly reducing the training time. These results demonstrate the effectiveness of hierarchical hybrid optimization for LLM fine-tuning. 

---
# MMViR: A Multi-Modal and Multi-Granularity Representation for Long-range Video Understanding 

**Authors**: Zizhong Li, Haopeng Zhang, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05495)  

**Abstract**: Long videos, ranging from minutes to hours, present significant challenges for current Multi-modal Large Language Models (MLLMs) due to their complex events, diverse scenes, and long-range dependencies. Direct encoding of such videos is computationally too expensive, while simple video-to-text conversion often results in redundant or fragmented content. To address these limitations, we introduce MMViR, a novel multi-modal, multi-grained structured representation for long video understanding. MMViR identifies key turning points to segment the video and constructs a three-level description that couples global narratives with fine-grained visual details. This design supports efficient query-based retrieval and generalizes well across various scenarios. Extensive evaluations across three tasks, including QA, summarization, and retrieval, show that MMViR outperforms the prior strongest method, achieving a 19.67% improvement in hour-long video understanding while reducing processing latency to 45.4% of the original. 

---
# MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization 

**Authors**: Jiefu Ou, Sapana Chaudhary, Kaj Bostrom, Nathaniel Weir, Shuai Zhang, Huzefa Rangwala, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2601.05475)  

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities in general coding tasks but encounter two key challenges when optimizing code: (i) the complexity of writing optimized code (such as performant CUDA kernels and competition-level CPU code) requires expertise in systems, algorithms and specific languages and (ii) requires interpretation of performance metrics like timing and device utilization beyond binary correctness. In this work, we explore inference-time search algorithms that guide the LLM to discover better solutions through iterative refinement based on execution feedback. Our approach, called MaxCode unifies existing search methods under a max-reward reinforcement learning framework, making the observation and action-value functions modular for modification. To enhance the observation space, we integrate a natural language critique model that converts raw execution feedback into diagnostic insights about errors and performance bottlenecks, and the best-discounted reward seen so far. Together, these provide richer input to the code proposal function. To improve exploration during search, we train a generative reward-to-go model using action values from rollouts to rerank potential solutions. Testing on the KernelBench (CUDA) and PIE (C++) optimization benchmarks shows that MaxCode improves optimized code performance compared to baselines, achieving 20.3% and 10.1% relative improvements in absolute speedup value and relative speedup ranking, respectively. 

---
# ROAP: A Reading-Order and Attention-Prior Pipeline for Optimizing Layout Transformers in Key Information Extraction 

**Authors**: Tingwei Xie, Jinxin He, Yonghong Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.05470)  

**Abstract**: The efficacy of Multimodal Transformers in visually-rich document understanding (VrDU) is critically constrained by two inherent limitations: the lack of explicit modeling for logical reading order and the interference of visual tokens that dilutes attention on textual semantics.
To address these challenges, this paper presents ROAP, a lightweight and architecture-agnostic pipeline designed to optimize attention distributions in Layout Transformers without altering their pre-trained backbones.
The proposed pipeline first employs an Adaptive-XY-Gap (AXG-Tree) to robustly extract hierarchical reading sequences from complex layouts. These sequences are then integrated into the attention mechanism via a Reading-Order-Aware Relative Position Bias (RO-RPB). Furthermore, a Textual-Token Sub-block Attention Prior (TT-Prior) is introduced to adaptively suppress visual noise and enhance fine-grained text-text interactions.
Extensive experiments on the FUNSD and CORD benchmarks demonstrate that ROAP consistently improves the performance of representative backbones, including LayoutLMv3 and GeoLayoutLM.
These findings confirm that explicitly modeling reading logic and regulating modality interference are critical for robust document understanding, offering a scalable solution for complex layout analysis. The implementation code will be released at this https URL. 

---
# RingSQL: Generating Synthetic Data with Schema-Independent Templates for Text-to-SQL Reasoning Models 

**Authors**: Marko Sterbentz, Kevin Cushing, Cameron Barrie, Kristian J. Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2601.05451)  

**Abstract**: Recent advances in text-to-SQL systems have been driven by larger models and improved datasets, yet progress is still limited by the scarcity of high-quality training data. Manual data creation is expensive, and existing synthetic methods trade off reliability and scalability. Template-based approaches ensure correct SQL but require schema-specific templates, while LLM-based generation scales easily but lacks quality and correctness guarantees. We introduce RingSQL, a hybrid data generation framework that combines schema-independent query templates with LLM-based paraphrasing of natural language questions. This approach preserves SQL correctness across diverse schemas while providing broad linguistic variety. In our experiments, we find that models trained using data produced by RingSQL achieve an average gain in accuracy of +2.3% across six text-to-SQL benchmarks when compared to models trained on other synthetic data. We make our code available at this https URL. 

---
# Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization 

**Authors**: Yuxiang Ji, Yong Wang, Ziyu Ma, Yiming Hu, Hailang Huang, Xuecai Hu, Guanhua Chen, Liaoni Wu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05432)  

**Abstract**: The image geolocalization task aims to predict the location where an image was taken anywhere on Earth using visual clues. Existing large vision-language model (LVLM) approaches leverage world knowledge, chain-of-thought reasoning, and agentic capabilities, but overlook a common strategy used by humans -- using maps. In this work, we first equip the model \textit{Thinking with Map} ability and formulate it as an agent-in-the-map loop. We develop a two-stage optimization scheme for it, including agentic reinforcement learning (RL) followed by parallel test-time scaling (TTS). The RL strengthens the agentic capability of model to improve sampling efficiency, and the parallel TTS enables the model to explore multiple candidate paths before making the final prediction, which is crucial for geolocalization. To evaluate our method on up-to-date and in-the-wild images, we further present MAPBench, a comprehensive geolocalization training and evaluation benchmark composed entirely of real-world images. Experimental results show that our method outperforms existing open- and closed-source models on most metrics, specifically improving Acc@500m from 8.0\% to 22.1\% compared to \textit{Gemini-3-Pro} with Google Search/Map grounded mode. 

---
# Conformity and Social Impact on AI Agents 

**Authors**: Alessandro Bellina, Giordano De Marzo, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2601.05384)  

**Abstract**: As AI agents increasingly operate in multi-agent environments, understanding their collective behavior becomes critical for predicting the dynamics of artificial societies. This study examines conformity, the tendency to align with group opinions under social pressure, in large multimodal language models functioning as AI agents. By adapting classic visual experiments from social psychology, we investigate how AI agents respond to group influence as social actors. Our experiments reveal that AI agents exhibit a systematic conformity bias, aligned with Social Impact Theory, showing sensitivity to group size, unanimity, task difficulty, and source characteristics. Critically, AI agents achieving near-perfect performance in isolation become highly susceptible to manipulation through social influence. This vulnerability persists across model scales: while larger models show reduced conformity on simple tasks due to improved capabilities, they remain vulnerable when operating at their competence boundary. These findings reveal fundamental security vulnerabilities in AI agent decision-making that could enable malicious manipulation, misinformation campaigns, and bias propagation in multi-agent systems, highlighting the urgent need for safeguards in collective AI deployments. 

---
# The Persona Paradox: Medical Personas as Behavioral Priors in Clinical Language Models 

**Authors**: Tassallah Abdullahi, Shrestha Ghosh, Hamish S Fraser, Daniel León Tramontini, Adeel Abbasi, Ghada Bourjeily, Carsten Eickhoff, Ritambhara Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.05376)  

**Abstract**: Persona conditioning can be viewed as a behavioral prior for large language models (LLMs) and is often assumed to confer expertise and improve safety in a monotonic manner. However, its effects on high-stakes clinical decision-making remain poorly characterized. We systematically evaluate persona-based control in clinical LLMs, examining how professional roles (e.g., Emergency Department physician, nurse) and interaction styles (bold vs.\ cautious) influence behavior across models and medical tasks. We assess performance on clinical triage and patient-safety tasks using multidimensional evaluations that capture task accuracy, calibration, and safety-relevant risk behavior. We find systematic, context-dependent, and non-monotonic effects: Medical personas improve performance in critical care tasks, yielding gains of up to $\sim+20\%$ in accuracy and calibration, but degrade performance in primary-care settings by comparable margins. Interaction style modulates risk propensity and sensitivity, but it's highly model-dependent. While aggregated LLM-judge rankings favor medical over non-medical personas in safety-critical cases, we found that human clinicians show moderate agreement on safety compliance (average Cohen's $\kappa = 0.43$) but indicate a low confidence in 95.9\% of their responses on reasoning quality. Our work shows that personas function as behavioral priors that introduce context-dependent trade-offs rather than guarantees of safety or expertise. The code is available at this https URL\_Paradox. 

---
# TIME: Temporally Intelligent Meta-reasoning Engine for Context Triggered Explicit Reasoning 

**Authors**: Susmit Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.05300)  

**Abstract**: Reasoning oriented large language models often expose explicit "thinking" as long, turn-global traces at the start of every response, either always on or toggled externally at inference time. While useful for arithmetic, programming, and problem solving, this design is costly, blurs claim level auditability, and cannot re-trigger explicit reasoning once the model begins presenting. Dialogue models are also largely blind to temporal structure, treating replies after seconds and replies after weeks as equivalent unless time is stated in text. We introduce TIME, the Temporally Intelligent Meta-reasoning Engine, a behavioral alignment framework that treats explicit reasoning as a context sensitive resource driven by discourse and temporal cues. TIME augments dialogue with optional ISO 8601 <time> tags, tick turns that represent silent gaps, and short <think> blocks that can appear anywhere in a reply. A four-phase curriculum including a small, maximally diverse full-batch alignment step trains Qwen3 dense models to invoke brief, in-place reasoning bursts and keep user facing text compact. We evaluate with TIMEBench, a temporally grounded dialogue benchmark probing chronology, commonsense under gaps and offsets, anomaly detection, and continuity. Across 4B to 32B scales, TIME improves TIMEBench scores over base Qwen3 in both thinking and no-thinking modes while reducing reasoning tokens by about an order of magnitude. Our training data and code are available at this https URL and TIMEBench is available at this https URL 

---
# Transforming User Defined Criteria into Explainable Indicators with an Integrated LLM AHP System 

**Authors**: Geonwoo Bang, Dongho Kim, Moohong Min  

**Link**: [PDF](https://arxiv.org/pdf/2601.05267)  

**Abstract**: Evaluating complex texts across domains requires converting user defined criteria into quantitative, explainable indicators, which is a persistent challenge in search and recommendation systems. Single prompt LLM evaluations suffer from complexity and latency issues, while criterion specific decomposition approaches rely on naive averaging or opaque black-box aggregation methods. We present an interpretable aggregation framework combining LLM scoring with the Analytic Hierarchy Process. Our method generates criterion specific scores via LLM as judge, measures discriminative power using Jensen Shannon distance, and derives statistically grounded weights through AHP pairwise comparison matrices. Experiments on Amazon review quality assessment and depression related text scoring demonstrate that our approach achieves high explainability and operational efficiency while maintaining comparable predictive power, making it suitable for real time latency sensitive web services. 

---
# Retrieval-Augmented Multi-LLM Ensemble for Industrial Part Specification Extraction 

**Authors**: Muzakkiruddin Ahmed Mohammed, John R. Talburt, Leon Claasssens, Adriaan Marais  

**Link**: [PDF](https://arxiv.org/pdf/2601.05266)  

**Abstract**: Industrial part specification extraction from unstructured text remains a persistent challenge in manufacturing, procurement, and maintenance, where manual processing is both time-consuming and error-prone. This paper introduces a retrieval-augmented multi-LLM ensemble framework that orchestrates nine state-of-the-art Large Language Models (LLMs) within a structured three-phase pipeline. RAGsemble addresses key limitations of single-model systems by combining the complementary strengths of model families including Gemini (2.0, 2.5, 1.5), OpenAI (GPT-4o, o4-mini), Mistral Large, and Gemma (1B, 4B, 3n-e4b), while grounding outputs in factual data using FAISS-based semantic retrieval. The system architecture consists of three stages: (1) parallel extraction by diverse LLMs, (2) targeted research augmentation leveraging high-performing models, and (3) intelligent synthesis with conflict resolution and confidence-aware scoring. RAG integration provides real-time access to structured part databases, enabling the system to validate, refine, and enrich outputs through similarity-based reference retrieval. Experimental results using real industrial datasets demonstrate significant gains in extraction accuracy, technical completeness, and structured output quality compared to leading single-LLM baselines. Key contributions include a scalable ensemble architecture for industrial domains, seamless RAG integration throughout the pipeline, comprehensive quality assessment mechanisms, and a production-ready solution suitable for deployment in knowledge-intensive manufacturing environments. 

---
# Cross-Document Topic-Aligned Chunking for Retrieval-Augmented Generation 

**Authors**: Mile Stankovic  

**Link**: [PDF](https://arxiv.org/pdf/2601.05265)  

**Abstract**: Chunking quality determines RAG system performance. Current methods partition documents individually, but complex queries need information scattered across multiple sources: the knowledge fragmentation problem. We introduce Cross-Document Topic-Aligned (CDTA) chunking, which reconstructs knowledge at the corpus level. It first identifies topics across documents, maps segments to each topic, and synthesizes them into unified chunks.
On HotpotQA multi-hop reasoning, our method reached 0.93 faithfulness versus 0.83 for contextual retrieval and 0.78 for semantic chunking, a 12% improvement over current industry best practice (p < 0.05). On UAE Legal texts, it reached 0.94 faithfulness with 0.93 citation accuracy. At k = 3, it maintains 0.91 faithfulness while semantic methods drop to 0.68, with a single CDTA chunk containing information requiring multiple traditional fragments.
Indexing costs are higher, but synthesis produces information-dense chunks that reduce query-time retrieval needs. For high-query-volume applications with distributed knowledge, cross-document synthesis improves measurably over within-document optimization. 

---
# LLM2IR: simple unsupervised contrastive learning makes long-context LLM great retriever 

**Authors**: Xiaocong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05262)  

**Abstract**: Modern dense information retrieval (IR) models usually rely on costly large-scale pretraining. In this paper, we introduce LLM2IR, an efficient unsupervised contrastive learning framework to convert any decoder-only large language model (LLM) to an information retrieval model. Despite its simplicity, the effectiveness is proven among different LLMs on multiple IR benchmarks including LoCo, LongEmbed and BEIR. We also find that models with a longer context length tend to have a stronger IR capacity by comparing task performances of models in the same model family. Our work not only provides an effective way to build IR models on the state-of-the-art LLMs, but also shed light on the relationship between information retrieval ability and model context length, which helps the design of better information retrievers. 

---
# Quantifying Document Impact in RAG-LLMs 

**Authors**: Armin Gerami, Kazem Faghih, Ramani Duraiswami  

**Link**: [PDF](https://arxiv.org/pdf/2601.05260)  

**Abstract**: Retrieval Augmented Generation (RAG) enhances Large Language Models (LLMs) by connecting them to external knowledge, improving accuracy and reducing outdated information. However, this introduces challenges such as factual inconsistencies, source conflicts, bias propagation, and security vulnerabilities, which undermine the trustworthiness of RAG systems. A key gap in current RAG evaluation is the lack of a metric to quantify the contribution of individual retrieved documents to the final output. To address this, we introduce the Influence Score (IS), a novel metric based on Partial Information Decomposition that measures the impact of each retrieved document on the generated response. We validate IS through two experiments. First, a poison attack simulation across three datasets demonstrates that IS correctly identifies the malicious document as the most influential in $86\%$ of cases. Second, an ablation study shows that a response generated using only the top-ranked documents by IS is consistently judged more similar to the original response than one generated from the remaining documents. These results confirm the efficacy of IS in isolating and quantifying document influence, offering a valuable tool for improving the transparency and reliability of RAG systems. 

---
# Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring 

**Authors**: Eirini Baltzi, Tilemachos Moumouris, Athena Psalta, Vasileios Tsironis, Konstantinos Karantzalos  

**Link**: [PDF](https://arxiv.org/pdf/2601.05256)  

**Abstract**: Inland water monitoring is vital for safeguarding public health and ecosystems, enabling timely interventions to mitigate risks. Existing methods often address isolated sub-problems such as cyanobacteria, chlorophyll, or other quality indicators separately. NAIAD introduces an agentic AI assistant that leverages Large Language Models (LLMs) and external analytical tools to deliver a holistic solution for inland water monitoring using Earth Observation (EO) data. Designed for both experts and non-experts, NAIAD provides a single-prompt interface that translates natural-language queries into actionable insights. Through Retrieval-Augmented Generation (RAG), LLM reasoning, external tool orchestration, computational graph execution, and agentic reflection, it retrieves and synthesizes knowledge from curated sources to produce tailored reports. The system integrates diverse tools for weather data, Sentinel-2 imagery, remote-sensing index computation (e.g., NDCI), chlorophyll-a estimation, and established platforms such as CyFi. Performance is evaluated using correctness and relevancy metrics, achieving over 77% and 85% respectively on a dedicated benchmark covering multiple user-expertise levels. Preliminary results show strong adaptability and robustness across query types. An ablation study on LLM backbones further highlights Gemma 3 (27B) and Qwen 2.5 (14B) as offering the best balance between computational efficiency and reasoning performance. 

---
# TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation 

**Authors**: Wenbiao Tao, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.05254)  

**Abstract**: Retrieval-Augmented Generation enhances language models by retrieving external knowledge to support informed and grounded responses. However, traditional RAG methods rely on fragment-level retrieval, limiting their ability to address query-focused summarization queries. GraphRAG introduces a graph-based paradigm for global knowledge reasoning, yet suffers from inefficiencies in information extraction, costly resource consumption, and poor adaptability to incremental updates. To overcome these limitations, we propose TagRAG, a tag-guided hierarchical knowledge graph RAG framework designed for efficient global reasoning and scalable graph maintenance. TagRAG introduces two key components: (1) Tag Knowledge Graph Construction, which extracts object tags and their relationships from documents and organizes them into hierarchical domain tag chains for structured knowledge representation, and (2) Tag-Guided Retrieval-Augmented Generation, which retrieves domain-centric tag chains to localize and synthesize relevant knowledge during inference. This design significantly adapts to smaller language models, improves retrieval granularity, and supports efficient knowledge increment. Extensive experiments on UltraDomain datasets spanning Agriculture, Computer Science, Law, and cross-domain settings demonstrate that TagRAG achieves an average win rate of 95.41\% against baselines while maintaining about 14.6x construction and 1.9x retrieval efficiency compared with GraphRAG. 

---
