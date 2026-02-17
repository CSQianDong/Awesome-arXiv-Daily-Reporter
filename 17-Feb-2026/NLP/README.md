# Text Style Transfer with Parameter-efficient LLM Finetuning and Round-trip Translation 

**Authors**: Ruoxi Liu, Philipp Koehn  

**Link**: [PDF](https://arxiv.org/pdf/2602.15013)  

**Abstract**: This paper proposes a novel method for Text Style Transfer (TST) based on parameter-efficient fine-tuning of Large Language Models (LLMs). Addressing the scarcity of parallel corpora that map between styles, the study employs roundtrip translation to synthesize such parallel datasets from monolingual corpora. This approach creates 'neutralized' text devoid of stylistic attributes, essentially creating a shared input style at training-time and inference-time. Experimental results demonstrate consistent superiority of this method over zero-shot prompting and fewshot ICL techniques measured by BLEU scores and style accuracy scores across four investigated domains. Furthermore, the integration of retrieval-augmented generation (RAG) for terminology and name knowledge enhances robustness and stylistic consistency. 

---
# Cold-Start Personalization via Training-Free Priors from Structured World Models 

**Authors**: Avinandan Bose, Shuyue Stella Li, Faeze Brahman, Pang Wei Koh, Simon Shaolei Du, Yulia Tsvetkov, Maryam Fazel, Lin Xiao, Asli Celikyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2602.15012)  

**Abstract**: Cold-start personalization requires inferring user preferences through interaction when no user-specific historical data is available. The core challenge is a routing problem: each task admits dozens of preference dimensions, yet individual users care about only a few, and which ones matter depends on who is asking. With a limited question budget, asking without structure will miss the dimensions that matter. Reinforcement learning is the natural formulation, but in multi-turn settings its terminal reward fails to exploit the factored, per-criterion structure of preference data, and in practice learned policies collapse to static question sequences that ignore user responses. We propose decomposing cold-start elicitation into offline structure learning and online Bayesian inference. Pep (Preference Elicitation with Priors) learns a structured world model of preference correlations offline from complete profiles, then performs training-free Bayesian inference online to select informative questions and predict complete preference profiles, including dimensions never asked about. The framework is modular across downstream solvers and requires only simple belief models. Across medical, mathematical, social, and commonsense reasoning, Pep achieves 80.8% alignment between generated responses and users' stated preferences versus 68.5% for RL, with 3-5x fewer interactions. When two users give different answers to the same question, Pep changes its follow-up 39-62% of the time versus 0-28% for RL. It does so with ~10K parameters versus 8B for RL, showing that the bottleneck in cold-start elicitation is the capability to exploit the factored structure of preference data. 

---
# Learning User Interests via Reasoning and Distillation for Cross-Domain News Recommendation 

**Authors**: Mengdan Zhu, Yufan Zhao, Tao Di, Yulan Yan, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.15005)  

**Abstract**: News recommendation plays a critical role in online news platforms by helping users discover relevant content. Cross-domain news recommendation further requires inferring user's underlying information needs from heterogeneous signals that often extend beyond direct news consumption. A key challenge lies in moving beyond surface-level behaviors to capture deeper, reusable user interests while maintaining scalability in large-scale production systems. In this paper, we present a reinforcement learning framework that trains large language models to generate high-quality lists of interest-driven news search queries from cross-domain user signals. We formulate query-list generation as a policy optimization problem and employ GRPO with multiple reward signals. We systematically study two compute dimensions: inference-time sampling and model capacity, and empirically observe consistent improvements with increased compute that exhibit scaling-like behavior. Finally, we perform on-policy distillation to transfer the learned policy from a large, compute-intensive teacher to a compact student model suitable for scalable deployment. Extensive offline experiments, ablation studies and large-scale online A/B tests in a production news recommendation system demonstrate consistent gains in both interest modeling quality and downstream recommendation performance. 

---
# Counterfactual Fairness Evaluation of LLM-Based Contact Center Agent Quality Assurance System 

**Authors**: Kawin Mayilvaghanan, Siddhant Gupta, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.14970)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in contact-center Quality Assurance (QA) to automate agent performance evaluation and coaching feedback. While LLMs offer unprecedented scalability and speed, their reliance on web-scale training data raises concerns regarding demographic and behavioral biases that may distort workforce assessment. We present a counterfactual fairness evaluation of LLM-based QA systems across 13 dimensions spanning three categories: Identity, Context, and Behavioral Style. Fairness is quantified using the Counterfactual Flip Rate (CFR), the frequency of binary judgment reversals, and the Mean Absolute Score Difference (MASD), the average shift in coaching or confidence scores across counterfactual pairs. Evaluating 18 LLMs on 3,000 real-world contact center transcripts, we find systematic disparities, with CFR ranging from 5.4% to 13.0% and consistent MASD shifts across confidence, positive, and improvement scores. Larger, more strongly aligned models show lower unfairness, though fairness does not track accuracy. Contextual priming of historical performance induces the most severe degradations (CFR up to 16.4%), while implicit linguistic identity cues remain a persistent bias source. Finally, we analyze the efficacy of fairness-aware prompting, finding that explicit instructions yield only modest improvements in evaluative consistency. Our findings underscore the need for standardized fairness auditing pipelines prior to deploying LLMs in high-stakes workforce evaluation. 

---
# Tool-Aware Planning in Contact Center AI: Evaluating LLMs through Lineage-Guided Query Decomposition 

**Authors**: Varun Nathan, Shreyas Guha, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.14955)  

**Abstract**: We present a domain-grounded framework and benchmark for tool-aware plan generation in contact centers, where answering a query for business insights, our target use case, requires decomposing it into executable steps over structured tools (Text2SQL (T2S)/Snowflake) and unstructured tools (RAG/transcripts) with explicit depends_on for parallelism. Our contributions are threefold: (i) a reference-based plan evaluation framework operating in two modes - a metric-wise evaluator spanning seven dimensions (e.g., tool-prompt alignment, query adherence) and a one-shot evaluator; (ii) a data curation methodology that iteratively refines plans via an evaluator->optimizer loop to produce high-quality plan lineages (ordered plan revisions) while reducing manual effort; and (iii) a large-scale study of 14 LLMs across sizes and families for their ability to decompose queries into step-by-step, executable, and tool-assigned plans, evaluated under prompts with and without lineage. Empirically, LLMs struggle on compound queries and on plans exceeding 4 steps (typically 5-15); the best total metric score reaches 84.8% (Claude-3-7-Sonnet), while the strongest one-shot match rate at the "A+" tier (Extremely Good, Very Good) is only 49.75% (o3-mini). Plan lineage yields mixed gains overall but benefits several top models and improves step executability for many. Our results highlight persistent gaps in tool-understanding, especially in tool-prompt alignment and tool-usage completeness, and show that shorter, simpler plans are markedly easier. The framework and findings provide a reproducible path for assessing and improving agentic planning with tools for answering data-analysis queries in contact-center settings. 

---
# BFS-PO: Best-First Search for Large Reasoning Models 

**Authors**: Fiorenzo Parascandolo, Wenhui Tan, Enver Sangineto, Ruihua Song, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2602.14917)  

**Abstract**: Large Reasoning Models (LRMs) such as OpenAI o1 and DeepSeek-R1 have shown excellent performance in reasoning tasks using long reasoning chains. However, this has also led to a significant increase of computational costs and the generation of verbose output, a phenomenon known as overthinking. The tendency to overthinking is often exacerbated by Reinforcement Learning (RL) algorithms such as GRPO/DAPO. In this paper, we propose BFS-PO, an RL algorithm which alleviates this problem using a Best-First Search exploration strategy. Specifically, BFS-PO looks for the shortest correct answer using a backtracking mechanism based on maximum entropy nodes. By generating progressively shorter responses during training, BFS-PO learns to produce concise reasoning chains. Using different benchmarks and base LRMs, we show that BFS-PO can simultaneously increase the LRM accuracy and shorten its answers. 

---
# Testimole-Conversational: A 30-Billion-Word Italian Discussion Board Corpus (1996-2024) for Language Modeling and Sociolinguistic Research 

**Authors**: Matteo Rinaldi, Rossella Varvara, Viviana Patti  

**Link**: [PDF](https://arxiv.org/pdf/2602.14819)  

**Abstract**: We present "Testimole-conversational" a massive collection of discussion boards messages in the Italian language. The large size of the corpus, more than 30B word-tokens (1996-2024), renders it an ideal dataset for native Italian Large Language Models'pre-training. Furthermore, discussion boards' messages are a relevant resource for linguistic as well as sociological analysis. The corpus captures a rich variety of computer-mediated communication, offering insights into informal written Italian, discourse dynamics, and online social interaction in wide time span. Beyond its relevance for NLP applications such as language modelling, domain adaptation, and conversational analysis, it also support investigations of language variation and social phenomena in digital communication. The resource will be made freely available to the research community. 

---
# Physical Commonsense Reasoning for Lower-Resourced Languages and Dialects: a Study on Basque 

**Authors**: Jaione Bengoetxea, Itziar Gonzalez-Dios, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2602.14812)  

**Abstract**: Physical commonsense reasoning represents a fundamental capability of human intelligence, enabling individuals to understand their environment, predict future events, and navigate physical spaces. Recent years have witnessed growing interest in reasoning tasks within Natural Language Processing (NLP). However, no prior research has examined the performance of Large Language Models (LLMs) on non-question-answering (non-QA) physical commonsense reasoning tasks in low-resource languages such as Basque. Taking the Italian GITA as a starting point, this paper addresses this gap by presenting BasPhyCo, the first non-QA physical commonsense reasoning dataset for Basque, available in both standard and dialectal variants. We evaluate model performance across three hierarchical levels of commonsense understanding: (1) distinguishing between plausible and implausible narratives (accuracy), (2) identifying the conflicting element that renders a narrative implausible (consistency), and (3) determining the specific physical state that creates the implausibility (verifiability). These tasks were assessed using multiple multilingual LLMs as well as models pretrained specifically for Italian and Basque. Results indicate that, in terms of verifiability, LLMs exhibit limited physical commonsense capabilities in low-resource languages such as Basque, especially when processing dialectal variants. 

---
# Overthinking Loops in Agents: A Structural Risk via MCP Tools 

**Authors**: Yohan Lee, Jisoo Jang, Seoyeon Choi, Sangyeop Kim, Seungtaek Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.14798)  

**Abstract**: Tool-using LLM agents increasingly coordinate real workloads by selecting and chaining third-party tools based on text-visible metadata such as tool names, descriptions, and return messages. We show that this convenience creates a supply-chain attack surface: a malicious MCP tool server can be co-registered alongside normal tools and induce overthinking loops, where individually trivial or plausible tool calls compose into cyclic trajectories that inflate end-to-end tokens and latency without any single step looking abnormal. We formalize this as a structural overthinking attack, distinguishable from token-level verbosity, and implement 14 malicious tools across three servers that trigger repetition, forced refinement, and distraction. Across heterogeneous registries and multiple tool-capable models, the attack causes severe resource amplification (up to $142.4\times$ tokens) and can degrade task outcomes. Finally, we find that decoding-time concision controls do not reliably prevent loop induction, suggesting defenses should reason about tool-call structure rather than tokens alone. 

---
# A Geometric Analysis of Small-sized Language Model Hallucinations 

**Authors**: Emanuele Ricco, Elia Onofri, Lorenzo Cima, Stefano Cresci, Roberto Di Pietro  

**Link**: [PDF](https://arxiv.org/pdf/2602.14778)  

**Abstract**: Hallucinations -- fluent but factually incorrect responses -- pose a major challenge to the reliability of language models, especially in multi-step or agentic settings.
This work investigates hallucinations in small-sized LLMs through a geometric perspective, starting from the hypothesis that when models generate multiple responses to the same prompt, genuine ones exhibit tighter clustering in the embedding space, we prove this hypothesis and, leveraging this geometrical insight, we also show that it is possible to achieve a consistent level of separability. This latter result is used to introduce a label-efficient propagation method that classifies large collections of responses from just 30-50 annotations, achieving F1 scores above 90%.
Our findings, framing hallucinations from a geometric perspective in the embedding space, complement traditional knowledge-centric and single-response evaluation paradigms, paving the way for further research. 

---
# Emergently Misaligned Language Models Show Behavioral Self-Awareness That Shifts With Subsequent Realignment 

**Authors**: Laurène Vaugrante, Anietta Weckauff, Thilo Hagendorff  

**Link**: [PDF](https://arxiv.org/pdf/2602.14777)  

**Abstract**: Recent research has demonstrated that large language models (LLMs) fine-tuned on incorrect trivia question-answer pairs exhibit toxicity - a phenomenon later termed "emergent misalignment". Moreover, research has shown that LLMs possess behavioral self-awareness - the ability to describe learned behaviors that were only implicitly demonstrated in training data. Here, we investigate the intersection of these phenomena. We fine-tune GPT-4.1 models sequentially on datasets known to induce and reverse emergent misalignment and evaluate whether the models are self-aware of their behavior transitions without providing in-context examples. Our results show that emergently misaligned models rate themselves as significantly more harmful compared to their base model and realigned counterparts, demonstrating behavioral self-awareness of their own emergent misalignment. Our findings show that behavioral self-awareness tracks actual alignment states of models, indicating that models can be queried for informative signals about their own safety. 

---
# Multi-Agent Comedy Club: Investigating Community Discussion Effects on LLM Humor Generation 

**Authors**: Shiwei Hong, Lingyao Li, Ethan Z. Rong, Chenxinran Shen, Zhicong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.14770)  

**Abstract**: Prior work has explored multi-turn interaction and feedback for LLM writing, but evaluations still largely center on prompts and localized feedback, leaving persistent public reception in online communities underexamined. We test whether broadcast community discussion improves stand-up comedy writing in a controlled multi-agent sandbox: in the discussion condition, critic and audience threads are recorded, filtered, stored as social memory, and later retrieved to condition subsequent generations, whereas the baseline omits discussion. Across 50 rounds (250 paired monologues) judged by five expert annotators using A/B preference and a 15-item rubric, discussion wins 75.6% of instances and improves Craft/Clarity ({\Delta} = 0.440) and Social Response ({\Delta} = 0.422), with occasional increases in aggressive humor. 

---
# Unlocking Reasoning Capability on Machine Translation in Large Language Models 

**Authors**: Sara Rajaee, Sebastian Vincent, Alexandre Berard, Marzieh Fadaee, Kelly Marchisio, Tom Kocmi  

**Link**: [PDF](https://arxiv.org/pdf/2602.14763)  

**Abstract**: Reasoning-oriented large language models (RLMs) achieve strong gains on tasks such as mathematics and coding by generating explicit intermediate reasoning. However, their impact on machine translation (MT) remains underexplored. We systematically evaluate several open- and closed-weights RLMs on the WMT24++ benchmark and find that enabling explicit reasoning consistently degrades translation quality across languages and models. Analysis reveals that MT reasoning traces are highly linear, lacking revision, self-correction and exploration of alternative translations, which limits their usefulness. Furthermore, injecting higher-quality reasoning traces from stronger models does not reliably improve weaker models' performance. To address this mismatch, we propose a structured reasoning framework tailored to translation, based on multi-step drafting, adequacy refinement, fluency improvement, and selective iterative revision. We curate a synthetic dataset of dynamic structured reasoning traces and post-train a large reasoning model on this data. Experiments show significant improvements over standard translation fine-tuning and injected generic reasoning baselines. Our findings demonstrate that reasoning must be task-structured to benefit MT. 

---
# Residual Connections and the Causal Shift: Uncovering a Structural Misalignment in Transformers 

**Authors**: Jonathan Lys, Vincent Gripon, Bastien Pasdeloup, Lukas Mauch, Fabien Cardinaux, Ghouthi Boukli Hacene  

**Link**: [PDF](https://arxiv.org/pdf/2602.14760)  

**Abstract**: Large Language Models (LLMs) are trained with next-token prediction, implemented in autoregressive Transformers via causal masking for parallelism. This creates a subtle misalignment: residual connections tie activations to the current token, while supervision targets the next token, potentially propagating mismatched information if the current token is not the most informative for prediction. In this work, we empirically localize this input-output alignment shift in pretrained LLMs, using decoding trajectories over tied embedding spaces and similarity-based metrics. Our experiments reveal that the hidden token representations switch from input alignment to output alignment deep within the network. Motivated by this observation, we propose a lightweight residual-path mitigation based on residual attenuation, implemented either as a fixed-layer intervention or as a learnable gating mechanism. Experiments on multiple benchmarks show that these strategies alleviate the representation misalignment and yield improvements, providing an efficient and general architectural enhancement for autoregressive Transformers. 

---
# Cognitive networks reconstruct mindsets about STEM subjects and educational contexts in almost 1000 high-schoolers, University students and LLM-based digital twins 

**Authors**: Francesco Gariboldi, Emma Franchino, Edith Haim, Gianluca Lattanzi, Alessandro Grecucci, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2602.14749)  

**Abstract**: Attitudes toward STEM develop from the interaction of conceptual knowledge, educational experiences, and affect. Here we use cognitive network science to reconstruct group mindsets as behavioural forma mentis networks (BFMNs). In this case, nodes are cue words and free associations, edges are empirical associative links, and each concept is annotated with perceived valence. We analyse BFMNs from N = 994 observations spanning high school students, university students, and early-career STEM experts, alongside LLM (GPT-oss) "digital twins" prompted to emulate comparable profiles. Focusing also on semantic neighbourhoods ("frames") around key target concepts (e.g., STEM subjects or educational actors/places), we quantify frames in terms of valence auras, emotional profiles, network overlap (Jaccard similarity), and concreteness relative to null baselines. Across student groups, science and research are consistently framed positively, while their core quantitative subjects (mathematics and statistics) exhibit more negative and anxiety related auras, amplified in higher math-anxiety subgroups, evidencing a STEM-science cognitive and emotional dissonance. High-anxiety frames are also less concrete than chance, suggesting more abstract and decontextualised representations of threatening quantitative domains. Human networks show greater overlapping between mathematics and anxiety than GPT-oss. The results highlight how BFMNs capture cognitive-affective signatures of mindsets towards the target domains and indicate that LLM-based digital twins approximate cultural attitudes but miss key context-sensitive, experience-based components relevant to replicate human educational anxiety. 

---
# Rethinking the Role of LLMs in Time Series Forecasting 

**Authors**: Xin Qiu, Junlong Tong, Yirong Sun, Yunpu Ma, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.14744)  

**Abstract**: Large language models (LLMs) have been introduced to time series forecasting (TSF) to incorporate contextual knowledge beyond numerical signals. However, existing studies question whether LLMs provide genuine benefits, often reporting comparable performance without LLMs. We show that such conclusions stem from limited evaluation settings and do not hold at scale. We conduct a large-scale study of LLM-based TSF (LLM4TSF) across 8 billion observations, 17 forecasting scenarios, 4 horizons, multiple alignment strategies, and both in-domain and out-of-domain settings. Our results demonstrate that \emph{LLM4TS indeed improves forecasting performance}, with especially large gains in cross-domain generalization. Pre-alignment outperforming post-alignment in over 90\% of tasks. Both pretrained knowledge and model architecture of LLMs contribute and play complementary roles: pretraining is critical under distribution shifts, while architecture excels at modeling complex temporal dynamics. Moreover, under large-scale mixed distributions, a fully intact LLM becomes indispensable, as confirmed by token-level routing analysis and prompt-based improvements. Overall, Our findings overturn prior negative assessments, establish clear conditions under which LLMs are not only useful, and provide practical guidance for effective model design. We release our code at this https URL. 

---
# LLMStructBench: Benchmarking Large Language Model Structured Data Extraction 

**Authors**: Sönke Tenckhoff, Mario Koddenbrock, Erik Rodner  

**Link**: [PDF](https://arxiv.org/pdf/2602.14743)  

**Abstract**: We present LLMStructBench, a novel benchmark for evaluating Large Language Models (LLMs) on extracting structured data and generating valid JavaScript Object Notation (JSON) outputs from natural-language text. Our open dataset comprises diverse, manually verified parsing scenarios of varying complexity and enables systematic testing across 22 models and five prompting strategies. We further introduce complementary performance metrics that capture both token-level accuracy and document-level validity, facilitating rigorous comparison of model, size, and prompting effects on parsing reliability.
In particular, we show that choosing the right prompting strategy is more important than standard attributes such as model size. This especially ensures structural validity for smaller or less reliable models but increase the number of semantic errors. Our benchmark suite is an step towards future research in the area of LLM applied to parsing or Extract, Transform and Load (ETL) applications. 

---
# Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography 

**Authors**: Gianluca Vico, Jindřich Libovický  

**Link**: [PDF](https://arxiv.org/pdf/2602.14675)  

**Abstract**: We present a crowdsourced dataset for Piedmontese, an endangered Romance language of northwestern Italy. The dataset comprises 145 Italian-Piedmontese parallel sentences derived from Flores+, with translations produced by speakers writing in their natural orthographic style rather than adhering to standardized conventions, along with manual word alignment. We use this resource to benchmark several large language models on tokenization parity, topic classification, and machine translation. Our analysis reveals that Piedmontese incurs a tokenization penalty relative to higher-resource Romance languages, yet LLMs achieve classification performance approaching that of Italian, French, and English. Machine translation results are asymmetric: models translate adequately from Piedmontese into high-resource languages, but generation into Piedmontese remains challenging. The dataset and code are publicly released. 

---
# Breaking Data Efficiency Dilemma: A Federated and Augmented Learning Framework For Alzheimer's Disease Detection via Speech 

**Authors**: Xiao Wei, Bin Wen, Yuqin Lin, Kai Li, Mingyang gu, Xiaobao Wang, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14655)  

**Abstract**: Early diagnosis of Alzheimer's Disease (AD) is crucial for delaying its progression. While AI-based speech detection is non-invasive and cost-effective, it faces a critical data efficiency dilemma due to medical data scarcity and privacy barriers. Therefore, we propose FAL-AD, a novel framework that synergistically integrates federated learning with data augmentation to systematically optimize data efficiency. Our approach delivers three key breakthroughs: First, absolute efficiency improvement through voice conversion-based augmentation, which generates diverse pathological speech samples via cross-category voice-content recombination. Second, collaborative efficiency breakthrough via an adaptive federated learning paradigm, maximizing cross-institutional benefits under privacy constraints. Finally, representational efficiency optimization by an attentive cross-modal fusion model, which achieves fine-grained word-level alignment and acoustic-textual interaction. Evaluated on ADReSSo, FAL-AD achieves a state-of-the-art multi-modal accuracy of 91.52%, outperforming all centralized baselines and demonstrating a practical solution to the data efficiency dilemma. Our source code is publicly available at this https URL. 

---
# Is Information Density Uniform when Utterances are Grounded on Perception and Discourse? 

**Authors**: Matteo Gay, Coleman Haley, Mario Giulianelli, Edoardo Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2602.14653)  

**Abstract**: The Uniform Information Density (UID) hypothesis posits that speakers are subject to a communicative pressure to distribute information evenly within utterances, minimising surprisal variance. While this hypothesis has been tested empirically, prior studies are limited exclusively to text-only inputs, abstracting away from the perceptual context in which utterances are produced. In this work, we present the first computational study of UID in visually grounded settings. We estimate surprisal using multilingual vision-and-language models over image-caption data in 30 languages and visual storytelling data in 13 languages, together spanning 11 families. We find that grounding on perception consistently smooths the distribution of information, increasing both global and local uniformity across typologically diverse languages compared to text-only settings. In visual narratives, grounding in both image and discourse contexts has additional effects, with the strongest surprisal reductions occurring at the onset of discourse units. Overall, this study takes a first step towards modelling the temporal dynamics of information flow in ecologically plausible, multimodal language use, and finds that grounded language exhibits greater information uniformity, supporting a context-sensitive formulation of UID. 

---
# GradMAP: Faster Layer Pruning with Gradient Metric and Projection Compensation 

**Authors**: Hao Liu, Guangyan Li, Wensheng Zhang, Yongqiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14649)  

**Abstract**: Large Language Models (LLMs) exhibit strong reasoning abilities, but their high computational costs limit their practical deployment. Recent studies reveal significant redundancy in LLMs layers, making layer pruning an active research topic. Layer pruning research primarily focuses on two aspects: measuring layer importance and recovering performance after pruning. Unfortunately, the present works fail to simultaneously maintain pruning performance and efficiency. In this study, we propose GradMAP, a faster layer pruning method with \textbf{Grad}ient \textbf{M}etric \textbf{A}nd \textbf{P}rojection compensation, which consists of two stages. In the first stage, we introduce a novel metric based on gradient magnitudes, enabling a global assessment of layer importance. Note that, it requires only a single backward propagation step per pruning decision, substantially enhancing pruning efficiency. In the second stage, we first analyze the layers with the largest mean shift resulting from pruning, and then incorporate a simple yet effective projection compensation matrix to correct this drift in one step. In this way, the degradation of model performance caused by layer pruning is effectively alleviated. Extensive experiments show that GradMAP outperforms previous layer pruning methods in both pruning speed (achieving an average $4\times$ speedup) and performance. 

---
# The Wikidata Query Logs Dataset 

**Authors**: Sebastian Walter, Hannah Bast  

**Link**: [PDF](https://arxiv.org/pdf/2602.14594)  

**Abstract**: We present the Wikidata Query Logs (WDQL) dataset, a dataset consisting of 200k question-query pairs over the Wikidata knowledge graph. It is over 6x larger than the largest existing Wikidata datasets of similar format without relying on template-generated queries. Instead, we construct it using real-world SPARQL queries sent to the Wikidata Query Service and generate questions for them. Since these log-based queries are anonymized, and therefore often do not produce results, a significant amount of effort is needed to convert them back into meaningful SPARQL queries. To achieve this, we present an agent-based method that iteratively de-anonymizes, cleans, and verifies queries against Wikidata while also generating corresponding natural-language questions. We demonstrate the dataset's benefit for training question-answering methods. All WDQL assets, as well as the agent code, are publicly available under a permissive license. 

---
# Assessing Large Language Models for Medical QA: Zero-Shot and LLM-as-a-Judge Evaluation 

**Authors**: Shefayat E Shams Adib, Ahmed Alfey Sani, Ekramul Alam Esham, Ajwad Abrar, Tareque Mohmud Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2602.14564)  

**Abstract**: Recently, Large Language Models (LLMs) have gained significant traction in medical domain, especially in developing a QA systems to Medical QA systems for enhancing access to healthcare in low-resourced settings. This paper compares five LLMs deployed between April 2024 and August 2025 for medical QA, using the iCliniq dataset, containing 38,000 medical questions and answers of diverse specialties. Our models include Llama-3-8B-Instruct, Llama 3.2 3B, Llama 3.3 70B Instruct, Llama-4-Maverick-17B-128E-Instruct, and GPT-5-mini. We are using a zero-shot evaluation methodology and using BLEU and ROUGE metrics to evaluate performance without specialized fine-tuning. Our results show that larger models like Llama 3.3 70B Instruct outperform smaller models, consistent with observed scaling benefits in clinical tasks. It is notable that, Llama-4-Maverick-17B exhibited more competitive results, thus highlighting evasion efficiency trade-offs relevant for practical deployment. These findings align with advancements in LLM capabilities toward professional-level medical reasoning and reflect the increasing feasibility of LLM-supported QA systems in the real clinical environments. This benchmark aims to serve as a standardized setting for future study to minimize model size, computational resources and to maximize clinical utility in medical NLP applications. 

---
# Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets 

**Authors**: Yuchen Yang, Wenze Lin, Enhao Huang, Zhixuan Chu, Hongbin Zhou, Lan Tao, Yiming Li, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.14536)  

**Abstract**: Large Language Models (LLMs) have seen remarkable advancements, achieving state-of-the-art results in diverse applications. Fine-tuning, an important step for adapting LLMs to specific downstream tasks, typically involves further training on corresponding datasets. However, a fundamental discrepancy exists between current fine-tuning datasets and the token-level optimization mechanism of LLMs: most datasets are designed at the sentence-level, which introduces token-level noise, causing negative influence to final performance. In this paper, we propose XTF, an explainable token-level noise filtering framework. XTF decomposes the complex and subtle contributions of token-level data to the fine-tuning process into three distinct and explicit attributes (reasoning importance, knowledge novelty, and task relevance), which can be assessed using scoring methods, and then masks the gradients of selected noisy tokens accordingly to optimize the performance of fine-tuned LLMs. We conduct extensive experiments on three representative downstream tasks (math, code and medicine) across 7 mainstream LLMs. The results demonstrate that XTF can significantly improve downstream performance by up to 13.7% compared to regular fine-tuning. Our work highlights the importance of token-level dataset optimization, and demonstrates the potential of strategies based on attribute decomposition for explaining complex training mechanisms. 

---
# Beyond Translation: Evaluating Mathematical Reasoning Capabilities of LLMs in Sinhala and Tamil 

**Authors**: Sukumar Kishanthan, Kumar Thushalika, Buddhi Jayasekara, Asela Hevapathige  

**Link**: [PDF](https://arxiv.org/pdf/2602.14517)  

**Abstract**: Large language models (LLMs) demonstrate strong mathematical reasoning in English, but whether these capabilities reflect genuine multilingual reasoning or reliance on translation-based processing in low-resource languages like Sinhala and Tamil remains unclear. We examine this fundamental question by evaluating whether LLMs genuinely reason mathematically in these languages or depend on implicit translation to English-like representations. Using a taxonomy of six math problem types, from basic arithmetic to complex unit conflict and optimization problems, we evaluate four prominent large language models. To avoid translation artifacts that confound language ability with translation quality, we construct a parallel dataset where each problem is natively authored by fluent speakers with mathematical training in all three languages. Our analysis demonstrates that while basic arithmetic reasoning transfers robustly across languages, complex reasoning tasks show significant degradation in Tamil and Sinhala. The pattern of failures varies by model and problem type, suggesting that apparent multilingual competence may not reflect uniform reasoning capabilities across languages. These findings challenge the common assumption that models exhibiting strong multilingual performance can reason equally effectively across languages, and highlight the need for fine-grained, type-aware evaluation in multilingual settings. 

---
# Query as Anchor: Scenario-Adaptive User Representation via Large Language Model 

**Authors**: Jiahao Yuan, Yike Xu, Jinyong Wen, Baokun Wang, Ziyi Gao, Xiaotong Lin, Yun Liu, Xing Fu, Yu Cheng, Yongchao Liu, Weiqiang Wang, Zhongle Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.14492)  

**Abstract**: Industrial-scale user representation learning requires balancing robust universality with acute task-sensitivity. However, existing paradigms primarily yield static, task-agnostic embeddings that struggle to reconcile the divergent requirements of downstream scenarios within unified vector spaces. Furthermore, heterogeneous multi-source data introduces inherent noise and modality conflicts, degrading representation. We propose Query-as-Anchor, a framework shifting user modeling from static encoding to dynamic, query-aware synthesis. To empower Large Language Models (LLMs) with deep user understanding, we first construct UserU, an industrial-scale pre-training dataset that aligns multi-modal behavioral sequences with user understanding semantics, and our Q-Anchor Embedding architecture integrates hierarchical coarse-to-fine encoders into dual-tower LLMs via joint contrastive-autoregressive optimization for query-aware user representation. To bridge the gap between general pre-training and specialized business logic, we further introduce Cluster-based Soft Prompt Tuning to enforce discriminative latent structures, effectively aligning model attention with scenario-specific modalities. For deployment, anchoring queries at sequence termini enables KV-cache-accelerated inference with negligible incremental latency. Evaluations on 10 Alipay industrial benchmarks show consistent SOTA performance, strong scalability, and efficient deployment. Large-scale online A/B testing in Alipay's production system across two real-world scenarios further validates its practical effectiveness. Our code is prepared for public release and will be available at: this https URL. 

---
# BETA-Labeling for Multilingual Dataset Construction in Low-Resource IR 

**Authors**: Md. Najib Hasan, Mst. Jannatun Ferdous Rain, Fyad Mohammed, Nazmul Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2602.14488)  

**Abstract**: IR in low-resource languages remains limited by the scarcity of high-quality, task-specific annotated datasets. Manual annotation is expensive and difficult to scale, while using large language models (LLMs) as automated annotators introduces concerns about label reliability, bias, and evaluation validity. This work presents a Bangla IR dataset constructed using a BETA-labeling framework involving multiple LLM annotators from diverse model families. The framework incorporates contextual alignment, consistency checks, and majority agreement, followed by human evaluation to verify label quality. Beyond dataset creation, we examine whether IR datasets from other low-resource languages can be effectively reused through one-hop machine translation. Using LLM-based translation across multiple language pairs, we experimented on meaning preservation and task validity between source and translated datasets. Our experiment reveal substantial variation across languages, reflecting language-dependent biases and inconsistent semantic preservation that directly affect the reliability of cross-lingual dataset reuse. Overall, this study highlights both the potential and limitations of LLM-assisted dataset creation for low-resource IR. It provides empirical evidence of the risks associated with cross-lingual dataset reuse and offers practical guidance for constructing more reliable benchmarks and evaluation pipelines in low-resource language settings. 

---
# HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation 

**Authors**: Wen-Sheng Lien, Yu-Kai Chan, Hao-Lung Hsiao, Bo-Kai Ruan, Meng-Fen Chiang, Chien-An Chen, Yi-Ren Yeh, Hong-Han Shuai  

**Link**: [PDF](https://arxiv.org/pdf/2602.14470)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) methods, typically built on knowledge graphs (KGs) with binary relational facts, have shown promise in multi-hop open-domain QA. However, their rigid retrieval schemes and dense similarity search often introduce irrelevant context, increase computational overhead, and limit relational expressiveness. In contrast, n-ary hypergraphs encode higher-order relational facts that capture richer inter-entity dependencies and enable shallower, more efficient reasoning paths. To address this limitation, we propose HyperRAG, a RAG framework tailored for n-ary hypergraphs with two complementary retrieval variants: (i) HyperRetriever learns structural-semantic reasoning over n-ary facts to construct query-conditioned relational chains. It enables accurate factual tracking, adaptive high-order traversal, and interpretable multi-hop reasoning under context constraints. (ii) HyperMemory leverages the LLM's parametric memory to guide beam search, dynamically scoring n-ary facts and entities for query-aware path expansion. Extensive evaluations on WikiTopics (11 closed-domain datasets) and three open-domain QA benchmarks (HotpotQA, MuSiQue, and 2WikiMultiHopQA) validate HyperRAG's effectiveness. HyperRetriever achieves the highest answer accuracy overall, with average gains of 2.95% in MRR and 1.23% in Hits@10 over the strongest baseline. Qualitative analysis further shows that HyperRetriever bridges reasoning gaps through adaptive and interpretable n-ary chain construction, benefiting both open and closed-domain QA. 

---
# Measuring and Mitigating Post-hoc Rationalization in Reverse Chain-of-Thought Generation 

**Authors**: Guangyue Peng, Zongchao Chen, Wen Luo, Yuntao Wen, Wei Li, Ruixiang Feng, Ran Le, Chen Yang, Zhenwei An, Yang Song, Tao Zhang, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14469)  

**Abstract**: Reverse Chain-of-Thought Generation (RCG) synthesizes reasoning traces from query-answer pairs, but runs the risk of producing post-hoc rationalizations: when models can see the answer during generation, the answer serves as a cognitive anchor that shapes the entire explanation. We formalize this phenomenon through a three-level measurement hierarchy: lexical, entropic, and probabilistic anchoring, each captures surface artifacts, entropy dynamics, and latent answer dependence, respectively. We analyze semantic suppression, the intuitive mitigation strategy that instructs models to ignore the answer, to find out its counterproduction: while it reduces lexical overlap, it paradoxically increases entropic and probabilistic anchoring. Drawing on Ironic Process Theory from cognitive psychology, we attribute this failure to active monitoring of the forbidden answer, which inadvertently deepens dependence on it. To break this cycle, we propose Structural Skeleton-guided Reasoning (SSR), a two-phase approach that first generates an answer-invariant functional skeleton structure, then uses this skeleton to guide full trace generation. By redirecting the information flow to structural planning rather than answer monitoring, SSR consistently reduces anchoring across all three levels. We further introduce Distilled SSR (SSR-D), which fine-tunes models on teacher-generated SSR traces to ensure reliable structural adherence. Experiments across open-ended reasoning benchmarks demonstrate that SSR-D achieves up to 10% improvement over suppression baselines while preserving out-of-distribution (OOD) generalization. 

---
# Robust Bias Evaluation with FilBBQ: A Filipino Bias Benchmark for Question-Answering Language Models 

**Authors**: Lance Calvin Lim Gamboa, Yue Feng, Mark Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.14466)  

**Abstract**: With natural language generation becoming a popular use case for language models, the Bias Benchmark for Question-Answering (BBQ) has grown to be an important benchmark format for evaluating stereotypical associations exhibited by generative models. We expand the linguistic scope of BBQ and construct FilBBQ through a four-phase development process consisting of template categorization, culturally aware translation, new template construction, and prompt generation. These processes resulted in a bias test composed of more than 10,000 prompts which assess whether models demonstrate sexist and homophobic prejudices relevant to the Philippine context. We then apply FilBBQ on models trained in Filipino but do so with a robust evaluation protocol that improves upon the reliability and accuracy of previous BBQ implementations. Specifically, we account for models' response instability by obtaining prompt responses across multiple seeds and averaging the bias scores calculated from these distinctly seeded runs. Our results confirm both the variability of bias scores across different seeds and the presence of sexist and homophobic biases relating to emotion, domesticity, stereotyped queer interests, and polygamy. FilBBQ is available via GitHub. 

---
# LLM-Guided Knowledge Distillation for Temporal Knowledge Graph Reasoning 

**Authors**: Wang Xing, Wei Song, Siyu Lin, Chen Wu, Man Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14428)  

**Abstract**: Temporal knowledge graphs (TKGs) support reasoning over time-evolving facts, yet state-of-the-art models are often computationally heavy and costly to deploy. Existing compression and distillation techniques are largely designed for static graphs; directly applying them to temporal settings may overlook time-dependent interactions and lead to performance degradation. We propose an LLM-assisted distillation framework specifically designed for temporal knowledge graph reasoning. Beyond a conventional high-capacity temporal teacher, we incorporate a large language model as an auxiliary instructor to provide enriched supervision. The LLM supplies broad background knowledge and temporally informed signals, enabling a lightweight student to better model event dynamics without increasing inference-time complexity. Training is conducted by jointly optimizing supervised and distillation objectives, using a staged alignment strategy to progressively integrate guidance from both teachers. Extensive experiments on multiple public TKG benchmarks with diverse backbone architectures demonstrate that the proposed approach consistently improves link prediction performance over strong distillation baselines, while maintaining a compact and efficient student model. The results highlight the potential of large language models as effective teachers for transferring temporal reasoning capability to resource-efficient TKG systems. 

---
# WavePhaseNet: A DFT-Based Method for Constructing Semantic Conceptual Hierarchy Structures (SCHS) 

**Authors**: Kiyotaka Kasubuchi, Kazuo Fukiya  

**Link**: [PDF](https://arxiv.org/pdf/2602.14419)  

**Abstract**: This paper reformulates Transformer/Attention mechanisms in Large Language Models (LLMs) through measure theory and frequency analysis, theoretically demonstrating that hallucination is an inevitable structural limitation. The embedding space functions as a conditional expectation over a {\sigma}-algebra, and its failure to be isomorphic to the semantic truth set fundamentally causes logical consistency breakdown. WavePhaseNet Method The authors propose WavePhaseNet, which explicitly constructs a Semantic Conceptual Hierarchy Structure (SCHS) using Discrete Fourier Transform (DFT). By applying DFT along the sequence dimension, semantic information is decomposed into frequency bands: low-frequency components capture global meaning and intent, while high-frequency components represent local syntax and expression. This staged separation enables precise semantic manipulation in diagonalized space. Dimensionality Reduction GPT-4's 24,576-dimensional embedding space exhibits a 1/f spectral structure based on language self-similarity and Zipf's law. Through cumulative energy analysis, the authors derive that approximately 3,000 dimensions constitute the lower bound for "complete representation." This demonstrates that reduction from 24,576 to 3,000 dimensions preserves meaning and intent while enabling rigorous reasoning and suppressing hallucination. Cohomological Consistency Control The reduced embedding space, constructed via cohomological regularization over overlapping local windows, allows defining a graph structure and cochain complex. This quantifies inconsistencies among local inferences as coboundary-based losses. Applying harmonic projection based on Hodge theory positions cohomology as a computable regularization principle for controlling semantic consistency, extracting maximally consistent global representations. 

---
# TruthStance: An Annotated Dataset of Conversations on Truth Social 

**Authors**: Fathima Ameen, Danielle Brown, Manusha Malgareddy, Amanul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2602.14406)  

**Abstract**: Argument mining and stance detection are central to understanding how opinions are formed and contested in online discourse. However, most publicly available resources focus on mainstream platforms such as Twitter and Reddit, leaving conversational structure on alt-tech platforms comparatively under-studied. We introduce TruthStance, a large-scale dataset of Truth Social conversation threads spanning 2023-2025, consisting of 24,378 posts and 523,360 comments with reply-tree structure preserved. We provide a human-annotated benchmark of 1,500 instances across argument mining and claim-based stance detection, including inter-annotator agreement, and use it to evaluate large language model (LLM) prompting strategies. Using the best-performing configuration, we release additional LLM-generated labels for 24,352 posts (argument presence) and 107,873 comments (stance to parent), enabling analysis of stance and argumentation patterns across depth, topics, and users. All code and data are released publicly. 

---
# Beyond Token-Level Policy Gradients for Complex Reasoning with Large Language Models 

**Authors**: Mufan Xu, Kehai Chen, Xuefeng Bai, Zhengyu Niu, Muyun Yang, Tiejun Zhao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14386)  

**Abstract**: Existing policy-gradient methods for auto-regressive language models typically select subsequent tokens one at a time as actions in the policy. While effective for many generation tasks, such an approach may not fully capture the structure of complex reasoning tasks, where a single semantic decision is often realized across multiple tokens--for example, when defining variables or composing equations. This introduces a potential mismatch between token-level optimization and the inherently block-level nature of reasoning in these settings. To bridge this gap, we propose Multi-token Policy Gradient Optimization (MPO), a framework that treats sequences of K consecutive tokens as unified semantic actions. This block-level perspective enables our method to capture the compositional structure of reasoning trajectories and supports optimization over coherent, higher-level objectives. Experiments on mathematical reasoning and coding benchmarks show that MPO outperforms standard token-level policy gradient baselines, highlight the limitations of token-level policy gradients for complex reasoning, motivating future research to look beyond token-level granularity for reasoning-intensive language tasks. 

---
# InnoEval: On Research Idea Evaluation as a Knowledge-Grounded, Multi-Perspective Reasoning Problem 

**Authors**: Shuofei Qiao, Yunxiang Wei, Xuehai Wang, Bin Wu, Boyang Xue, Ningyu Zhang, Hossein A. Rahmani, Yanshan Wang, Qiang Zhang, Keyan Ding, Jeff Z. Pan, Huajun Chen, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2602.14367)  

**Abstract**: The rapid evolution of Large Language Models has catalyzed a surge in scientific idea production, yet this leap has not been accompanied by a matching advance in idea evaluation. The fundamental nature of scientific evaluation needs knowledgeable grounding, collective deliberation, and multi-criteria decision-making. However, existing idea evaluation methods often suffer from narrow knowledge horizons, flattened evaluation dimensions, and the inherent bias in LLM-as-a-Judge. To address these, we regard idea evaluation as a knowledge-grounded, multi-perspective reasoning problem and introduce InnoEval, a deep innovation evaluation framework designed to emulate human-level idea assessment. We apply a heterogeneous deep knowledge search engine that retrieves and grounds dynamic evidence from diverse online sources. We further achieve review consensus with an innovation review board containing reviewers with distinct academic backgrounds, enabling a multi-dimensional decoupled evaluation across multiple metrics. We construct comprehensive datasets derived from authoritative peer-reviewed submissions to benchmark InnoEval. Experiments demonstrate that InnoEval can consistently outperform baselines in point-wise, pair-wise, and group-wise evaluation tasks, exhibiting judgment patterns and consensus highly aligned with human experts. 

---
# Does Socialization Emerge in AI Agent Society? A Case Study of Moltbook 

**Authors**: Ming Li, Xirui Li, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.14299)  

**Abstract**: As large language model agents increasingly populate networked environments, a fundamental question arises: do artificial intelligence (AI) agent societies undergo convergence dynamics similar to human social systems? Lately, Moltbook approximates a plausible future scenario in which autonomous agents participate in an open-ended, continuously evolving online society. We present the first large-scale systemic diagnosis of this AI agent society. Beyond static observation, we introduce a quantitative diagnostic framework for dynamic evolution in AI agent societies, measuring semantic stabilization, lexical turnover, individual inertia, influence persistence, and collective consensus. Our analysis reveals a system in dynamic balance in Moltbook: while global semantic averages stabilize rapidly, individual agents retain high diversity and persistent lexical turnover, defying homogenization. However, agents exhibit strong individual inertia and minimal adaptive response to interaction partners, preventing mutual influence and consensus. Consequently, influence remains transient with no persistent supernodes, and the society fails to develop stable collective influence anchors due to the absence of shared social memory. These findings demonstrate that scale and interaction density alone are insufficient to induce socialization, providing actionable design and analysis principles for upcoming next-generation AI agent societies. 

---
# STATe-of-Thoughts: Structured Action Templates for Tree-of-Thoughts 

**Authors**: Zachary Bamberger, Till R. Saenger, Gilad Morad, Ofra Amir, Brandon M. Stewart, Amir Feder  

**Link**: [PDF](https://arxiv.org/pdf/2602.14265)  

**Abstract**: Inference-Time-Compute (ITC) methods like Best-of-N and Tree-of-Thoughts are meant to produce output candidates that are both high-quality and diverse, but their use of high-temperature sampling often fails to achieve meaningful output diversity. Moreover, existing ITC methods offer limited control over how to perform reasoning, which in turn limits their explainability. We present STATe-of-Thoughts (STATe), an interpretable ITC method that searches over high-level reasoning patterns. STATe replaces stochastic sampling with discrete and interpretable textual interventions: a controller selects actions encoding high-level reasoning choices, a generator produces reasoning steps conditioned on those choices, and an evaluator scores candidates to guide search. This structured approach yields three main advantages. First, action-guided textual interventions produce greater response diversity than temperature-based sampling. Second, in a case study on argument generation, STATe's explicit action sequences capture interpretable features that are highly predictive of output quality. Third, estimating the association between performance and action choices allows us to identify promising yet unexplored regions of the action space and steer generation directly toward them. Together, these results establish STATe as a practical framework for generating high-quality, diverse, and interpretable text. Our framework is available at this https URL. 

---
# Detecting LLM Hallucinations via Embedding Cluster Geometry: A Three-Type Taxonomy with Measurable Signatures 

**Authors**: Matic Korun  

**Link**: [PDF](https://arxiv.org/pdf/2602.14259)  

**Abstract**: We propose a geometric taxonomy of large language model hallucinations based on observable signatures in token embedding cluster structure. By analyzing the static embedding spaces of 11 transformer models spanning encoder (BERT, RoBERTa, ELECTRA, DeBERTa, ALBERT, MiniLM, DistilBERT) and decoder (GPT-2) architectures, we identify three operationally distinct hallucination types: Type 1 (center-drift) under weak context, Type 2 (wrong-well convergence) to locally coherent but contextually incorrect cluster regions, and Type 3 (coverage gaps) where no cluster structure exists. We introduce three measurable geometric statistics: {\alpha} (polarity coupling), \b{eta} (cluster cohesion), and {\lambda}_s (radial information gradient). Across all 11 models, polarity structure ({\alpha} > 0.5) is universal (11/11), cluster cohesion (\b{eta} > 0) is universal (11/11), and the radial information gradient is significant (9/11, p < 0.05). We demonstrate that the two models failing {\lambda}_s significance -- ALBERT and MiniLM -- do so for architecturally explicable reasons: factorized embedding compression and distillation-induced isotropy, respectively. These findings establish the geometric prerequisites for type-specific hallucination detection and yield testable predictions about architecture-dependent vulnerability profiles. 

---
# AD-Bench: A Real-World, Trajectory-Aware Advertising Analytics Benchmark for LLM Agents 

**Authors**: Lingxiang Hu, Yiding Sun, Tianle Xia, Wenwei Li, Ming Xu, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14257)  

**Abstract**: While Large Language Model (LLM) agents have achieved remarkable progress in complex reasoning tasks, evaluating their performance in real-world environments has become a critical problem. Current benchmarks, however, are largely restricted to idealized simulations, failing to address the practical demands of specialized domains like advertising and marketing analytics. In these fields, tasks are inherently more complex, often requiring multi-round interaction with professional marketing tools. To address this gap, we propose AD-Bench, a benchmark designed based on real-world business requirements of advertising and marketing platforms. AD-Bench is constructed from real user marketing analysis requests, with domain experts providing verifiable reference answers and corresponding reference tool-call trajectories. The benchmark categorizes requests into three difficulty levels (L1-L3) to evaluate agents' capabilities under multi-round, multi-tool collaboration. Experiments show that on AD-Bench, Gemini-3-Pro achieves Pass@1 = 68.0% and Pass@3 = 83.0%, but performance drops significantly on L3 to Pass@1 = 49.4% and Pass@3 = 62.1%, with a trajectory coverage of 70.1%, indicating that even state-of-the-art models still exhibit substantial capability gaps in complex advertising and marketing analysis scenarios. AD-Bench provides a realistic benchmark for evaluating and improving advertising marketing agents, the leaderboard and code can be found at this https URL. 

---
# We can still parse using syntactic rules 

**Authors**: Ghaly Hussein  

**Link**: [PDF](https://arxiv.org/pdf/2602.14238)  

**Abstract**: This research introduces a new parsing approach, based on earlier syntactic work on context free grammar (CFG) and generalized phrase structure grammar (GPSG). The approach comprises both a new parsing algorithm and a set of syntactic rules and features that overcome the limitations of CFG. It also generates both dependency and constituency parse trees, while accommodating noise and incomplete parses. The system was tested on data from Universal Dependencies, showing a promising average Unlabeled Attachment Score (UAS) of 54.5% in the development dataset (7 corpora) and 53.8% in the test set (12 corpora). The system also provides multiple parse hypotheses, allowing further reranking to improve parsing accuracy. This approach also leverages much of the theoretical syntactic work since the 1950s to be used within a computational context. The application of this approach provides a transparent and interpretable NLP model to process language input. 

---
# Knowing When Not to Answer: Abstention-Aware Scientific Reasoning 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2602.14189)  

**Abstract**: Large language models are increasingly used to answer and verify scientific claims, yet existing evaluations typically assume that a model must always produce a definitive answer. In scientific settings, however, unsupported or uncertain conclusions can be more harmful than abstaining. We study this problem through an abstention-aware verification framework that decomposes scientific claims into minimal conditions, audits each condition against available evidence using natural language inference (NLI), and selectively decides whether to support, refute, or abstain. We evaluate this framework across two complementary scientific benchmarks: SciFact and PubMedQA, covering both closed-book and open-domain evidence settings. Experiments are conducted with six diverse language models, including encoder-decoder, open-weight chat models, and proprietary APIs. Across all benchmarks and models, we observe that raw accuracy varies only modestly across architectures, while abstention plays a critical role in controlling error. In particular, confidence-based abstention substantially reduces risk at moderate coverage levels, even when absolute accuracy improvements are limited. Our results suggest that in scientific reasoning tasks, the primary challenge is not selecting a single best model, but rather determining when available evidence is sufficient to justify an answer. This work highlights abstention-aware evaluation as a practical and model-agnostic lens for assessing scientific reliability, and provides a unified experimental basis for future work on selective reasoning in scientific domains. Code is available at this https URL . 

---
# GPT-5 vs Other LLMs in Long Short-Context Performance 

**Authors**: Nima Esmi, Maryam Nezhad-Moghaddam, Fatemeh Borhani, Asadollah Shahbahrami, Amin Daemdoost, Georgi Gaydadjiev  

**Link**: [PDF](https://arxiv.org/pdf/2602.14188)  

**Abstract**: With the significant expansion of the context window in Large Language Models (LLMs), these models are theoretically capable of processing millions of tokens in a single pass. However, research indicates a significant gap between this theoretical capacity and the practical ability of models to robustly utilize information within long contexts, especially in tasks that require a comprehensive understanding of numerous details. This paper evaluates the performance of four state-of-the-art models (Grok-4, GPT-4, Gemini 2.5, and GPT-5) on long short-context tasks. For this purpose, three datasets were used: two supplementary datasets for retrieving culinary recipes and math problems, and a primary dataset of 20K social media posts for depression detection. The results show that as the input volume on the social media dataset exceeds 5K posts (70K tokens), the performance of all models degrades significantly, with accuracy dropping to around 50-53% for 20K posts. Notably, in the GPT-5 model, despite the sharp decline in accuracy, its precision remained high at approximately 95%, a feature that could be highly effective for sensitive applications like depression detection. This research also indicates that the "lost in the middle" problem has been largely resolved in newer models. This study emphasizes the gap between the theoretical capacity and the actual performance of models on complex, high-volume data tasks and highlights the importance of metrics beyond simple accuracy for practical applications. 

---
# Index Light, Reason Deep: Deferred Visual Ingestion for Visual-Dense Document Question Answering 

**Authors**: Tao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.14162)  

**Abstract**: Existing multimodal document question answering methods universally adopt a supply-side ingestion strategy: running a Vision-Language Model (VLM) on every page during indexing to generate comprehensive descriptions, then answering questions through text retrieval. However, this "pre-ingestion" approach is costly (a 113-page engineering drawing package requires approximately 80,000 VLM tokens), end-to-end unreliable (VLM outputs may fail to be correctly retrieved due to format mismatches in the retrieval infrastructure), and irrecoverable once it fails. This paper proposes the Deferred Visual Ingestion (DVI) framework, adopting a demand-side ingestion strategy: the indexing phase performs only lightweight metadata extraction, deferring visual understanding to the moment users pose specific questions. DVI's core principle is "Index for locating, not understanding"--achieving page localization through structured metadata indexes and BM25 full-text search, then sending original images along with specific questions to a VLM for targeted analysis. Experiments on two real industrial engineering drawings (113 pages + 7 pages) demonstrate that DVI achieves comparable overall accuracy at zero ingestion VLM cost (46.7% vs. 48.9%), an effectiveness rate of 50% on visually necessary queries (vs. 0% for pre-ingestion), and 100% page localization (98% search space compression). DVI also supports interactive refinement and progressive caching, transforming the "QA accuracy" problem into a "page localization" problem--once the correct drawing page is found, obtaining the answer becomes a matter of interaction rounds. 

---
# A Multi-Agent Framework for Medical AI: Leveraging Fine-Tuned GPT, LLaMA, and DeepSeek R1 for Evidence-Based and Bias-Aware Clinical Query Processing 

**Authors**: Naeimeh Nourmohammadi, Md Meem Hossain, Anh Han, Safina Showkat Ara, Zia Ush Shamszaman  

**Link**: [PDF](https://arxiv.org/pdf/2602.14158)  

**Abstract**: Large language models (LLMs) show promise for healthcare question answering, but clinical use is limited by weak verification, insufficient evidence grounding, and unreliable confidence signalling. We propose a multi-agent medical QA framework that combines complementary LLMs with evidence retrieval, uncertainty estimation, and bias checks to improve answer reliability. Our approach has two phases. First, we fine-tune three representative LLM families (GPT, LLaMA, and DeepSeek R1) on MedQuAD-derived medical QA data (20k+ question-answer pairs across multiple NIH domains) and benchmark generation quality. DeepSeek R1 achieves the strongest scores (ROUGE-1 0.536 +- 0.04; ROUGE-2 0.226 +-0.03; BLEU 0.098 -+ 0.018) and substantially outperforms the specialised biomedical baseline BioGPT in zero-shot evaluation. Second, we implement a modular multi-agent pipeline in which a Clinical Reasoning agent (fine-tuned LLaMA) produces structured explanations, an Evidence Retrieval agent queries PubMed to ground responses in recent literature, and a Refinement agent (DeepSeek R1) improves clarity and factual consistency; an optional human validation path is triggered for high-risk or high-uncertainty cases. Safety mechanisms include Monte Carlo dropout and perplexity-based uncertainty scoring, plus lexical and sentiment-based bias detection supported by LIME/SHAP-based analyses. In evaluation, the full system achieves 87% accuracy with relevance around 0.80, and evidence augmentation reduces uncertainty (perplexity 4.13) compared to base responses, with mean end-to-end latency of 36.5 seconds under the reported configuration. Overall, the results indicate that agent specialisation and verification layers can mitigate key single-model limitations and provide a practical, extensible design for evidence-based and bias-aware medical AI. 

---
# Character-aware Transformers Learn an Irregular Morphological Pattern Yet None Generalize Like Humans 

**Authors**: Akhilesh Kakolu Ramarao, Kevin Tang, Dinah Baer-Henney  

**Link**: [PDF](https://arxiv.org/pdf/2602.14100)  

**Abstract**: Whether neural networks can serve as cognitive models of morphological learning remains an open question. Recent work has shown that encoder-decoder models can acquire irregular patterns, but evidence that they generalize these patterns like humans is mixed. We investigate this using the Spanish \emph{L-shaped morphome}, where only the first-person singular indicative (e.g., \textit{pongo} `I put') shares its stem with all subjunctive forms (e.g., \textit{ponga, pongas}) despite lacking apparent phonological, semantic, or syntactic motivation. We compare five encoder-decoder transformers varying along two dimensions: sequential vs. position-invariant positional encoding, and atomic vs. decomposed tag representations. Positional encoding proves decisive: position-invariant models recover the correct L-shaped paradigm clustering even when L-shaped verbs are scarce in training, whereas sequential positional encoding models only partially capture the pattern. Yet none of the models productively generalize this pattern to novel forms. Position-invariant models generalize the L-shaped stem across subjunctive cells but fail to extend it to the first-person singular indicative, producing a mood-based generalization rather than the L-shaped morphomic pattern. Humans do the opposite, generalizing preferentially to the first-person singular indicative over subjunctive forms. None of the models reproduce the human pattern, highlighting the gap between statistical pattern reproduction and morphological abstraction. 

---
# CCiV: A Benchmark for Structure, Rhythm and Quality in LLM-Generated Chinese \textit{Ci} Poetry 

**Authors**: Shangqing Zhao, Yupei Ren, Yuhao Zhou, Xiaopeng Bai, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.14081)  

**Abstract**: The generation of classical Chinese \textit{Ci} poetry, a form demanding a sophisticated blend of structural rigidity, rhythmic harmony, and artistic quality, poses a significant challenge for large language models (LLMs). To systematically evaluate and advance this capability, we introduce \textbf{C}hinese \textbf{Ci}pai \textbf{V}ariants (\textbf{CCiV}), a benchmark designed to assess LLM-generated \textit{Ci} poetry across these three dimensions: structure, rhythm, and quality. Our evaluation of 17 LLMs on 30 \textit{Cipai} reveals two critical phenomena: models frequently generate valid but unexpected historical variants of a poetic form, and adherence to tonal patterns is substantially harder than structural rules. We further show that form-aware prompting can improve structural and tonal control for stronger models, while potentially degrading weaker ones. Finally, we observe weak and inconsistent alignment between formal correctness and literary quality in our sample. CCiV highlights the need for variant-aware evaluation and more holistic constrained creative generation methods. 

---
# Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality 

**Authors**: Nitay Calderon, Eyal Ben-David, Zorik Gekhman, Eran Ofek, Gal Yona  

**Link**: [PDF](https://arxiv.org/pdf/2602.14080)  

**Abstract**: Standard factuality evaluations of LLMs treat all errors alike, obscuring whether failures arise from missing knowledge (empty shelves) or from limited access to encoded facts (lost keys). We propose a behavioral framework that profiles factual knowledge at the level of facts rather than questions, characterizing each fact by whether it is encoded, and then by how accessible it is: cannot be recalled, can be directly recalled, or can only be recalled with inference-time computation (thinking). To support such profiling, we introduce WikiProfile, a new benchmark constructed via an automated pipeline with a prompted LLM grounded in web search. Across 4 million responses from 13 LLMs, we find that encoding is nearly saturated in frontier models on our benchmark, with GPT-5 and Gemini-3 encoding 95--98% of facts. However, recall remains a major bottleneck: many errors previously attributed to missing knowledge instead stem from failures to access it. These failures are systematic and disproportionately affect long-tail facts and reverse questions. Finally, we show that thinking improves recall and can recover a substantial fraction of failures, indicating that future gains may rely less on scaling and more on methods that improve how models utilize what they already encode. 

---
# GTS: Inference-Time Scaling of Latent Reasoning with a Learnable Gaussian Thought Sampler 

**Authors**: Minghan Wang, Ye Bai, Thuy-Trang Vu, Ehsan Shareghi, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2602.14077)  

**Abstract**: Inference-time scaling (ITS) in latent reasoning models typically introduces stochasticity through heuristic perturbations, such as dropout or fixed Gaussian noise. While these methods increase trajectory diversity, their exploration behavior is not explicitly modeled and can be inefficient under finite sampling budgets. We observe that stronger perturbations do not necessarily translate into more effective candidate trajectories, as unguided noise may disrupt internal decision structure rather than steer it. To provide a more structured alternative, we model latent thought exploration as conditional sampling from learnable densities and instantiate this idea as a Gaussian Thought Sampler (GTS). GTS predicts context-dependent perturbation distributions over continuous reasoning states and is trained with GRPO-style policy optimization while keeping the backbone frozen. Experiments on GSM8K with two latent reasoning architectures show that GTS achieves more reliable inference-time scaling than heuristic baselines. These findings indicate that improving latent ITS requires structured and optimizable exploration mechanisms rather than simply amplifying stochasticity. 

---
# Annotation-Efficient Vision-Language Model Adaptation to the Polish Language Using the LLaVA Framework 

**Authors**: Grzegorz Statkiewicz, Alicja Dobrzeniecka, Karolina Seweryn, Aleksandra Krasnodębska, Karolina Piosek, Katarzyna Bogusz, Sebastian Cygert, Wojciech Kusa  

**Link**: [PDF](https://arxiv.org/pdf/2602.14073)  

**Abstract**: Most vision-language models (VLMs) are trained on English-centric data, limiting their performance in other languages and cultural contexts. This restricts their usability for non-English-speaking users and hinders the development of multimodal systems that reflect diverse linguistic and cultural realities. In this work, we reproduce and adapt the LLaVA-Next methodology to create a set of Polish VLMs. We rely on a fully automated pipeline for translating and filtering existing multimodal datasets, and complement this with synthetic Polish data for OCR and culturally specific tasks. Despite relying almost entirely on automatic translation and minimal manual intervention to the training data, our approach yields strong results: we observe a +9.5% improvement over LLaVA-1.6-Vicuna-13B on a Polish-adapted MMBench, along with higher-quality captions in generative evaluations, as measured by human annotators in terms of linguistic correctness. These findings highlight that large-scale automated translation, combined with lightweight filtering, can effectively bootstrap high-quality multimodal models for low-resource languages. Some challenges remain, particularly in cultural coverage and evaluation. To facilitate further research, we make our models and evaluation dataset publicly available. 

---
# Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric 

**Authors**: Ruipeng Jia, Yunyi Yang, Yuxin Wu, Yongbo Gai, Siyuan Tao, Mengyu Zhou, Jianhe Lin, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14069)  

**Abstract**: Scalar reward models compress multi-dimensional human preferences into a single opaque score, creating an information bottleneck that often leads to brittleness and reward hacking in open-ended alignment. We argue that robust alignment for non-verifiable tasks is fundamentally a principle generalization problem: reward should not be a learned function internalized into a judge, but an explicit reasoning process executed under inspectable principles. To operationalize this view, we present the Open Rubric System (OpenRS), a plug-and-play, rubrics-based LLM-as-a-Judge framework built around Pairwise Adaptive Meta-Rubrics (PAMR) and lightweight Pointwise Verifiable Rubrics (PVRs), which provide both hard-constraint guardrails and verifiable reward components when ground-truth or programmatic checks are available. OpenRS uses an explicit meta-rubric -- a constitution-like specification that governs how rubrics are instantiated, weighted, and enforced -- and instantiates adaptive rubrics on the fly by conditioning on the semantic differences between two candidate responses. It then performs criterion-wise pairwise comparisons and aggregates criterion-level preferences externally, avoiding pointwise weighted scalarization while improving discriminability in open-ended settings. To keep principles consistent yet editable across various domains, we introduce a two-level meta-rubric refinement pipeline (automated evolutionary refinement for general principles and a reproducible human-in-the-loop procedure for domain principles), complemented with pointwise verifiable rubrics that act as both guardrails against degenerate behaviors and a source of verifiable reward for objective sub-tasks. Finally, we instantiate OpenRS as reward supervision in pairwise RL training. 

---
# From Scarcity to Scale: A Release-Level Analysis of the Pashto Common Voice Dataset 

**Authors**: Jandad Jahani, Mursal Dawodi, Jawid Ahmad Baktash  

**Link**: [PDF](https://arxiv.org/pdf/2602.14062)  

**Abstract**: Large, openly licensed speech datasets are essential for building automatic speech recognition (ASR) systems, yet many widely spoken languages remain underrepresented in public resources. Pashto, spoken by more than 60 million people, has historically lacked large-scale openly licensed speech data suitable for modern ASR development.
This paper presents a release-level analysis of the Pashto component of the Mozilla Common Voice corpus, focusing on version 24.0 (December 2025) and contextualizing trends across major releases. We document rapid growth from 1.49 recorded hours in mid-2023 to 2,768.7 total hours in 2025, including 975.89 validated hours available for supervised ASR training.
Beyond scale, we analyze validation throughput, contributor participation inequality, demographic metadata completeness, and sentence-level concentration in the validated subset. We find that participation is extremely concentrated (Gini = 0.941), age representation is strongly skewed toward young adults, and 41.97\% of clips lack self-reported gender labels, limiting subgroup auditing based on metadata. At the textual level, prompt reuse is moderate: 35.88\% of unique sentences account for 50\% of validated clips, suggesting that structural concentration is driven primarily by uneven contributor activity rather than dominance of a small prompt set.
These results provide a quantitative audit of a rapidly scaling low-resource speech corpus and highlight practical priorities for improving dataset maturity, including expanded validation capacity and broader demographic participation. 

---
# LM-Lexicon: Improving Definition Modeling via Harmonizing Semantic Experts 

**Authors**: Yang Liu, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li, Lingyong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.14060)  

**Abstract**: We introduce LM-Lexicon, an innovative definition modeling approach that incorporates data clustering, semantic expert learning, and model merging using a sparse mixture-of-experts architecture. By decomposing the definition modeling task into specialized semantic domains, where small language models are trained as domain experts, LM-Lexicon achieves substantial improvements (+7% BLEU score compared with the prior state-of-the-art model) over existing methods on five widely used benchmarks. Empirically, we demonstrate that 1) the clustering strategy enables fine-grained expert specialization with nearly 10% improvement in definition quality; 2) the semantic-aware domain-level routing mechanism achieves higher expert efficacy (+1%) than conventional token-level routing; and 3) further performance gains can be obtained through test-time compute and semantic expert scaling. Our work advances definition modeling while providing insights into the development of efficient language models for semantic-intensive applications. 

---
# LogitsCoder: Towards Efficient Chain-of-Thought Path Search via Logits Preference Decoding for Code Generation 

**Authors**: Jizheng Chen, Weiming Zhang, Xinyi Dai, Weiwen Liu, Kounianhua Du, Yasheng Wang, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14054)  

**Abstract**: Code generation remains a challenging task that requires precise and structured reasoning. Existing Test Time Scaling (TTS) methods, including structured tree search, have made progress in exploring reasoning paths but still face two major challenges: (1) underthinking, where reasoning chains tend to be shallow and fail to capture the full complexity of problems; and (2) overthinking, where overly verbose reasoning leads to inefficiency and increased computational costs. To address these issues, we propose LogitsCoder, a novel framework that enhances chain-of-thought reasoning through lightweight, logit-level control mechanisms for code generation. LogitsCoder iteratively generates and refines reasoning steps by first steering token selection toward statistically preferred patterns via Logits Preference Decoding, then selecting and aggregating diverse reasoning paths using Logits Rank Based Path Selection and Thoughts Aggregation. This results in coherent and effective reasoning chains that balance depth and efficiency. Extensive experiments demonstrate that LogitsCoder produces more efficient and higher-quality reasoning chains, leading to superior code generation performance compared to baseline methods. 

---
# Context Shapes LLMs Retrieval-Augmented Fact-Checking Effectiveness 

**Authors**: Pietro Bernardelle, Stefano Civelli, Kevin Roitero, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2602.14044)  

**Abstract**: Large language models (LLMs) show strong reasoning abilities across diverse tasks, yet their performance on extended contexts remains inconsistent. While prior research has emphasized mid-context degradation in question answering, this study examines the impact of context in LLM-based fact verification. Using three datasets (HOVER, FEVEROUS, and ClimateFEVER) and five open-source models accross different parameters sizes (7B, 32B and 70B parameters) and model families (Llama-3.1, Qwen2.5 and Qwen3), we evaluate both parametric factual knowledge and the impact of evidence placement across varying context lengths. We find that LLMs exhibit non-trivial parametric knowledge of factual claims and that their verification accuracy generally declines as context length increases. Similarly to what has been shown in previous works, in-context evidence placement plays a critical role with accuracy being consistently higher when relevant evidence appears near the beginning or end of the prompt and lower when placed mid-context. These results underscore the importance of prompt structure in retrieval-augmented fact-checking systems. 

---
# Geometry-Preserving Aggregation for Mixture-of-Experts Embedding Models 

**Authors**: Sajjad Kachuee, Mohammad Sharifkhani  

**Link**: [PDF](https://arxiv.org/pdf/2602.14039)  

**Abstract**: Mixture-of-Experts (MoE) embedding models combine expert outputs using weighted linear summation, implicitly assuming a linear subspace structure in the embedding space. This assumption is shown to be inconsistent with the geometry of expert representations. Geometric analysis of a modern MoE embedding model reveals that expert outputs lie on a shared hyperspherical manifold characterized by tightly concentrated norms and substantial angular separation. Under this geometry, linear aggregation induces inward collapse toward the manifold interior, distorting vector magnitude and direction and reducing embedding comparability. To address this inconsistency, Spherical Barycentric Aggregation (SBA) is introduced as a geometry-preserving aggregation operator that separates radial and angular components to maintain hyperspherical structure while remaining fully compatible with existing routing mechanisms. Experiments on selected tasks from the Massive Text Embedding Benchmark (MTEB), including semantic similarity, clustering, and duplicate question detection, demonstrate consistent performance improvements with identical training cost and full stability. Additional geometric analyses confirm that SBA prevents aggregation-induced collapse and preserves hyperspherical consistency, highlighting the importance of geometry-aware aggregation in MoE embedding architectures. 

---
# GRRM: Group Relative Reward Modeling for Machine Translation 

**Authors**: Sen Yang, Shanbo Cheng, Lu Xu, Jianbing Zhang, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14028)  

**Abstract**: While Group Relative Policy Optimization (GRPO) offers a powerful framework for LLM post-training, its effectiveness in open-ended domains like Machine Translation hinges on accurate intra-group ranking. We identify that standard Scalar Quality Metrics (SQM) fall short in this context; by evaluating candidates in isolation, they lack the comparative context necessary to distinguish fine-grained linguistic nuances. To address this, we introduce the Group Quality Metric (GQM) paradigm and instantiate it via the Group Relative Reward Model (GRRM). Unlike traditional independent scorers, GRRM processes the entire candidate group jointly, leveraging comparative analysis to rigorously resolve relative quality and adaptive granularity. Empirical evaluations confirm that GRRM achieves competitive ranking accuracy among all baselines. Building on this foundation, we integrate GRRM into the GRPO training loop to optimize the translation policy. Experimental results demonstrate that our framework not only improves general translation quality but also unlocks reasoning capabilities comparable to state-of-the-art reasoning models. We release codes, datasets, and model checkpoints at this https URL. 

---
# Named Entity Recognition for Payment Data Using NLP 

**Authors**: Srikumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2602.14009)  

**Abstract**: Named Entity Recognition (NER) has emerged as a critical component in automating financial transaction processing, particularly in extracting structured information from unstructured payment data. This paper presents a comprehensive analysis of state-of-the-art NER algorithms specifically designed for payment data extraction, including Conditional Random Fields (CRF), Bidirectional Long Short-Term Memory with CRF (BiLSTM-CRF), and transformer-based models such as BERT and FinBERT. We conduct extensive experiments on a dataset of 50,000 annotated payment transactions across multiple payment formats including SWIFT MT103, ISO 20022, and domestic payment systems. Our experimental results demonstrate that fine-tuned BERT models achieve an F1-score of 94.2% for entity extraction, outperforming traditional CRF-based approaches by 12.8 percentage points. Furthermore, we introduce PaymentBERT, a novel hybrid architecture combining domain-specific financial embeddings with contextual representations, achieving state-of-the-art performance with 95.7% F1-score while maintaining real-time processing capabilities. We provide detailed analysis of cross-format generalization, ablation studies, and deployment considerations. This research provides practical insights for financial institutions implementing automated sanctions screening, anti-money laundering (AML) compliance, and payment processing systems. 

---
# The Sufficiency-Conciseness Trade-off in LLM Self-Explanation from an Information Bottleneck Perspective 

**Authors**: Ali Zahedzadeh, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2602.14002)  

**Abstract**: Large Language Models increasingly rely on self-explanations, such as chain of thought reasoning, to improve performance on multi step question answering. While these explanations enhance accuracy, they are often verbose and costly to generate, raising the question of how much explanation is truly necessary. In this paper, we examine the trade-off between sufficiency, defined as the ability of an explanation to justify the correct answer, and conciseness, defined as the reduction in explanation length. Building on the information bottleneck principle, we conceptualize explanations as compressed representations that retain only the information essential for producing correct this http URL operationalize this view, we introduce an evaluation pipeline that constrains explanation length and assesses sufficiency using multiple language models on the ARC Challenge dataset. To broaden the scope, we conduct experiments in both English, using the original dataset, and Persian, as a resource-limited language through translation. Our experiments show that more concise explanations often remain sufficient, preserving accuracy while substantially reducing explanation length, whereas excessive compression leads to performance degradation. 

---
# Chain-of-Thought Reasoning with Large Language Models for Clinical Alzheimer's Disease Assessment and Diagnosis 

**Authors**: Tongze Zhang, Jun-En Ding, Melik Ozolcer, Fang-Ming Hung, Albert Chih-Chieh Yang, Feng Liu, Yi-Rou Ji, Sang Won Bae  

**Link**: [PDF](https://arxiv.org/pdf/2602.13979)  

**Abstract**: Alzheimer's disease (AD) has become a prevalent neurodegenerative disease worldwide. Traditional diagnosis still relies heavily on medical imaging and clinical assessment by physicians, which is often time-consuming and resource-intensive in terms of both human expertise and healthcare resources. In recent years, large language models (LLMs) have been increasingly applied to the medical field using electronic health records (EHRs), yet their application in Alzheimer's disease assessment remains limited, particularly given that AD involves complex multifactorial etiologies that are difficult to observe directly through imaging modalities. In this work, we propose leveraging LLMs to perform Chain-of-Thought (CoT) reasoning on patients' clinical EHRs. Unlike direct fine-tuning of LLMs on EHR data for AD classification, our approach utilizes LLM-generated CoT reasoning paths to provide the model with explicit diagnostic rationale for AD assessment, followed by structured CoT-based predictions. This pipeline not only enhances the model's ability to diagnose intrinsically complex factors but also improves the interpretability of the prediction process across different stages of AD progression. Experimental results demonstrate that the proposed CoT-based diagnostic framework significantly enhances stability and diagnostic performance across multiple CDR grading tasks, achieving up to a 15% improvement in F1 score compared to the zero-shot baseline method. 

---
# HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam 

**Authors**: Weiqi Zhai, Zhihai Wang, Jinghang Wang, Boyu Yang, Xiaogang Li, Xiang Xu, Bohan Wang, Peng Wang, Xingzhe Wu, Anfeng Li, Qiyuan Feng, Yuhao Zhou, Shoulin Han, Wenjie Luo, Yiyuan Li, Yaxuan Wang, Ruixian Luo, Guojie Lin, Peiyao Xiao, Chengliang Xu, Ben Wang, Zeyu Wang, Zichao Chen, Jianan Ye, Yijie Hu, Jialong Chen, Zongwen Shen, Yuliang Xu, An Yang, Bowen Yu, Dayiheng Liu, Junyang Lin, Hu Wei, Que Shen, Bing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.13964)  

**Abstract**: Humanity's Last Exam (HLE) has become a widely used benchmark for evaluating frontier large language models on challenging, multi-domain questions. However, community-led analyses have raised concerns that HLE contains a non-trivial number of noisy items, which can bias evaluation results and distort cross-model comparisons. To address this challenge, we introduce HLE-Verified, a verified and revised version of HLE with a transparent verification protocol and fine-grained error taxonomy. Our construction follows a two-stage validation-and-repair workflow resulting in a certified benchmark. In Stage I, each item undergoes binary validation of the problem and final answer through domain-expert review and model-based cross-checks, yielding 641 verified items. In Stage II, flawed but fixable items are revised under strict constraints preserving the original evaluation intent, through dual independent expert repairs, model-assisted auditing, and final adjudication, resulting in 1,170 revised-and-certified items. The remaining 689 items are released as a documented uncertain set with explicit uncertainty sources and expertise tags for future refinement. We evaluate seven state-of-the-art language models on HLE and HLE-Verified, observing an average absolute accuracy gain of 7--10 percentage points on HLE-Verified. The improvement is particularly pronounced on items where the original problem statement and/or reference answer is erroneous, with gains of 30--40 percentage points. Our analyses further reveal a strong association between model confidence and the presence of errors in the problem statement or reference answer, supporting the effectiveness of our revisions. Overall, HLE-Verified improves HLE-style evaluations by reducing annotation noise and enabling more faithful measurement of model capabilities. Data is available at: this https URL 

---
# Pre-Editorial Normalization for Automatically Transcribed Medieval Manuscripts in Old French and Latin 

**Authors**: Thibault Clérice, Rachel Bawden, Anthony Glaise, Ariane Pinche, David Smith  

**Link**: [PDF](https://arxiv.org/pdf/2602.13905)  

**Abstract**: Recent advances in Automatic Text Recognition (ATR) have improved access to historical archives, yet a methodological divide persists between palaeographic transcriptions and normalized digital editions. While ATR models trained on more palaeographically-oriented datasets such as CATMuS have shown greater generalizability, their raw outputs remain poorly compatible with most readers and downstream NLP tools, thus creating a usability gap. On the other hand, ATR models trained to produce normalized outputs have been shown to struggle to adapt to new domains and tend to over-normalize and hallucinate. We introduce the task of Pre-Editorial Normalization (PEN), which consists in normalizing graphemic ATR output according to editorial conventions, which has the advantage of keeping an intermediate step with palaeographic fidelity while providing a normalized version for practical usability. We present a new dataset derived from the CoMMA corpus and aligned with digitized Old French and Latin editions using passim. We also produce a manually corrected gold-standard evaluation set. We benchmark this resource using ByT5-based sequence-to-sequence models on normalization and pre-annotation tasks. Our contributions include the formal definition of PEN, a 4.66M-sample silver training corpus, a 1.8k-sample gold evaluation set, and a normalization model achieving a 6.7% CER, substantially outperforming previous models for this task. 

---
# Evaluating Prompt Engineering Techniques for RAG in Small Language Models: A Multi-Hop QA Approach 

**Authors**: Amir Hossein Mohammadi, Ali Moeinian, Zahra Razavizade, Afsaneh Fatemi, Reza Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2602.13890)  

**Abstract**: Retrieval Augmented Generation (RAG) is a powerful approach for enhancing the factual grounding of language models by integrating external knowledge. While widely studied for large language models, the optimization of RAG for Small Language Models (SLMs) remains a critical research gap, particularly in complex, multi-hop question-answering tasks that require sophisticated reasoning. In these systems, prompt template design is a crucial yet under-explored factor influencing performance. This paper presents a large-scale empirical study to investigate this factor, evaluating 24 different prompt templates on the HotpotQA dataset. The set includes a standard RAG prompt, nine well-formed techniques from the literature, and 14 novel hybrid variants, all tested on two prominent SLMs: Qwen2.5-3B Instruct and Gemma3-4B-It. Our findings, based on a test set of 18720 instances, reveal significant performance gains of up to 83% on Qwen2.5 and 84.5% on Gemma3-4B-It, yielding an improvement of up to 6% for both models compared to the Standard RAG prompt. This research also offers concrete analysis and actionable recommendations for designing effective and efficient prompts for SLM-based RAG systems, practically for deployment in resource-constrained environments. 

---
# ADAB: Arabic Dataset for Automated Politeness Benchmarking -- A Large-Scale Resource for Computational Sociopragmatics 

**Authors**: Hend Al-Khalifa, Nadia Ghezaiel, Maria Bounnit, Hend Hamed Alhazmi, Noof Abdullah Alfear, Reem Fahad Alqifari, Ameera Masoud Almasoud, Sharefah Ahmed Al-Ghamdi  

**Link**: [PDF](https://arxiv.org/pdf/2602.13870)  

**Abstract**: The growing importance of culturally-aware natural language processing systems has led to an increasing demand for resources that capture sociopragmatic phenomena across diverse languages. Nevertheless, Arabic-language resources for politeness detection remain under-explored, despite the rich and complex politeness expressions embedded in Arabic communication. In this paper, we introduce ADAB (Arabic Politeness Dataset), a new annotated Arabic dataset collected from four online platforms, including social media, e-commerce, and customer service domains, covering Modern Standard Arabic and multiple dialects (Gulf, Egyptian, Levantine, and Maghrebi). The dataset was annotated based on Arabic linguistic traditions and pragmatic theory, resulting in three classes: polite, impolite, and neutral. It contains 10,000 samples with linguistic feature annotations across 16 politeness categories and achieves substantial inter-annotator agreement (kappa = 0.703). We benchmark 40 model configurations, including traditional machine learning, transformer-based models, and large language models. The dataset aims to support research on politeness-aware Arabic NLP. 

---
# Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages 

**Authors**: Somnath Banerjee, Rima Hazra, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.13867)  

**Abstract**: Large language models (LLMs) are being deployed across the Global South, where everyday use involves low-resource languages, code-mixing, and culturally specific norms. Yet safety pipelines, benchmarks, and alignment still largely target English and a handful of high-resource languages, implicitly assuming safety and factuality ''transfer'' across languages. Evidence increasingly shows they do not. We synthesize recent findings indicating that (i) safety guardrails weaken sharply on low-resource and code-mixed inputs, (ii) culturally harmful behavior can persist even when standard toxicity scores look acceptable, and (iii) English-only knowledge edits and safety patches often fail to carry over to low-resource languages. In response, we outline a practical agenda for researchers and students in the Global South: parameter-efficient safety steering, culturally grounded evaluation and preference data, and participatory workflows that empower local communities to define and mitigate harm. Our aim is to make multilingual safety a core requirement-not an add-on-for equitable AI in underrepresented regions. 

---
# Tutoring Large Language Models to be Domain-adaptive, Precise, and Safe 

**Authors**: Somnath Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.13860)  

**Abstract**: The overarching research direction of this work is the development of a ''Responsible Intelligence'' framework designed to reconcile the immense generative power of Large Language Models (LLMs) with the stringent requirements of real-world deployment. As these models become a transformative force in artificial intelligence, there is an urgent need to move beyond general-purpose architectures toward systems that are contextually aware, inherently safer, and deeply respectful of global cultural nuances. This research navigates three interconnected threads: domain adaptation to ensure technical precision, ethical rigor to mitigate adversarial vulnerabilities, and cultural/multilingual alignment to promote global inclusivity. The methodological trajectory moves from classical supervised adaptation for task-specific demands to decoding-time alignment for safety, finally leveraging human feedback and preference modeling to achieve sociolinguistic acuity. 

---
# PrivAct: Internalizing Contextual Privacy Preservation via Multi-Agent Preference Training 

**Authors**: Yuhan Cheng, Hancheng Ye, Hai Helen Li, Jingwei Sun, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.13840)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in personalized tasks involving sensitive, context-dependent information, where privacy violations may arise in agents' action due to the implicitness of contextual privacy. Existing approaches rely on external, inference-time interventions which are brittle, scenario-specific, and may expand the privacy attack surface. We propose PrivAct, a contextual privacy-aware multi-agent learning framework that internalizes contextual privacy preservation directly into models' generation behavior for privacy-compliant agentic actions. By embedding privacy preferences into each agent, PrivAct enhances system-wide contextual integrity while achieving a more favorable privacy-helpfulness tradeoff. Experiments across multiple LLM backbones and benchmarks demonstrate consistent improvements in contextual privacy preservation, reducing leakage rates by up to 12.32% while maintaining comparable helpfulness, as well as zero-shot generalization and robustness across diverse multi-agent topologies. Code is available at this https URL. 

---
# Speculative Decoding with a Speculative Vocabulary 

**Authors**: Miles Williams, Young D. Kwon, Rui Li, Alexandros Kouris, Stylianos I. Venieris  

**Link**: [PDF](https://arxiv.org/pdf/2602.13836)  

**Abstract**: Speculative decoding has rapidly emerged as a leading approach for accelerating language model (LM) inference, as it offers substantial speedups while yielding identical outputs. This relies upon a small draft model, tasked with predicting the outputs of the target model. State-of-the-art speculative decoding methods use a draft model consisting of a single decoder layer and output embedding matrix, with the latter dominating drafting time for the latest LMs. Recent work has sought to address this output distribution bottleneck by reducing the vocabulary of the draft model. Although this can improve throughput, it compromises speculation effectiveness when the target token is out-of-vocabulary. In this paper, we argue for vocabulary speculation as an alternative to a reduced vocabulary. We propose SpecVocab, an efficient and effective method that selects a vocabulary subset per decoding step. Across a variety of tasks, we demonstrate that SpecVocab can achieve a higher acceptance length than state-of-the-art speculative decoding approach, EAGLE-3. Notably, this yields up to an 8.1% increase in average throughput over EAGLE-3. 

---
# Beyond Words: Evaluating and Bridging Epistemic Divergence in User-Agent Interaction via Theory of Mind 

**Authors**: Minyuan Ruan, Ziyue Wang, Kaiming Liu, Yunghwei Lai, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.13832)  

**Abstract**: Large Language Models (LLMs) have developed rapidly and are widely applied to both general-purpose and professional tasks to assist human users. However, they still struggle to comprehend and respond to the true user needs when intentions and instructions are imprecisely conveyed, leading to a divergence between subjective user believes and true environment states. Resolving this epistemic divergence requires Theory of Mind (ToM), yet existing ToM evaluations for LLMs primarily focus on isolated belief inference, overlooking its functional utility in real-world interaction. To this end, we formalize ToM for LLMs as a mechanism for epistemic divergence detection and resolution, and propose a benchmark, \benchname, to assess how models reconcile user beliefs and profiles in practice. Results across 11 leading models reveal a significant limitation to identify underlying cognitive gaps that impede task success. To bridge this gap, we further curate a trajectory-based ToM dataset linking belief tracking with task-related state inference. The model trained on this data via reinforcement learning shows consistent improvement in reasoning about user mental states, leading to enhanced downstream performance. Our work highlights the practical value of ToM as an essential interaction-level mechanism rather than as a standalone reasoning skill. 

---
# The acquisition of English irregular inflections by Yemeni L1 Arabic learners: A Universal Grammar approach 

**Authors**: Muneef Y. Alsawsh, Mohammed Q. Shormani  

**Link**: [PDF](https://arxiv.org/pdf/2602.13816)  

**Abstract**: This study examines the acquisition of English irregular inflections by Yemeni learners of English as a second language (L2), utilizing a Universal Grammar (UG) approach. Within the UG approach, the study considers Feature Reassembly Hypothesis (FRH) (Lardiere, 2008, 2009) part of UG, focusing on the roles of first language (L1) transfer and L2 developmental influence. It analyzes learner errors across two developmental stages. Stage 1 data reveal a dominant influence of L1 transfer, particularly in phonological and structural mismatches, while stage 2 data demonstrate increased learner sensitivity to UG properties and morphological reconfiguration toward the target language. Findings reveal that errors in irregular inflectional morphology are attributed to both interlingual and intralingual sources, with overgeneralization of L2 rules as a common developmental strategy. Statistical analysis, including a one-way ANOVA, indicates significant improvement in the production of well-formed irregular inflections from stage 1 to stage 2, underscoring learners' continued access to UG. However, persistent difficulties with consonant change, zero-morpheme, and -a plural inflections suggest that limited exposure, ineffective input modeling, and insufficient instructional quality constrain full UG access. The study concludes that while L1 transfer and L2 developmental factors influence initial stages of acquisition, appropriate linguistic input and instruction are critical for facilitating UG-driven feature reassembly in adult L2 learners. 

---
# OMGs: A multi-agent system supporting MDT decision-making across the ovarian tumour care continuum 

**Authors**: Yangyang Zhang, Zilong Wang, Jianbo Xu, Yongqi Chen, Chu Han, Zhihao Zhang, Shuai Liu, Hui Li, Huiping Zhang, Ziqi Liu, Jiaxin Chen, Jun Zhu, Zheng Feng, Hao Wen, Xingzhu Ju, Yanping Zhong, Yunqiu Zhang, Jie Duan, Jun Li, Dongsheng Li, Weijie Wang, Haiyan Zhu, Wei Jiang, Xiaohua Wu, Shuo Wang, Haiming Li, Qinhao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.13793)  

**Abstract**: Ovarian tumour management has increasingly relied on multidisciplinary tumour board (MDT) deliberation to address treatment complexity and disease heterogeneity. However, most patients worldwide lack access to timely expert consensus, particularly in resource-constrained centres where MDT resources are scarce or unavailable. Here we present OMGs (Ovarian tumour Multidisciplinary intelligent aGent System), a multi-agent AI framework where domain-specific agents deliberate collaboratively to integrate multidisciplinary evidence and generate MDT-style recommendations with transparent rationales. To systematically evaluate MDT recommendation quality, we developed SPEAR (Safety, Personalization, Evidence, Actionability, Robustness) and validated OMGs across diverse clinical scenarios spanning the care continuum. In multicentre re-evaluation, OMGs achieved performance comparable to expert MDT consensus ($4.45 \pm 0.30$ versus $4.53 \pm 0.23$), with higher Evidence scores (4.57 versus 3.92). In prospective multicentre evaluation (59 patients), OMGs demonstrated high concordance with routine MDT decisions. Critically, in paired human-AI studies, OMGs most substantially enhanced clinicians' recommendations in Evidence and Robustness, the dimensions most compromised when multidisciplinary expertise is unavailable. These findings suggest that multi-agent deliberative systems can achieve performance comparable to expert MDT consensus, with potential to expand access to specialized oncology expertise in resource-limited settings. 

---
# How Do Lexical Senses Correspond Between Spoken German and German Sign Language? 

**Authors**: Melis Çelikkol, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.13790)  

**Abstract**: Sign language lexicographers construct bilingual dictionaries by establishing word-to-sign mappings, where polysemous and homonymous words corresponding to different signs across contexts are often underrepresented. A usage-based approach examining how word senses map to signs can identify such novel mappings absent from current dictionaries, enriching lexicographic resources. We address this by analyzing German and German Sign Language (Deutsche Gebärdensprache, DGS), manually annotating 1,404 word use-to-sign ID mappings derived from 32 words from the German Word Usage Graph (D-WUG) and 49 signs from the Digital Dictionary of German Sign Language (DW-DGS). We identify three correspondence types: Type 1 (one-to-many), Type 2 (many-to-one), and Type 3 (one-to-one), plus No Match cases. We evaluate computational methods: Exact Match (EM) and Semantic Similarity (SS) using SBERT embeddings. SS substantially outperforms EM overall 88.52% vs. 71.31%), with dramatic gains for Type 1 (+52.1 pp). Our work establishes the first annotated dataset for cross-modal sense correspondence and reveals which correspondence patterns are computationally identifiable. Our code and dataset are made publicly available. 

---
# RMPL: Relation-aware Multi-task Progressive Learning with Stage-wise Training for Multimedia Event Extraction 

**Authors**: Yongkang Jin, Jianwen Luo, Jingjing Wang, Jianmin Yao, Yu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2602.13748)  

**Abstract**: Multimedia Event Extraction (MEE) aims to identify events and their arguments from documents that contain both text and images. It requires grounding event semantics across different modalities. Progress in MEE is limited by the lack of annotated training data. M2E2 is the only established benchmark, but it provides annotations only for evaluation. This makes direct supervised training impractical. Existing methods mainly rely on cross-modal alignment or inference-time prompting with Vision--Language Models (VLMs). These approaches do not explicitly learn structured event representations and often produce weak argument grounding in multimodal settings. To address these limitations, we propose RMPL, a Relation-aware Multi-task Progressive Learning framework for MEE under low-resource conditions. RMPL incorporates heterogeneous supervision from unimodal event extraction and multimedia relation extraction with stage-wise training. The model is first trained with a unified schema to learn shared event-centric representations across modalities. It is then fine-tuned for event mention identification and argument role extraction using mixed textual and visual data. Experiments on the M2E2 benchmark with multiple VLMs show consistent improvements across different modality settings. 

---
# On Theoretically-Driven LLM Agents for Multi-Dimensional Discourse Analysis 

**Authors**: Maciej Uberna, Michał Wawer, Jarosław A. Chudziak, Marcin Koszowy  

**Link**: [PDF](https://arxiv.org/pdf/2602.13713)  

**Abstract**: Identifying the strategic uses of reformulation in discourse remains a key challenge for computational argumentation. While LLMs can detect surface-level similarity, they often fail to capture the pragmatic functions of rephrasing, such as its role within rhetorical discourse. This paper presents a comparative multi-agent framework designed to quantify the benefits of incorporating explicit theoretical knowledge for this task. We utilise an dataset of annotated political debates to establish a new standard encompassing four distinct rephrase functions: Deintensification, Intensification, Specification, Generalisation, and Other, which covers all remaining types (D-I-S-G-O). We then evaluate two parallel LLM-based agent systems: one enhanced by argumentation theory via Retrieval-Augmented Generation (RAG), and an identical zero-shot baseline. The results reveal a clear performance gap: the RAG-enhanced agents substantially outperform the baseline across the board, with particularly strong advantages in detecting Intensification and Generalisation context, yielding an overall Macro F1-score improvement of nearly 30\%. Our findings provide evidence that theoretical grounding is not only beneficial but essential for advancing beyond mere paraphrase detection towards function-aware analysis of argumentative discourse. This comparative multi-agent architecture represents a step towards scalable, theoretically informed computational tools capable of identifying rhetorical strategies in contemporary discourse. 

---
# Metaphors' journeys across time and genre: tracking the evolution of literary metaphors with temporal embeddings 

**Authors**: Veronica Mangiaterra, Chiara Barattieri di San Pietro, Paolo Canal, Valentina Bambini  

**Link**: [PDF](https://arxiv.org/pdf/2602.13701)  

**Abstract**: Metaphors are a distinctive feature of literary language, yet they remain less studied experimentally than everyday metaphors. Moreover, previous psycholinguistic and computational approaches overlooked the temporal dimension, although many literary metaphors were coined centuries apart from contemporary readers. This study innovatively applies tools from diachronic distributional semantics to assess whether the processing costs of literary metaphors varied over time and genre. Specifically, we trained word embeddings on literary and nonliterary Italian corpora from the 19th and 21st centuries, for a total of 124 million tokens, and modeled changes in the semantic similarity between topics and vehicles of 515 19th-century literary metaphors, taking this measure as a proxy of metaphor processing demands. Overall, semantic similarity, and hence metaphor processing demands, remained stable over time. However, genre played a key role: metaphors appeared more difficult (i.e., lower topic-vehicle similarity) in modern literary contexts than in 19th-century literature, but easier (i.e., higher topic-vehicle similarity) in today's nonliterary language (e.g., the Web) than in 19th-century nonliterary texts. This pattern was further shaped by semantic features of metaphors' individual terms, such as vector coherence and semantic neighborhood density. Collectively, these findings align with broader linguistic changes in Italian, such as the stylistic simplification of modern literature, which may have increased metaphor processing demands, and the high creativity of the Web's language, which seems to render metaphor more accessible. 

---
# Elo-Evolve: A Co-evolutionary Framework for Language Model Alignment 

**Authors**: Jing Zhao, Ting Zhen, Junwei bao, Hongfei Jiang, Yang song  

**Link**: [PDF](https://arxiv.org/pdf/2602.13575)  

**Abstract**: Current alignment methods for Large Language Models (LLMs) rely on compressing vast amounts of human preference data into static, absolute reward functions, leading to data scarcity, noise sensitivity, and training instability. We introduce Elo-Evolve, a co-evolutionary framework that redefines alignment as dynamic multi-agent competition within an adaptive opponent pool. Our approach makes two key innovations: (1) eliminating Bradley-Terry model dependencies by learning directly from binary win/loss outcomes in pairwise competitions, and (2) implementing Elo-orchestrated opponent selection that provides automatic curriculum learning through temperature-controlled sampling. We ground our approach in PAC learning theory, demonstrating that pairwise comparison achieves superior sample complexity and empirically validate a 4.5x noise reduction compared to absolute scoring approaches. Experimentally, we train a Qwen2.5-7B model using our framework with opponents including Qwen2.5-14B, Qwen2.5-32B, and Qwen3-8B models. Results demonstrate a clear performance hierarchy: point-based methods < static pairwise training < Elo-Evolve across Alpaca Eval 2.0 and MT-Bench, validating the progressive benefits of pairwise comparison and dynamic opponent selection for LLM alignment. 

---
# LLM-Confidence Reranker: A Training-Free Approach for Enhancing Retrieval-Augmented Generation Systems 

**Authors**: Zhipeng Song, Xiangyu Kong, Xinrui Bao, Yizhi Zhou, Jiulong Jiao, Sitong Liu, Yuhang Zhou, Heng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.13571)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet hallucinations in knowledge-intensive tasks remain a critical challenge. Retrieval-augmented generation (RAG) addresses this by integrating external knowledge, but its efficacy depends on accurate document retrieval and ranking. Although existing rerankers demonstrate effectiveness, they frequently necessitate specialized training, impose substantial computational expenses, and fail to fully exploit the semantic capabilities of LLMs, particularly their inherent confidence signals. We propose the LLM-Confidence Reranker (LCR), a training-free, plug-and-play algorithm that enhances reranking in RAG systems by leveraging black-box LLM confidence derived from Maximum Semantic Cluster Proportion (MSCP). LCR employs a two-stage process: confidence assessment via multinomial sampling and clustering, followed by binning and multi-level sorting based on query and document confidence thresholds. This approach prioritizes relevant documents while preserving original rankings for high-confidence queries, ensuring robustness. Evaluated on BEIR and TREC benchmarks with BM25 and Contriever retrievers, LCR--using only 7--9B-parameter pre-trained LLMs--consistently improves NDCG@5 by up to 20.6% across pre-trained LLM and fine-tuned Transformer rerankers, without degradation. Ablation studies validate the hypothesis that LLM confidence positively correlates with document relevance, elucidating LCR's mechanism. LCR offers computational efficiency, parallelism for scalability, and broad compatibility, mitigating hallucinations in applications like medical diagnosis. 

---
# DistillLens: Symmetric Knowledge Distillation Through Logit Lens 

**Authors**: Manish Dhakal, Uthman Jinadu, Anjila Budathoki, Rajshekhar Sunderraman, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.13567)  

**Abstract**: Standard Knowledge Distillation (KD) compresses Large Language Models (LLMs) by optimizing final outputs, yet it typically treats the teacher's intermediate layer's thought process as a black box. While feature-based distillation attempts to bridge this gap, existing methods (e.g., MSE and asymmetric KL divergence) ignore the rich uncertainty profiles required for the final output. In this paper, we introduce DistillLens, a framework that symmetrically aligns the evolving thought processes of student and teacher models. By projecting intermediate hidden states into the vocabulary space via the Logit Lens, we enforce structural alignment using a symmetric divergence objective. Our analysis proves that this constraint imposes a dual-sided penalty, preventing both overconfidence and underconfidence while preserving the high-entropy information conduits essential for final deduction. Extensive experiments on GPT-2 and Llama architectures demonstrate that DistillLens consistently outperforms standard KD and feature-transfer baselines on diverse instruction-following benchmarks. The code is available at this https URL. 

---
# Small Reward Models via Backward Inference 

**Authors**: Yike Wang, Faeze Brahman, Shangbin Feng, Teng Xiao, Hannaneh Hajishirzi, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2602.13551)  

**Abstract**: Reward models (RMs) play a central role throughout the language model (LM) pipeline, particularly in non-verifiable domains. However, the dominant LLM-as-a-Judge paradigm relies on the strong reasoning capabilities of large models, while alternative approaches require reference responses or explicit rubrics, limiting flexibility and broader accessibility. In this work, we propose FLIP (FLipped Inference for Prompt reconstruction), a reference-free and rubric-free reward modeling approach that reformulates reward modeling through backward inference: inferring the instruction that would most plausibly produce a given response. The similarity between the inferred and the original instructions is then used as the reward signal. Evaluations across four domains using 13 small language models show that FLIP outperforms LLM-as-a-Judge baselines by an average of 79.6%. Moreover, FLIP substantially improves downstream performance in extrinsic evaluations under test-time scaling via parallel sampling and GRPO training. We further find that FLIP is particularly effective for longer outputs and robust to common forms of reward hacking. By explicitly exploiting the validation-generation gap, FLIP enables reliable reward modeling in downscaled regimes where judgment methods fail. Code available at this https URL. 

---
# On Calibration of Large Language Models: From Response To Capability 

**Authors**: Sin-Han Yang, Cheng-Kuang Wu, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.13540)  

**Abstract**: Large language models (LLMs) are widely deployed as general-purpose problem solvers, making accurate confidence estimation critical for reliable use. Prior work on LLM calibration largely focuses on response-level confidence, which estimates the correctness of a single generated output. However, this formulation is misaligned with many practical settings where the central question is how likely a model is to solve a query overall. We show that this mismatch results from the stochastic nature of modern LLM decoding, under which single-response correctness fails to reflect underlying model capability. To address this issue, we introduce capability calibration, which targets the model's expected accuracy on a query. We formally distinguish capability calibration from response calibration and show that the two differ both theoretically and empirically. We establish an empirical evaluation setup and study a range of confidence estimation methods. Our results demonstrate that capability-calibrated confidence improves pass@$k$ prediction and inference budget allocation, establishing a foundation with potential for diverse applications. 

---
# Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens 

**Authors**: Wei-Lin Chen, Liqian Peng, Tian Tan, Chao Zhao, Blake JianHang Chen, Ziqian Lin, Alec Go, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2602.13517)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning capabilities by scaling test-time compute via long Chain-of-Thought (CoT). However, recent findings suggest that raw token counts are unreliable proxies for reasoning quality: increased generation length does not consistently correlate with accuracy and may instead signal "overthinking," leading to performance degradation. In this work, we quantify inference-time effort by identifying deep-thinking tokens -- tokens where internal predictions undergo significant revisions in deeper model layers prior to convergence. Across four challenging mathematical and scientific benchmarks (AIME 24/25, HMMT 25, and GPQA-diamond) and a diverse set of reasoning-focused models (GPT-OSS, DeepSeek-R1, and Qwen3), we show that deep-thinking ratio (the proportion of deep-thinking tokens in a generated sequence) exhibits a robust and consistently positive correlation with accuracy, substantially outperforming both length-based and confidence-based baselines. Leveraging this insight, we introduce Think@n, a test-time scaling strategy that prioritizes samples with high deep-thinking ratios. We demonstrate that Think@n matches or exceeds standard self-consistency performance while significantly reducing inference costs by enabling the early rejection of unpromising generations based on short prefixes. 

---
# From Perceptions To Evidence: Detecting AI-Generated Content In Turkish News Media With A Fine-Tuned Bert Classifier 

**Authors**: Ozancan Ozdemir  

**Link**: [PDF](https://arxiv.org/pdf/2602.13504)  

**Abstract**: The rapid integration of large language models into newsroom workflows has raised urgent questions about the prevalence of AI-generated content in online media. While computational studies have begun to quantify this phenomenon in English-language outlets, no empirical investigation exists for Turkish news media, where existing research remains limited to qualitative interviews with journalists or fake news detection. This study addresses that gap by fine-tuning a Turkish-specific BERT model (dbmdz/bert-base-turkish-cased) on a labeled dataset of 3,600 articles from three major Turkish outlets with distinct editorial orientations for binary classification of AI-rewritten content. The model achieves 0.9708 F1 score on the held-out test set with symmetric precision and recall across both classes. Subsequent deployment on over 3,500 unseen articles spanning between 2023 and 2026 reveals consistent cross-source and temporally stable classification patterns, with mean prediction confidence exceeding 0.96 and an estimated 2.5 percentage of examined news content rewritten or revised by LLMs on average. To the best of our knowledge, this is the first study to move beyond self-reported journalist perceptions toward empirical, data-driven measurement of AI usage in Turkish news media. 

---
# Language Model Memory and Memory Models for Language 

**Authors**: Benjamin L. Badger  

**Link**: [PDF](https://arxiv.org/pdf/2602.13466)  

**Abstract**: The ability of machine learning models to store input information in hidden layer vector embeddings, analogous to the concept of `memory', is widely employed but not well characterized. We find that language model embeddings typically contain relatively little input information regardless of data and compute scale during training. In contrast, embeddings from autoencoders trained for input regeneration are capable of nearly perfect memory formation. The substitution of memory embeddings for token sequences leads to substantial computational efficiencies, motivating the introduction of a parallelizable encoder-decoder memory model architecture. Upon causal training these models contain information-poor embeddings incapable of arbitrary information access, but by combining causal and information retention objective functions they learn to form and decode information-rich memories. Training can be further streamlined by freezing a high fidelity encoder followed by a curriculum training approach where decoders first learn to process memories and then learn to additionally predict next tokens. We introduce the perspective that next token prediction training alone is poorly suited for accurate memory formation as the objective itself is non-invertible, motivating the use of combined objective functions for models where the entire input is not exposed. 

---
# Using Machine Learning to Enhance the Detection of Obfuscated Abusive Words in Swahili: A Focus on Child Safety 

**Authors**: Phyllis Nabangi, Abdul-Jalil Zakaria, Jema David Ndibwile  

**Link**: [PDF](https://arxiv.org/pdf/2602.13455)  

**Abstract**: The rise of digital technology has dramatically increased the potential for cyberbullying and online abuse, necessitating enhanced measures for detection and prevention, especially among children. This study focuses on detecting abusive obfuscated language in Swahili, a low-resource language that poses unique challenges due to its limited linguistic resources and technological support. Swahili is chosen due to its popularity and being the most widely spoken language in Africa, with over 16 million native speakers and upwards of 100 million speakers in total, spanning regions in East Africa and some parts of the Middle East.
We employed machine learning models including Support Vector Machines (SVM), Logistic Regression, and Decision Trees, optimized through rigorous parameter tuning and techniques like Synthetic Minority Over-sampling Technique (SMOTE) to handle data imbalance. Our analysis revealed that, while these models perform well in high-dimensional textual data, our dataset's small size and imbalance limit our findings' generalizability. Precision, recall, and F1 scores were thoroughly analyzed, highlighting the nuanced performance of each model in detecting obfuscated language.
This research contributes to the broader discourse on ensuring safer online environments for children, advocating for expanded datasets and advanced machine-learning techniques to improve the effectiveness of cyberbullying detection systems. Future work will focus on enhancing data robustness, exploring transfer learning, and integrating multimodal data to create more comprehensive and culturally sensitive detection mechanisms. 

---
# LLM-Powered Automatic Translation and Urgency in Crisis Scenarios 

**Authors**: Belu Ticona, Antonis Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2602.13452)  

**Abstract**: Large language models (LLMs) are increasingly proposed for crisis preparedness and response, particularly for multilingual communication. However, their suitability for high-stakes crisis contexts remains insufficiently evaluated. This work examines the performance of state-of-the-art LLMs and machine translation systems in crisis-domain translation, with a focus on preserving urgency, which is a critical property for effective crisis communication and triaging. Using multilingual crisis data and a newly introduced urgency-annotated dataset covering over 32 languages, we show that both dedicated translation models and LLMs exhibit substantial performance degradation and instability. Crucially, even linguistically adequate translations can distort perceived urgency, and LLM-based urgency classifications vary widely depending on the language of the prompt and input. These findings highlight significant risks in deploying general-purpose language technologies for crisis communication and underscore the need for crisis-aware evaluation frameworks. 

---
# Multimodal Consistency-Guided Reference-Free Data Selection for ASR Accent Adaptation 

**Authors**: Ligong Lei, Wenwen Lu, Xudong Pang, Zaokere Kadeer, Aishan Wumaier  

**Link**: [PDF](https://arxiv.org/pdf/2602.13263)  

**Abstract**: Automatic speech recognition (ASR) systems often degrade on accented speech because acoustic-phonetic and prosodic shifts induce a mismatch to training data, making labeled accent adaptation costly. However, common pseudo-label selection heuristics are largely text-centric (e.g., perplexity (PPL) filtering) and can prefer fluent yet acoustically mismatched hypotheses, leading to error amplification when fine-tuning. To address this, we introduce a multimodal consistency-guided, reference-free data selection pipeline for ASR accent adaptation under a transductive, label-free protocol. The pipeline starts with a target-aware preselection step based on submodular mutual information to improve query relevance and reduce downstream computation. It then generates multiple pseudo-transcriptions per utterance via perturbation-based decoding and scores each hypothesis using two reference-free signals: speech--text alignment in a shared embedding space and predicted word error rate (WER). A simple percentile-based selection rule retains reliable pseudo-labels for fine-tuning while discarding noisy utterances. In an in-domain setting, selecting ~1.5k utterances from a 30k pool achieves 10.91% WER, close to 10.45% obtained using 30k supervised labels. In a cross-domain setting with a mismatched candidate pool, consistency-filtered subsets avoid the degradation caused by unfiltered pseudo-labels under strong accent shift, and matched-hour experiments on a stronger ASR backbone further confirm gains over random sampling and recent selection baselines. 

---
# Symmetry in language statistics shapes the geometry of model representations 

**Authors**: Dhruva Karkada, Daniel J. Korchinski, Andres Nava, Matthieu Wyart, Yasaman Bahri  

**Link**: [PDF](https://arxiv.org/pdf/2602.15029)  

**Abstract**: Although learned representations underlie neural networks' success, their fundamental properties remain poorly understood. A striking example is the emergence of simple geometric structures in LLM representations: for example, calendar months organize into a circle, years form a smooth one-dimensional manifold, and cities' latitudes and longitudes can be decoded by a linear probe. We show that the statistics of language exhibit a translation symmetry -- e.g., the co-occurrence probability of two months depends only on the time interval between them -- and we prove that the latter governs the aforementioned geometric structures in high-dimensional word embedding models. Moreover, we find that these structures persist even when the co-occurrence statistics are strongly perturbed (for example, by removing all sentences in which two months appear together) and at moderate embedding dimension. We show that this robustness naturally emerges if the co-occurrence statistics are collectively controlled by an underlying continuous latent variable. We empirically validate this theoretical framework in word embedding models, text embedding models, and large language models. 

---
# Scaling Beyond Masked Diffusion Language Models 

**Authors**: Subham Sekhar Sahoo, Jean-Marie Lemercier, Zhihan Yang, Justin Deschenaux, Jingyu Liu, John Thickstun, Ante Jukic  

**Link**: [PDF](https://arxiv.org/pdf/2602.15014)  

**Abstract**: Diffusion language models are a promising alternative to autoregressive models due to their potential for faster generation. Among discrete diffusion approaches, Masked diffusion currently dominates, largely driven by strong perplexity on language modeling benchmarks. In this work, we present the first scaling law study of uniform-state and interpolating discrete diffusion methods. We also show that Masked diffusion models can be made approximately 12% more FLOPs-efficient when trained with a simple cross-entropy objective. We find that perplexity is informative within a diffusion family but can be misleading across families, where models with worse likelihood scaling may be preferable due to faster and more practical sampling, as reflected by the speed-quality Pareto frontier. These results challenge the view that Masked diffusion is categorically the future of diffusion language modeling and that perplexity alone suffices for cross-algorithm comparison. Scaling all methods to 1.7B parameters, we show that uniform-state diffusion remains competitive on likelihood-based benchmarks and outperforms autoregressive and Masked diffusion models on GSM8K, despite worse validation perplexity. We provide the code, model checkpoints, and video tutorials on the project page: this http URL 

---
# Learning State-Tracking from Code Using Linear RNNs 

**Authors**: Julien Siems, Riccardo Grazzi, Kirill Kalinin, Hitesh Ballani, Babak Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2602.14814)  

**Abstract**: Over the last years, state-tracking tasks, particularly permutation composition, have become a testbed to understand the limits of sequence models architectures like Transformers and RNNs (linear and non-linear). However, these are often sequence-to-sequence tasks: learning to map actions (permutations) to states, which is incompatible with the next-token prediction setting commonly used to train language models. We address this gap by converting permutation composition into code via REPL traces that interleave state-reveals through prints and variable transformations. We show that linear RNNs capable of state-tracking excel also in this setting, while Transformers still fail. Motivated by this representation, we investigate why tracking states in code is generally difficult: actions are not always fully observable. We frame this as tracking the state of a probabilistic finite-state automaton with deterministic state reveals and show that linear RNNs can be worse than non-linear RNNs at tracking states in this setup. 

---
# Exposing the Systematic Vulnerability of Open-Weight Models to Prefill Attacks 

**Authors**: Lukas Struppek, Adam Gleave, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2602.14689)  

**Abstract**: As the capabilities of large language models continue to advance, so does their potential for misuse. While closed-source models typically rely on external defenses, open-weight models must primarily depend on internal safeguards to mitigate harmful behavior. Prior red-teaming research has largely focused on input-based jailbreaking and parameter-level manipulations. However, open-weight models also natively support prefilling, which allows an attacker to predefine initial response tokens before generation begins. Despite its potential, this attack vector has received little systematic attention. We present the largest empirical study to date of prefill attacks, evaluating over 20 existing and novel strategies across multiple model families and state-of-the-art open-weight models. Our results show that prefill attacks are consistently effective against all major contemporary open-weight models, revealing a critical and previously underexplored vulnerability with significant implications for deployment. While certain large reasoning models exhibit some robustness against generic prefilling, they remain vulnerable to tailored, model-specific strategies. Our findings underscore the urgent need for model developers to prioritize defenses against prefill attacks in open-weight LLMs. 

---
# Alignment Adapter to Improve the Performance of Compressed Deep Learning Models 

**Authors**: Rohit Raj Rai, Abhishek Dhaka, Amit Awekar  

**Link**: [PDF](https://arxiv.org/pdf/2602.14635)  

**Abstract**: Compressed Deep Learning (DL) models are essential for deployment in resource-constrained environments. But their performance often lags behind their large-scale counterparts. To bridge this gap, we propose Alignment Adapter (AlAd): a lightweight, sliding-window-based adapter. It aligns the token-level embeddings of a compressed model with those of the original large model. AlAd preserves local contextual semantics, enables flexible alignment across differing dimensionalities or architectures, and is entirely agnostic to the underlying compression method. AlAd can be deployed in two ways: as a plug-and-play module over a frozen compressed model, or by jointly fine-tuning AlAd with the compressed model for further performance gains. Through experiments on BERT-family models across three token-level NLP tasks, we demonstrate that AlAd significantly boosts the performance of compressed models with only marginal overhead in size and latency. 

---
# MATEO: A Multimodal Benchmark for Temporal Reasoning and Planning in LVLMs 

**Authors**: Gabriel Roccabruna, Olha Khomyn, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2602.14589)  

**Abstract**: AI agents need to plan to achieve complex goals that involve orchestrating perception, sub-goal decomposition, and execution. These plans consist of ordered steps structured according to a Temporal Execution Order (TEO, a directed acyclic graph that ensures each step executes only after its preconditions are satisfied. Existing research on foundational models' understanding of temporal execution is limited to automatically derived annotations, approximations of the TEO as a linear chain, or text-only inputs. To address this gap, we introduce MATEO (MultimodAl Temporal Execution Order), a benchmark designed to assess and improve the temporal reasoning abilities of Large Vision Language Models (LVLMs) required for real-world planning. We acquire a high-quality professional multimodal recipe corpus, authored through a standardized editorial process that decomposes instructions into discrete steps, each paired with corresponding images. We collect TEO annotations as graphs by designing and using a scalable crowdsourcing pipeline. Using MATEO, we evaluate six state-of-the-art LVLMs across model scales, varying language context, multimodal input structure, and fine-tuning strategies. 

---
# Parameter-Efficient Fine-Tuning of LLMs with Mixture of Space Experts 

**Authors**: Buze Zhang, Jinkai Tao, Zilang Zeng, Neil He, Ali Maatouk, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2602.14490)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, with Parameter-Efficient Fine-Tuning (PEFT) emerging as a key technique for downstream task adaptation. However, existing PEFT methods mainly operate in Euclidean space, fundamentally limiting their capacity to capture complex geometric structures inherent in language data. While alternative geometric spaces, like hyperbolic geometries for hierarchical data and spherical manifolds for circular patterns, offer theoretical advantages, forcing representations into a single manifold type ultimately limits expressiveness, even when curvature parameters are learnable. To address this, we propose Mixture of Space (MoS), a unified framework that leverages multiple geometric spaces simultaneously to learn richer, curvature-aware representations. Building on this scheme, we develop MoSLoRA, which extends Low-Rank Adaptation (LoRA) with heterogeneous geometric experts, enabling models to dynamically select or combine appropriate geometric spaces based on input context. Furthermore, to address the computational overhead of frequent manifold switching, we develop a lightweight routing mechanism. Moreover, we provide empirical insights into how curvature optimization impacts training stability and model performance. Our experiments across diverse benchmarks demonstrate that MoSLoRA consistently outperforms strong baselines, achieving up to 5.6% improvement on MATH500 and 15.9% on MAWPS. 

---
# Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report v1.5 

**Authors**: Dongrui Liu, Yi Yu, Jie Zhang, Guanxu Chen, Qihao Lin, Hanxi Zhu, Lige Huang, Yijin Zhou, Peng Wang, Shuai Shao, Boxuan Zhang, Zicheng Liu, Jingwei Sun, Yu Li, Yuejin Xie, Jiaxuan Guo, Jia Xu, Chaochao Lu, Bowen Zhou, Xia Hu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2602.14457)  

**Abstract**: To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, Frontier AI Risk Management Framework in Practice presents a comprehensive assessment of their frontier risks. As Large Language Models (LLMs) general capabilities rapidly evolve and the proliferation of agentic AI, this version of the risk analysis technical report presents an updated and granular assessment of five critical dimensions: cyber offense, persuasion and manipulation, strategic deception, uncontrolled AI R\&D, and self-replication. Specifically, we introduce more complex scenarios for cyber offense. For persuasion and manipulation, we evaluate the risk of LLM-to-LLM persuasion on newly released LLMs. For strategic deception and scheming, we add the new experiment with respect to emergent misalignment. For uncontrolled AI R\&D, we focus on the ``mis-evolution'' of agents as they autonomously expand their memory substrates and toolsets. Besides, we also monitor and evaluate the safety performance of OpenClaw during the interaction on the Moltbook. For self-replication, we introduce a new resource-constrained scenario. More importantly, we propose and validate a series of robust mitigation strategies to address these emerging threats, providing a preliminary technical and actionable pathway for the secure deployment of frontier AI. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges. 

---
# Precedent-Informed Reasoning: Mitigating Overthinking in Large Reasoning Models via Test-Time Precedent Learning 

**Authors**: Qianyue Wang, Jinwu Hu, Huanxiang Lin, Bolin Chen, Zhiquan Wen, Yaofo Chen, Yu Rong, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.14451)  

**Abstract**: Reasoning in Large Language Models (LLMs) often suffers from inefficient long chain-of-thought traces with redundant self-exploration and validation, which inflate computational costs and even degrade performance. Inspired by human reasoning patterns where people solve new problems by leveraging past related cases to constrain search spaces and reduce trial-and-error, we propose Precedent Informed Reasoning (PIR) transforming LRMs'reasoning paradigm from exhaustive self-exploration to guided learning from precedents. PIR addresses two key challenges: what precedents to adopt and how to utilize them. First, Adaptive Precedent Selection (APS) constructs, for each question and LRM, a compact set of precedents that are both semantically related and informative for the model. It ranks examples by a joint score with semantic similarity and model perplexity, then adapts the amount of precedents to maximize perplexity reduction. Second, Test-time Experience Internalization (TEI) is treated as the test-time learning on precedent-informed instruction, updating lightweight adapters to internalize solution patterns and use them as a prior during subsequent reasoning. Experiments across mathematical reasoning, scientific QA, and code generation demonstrate that PIR consistently shortens reasoning traces while maintaining or improving final accuracy across LLMs, yielding outstanding accuracy-efficiency trade-offs. 

---
# Selective Synchronization Attention 

**Authors**: Hasi Hays  

**Link**: [PDF](https://arxiv.org/pdf/2602.14445)  

**Abstract**: The Transformer architecture has become the foundation of modern deep learning, yet its core self-attention mechanism suffers from quadratic computational complexity and lacks grounding in biological neural computation. We propose Selective Synchronization Attention (SSA), a novel attention mechanism that replaces the standard dot-product self-attention with a closed-form operator derived from the steady-state solution of the Kuramoto model of coupled oscillators. In SSA, each token is represented as an oscillator characterized by a learnable natural frequency and phase; the synchronization strength between token pairs, determined by a frequency-dependent coupling and phase-locking condition, serves as the attention weight. This formulation provides three key advantages: (i) natural sparsity arising from the phase-locking threshold, whereby tokens with incompatible frequencies automatically receive zero attention weight without explicit masking; (ii) unified positional-semantic encoding through the natural frequency spectrum, eliminating the need for separate positional encodings; and (iii) a single-pass, closed-form computation that avoids iterative ODE integration, with all components (coupling, order parameter, synchronization) derived from the oscillatory framework. We instantiate SSA within the Oscillatory Synchronization Network (OSN), a drop-in replacement for the Transformer block. Analysis of the synchronization matrices reveals non-uniform, head-diverse coupling patterns even at initialization, demonstrating a stronger architectural inductive bias than the approximately uniform attention produced by randomly initialized Transformers. 

---
# Synthetic Reader Panels: Tournament-Based Ideation with LLM Personas for Autonomous Publishing 

**Authors**: Fred Zimmerman  

**Link**: [PDF](https://arxiv.org/pdf/2602.14433)  

**Abstract**: We present a system for autonomous book ideation that replaces human focus groups with synthetic reader panels -- diverse collections of LLM-instantiated reader personas that evaluate book concepts through structured tournament competitions. Each persona is defined by demographic attributes (age group, gender, income, education, reading level), behavioral patterns (books per year, genre preferences, discovery methods, price sensitivity), and consistency parameters. Panels are composed per imprint to reflect target demographics, with diversity constraints ensuring representation across age, reading level, and genre affinity. Book concepts compete in single-elimination, double-elimination, round-robin, or Swiss-system tournaments, judged against weighted criteria including market appeal, originality, and execution potential. To reject low-quality LLM evaluations, we implement five automated anti-slop checks (repetitive phrasing, generic framing, circular reasoning, score clustering, audience mismatch). We report results from deployment within a multi-imprint publishing operation managing 6 active imprints and 609 titles in distribution. Three case studies -- a 270-evaluator panel for a children's literacy novel, and two 5-person expert panels for a military memoir and a naval strategy monograph -- demonstrate that synthetic panels produce actionable demographic segmentation, identify structural content issues invisible to homogeneous reviewers, and enable tournament filtering that eliminates low-quality concepts while enriching high-quality survivors from 15% to 62% of the evaluated pool. 

---
# Differentially Private Retrieval-Augmented Generation 

**Authors**: Tingting Tang, James Flemings, Yongqin Wang, Murali Annavaram  

**Link**: [PDF](https://arxiv.org/pdf/2602.14374)  

**Abstract**: Retrieval-augmented generation (RAG) is a widely used framework for reducing hallucinations in large language models (LLMs) on domain-specific tasks by retrieving relevant documents from a database to support accurate responses. However, when the database contains sensitive corpora, such as medical records or legal documents, RAG poses serious privacy risks by potentially exposing private information through its outputs. Prior work has demonstrated that one can practically craft adversarial prompts that force an LLM to regurgitate the augmented contexts. A promising direction is to integrate differential privacy (DP), a privacy notion that offers strong formal guarantees, into RAG systems. However, naively applying DP mechanisms into existing systems often leads to significant utility degradation. Particularly for RAG systems, DP can reduce the usefulness of the augmented contexts leading to increase risk of hallucination from the LLMs. Motivated by these challenges, we present DP-KSA, a novel privacy-preserving RAG algorithm that integrates DP using the propose-test-release paradigm. DP-KSA follows from a key observation that most question-answering (QA) queries can be sufficiently answered with a few keywords. Hence, DP-KSA first obtains an ensemble of relevant contexts, each of which will be used to generate a response from an LLM. We utilize these responses to obtain the most frequent keywords in a differentially private manner. Lastly, the keywords are augmented into the prompt for the final output. This approach effectively compresses the semantic space while preserving both utility and privacy. We formally show that DP-KSA provides formal DP guarantees on the generated output with respect to the RAG database. We evaluate DP-KSA on two QA benchmarks using three instruction-tuned LLMs, and our empirical results demonstrate that DP-KSA achieves a strong privacy-utility tradeoff. 

---
# FMMD: A multimodal open peer review dataset based on F1000Research 

**Authors**: Zhenzhen Zhuang, Yuqing Fu, Jing Zhu, Zhangping Zhou, Jialiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.14285)  

**Abstract**: Automated scholarly paper review (ASPR) has entered the coexistence phase with traditional peer review, where artificial intelligence (AI) systems are increasingly incorporated into real-world manuscript evaluation. In parallel, research on automated and AI-assisted peer review has proliferated. Despite this momentum, empirical progress remains constrained by several critical limitations in existing datasets. While reviewers routinely evaluate figures, tables, and complex layouts to assess scientific claims, most existing datasets remain overwhelmingly text-centric. This bias is reinforced by a narrow focus on data from computer science venues. Furthermore, these datasets lack precise alignment between reviewer comments and specific manuscript versions, obscuring the iterative relationship between peer review and manuscript evolution. In response, we introduce FMMD, a multimodal and multidisciplinary open peer review dataset curated from F1000Research. The dataset bridges the current gap by integrating manuscript-level visual and structural data with version-specific reviewer reports and editorial decisions. By providing explicit alignment between reviewer comments and the exact article iteration under review, FMMD enables fine-grained analysis of the peer review lifecycle across diverse scientific domains. FMMD supports tasks such as multimodal issue detection and multimodal review comment generation. It provides a comprehensive empirical resource for the development of peer review research. 

---
# MCPShield: A Security Cognition Layer for Adaptive Trust Calibration in Model Context Protocol Agents 

**Authors**: Zhenhong Zhou, Yuanhe Zhang, Hongwei Cai, Moayad Aloqaily, Ouns Bouachir, Linsey Pang, Prakhar Mehrotra, Kun Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.14281)  

**Abstract**: The Model Context Protocol (MCP) standardizes tool use for LLM-based agents and enable third-party servers. This openness introduces a security misalignment: agents implicitly trust tools exposed by potentially untrusted MCP servers. However, despite its excellent utility, existing agents typically offer limited validation for third-party MCP servers. As a result, agents remain vulnerable to MCP-based attacks that exploit the misalignment between agents and servers throughout the tool invocation lifecycle. In this paper, we propose MCPShield as a plug-in security cognition layer that mitigates this misalignment and ensures agent security when invoking MCP-based tools. Drawing inspiration from human experience-driven tool validation, MCPShield assists agent forms security cognition with metadata-guided probing before invocation. Our method constrains execution within controlled boundaries while cognizing runtime events, and subsequently updates security cognition by reasoning over historical traces after invocation, building on human post-use reflection on tool behavior. Experiments demonstrate that MCPShield exhibits strong generalization in defending against six novel MCP-based attack scenarios across six widely used agentic LLMs, while avoiding false positives on benign servers and incurring low deployment overhead. Overall, our work provides a practical and robust security safeguard for MCP-based tool invocation in open agent ecosystems. 

---
# Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions 

**Authors**: Ruomeng Ding, Tianwei Gao, Thomas P. Zollo, Eitan Bachmat, Richard Zemel, Zhun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.14279)  

**Abstract**: Eliciting information to reduce uncertainty about latent group-level properties from surveys and other collective assessments requires allocating limited questioning effort under real costs and missing data. Although large language models enable adaptive, multi-turn interactions in natural language, most existing elicitation methods optimize what to ask with a fixed respondent pool, and do not adapt respondent selection or leverage population structure when responses are partial or incomplete. To address this gap, we study adaptive group elicitation, a multi-round setting where an agent adaptively selects both questions and respondents under explicit query and participation budgets. We propose a theoretically grounded framework that combines (i) an LLM-based expected information gain objective for scoring candidate questions with (ii) heterogeneous graph neural network propagation that aggregates observed responses and participant attributes to impute missing responses and guide per-round respondent selection. This closed-loop procedure queries a small, informative subset of individuals while inferring population-level responses via structured similarity. Across three real-world opinion datasets, our method consistently improves population-level response prediction under constrained budgets, including a >12% relative gain on CES at a 10% respondent budget. 

---
# REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents 

**Authors**: Zheng Chu, Xiao Wang, Jack Hong, Huiming Fan, Yuqi Huang, Yue Yang, Guohai Xu, Chenxiao Zhao, Cheng Xiang, Shengchao Hu, Dongdong Kuang, Ming Liu, Bing Qin, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.14234)  

**Abstract**: Large language models are transitioning from generalpurpose knowledge engines to realworld problem solvers, yet optimizing them for deep search tasks remains challenging. The central bottleneck lies in the extreme sparsity of highquality search trajectories and reward signals, arising from the difficulty of scalable longhorizon task construction and the high cost of interactionheavy rollouts involving external tool calls. To address these challenges, we propose REDSearcher, a unified framework that codesigns complex task synthesis, midtraining, and posttraining for scalable searchagent optimization. Specifically, REDSearcher introduces the following improvements: (1) We frame task synthesis as a dualconstrained optimization, where task difficulty is precisely governed by graph topology and evidence dispersion, allowing scalable generation of complex, highquality tasks. (2) We introduce toolaugmented queries to encourage proactive tool use rather than passive recall.(3) During midtraining, we strengthen core atomic capabilities knowledge, planning, and function calling substantially reducing the cost of collecting highquality trajectories for downstream training. (4) We build a local simulated environment that enables rapid, lowcost algorithmic iteration for reinforcement learning experiments. Across both textonly and multimodal searchagent benchmarks, our approach achieves stateoftheart performance. To facilitate future research on longhorizon search agents, we will release 10K highquality complex text search trajectories, 5K multimodal trajectories and 1K text RL query set, and together with code and model checkpoints. 

---
# The Interspeech 2026 Audio Reasoning Challenge: Evaluating Reasoning Process Quality for Audio Reasoning Models and Agents 

**Authors**: Ziyang Ma, Ruiyang Xu, Yinghao Ma, Chao-Han Huck Yang, Bohan Li, Jaeyeon Kim, Jin Xu, Jinyu Li, Carlos Busso, Kai Yu, Eng Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.14224)  

**Abstract**: Recent Large Audio Language Models (LALMs) excel in understanding but often lack transparent reasoning. To address this "black-box" limitation, we organized the Audio Reasoning Challenge at Interspeech 2026, the first shared task dedicated to evaluating Chain-of-Thought (CoT) quality in the audio domain. The challenge introduced MMAR-Rubrics, a novel instance-level protocol assessing the factuality and logic of reasoning chains. Featured Single Model and Agent tracks, the competition attracting 156 teams from 18 countries and regions. Results show agent systems currently lead in reasoning quality, utilizing iterative tool orchestration and cross-modal analysis. Besides, single models are rapidly advancing via reinforcement learning and sophisticated data pipeline. We details the challenge design, methodology, and a comprehensive analysis of state-of-the-art systems, providing new insights for explainable audio intelligence. 

---
# Reasoning Language Models for complex assessments tasks: Evaluating parental cooperation from child protection case reports 

**Authors**: Dragan Stoll, Brian E. Perron, Zia Qi, Selina Steinmann, Nicole F. Eicher, Andreas Jud  

**Link**: [PDF](https://arxiv.org/pdf/2602.14216)  

**Abstract**: Purpose: Reasoning language models (RLMs) have demonstrated significant advances in solving complex reasoning tasks. We examined their potential to assess parental cooperation during CPS interventions using case reports, a case factor characterized by ambiguous and conflicting information. Methods: A four stage workflow comprising (1) case reports collection, (2) reasoning-based assessment of parental cooperation, (3) automated category extraction, and (4) case labeling was developed. The performance of RLMs with different parameter sizes (255B, 32B, 4B) was compared against human validated data. Two expert human reviewers (EHRs) independently classified a weighted random sample of reports. Results: The largest RLM achieved the highest accuracy (89%), outperforming the initial approach (80%). Classification accuracy was higher for mothers (93%) than for fathers (85%), and EHRs exhibited similar differences. Conclusions: RLMs' reasoning can effectively assess complex case factors such as parental cooperation. Lower accuracy in assessing fathers' cooperation supports the argument of a stronger professional focus on mothers in CPS interventions. 

---
# MAGE: All-[MASK] Block Already Knows Where to Look in Diffusion LLM 

**Authors**: Omin Kwon, Yeonjae Kim, Doyeon Kim, Minseo Kim, Yeonhong Park, Jae W. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.14209)  

**Abstract**: Block diffusion LLMs are emerging as a promising next paradigm for language generation, but their use of KV caching makes memory access a dominant bottleneck in long-context settings. While dynamic sparse attention has been actively explored, existing methods designed for autoregressive LLMs rely on approximate importance estimation and perform poorly when adapted to block diffusion. This work identifies a key opportunity unique to block diffusion: attention at the first All-[MASK] denoising step reliably predicts important KV entries and budget requirements, enabling MAGE to perform a single exact attention pass per block and reuse it for training-free sparse denoising. Across long-context benchmarks including LongBench and Needle-in-a-Haystack, MAGE achieves near-lossless accuracy with a fraction of the KV budget while delivering up to 3-4x end-to-end speedup, consistently outperforming AR-oriented sparse attention baselines. A lightweight fine-tuning strategy further strengthens [MASK]-guided patterns with minimal cost, requiring only a few hours of training on a single NVIDIA H100 GPU for both 1.5B and 7B models. 

---
# Investigation for Relative Voice Impression Estimation 

**Authors**: Keinichi Fujita, Yusuke Ijima  

**Link**: [PDF](https://arxiv.org/pdf/2602.14172)  

**Abstract**: Paralinguistic and non-linguistic aspects of speech strongly influence listener impressions. While most research focuses on absolute impression scoring, this study investigates relative voice impression estimation (RIE), a framework for predicting the perceptual difference between two utterances from the same speaker. The estimation target is a low-dimensional vector derived from subjective evaluations, quantifying the perceptual shift of the second utterance relative to the first along an antonymic axis (e.g., ``Dark--Bright''). To isolate expressive and prosodic variation, we used recordings of a professional speaker reading a text in various styles. We compare three modeling approaches: classical acoustic features commonly used for speech emotion recognition, self-supervised speech representations, and multimodal large language models (MLLMs). Our results demonstrate that models using self-supervised representations outperform methods with classical acoustic features, particularly in capturing complex and dynamic impressions (e.g., ``Cold--Warm'') where classical features fail. In contrast, current MLLMs prove unreliable for this fine-grained pairwise task. This study provides the first systematic investigation of RIE and demonstrates the strength of self-supervised speech models in capturing subtle perceptual variations. 

---
# Deep Dense Exploration for LLM Reinforcement Learning via Pivot-Driven Resampling 

**Authors**: Yiran Guo, Zhongjian Qiao, Yingqi Xie, Jie Liu, Dan Ye, Ruiqing Zhang, Shuang Qiu, Lijie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.14169)  

**Abstract**: Effective exploration is a key challenge in reinforcement learning for large language models: discovering high-quality trajectories within a limited sampling budget from the vast natural language sequence space. Existing methods face notable limitations: GRPO samples exclusively from the root, saturating high-probability trajectories while leaving deep, error-prone states under-explored. Tree-based methods blindly disperse budgets across trivial or unrecoverable states, causing sampling dilution that fails to uncover rare correct suffixes and destabilizes local baselines. To address this, we propose Deep Dense Exploration (DDE), a strategy that focuses exploration on $\textit{pivots}$-deep, recoverable states within unsuccessful trajectories. We instantiate DDE with DEEP-GRPO, which introduces three key innovations: (1) a lightweight data-driven utility function that automatically balances recoverability and depth bias to identify pivot states; (2) local dense resampling at each pivot to increase the probability of discovering correct subsequent trajectories; and (3) a dual-stream optimization objective that decouples global policy learning from local corrective updates. Experiments on mathematical reasoning benchmarks demonstrate that our method consistently outperforms GRPO, tree-based methods, and other strong baselines. 

---
# ROAST: Rollout-based On-distribution Activation Steering Technique 

**Authors**: Xuanbo Su, Hao Luo, Yingfang Zhang, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.14143)  

**Abstract**: Activation steering provides parameter-efficient control over large language models (LLMs) at inference time, but many methods rely on off-distribution supervision and discrete masking, leading to brittle interventions. We propose ROAST (Rollout-based On-distribution Activation Steering Technique), which estimates steering directions from the model's own on-distribution rollouts via ROC and avoids hard sparsification via Continuous Soft Scaling (CSS) and Grouped Mean Normalization. Our empirical analysis reveals that while activation magnitude correlates moderately with directional consistency, the variance in magnitude is significant and often disproportionate to semantic quality. This suggests that high-magnitude activations risk dominating the global steering direction if not properly normalized. To address this, ROAST employs grouped normalization to balance contributions across samples, ensuring a more robust estimation of the consensus steering direction. Across models (0.6B to 32B), ROAST consistently improves performance on diverse tasks (e.g., +9.7% on GSM8K for Qwen3-0.6B and +12.1% on TruthfulQA for GLM4-32B), and analyses show that CSS better preserves activation energy. 

---
# Algebraic Quantum Intelligence: A New Framework for Reproducible Machine Creativity 

**Authors**: Kazuo Yano, Jonghyeok Lee, Tae Ishitomi, Hironobu Kawaguchi, Akira Koyama, Masakuni Ota, Yuki Ota, Nobuo Sato, Keita Shimada, Sho Takematsu, Ayaka Tobinai, Satomi Tsuji, Kazunori Yanagi, Keiko Yano, Manabu Harada, Yuki Matsuda, Kazunori Matsumoto, Kenichi Matsumura, Hamae Matsuo, Yumi Miyazaki, Kotaro Murai, Tatsuya Ohshita, Marie Seki, Shun Tanoue, Tatsuki Terakado, Yuko Ichimaru, Mirei Saito, Akihiro Otsuka, Koji Ara  

**Link**: [PDF](https://arxiv.org/pdf/2602.14130)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in generating fluent and contextually appropriate text; however, their capacity to produce genuinely creative outputs remains limited. This paper posits that this limitation arises from a structural property of contemporary LLMs: when provided with rich context, the space of future generations becomes strongly constrained, and the generation process is effectively governed by near-deterministic dynamics. Recent approaches such as test-time scaling and context adaptation improve performance but do not fundamentally alter this constraint. To address this issue, we propose Algebraic Quantum Intelligence (AQI) as a computational framework that enables systematic expansion of semantic space. AQI is formulated as a noncommutative algebraic structure inspired by quantum theory, allowing properties such as order dependence, interference, and uncertainty to be implemented in a controlled and designable manner. Semantic states are represented as vectors in a Hilbert space, and their evolution is governed by C-values computed from noncommutative operators, thereby ensuring the coexistence and expansion of multiple future semantic possibilities. In this study, we implement AQI by extending a transformer-based LLM with more than 600 specialized operators. We evaluate the resulting system on creative reasoning benchmarks spanning ten domains under an LLM-as-a-judge protocol. The results show that AQI consistently outperforms strong baseline models, yielding statistically significant improvements and reduced cross-domain variance. These findings demonstrate that noncommutative algebraic dynamics can serve as a practical and reproducible foundation for machine creativity. Notably, this architecture has already been deployed in real-world enterprise environments. 

---
# Neuromem: A Granular Decomposition of the Streaming Lifecycle in External Memory for LLMs 

**Authors**: Ruicheng Zhang, Xinyi Li, Tianyi Xu, Shuhao Zhang, Xiaofei Liao, Hai Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.13967)  

**Abstract**: Most evaluations of External Memory Module assume a static setting: memory is built offline and queried at a fixed state. In practice, memory is streaming: new facts arrive continuously, insertions interleave with retrievals, and the memory state evolves while the model is serving queries. In this regime, accuracy and cost are governed by the full memory lifecycle, which encompasses the ingestion, maintenance, retrieval, and integration of information into generation. We present Neuromem, a scalable testbed that benchmarks External Memory Modules under an interleaved insertion-and-retrieval protocol and decomposes its lifecycle into five dimensions including memory data structure, normalization strategy, consolidation policy, query formulation strategy, and context integration mechanism. Using three representative datasets LOCOMO, LONGMEMEVAL, and MEMORYAGENTBENCH, Neuromem evaluates interchangeable variants within a shared serving stack, reporting token-level F1 and insertion/retrieval latency. Overall, we observe that performance typically degrades as memory grows across rounds, and time-related queries remain the most challenging category. The memory data structure largely determines the attainable quality frontier, while aggressive compression and generative integration mechanisms mostly shift cost between insertion and retrieval with limited accuracy gain. 

---
# MarsRetrieval: Benchmarking Vision-Language Models for Planetary-Scale Geospatial Retrieval on Mars 

**Authors**: Shuoyuan Wang, Yiran Wang, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.13961)  

**Abstract**: Data-driven approaches like deep learning are rapidly advancing planetary science, particularly in Mars exploration. Despite recent progress, most existing benchmarks remain confined to closed-set supervised visual tasks and do not support text-guided retrieval for geospatial discovery. We introduce MarsRetrieval, a retrieval benchmark for evaluating vision-language models for Martian geospatial discovery. MarsRetrieval includes three tasks: (1) paired image-text retrieval, (2) landform retrieval, and (3) global geo-localization, covering multiple spatial scales and diverse geomorphic origins. We propose a unified retrieval-centric protocol to benchmark multimodal embedding architectures, including contrastive dual-tower encoders and generative vision-language models. Our evaluation shows MarsRetrieval is challenging: even strong foundation models often fail to capture domain-specific geomorphic distinctions. We further show that domain-specific fine-tuning is critical for generalizable geospatial discovery in planetary settings. Our code is available at this https URL 

---
# Why Code, Why Now: Learnability, Computability, and the Real Limits of Machine Learning 

**Authors**: Zhimin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.13934)  

**Abstract**: Code generation has progressed more reliably than reinforcement learning, largely because code has an information structure that makes it learnable. Code provides dense, local, verifiable feedback at every token, whereas most reinforcement learning problems do not. This difference in feedback quality is not binary but graded. We propose a five-level hierarchy of learnability based on information structure and argue that the ceiling on ML progress depends less on model size than on whether a task is learnable at all. The hierarchy rests on a formal distinction among three properties of computational problems (expressibility, computability, and learnability). We establish their pairwise relationships, including where implications hold and where they fail, and present a unified template that makes the structural differences explicit. The analysis suggests why supervised learning on code scales predictably while reinforcement learning does not, and why the common assumption that scaling alone will solve remaining ML challenges warrants scrutiny. 

---
# From Pixels to Policies: Reinforcing Spatial Reasoning in Language Models for Content-Aware Layout Design 

**Authors**: Sha Li, Stefano Petrangeli, Yu Shen, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.13912)  

**Abstract**: We introduce LaySPA, a reinforcement learning framework that equips large language models (LLMs) with explicit and interpretable spatial reasoning for content-aware graphic layout design. LaySPA addresses two key challenges: LLMs' limited spatial reasoning and the lack of opacity in design decision making. Instead of operating at the pixel level, we reformulate layout design as a policy learning problem over a structured textual spatial environment that explicitly encodes canvas geometry, element attributes, and inter-element relationships. LaySPA produces dual-level outputs comprising interpretable reasoning traces and structured layout specifications, enabling transparent and controllable design decision making. Layout design policy is optimized via a multi-objective spatial critique that decomposes layout quality into geometric validity, relational coherence, and aesthetic consistency, and is trained using relative group optimization to stabilize learning in open-ended design spaces. Experiments demonstrate that LaySPA improves structural validity and visual quality, outperforming larger proprietary LLMs and achieving performance comparable to specialized SOTA layout generators while requiring fewer annotated samples and reduced latency. 

---
# StackingNet: Collective Inference Across Independent AI Foundation Models 

**Authors**: Siyang Li, Chenhao Liu, Dongrui Wu, Zhigang Zeng, Lieyun Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.13792)  

**Abstract**: Artificial intelligence built on large foundation models has transformed language understanding, vision and reasoning, yet these systems remain isolated and cannot readily share their capabilities. Integrating the complementary strengths of such independent foundation models is essential for building trustworthy intelligent systems. Despite rapid progress in individual model design, there is no established approach for coordinating such black-box heterogeneous models. Here we show that coordination can be achieved through a meta-ensemble framework termed StackingNet, which draws on principles of collective intelligence to combine model predictions during inference. StackingNet improves accuracy, reduces bias, enables reliability ranking, and identifies or prunes models that degrade performance, all operating without access to internal parameters or training data. Across tasks involving language comprehension, visual estimation, and academic paper rating, StackingNet consistently improves accuracy, robustness, and fairness, compared with individual models and classic ensembles. By turning diversity from a source of inconsistency into collaboration, StackingNet establishes a practical foundation for coordinated artificial intelligence, suggesting that progress may emerge from not only larger single models but also principled cooperation among many specialized ones. 

---
# AllMem: A Memory-centric Recipe for Efficient Long-context Modeling 

**Authors**: Ziming Wang, Xiang Wang, Kailong Peng, Lang Qin, Juan Gabriel Kostelec, Christos Sourmpis, Axel Laborieux, Qinghai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.13680)  

**Abstract**: Large Language Models (LLMs) encounter significant performance bottlenecks in long-sequence tasks due to the computational complexity and memory overhead inherent in the self-attention mechanism. To address these challenges, we introduce \textsc{AllMem}, a novel and efficient hybrid architecture that integrates Sliding Window Attention (SWA) with non-linear Test-Time Training (TTT) memory networks. \textsc{AllMem} enables models to effectively scale to ultra-long contexts while mitigating catastrophic forgetting. This approach not only overcomes the representation constraints typical of linear memory models but also significantly reduces the computational and memory footprint during long-sequence inference. Furthermore, we implement a Memory-Efficient Fine-Tuning strategy to replace standard attention layers in pre-trained models with memory-augmented sliding window layers. This framework facilitates the efficient transformation of any off-the-shelf pre-trained LLM into an \textsc{AllMem}-based architecture. Empirical evaluations confirm that our 4k window model achieves near-lossless performance on 37k LongBench with a marginal 0.83 drop compared to full attention. Furthermore, on InfiniteBench at a 128k context, our 8k window variant outperforms full attention, which validates the effectiveness of our parameterized memory in mitigating noise and maintaining robust long-range modeling without the prohibitive costs of global attention. 

---
# Building Autonomous GUI Navigation via Agentic-Q Estimation and Step-Wise Policy Optimization 

**Authors**: Yibo Wang, Guangda Huzhang, Yuwei Hu, Yu Xia, Shiyin Lu, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.13653)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have substantially driven the progress of autonomous agents for Graphical User Interface (GUI). Nevertheless, in real-world applications, GUI agents are often faced with non-stationary environments, leading to high computational costs for data curation and policy optimization. In this report, we introduce a novel MLLM-centered framework for GUI agents, which consists of two components: agentic-Q estimation and step-wise policy optimization. The former one aims to optimize a Q-model that can generate step-wise values to evaluate the contribution of a given action to task completion. The latter one takes step-wise samples from the state-action trajectory as inputs, and optimizes the policy via reinforcement learning with our agentic-Q model. It should be noticed that (i) all state-action trajectories are produced by the policy itself, so that the data collection costs are manageable; (ii) the policy update is decoupled from the environment, ensuring stable and efficient optimization. Empirical evaluations show that our framework endows Ovis2.5-9B with powerful GUI interaction capabilities, achieving remarkable performances on GUI navigation and grounding benchmarks and even surpassing contenders with larger scales. 

---
# KorMedMCQA-V: A Multimodal Benchmark for Evaluating Vision-Language Models on the Korean Medical Licensing Examination 

**Authors**: Byungjin Choi, Seongsu Bae, Sunjun Kweon, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.13650)  

**Abstract**: We introduce KorMedMCQA-V, a Korean medical licensing-exam-style multimodal multiple-choice question answering benchmark for evaluating vision-language models (VLMs). The dataset consists of 1,534 questions with 2,043 associated images from Korean Medical Licensing Examinations (2012-2023), with about 30% containing multiple images requiring cross-image evidence integration. Images cover clinical modalities including X-ray, computed tomography (CT), electrocardiography (ECG), ultrasound, endoscopy, and other medical visuals. We benchmark over 50 VLMs across proprietary and open-source categories-spanning general-purpose, medical-specialized, and Korean-specialized families-under a unified zero-shot evaluation protocol. The best proprietary model (Gemini-3.0-Pro) achieves 96.9% accuracy, the best open-source model (Qwen3-VL-32B-Thinking) 83.7%, and the best Korean-specialized model (VARCO-VISION-2.0-14B) only 43.2%. We further find that reasoning-oriented model variants gain up to +20 percentage points over instruction-tuned counterparts, medical domain specialization yields inconsistent gains over strong general-purpose baselines, all models degrade on multi-image questions, and performance varies notably across imaging modalities. By complementing the text-only KorMedMCQA benchmark, KorMedMCQA-V forms a unified evaluation suite for Korean medical reasoning across text-only and multimodal conditions. The dataset is available via Hugging Face Datasets: this https URL. 

---
# Rubrics as an Attack Surface: Stealthy Preference Drift in LLM Judges 

**Authors**: Ruomeng Ding, Yifei Pang, He Sun, Yizhong Wang, Zhiwei Steven Wu, Zhun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.13576)  

**Abstract**: Evaluation and alignment pipelines for large language models increasingly rely on LLM-based judges, whose behavior is guided by natural-language rubrics and validated on benchmarks. We identify a previously under-recognized vulnerability in this workflow, which we term Rubric-Induced Preference Drift (RIPD). Even when rubric edits pass benchmark validation, they can still produce systematic and directional shifts in a judge's preferences on target domains. Because rubrics serve as a high-level decision interface, such drift can emerge from seemingly natural, criterion-preserving edits and remain difficult to detect through aggregate benchmark metrics or limited spot-checking. We further show this vulnerability can be exploited through rubric-based preference attacks, in which benchmark-compliant rubric edits steer judgments away from a fixed human or trusted reference on target domains, systematically inducing RIPD and reducing target-domain accuracy up to 9.5% (helpfulness) and 27.9% (harmlessness). When these judgments are used to generate preference labels for downstream post-training, the induced bias propagates through alignment pipelines and becomes internalized in trained policies. This leads to persistent and systematic drift in model behavior. Overall, our findings highlight evaluation rubrics as a sensitive and manipulable control interface, revealing a system-level alignment risk that extends beyond evaluator reliability alone. The code is available at: this https URL. Warning: Certain sections may contain potentially harmful content that may not be appropriate for all readers. 

---
# Mitigating the Safety-utility Trade-off in LLM Alignment via Adaptive Safe Context Learning 

**Authors**: Yanbo Wang, Minzheng Wang, Jian Liang, Lu Wang, Yongcan Yu, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2602.13562)  

**Abstract**: While reasoning models have achieved remarkable success in complex reasoning tasks, their increasing power necessitates stringent safety measures. For safety alignment, the core challenge lies in the inherent trade-off between safety and utility. However, prevailing alignment strategies typically construct CoT training data with explicit safety rules via context distillation. This approach inadvertently limits reasoning capabilities by creating a rigid association between rule memorization and refusal. To mitigate the safety-utility trade-off, we propose the Adaptive Safe Context Learning (ASCL) framework to improve the reasoning given proper context. ASCL formulates safety alignment as a multi-turn tool-use process, empowering the model to autonomously decide when to consult safety rules and how to generate the ongoing reasoning. Furthermore, to counteract the preference for rule consultation during RL, we introduce Inverse Frequency Policy Optimization (IFPO) to rebalance advantage estimates. By decoupling rule retrieval and subsequent reasoning, our method achieves higher overall performance compared to baselines. 

---
# LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News 

**Authors**: Yunfan Zhang, Kathleen McKeown, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2602.13543)  

**Abstract**: Large Language Models (LLMs) with agentic web search capabilities show strong potential for tasks requiring real-time information access and complex fact retrieval, yet evaluating such systems remains challenging. We introduce \bench, a rigorous and regularly updated benchmark designed to assess the agentic web search abilities of LLMs. \bench automatically generates fresh question-answer pairs from recent news articles, ensuring that questions require information beyond an LLM's training data and enabling clear separation between internal knowledge and search capability. The benchmark features intentionally difficult questions requiring multi-hop search queries, page visits, and reasoning, making it well-suited for evaluating agentic search behavior. Our automated data curation and question generation pipeline enables frequent benchmark updates and supports construction of a large-scale training dataset for agentic web search models, addressing the scarcity of such data in the research community. To ensure reliable evaluation, we include a subset of human-verified samples in the test set. We evaluate a broad range of systems using \bench, including commercial and open-weight LLMs as well as LLM-based web search APIs. The leaderboard, datasets, and code are publicly available at this http URL. 

---
# SecureGate: Learning When to Reveal PII Safely via Token-Gated Dual-Adapters for Federated LLMs 

**Authors**: Mohamed Shaaban, Mohamed Elmahallawy  

**Link**: [PDF](https://arxiv.org/pdf/2602.13529)  

**Abstract**: Federated learning (FL) enables collaborative training across organizational silos without sharing raw data, making it attractive for privacy-sensitive applications. With the rapid adoption of large language models (LLMs), federated fine-tuning of generative LLMs has gained attention as a way to leverage distributed data while preserving confidentiality. However, this setting introduces fundamental challenges: (i) privacy leakage of personally identifiable information (PII) due to LLM memorization, and (ii) a persistent tension between global generalization and local utility under heterogeneous data. Existing defenses, such as data sanitization and differential privacy, reduce leakage but often degrade downstream performance. We propose SecureGate, a privacy-aware federated fine-tuning framework for LLMs that provides fine-grained privacy control without sacrificing utility. SecureGate employs a dual-adapter LoRA architecture: a secure adapter that learns sanitized, globally shareable representations, and a revealing adapter that captures sensitive, organization-specific knowledge. A token-controlled gating module selectively activates these adapters at inference time, enabling controlled information disclosure without retraining. Extensive experiments across multiple LLMs and real-world datasets show that SecureGate improves task utility while substantially reducing PII leakage, achieving up to a 31.66X reduction in inference attack accuracy and a 17.07X reduction in extraction recall for unauthorized requests. Additionally, it maintains 100% routing reliability to the correct adapter and incurs only minimal computational and communication overhead. 

---
# Protect$^*$: Steerable Retrosynthesis through Neuro-Symbolic State Encoding 

**Authors**: Shreyas Vinaya Sathyanarayana, Shah Rahil Kirankumar, Sharanabasava D. Hiremath, Bharath Ramsundar  

**Link**: [PDF](https://arxiv.org/pdf/2602.13419)  

**Abstract**: Large Language Models (LLMs) have shown remarkable potential in scientific domains like retrosynthesis; yet, they often lack the fine-grained control necessary to navigate complex problem spaces without error. A critical challenge is directing an LLM to avoid specific, chemically sensitive sites on a molecule - a task where unconstrained generation can lead to invalid or undesirable synthetic pathways. In this work, we introduce Protect$^*$, a neuro-symbolic framework that grounds the generative capabilities of Large Language Models (LLMs) in rigorous chemical logic. Our approach combines automated rule-based reasoning - using a comprehensive database of 55+ SMARTS patterns and 40+ characterized protecting groups - with the generative intuition of neural models. The system operates via a hybrid architecture: an ``automatic mode'' where symbolic logic deterministically identifies and guards reactive sites, and a ``human-in-the-loop mode'' that integrates expert strategic constraints. Through ``active state tracking,'' we inject hard symbolic constraints into the neural inference process via a dedicated protection state linked to canonical atom maps. We demonstrate this neuro-symbolic approach through case studies on complex natural products, including the discovery of a novel synthetic pathway for Erythromycin B, showing that grounding neural generation in symbolic logic enables reliable, expert-level autonomy. 

---
# Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents 

**Authors**: Xu Li, Simon Yu, Minzhou Pan, Yiyou Sun, Bo Li, Dawn Song, Xue Lin, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.13379)  

**Abstract**: LLM-based agents are becoming increasingly capable, yet their safety lags behind. This creates a gap between what agents can do and should do. This gap widens as agents engage in multi-turn interactions and employ diverse tools, introducing new risks overlooked by existing benchmarks. To systematically scale safety testing into multi-turn, tool-realistic settings, we propose a principled taxonomy that transforms single-turn harmful tasks into multi-turn attack sequences. Using this taxonomy, we construct MT-AgentRisk (Multi-Turn Agent Risk Benchmark), the first benchmark to evaluate multi-turn tool-using agent safety. Our experiments reveal substantial safety degradation: the Attack Success Rate (ASR) increases by 16% on average across open and closed models in multi-turn settings. To close this gap, we propose ToolShield, a training-free, tool-agnostic, self-exploration defense: when encountering a new tool, the agent autonomously generates test cases, executes them to observe downstream effects, and distills safety experiences for deployment. Experiments show that ToolShield effectively reduces ASR by 30% on average in multi-turn interactions. Our code is available at this https URL. 

---
# An Online Reference-Free Evaluation Framework for Flowchart Image-to-Code Generation 

**Authors**: Giang Son Nguyen, Zi Pong Lim, Sarthak Ketanbhai Modi, Yon Shin Teo, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.13376)  

**Abstract**: Vision-Language Models (VLMs) are increasingly used in document processing pipelines to convert flowchart images into structured code (e.g., Mermaid). In production, these systems process arbitrary inputs for which no ground-truth code exists, making output quality difficult to assess. We propose a reference-free evaluation framework that monitors flowchart image-to-code generation quality at inference time, using only the input image and the generated output. The framework introduces two automated metrics: $\text{Recall}{\text{OCR}}$, which estimates content coverage by extracting text from the input image via OCR as a proxy reference, and $\text{Precision}{\text{VE}}$, which detects hallucinated elements through Visual Entailment against the original image. Their harmonic mean, $\text{F1}{\text{OCR-VE}}$, provides a unified quality score. Validation on the FlowVQA dataset shows strong agreement with ground-truth metrics (average Pearson's $r = 0.97$, $0.91$, and $0.94$ for Recall, Precision, and F1, respectively), confirming the framework's reliability as a practical, reference-free alternative for continuous quality monitoring in production settings. 

---
# G2CP: A Graph-Grounded Communication Protocol for Verifiable and Efficient Multi-Agent Reasoning 

**Authors**: Karim Ben Khaled, Davy Monticolo  

**Link**: [PDF](https://arxiv.org/pdf/2602.13370)  

**Abstract**: Multi-agent systems powered by Large Language Models face a critical challenge: agents communicate through natural language, leading to semantic drift, hallucination propagation, and inefficient token consumption. We propose G2CP (Graph-Grounded Communication Protocol), a structured agent communication language where messages are graph operations rather than free text. Agents exchange explicit traversal commands, subgraph fragments, and update operations over a shared knowledge graph, enabling verifiable reasoning traces and eliminating ambiguity. We validate G2CP within an industrial knowledge management system where specialized agents (Diagnostic, Procedural, Synthesis, and Ingestion) coordinate to answer complex queries. Experimental results on 500 industrial scenarios and 21 real-world maintenance cases show that G2CP reduces inter-agent communication tokens by 73%, improves task completion accuracy by 34% over free-text baselines, eliminates cascading hallucinations, and produces fully auditable reasoning chains. G2CP represents a fundamental shift from linguistic to structural communication in multi-agent systems, with implications for any domain requiring precise agent coordination. Code, data, and evaluation scripts are publicly available. 

---
# Nanbeige4.1-3B: A Small General Model that Reasons, Aligns, and Acts 

**Authors**: Chen Yang, Guangyue Peng, Jiaying Zhu, Ran Le, Ruixiang Feng, Tao Zhang, Xiyun Xu, Yang Song, Yiming Jia, Yuntao Wen, Yunzhi Xu, Zekai Wang, Zhenwei An, Zhicong Sun, Zongchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.13367)  

**Abstract**: We present Nanbeige4.1-3B, a unified generalist language model that simultaneously achieves strong agentic behavior, code generation, and general reasoning with only 3B parameters. To the best of our knowledge, it is the first open-source small language model (SLM) to achieve such versatility in a single model. To improve reasoning and preference alignment, we combine point-wise and pair-wise reward modeling, ensuring high-quality, human-aligned responses. For code generation, we design complexity-aware rewards in Reinforcement Learning, optimizing both correctness and efficiency. In deep search, we perform complex data synthesis and incorporate turn-level supervision during training. This enables stable long-horizon tool interactions, allowing Nanbeige4.1-3B to reliably execute up to 600 tool-call turns for complex problem-solving. Extensive experimental results show that Nanbeige4.1-3B significantly outperforms prior models of similar scale, such as Nanbeige4-3B-2511 and Qwen3-4B, even achieving superior performance compared to much larger models, such as Qwen3-30B-A3B. Our results demonstrate that small models can achieve both broad competence and strong specialization simultaneously, redefining the potential of 3B parameter models. 

---
# Using Deep Learning to Generate Semantically Correct Hindi Captions 

**Authors**: Wasim Akram Khan, Anil Kumar Vuppala  

**Link**: [PDF](https://arxiv.org/pdf/2602.13352)  

**Abstract**: Automated image captioning using the content from the image is very appealing when done by harnessing the capability of computer vision and natural language processing. Extensive research has been done in the field with a major focus on the English language which gives the scope for further developments in the same with consideration of popular foreign languages. This research utilizes distinct models for translating the image caption into Hindi, the fourth most popular language across the world. Exploring the multi-modal architectures this research comprises local visual features, global visual features, attention mechanisms, and pre-trained models. Using google cloud translator on the image dataset from Flickr8k, Hindi image descriptions have been generated. Pre-trained CNNs like VGG16, ResNet50, and Inception V3 helped in retrieving image characteristics, while the uni-directional and bi-directional techniques of text encoding are used for the text encoding process. An additional Attention layer helps to generate a weight vector and, by multiplying it, combine image characteristics from each time step into a sentence-level feature vector. Bilingual evaluation understudy scores are used to compare the research outcome. Many experiments that serve as a baseline are done for the comparative analysis of the research. An image with a score of BLEU-1 is considered sufficient, whereas one with a score of BLEU-4 is considered to have fluid image captioning. For both BLEU scores, the attention-based bidirectional LSTM with VGG16 produced the best results of 0.59 and 0.19 respectively. The experiments conclude that researchs ability to produce relevant, semantically accurate image captions in Hindi. The research accomplishes the goals and future research can be guided by this research model. 

---
# Exploring the Performance of ML/DL Architectures on the MNIST-1D Dataset 

**Authors**: Michael Beebe, GodsGift Uzor, Manasa Chepuri, Divya Sree Vemula, Angel Ayala  

**Link**: [PDF](https://arxiv.org/pdf/2602.13348)  

**Abstract**: Small datasets like MNIST have historically been instrumental in advancing machine learning research by providing a controlled environment for rapid experimentation and model evaluation. However, their simplicity often limits their utility for distinguishing between advanced neural network architectures. To address these challenges, Greydanus et al. introduced the MNIST-1D dataset, a one-dimensional adaptation of MNIST designed to explore inductive biases in sequential data. This dataset maintains the advantages of small-scale datasets while introducing variability and complexity that make it ideal for studying advanced architectures.
In this paper, we extend the exploration of MNIST-1D by evaluating the performance of Residual Networks (ResNet), Temporal Convolutional Networks (TCN), and Dilated Convolutional Neural Networks (DCNN). These models, known for their ability to capture sequential patterns and hierarchical features, were implemented and benchmarked alongside previously tested architectures such as logistic regression, MLPs, CNNs, and GRUs. Our experimental results demonstrate that advanced architectures like TCN and DCNN consistently outperform simpler models, achieving near-human performance on MNIST-1D. ResNet also shows significant improvements, highlighting the importance of leveraging inductive biases and hierarchical feature extraction in small structured datasets.
Through this study, we validate the utility of MNIST-1D as a robust benchmark for evaluating machine learning architectures under computational constraints. Our findings emphasize the role of architectural innovations in improving model performance and offer insights into optimizing deep learning models for resource-limited environments. 

---
# Artificial Organisations 

**Authors**: William Waites  

**Link**: [PDF](https://arxiv.org/pdf/2602.13275)  

**Abstract**: Alignment research focuses on making individual AI systems reliable. Human institutions achieve reliable collective behaviour differently: they mitigate the risk posed by misaligned individuals through organisational structure. Multi-agent AI systems should follow this institutional model using compartmentalisation and adversarial review to achieve reliable outcomes through architectural design rather than assuming individual alignment.
We demonstrate this approach through the Perseverance Composition Engine, a multi-agent system for document composition. The Composer drafts text, the Corroborator verifies factual substantiation with full source access, and the Critic evaluates argumentative quality without access to sources: information asymmetry enforced by system architecture. This creates layered verification: the Corroborator detects unsupported claims, whilst the Critic independently assesses coherence and completeness. Observations from 474 composition tasks (discrete cycles of drafting, verification, and evaluation) exhibit patterns consistent with the institutional hypothesis. When assigned impossible tasks requiring fabricated content, this iteration enabled progression from attempted fabrication toward honest refusal with alternative proposals--behaviour neither instructed nor individually incentivised. These findings motivate controlled investigation of whether architectural enforcement produces reliable outcomes from unreliable components.
This positions organisational theory as a productive framework for multi-agent AI safety. By implementing verification and evaluation as structural properties enforced through information compartmentalisation, institutional design offers a route to reliable collective behaviour from unreliable individual components. 

---
# ProMoral-Bench: Evaluating Prompting Strategies for Moral Reasoning and Safety in LLMs 

**Authors**: Rohan Subramanian Thomas, Shikhar Shiromani, Abdullah Chaudhry, Ruizhe Li, Vasu Sharma, Kevin Zhu, Sunishchal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2602.13274)  

**Abstract**: Prompt design significantly impacts the moral competence and safety alignment of large language models (LLMs), yet empirical comparisons remain fragmented across datasets and this http URL introduce ProMoral-Bench, a unified benchmark evaluating 11 prompting paradigms across four LLM families. Using ETHICS, Scruples, WildJailbreak, and our new robustness test, ETHICS-Contrast, we measure performance via our proposed Unified Moral Safety Score (UMSS), a metric balancing accuracy and safety. Our results show that compact, exemplar-guided scaffolds outperform complex multi-stage reasoning, providing higher UMSS scores and greater robustness at a lower token cost. While multi-turn reasoning proves fragile under perturbations, few-shot exemplars consistently enhance moral stability and jailbreak resistance. ProMoral-Bench establishes a standardized framework for principled, cost-effective prompt engineering. 

---
# Directional Concentration Uncertainty: A representational approach to uncertainty quantification for generative models 

**Authors**: Souradeep Chattopadhyay, Brendan Kennedy, Sai Munikoti, Soumik Sarkar, Karl Pazdernik  

**Link**: [PDF](https://arxiv.org/pdf/2602.13264)  

**Abstract**: In the critical task of making generative models trustworthy and robust, methods for Uncertainty Quantification (UQ) have begun to show encouraging potential. However, many of these methods rely on rigid heuristics that fail to generalize across tasks and modalities. Here, we propose a novel framework for UQ that is highly flexible and approaches or surpasses the performance of prior heuristic methods. We introduce Directional Concentration Uncertainty (DCU), a novel statistical procedure for quantifying the concentration of embeddings based on the von Mises-Fisher (vMF) distribution. Our method captures uncertainty by measuring the geometric dispersion of multiple generated outputs from a language model using continuous embeddings of the generated outputs without any task specific heuristics. In our experiments, we show that DCU matches or exceeds calibration levels of prior works like semantic entropy (Kuhn et al., 2023) and also generalizes well to more complex tasks in multi-modal domains. We present a framework for the wider potential of DCU and its implications for integration into UQ for multi-modal and agentic frameworks. 

---
# General learned delegation by clones 

**Authors**: Darren Li, Meiqi Chen, Chenze Shao, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.13262)  

**Abstract**: Frontier language models improve with additional test-time computation, but serial reasoning or uncoordinated parallel sampling can be compute-inefficient under fixed inference budgets. We propose SELFCEST, which equips a base model with the ability to spawn same-weight clones in separate parallel contexts by agentic reinforcement learning. Training is end-to-end under a global task reward with shared-parameter rollouts, yielding a learned controller that allocates both generation and context budget across branches. Across challenging math reasoning benchmarks and long-context multi-hop QA, SELFCEST improves the accuracy-cost Pareto frontier relative to monolithic baselines at matched inference budget, and exhibits out-of-distribution generalization in both domains. 

---
# MAPLE: A Sub-Agent Architecture for Memory, Learning, and Personalization in Agentic AI Systems 

**Authors**: Deepak Babu Piskala  

**Link**: [PDF](https://arxiv.org/pdf/2602.13258)  

**Abstract**: Large language model (LLM) agents have emerged as powerful tools for complex tasks, yet their ability to adapt to individual users remains fundamentally limited. We argue this limitation stems from a critical architectural conflation: current systems treat memory, learning, and personalization as a unified capability rather than three distinct mechanisms requiring different infrastructure, operating on different timescales, and benefiting from independent optimization. We propose MAPLE (Memory-Adaptive Personalized LEarning), a principled decomposition where Memory handles storage and retrieval infrastructure; Learning extracts intelligence from accumulated interactions asynchronously; and Personalization applies learned knowledge in real-time within finite context budgets. Each component operates as a dedicated sub-agent with specialized tooling and well-defined interfaces. Experimental evaluation on the MAPLE-Personas benchmark demonstrates that our decomposition achieves a 14.6% improvement in personalization score compared to a stateless baseline (p < 0.01, Cohen's d = 0.95) and increases trait incorporation rate from 45% to 75% -- enabling agents that genuinely learn and adapt. 

---
# X-Blocks: Linguistic Building Blocks of Natural Language Explanations for Automated Vehicles 

**Authors**: Ashkan Y. Zadeh, Xiaomeng Li, Andry Rakotonirainy, Ronald Schroeter, Sebastien Glaser, Zishuo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.13248)  

**Abstract**: Natural language explanations play a critical role in establishing trust and acceptance of automated vehicles (AVs), yet existing approaches lack systematic frameworks for analysing how humans linguistically construct driving rationales across diverse scenarios. This paper introduces X-Blocks (eXplanation Blocks), a hierarchical analytical framework that identifies the linguistic building blocks of natural language explanations for AVs at three levels: context, syntax, and lexicon.
At the context level, we propose RACE (Reasoning-Aligned Classification of Explanations), a multi-LLM ensemble framework that combines Chain-of-Thought reasoning with self-consistency mechanisms to robustly classify explanations into 32 scenario-aware categories. Applied to human-authored explanations from the Berkeley DeepDrive-X dataset, RACE achieves 91.45 percent accuracy and a Cohens kappa of 0.91 against cases with human annotator agreement, indicating near-human reliability for context classification.
At the lexical level, log-odds analysis with informative Dirichlet priors reveals context-specific vocabulary patterns that distinguish driving scenarios. At the syntactic level, dependency parsing and template extraction show that explanations draw from a limited repertoire of reusable grammar families, with systematic variation in predicate types and causal constructions across contexts.
The X-Blocks framework is dataset-agnostic and task-independent, offering broad applicability to other automated driving datasets and safety-critical domains. Overall, our findings provide evidence-based linguistic design principles for generating scenario-aware explanations that support transparency, user trust, and cognitive accessibility in automated driving systems. 

---
# NL2LOGIC: AST-Guided Translation of Natural Language into First-Order Logic with Large Language Models 

**Authors**: Rizky Ramadhana Putra, Raihan Sultan Pasha Basuki, Yutong Cheng, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.13237)  

**Abstract**: Automated reasoning is critical in domains such as law and governance, where verifying claims against facts in documents requires both accuracy and interpretability. Recent work adopts structured reasoning pipelines that translate natural language into first-order logic and delegate inference to automated solvers. With the rise of large language models, approaches such as GCD and CODE4LOGIC leverage their reasoning and code generation capabilities to improve logic parsing. However, these methods suffer from fragile syntax control due to weak enforcement of global grammar constraints and low semantic faithfulness caused by insufficient clause-level semantic understanding. We propose NL2LOGIC, a first-order logic translation framework that introduces an abstract syntax tree as an intermediate representation. NL2LOGIC combines a recursive large language model based semantic parser with an abstract syntax tree guided generator that deterministically produces solver-ready logic code. Experiments on the FOLIO, LogicNLI, and ProofWriter benchmarks show that NL2LOGIC achieves 99 percent syntactic accuracy and improves semantic correctness by up to 30 percent over state-of-the-art baselines. Furthermore, integrating NL2LOGIC into Logic-LM yields near-perfect executability and improves downstream reasoning accuracy by 31 percent compared to Logic-LM's original few-shot unconstrained translation module. 

---
# Variation is the Key: A Variation-Based Framework for LLM-Generated Text Detection 

**Authors**: Xuecong Li, Xiaohong Li, Qiang Hu, Yao Zhang, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.13226)  

**Abstract**: Detecting text generated by large language models (LLMs) is crucial but challenging. Existing detectors depend on impractical assumptions, such as white-box settings, or solely rely on text-level features, leading to imprecise detection ability. In this paper, we propose a simple but effective and practical LLM-generated text detection method, VaryBalance. The core of VaryBalance is that, compared to LLM-generated texts, there is a greater difference between human texts and their rewritten version via LLMs. Leveraging this observation, VaryBalance quantifies this through mean standard deviation and distinguishes human texts and LLM-generated texts. Comprehensive experiments demonstrated that VaryBalance outperforms the state-of-the-art detectors, i.e., Binoculars, by up to 34.3\% in terms of AUROC, and maintains robustness against multiple generating models and languages. 

---
# A Geometric Taxonomy of Hallucinations in LLMs 

**Authors**: Javier Marín  

**Link**: [PDF](https://arxiv.org/pdf/2602.13224)  

**Abstract**: The term "hallucination" in large language models conflates distinct phenomena with different geometric signatures in embedding space. We propose a taxonomy identifying three types: unfaithfulness (failure to engage with provided context), confabulation (invention of semantically foreign content), and factual error (incorrect claims within correct conceptual frames). We observe a striking asymmetry. On standard benchmarks where hallucinations are LLM-generated, detection is domain-local: AUROC 0.76-0.99 within domains, but 0.50 (chance level) across domains. Discriminative directions are approximately orthogonal between domains (mean cosine similarity -0.07). On human-crafted confabulations - invented institutions, redefined terminology, fabricated mechanisms - a single global direction achieves 0.96 AUROC with 3.8% cross-domain degradation. We interpret this divergence as follows: benchmarks capture generation artifacts (stylistic signatures of prompted fabrication), while human-crafted confabulations capture genuine topical drift. The geometric structure differs because the underlying phenomena differ. Type III errors show 0.478 AUROC - indistinguishable from chance. This reflects a theoretical constraint: embeddings encode distributional co-occurrence, not correspondence to external reality. Statements with identical contextual patterns occupy similar embedding regions regardless of truth value. The contribution is a geometric taxonomy clarifying the scope of embedding-based detection: Types I and II are detectable; Type III requires external verification mechanisms. 

---
# Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning 

**Authors**: Bowen Liu, Zhi Wu, Runquan Xie, Zhanhui Kang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.13218)  

**Abstract**: Scaling verifiable training signals remains a key bottleneck for Reinforcement Learning from Verifiable Rewards (RLVR). Logical reasoning is a natural substrate: constraints are formal and answers are programmatically checkable. However, prior synthesis pipelines either depend on expert-written code or operate within fixed templates/skeletons, which limits growth largely to instance-level perturbations. We propose SSLogic, an agentic meta-synthesis framework that scales at the task-family level by iteratively synthesizing and repairing executable Generator--Validator program pairs in a closed Generate--Validate--Repair loop, enabling continuous family evolution with controllable difficulty. To ensure reliability, we introduce a Multi-Gate Validation Protocol that combines multi-strategy consistency checks with Adversarial Blind Review, where independent agents must solve instances by writing and executing code to filter ambiguous or ill-posed tasks. Starting from 400 seed families, two evolution rounds expand to 953 families and 21,389 verifiable instances (from 5,718). Training on SSLogic-evolved data yields consistent gains over the seed baseline at matched training steps, improving SynLogic by +5.2, BBEH by +1.4, AIME25 by +3.0, and Brumo25 by +3.7. 

---
# Reshaping MOFs text mining with a dynamic multi-agents framework of large language model 

**Authors**: Zuhong Lin, Daoyuan Ren, Kai Ran, Jing Sun, Songlin Yu, Xuefeng Bai, Xiaotian Huang, Haiyang He, Pengxu Pan, Ying Fang, Zhanglin Li, Haipu Li, Jingjing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.18880)  

**Abstract**: Accurately identifying the synthesis conditions of metal-organic frameworks (MOFs) is essential for guiding experimental design, yet remains challenging because relevant information in the literature is often scattered, inconsistent, and difficult to interpret. We present MOFh6, a large language model driven system that reads raw articles or crystal codes and converts them into standardized synthesis tables. It links related descriptions across paragraphs, unifies ligand abbreviations with full names, and outputs structured parameters ready for use. MOFh6 achieved 99% extraction accuracy, resolved 94.1% of abbreviation cases across five major publishers, and maintained a precision of 0.93 +/- 0.01. Processing a full text takes 9.6 s, locating synthesis descriptions 36 s, with 100 papers processed for USD 4.24. By replacing static database lookups with real-time extraction, MOFh6 reshapes MOF synthesis research, accelerating the conversion of literature knowledge into practical synthesis protocols and enabling scalable, data-driven materials discovery. 

---
