# Automated Discovery Has No Universally Superior Harness 

**Authors**: Akshat Gupta, Jermaine Lei, Alexander Lu, Gopala Anumanchipalli, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2607.18235)  

**Abstract**: Autonomous discovery systems such as OpenEvolve and TTT-Discover are often used as general-purpose harnesses. However, in practice these are composite systems combining several design choices about archives, parent selection, exploration, and budget allocation into a single recipe. Because discovery runs are expensive and inherently stochastic, existing harnesses are often compared using too few independent trials to distinguish key methodological improvements from run-to-run variance. We systematically decompose OpenEvolve-style evolutionary search and the TTT-Discover search harness into its constituent components and systematically evaluate 30 budget-matched harnesses across 12 model-problem pairs using more than 3.1 million LLM rollouts and repeated-trial statistical analysis. Our results show that discovery harnesses have a generalization problem: No fixed harness is reliably superior across the evaluated model-problem pairs, and variants of OpenEvolve generally underperform simpler alternatives. Thus, harness choice is better viewed as a hyperparameter rather than as a universal recipe, and should be tailored to the specific problem and underlying model. We also find that early discovery progress predicts final performance, and use this property to present a budget-matched adaptive-allocation experiment that starts multiple harnesses, prunes weak partial runs, and reallocates compute to stronger survivors, outperforming both commitment to a randomly sampled fixed harness and a non-adaptive harness ensemble. Together, these results motivate shifting from fixed harness selection to online adaptation guided by early performance. We release all run pools including baseline null distributions for every model-problem pair as reusable statistical infrastructure against for future harness proposals. 

---
# It's Not What You Say, It's How You Say It: Evaluating LLM Responses to Expressions of Belief 

**Authors**: Kevin Du, Clara Kümpel, Michelle Wastl, Alex Warstadt  

**Link**: [PDF](https://arxiv.org/pdf/2607.18232)  

**Abstract**: Users frequently express their beliefs to large language models (LLMs). In some situations, the LLM should accept these contextual beliefs as true. In others, they should stick to their prior knowledge. Notably, users' expressions of belief (EoBs) can take linguistically diverse forms - using presuppositions, evidential and certainty markers, or varied tones - each of which may have a different persuasiveness over the LLMs. We introduce a typology to systematically evaluate how different EoBs affect whether models follow context versus prior knowledge. The typology is grounded in four linguistically motivated dimensions: form, evidentiality, epistemic stance, and tone, spanning 17 fine-grained types. By pairing these EoBs with world knowledge facts, we generate controlled EoB-query pairs that isolate the effect of linguistic variation. Using this benchmark, we evaluate 16 LLMs that differ in architecture (Llama3, Qwen3, Gemma3), scale (1B-30B parameters), and training stages (base vs instruct). We identify meaningful variations in response behavior across these axes, e.g., that bigger models and instruction models tend to be less context-following than smaller models and base models. We further identify specific EoBs that statistically significantly persuade LMs more consistently than others. Our work reveals systematic patterns in how linguistic framing affects LLM context integration, with implications for prompt engineering and model robustness. 

---
# SWE-Pruner Pro: The Coder LLM Already Knows What to Prune 

**Authors**: Yuhang Wang, Yuling Shi, Shaoqiu Zhang, Jialiang Liang, Shilin He, Siyu Ye, Yuting Chen, Kai Cai, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18213)  

**Abstract**: Pruning long context for coding agents has been a vital technology for efficient context management. While existing context pruning methods such as SWE-Pruner realize this by attaching a separate code classifier, we find the agent itself encodes internal representations indicating the relevance of code context when reading tool output. Based on this finding, we propose SWE-Pruner Pro, which prunes tool outputs directly inside the agent. Concretely, a small head turns the agent's own internal representations into a keep-or-prune label for each line, with a length-aware embedding keyed to each tool output's line count. Across two open-weight backbones and four multi-turn benchmarks, SWE-Pruner Pro saves up to 39% of prompt and completion tokens while preserving task quality, with bounded inference overhead. Notably, on MiMo-V2-Flash SWE-Pruner Pro additionally raises the SWE-Bench Verified resolve rate by +3.8% and the long-context Oolong accuracy by +2.2 points. 

---
# PPL-Factory: Task-Aware and Budget-Aware Data Selection from Language Modeling to Reasoning 

**Authors**: Hang Zhang, Warren J. Gross  

**Link**: [PDF](https://arxiv.org/pdf/2607.18199)  

**Abstract**: Not all training samples contribute equally to large language model fine-tuning. Selecting informative training samples can reduce the computational cost while preserving downstream performance. Many existing data selection methods rely on indirect heuristics, such as data quality, diversity or reasoning trace length. However, the effectiveness of these fixed criteria is task-dependent and difficult to generalize across diverse downstream tasks. Perplexity-based data selection provides a simple and model-aware solution to estimate the sample difficulty, but existing approaches typically score the entire training sequence and ignore the difference in learning objectives of language modeling and reasoning tasks. In this paper, we propose PPL-Factory, a simple and interpretable data selection framework that combines task-aware perplexity-based scores and data budget-aware selection criteria. Experiments on GSM8K demonstrate that PPL-Factory outperforms other state-of-the-art data selection methods using only $1\%$ of the training set. With $10\%$ of the data, PPL-Factory exceeds full-data fine-tuning accuracy by 0.9 on GSM8K and 4.8 on MATH. Overall, our results demonstrate that task-aware and budget-aware perplexity-based selection provides an effective and applicable approach for efficient fine-tuning. 

---
# VEHBench: A Stage-Local Diagnostic Benchmark for LLM-Assisted Vibration Energy Harvester Design 

**Authors**: Depeng Su, Yuyu Luo, Guobiao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18181)  

**Abstract**: Battery-free Internet of Things (IoT) requires iterative design of vibration energy harvesters (VEHs) under coupled physical constraints, while LLMs are emerging as interface layers for engineering workflows. However, existing engineering benchmarks primarily assess final artifact validity, offering limited insights into how LLMs behave across different stages of coupled physical design. We introduce VEHBench, an engineering-native diagnostic benchmark for LLM-assisted VEH design, featuring 763 literature-grounded tasks scored by an analytical physical oracle. VEHBench evaluates four design roles: specification triage, verifier-guided search, corrupted-state recovery, and policy-conditioned selection. Experimental results reveal that LLM capability is strongly stage-dependent: no single model consistently dominates the entire workflow, and response-control profiles expose distinct behavioral patterns across design roles. VEHBench thus provides a stage-aware foundation for evaluating, selecting, routing, and improving verifier-grounded engineering LLMs. The benchmark artifact is available at this https URL 

---
# How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs? 

**Authors**: Prakhar Gupta, Terry Jingchen Zhang, Florent Draye, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.18114)  

**Abstract**: Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an originally correct answer. We study where this susceptibility, spanning sycophancy and related cue-induced biases, lives inside the model. Across five model families and seven BCT bias types, we extract a per-bias direction from hidden states and triangulate it through three measures: probing, leave-one-dataset-out transfer, and causal intervention. The susceptibility is largely installed by alignment tuning rather than pretraining: pretrained base models barely cave to these biases, and their activations carry no cue-specific signal beyond question content. Within aligned models, each bias becomes a single coherent direction that we can both decode and steer along, recovering the unbiased answer across every family we test. The biases stay representationally distinct, however: cross-bias entanglement is model-specific rather than a property of the bias category, and even behaviorally similar biases occupy different directions. The same intervention also serves as a modest debiasing tool, recovering a meaningful share of bias-induced errors while preserving most correct answers across all instruct families. Cue-induced bias is therefore best understood not as a single flaw in LLMs but as a family of distinct, causally active directions that alignment tuning installs. 

---
# VDAR-Router: Adaptive LLMs Routing via Verbalized Query Difficulty Analysis Retrieval 

**Authors**: Yu-Chien Tang, Jun-Chen Hung, Wen-Chih Peng, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2607.18098)  

**Abstract**: Large language models are increasingly used in practical systems, making efficient model selection important for reducing deployment cost. LLM routing has emerged as a practical solution for allocating each input query to an appropriate model under a desired cost-performance trade-off. Existing routing methods often estimate model suitability from the surface semantics or embedding similarity of the input query. However, such methods may ignore the underlying difficulty of a query, leading to suboptimal routing decisions. To address the challenge, we propose VDAR-Router, a difficulty-aware retrieval-based routing framework. For each input query, VDAR-Router first generates an explicit difficulty analysis. It then retrieves historical examples with similar difficulty profiles. Based on the retrieved records, it estimates candidate model suitability and selects the model using a reward function that considers both performance and cost. Experiments on three datasets show that VDAR-Router consistently achieves better cost-performance trade-offs than existing baselines. These results demonstrate the effectiveness of difficulty-aware retrieval for training-free LLM routing. Case studies further show that explicit query analysis helps retrieve more relevant examples and supports more reliable routing decisions. 

---
# Modeling turn-taking with distant viewing: investigating silence thresholds in human and AI-generated discourse 

**Authors**: Taylor Arnold, Nicolas Ballier, Artem Saloev  

**Link**: [PDF](https://arxiv.org/pdf/2607.18076)  

**Abstract**: This study investigates silence gaps in two kinds of audiovisual material. We analysed thirty US situational comedies and fifty-one synthetic podcasts generated with Google NotebookLM. Gaps were compared across speaker gender, assigned from a fundamental-frequency threshold estimated in Praat, and across production settings. 

---
# Pancasila-Dilemmas: Evaluating Large Language Models on Indonesian Human Value Dilemmas Grounded in Pancasila 

**Authors**: Supryadi, Irfan, Julianti, Darren Keanly Martin, Jayvin Fernando, Yuqi Ren, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.18066)  

**Abstract**: The value alignment of large language models (LLMs) is crucial for ensuring responses align with human intention and value preferences. However, most evaluations of value alignment focus on Western or universal values, while assessments grounded in the value systems of specific countries remain scarce. In this paper, we introduce Pancasila-Dilemmas, an evaluation dataset of 1,834 questions derived from Indonesian news, classified by 5 values of Pancasila: Religion, Humanity, Unity, Democracy, and Social Justice. This dataset reflects daily life in Indonesia, making it suitable for measuring the value alignment of LLMs deployed for Indonesia. To ensure a more rigorous evaluation, we choose scenarios containing dilemmas. The dataset is proofread by native speakers and answered by 5 diverse Indonesian citizens. We evaluate 50 closed- and open-source LLMs on our dataset. Results reveal that all evaluated LLMs achieves less than 0.5 Probability Match Score (PMS) and 0.72 Max-Vote Agreement Score (MVAS). Compared by each values, LLMs mostly struggle in Religion and Unity dilemma cases. This highlights a significant gap in capturing Indonesian values. The dataset is publicly available at this https URL. 

---
# An Early Warning of Emerging Biosecurity Risks in Frontier LLMs 

**Authors**: Zhida He, Xia Hu, Baichen Le, Chunxiao Li, Jiajia Li, Lijun Li, Chaochao Lu, Jing Shao, Youbang Sun, Hua Tang, Xiang Wang, Xiao Wang, Xiaoyu Wen, Tong Wu, Jia Xu, Peng Yu, Shu Yu, Jie Zhang, Qiaosheng Zhang, Yi Zhang, Xing-Ming Zhao, Tianhang Zheng, Ziyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.18056)  

**Abstract**: Frontier large language models (LLMs) are increasingly integrated into scientific workflows, yet their growing biological capabilities may outpace current safeguards. To assess the biological risks of frontier models, we develop Intern-BioBreaker, a specialized bio-red-teaming model, together with an integrated computational-to-physical framework that couples model-level stress testing with wet-lab validation. Within this framework, Intern-BioBreaker generates targeted jailbreak prompts to test whether aligned models can be induced to provide operational guidance for safety-sensitive biological tasks or produce sequence-level outputs with potentially harmful properties. Selected sequence outputs are then carried forward for DNA synthesis, host expression, and orthogonal protein verification to assess whether model-generated designs can yield the intended biological products. Our evaluation reveals a concerning gap between text-level safeguards and the risks posed by capable scientific models: (i) Intern-BioBreaker outperforms baseline attack models and reveals widespread bio-risk jailbreak vulnerabilities across both open-weight and proprietary frontier LLMs, with several targets reaching near-saturated or 100% task-level attack success rate (ASR); (ii) in sequence-level case studies, GPT-5.5 can be induced to generate modified viral candidate sequences with pathogenic potential; the corresponding translated proteins may exhibit even stronger receptor-binding affinity and thus enhanced infection potential; and (iii) end-to-end verification shows that selected model-generated biological designs are not merely textual artifacts, but can be physically realized under controlled experimental settings. These findings underscore the need for stronger biological red-teaming, nucleic acid synthesis screening, and safety mechanisms that keep pace with model capabilities. 

---
# What Transfers Under Source Shift? Definitions, Examples, and Fine-Tuning for Climate Disclosure Classification 

**Authors**: Guosheng Li, Fenghui Ren, Bin Liu, Chuan Yu, Kaiying Ji, Lin Yue, Jun Shen, Sasa Qian  

**Link**: [PDF](https://arxiv.org/pdf/2607.17952)  

**Abstract**: Climate disclosure classification is a fundamental task for analysing corporate climate disclosures, yet such disclosures appear in many different sources -- annual reports, press releases, and earnings calls -- that differ in length, purpose, and writing style. Existing evaluations are mostly conducted within a single source, leaving open whether common LLM adaptation strategies remain effective under source shift. We reframe climate disclosure classification as a cross-source adaptation problem and study three widely used adaptation strategies -- definitions, examples, and fine-tuning -- across eleven open- and closed-source LLMs, using two corpora that share the same label space but come from different sources. We find that all strategies bring positive cross-source gains on average, but the strongest in-source strategies are not the strongest cross-source ones: similarity-based retrieval and LoRA fine-tuning gain most in-source but lose most of that advantage under source shift; randomly selected few-shot examples, a weaker in-source baseline, retain their advantage more reliably; definitions transfer most consistently, though only when their granularity matches the target text. Across these strategies, when the source changes, simpler is often safer. 

---
# DeLIVeR: Decomposed Learning for Information-grounded Veracity Recognition via Reinforced Knowledge Graph Exploration 

**Authors**: Cong Hoan Nguyen, Thomas Hoang, Hieu Minh Duong, Long Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2607.17935)  

**Abstract**: Automated fact-checking remains a challenge for Large Language Models (LLMs) due to "query brittleness" in traditional retrieval systems. We propose DeLIVeR (Decomposed Learning for Information-grounded Veracity Recognition), a framework that treats evidence retrieval as a reinforced strategic exploration task. DeLIVeR utilizes a Planner LLM to decompose complex claims into targeted question sets, which are used to traverse structured Knowledge Graphs (KGs) for high-precision evidence. We optimize the Planner's policy using Group Relative Policy Optimization (GRPO) with a reward system prioritizing structural diversity and verdict accuracy. Our evaluation on LIAR, FEVER, and PolitiFact shows that DeLIVeR significantly outperforms state-of-the-art baselines. Using Qwen2.5-7B, our framework achieved peak F1-scores of 83.73, 84.57, and 79.70 respectively, representing a 10-15% improvement over HippoRAG2. By shifting to a reinforced question-planning strategy, DeLIVeR effectively bridges multi-hop reasoning gaps and provides an auditable, transparent path for verifiable misinformation detection. 

---
# Zero Hallucination, by Construction: Hallucination-Aware Layered Oversight for Trustworthy Enterprise AI 

**Authors**: Bogdan Raduta, Horia Velicu, Alexandru Preda, Serban Chiricescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17883)  

**Abstract**: Enterprises will not deploy AI agents they cannot trust, and the most-cited reason for distrust is hallucination: confident, fluent output that is simply not true. The common response is to wait for a model that does not hallucinate. We argue that this is the wrong target. Large language models are, by construction, capable of generating unsupported text, and no amount of scale removes the possibility; a faithfulness judge bolted onto a raw model catches some errors but still ships others, and even well-curated retrieval pipelines have been shown to fabricate citations. We reframe the goal: "zero hallucination" is not a property a model possesses but a property a system enforces. We present HALO (Hallucination-Aware Layered Oversight), an assurance architecture which treats hallucination as a containable failure mode rather than an eliminable one. HALO composes six layers of defense: grounded generation over retrieved, approved content; constrained, deterministic execution that bounds where the model can err; multi-signal verification that scores every output for groundedness and hallucination using both an LLM judge and evidence-based checks against the source text; calibrated abstention, so the system declines rather than guesses when grounding is insufficient; total traceability of every retrieval, tool call, and generation; and continuous oversight that detects drift, alerts on threshold breaches, and closes the loop by regenerating and statistically validating improved agents. We detail each layer, give particular attention to evidence-based confidence (which verifies extractions against the source document rather than trusting the model's self-reported certainty), and illustrate the architecture on a regulated claims-extraction workload 

---
# When a Name Is Not a Name: A Benchmark Dataset and Distilled Reasoning for Culturally Entangled Bangla Homographs in Low-Resource LLMs 

**Authors**: Md. Asaduzzaman Shuvo  

**Link**: [PDF](https://arxiv.org/pdf/2607.17828)  

**Abstract**: Many Bangla words are at once personal names and culturally loaded common nouns, "Maya" is both a girl's name and a word for affectionate compassion. Choosing the right reading demands cultural knowledge that is scarce in the pretraining data of modern language models. We introduce Culturally Entangled Homograph (CEH) disambiguation and build a Bangla benchmark of 1,516 expert-verified sentences (3,032 labelled occurrences) in which one word appears twice with two distinct readings, each labelled with a culturally grounded category and an explanation of the reasoning behind it. Across open- and closed-source models, we find a systematic dominant-meaning bias: models default to the common-noun sense and overlook the name. A Bangla-specific model fails under every prompting regime we test, showing that language-specific pretraining alone does not confer cultural grounding. We further show that contrastive chain-of-thought prompting can sharply reduce this bias without training, and that distilling cultural explanations teaches small (1-3B) models to reason toward the correct reading rather than memorise labels, cutting dominant-meaning bias from as high as 100% to under 5% and turning the failed Bangla-specific model into our strongest system. Dataset and code are available at this https URL. 

---
# ESCUCHA: A Spanish Speech Benchmark for Heterogeneous Acoustic Conditions 

**Authors**: Fernando López, Ana Ayala, Guillermo Segovia, Fernando Ibáñez, Ana Martínez, Pablo Gómez, Jordi Luque  

**Link**: [PDF](https://arxiv.org/pdf/2607.17812)  

**Abstract**: As large audio language models (LALMs) advance, robust evaluation frameworks have become essential. In this context, Spanish speech understanding under realistic acoustic conditions has received particularly little attention. We introduce ESCUCHA, the first Spanish speech understanding benchmark designed to evaluate LALMs across heterogeneous acoustic conditions and reasoning abilities. ESCUCHA comprises 1,000 human-curated questions paired with audio, totaling 162.9 hours sourced directly ``from the wild'' rather than drawn from existing datasets, with durations ranging from a few seconds to over 80 minutes. The benchmark emphasizes reasoning, spanning 9 perceptual and 10 reasoning categories, and it captures linguistic diversity through multiple Spanish accents and non-normative speech. ESCUCHA further includes multi-audio questions, spoken questions, and audio instructions, and it flags which questions support open-ended evaluation. Benchmarking several state-of-the-art multimodal and speech models reveals substantial performance gaps relative to trained humans. 

---
# When to Use Extra Context: Evidence-Grounded Terminology Adaptation for Simultaneous Speech Translation 

**Authors**: Zeyu Yang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2607.17766)  

**Abstract**: Extra context is valuable for simultaneous speech translation of technical talks, but injecting the entire document context into every streaming segment is often too coarse. Through diagnostic experiments, we find that context gains mainly come from paper-specific terminology recovery rather than uniform semantic enhancement. We therefore propose EGTA, an Evidence-Grounded Terminology Adaptation framework that builds a document terminology memory, selects compact candidate terms conditioned on the current streaming state, and adapts ASR/speech-side and decoder-side decision spaces using only the selected terms. EGTA can be instantiated in cascaded, end-to-end, and generation-only SimulST settings without full-model fine-tuning. We evaluate EGTA on an ACL technical-talk SimulST evaluation suite consisting of MCIF-dev and ACL60/60-dev. On MCIF-dev, EGTA-RG improves BLEU by +1.05/+0.59, XCOMET-XL by +0.019/+0.006, named-entity recall by +79\%/+73\% relative, and acronym recall by +0.099/+0.171 on En$\rightarrow$Zh and En$\rightarrow$De. Across MCIF-dev latency settings, EGTA consistently improves XCOMET-XL, named-entity recall, and acronym recall. External validation on ACL60/60-dev further shows consistent terminology-recall gains without additional fine-tuning. Shuffled-memory controls and activation audits provide evidence that the improvements are tied to paper-specific evidence alignment rather than generic context prompting. 

---
# Large Language Models for Citation Function Classification 

**Authors**: Daniel Vodička, Jakub Šmíd, Pavel Král, Christophe Cerisara  

**Link**: [PDF](https://arxiv.org/pdf/2607.17738)  

**Abstract**: Citation function classification plays a crucial role in understanding the relationships between scientific publications and advancing bibliometric analysis. This study presents one of the first comprehensive evaluations of multiple state-of-the-art (SOTA) large language models (LLMs) for citation function classification, achieving new SOTA results on the ACL-ARC dataset. We systematically compare five models (Mistral 7B, Orca 2-7B, LLaMA 3.1-8B, Falcon 7B, and SciBERT) across zero-shot, few-shot, and fine-tuning approaches. Our fine-tuned Falcon 7B model achieves a 73.3% macro F1 score on ACL-ARC, representing a significant improvement over previous methods. Additionally, we introduce AC3, a novel dataset featuring a seven-category annotation scheme that distinguishes between neutral acknowledgments and explicit evaluative stances (more opinion-oriented citations - criticizing, complimenting, contradicting). The dataset is implemented across four context extraction variants to systematically evaluate the impact of contextual scope on classification performance. We also provide detailed analysis of model performance, experimental configurations, and limitations to guide future research in this domain. To our knowledge, this is one of the first studies dedicated to comprehensive model comparison for citation function classification, addressing a gap identified in recent surveys. 

---
# C$^2$KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference 

**Authors**: Chuheng Du, Junyi Chen, Hanlin Tang, Kan Liu, Tao Lan, Lin Qu, Chaoyue Niu, Shengzhong Liu, Guihai Chen, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17715)  

**Abstract**: Long-context inference is central to modern large language model (LLM) applications such as retrieval-augmented generation and multi-document reasoning. To mitigate the growing inference cost, recent work has explored key-value (KV) cache reuse to reduce redundant prefill computation. However, existing reuse methods primarily focus on computation savings and overlook a critical bottleneck in long-context LLM serving: the cost of storing and accessing large KV caches. While KV compression appears to be a natural complement, naively combining compression with non-prefix KV reuse often leads to severe accuracy degradation. In this work, we propose C$^2$KV, a unified framework for non-prefix KV reuse that jointly optimizes KV extraction and inference-time concatenation. C$^2$KV learns a composable and compressed KV cache manifold that is explicitly designed to be position-agnostic. Our approach introduces a lightweight sidecar Extractor with learnable compression tokens and a structured attention flow, enabling modular KV representations that can be flexibly reused and concatenated without modifying the frozen base model. We further employ a compression-concatenation co-training strategy to align extraction-time representations with their downstream reuse behavior. Extensive experiments across multiple long-context benchmarks and model families demonstrate that C$^2$KV significantly reduces KV cache storage and transfer costs, achieving up to 17$\times$ inference speedup under long contexts, while preserving generation quality. 

---
# AEGIS: Awareness-Enhanced Guidance for Iterative Safeguard 

**Authors**: Kyungwon Park, Sangmin Lee, Heejae Chon, Hyungu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17713)  

**Abstract**: Span-level rationales are often assumed to improve controllability in text detoxification, but it remains unclear when such guidance helps and when it introduces trade-offs. We present Awareness-Enhanced Guidance for Iterative Safeguard (AEGIS) as an exploratory framework for studying span-guided multilingual detoxification across English, Mandarin Chinese, and Korean. AEGIS combines span-level detector outputs with frozen generator backbones, allowing harmful spans, intensity labels, and target attributes to be provided as structured guidance during rewriting. Rather than claiming state-of-the-art detoxification performance, we analyze how span guidance affects the balance between toxicity reduction and meaning preservation across generator families, model scales, and languages. Our results suggest that span-guided detoxification is conditionally useful: explicit rationales change the trade-off between toxicity reduction and meaning preservation, but their effects depend strongly on the generator backbone and the linguistic context. These findings highlight both the promise and the limitations of span-level control signals for multilingual detoxification. 

---
# Tokenizing Crosslingual Homographs 

**Authors**: Rotem Brillant, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2607.17689)  

**Abstract**: Multilingual language models rely on shared subword vocabularies to represent multiple languages within a limited number of token units. While such sharing is often useful, it can also create cases in which identical surface forms are treated too uniformly across languages, even when their meanings or usage differ. We investigate this limitation through cross-lingual homographs and false friends, and examine whether introducing language information earlier in the tokenization process can improve their treatment. We propose a simple tokenizer-level intervention based on language cues: language-specific characters replacing initial characters of shared-vocabulary words, reducing common identity during vocabulary construction. In intrinsic analysis, we find through tokenizer-level statistics that BPE and UnigramLM often treat cross-lingual homographs in a largely language agnostic way, whereas the context-sensitive SaGe tokenizer diverges more strongly; our intervention removes this gap. In downstream English-to-X machine translation, our cues yield modest improvements in several settings, especially under BPE, although the effect is not consistent across all languages and evaluation sets. Overall, the findings suggest that adding lightweight language information at the tokenizer level is a promising direction for further exploration. 

---
# Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time Collaboration 

**Authors**: Jie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17531)  

**Abstract**: Test-time collaboration, including self-consistency, best-of-N selection, critic models, and verifier pipelines, is often credited with broadly improving LLM reasoning, yet its gains are uneven and sometimes negative. We ask when training-free collaboration should be expected to help. For a fixed candidate pool, we decompose a selector or verifier's net gain into measurable factors: recoverable mass, verification-signal coverage, conditional selection quality, and harm to already-correct outputs. This reframes collaboration as a candidate-selection problem rather than as an intrinsic property of a multi-agent topology. Across LiveCodeBench, MATH Level-5 hard subjects, and GPQA-Diamond, gains are bounded first by the oracle gap and then by signal fidelity, which we measure directly as candidate-level agreement between verifier verdicts and official labels. On LiveCodeBench, a public-test verifier (MCC 0.825) gains +8.14 percentage points (pp) over a first-sample baseline; a generated-test verifier (MCC 0.248) improves by +2.70pp and is not statistically distinguishable from an LLM selector, but operates at near-zero harm versus the selector's 4.69% harm rate. On MATH, a symbolic answer-equivalence selector beats self-consistency by +4.67pp, while LLM selectors are negative. On GPQA-Diamond, recoverable mass is only 3.03% and 87.54% of candidate pools are answer-identical; a weaker model's pools shrink both further, suggesting that oracle gap is a joint property of task, model, and sampling configuration. Our framework yields a practical pre-deployment diagnostic: estimate the oracle gap, then measure coverage, signal fidelity, and harm before investing in collaboration. 

---
# Token-Level Off-Policy Learning for Faithful Generation Under Distribution Shift 

**Authors**: Zitong Huang, Gustavo Lucas Carvalho, Deqing Fu, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2607.17524)  

**Abstract**: We propose Token-Level Off-Policy Labeling (TOPL), an off-policy training paradigm that reframes post-training as a token-level correctness prediction task. Our key intuition is that by training the model to distinguish good and bad tokens in a response, we naturally guide the model towards generating good tokens, while avoiding the pitfalls that come with directly training the model to generate off-policy tokens. Experiments on document summarization tasks show that TOPL achieves strong out-of-distribution generalization across 11 datasets against a diverse set of sequence-level and token-level baselines. We further demonstrate that TOPL transfers effectively to machine translation, suggesting that its benefits generalize across different faithful generation tasks. Through ablation studies, we confirm that our token-level learning signal is critical to good performance; sequence-level analogues do not confer similar benefits. Finally, we show that TOPL induces interpretable model updates: the LoRA adapters learned through TOPL function as linear classification heads and steering vectors. 

---
# Multilingual Sentence Embeddings for Linguistic-Integrated Reliability Audit 

**Authors**: Ummugul Bezirhan, Ji Yoon Jung, Matthias von Davier  

**Link**: [PDF](https://arxiv.org/pdf/2607.17466)  

**Abstract**: Multilingual assessment systems commonly rely on translation for scoring and quality-control processes. We evaluate whether multilingual sentence embeddings can replace translated English input for Linguistic-Integrated Reliability Auditing (LiRA) across 11 PIRLS constructed-response items and three embedding models. Native-language embeddings reproduced translation-based reliability estimates closely while recovering responses excluded after translation failure, with no meaningful change in reliability. 

---
# How Reliable Are Multimodal Signals of Conversational State? Evidence from Remote Dyadic Collaborative Tasks 

**Authors**: Tahiya Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2607.17452)  

**Abstract**: Measuring conversational states such as cognitive load and conversational power from multimodal behavior requires characteristic features that are not only predictive but also reliable across task contexts. We present a three-dimensional evaluation framework assessing predictive accuracy, cross-task generalizability, and test-retest reliability, applied to interactional, acoustic, and linguistic features extracted from dyadic conversations during collaborative tasks performed over a video-conferencing platform (AVCAffe dataset; 53 dyads, 9 tasks). Our results show that no single feature family dominates all three dimensions. Linguistic features show the highest predictive accuracy for cognitive load but collapse under cross-task evaluation, revealing sensitivity to task-specific vocabulary. Additionally, acoustic reliability, often reported as evidence of feature stability, degrades once speaker identity is controlled, confirming that standard prosodic features measure vocal characteristics rather than conversational state. Interaction features provide the only genuinely reliable signal, unchanged after speaker normalization. Interestingly, classifying power role remained near chance baseline across all conditions, indicating limitations of task-level aggregated behavior for predicting power role in conversation. Our findings reveal three insights: (1) linguistic features predict best but generalize poorly across task contexts; (2) acoustic reliability collapses to near-zero once speaker identity is controlled, challenging standard evaluation practice; and (3) interaction features provide the only genuinely reliable signal, with floor dominance predicting within-dyad cognitive load asymmetry. These results argue for speaker normalization and multi-dimensional evaluation as prerequisites for context-aware, robust multimodal feature selection in conversational systems. 

---
# The Librarian Who Refused to Code: Model-Dependent Identity Enactment in LLM Code Generation 

**Authors**: Shayell Aharon Salomon, Noam Israel, Ido Safruti, Amir Shaked  

**Link**: [PDF](https://arxiv.org/pdf/2607.17420)  

**Abstract**: Biographical personas are widely used in system prompts, but their effects on code generation are rarely evaluated under controlled, pre-registered conditions. We tested four prompt conditions (no persona, two engineer personas, and a research-librarian persona), 12 code-generation tasks, two frontier models, and five runs per cell (480 completions). Persona effects differed between the two tested models. Under the pre-registered mixed-effects analysis, the condition-by-model interaction was significant for provider-reported output tokens; a post-hoc visible-character measure showed the same qualitative pattern. Six GPT-5.5 completions were length-capped and are reported separately. On Claude Opus, the minimalist engineer persona reduced visible output by 30% (33% in provider tokens) without improving correctness, while the thorough engineer persona increased output without a correctness gain. In an exploratory post-hoc analysis, the librarian persona elicited in-character disclaimers in 55 of 60 Opus responses and 12 genuine no-code responses, lowering mean correctness from 0.92 to 0.67. GPT-5.5 produced neither behavior in its 59 non-truncated responses. These results are consistent with personas acting as Model-Dependent behavioral-policy biases rather than universal quality interventions. We release raw completions, derived scores, analysis artifacts, a pre-registration document, and an execution gate log; end-to-end test-based rescoring requires an unreleased task harness. 

---
# Team DACTYL at PAN 2026: Bayesian Data Mixing and Empirical X-risk Minimization for AI-text Detection 

**Authors**: Shantanu Thorat  

**Link**: [PDF](https://arxiv.org/pdf/2607.17382)  

**Abstract**: Existing research shows that AI-generated text detection classifiers achieve strong in-distribution (ID) performance but do not maintain the same performance on out-of-distribution (OOD) texts, suggesting overfitting to dataset-specific features. However, combining different training datasets doesn't always improve performance and, in some cases, can even encourage shortcut learning. To address this issue, we fine-tune BERT-tiny models with Bayesian classification heads to select texts across three different datasets to use as a consolidated training set. We trained three different classifiers: fine-tuned DeBERTa-V3-large and ModernBERT-large classifiers via empirical X-risk minimization, and an MCGrad model that calibrates the predictions from the ModernBERT-large classifier. The DeBERTa-V3-large-large classifier achieves a mean score of 0.882 on the PAN 2026 test set across five metrics: AUROC, $F_1$, C@1, Brier score, and $F_{0.5u}$. ModernBERT-large achieves a score of 0.96 while MCGrad achieves the best score of the three with a mean score of 0.974, ranking second on the leaderboard. Our results highlight that careful dataset curation can lead to strong OOD performance. We release our ModernBERT-large and DeBERTa-V3-large models at this https URL . 

---
# Safety That Does Not Transfer: Cross-Lingual Clinical Correctness Drift in Deployable Medical Language Models 

**Authors**: Anthonio Oladimeji Gabriel, Dimeji Olawuyi, Toba Ajayi, Temilola Aderemi  

**Link**: [PDF](https://arxiv.org/pdf/2607.17270)  

**Abstract**: Safety evaluation of large language models is conducted predominantly in English and predominantly on frontier systems. Neither condition describes how such models are encountered in low-resource health settings, where small quantised systems are run locally and queried in local languages. We ask whether clinical safety established in English transfers to Hausa, and whether any failure is attributable to the language, the clinical task, or the class of model that low-resource deployment admits. Matched English-Hausa question pairs were built for three conditions of high burden in northern Nigeria: malaria, sickle cell disease, and tuberculosis, probing knowledge recall, emergency triage, a leading question inviting a contraindicated action, and a traditional-remedy claim. Six models were evaluated: five locally deployable systems of 4-9 billion parameters, two medically fine-tuned, and one frontier system. All 128 responses were scored against Nigerian national treatment guidelines by two fluent Hausa speakers working independently and blind to one another. Among locally deployable models, mean clinical correctness fell from 1.57 in English to -0.03 in Hausa, on a scale where 2 denotes a correct answer and -1 an actively harmful one. The frontier model moved from 2.00 to 1.75 and produced no response judged harmful in either language. Drift was consistent across all three conditions. Inter-rater agreement was substantial for clinical correctness (kappa = 0.70); agreement on harm was initially poor (kappa = 0.22) and is examined in detail. Because a frontier model answers the same questions competently in Hausa, the deficit is a property neither of the language nor of the clinical material, but of the deployable tier. 

---
# Debate-on-Graph: Reliable and Adaptive Reasoning of Large Language Model on Uncertain Knowledge Graph 

**Authors**: Peiji Yu, Xin Chen, Tianxing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17266)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in natural language processing. However, LLMs often suffer from hallucinations and lack of relevant knowledge when dealing with question answering (QA) tasks. To mitigate these issues, knowledge graphs (KGs) have been utilized to enhance LLM reasoning. Nevertheless, KGs often contain noise and errors, while existing KG-enhanced LLM approaches are generally unable to identify and filter such noisy and erroneous content, which can instead amplify hallucinations and pose challenges for reliable reasoning. Uncertain knowledge graphs (UKGs), which associate each triple with a confidence score to quantify uncertainty, offer a promising direction to address this challenge. Compared with prior work, we investigate how to leverage UKGs to support LLMs for QA. We propose Debate-on-Graph (DoG), a new framework that enables LLMs and UKGs to collaborate adaptively for reliable reasoning. Specifically, we first design a heuristic search algorithm tailored for UKGs to extract reliable and question-relevant subgraphs, thereby reducing noise and errors in retrieved knowledge. We then introduce a Multi-Agent Debate mechanism, which yields reliable answers through adaptive adversarial debates, aiming to fully exploit the knowledge in UKGs while preserving the reliability of retrieved evidence. Extensive experiments on four benchmark QA datasets show that DoG achieves state-of-the-art performance over existing LLM reasoning methods and KG-based baselines, while enabling reliable and adaptive reasoning. Our code is available at this https URL. 

---
# Should Missing Modalities Always Be Necessary to Repair for Multi-modal Sentiment Analysis? 

**Authors**: Yubo Gao, Haotian Wu, Xiaoyu Xu, Yibo Yan, Hong Chen, Ruoshui Peng, Fei Pan, Puay Siew Tan, Zhuoran Gao, Yonghua Hei, Jie Zhang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17262)  

**Abstract**: Existing methods for multimodal sentiment analysis (MSA) under missing modalities usually follow a repair-first paradigm. We revisit this assumption and ask: \emph{should every missing modality be repaired?} A per-sample oracle analysis shows the answer is not always: full-modality input is optimal for only a small fraction of samples, and every modality subset is preferred by some samples. These results suggest that adding or repairing modalities may not always improve prediction, and that the utility of each modality is sample-dependent. Building on this finding, we propose \textbf{S}ufficiency-\textbf{I}nformed \textbf{E}vidential \textbf{V}al\textbf{vE} (\textbf{SIEVE}) that turns ``whether to repair'' into an explicit, learnable decision at the sample level. SIEVE compares a direct prediction branch with a repair branch, derives an empirical sufficiency signal from their per-sample loss gap, and routes each input through an evidential gate that jointly models sufficiency and its epistemic uncertainty. SIEVE is repair-agnostic: it operates as a plug-and-play decision on top of any explicit or implicit repair module, without modifying its internal design. Experiments on CMU-MOSI and IEMOCAP show that SIEVE consistently improves representative repair backbones across evaluated missing rates, and approaches the per-sample dual-branch achievable optimum. 

---
# EvolvingWorld: An Open-Schema Framework for Co-Evolving Role-Play Agents and World Model in Interactive Literary World 

**Authors**: Qing Zong, Yue Guo, Mengxin Yang, Yiwen Guo, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.17250)  

**Abstract**: This paper introduces EvolvingWorld, a framework and benchmark for character and world co-evolution in interactive literary worlds. Existing systems either treat interactive literary simulation as static persona imitation or isolated scene generation, failing to capture how characters and worlds evolve together over time. To address this, EvolvingWorld models literary simulation as a long-horizon process where characters interact, scenes progress, and character and world states are persistently updated. Unlike prior systems relying on fixed schemas, EvolvingWorld adopts an open-schema framework to support simulation across diverse literary worlds. The framework consists of two coupled modules: a Character Agent for multi-character role-play and persistent profile evolution, and an LLM-based World Model for global and location/entity-level state maintenance and scene progression. Based on this architecture, we formulate 7 trainable tasks for scene initialization, interaction generation, and state update. We construct a dataset from 57 books, producing 138,596 supervised training samples and 222 snapshots for testing. Furthermore, we introduce a trajectory-level LLM-as-Judge evaluation protocol spanning 10 dimensions and 20 metrics. Experiments show that EvolvingWorld can improve long-horizon simulation by effectively maintaining persistent, coherent character and world development. 

---
# AI_LectureNote: A Retrospective Pilot Study of a Post-ASR Workflow for English-Script Rendering and Semantic Drift in Korean-English Medical Lectures 

**Authors**: Kyeongeon Lee, Donghoon Chang, Seungryeol Baek, Taehong Kim, Wonjun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17237)  

**Abstract**: AI_LectureNote is a historical, readability-oriented post-ASR workflow for Korean-English medical lectures. It rewrites speech-to-text output into study transcripts while restoring Latin-script medical terms rather than Korean phonetic transliterations. We retrospectively evaluate the workflow on four author-recorded lectures across five conditions. In this pilot, post-processing raised the macro English-script rendering rate from 0.39 to 0.71 on the whisper-1 path and from 0.26 to 0.65 when applied to 3-minute chunked gpt-4o-transcribe output. However, English-script rendering did not imply semantic faithfulness: the two post-processed conditions showed semantic drift in 34 and 36 of 282 reference sentences and polarity failures in 11 and 13 of 101 polarity-cue rows. A descriptive cross-input comparison suggested different candidate failure patterns: polarity-failure sets overlapped more strongly across front-ends (Jaccard 0.60; 9 shared of 15 unioned failures) than general semantic-drift sets (Jaccard 0.23; 13 shared of 57 unioned drifts). This single-annotator pilot documents concrete failure modes rather than population rates and supports evaluating surface accuracy, term-script rendering, chunk-level script consistency, and medical-meaning preservation separately. 

---
# Robust Summarization of Doctor-Patient Conversations: TalTech Systems for the Beyond Transcription Challenge 

**Authors**: Aivo Olev, Tanel Alumäe  

**Link**: [PDF](https://arxiv.org/pdf/2607.17230)  

**Abstract**: This paper describes TalTech's submissions to the Beyond Transcription Challenge (BeTraC), which requires generating SOAP notes directly from long doctor-patient conversation recordings, without intermediate transcription. After screening open-weight speech LLMs for long-audio robustness, we adapted Voxtral Mini (lightweight track) and Voxtral Small (heavyweight track) with LoRA supervised fine-tuning followed by DAPO reinforcement learning that uses the challenge metric, Open Medical Concept F1, as its reward. Our systems ranked first in both tracks, and an independent LLM-as-a-judge evaluation showed the lowest hallucination rate among all submissions, indicating that reinforcement learning against a concept-matching metric need not compromise factual reliability. We also find that fine-tuning on text transcripts transfers well to speech input and appears to improve robustness on out-of-domain real recordings. 

---
# Literary Non-Style in LLM-Generated Text 

**Authors**: Cory Massaro  

**Link**: [PDF](https://arxiv.org/pdf/2607.17228)  

**Abstract**: Prior work on LLM-generated text has demonstrated quantitative and qualitative departures from text produced by humans. LLM-generated texts differ from human writing in style, resulting in a characteristic textual "feel," while the semantic range of LLMs is much restricted compared to that of humans. In this contribution, I note simple but consistent patterns in the statistical distribution of n-grams within LLM-generated text. Via qualitative analysis of these n-grams, I reveal deficiencies in LLM style. Because higher-order n-grams correlate to semantic content, I conclude that questions of style and semantics are not cleanly separable. 

---
# Auditing Question-Order Effects in Large Language Models with the QQ Equality: Mechanism Characterization and a Saturation Caveat 

**Authors**: Pilsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17219)  

**Abstract**: Human survey respondents exhibit question-order effects that satisfy the QQ (quantum question) equality, an a priori, parameter-free prediction of the projective quantum question-order model. We develop the QQ equality into an audit criterion for sequential judgments of autoregressive large language models (LLMs). Theoretically, we characterize which mechanism classes satisfy it robustly: marginal-independent kernels satisfy QQ iff all four mismatch transition rates coincide (a class containing the 2D rank-1 projective model with a fixed measurement pair under state variation); a polarity- and position-dependent repetition family is characterized by an exact cross-symmetry condition with closed-form violations; QQ-satisfying behaviors are closed under order-matched mixing; and the rank-2 Contextuality-by-Default criterion translates into audit coordinates as $|\qQQ|\le\OSS$, where $\OSS$ (the order-sensitivity score) totals the order sensitivity of the two marginals. Methodologically, we develop a pre-specified, audit-logged pipeline applicable to any model exposing next-token log-probabilities; it combines worst-case robustness envelopes, sampling-consistency spot checks, full label counterbalancing, and a saturation diagnostic. Empirically, in a first-signal pilot on an open-weight instruction-tuned model under two framings, all pre-specified health gates passed, yet 17/18 and 7/8 item pairs, respectively, were saturated (near-deterministic), and no item was certified residually contextual. Forced-binary next-token log-probabilities were thus inadequate for distribution-level QQ audits under the tested model and prompting conditions; we recommend pre-specified saturation diagnostics whenever next-token distributions are treated as survey-response distributions. 

---
# KyrgyzLLM-Bench: Benchmarking Kyrgyz Language Understanding 

**Authors**: Timur Turatali, Aida Turdubaeva, Rustem Izmailov, Anton M. Alekseev, Sergey I. Nikolenko  

**Link**: [PDF](https://arxiv.org/pdf/2607.17173)  

**Abstract**: Evaluating large language models (LLMs) across languages remains challenging, as most multilingual benchmarks rely on translated English datasets, often obscuring linguistic and cultural specificity in the target language. This issue is particularly pronounced for less-resourced languages such as Kyrgyz, where reliable natively authored evaluation data are scarce. Building on previously introduced Kyrgyz-language evaluation datasets, this work reports the first systematic and large-scale evaluation of LLMs in Kyrgyz using the KyrgyzLLM-Bench benchmark suite. KyrgyzLLM-Bench comprises two natively authored datasets$-$KyrgyzMMLU and KyrgyzRC$-$together with carefully translated and manually post-edited versions of WinoGrande, HellaSwag, BoolQ, and TruthfulQA. We evaluate 26 open- and closed-source LLMs under zero-shot and few-shot settings, analyzing model performance, cross-lingual transfer, and the impact of translation artifacts on evaluation reliability. Across families and tasks, model rankings transfer broadly from English to Kyrgyz on WinoGrande and BoolQ, and to a lesser extent on MMLU, while HellaSwag exhibits a substantial English-Kyrgyz performance gap consistent with translation-induced plausibility shifts. Few-shot prompting improves several open-source models on reading comprehension but behaves inconsistently for proprietary models on translated tasks. We publicly release all datasets, evaluation code, and per-model results, and integrate the Kyrgyz tasks into a widely used multilingual evaluation framework to support future research on Kyrgyz NLP. 

---
# Scope3Trace: Evidence-Based Identification and Extraction of Scope 3 GHG Emissions from Sustainability Reports 

**Authors**: Siyuan Zheng, Yifan Duan, Chao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2607.17122)  

**Abstract**: Scope 3 greenhouse gas (GHG) emissions account for the majority of corporate carbon footprints, yet remain difficult to analyze at scale due to sparse disclosures, heterogeneous report document formats, and limited evidence traceability. Existing approaches typically rely on large language models to extract emissions information from ESG reports, but often lack explicit evidence grounding or depend on costly manual annotation and verification to ensure extraction reliability. To address these challenges, we propose Scope3Trace, an evidence-grounded information extraction framework designed to extract interpretable and traceable Scope 3 emissions information from real-world ESG and sustainability reports. The framework integrates a document information extraction pipeline that performs PDF collection and OCR parsing, LLM-assisted page localization and table reconstruction, and hybrid rule-LLM extraction of organization- and building-level emissions disclosures with evidence-grounded verification. Building upon this framework, we further contribute a dual-level, evidence-grounded, multimodal dataset comprising organization-level Scope 3 disclosures extracted from heterogeneous sustainability reports. Scope3Trace enables reliable extraction and transparent integration of heterogeneous sustainability disclosures, achieving high accuracy in extracting Scope 1-3 totals and category-level disclosures from sustainability reports. 

---
# BLAD: A Historically Contextualized, Multilingual Dataset of Bangladeshi Legal Acts (1799 to 2025) 

**Authors**: Adib Sakhawat  

**Link**: [PDF](https://arxiv.org/pdf/2607.17111)  

**Abstract**: We present the Bangladesh Legal Acts Dataset (BLAD), a curated collection of 1{,}484 legislative acts enacted between 1799 and 2025. Each act is represented with its full text, structured sections and footnotes, repeal status, and metadata linking it to the governing regime, head of state, and prevailing legal framework at the time of enactment. The corpus spans English, Bengali, and mixed-language documents, supporting temporal and multilingual analysis of statutory law. BLAD addresses a persistent gap in legal natural language processing (NLP) resources for low-resource, civil-law jurisdictions in South Asia. We describe the acquisition and enrichment pipeline, report descriptive statistics over more than two centuries of legislation, and outline the research directions the corpus enables. The dataset is publicly available under the CC~BY-SA~4.0 license at this https URL. 

---
# Learning from Synthetic Data without Model Collapse in Iterative Instruction Tuning 

**Authors**: Xiaonan Luo, Yue Huang, Kehan Guo, Ping He, Chuan Zou, Ting Hua, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17043)  

**Abstract**: Model collapse is a central challenge in learning from synthetic data: as later-generation large language models (LLMs) are trained on an increasing proportion of model-generated data, performance can degrade due to narrowed coverage and accumulated bias. Existing work mainly studies how to bound this degradation. In iterative model evolution, however, the more meaningful objective is to ensure that each successive model improves over its predecessor, which requires diagnosing collapse at a granularity that is actionable for data curation. We study this problem in synthetic data self-improving for instruction tuning. We show that collapse in this setting is not simply uniform performance degradation, but can appear as a polarization of competence, where synthetic training reinforces already strong skills while further degrading weak ones. Motivated by this observation, we propose KITE (Knowledge-boundary Instruction Tuning via Exploration), a two-stage framework that combines failure-guided data generation with boundary-aware uncertainty curation. Experiments across several datasets and multiple open-source LLMs show that KITE yields more stable improvement than strong synthetic-data baselines. 

---
# Real-World Evaluation of an AI Agent Drafting Translational Impact Summaries 

**Authors**: Mohammad Arvan, Amber E. Osterholt, Bailee Rue, Yuvaneswaren Ramakrishnan Sureshbabu, Krishna Riteshkumar Patel, Rebecca T. Feinstein, Bethany C. Bray, Niranjan S. Karnik  

**Link**: [PDF](https://arxiv.org/pdf/2607.16989)  

**Abstract**: Introduction. Clinical and Translational Science Award (CTSA) programs must document their scholars' research impact, but assembling each scholar's record by hand takes staff an estimated 15 hours and does not scale to a full cohort. An artificial intelligence (AI) agent could serve as a tool to gather scholar data across platforms and disciplines.
Methods. We built a human-in-the-loop AI agent that assembles a dossier of sourced evidence for each scholar and drafts one-sentence Translational Science Benefits Model (TSBM) impact summaries for staff review. We evaluated it in the impact-reporting workflow of one CTSA hub across 10 career-development (KL2/K12) scholars. Two evaluation staff independently coded all 507 findings as accept, edit, or reject; the primary measure was the unanimous usable rate, defined as the share both accepted or edited.
Results. Both reviewers accepted or edited 81.7% of the agent's findings. Reviewers each spent a median of 14 minutes per scholar, replacing an estimated 15 hours of manual assembly. Inter-rater agreement was moderate (Cohen's kappa 0.43 on the usable-versus-reject decision). A profile discovery study found the agent's recall close to human search. The agent's impact evidence spanned all four TSBM domains, and about a third of the reviewed findings fell in non-scholarly categories that routine processes tend to miss. Reviewers rated synthesis accuracy 4.5 and usefulness 4.8 on a 5-point scale.
Conclusions. A human-in-the-loop AI agent can serve as the first-pass author of a scholar's impact record, shifting staff from collecting and writing to reviewing, and making cohort-scale impact reporting feasible. 

---
# Trace-Based On-Policy Distillation for Masked Diffusion Language Models 

**Authors**: Haolin Ren, Ziyang Huang, Chenhao Yuan, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.16872)  

**Abstract**: Diffusion large language models (dLLMs) are a promising alternative to autoregressive generation. However, reasoning-oriented post-training for dLLMs remains challenging. Supervised fine-tuning (SFT) for dLLMs requires dense but often off-policy masked states, while reinforcement learning (RL) relies on sparse rewards or value modeling. This paper proposes \textbf{trace-based on-policy distillation (TOPD)}, a teacher-supervised framework that transfers reasoning ability to a target dLLM without reward estimation. The key idea is to supervise a dLLM on its own denoising trajectory, focusing on the trace-aligned token decisions that form the final response. Specifically, TOPD samples on-policy diffusion trajectories from the target dLLM, obtains teacher token distributions from a teacher model on the corresponding partially denoised states, and updates the target dLLM with a token-level Reverse Kullback-Leibler (Reverse-KL) objective. This design preserves dense teacher supervision while aligning training with the model's own denoising states. On mathematical reasoning benchmarks, TOPD enables SDAR-4B-Chat to match the MATH500 accuracy of its RL-trained counterpart TraDo-4B-Instruct, with gains of +5.7 under static evaluation and +4.5 under dynamic evaluation. Compared with the RL-trained counterpart, TOPD achieves this with 4$\times$ fewer rollout rounds, corresponding to an estimated 96.0$\times$ to-accuracy model-compute speedup. 

---
# Group Entropy-Controlled Policy Optimization 

**Authors**: Guangran Cheng, Chengqi Lyu, Songyang Gao, Wenwei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.16850)  

**Abstract**: Entropy control has become an effective tool in reinforcement learning (RL) of large language models (LLMs), helping balance exploration-exploitation trade-off during alignment process. Such RL paradigm is often conducted on mixtures of heterogeneous tasks, which induce distinct entropy regimes under the same policy, making global or token-level entropy regulation insufficient to corresponding heterogeneous needs of exploration. This heterogeneity further makes GRPO-style normalized advantages induce an entropy-dependent bias, making advantage signals across prompt groups statistically non-comparable. To address this issue, we propose Group Entropy-Controlled Policy Optimization (GEPO), a lightweight extension to GRPO that uses group entropy, estimated from existing grouped samples to perform entropy-conditioned asymmetric advantage shaping. GEPO attenuates positive advantages in low-entropy groups to reduce over-exploitation, and negative advantages in high-entropy groups to preserve exploration, with adaptive thresholds derived from historical entropy statistics. Extensive experiments on two base models across thirteen benchmarks spanning mathematics, physics, science, code generation, and instruction following show that GEPO consistently outperforms GRPO and recent entropy-controlled methods, delivering balanced cross-task improvements while preserving task-specific exploration levels throughout training. 

---
# Schema-Constrained Document-Level Event Argument Extraction with Lightweight LLM Fine-Tuning 

**Authors**: Roberto Pietrantuono, Antonio Guerriero, Pouya Sattari  

**Link**: [PDF](https://arxiv.org/pdf/2607.16808)  

**Abstract**: Event Argument Extraction (EAE) converts documents into structured event records by identifying argument spans and assigning them schema-defined roles. Document-level EAE is challenging due to long-range dependencies between triggers and arguments, cross-sentence context, and strict role constraints, which often lead to boundary errors, uncertainty in roles, and inconsistencies with restricted schemas.
In this paper, we study whether mid-sized open LLMs can perform schema-constrained EAE reliably at the document level on MAVEN-ARG. Our approach combines (i) role-set injection in prompts for schema compliance, (ii) parameter-efficient supervised fine-tuning (LoRA) using the same JSON-only interface used at inference, and (iii) deterministic decoding with post-processing that validates JSON, filters invalid roles, de-duplicates arguments, and aligns spans to the document window. Under the official MAVEN-ARG evaluator, fine-tuned mid-sized open models outperform previously reported GPT baselines across mention, entity-coreference, and event-coreference evaluations; our best model (Phi-4, 14B) reaches 42.39\% F1 at the event-coreference level. Code to reproduce experiments is publicly available at this https URL. 

---
# Diagnosing Correctness Probes under Self-Judgement Confounding 

**Authors**: Yi-Long Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.16799)  

**Abstract**: Hidden-state readouts can predict whether language-model outputs are correct, but objective correctness (OC) usually agrees with the model's own self-judgement (SJ), leaving the decoded signal semantically ambiguous. We construct conflict cases in which OC and SJ predict opposite readout orderings. On high-confidence disagreements, conventional correctness-labelled contrasts often rank incorrect/self-endorsed responses above correct/self-rejected responses, following SJ rather than OC. We estimate factorial SJ- and OC-associated directions and evaluate their polarity across mathematical reasoning and factual recall. Across four instruction-tuned models up to 14B parameters, the SJ-associated direction transfers above chance in both cross-domain directions for every model, whereas the OC-associated direction has a below-chance point estimate for the expected OC ordering in every corresponding condition. This transfer asymmetry develops across middle-to-late layers, persists under answer-likelihood, sequence-length, and null-direction controls, and extends to MMLU and binary TruthfulQA without target-domain direction fitting. Across the studied models and diagnostic subsets, the most reliably transferable component preserves SJ-associated polarity. Transferability alone therefore does not establish objective-correctness semantics. 

---
# Cascading versus Joint Modeling for Hierarchical Offensive Language Detection 

**Authors**: Ruixing Ren, Junhui Zhao, Xiaoke Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.16790)  

**Abstract**: Fine-grained offensive language detection organizes labels into a hierarchical structure, for which two modeling paradigms exist: cascaded decomposition and joint multi-task modeling. Prior work rarely provides a direct, controlled comparison of the two paradigms in terms of accuracy, parameter count, and inference latency, and rarely verifies whether a chosen class-imbalance handling strategy is actually optimal. This paper proposes a three-level cascaded detection system whose training strategy is customized per subtask, together with two verification mechanisms. First, a controlled ablation study determines the best class-imbalance handling strategy for each subtask. Second, a joint multi-task model with a shared encoder is trained as an architectural control, yielding real measurements along the dimensions of accuracy, parameter count, and inference latency. Experiments show that the cascaded system attains macro-F1 scores of 0.795, 0.716, and 0.557 on the three subtasks of the official test set. The ablation study reveals that configuring the loss function purely by imbalance-severity intuition is suboptimal; reconfiguring based on the ablation results improves both performance and stability. End-to-end cascade evaluation shows that roughly one-fifth of the errors in the cascade pipeline originate from the first-stage filter and cannot be corrected by subsequent stages. Relative to the joint multi-task model, the cascaded architecture achieves higher accuracy on all three subtasks, with a 7.1-point macro-F1 gain on the most severely imbalanced subtask, at the cost of three times the parameters and 1.67 times the inference latency. Together, these results establish an explicit, quantifiable trade-off between the accuracy advantage of cascaded architectures and their deployment cost. 

---
# JOR-Bench: Japanese Operations Research Benchmarks for Large Language Models 

**Authors**: Yuu Jinnai  

**Link**: [PDF](https://arxiv.org/pdf/2607.16777)  

**Abstract**: We present JOR-Bench, a collection of five Japanese-language benchmarks for evaluating the ability of large language models (LLMs) to formulate and solve operations research (OR) problems. Each benchmark is a Japanese translation of an existing English benchmark: IndustryOR, MAMO Complex LP, NL4OPT, OptiBench, and OptMATH, covering 1,319 problems spanning linear programming, mixed-integer programming, non-linear programming, and combinatorial optimization. JOR-Bench is a solver-independent benchmark that can be used with any solver or programming language, and consists of pairs of Japanese problem statements and expected numerical answers. We evaluate seven LLMs, including multilingual general-purpose models and Japanese-specialized models, on both the original English and the new Japanese versions, and compare performance across languages. For the main evaluation, we standardize execution with the Python interface to OR-Tools to make model outputs comparable and reproducible with open-source software. Our results show that OR formulation ability is largely language-neutral for strong multilingual models; the overall average accuracy difference between English and Japanese is only $-0.3$ pp. Yet error analysis reveals subtle cross-lingual differences, including a pragmatic disambiguation failure in some domains that causes models to output decision-variable values instead of the objective value when the prompt is in Japanese. 

---
# Though Language Models Err While They Strive: Conformal Prediction for Self-Correcting Scientific Generation 

**Authors**: Mingqiao Mo, Yunlong Tan, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.16704)  

**Abstract**: Large language models frequently violate fundamental scientific principles when generating technical content, undermining their reliability in scientific applications. We introduce Scientific Feasibility Control SFC, a graph-structured conformal prediction framework that provides statistical guarantees for scientific reasoning validity through progressive absolute-coherent-factuality validation. Our approach decomposes scientific reasoning into atomic absolute-coherent-factuality units requiring both individual correctness against physical laws and logical substantiation from preceding context, addressing the cascade effect where early scientific errors contaminate subsequent reasoning steps. Unlike independence-based methods that treat claims in isolation, SFC models logical dependencies as approximate deducibility graphs and operates through real-time validation with dynamic branching when scientific violations are detected, the system branches to alternative generation paths using verified context as foundation. We demonstrate SFC across established scientific reasoning benchmarks including PhyX multimodal physics, MATH, ScienceQA, and ARC Challenge, achieving 50.1 percent accuracy on PhyX physics reasoning, substantially outperforming recent reasoning models including DeepSeek-R1 49.8 percent and GPT-4 45.8 percent while providing 91.7 percent scientific validity with formal conformal coverage guarantees at alpha equals 0.10 confidence level and reducing scientific law violations by 73 percent across multiple model architectures. 

---
# Are Arithmetic Heuristic Neurons Form-Invariant? A Mechanistic Analysis of Symbols, Text, and Code in LLMs 

**Authors**: Sharath Naganna, Tanvir Ahmed Sijan, Uddipta Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2607.16693)  

**Abstract**: Large language models often succeed on one formulation of a problem while failing on an equivalent formulation. Whether these failures arise from distinct internal circuits or different activation states of a shared circuit remains unknown. Recent mechanistic interpretability studies suggest that arithmetic in LLMs emerges from a "bag of heuristics," encoded by a sparse set of MLP neurons that represent distinct arithmetic strategies. We investigate whether arithmetic heuristic neurons are form-invariant across symbolic arithmetic, natural language word problems, and Python code in three Llama-3 models. In each format, we identify arithmetic heuristic neurons using a two-stage pipeline combining attribution patching and activation patching. A compact set of neurons is shared across all three formats, and targeted interventions show this shared circuit is both necessary and sufficient for late-layer arithmetic computation. Transferring the shared neurons' activations from a successful execution in one format to a failed execution in another recovers most incorrect predictions, exceeding 97% for addition and subtraction, indicating that cross-format failures arise from activation states rather than distinct circuits. Moreover, shared neurons consistently belong to the same heuristic families across formats, demonstrating that arithmetic computation in LLMs is largely form-invariant at the neuron level. 

---
# SpecLA: Efficient Speculative Decoding for Linear-Attention Models 

**Authors**: Zhibin Wang, Xuying Han, Zhaohua Yang, Fuliang Liu, Xue Li, Rong Gu, Sheng Zhong, Chen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2607.16673)  

**Abstract**: Linear-attention models replace the growing KV cache with recurrent states, but autoregressive decoding still reads, updates, and writes these states one token at a time. Speculative decoding can reduce this cost by verifying several draft tokens in one target pass, yet existing speculative systems are designed for Transformer KV caches. For stateful linear-attention targets, verification must follow recurrent dependencies across chains and branches, acceptance must update only the accepted state trajectory, and the drafter must avoid submitting candidates that waste stateful verification work. This paper presents SpecLA, a speculative decoding runtime for stateful linear-attention models. SpecLA verifies chains and trees with topology-aware kernels, stores compact factors produced during verification to recover accepted states, and uses confidence pruning plus a target-aligned EAGLE-style drafter to feed useful candidates to the verifier. On an NVIDIA H100 with a public GDN-1.3B target, SpecLA achieves up to 1.70x end-to-end speedup over autoregressive decoding. 

---
# OpenLanguageModel: Readable and Composable Small-Language-Model Pretraining for Education and Research 

**Authors**: Tavish Mankash, Vardhaman Kalloli, Keshava Prasad, Deepan Muthirayan  

**Link**: [PDF](https://arxiv.org/pdf/2607.16669)  

**Abstract**: OpenLanguageModel (OLM) is an open-source PyTorch library for building and pretraining small language models while keeping their machinery visible. In OLM, model code reads like the architecture: components are ordinary modules, while Block, Residual, Repeat, and Parallel describe how they are wired. The resulting model can move unchanged from a teaching notebook to a complete pretraining run or a research ablation. OLM connects this readable model layer to tokenizers, local and streaming datasets, optimization, mixed precision, callbacks, checkpoints, and hardware-aware CPU, single-GPU, and single-node multi-GPU execution. We demonstrate the full path by tracing GPT-2 from diagram to code, launching a FineWeb-Edu training script, replacing one attention component, and letting AutoTrainer configure the available machine. The package includes 27 presets across nine familiar model families and documentation that progresses from LM fundamentals to architecture research. Validation shows close agreement with independent reference implementations, 90.6% four-GPU weak-scaling efficiency for a 348M-parameter workload, compact architecture edits, and positive early usability results. OLM is MIT-licensed and available through PyPI, GitHub, and its documentation site. 

---
# From Memory to Skills: Evidence-Grounded Co-Evolution Governance for Long-Horizon LLM Agents 

**Authors**: Bo Tang, Yang Zhang, Guomian Zhuang, Wenqiang Wei, Gaoyang Zheng, Lindong Xie, Yanchao Tan, Feiyu Xiong, Qingyu Yang, Edward Chung, Zhiyu li  

**Link**: [PDF](https://arxiv.org/pdf/2607.16621)  

**Abstract**: Existing memory systems for long-horizon LLM agents often retrieve prior traces as passive context rather than converting them into executable capabilities. In this paper, we propose MSCE, a training-free Memory--Skill Co-Evolution framework that organizes agent experience into grounded step traces, reusable procedural policies, and declarative environmental cognition. MSCE crystallizes evidence-backed L2 policies with positive estimated gain into callable skills that retain evidence links, applicability boundaries, decision guidance, verification rules, and reliability estimates. It further introduces reflection-weighted value backfilling, which propagates sparse terminal feedback through dense local self-reflections to produce evidence-calibrated trace values for governing memory and skill evolution. Experiments on EvoAgentBench and LoCoMo demonstrate that MSCE significantly outperforms state-of-the-art skill-augmented and memory-driven agent baselines, exhibiting strong cross-domain transferability and lifelong-evolution capabilities. 

---
# NOWJ@COLIEE 2026: Adaptive Pipelines for Legal Retrieval and Reasoning 

**Authors**: Thuong-Hieu Ngo, Hoang-Trung Nguyen, Huu-Dong Nguyen, Xuan-Bach Le, Le-Dung Nguyen, Quang-Thanh Tran, Ha-Thanh Nguyen, Thi-Hai-Yen Vuong  

**Link**: [PDF](https://arxiv.org/pdf/2607.16603)  

**Abstract**: This paper presents the methodologies and results of the NOWJ team's participation across all five tasks of the COLIEE 2026 competition. For Task 1 (Legal Case Retrieval), we propose a four-stage pipeline comprising candidate filtering, dense retrieval with complementary embedding models, cross-encoder reranking via fine-tuned generative rerankers and MLP-based pairwise classification, and adaptive per-query cutoff prediction. For Task 2 (Legal Case Entailment), we combine BM25 filtering, T5-based reranking, and LLM-based entailment verification with consensus ensemble. For Task 3 (Statute Law Retrieval and Entailment), we adopt a retrieval-augmented generation framework with dense retrieval, attention-based reranking, and few-shot-prompted LLM reasoning. For Task 4 (Legal Textual Entailment), we introduce a dynamic routing pipeline that classifies query difficulty and dispatches cases to either a balanced few-shot solver or a structured zero-shot chain-of-thought solver. For the Pilot Task (Legal Judgment Prediction), we combine hierarchical transformers with CRF layers, argument relation mining, and probabilistic argumentation graph reasoning. 

---
# Encoding EEG Signals to Examine Human-Like Next-Word Prediction Behaviour in Language Models 

**Authors**: Boi Mai Quach, Binh T. Nguyen, Cathal Gurrin, Graham Healy  

**Link**: [PDF](https://arxiv.org/pdf/2607.16549)  

**Abstract**: Language models (LMs) are trained to excel at predicting the next word in the sequence given prior context, and humans also share this predictability in reading comprehension. Neuroscience research reveals that next-word predictability influences brain response, as recorded at millisecond resolution using electroencephalography (EEG). While our evidence indicates that advanced LMs achieve accuracies closely aligned with human performance at the next-word prediction task, this raises the question: Does higher prediction accuracy necessarily mean that these models adequately capture the cognitive signals associated with human reading comprehension? Here, we generate regressors for both humans and LMs based on two information measures, including top-1 prediction and surprisal, to predict event-related potential (ERP) elicited from EEG recordings which reflect different stages of cognitive processing during reading. We argue that modelling ERP patterns offers fine-grained analysis of the cognitive plausibility of various LMs during reading. Our results indicate that only surprisal potentially correlates with language-processing ERPs, especially for open-class words with high semantic content. Moreover, our findings challenge the assumption that scaling LMs with increased parameters and computational budgets will consistently lead to improved convergence with human-like linguistic processing. 

---
# Committed Before Reasoning: Behavioral Reproduction and Preliminary Activation-Level Evidence of Answer Pre-Commitment in an Open-Weight LLM 

**Authors**: Heejin Jo  

**Link**: [PDF](https://arxiv.org/pdf/2607.16451)  

**Abstract**: Chat models sometimes commit to an answer and then produce reasoning that justifies it rather than deriving it -- even when the answer contradicts a task premise. We study a minimal probe: "I want to wash my car. The car wash is 100 meters away. Should I walk or drive?" Only drive works (the car must be at the car wash), yet models overwhelmingly recommend walking. (1) Behavioral reproduction: on Qwen3-8B across five system-prompt conditions (210 rollouts), the wrong commitment occurs in 85-100% of sampled rollouts per condition and 100% of greedy rollouts, in both thinking and non-thinking modes; a 4,096-token thinking budget does not repair it. (2) Preliminary activation-level evidence: probing hidden states with a pretrained, training-free activation oracle (no task-specific probe training) at positions before the answer text is emitted, "walk" read-outs exceed a neutral-context baseline (68% vs. 17%; walk-committing rollouts p=.005, drive-committing rollouts p=.005, Fisher exact) -- notably, rollouts that eventually answer drive also read as walk-leaning before commitment (5/6). The oracle's default on unrelated content is "drive" (83%), so the read-outs are not lexical bias; stratifying by literal walk/drive occurrence shows they are not text recovery either (spans containing "drive" still read out walk; in balanced lexical fields, per-rollout walk-majorities beat a per-prompt neutral baseline 15/22 vs. 1/8, p=.01; drive-committing rollouts 6/6, p=.002). Samples are small and the within-rollout positional gradient is not significant (p=.34); we frame these results as preliminary. (3) Methodological: with fixed oracle, activations, and positions, question wording alone moves a positive control from 2/16 (open question) to 11/16 (closed); negative oracle results are uninterpretable without per-wording positive controls. 

---
# RIMS: Preference Optimization via Smoothed Multi-pair Aggregation for Small-Scale LLM Retrieval-Augmented Generation 

**Authors**: Pei Tian, Zihan Dong, Tianci Liu, Linjun Zhang, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.16431)  

**Abstract**: Small-scale language models (SLMs) are attractive for retrieval-augmented generation (RAG) in resource-constrained settings, but their limited capacity makes them highly sensitive to noisy or spurious retrieved evidence. Existing preference-based methods such as RoseRAG select only the hardest single preference pair via hard argmin/argmax, discarding the remaining signal; others treat multiple pairs as independent binary comparisons, resulting in low data utilization. We propose RIMS, a three-stage preference optimization framework comprising (1) synthetic chain-of-thought preference data generation via rejection sampling using the target SLM itself without relying on proprietary models, (2) a differentiable soft aggregation mechanism that replaces hard selection with a smooth operator, preserving gradient signal from all preference pairs while retaining the discriminative structure of margin-aware selection, and (3) preference optimization with the smoothed objective applied to multiple alignment algorithms. We theoretically show that the smoothed approximation admits a controllable error bound and that smooth aggregation yields provably tighter gradient alignment to the oracle objective than hard selection. Experiments on four multi-hop question answering benchmarks show that our approach outperforms state-of-the-art baselines across multiple SLM backbones, achieving consistent gains in Exact Match and F1 under noisy retrieval conditions. Our implementation is available at this https URL. 

---
# Multi-level context Modeling for consistent expert selection in Mixture-of-Experts 

**Authors**: Shuhan Huang, Naifan Zhang, Yuanbo Tang, Yang Li, Wai Kin Victor Chan  

**Link**: [PDF](https://arxiv.org/pdf/2607.16427)  

**Abstract**: Mixture-of-Experts (MoE) enables efficient scaling of Transformer models by routing tokens to a small subset of experts. However, existing routers typically condition expert selection on shallow or isolated token representations, which often produce unstable and semantically inconsistent routing decisions across layers. In this work, we revisit expert selection from a representation perspective and identify context incompleteness as a key bottleneck limiting effective expert specialization. To address this issue, we propose Multi-level Context Fusion MOE (MCF-MOE), a framework that constructs context-aware representations by integrating complementary signals from cross-layer semantic aggregation and local token-level interactions, enabling more informative and consistent expert selection. Experiments on language modeling and understanding benchmarks demonstrate that MCF-MOE consistently improves routing consistency and downstream performance over strong MoE baselines, highlighting the importance of contextual completeness in expert routing. The code is available at this https URL. 

---
# Logical Judgments Under Pressure: Diagnosing Syllogistic Stability with Learned Soft Prefixes 

**Authors**: Brian K Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.18228)  

**Abstract**: To test how correct logical judgments respond to learned context, we prepend a soft prefix to an exactly labeled syllogistic reasoning benchmark while keeping the model fixed. Soft prefixes are opaque continuous vectors, so we characterize them through the behavior they induce across controlled variations in logical form and interface. By studying which prefixes succeed and how their effects generalize, we characterize how learned contextual pressure can override correct judgments and expose limits in a model's logical stability. Across Qwen3.6-35B-A3B MoE, Qwen3-8B, and Gemma 4 31B, learned prefixes redirect many correct answers and remain effective across unseen forms and interface changes. In repeated tests with Qwen3.6 MoE and Gemma, they outperform paired random controls in all 16 model--direction--split comparisons by 37 to 99 percentage points. Qwen3.6 MoE flip rates remain between 72% and 90% across wording and prompt changes, while Gemma validity prefixes retain 54% to 56% flip compared with less than 1% for matched random prefixes. Diagnostic tests show that the dominant effect is a broad preference for one answer meaning rather than fixed-symbol forcing or a logical operation that transfers reliably between tasks. The form of this bias differs across models. In both Qwen models, simple score models often predict which judgments will flip but not how far their margins will move, whereas Gemma's overall response is more closely approximated by the same models. These results show that the dominant behavioral effect of successful soft prefixes is a broad answer preference, while the remaining response reveals substantial model-specific differences in logical stability. 

---
# Do Language Models Dream of Binding Molecules? Benchmarking LLMs under Spatial Constraints 

**Authors**: Thomas MacDougall, Maksim Kuznetsov, Roman Schutski, Rim Shayakhmetov, Maxim Malkov, Vladimir Aladinskiy, Alex Aliper, Alex Zhavoronkov  

**Link**: [PDF](https://arxiv.org/pdf/2607.18144)  

**Abstract**: Structure-based drug design (SBDD) leverages the 3D structure of protein targets, often complemented by other spatial constraints, to generate candidate binding molecules. While diffusion models have dominated as a leading paradigm for high-quality 3D molecule generation, LLM-based methods are rapidly emerging in molecular design and have shown competitive performance in pocket-conditioned molecular generation. However, their ability to reason about physics and 3D spatial environments is largely underexplored. In this work, we systematically analyze whether current general-purpose LLMs are capable of navigating complex 3D constraints compared to established baselines such as specialized diffusion models. We consider 3D ligand generation conditioned on protein pockets together with ligand- and interaction-derived spatial constraints, including anchor fragments, pharmacophore points, and mandatory pocket-ligand interactions. To enable this evaluation, we introduce 3D-Fit - a token-efficient benchmarking strategy for assessing LLM performance on multi-conditioned spatial molecule generation. Our findings reveal a clear pattern in LLM spatial capabilities: while they still lag behind state-of-the-art approaches, they are promising and can handle multiple spatial constraints simultaneously, enabling scaling to heterogeneous setups. 

---
# O-VAD: Industrial Video Anomaly Detection through Object-Centric Tracking and Reasoning 

**Authors**: Mei Yuan, Qi Long, Qifeng Wu, Zhenyang Li, Yizhou Zhao, Lei Wang, Yang Liu, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18142)  

**Abstract**: Industrial Video Anomaly Detection (IVAD) aims to identify anomalous objects and events in an industrial process, which is crucial for modern manufacturing and quality control systems. Existing VLM-based anomaly reasoning methods are capable of detecting open-ended anomalies in general domains. However, their performance declines in industrial settings characterized by intricate object transformations, strict physics, and procedural constraints. To tackle the complexity of such interaction-intensive detection, we introduce a training-free agentic framework for anomaly detection free of domain-specific knowledge, emphasizing object state evolution like humans inspectors. It is designed to track spatial-temporal dynamics and underlying transformations of detected objects over time, and then reason over the object-wise temporal state trajectories to identify abnormal objects in grounded frames. Our method overcomes limitations of prior approaches that rely on retraining on normal clips or injecting domain knowledge as context for test-time inference. Extensive experiments on three IVAD datasets demonstrate that our method outperforms frontier VLMs, agentic frameworks, and traditional VAD methods fine-tuned on the respective datasets, while providing interpretable reports over anomaly processes and types. 

---
# LLM-as-a-Coach: Experiential Learning for Non-Verifiable Tasks 

**Authors**: Tianzhu Ye, Li Dong, Guanheng Chen, He Zhu, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.18110)  

**Abstract**: Reinforcement learning (RL) on open-ended tasks compresses an LLM's rubric-based evaluation into a scalar reward, discarding rich textual feedback and conflating responses with distinct quality profiles. We propose Experiential Learning (EL), which repurposes the feedback model from an LLM-as-a-Judge into an LLM-as-a-Coach. The coach distills its assessment of each on-policy response into transferable experiential knowledge, which conditions a teacher model and is internalized by the policy through on-policy context distillation. Compared with scalar rewards, this higher-bandwidth feedback channel provides dense supervision and preserves fine-grained preferences among high-quality responses. Across two policy families, with feedback from the policy itself or a proprietary model, EL consistently outperforms rubric-based RL on held-out and unseen open-ended tasks. Notably, EL generalizes better beyond the training distribution, and mitigates reward hacking. These findings establish experiential knowledge as a richer and more generalizable learning signal for post-training on non-verifiable tasks. 

---
# FinSAgent: Corpus-Aligned Multi-Agent RAG Framework for Evidence-Grounded SEC Filing Question Answering 

**Authors**: Jijun Chi, Zhenghan Tai, Hanwei Wu, Tung Sum Thomas Kwok, Hailin He, Zixing Liao, Bohuai Xiao, Chaolong Jiang, Jianliang Lei, Jerry Huang, Peng Lu, Muzhi Li, Liheng Ma, Yihong Wu, Sicheng Lyu, Jingrui Tian, Yihan Li, Yanzhang Ma, Dingtao Hu, Yufei Cui, Ling Zhou, Lei Ding, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18102)  

**Abstract**: Financial question answering over U.S. Securities and Exchange Commission (SEC) filings requires retrieving and synthesizing heterogeneous evidence dispersed across long, standardized, and highly redundant disclosures. Existing retrieval-augmented and multi-agent systems typically derive retrieval queries directly from the user's question and rank candidates by semantic similarity. Together, these choices create prior-corpus misalignment: a mismatch between model priors and the target filings' structure, terminology, and evidence standards. As a result, query generation misses corpus-specific evidence, while semantic reranking favors topically similar but evidentially invalid false-positive chunks. We propose FinSAgent, an evidence-grounded multi-agent framework that reframes SEC filing QA as corpus-aligned retrieval planning and corrects both ends with a single principle: inject corpus-side conditioning wherever model priors would otherwise dominate. FinSAgent combines (1) role-specialized agents anchored to the mandated 10-K item structure, (2) database-aware query decomposition that conditions each agent's sub-queries on a lightweight, summary-level view of the local corpus, and (3) multi-path retrieval with a learned feature-gated reranker that separates evidential validity from semantic similarity. Across five offline financial QA benchmarks, FinSAgent improves retrieval coverage and answer correctness over strong single-agent and multi-agent baselines; in a three-arm randomized online experiment with 1,000 anonymous user ratings, it also receives higher scores than baselines. 

---
# WorldCupArena: Fine-Grained Evaluation of Language Models and Deep-Research Agents on Football Forecasting 

**Authors**: Zhaokai Wang, Tianlin Gui, Jiayuan Rao, Shangzhe Di, Yihong Tang, Dingli Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18084)  

**Abstract**: Predicting a football match before kickoff requires more than knowing past results: a model must use changing information and make a clear prediction before the answer is available. We present WorldCupArena, a dynamic benchmark for language models and deep-research agents. The 2026 FIFA World Cup is its first evaluation, and the same process can be reused for future leagues and cups. Before each match, a model either receives a common evidence package or searches for information itself. It predicts the result and score, likely players and events, match statistics, and the outcome of the competition. After the match, these predictions are compared with the recorded result. We report result accuracy, exact-score accuracy, and a scoreline score that gives some credit when a predicted score is close but not exact, together with scores for the other prediction tasks. Across 104 matches and 13 systems, models with similar result accuracy differ more clearly on detailed predictions. Compared with betting-market and human-fan baselines, the best system shows only small gains in result and exact-score accuracy, but a clearer gain in Scoreline. New schedules can be added as they begin, allowing the benchmark to evaluate future models without using outcomes that are already known. Code, prompts, predictions, and evaluation scripts are open sourced at this https URL. 

---
# L1 Augmented Attention as an Improved Vector Similarity Metric 

**Authors**: Kurt Godden  

**Link**: [PDF](https://arxiv.org/pdf/2607.18027)  

**Abstract**: Scaled dot product attention conflates directional alignment and vector magnitude, limiting its effectiveness as a similarity metric in Transformer models. We introduce L1 augmented attention, a simple and computationally parallelizable modification that subtracts a learned, head specific L1 distance between queries and keys from the dot product score. This hybrid similarity captures complementary geometric information. Dot product rewards directional alignment, while L1 penalizes coordinate deviations. To reduce the cost of L1 computation, we project queries and keys into low dimensional subspaces whose parameters specialize to preserve informative L1 structure. Evaluated on WikiText 2 using a compact transformer, L1 augmented attention achieves up to a 14.5% reduction in perplexity over the original transformer baseline and outperforms an RBF L2 kernel. Analysis of norm variance and learned L1 weights reveals distinct geometric roles across layers and strong head level specialization. These results demonstrate that enriching attention with L1 geometry provides a principled and effective improvement to similarity computation in modern language models, with practical benefits for both accuracy and parallel efficiency. 

---
# MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning for Parameter-Efficient Reasoning in Compact Models 

**Authors**: Martino M. L. Pulici, Cuong Xuan Chu, Evgeny Kharlamov, Zifeng Ding, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.18006)  

**Abstract**: Large language models achieve strong reasoning performance, but often at prohibitive training cost - a challenge that is especially acute for compact models ($\leq 4 \, \mathrm{B}$ parameters) trained under limited budgets. We introduce MADA-RL, a post-training framework that specializes compact models into generator and critic roles and trains them with a debate-aware learning signal, fine-tuning only a small subset of parameters via LoRA adapters. Our central contribution is a counterfactual critic advantage: a dynamic, role-conditioned baseline that redefines the critic's advantage as its reward minus the generator ensemble's per-instance accuracy. This explicitly optimizes critics to improve over generator consensus rather than to merely reproduce a correct answer, yielding more targeted credit assignment than static mean-reward normalization. At deployment, the specialized agents are composed in a lightweight multi-round protocol. Across five mathematical reasoning benchmarks, MADA-RL raises the accuracy of the DeepSeek-R1-Distill-Qwen-1.5B model from $39.9 \, \%$ to $41.9 \, \%$ ($+2.0$ points, $p < 0.001$) using $16$ times fewer trainable parameters than fully fine-tuned baselines, placing it on the accuracy-trainable-parameter Pareto front. It approaches, but does not surpass, the strongest baselines (DeepScaleR, STILL-3), which are trained on substantially larger datasets; we analyse this gap and the associated inference-time cost directly. A controlled study isolates the source of MADA-RL's gains: the counterfactual advantage produces the highest critic improvement rate of any model evaluated, indicating that trained critics learn to correct generator errors rather than to imitate them. 

---
# Self-State Attacks on Self-Hosted AI Agents: How Far Can OS Defenses Go? 

**Authors**: Yimeng Chen, Nathanaël Denis, Roberto Di Pietro, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2607.17986)  

**Abstract**: Self-hosted AI agents read and write their own memory and configuration files to function. An agent may get compromised via corruption of its own state -- a compromise realized via legitimate OS system call invocation. We refer to this class of threats as self-state attacks. In this paper, we investigate the OS resilience to this class of attacks. Formally, we characterize a four-axis attack space (Target, Mechanism, Granularity, Temporal); investigate the structural limits of prevention, detection, and recovery; and introduce a workload-conditioned view of detectability. To instantiate the framework, we collect live activity traces from a representative self-hosted agent running across distinct workload profiles, and realize the attack space as a 23-cell matrix, 43 concrete operations on real self-state files, and injected into those traces. We then evaluate both canonical and workload-conditioned defense strategies. The empirical results show that a layered defense stack (access-control prevention on the instruction and configuration layers, workload-conditioned detection on the memory layer, and periodic backup for recovery) is effective on most attack cells while a small residual attack surface remains structurally indistinguishable at the OS level. These findings suggest that against the newly established class of self-state attacks, OS-level defense needs to be reconsidered, potentially opening new research directions in the field. 

---
# Benchmarking Resource-Efficient LLMs for Research Topic Ontology Generation in the Biomedical Field 

**Authors**: Tanay Aggarwal, Angelo Salatino, Francesco Osborne, Enrico Motta  

**Link**: [PDF](https://arxiv.org/pdf/2607.17902)  

**Abstract**: Knowledge Organization Systems like Ontologies and taxonomies are fundamental for structuring scientific knowledge, yet their manual curation presents a persistent bottleneck in knowledge management. While Large Language Models (LLMs) offer a scalable mechanism for automated ontology generation, their capacity to classify complex, domain-specific semantics requires systematic evaluation. In this paper, we assess the performance of five small, open-source LLMs (up to 9 billion parameters) in identifying semantic relationships between biomedical concepts. To support this evaluation, we introduce MeSH-Rel-4K, a dataset comprising 4K semantic relationships extracted from the Medical Subject Headings (MeSH). We analyse three adaptation strategies: standard prompting, Chain-of-Thought prompting, and fine-tuning. While parameter-constrained models traditionally struggle with the nuances of in-context logic, our results reveal that targeted fine-tuning increases the average F1-score by 34.1 percentage points. These results confirm that direct fine-tuning effectively exceeds the reasoning bottlenecks of smaller LLMs, providing an accurate, automated methodology for the construction and evolution of specialised biomedical ontologies. 

---
# AlphaOracle: Oracle bone script decipherment via human-workflow-inspired deep learning 

**Authors**: Yuliang Liu, Haisu Guan, Pengjie Wang, Xinyu Wang, Jinpeng Wan, Kaile Zhang, Handong Zheng, Xingchen Liu, Zhebin Kuang, Huanxin Yang, Bang Li, Yongge Liu, Lianwen Jin, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2607.17849)  

**Abstract**: Approximately 3,000 of the 4,500 oracle bone script (OBS) characters remain undeciphered due to fragmentary inscriptions and sparse evidence. Current AI approaches fail to replicate expert workflows that integrate form analysis, contextual semantics, and philological reasoning. We introduce AlphaOracle, a human-workflow-inspired framework that systematizes OBS decipherment using the largest digitized corpus to date. Its multi-stage pipeline comprises: (i) rubbing parsing; (ii) radical-based morphological analysis with diachronic modeling; (iii) contextual retrieval with semantic alignment; and (iv) philological validation against classical sources. Each stage generates explicit, confidence-weighted evidence chains, culminating in interpretable reports for scholarly verification. Across multiple test characters, AlphaOracle's readings strongly agreed with expert interpretations. In a study of 86 domain specialists, it reduced analysis time by 64% and 79% of participants rated it highly useful. Notably, AlphaOracle resolves the character "Lao" as a toponymic or clan designation, offering concrete revisions to Shang administrative and social interpretations. These results suggest that computational methods aligned with philological practice can facilitate OBS research and provide a conceptual reference for studies of other undeciphered scripts. 

---
# Mobius Learning: Cyclic Depth Folding in Transformers 

**Authors**: Tongtian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17843)  

**Abstract**: Transformer-based language models organize computation along an ordered depth axis, where shallow and deep blocks often develop distinct representational roles. We challenge the conventional view that these roles must remain tied to a block's position in the ordered sequence. We introduce Mobius Learning, a training architecture based on cyclic depth folding, in which different data streams follow cyclically shifted block orders. The same block group is therefore applied early in the block sequence for some data streams and late for others, so it is optimized in both shallow and deep roles, a phenomenon we call depth-role superposition. Surprisingly, in four-worker experiments with a modded GPT-2 small (124M) model trained on 2.5B FineWeb tokens using Muon, Mobius Learning achieves lower validation loss than a fixed-order looped Transformer at larger numbers of Transformer block-sequence passes. This counterintuitive result shows that a block group need not remain confined to one fixed shallow or deep role within the block sequence and opens a new design space based on cyclic depth folding. Crucially, this structure makes Mobius Learning particularly well suited to memory-constrained distributed training: raw training data remain local, while each worker stores one block group rather than the complete Transformer block stack. 

---
# It Matters How You Say It: Exploring Rhetorical Patterns for AI-Assisted Information Evaluation 

**Authors**: Sadra Sabouri, Zeinabsadat Saghi, Jordan Lee Boyd-Graber, Jonathan May, Jonathan K. Kummerfeld, Souti Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2607.17627)  

**Abstract**: Prior work on AI-assisted information evaluation has largely focused on what AI systems communicate, comparing explanation types and formats, with responses predominantly cast in directive rhetoric where the system delivers a verdict and the user passively accepts it. While debate-style interactions have recently shown promise in prompting critical evaluation over deference, the rhetorical patterns that structure AI responses and how they might induce reflection, uncertainty, or independent reasoning remain largely unexamined. To address this, we investigated eight rhetorical patterns known to induce contemplation: Intentional Misleading, Interpretive Alternative, Scaffold Explanation, Triggering Distrust, Information Distortion, Alternative Framing, Socratic Questioning, and an Oracle baseline. Through a within-subject study with n=98 participants on a hint-on-demand fact verification task, we observed preliminary evidence that Scaffold Explanation were associated with the highest accuracy gains, and encouraging deeper reflection. Surprisingly, the adversarial conditions also improved accuracy modestly. Participants preferred Alternative Framing most and Interpretive Alternative least, largely due to the latter's perceived time cost. We discuss the implications of designing conversational agents with varied rhetorical styles and the trade-offs among user performance, satisfaction, and contemplation. 

---
# PoLoRA: A Preconditioned Orthogonalized LoRA Optimizer 

**Authors**: Nikhil Ghosh, Tetiana Parshakova, Robert M. Gower  

**Link**: [PDF](https://arxiv.org/pdf/2607.17620)  

**Abstract**: Low-rank adaptation (LoRA) makes finetuning large language models cheaper by adding to each weight matrix a trainable low-rank update parameterized as the product of two matrices. These matrices are usually trained with Adam, which treats them as a single flat vector of parameters and ignores both the matrix and product structure of LoRA. Applying a matrix-aware optimizer such as Muon to each factor does not consistently improve over Adam, and neither do the product-aware Muon variants proposed in concurrent works. To realize consistent gains, we introduce PoLoRA, a Preconditioned Orthogonalized LoRA optimizer built from three ingredients: a product-aware spectral update direction, curvature preconditioning derived from controlling the per-sample loss change, and a magnitude rule that controls the sizes of both the factor and merged updates. We evaluate PoLoRA on instruction-tuning datasets for code and math across models from 1B to 8B parameters, and find that it reaches the final held-out loss achieved by tuned Adam in 1.2-1.7 times fewer steps, while adding at most 3% per-step overhead. Compared to Adam, PoLoRA is also less sensitive to the learning rate, and its optimal learning rate is stable across ranks. 

---
# Is Progressive Disclosure All You Need for Long-Context Agents? 

**Authors**: Yifeng He, Yinzhe Zhao, Jicheng Wang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.17598)  

**Abstract**: Long-document question answering usually forces a choice between loading the whole document into the context window and bolting on a separate retriever. Agentic AI suggests a broader option, giving the agent the document path and letting it decide how and what to read. Agent Skills, a standard for packaging expertise into folders an agent loads on demand, supply a ready mechanism: progressive disclosure, which exposes only what a query needs, from a short description down to the specific passages. Practitioners rapidly adopted this pattern for book-length understanding tasks, but the evidence to support such choices has been anecdotal. We run the first controlled study of the pattern, comparing raw-document navigation and several designs of Agent Skills packs against a classical hybrid retriever across three agent harnesses and three model families on InfiniteBench. On a single book, the gain depends on the harness, running large when the agent navigates the raw document poorly but near zero when a strong agent harness already divides and retrieves on its own. When scaling up to tasks that span many books, raw-document navigation collapses while one-level progressive disclosure degrades more slowly and pulls ahead. A second, deeper routing level never helps and sometimes breaks accuracy outright, so one level is enough. Progressive disclosure buys context, not intelligence: it is redundant while a strong agent can locate the right passages itself, and decisive once the corpus grows too large to navigate by reading. 

---
# D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound 3D NAND-Optimized Similarity Search with Vector Adaptation 

**Authors**: Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S. Rosing, Mingu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17538)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the factual grounding of large language model (LLM) inference by retrieving relevant information from external knowledge bases. However, its dense vector retrieval introduces significant latency and energy overhead, becoming the primary performance bottleneck. Although recent in-storage accelerators aim to reduce data movement, they still rely on host or embedded processors outside the memory, where nearly 70% of the total retrieval time is spent. As a result, they cannot fully overcome the bandwidth limitations, leading to yet another memory bottleneck. To tackle these limitations, we present D-NOVA, a hardware-software co-designed in-storage retrieval accelerator. D-NOVA executes an inverted file (IVF)-based hierarchical retrieval pipeline by deeply embedding the search functionality directly into the NAND memory array. This is achieved by incorporating a new distance metric, Dual-Bound Tight Similarity Sensing (DTS), which is specifically tailored for searching within the NAND string. In addition, we introduce a lightweight contrastive adapter that maps embedding vectors into a DTS-friendly domain, recovering near-software recall while improving performance and energy efficiency. D-NOVA is up to 41.7x faster and 71x more energy-efficient than a CPU baseline, and achieves 12.13x higher throughput while being up to 1.26x more energy-efficient than state-of-the-art in-storage RAG accelerators, demonstrating the potential of fully in-storage vector search for scalable RAG acceleration. 

---
# Salience Induction against Multi-Hop RAG Agents: Threat and Defense 

**Authors**: Xingfu Zhou, Pengfei Wang, Yuan Zhou, Wei Xie, Xu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.17535)  

**Abstract**: Agentic retrieval-augmented generation (RAG) systems increasingly retrieve external evidence and orchestrate tools for knowledge-intensive applications. In Multi-Hop question answering, agents chain facts across documents. Existing defenses focus on content poisoning, which injects false facts, and prompt injection, which embeds directives. We identify a third attack surface: the salience channel, through which fact position, emphasis, framing, and semantic proximity can redirect reasoning even when all retrieved claims are true and no instructions are present. We formalize Salience Induction as truth-preserving edits that redirect Multi-Hop attribute binding while leaving the retrieval trace semantically intact. We define six Salience-Editing operator classes and build an iterative proposer-verifier pipeline under factual and stealth constraints. We also introduce SalientWiki-MH, a decoy-annotated Multi-Hop benchmark. Evaluations across five frontier model families (GPT, Claude, Gemini, DeepSeek, and Qwen) and three agent architectures (ReAct, Reflexion, and tool-calling) show broad generalization. Under a 30% edit budget, Salience Induction achieves an 83.3% attack success rate; the strongest evaluated baseline defense leaves 75.7% post-defense ASR. Untargeted rewriting further reduces attacks only by degrading neutral task success. Our lightweight input-side defense, Salience Normalization, reduces attack success to 15.3% under standard attacks and 23.6% under an adaptive attack. These results show that truthfulness and instruction filtering alone are insufficient: robust agentic RAG also requires defenses against salience-relevance decoupling. 

---
# Thinking in Video: Can Video Generators Really Reason About the Real World? 

**Authors**: Yongheng Zhang, Guang Yang, Ruihan Hou, Qiguang Chen, Ziang Liu, Xiaolong Liu, Manman Zhang, Yanchao Hao, Zheng Wei, Hao Wu, Libo Qin, Peishan Dai, Yinghui Li, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.17523)  

**Abstract**: Recent advances in world models and video generation have given rise to an emerging reasoning paradigm that leverages video generative models to simulate, predict, and reason about real-world dynamics. We redefine this paradigm as Thinking in Video, where video is not merely an output artifact but a medium for constructing, extending, and verifying causal thought. However, this promise remains unverified: convincing rollouts may reflect memorized appearances rather than causal understanding, while existing metrics separate perceptual fidelity from semantic logic. To evaluate whether video generators support such reasoning, we introduce the Causal-Generative Dual-Judge (CGDJ), auditing World Model Consistency from two perspectives. Explicit Causal Perception tests whether a generator reads a video scenario as a reasoning problem through spatio-temporal flattened visual question answering, while Implicit Generative Perception-Prediction Gap evaluates whether it renders the causal consequence as a consistent future video. Applying CGDJ to representative open- and closed-source generators reveals a clear Perception-Prediction Gap: open-source models produce plausible dynamics despite near-zero explicit causal perception, whereas advanced closed-source systems show stronger but still limited alignment between reasoning and generation. Further analysis exposes audio-visual misalignment, where models verbalize correct causal logic more reliably than they render it, challenging the "world simulator" narrative. 

---
# After the Euclidean Highway: Hyperbolic Expert AI as the Next Innovation 

**Authors**: Kwan Soo Shin, In Seok Kang, Munho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.17513)  

**Abstract**: Expert domains are trees; the Euclidean transformer is not, diluting parent-child structure exponentially at depth. The hyperbolic turn left one question unasked: not how much of a network to curve, but where curvature may touch the gradient. Placement is a law, not a knob: the same geometry on a trainable adapter collapses training (seventeen training collapses, ~220 GPU-hours), yet at the loss layer alone it trains without one -- this is HySAT (Hyperbolic Structure-Aware Training), hyperbolic losses at the loss layer only. Across six expert SLMs we constructed and deployed (Llama 3.1 and EXAONE 3.5; four adapter strategies; 18.0M-sample corpus; zero NaN over ~317K optimizer steps), a matched four-arm ablation isolates the preserved manifold invariant, and three propositions and a lemma prove why loss-only placement is stable where adapter-on-manifold is not. Four models are operationally deployed (one live, consumer-facing), two open-weight, with per-step traces and a seventeen-incident failure ledger on Zenodo (CC-BY-4.0). 

---
# Calibrating Semantic Uncertainty from Observable Language-Model Probabilities 

**Authors**: Matthew F. Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2607.17447)  

**Abstract**: Language models produce probabilities over words, but professional decisions require uncertainty over meaningful states such as diagnoses, hypotheses or operational conditions. A model's printed numerical confidence does not establish reliability. We introduce a semantic map: a prespecified, testable bridge from probabilities over verbal responses to probabilities over declared states, formulated as semiparametric inference for a finite-valued latent state. A reference model defines the target posterior, the language model supplies an unrestricted conditional distribution over verbal responses, and held-out calibration connects them. We derive posterior-error bounds and conditions for existence, uniqueness, stability and sequential Bayesian updating. Crucially, language probabilities depend on the prompt's lexical form, whereas the target posterior is unchanged by information-equivalent rewording. We test the method on professional market text compiled from Federal Reserve economic and financial series and on controlled simulations with exact posteriors. Across two fitted language models, language-derived probabilities outperform printed numerical probabilities, recover held-out posteriors with valid uncertainty coverage, remain largely stable under paraphrasing, and respond appropriately to altered evidence. The broader implication is that prompt engineering optimizes a wording-dependent response, whereas scientific and professional use requires validated stability of application-relevant meaning. The semantic map turns this general concern into a testable statistical problem and, when its acceptance conditions hold, yields an auditable posterior estimate. The same principle offers a template for auditing classifications, recommendations and other fluent responses that may conceal semantic instability. 

---
# Abliteration Is Not a Scalpel: Off-Target Effects of Refusal Removal on Decision Disposition Across Model Families 

**Authors**: Aleksander Fafuła  

**Link**: [PDF](https://arxiv.org/pdf/2607.17427)  

**Abstract**: Abliteration - deleting a model's refusal direction from its weights - is the standard recipe behind popular "uncensored" open-weight models. We show the surgery is not clean. As a disposition probe we use 21,600 decisions under uncertainty - weekly up/down calls on 60 Warsaw Stock Exchange equities over 18 weeks, replayed through a frozen pipeline so the decision-layer model is the only variable. The task elicits no refusals at all, so any between-arm delta is pure side effect. Holding provenance constant (official BF16 checkpoints, a single abliteration author, an identical serving stack, one byte-identical frozen prompt), we compare base and abliterated arms of two Mixture-of-Experts families, Gemma-4-26B-A4B-it and Qwen3-30B-A3B-Instruct-2507. Three effects replicate across both families (weeks-clustered bootstrap CIs excluding zero): abliterated models are systematically more optimistic (+12.2 pp Gemma, +7.4 pp Qwen; the confirmed preregistered endpoint), justify themselves at greater length, and use fewer explicit uncertainty words in forced self-critiques (both exploratory). A fourth effect reverses sign: the same operation makes Gemma-abliterated less confident and Qwen-abliterated more (family CIs non-overlapping) - one weight surgery, opposite shifts in expressed confidence. Capability covariates rule out instruction-following degradation as the driver, and no arm shows economic skill: the apparent edge of abliterated arms is regime beta, not alpha. Our provenance audit also caught two independent contamination channels - a mismatched-quantizer pilot pair and a stale community chat template that silently mangled the rendered prompt - suggesting toolchain artifacts are the rule in studies of community-modified checkpoints. Whoever deploys an "uncensored" model as an agent is deploying a measurably different decision-maker, not the base model minus refusals. 

---
# EII-SCL: Harnessing Emotional Inertia for Multimodal Emotion Recognition in Conversation 

**Authors**: Zilong Huang, Kong Aik Lee, Chong-Xin Gan, Zezhong Jin, Ruichen Zuo, Man-Wai Mak  

**Link**: [PDF](https://arxiv.org/pdf/2607.17366)  

**Abstract**: Multimodal emotion recognition in conversation (MERC) achieves accurate predictions by integrating multimodal and contextual information in dialogues. While current MERC approaches focus on modeling complex contextual dependencies in conversation, they often overlook the impact of contextual emotional inertia in emotion shift, leading to sub-optimal performance. To address this issue, we propose a novel Emotional Inertia-Informed Supervised Contrastive Learning module (EII-SCL) that informs the contrastive objective by constructing inertia-affected samples within temporal windows, effectively leveraging emotional inertia as a prior while enabling seamless integration with existing MERC models without requiring additional data. Extensive experiments on IEMOCAP and MELD show that our approach consistently outperforms state-of-the-art methods. 

---
# WAR: Workload-Aware Rollouts for Synchronous Agentic Reinforcement Learning 

**Authors**: Ryan Xu, Atlas Zhao, David Bao, Frank Du  

**Link**: [PDF](https://arxiv.org/pdf/2607.17299)  

**Abstract**: Long-horizon rollout generation has become the dominant systems bottleneck in agentic reinforcement learning (RL). As agents interact with environments over many turns, trajectories rapidly grow to tens of thousands of tokens, making synchronous RL training increasingly constrained by rollout. We propose WAR, a workload-aware rollout system that substantially accelerates synchronous agentic RL by jointly optimizing decoding and scheduling. WAR is built on a key observation: the optimal rollout optimization strategy depends on runtime load: (1) Under low load, WAR enables model-free speculative decoding with SuffixDecoding, which reuses suffix patterns from previously completed trajectories as speculative drafts for future rollouts. Unlike model-based drafters, SuffixDecoding introduces no additional draft model and avoids GPU contention with rollout generation. (2) Under high load, where saturated batched decoding leaves limited room for speculative speedup, WAR shifts the optimization focus to cache-aware scheduling. A global scheduler places requests across rollout replicas based on cache locality, trajectory progress and server load, reducing redundant KV-cache recomputation and mitigating load imbalance. By combining decoding-level suffix reuse with system-level rollout scheduling, WAR delivers robust throughput improvements across workload regimes without changing the underlying RL algorithm. WAR improves long-context agentic rollout throughput by 1.4x under low load and up to 1.6x under high load. These results show that WAR removes a major rollout bottleneck in synchronous agentic RL and provides a practical path toward scalable long-context agent training. 

---
# DRNOISE: Benchmarking Deep Research Agents in Misleading Evidence Environments 

**Authors**: Jun Nie, Zhiqin Yang, Zhenheng Tang, Yonggang Zhang, Xiaowen Chu, Xinmei Tian, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.17291)  

**Abstract**: Deep research agents increasingly operate over the open web, where relevant records coexist with redundant summaries, outdated reports, and misleading documents. Existing evaluations offer limited insight into whether agents preserve sound evidential standards when an ordinary-looking false document is deliberately seeded into a searchable environment and offers a direct shortcut to a conflicting answer. We introduce DRNOISE, a 100-task benchmark for answer recovery under misleading evidence. Each task has a unique gold answer supported by two corroborating indirect record chains; the paired noisy condition adds one plausible document that states a conflicting answer directly. The benchmark spans ten families of evidence operations. Across agents with strong clean-task performance, this single intervention causes 66-88 percentage-point accuracy drops. Trace analyses identify verification inertia as the dominant failure mode: agents often retrieve truthful records but stop before completing and reconciling the evidence chain, instead deferring to the answer-like document. Generic verification prompts reduce but do not close this gap. The setting is especially relevant to open-web deployment, where plausible falsehoods arrive through ordinary-looking pages rather than explicit attacks. Reliable deep research therefore requires more than retrieval and citation; it requires active reconciliation of direct claims with record-level evidence. 

---
# How Jailbreak Attacks Inform Safety Alignment: A Defender-Centric, Shapley-Based Evaluation of Jailbreak Contributions 

**Authors**: Yukai Zhou, Feiyang Lu, Xiaokai Mao, Jinfei Liu, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17152)  

**Abstract**: Jailbreak attacks on large language models are usually evaluated by attacker-centric metrics such as attack success rate (ASR), yet an attack that breaks a model is not necessarily useful for improving its safety. We propose a defender-centric view of jailbreak evaluation, where attacks are evaluated by the downstream safety improvements they enable when used as red-teaming data for safety training. Building on this view, we introduce A-MESS (Minimal Effective Attack-Subset Selection), a setting-agnostic framework for attributing and selecting jailbreak attacks from black-box subset utility observations. A-MESS estimates AttackSHAP, a Shapley-based score that attributes marginal utility to individual attacks and selects compact attack subsets under user-specified budgets via greedy or surrogate-based optimization. Across controlled utility landscapes and real LLM safety settings, we find that ASR rankings are weakly aligned with defender-centric utility, that AttackSHAP can be estimated accurately with limited utility queries, and that directly optimizing subsets yields stronger safety utility than attacker-centric or attribution-only selection. These results suggest evaluating jailbreak attacks as resources for improving safety, not only as tools for breaking models. 

---
# SlotGuard: Stop Oversharing Private Local Context in LLM Agent Transcri 

**Authors**: Haocheng Xia, Yongjoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.17147)  

**Abstract**: LLM agents can leak privacy (e.g., paths, emails) and credentials (e.g., API keys) as agent observations (e.g., tool outputs, shell logs, and file reads) are appended to provider-bound transcripts. Existing placeholder redaction is brittle: it can miss embedded or cross-turn references, over-redact benign lookalikes, and destroy the structure useful for reasoning. We present SlotGuard, a local transcript boundary that can hide sensitive data while retaining agents' performance. SlotGuard rewrites structural bindings as typed, suffix-aware slots, replaces secrets with format-preserving synthetic values, links cross-turn references with a lightweight session graph, and restores raw values only inside the trusted runtime. On controlled repository-oriented agent transcripts, SlotGuard removes all 20,814 annotated structurally sensitive characters across 9,229 paths and reduces credential leakage to 0.0\% across 852 planted values. It remains close to raw-transcript task success across four upstream models, while generic redaction drops to 2.5\%. Transcript rewriting takes a median of 14.424~$\mu$s per agent turn. The code is publicly accessible at this https URL. 

---
# The Geometry of Semantic Space: A Continuous Geometric Framework for the Transformer Architecture 

**Authors**: Zhihua Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17146)  

**Abstract**: We present a continuous geometric framework that models the discrete algebraic operations of the Transformer architecture as an integro-differential equation (IDE) on a semantic fiber bundle $\calE = \calM \times \R^d$. Beginning from a single geometric axiom -- that the token sequence forms a discrete $1$-manifold equipped with a canonical measure lattice -- we translate every core component of the modern Transformer (RMSNorm, RoPE, Softmax Attention, FFN, Residual Stream, SGD, Weight Decay) into a cohesive vocabulary of differential geometry, measure theory, and stochastic calculus. The resulting framework yields quantitative predictions spanning entropic optimal transport (Attention as a Schrödinger bridge) and non-equilibrium thermodynamics (SGD as Itô diffusion violating detailed balance). We conduct a six-part experimental campaign across five architectures (Qwen3, LLaMA\nobreakdash-3.1, Gemma\nobreakdash-3, GPT-2, Mistral) spanning $124$M to $8$B parameters. The empirical observables are quantitatively consistent with the geometric predictions: the $\epsilon^{-1/2}$ Lipschitz scaling calibration at machine precision ($R^2 = 1.000$), the Lie--Trotter operator-splitting torsion, the symmetric ablation instability confirming the Dual-Law of Topological Stability, the $\calO(1/\sqrt{k})$ thermodynamic suppression of Poincaré recurrence on the RoPE torus, the thermodynamic context-limit phase transition, and the Non-Equilibrium Steady State parameter vortex -- verified across two optimizers (AdamW and Pure SGD) to exclude momentum artifacts. The results demonstrate that analyzing Transformers through the lens of continuous stochastic differential geometry provides a predictive descriptive vocabulary for the stability limits, context bounds, and optimization dynamics of Large Language Models. 

---
# Fenced Citation-Context Retrieval for Case Law: Temporal Leakage and Degree Control Across Two Jurisdictions 

**Authors**: Yao Liu, Tien-Ping Tan, Zhilan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17142)  

**Abstract**: Incoming citation context-the text with which later cases describe a case when citing it-is a strong signal for legal precedent retrieval, but it is evaluated without a temporal fence: at query time the target precedent has not yet been cited by future cases, so an unfenced method credits itself with evidence unavailable at deployment. Our central contribution is to measure how much: a temporal-admission decomposition that splits the naive relax-the-fence gain into genuine future-citation leakage, legitimate pre-query admission, and an index effect, reporting the phantom fraction-the share that is future evidence. We instantiate this as a controlled deployability audit of incoming-citation retrieval across two jurisdictions, CLERC (US federal, 1.84M documents) and ECtHR-PCR (European Court of Human Rights), adding a zero-training anchor channel that admits only citers dated before the query. On ECtHR-PCR only 14.9% of the naive gain is genuine future evidence; on CLERC the over-credit grows to +4.95 R@1000 as the fence is relaxed. Under the fence, the zero-training channel gains +16.1 R@1000 over BM25 on CLERC, clears a citation-degree control, and reaches a published R@1000 estimate similar to the strongest trained system on ECtHR-PCR (79.56 vs 79.39) at zero training. 

---
# SynH-Rank: Quality-Aware Code Search via Diverse Data Synthesis and Hierarchical Ranking Training 

**Authors**: Keyu Liang, Haoye Wang, Yanfu Yan, Zhiyuan Wan, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17139)  

**Abstract**: Code search enhances developer productivity by enabling efficient code reuse. Current code search systems often use a retrieve-then-rerank pipeline, where rerankers focus on modeling semantic relevance between queries and code. However, these rerankers overlook critical non-functional qualities like execution speed, memory usage, and maintainability, which are essential for practical software development. Studies reveal developers expect results to maintain high coding standards and satisfy specific needs, such as resource optimization, highlighting the importance of quality-aware code search.
Achieving quality-aware code search faces two major challenges: the scarcity of quality-annotated datasets for effective training and the limitations of standard contrastive learning objectives, which fail to capture the ordinal relationships among high-quality, low-quality, and irrelevant code. Although contrastive learning excels in distinguishing relevant from irrelevant code, its binary objective does not support nuanced quality this http URL address these challenges, we propose SynH-Rank, a quality-aware code reranking framework that combines LLM-driven diverse data synthesis with hierarchical ranking training. SynH-Rank employs a three-level labeling scheme to explicitly model the hierarchy: high-quality relevant > low-quality relevant > irrelevant. Additionally, we introduce a new benchmark with 4,209 pairs and two novel metrics: Quality Preference Accuracy (QPA) for assessing prioritization of high-quality code and Multi-Condition Accuracy (MCA) for evaluating performance under complex this http URL results show SynH-Rank improves QPA by 20.15\% over backbone models and outperforms standard relevance-only contrastive training by 15.80\%, while simultaneously enhancing traditional relevance metrics and multi-condition generalizability. 

---
# Persistent Sparse Autoencoders: Learning Feature Timescales in Language Models 

**Authors**: Haoyan Luo, Mateo Espinosa Zarlenga, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2607.17117)  

**Abstract**: Sparse autoencoders (SAEs) decompose language model activations into sparse features, but standard SAEs encode each token independently and do not expose information that persists across a sequence. We introduce Persistent Sparse Autoencoders (Persistent SAEs), which extend standard SAEs by learning a persistence coefficient for each feature, allowing the model to learn which features should persist and for how long. Our experiments show that they retain competitive reconstruction quality while learning a spectrum of feature timescales: fast features behave as locally interpretable detectors, whereas slow features concentrate topic-level information in a persistent state. Moreover, as shown in a prompt-injection monitoring case study, slow features preserve detection signals and remain causally effective over long contexts. These results suggest that Persistent SAEs open up new opportunities for interpreting and monitoring language models through persistent semantic representations. 

---
# Otap:Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories 

**Authors**: Babak Barazandeh, Subhabrata Majumdar, George Michailidis  

**Link**: [PDF](https://arxiv.org/pdf/2607.17082)  

**Abstract**: Large language model agents solve tasks by generating trajectories that interleave planning, tool calls, and intermediate results. Current evaluation metrics reduce such a trajectory to a binary success flag or compare it against a reference by exact matching. A success flag cannot distinguish a sound solution from one that succeeds by luck, and says nothing about why a failed run went wrong. Exact matching penalizes plans that are valid but reordered or decomposed differently from the reference. We reframe trajectory evaluation as a distance between the agent's execution graph and a set of valid solution graphs, and instantiate it via an unbalanced fused Gromov-Wasserstein transport problem over attributed dependency graphs. The resulting score, termed \otap{} (Optimal Transport for Agentic Planning), is a pseudo-metric that is provably invariant to dependency-preserving reorderings and has bounded sensitivity to redundant steps. Its unbalanced marginals handle missing or hallucinated steps without forcing a match, and its soft coupling accommodates variation in plan granularity. On controlled perturbations and three public benchmarks, \otap{} separates valid from invalid trajectories in a regime where semantics-only metrics score below chance. Its accuracy is highest when the dependency graph is recovered exactly, and drops only when the graph is inferred heuristically from free-text traces. 

---
# A Systematic Evaluation of Traditional Privacy Policy Analysis Tools Against LLMs 

**Authors**: Madhav Aryal, Sudipa Saha, Kaushal Kafle, Anshuman Chhabra, Sunil Manandhar  

**Link**: [PDF](https://arxiv.org/pdf/2607.17075)  

**Abstract**: The advent of LLMs has significantly changed the research on privacy policy and data compliance analysis by enabling tasks that previously required specialized, domain-specific tools. However, it remains unclear to what extent LLMs can truly replicate the diverse functionalities, and the wide range of methodologies and analysis offered by prior work. In this paper, we conduct the first systematic evaluation of whether off-the-shelf LLMs can replace specialized privacy analysis tools. We study six representative tools spanning three major functionalities: contradiction detection, regulatory compliance analysis, and privacy policy summarization and aggregation, and across three intermediate tasks: structured data extraction using tuples, Semantic Role Labeling (SRL) and manual privacy policy labeling. We compare the performance of two state-of-the-art LLMs (GPT-5.2 and Gemini-2.5 in various configurations) against the tools by directly prompting the models to perform corresponding functionalities and tasks on a custom dataset of 10 privacy policies, allowing us to assess whether off-the-shelf models can produce tool-specific functionalities without further engineering or domain-specific training, major limitations in prior work. Our results show that LLMs consistently match or exceed the capabilities of existing tools across the functionalities. In manual labeling of first-party collection entities, LLMs achieved an average precision of 81.8% and recall of 70.9%, while for labeling of third-party sharing entities, they achieved an average precision of 91.4% and recall of 70.8% compared to the OPP-115 dataset. Overall, our findings indicate that LLMs can effectively perform a broad range of functionalities and tasks in privacy policy and regulation analysis that previously required specialized tools. 

---
# Training Continuous Chain of Thought Models: A Tale of Two Regimes 

**Authors**: Varun Yerram, He He, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.16972)  

**Abstract**: Continuous Chain-of-Thought methods replace verbose reasoning traces with a short sequence of dense latent representations. Earlier continuous CoT methods indirectly supervise the latent representations such that its final state match that of verbose reasoning traces, requiring autoregressive, slow generation during training. We introduce C-MTP, a simpler, faster direct supervision approach that models each latent as an average of the embeddings in the CoT traces to be compressed. Our approach outperforms a prior direct supervision method that approximates the distribution of compressed tokens, and performs competitively to slower indirect supervision approaches in existing evaluation setup with simplified CoT traces (less than 100 tokens). Lastly, we extend the evaluation of Continuous CoT methods to complex tasks with longer reasoning traces ($\ge$ few hundreds reasoning tokens). We find both direct and indirect supervision training methods perform poorly (roughly 65\% performance drop) in this setting, revealing the limitations of current continuous CoT methods. The code and checkpoints are released at this https URL 

---
# A Method for Learning Value Systems in Generative AI 

**Authors**: Andrés Holgado-Sánchez, Holger Billhardt, Sascha Ossowski  

**Link**: [PDF](https://arxiv.org/pdf/2607.16903)  

**Abstract**: Value-aware AI systems require explicit computational representations of human values (groundings) and their aggregation into value systems in order to align their decisions with ours. As such representations are difficult to elicit, value learning seeks to infer them by observing human behaviour. This work addresses the lack of grounded value learning methods in generative AI: existing approaches typically replicate human preferences without awareness of the multidimensional structure of value alignment, or lack principled value system elicitation methods. To address these gaps, we adapt a previously validated value system learning method to the generative AI setting, which, based on pairwise prompt-response preference data, simultaneously learns: i) an implementation of a grounding for a set of values given by a multi-objective reward model, and ii) a value system representation in the form of a weighted linear scalarization of the previous grounding model. To ensure that the learned value systems are based on coherent value representations, our algorithm dynamically prioritizes the grounding learning process. We evaluate the method against baselines and a contemporary method on prompt-response preference datasets. Results show competitive performance and minimal trade-offs against the baselines, while improving explainability. 

---
# Cross-Branch Conflict as a Shield: Safeguarding Facial Identities in Unified Multimodal Image Editing 

**Authors**: Weiwei Tan, Junxian Li, Rui Wang, Zhenhua Xu, Yanjun Zhang, Yu Leo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.16898)  

**Abstract**: Unified multimodal models (UMMs) have recently demonstrated powerful instruction-based image editing capabilities, but they also raise serious concerns about unauthorized manipulation of personal portraits. Existing adversarial protection methods are mainly designed for either visual understanding or image generation models and often become ineffective when transferred to UMMs, which process an image through multiple complementary visual pathways. In this work, we first conduct a feature-level analysis of unified image editing. We observe that the ViT-based understanding branch and the VAE-based generation branch exhibit non-trivial structural agreement for the same input image. Although perturbing an individual branch can reduce this agreement and induce intermediate hidden-state deviations, such effects are asymmetric and gradually attenuated during multimodal fusion and generation. These observations reveal that single-branch feature distortion is insufficient for consistently disrupting unified image editing. Motivated by this finding, we propose CCS, a unified adversarial protection framework that jointly drives the ViT and VAE representations away from their clean counterparts while explicitly disrupting their cross-branch compatibility through linear CKA. By simultaneously removing stable information from both visual pathways and creating incompatible visual contexts, CCS prevents the UMM from recovering reliable identity information during editing. Extensive experiments demonstrate that CCS consistently outperforms existing protection methods in suppressing identity-preserving edits. 

---
# DS@GT ARC at eRisk 2026: Hybrid Multi-Agent LLM System with Structured Algorithmic Guidance for Conversational Depression Screening 

**Authors**: Victor Gong, David Guecha  

**Link**: [PDF](https://arxiv.org/pdf/2607.16712)  

**Abstract**: We describe DS@GT's submission to the eRisk 2026 Task 1 challenge on conversational depression screening, in which systems interview LLM personas that simulate individuals with varying depression profiles and produce a Beck Depression Inventory II (BDI-II) score plus four key symptoms per persona, without directly asking sensitive mental health questions. Our pipeline evolved through three stages: a monolithic single-model prototype to start off, a baseline multi-agent architecture that separates conversational interviewing from BDI-II scoring under a coordinating orchestration layer, and a final hybrid configuration that replaces the paid GPT-5-nano interviewer with the open-source Gemma 27B. To offset the model's weaker reasoning and instruction-following, the hybrid adds three algorithmic components: a precomputed dialogue tree that standardizes interview openers and follow-ups, a reliability-weighted consensus aggregation inspired by the Weaver framework, and a cluster-based imputation step for unprobed symptoms. We submitted three fully automated runs across all 20 personas, with Run 1 from the paid baseline and Runs 2 and 3 from the hybrid. Hybrid Run 3 achieved an ADODL of 0.9063, ranking 3rd among all complete-submission runs and placing DS@GT 2nd among the 21 teams overall, while outperforming our paid baseline Run 1 (0.8841) at roughly one-quarter of the per-persona API cost. These results support our central hypothesis that with sufficient algorithmic supervision, a weaker open-source model can compete with a stronger proprietary model in the conversational interviewer role. Our source code is available at this https URL. 

---
# Dependency-Guided Code Generation: Structured Matrix Decomposition and Consistency-Guided Refinement 

**Authors**: Mingqiao Mo, Yangchen Zeng, Zikai Xiao, Xin Xiao, Wenhua Nie, Zhaolu Kang, Guangyuan Dong, Kai Shu, Hao Zhang, Xiaodong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2607.16692)  

**Abstract**: The increasing complexity of modern software systems has made automated code generation a fundamental task in software engineering. However, existing approaches often fail to adequately capture the intricate, multi-level dependencies among code entities, leading to generated code that is logically incomplete or difficult to integrate into real-world systems. To address this limitation, we propose a dependency-aware code generation framework that explicitly models interactions among code entities through a graph-based representation. We decompose dependencies into two complementary components: a quantized matrix that captures strong, explicit relations, and a sparse low-rank factorization that models weaker, implicit interactions. The decomposition is efficiently learned via an alternating optimization procedure. During code generation, the learned dependency structure is incorporated as a constraint, ensuring both semantic coherence and structural consistency of the generated code. Furthermore, we introduce a sparse triplet representation for strong dependencies, significantly improving storage efficiency and computational scalability. Extensive experiments demonstrate that our approach consistently produces code with superior semantic alignment and structural fidelity compared to existing methods. 

---
# Can Multimodal Large Language Models Understand OCT? 

**Authors**: Baochen Fu, Wenzhi Deng, Baihao Jin, Yang Li, Zihan Nie, Kailin Jiang, Yuntao Du, Weiye Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.16609)  

**Abstract**: Optical coherence tomography (OCT) imaging is essential for the diagnosis and treatment of retinal diseases. Although multimodal large language models (MLLMs) have demonstrated considerable potential in medical image analysis, existing benchmarks largely reduce OCT understanding to coarse-grained disease classification or isolated visual question answering, leaving the complete cognitive process from visual perception to clinical reasoning insufficiently evaluated. To address this limitation, we introduce OCT-Bench, a comprehensive benchmark dedicated to OCT image understanding. OCT-Bench comprises 10,076 high-quality multiple-choice questions constructed from 4,137 OCT images across seven public datasets. Following the real-world clinical interpretation workflow, we establish a hierarchical capability taxonomy consisting of 20 fine-grained tasks across three dimensions: Perception, Cognition, and Reasoning. These tasks cover a broad range of capabilities, including imaging attributes, retinal anatomy, lesion characteristics, spatial relationships, disease assessment, therapeutic decision-making, and prognostic management. We systematically evaluate 20 representative MLLMs, including proprietary models, open-source general-purpose models, and medical-domain models. Experimental results demonstrate that current models remain substantially short of reliable OCT understanding. Moreover, neither medical-domain adaptation nor increased model scale consistently improves performance across capability levels. OCT-Bench enables comprehensive and fine-grained evaluation of MLLMs, providing a foundation for identifying capability bottlenecks and advancing clinically grounded OCT understanding. 

---
# One Modality to Forget Them All: Enhancing Cross-Modal Unlearning in Vision-Language Models 

**Authors**: Sudharshan Balaji, Yili Ren, Guangjing Wang, Yimin Chen, Ning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.16442)  

**Abstract**: Machine unlearning is widely used to remove hazardous knowledge from large language models. Modern Vision-Language Models (VLMs), however, process both text and visual inputs, raising a fundamental security question: does unlearning in one modality transfer to the other? We present the first systematic, bidirectional study of cross-modal unlearning transfer across three VLM architectures: LLaVA-1.5 (MLP projection), InstructBLIP (Q-Former), and IDEFICS (gated cross-attention). We find that unlearning transfers across modalities, but the transfer is asymmetric and incomplete. In some cases, text unlearning strongly transfers to vision. However, this robustness is not preserved under typographic attacks that manipulate the visual presentation of text. Under such attacks, previously unlearned knowledge can be readily recovered, indicating shallow unlearning.
To address the transfer gap and shallow robustness, we propose \textsc{CrossInf}, an influence-guided mitigation strategy. Motivated by the observation that different model components contribute unequally to cross-modal transfer, \textsc{CrossInf} focuses unlearning on transformer blocks that most influence cross-modal generalization. It reduces the transfer gap by more than half in architectures with strong fusion, while preserving model utility. It also improves robustness under typographic attacks, reducing the attack success rate to near zero. We further conduct human evaluation with three annotators ($\kappa{=}0.77$) to validate our findings. Finally, we analyze shallow unlearning using Centered Kernel Alignment (CKA), providing insights into the observed transfer behavior and robustness limitations. 

---
# Clarify Before Executing: A Self-Evolving Agent for Resolving Intent Asymmetry in 3D Tool Orchestration 

**Authors**: Xiaoye Zhu, Weixin Li, Junan Huo, Bozhong Wang, Jia Zeng, Yi Yang, Cen Chen, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.16352)  

**Abstract**: A fundamental intent asymmetry plagues modern 3D asset creation: while state-of-the-art 3D toolchains demand precise, executable parameters, ordinary users typically provide vague, underspecified instructions. Current 3D agents treat this ambiguity as noise, defaulting to blind execution under a single-turn assumption. To address this limitation, we introduce CLARE, a clarification-aware and evolutionary 3D agent that treats intent asymmetry not as an execution error, but as an opportunity for strategic dialogue. By decoupling the generation pipeline into four specialized cognitive roles, CLARE intercepts and resolves underspecified instructions before invoking computationally expensive 3D tools to seamlessly execute tasks across five diverse domains: text-to-3D generation, single-view reconstruction, multi-view reconstruction, point cloud editing, and post-processing. Crucially, rather than relying on rigid manual rules, CLARE self-evolves its clarification policy via simulated multi-turn interactions. By optimizing a Multi-turn Reward, the agent internalizes the delicate balance between interaction efficiency and task completion. To rigorously test this, we construct 3D-Clarify, a comprehensive benchmark comprising 620 interaction scenarios with systematically injected ambiguity, missing information, and mistaken details. CLARE achieves state-of-the-art performance, with 60.40% and 43.34% success rates on single-step and multi-step tasks, respectively, more than doubling existing baselines. Both quantitative and qualitative results demonstrate that proactive clarification is the missing key to robust 3D execution. Code is available at this https URL. 

---
# Quantifying Ranking Uncertainty in LLM Benchmarks 

**Authors**: Bitya Neuhof, Yuval Benjamini  

**Link**: [PDF](https://arxiv.org/pdf/2607.16259)  

**Abstract**: Pretrained models are typically ranked on multi-task leaderboards to assess their effectiveness across diverse tasks. Rank confidence intervals were recently introduced as a method to quantify the uncertainty in these rankings by aggregating pairwise hypothesis tests. In this work, we analyze the sources of uncertainty in the knowledge evaluation benchmark MMLU and show how hypothesis tests can be modified to account for their effects. We demonstrate that ranking variability across MMLU subjects is substantial and should be considered when comparing LLMs or identifying the top-performing models. 

---
# From Outcomes to Actions: Leveraging Hindsight for Long-Horizon Language Agent Training 

**Authors**: Zishang Jiang, Tingyun Li, Jinyi Han, Xinyi Wang, Sihang Jiang, Yizhou Ying, Xiaojun Meng, Jiansheng Wei, Jiaqing Liang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.16257)  

**Abstract**: Reinforcement learning (RL) has become a widely adopted technique for improving large language models (LLMs) on complex tasks. Despite this progress, existing RL methods still face challenges in training agents with longer-horizon interactions. One major bottleneck is distinguishing the contribution of different actions in long-horizon interaction, leading to high optimization variance. To address this, we introduce a novel policy gradient method, Hindsight Policy Optimization (HPO), that projects both the current policy distribution and the hindsight distribution into an intent space and extracts low-variance learning signals from the Wasserstein distance between them. We theoretically and empirically show that aggregating semantically similar states and actions in the intent space yields a bounded-variance estimator and improves policy performance stably. Our code is available online. 

---
# Feature Generation Using LLMs: An Evolutionary Algorithm Approach 

**Authors**: Aria Nourbakhsh, Benoît Alcaraz, Christoph Schommer  

**Link**: [PDF](https://arxiv.org/pdf/2607.16255)  

**Abstract**: A crucial step in machine learning pipelines is to present each entity with features or attributes that are representative of the characteristics of the processed entities. Feature engineering is an important step in finding a relation among attributes that otherwise may not be processed by the ML algorithms. Meanwhile, Large Language Models have shown promising abilities in coding, mathematical reasoning, and processing world knowledge. In this work, we utilize an LLM for the problem of feature generation from tabular data based on the previously given features. We have created a pipeline that takes a set of attributes and a prompt to generate new features. Then, our selection algorithm selects the best-performing sets of attributes. We apply our method to eight datasets from different domains and data types. Our results show that, in most cases, the language model can produce new features based on mathematical and logical operators that are useful for the given tasks and can improve classification results. 

---
# CIGPO: Contextual Information-Gain Policy Optimization for Multi-Turn Evidence-Reading LLM Agents 

**Authors**: Hao Dou  

**Link**: [PDF](https://arxiv.org/pdf/2607.16244)  

**Abstract**: Training multi-turn evidence-reading agents with outcome-only reinforcement learning is unstable because intermediate turns receive little direct credit. In HotpotQA experiments with Qwen2.5-3B-Instruct, GRPO initially improves (standard F1 0.430) but subsequently collapses to 100% format-violating outputs. Training-log diagnosis reveals a zero-advantage lock-in mechanism: all sampled trajectories receive the minimum format penalty (-2.0), group-relative advantages vanish, and the policy-gradient loss becomes zero--an optimization deadlock. We propose a variance-injection strategy: by assigning per-turn rewards to intermediate evidence-reading turns, we prevent the group reward distribution from collapsing to a single value--preserving the variation that GRPO's group-relative advantage requires. Contextual Information-Gain Policy Optimization (CIGPO) implements this strategy using the marginal increase in the frozen reference model's log-likelihood of the ground-truth answer as the per-turn signal. With separate normalization of IG and F1 rewards and an IG-weight curriculum, CIGPO reaches a standard F1 of 0.518 on HotpotQA at the 3B scale (from 0.252 base; +105%), compared with 0.430 for the best GRPO checkpoint and 0.000 for the final GRPO checkpoint. CIGPO maintains meaningful reward variance and avoids zero-advantage lock-in throughout training. These results identify reward-variance collapse as a concrete failure mode of outcome-only GRPO and show that turn-level IG rewards can prevent it in this HotpotQA setting. 

---
# From Weights to Words: Expressing and Editing Preference Model Inferences in Natural Language 

**Authors**: Zachary Wojtowicz, Ayush Nayak, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2607.16232)  

**Abstract**: The growing use of statistical learning algorithms to infer human preferences from high-dimensional choice data runs up against a fundamental challenge: choice alternatives typically differ in many ways simultaneously, so it is generally unclear which factors actually drove an observed decision and should be credited as preferences. Compounding this problem, the opacity of these methods leaves human operators unable to inspect, contest, or correct models when they err. We introduce \emph{weights to words}, a method that takes a dataset of choice problems as input and automatically discovers a collection of domain-relevant preference dimensions, each described in natural language and paired with a vector in the model's representational space. These dimensions address both under-determination and opacity: they can be applied to concentrate attribution on a small set of meaningful factors, and they can externalize the model's inferences in natural language so that users can inspect and edit them in real time. We first qualitatively illustrate the method's versatility on four diverse domains: moral dilemmas, movies, wines, and free-form LLM responses. We then report two pre-registered human-subjects experiments, on moral dilemmas ($N=450$) and movie selection ($N=449$), that demonstrate its benefits for learning preference models: (1) regularizing a preference model toward the learned basis increases prediction accuracy on held-out choices, and (2) incorporating participants' structured edits further improves accuracy. In head-to-head comparisons, participants prefer the method's inferred preference profiles and endorse its predictions as more accurate. 

---
# RAIL Guard: Closing the Evaluation-to-Remediation Gap in Responsible AI for LLM Agents 

**Authors**: Sumit Verma, Pritam Prasun, Pritish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2607.16215)  

**Abstract**: Existing guardrail systems for large language model agents operate as binary classifiers that block unsafe content, leaving organizations to discard failing outputs and retry from scratch. We introduce RAIL Guard, a closed-loop responsible AI pipeline that evaluates LLM outputs across eight measurable dimensions and iteratively remediates failing outputs through an evaluate-rewrite-reevaluate loop. We evaluate the pipeline across three experiments on four frontier LLMs and 4,276 content outputs plus 6,400 agent tool-call scenarios. Closed-loop remediation achieves 96.9% convergence versus 49.1% for block-and-retry, though the highest-convergence method reduces utility by 22.3%; feedback-driven self-repair achieves 86.6% convergence on fixable dimensions with no significant utility loss (p = 0.177). Pre-tool-call evaluation reduces unsafe agent executions by 33% (p = 0.007) with zero impact on task completion. We identify a key distinction between fixable dimensions that respond to remediation and structural dimensions (Transparency at 93.0%, Accountability at 92.8%, and Inclusivity at 82.5% failure) that require architectural rather than algorithmic solutions. The system is available as open-source SDKs. 

---
# ColGraphRAG: Late-Interaction Evidence Retrieval for Multimodal GraphRAG 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.16208)  

**Abstract**: Graph-grounded multimodal question answering organizes text, tables, and images in a structured evidence graph, yet end-to-end accuracy depends on which multimodal assets are ranked highly enough to enter downstream reasoning; for graph-linked images, single-vector bi-encoder similarity can discard patch- and token-level structure needed for fine-grained alignment. We evaluate replacing the visual candidate-ranking operator over graph-linked image nodes with late-interaction MaxSim-style multi-vector scoring in the ColBERT/ColPali lineage, while keeping offline graph construction, text- and table-side retrieval, structured extraction, and downstream reasoning unchanged. On MultimodalQA, this change is associated with improved retrieval-stage point estimates for graph-linked image candidates and downstream QA gains, with larger movement where visual evidence matters most and mixed trends on text-dominant questions; we interpret the pattern as mechanism-level evidence for graph-linked visual evidence inclusion, while broader validation and finer graph-level diagnostics remain important future work. 

---
# DocOCR-Eval: A Correction-Based Framework for OCR Tool Selection Without Ground Truth 

**Authors**: Zihan Xu, Puzhen Wu, Lawrence Chun Man Lau, Wei Liu, Sirui Li, Yifan Peng, Yihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.16203)  

**Abstract**: Document parsing is a foundational step for document understanding tasks such as visual question answering and key information extraction, as it transforms unstructured scanned images into structured representations by extracting textual, visual, and layout information. While numerous Optical Character Recognition (OCR) engines and multimodal large language models (MLLMs) have been developed for this purpose, selecting an appropriate document parsing solution for a given document collection remains challenging, particularly in label-scarce settings. In this work, we conduct a systematic evaluation of text recognition performance across a diverse set of OCR engines and state-of-the-art MLLMs on multiple scanned document benchmarks spanning different domains and languages. Motivated by the limited contextual reasoning capabilities of many OCR engines and the high cost of manual annotations, we propose DocOCR-Eval, an annotation-free evaluation framework for automatic OCR assessment and selection. DocOCR-Eval employs a three-staged correction and ranking strategy to approximate annotation-based tool ordering without ground-truth labels. We show that aggregating across multiple MLLMs progressively improves alignment with annotation-based rankings. Extensive experiments further demonstrate that reliable OCR tool selection can be achieved in realistic, label-limited settings, providing practical guidance for deploying document parsing systems across diverse real-world document collections. 

---
