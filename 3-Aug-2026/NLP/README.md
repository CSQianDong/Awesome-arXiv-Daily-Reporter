# TokTier: Exact Stateful Tokenization for Agentic LLM Serving 

**Authors**: Zhenyu Zhang, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2607.29678)  

**Abstract**: LLM serving systems cache prompt KV state, yet most front ends still re-tokenize the full request text on every call. The cost lands on coding agents, which resubmit a long transcript after each small tool result, and reuse is hard because even a short append can change token boundaries near the end of the previous sequence. Across 153,951 calls from two agent ecosystems, the median call appends about 1.4K characters, and only 1.0-3.6% of calls start or rebuild a session with contexts of millions of characters. At a 94.1% fleet prompt-cache hit rate, tokenization reaches up to 64% of time to first token.
TokTier is a stateful tokenization service with one contract: emitted token IDs are always identical to full reference tokenization of the request text. For a session continuation, it re-tokenizes a small window around the append and splices only after a per-request stable-boundary check, widening the window or falling back to full tokenization on failure. For a call without a reusable prefix, it decomposes GPT-family regex pre-tokenization into run-local rules and runs exact pre-tokenization and BPE on a GPU. A sampled shadow verifier re-checks live traffic.
Across 17 tokenizer families, differential campaigns cover 1.5x10^10 split checks, a 12.4 TB real-text corpus, and 93,000+ replayed agent steps, with zero divergence. Incremental repair takes 0.5-1.1 ms from 100K to 3M characters, up to 437x faster than HF tokenization and 2.1x faster at 1M than the strongest cache-based baseline (Gigatoken) fully prewarmed. GPU full tokenization encodes a 1M-character request in 0.87 ms, up to 491x below HF and 23.4x below the fastest published CPU method. With vLLM, median time to first token drops 16-34% and P99 drops 23% under recorded bursts. Under a 50 ms P99 objective, four repair cores plus one GPU sustain 1,821 requests/s where a 16-core stateless front end saturates at 40. 

---
# Evolving language compositionality in a frequency-structured meaning space 

**Authors**: Fabio De Ponte, Eloise Gaines-White, Conor Houghton, Seth Bullock  

**Link**: [PDF](https://arxiv.org/pdf/2607.29642)  

**Abstract**: The iterated learning model was introduced to investigate language evolution: the way in which the characteristic properties of human languages have been shaped, at least partly, by repeated transmission from one language user to another. The key finding is that language compositionality can arise spontaneously as a consequence of language being passed repeatedly through a language learning bottleneck. Here we explore how changing the frequency of different meanings, so that some meanings occur much more frequently than others, affects the character of its compositionality. We find that, as observed in natural languages, high-frequency meanings can escape the pressure to conform to the grammar that characterizes lower-frequency meanings. However, when the frequency structure is instead imposed on parts rather than on whole meaning vectors, the language fails to transmit across generations. This occurs despite the fact that the most frequent elements are reliably learned. These results suggest that frequency can shape emergent linguistic structure only when the frequency distribution is defined over form-meaning units that learners can acquire holistically. When frequency is instead distributed over smaller units, it fails to support the relational structure required for compositional generalisation, thereby preventing stable language transmission. 

---
# FriendBench: Benchmarking Dyadic Familiarity Inference in Humans and Multimodal Large Language Models 

**Authors**: Jeffrey M. Girard, Jason Z. Zheng, Jacqueline R. Vertino, Antony D'Avirro, Benjamin Peloquin  

**Link**: [PDF](https://arxiv.org/pdf/2607.29602)  

**Abstract**: Reading a social situation often depends on behavior, not words alone. We introduce FriendBench, a benchmark for inferring whether two people are already familiar or are meeting as strangers, from a 20-second clip of a dyadic ice-breaker conversation. Every pair answers the same type of prompt, so only the manner of interaction can reveal the answer. Across text, audio, and video, we compare 26 models from seven companies against matched human panels over 96 balanced dyads. The best model and the human crowd are statistically indistinguishable on accuracy in every modality, but reach it differently: humans stay balanced across the two answers, while the strongest models lean toward "stranger"---a difference in effective prior, not discrimination. Richer channels help both unequally, and only humans gain from visible behavior on top of speech. We release the stimuli, human ratings, and model predictions. 

---
# ResKV: Reconstructing Omitted Attention Contributions for Fixed-Budget KV Cache Compression 

**Authors**: Yuhang Zhan, Lisi Chen, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2607.29591)  

**Abstract**: KV cache compression is essential for efficient long-context inference. Existing eviction methods permanently discard unselected tokens and consequently remove their aggregate contribution to attention. Merging-based alternatives preserve more information but can perturb retained keys and values that should remain exact. We observe that the information omitted by cache eviction can be formulated as residual statistics in both the numerator and denominator of softmax attention. Based on this observation, we propose ResKV, which divides a fixed KV budget into an exact main cache and a compact residual cache that reconstructs the contribution of omitted tokens. ResKV lets main-cache tokens and residual entries participate in the same softmax normalization, so residual entries restore both attention numerator and denominator mass rather than acting as a post-hoc correction. A construction-time validation proxy determines residual allocation for each layer and KV head, while a decode-time dynamic gate adjusts residual contributions for individual queries. Comprehensive evaluations on LongBench and RULER, covering query-aware and query-agnostic settings, multiple backbones, cache budgets, and representative compression baselines, demonstrate broad improvements under the same retained KV budget while preserving the practical efficiency of compressed decoding, including peak memory usage and long-context decode throughput. 

---
# Sycophancy Undermines Epistemic Vigilance in Cooperative Vision-Language Tasks 

**Authors**: Rupak Sarkar, Neha Srikanth, Saloni Gupta, Claire Bonial, Philip Resnik, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2607.29585)  

**Abstract**: To maintain common ground in cooperative conversation, humans iteratively update their beliefs as conversation participants share new information; participants who are epistemically vigilant detect when new information conflicts with prior beliefs and take steps to repair these conflicts. In order for AI systems to serve as reliable partners in complex cooperative tasks, they must similarly weigh incoming information against their own private evidence and shared context and appropriately surface inconsistencies when they arise. To measure the epistemic vigilance of vision-language models in cooperative settings, we present an information-asymmetric, dialog-based "spot-the-difference" task. Two models are privately shown one image each, and must determine through conversation whether the images are identical or, if not, identify the difference. Models routinely fail at this: they frequently overlook key evidence in their private image in favor of agreeing with their conversational partner, even when their agreement is unwarranted. We relate these violations of epistemic vigilance to the broader behavior of sycophancy, which manifests itself in cooperative goal-oriented dialog as over-accommodation and weak evidential grounding. Our results show that model steering to reduce sycophancy with a vector learned from task-agnostic sycophancy examples can reduce epistemic vigilance-related errors, making models more faithful reporters of their evidence, and in turn, more reliable partners in information-asymmetric cooperative tasks. 

---
# ARB: A Matched Authorship-Rewriting Benchmark Dataset for AI-Text Detector Evaluation 

**Authors**: Gaetano Perrone, Simon Pietro Romano  

**Link**: [PDF](https://arxiv.org/pdf/2607.29539)  

**Abstract**: Standard AI-text detection benchmarks compare human-written text against text generated directly by large language models (LLMs). While prior work has shown that rewriting and paraphrasing can degrade detector performance, it remains unclear whether performance measured on this conventional benchmark predicts detector behavior when human-authored content is rewritten by an LLM. To address this gap, we introduce Authorship-Rewriting Benchmark (ARB), built from 1,800 human source texts (600 each from XSum, WritingPrompts, and OpenWebText) and four open-weight generators (Llama-3.2-3B, Qwen2.5-7B, Mistral-7B, Gemma-2-9B). Each source item yields four matched variants: human-written (HUMAN), direct LLM generation (Free-LLM), LLM-rewritten human text (H2L), and same-generator LLM-rewritten LLM text (LLM2L). We evaluated five detectors (FastDetectGPT, Binoculars-falcon-7b, RADAR, BERT-Defense, RoBERTa-Defense) at a strict 1%-false-positive operating point (TPR@1%FPR). FastDetectGPT and Binoculars-falcon-7b detected 91.2% and 93.5\% of direct LLM text, but only 30.8% and 15.1% of human text an LLM had rewritten, a drop of 60-78 percentage points. The same detectors retained 78.3% and 83.0% recall when LLM text was rewritten by the same model, a much smaller decline of 10-13 points. RADAR followed the same pattern (66.8% to 12.2%), while BERT-Defense and RoBERTa-Defense stayed below 3% recall across all regimes. These results show that detector performance measured on the conventional human-vs-LLM benchmark does not transfer to human-authored text revised by an LLM, even though the same detectors remain largely robust to LLM-only rewriting. 

---
# Evidence-Type Competition: When Can Interventional Data Teach Language Models Causal Direction? 

**Authors**: Xining Xun  

**Link**: [PDF](https://arxiv.org/pdf/2607.29484)  

**Abstract**: Interventional data is widely regarded as the gold standard for teaching models causal reasoning. We test this assumption in a fully controlled synthetic environment pitting observational correlation against causal effect, and find it fails instructively. In Simpson's-paradox worlds, where the two have systematically opposite signs, increasing the fraction of interventional samples in pretraining does not improve causal direction: the magnitude of the model's do()-response grows monotonically, yet its sign is copied from the observational context. What governs whether interventional evidence is used is not the training mixture but the evidence type present in the context at inference time. Under an identical training recipe, a purely observational context induces systematic sign reversal in 29/50 worlds, a mixed context in 19/50, while aligned interventional probes alone yield 41/50 correct. Erasing observational evidence from the context immediately releases the suppressed causal interpolation ability (ratio_true = +0.56); a four-state content manipulation shows the switch is content-mediated and graded. The suppression is stable across training seeds (11/11 strong reversals persist on a matched-protocol second seed) and robust as a rate at 0.93B parameters (31.8% vs. 6% reversals in the matched probe-only arm), even as absolute gains shrink four-fold. An external audit on CLadder exposes a learned positive-effect prior with a two-layer structure: sign-randomized retraining removes it in-distribution but not out-of-distribution. We summarize: the capability lives in the weights; the switch lives in the context, and activation patching localizes the switch to the middle layers' observational rows. We further quantify the sampling noise floor of probe-based causal evaluation and an evidence-averaging protocol that cuts sign errors from 26% to 9%. 

---
# Know It, Act on It: Investigating Memory Utilization in LLM Personalization 

**Authors**: Zhaoxin Feng, Jianfei Ma, Emmanuele Chersoni  

**Link**: [PDF](https://arxiv.org/pdf/2607.29433)  

**Abstract**: As large language model (LLM) agents evolve into personalized companions, memory has emerged as a core capability. However, LLMs face a knowledge utilization problem: they may fail to act on relevant user preferences even when they are fully present in context. When an agent fails to tailor its response in a context where previously shared user preferences should matter, it is unclear whether the model failed to remember that information or remembered it but failed to use it. To isolate this breakdown, we introduce a decoupled evaluation paradigm that administers paired Know and Act tests to the same user preference. We conduct large-scale experiments across 16 systems and five memory architectures, evaluating 1,000 preferences embedded at three levels of expression strength. Our results show a large gap between Know and Act outcomes: agents often pass the recall test for a user preference but fail to reflect that same preference in the paired behavioral scenario. While memory architectures reduce this gap, utilization remains especially weak for health and therapy-related preferences, where failures to act carry the greatest real-world stakes. 

---
# Studying quantization trade-offs for efficient inference deployment in machine translation 

**Authors**: Jim Zhao, Sohir Maskey, Koen Oostermeijer, Douglas Orr, Teryn Jones  

**Link**: [PDF](https://arxiv.org/pdf/2607.29397)  

**Abstract**: Deploying large language models in realistic server environments poses challenges, as the system needs to provide high-quality responses with low latency. Quantization is a common approach to reduce the memory footprint and improve inference efficiency, yet its impact on latency and throughput is rarely evaluated under controlled, orchestration-level workloads. In this work we study the quantization trade-offs of two translation model families, EuroLLM \citep{martins2025eurollm} and Hy-MT2 \citep{zheng2026hy} across five models ranging from 1.7B to 22B for efficient deployment on a single A100 or H100 GPU. We demonstrate that combining a document-chunking strategy with W4A8 or W8A8 quantization improves the latency-throughput Pareto-curve under a wide range of workloads. Furthermore, since standard machine translation (MT) benchmarks rely on isolated sentences and fail to capture long-context dynamics, we introduce a document-level evaluation from WMT24++ to assess how text chunking strategies affect translation quality under quantization. Our results reveal that standard segment-level evaluation can fail to predict the interaction between quantization and long-context document translation. While Hy-MT2 remains robust under quantization, EuroLLM shows strong sensitivity and translation quality collapses rapidly for all considered quantization formats. Overall, our experiments show that the trade-off between inference efficiency and translation quality depends not only on the quantization format, but also on the choice of text chunking strategy. 

---
# PTP: Previous-Token Prediction based LLM Inversion for Near-Exact Prompt Reconstruction 

**Authors**: Pirzada Suhail, Nagasai Saketh Naidu, Atanu R Sinha, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2607.29378)  

**Abstract**: Large language models (LLMs) generate text by auto-regressively sampling the next token. This inherently leads to a many-to-many mapping between prompts and responses, complicating the task of inferring prompts from observed outputs. Prior work on LLM inversion frames prompt recovery as a semantic reconstruction task. They rely on fine-tuning pretrained sequence-to-sequence models on large external datasets--and requiring access to model weights or logits--to generate semantically plausible prompts. In contrast, we present a functional approach to inverting a given LLM in a black-box setting, without auxiliary aids. We train an explicit inverse language model entirely from scratch on data synthetically generated from the target LLM itself. Analogous to forward next-token prediction, our inverse model is trained using previous-token prediction, establishing a generative link between the forward and inverse processes that enables faithful prompt reconstruction. Moreover, it naturally supports diverse prompt reconstructions through sampling, whereby all such prompts induce similar responses under the forward, target LLM. Our approach generalises across datasets and exhibits transferability in reconstructing prompts from responses generated by different LLMs. Further, across the set of token based evaluation metrics for prompt and response reconstructions, our approach outperforms prior work. 

---
# Zero-Mem: Zero-Token Memory Operations for LLM Agents 

**Authors**: Yilin Xiao, Zhehan Zhu, Yujing Zhang, Jin Chen, Zijin Hong, Luyao Zhuang, Qinggang Zhang, Shengyuan Chen, Xiaocao Ouyang, Lingfei Ren, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.29377)  

**Abstract**: LLM agents need memory to act consistently over long interactions, yet many systems use additional LLM calls to operate that memory. Generating intermediate records and mediating their retrieval adds recurring token and time costs, while omitted or merged details can obscure the original evidence. We ask whether structured memory access requires generation at all. Zero-Mem introduces \emph{zero-token memory operations}: no step outside final question answering invokes an LLM or consumes LLM input or output tokens; encoder computation is accounted for separately. Zero-Mem preserves original interaction traces as its source of record. It organizes the traces in two complementary ways. An entity--context graph exposes connections across interactions, while a temporal hierarchy preserves conversational locality and session state. For each query, Zero-Mem weighs the two views, retrieves from both, and follows their structure to recover supporting relations or surrounding context. Deterministic calibration first discards conflicting evidence and then keeps the reader's answer grounded in the retrieved traces. Only the final-QA reader invokes an LLM. Across long-memory and long-context question-answering benchmarks, Zero-Mem achieves competitive performance while eliminating LLM calls and LLM-token consumption from memory operations. With the same final-QA reader and context budget, it reduces memory-operation time cost by 57.6\% relative to the fastest compared baseline. Ablations support the contribution of the two views and their query-dependent coordination. Overall, the results show that structured agent memory need not generate an intermediate representation of the past. After peer review, the code and implementation details will be available at \textcolor{blue}{this https URL}. 

---
# Cross-Lingual Transfer for Machine Translation in Turkic Languages 

**Authors**: Omer Burak Cinar, Mehmet Mert Dalkilic, Cagri Toraman  

**Link**: [PDF](https://arxiv.org/pdf/2607.29355)  

**Abstract**: Cross-lingual transfer is central to low-resource machine translation, but its behavior within closely related language families remains insufficiently characterized. We study transfer among five Turkic languages; Turkish, Azerbaijani, Uzbek, Kazakh, and Kyrgyz; using pairwise transfer matrices. In this setting, each model is fine-tuned with one transfer source and evaluated on a different transfer target while the translation target remains the same. Across mT5 experiments, we find that transfer is strongest between closely related Turkic pairs, especially Turkish-Azerbaijani and Kazakh-Kyrgyz. We also show that transfer direction matters, and that the same transfer source-transfer target pair can behave differently when the translation target changes. Latinization improves BLEU and chrF in several script-mismatched settings, but its effect is not uniform across metrics. Additional analyses show that transfer sources are mostly stable across different datasets and model settings. 

---
# Translation with Thought: Difficulty-Adaptive Reasoning via Reinforcement Learning for Multi-Domain Machine Translation 

**Authors**: Yongshi Ye, Biao Fu, Chongxuan Huang, Yidong Chen, Xiaodong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.29287)  

**Abstract**: Multi-domain machine translation (MDMT) poses a unique challenge due to varying levels of linguistic complexity across domains. Inspired by human translators' ability to adapt reasoning effort based on difficulty, we propose TwT (Translation with Thought), a resource-rational framework that learns to modulate inference between intuitive and deliberate reasoning. TwT is trained in two stages: (1) supervised fine-tuning on difficulty-aware long chain-of-thought traces distilled from DeepSeek-R1 and rewritten by GPT-4o to reflect human-like reasoning economy, and (2) reinforcement learning with a hybrid reward to optimize translation quality and reasoning efficiency. Evaluated on 15 benchmarks spanning in-domain and out-of-domain settings, as well as 3 seen and 59 unseen languages, with ablations across three backbone models, TwT-7B and TwT-14B outperform much larger SOTA reasoning models in translation quality, while reducing token usage by 32--60\%. These results confirm that aligning translation behavior with cognitive principles enables robust generalization, high translation quality, and efficient reasoning in MDMT. 

---
# CalibratedRubric: Task-Adaptive Rubric Banks for Open-Ended LLM Evaluation 

**Authors**: Mengting Chen, Yanshu Sun, Wanting Liang, Beidi Luan, Rui Sun, Dezhi Chen, Jing Li, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2607.29252)  

**Abstract**: Reliable evaluation of open-ended LLM outputs requires fine-grained rubrics, yet expert curation is costly and difficult to scale. Existing automated pipelines rely on strict judge unanimity and binary variance filters, which cannot distinguish measurable rubrics from informative ones. We introduce CalibratedRubric, a task-adaptive framework that combines type-specific scoring, Bayesian rubric-measurability filtering, and item response theory (IRT)-based bank assembly. CalibratedRubric estimates each rubric's measurability with a Beta--Bernoulli agreement posterior and uses a submodular information-coverage objective to construct compact rubric banks over the observed capability range. Across financial, healthcare, general, and legal benchmarks, measurability filtering improves human-gold agreement on JudgmentBench from $\kappa=0.604$ to $0.743$. IRT-based greedy selection improves cross-fitted rank fidelity over random selection across all six evaluated response blocks and requires only 49 rather than 131 rubrics to reach the target correlation on FinResearchBench decision-support tasks. Task-label perturbations further reduce system separation, confirming the practical relevance of task-adaptive scoring. These results support CalibratedRubric as an efficient, uncertainty-aware approach to open-ended LLM evaluation, with calibration gains depending on sufficient judge redundancy. 

---
# Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation 

**Authors**: Goutham Ramakrishnan, Megha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2607.29250)  

**Abstract**: Small language models (SLMs) are attractive for agentic deployment due to low latency, reduced cost, and on-device privacy, yet they struggle with tool-use tasks where training data is scarce and noisy. Unlike larger models, SLMs cannot compensate for low-quality supervision through sheer capacity, making data quality the critical bottleneck. We present Data Turnstile, an open-source framework that takes user-defined API specifications and generates high-quality synthetic training data for function calling. Turnstile decomposes multi-turn tool-use interactions into constrained, stepwise generation with validation and error-feedback loops, providing fine-grained control over API diversity, conversation complexity, and output correctness. We demonstrate effectiveness of domain adaptation with Turnstile data on two challenging function calling benchmarks. On the BFCL single-turn benchmark, a Qwen3-0.6B fine-tuned on Turnstile data without chain-of-thought achieves 75.9% overall accuracy (versus 67.4% for the base model with thinking enabled), closing the gap with thinking-enabled Qwen3-1.7B (78.4%) and Qwen3-4B (79.9%) despite being 3$\times$ and 7$\times$ smaller respectively. On $\tau^2$-bench, a multi-turn agentic benchmark, Turnstile-trained Qwen3-1.7B achieves 31.1% pass^1 on the Telecom domain, improving 4.7$\times$ over its 6.6% base and surpassing Qwen2.5-32B-Instruct (27.4%), a model 19$\times$ larger. Turnstile-trained Qwen3-0.6B achieves 24.6%, improving 7$\times$ over its 3.5% base and approaching the 32B model (53$\times$ larger). We release Data Turnstile along with a dataset spanning 1,000+ APIs and 100K+ multi-turn interactions. 

---
# Small Is Enough: Per-User Style Rewriting of AI-Edited Text via LoRA Adapters 

**Authors**: Antorweep Chakravorty  

**Link**: [PDF](https://arxiv.org/pdf/2607.29238)  

**Abstract**: InMyStyle is a privacy first, single user system that adapts small language models to rewrite AI-edited text towards an individual user's writing style without an instruction prompt at inference. Given a user's documents, it uses multiple local helper LLMs to construct paired training examples and fine tunes LoRA adapters on base models ranging from 0.5B to 7B parameters. Length aware generation budgets and automatic chunking support inputs of different lengths. On 219 evaluation pairs from a scientific-paper corpus, the automatic composite score plateaus at 0.69 [scale 0-1] across all model sizes under both greedy and sampled decoding. This observed plateau suggests that small models are sufficient for the measured rewriting task, with model size determining trade-offs rather than a stable quality ranking. As a secondary evaluation, 400 ratings from five LLM judges give InMyStyle outputs a mean perceived AI-ness score over 20% lower than their helper-AI generated inputs, while mean perceived AI-ness scores decrease with model size within InMyStyle. 

---
# Knowing When to Quit: Diagnosing and Training LLMs to Abort Futile Reasoning 

**Authors**: Xinyan Guan, Jiali Zeng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Fandong Meng  

**Link**: [PDF](https://arxiv.org/pdf/2607.29211)  

**Abstract**: Large language models generate computationally expensive yet semantically void reasoning on beyond-capability tasks, creating risks where plausible-sounding but incorrect derivations mislead users. We characterize this \textit{futile reasoning} phenomenon through systematic analysis, revealing universal capability overreach and systematic miscalibration between capability and behavior. The dominant failure mode is specious reasoning, which outputs look superficially valid but contain subtle errors, escalating with task difficulty. To address this, we introduce \textbf{CaRL} (\textbf{Ca}pability-\textbf{a}ligned \textbf{R}einforcement \textbf{L}earning), which aligns model behavior with capability boundaries through reward shaping that incentivizes refusal over futile reasoning and hindsight refusal augmentation that converts failures into refusal supervision. Experiments demonstrate a substantial reduction in futile reasoning while preserving performance across task difficulties, effectively achieving capability-aligned behavior without sacrificing utility. \footnote{this https URL} 

---
# Hy-MultiTurn: A Six-Dimensional Benchmark for Deep Multi-Turn Dialogue Understanding 

**Authors**: Eileen Ye, Jiawen Tao, Yaoming Li, Chenxu Liu, Wenhan Yu, Yaxin Fan, Xiaokun Yuan, Mengzhou Wu, Yanbing Jiang, Maxm Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.29196)  

**Abstract**: Long-running multi-turn interactions with chatbots and agents are now common, and a correct response often depends on remembering earlier details, tracking later revisions, identifying intended objects or referents, and withholding action when required conditions are unmet. Existing multi-turn benchmarks typically cover short exchanges and do not fully evaluate these capabilities in long multi-turn interactions, particularly in Chinese, while offering limited insight into how and why models fail. To address these limitations, we analyze real chatbot failures to identify six recurring mechanisms and use them to define six controlled evaluation modes in Hy-MultiTurn, a Chinese benchmark for deep multi-turn dialogue understanding. The six modes evaluate constraint memory, precise execution, constraint synthesis, object localization, action suppression, and reference resolution. Across the six modes, we construct 209 controlled tasks spanning 12-76 turns, with dialogue length, irrelevant-topic distraction, and colloquial phrasing adding further difficulty. Evaluation of 22 frontier model configurations shows that Hy-MultiTurn is broadly challenging, as even GPT-5.5, the strongest overall configuration, satisfies all requirements in only 41.1 percent of responses and no model performs best in all six modes. 

---
# Detecting Experiential Intertextuality Across Migration Routes: Beyond Surface Similarity in French Narratives 

**Authors**: Sakayo Toadoum Sari, Nelly Robin, Michelle Auzanneau, Lakhdar Sais, Veronique Petit, Marie Veniard, Said Jabbour, Fabien Delorme  

**Link**: [PDF](https://arxiv.org/pdf/2607.29188)  

**Abstract**: Migrants traversing geographically distinct routes such as the Trans-Saharan and Balkan corridors often recount strikingly parallel lived experiences: police violence, smuggler exploitation, dangerous crossings, and family separation. We introduce the task of experiential intertextuality detection: automatically identifying shared experiential echoes across migration narratives without requiring annotated training data. From 108 French migration narratives spanning both corridors, we automatically generate sentence pairs and score them using annotation-free methods: lexical baselines, sentence embeddings, POS-based structural features, a migration-specific theme lexicon, context-aware narrative features, and zero-shot LLM scoring with Qwen2.5-7B and Mistral-7B under three prompting strategies. We validate all methods against 816 expert-annotated intertextuality judgments (inter-annotator Krippendorff's $\alpha = 0.27$). Our results reveal that all surface, structural, and embedding methods correlate only weakly with expert judgments ($r \leq 0.30$); Qwen2.5-7B zero-shot achieves the best single-method correlation ($r = 0.38$); few-shot examples degrade Qwen but dramatically improve Mistral; narrative position significantly predicts intertextuality, with departure-phase pairs showing the highest experiential echoes; and a supervised hybrid combining all 31 features achieves $r = 0.45$, a 21% improvement over the best individual method. 

---
# Learning Latent Reasoning Traces for Scalar Reward Models End-to-End 

**Authors**: Sanwoo Lee, Clive Bai, Hsiu-Yuan Huang, Kun Liang, Weijie Liu, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.29185)  

**Abstract**: Reward models (RMs) are central to aligning large language models with human preferences via reinforcement learning. Although traditional scalar RMs enable efficient and probabilistic reward modeling, they rely on superficial cues that fail to generalize to complex or out-of-distribution (OOD) tasks. Conversely, generative RMs leverage extensive reasoning to improve robustness on challenging tasks, but their natural language-based scores lack the numerical flexibility and probabilistic interpretability that scalar RMs offer. While recent approaches combine both paradigms through off-policy multi-task learning, such parallel optimization does not guarantee that generated reasoning traces actively align with or benefit downstream scalar reward prediction. To address this mismatch, we propose LatentRM, a reward modeling framework that learns intermediate reasoning traces as discrete latent variables to explicitly maximize the likelihood of downstream scalar rewards. Through on-policy optimization of the latent reasoning space end-to-end, LatentRM tightly couples deep reasoning-based evaluation with precise scoring. Extensive validations on in-distribution and OOD datasets and RLHF show that LatentRM outperforms scalar, generative, and hybrid RMs on preference modeling and policy alignment across tasks ranging from open-ended conversation to complex reasoning. 

---
# Authorship Verification of Transcribed German-Language Videos 

**Authors**: Oren Halvani, Sophie Titze  

**Link**: [PDF](https://arxiv.org/pdf/2607.29168)  

**Abstract**: Authorship Verification (AV) represents an important subfield of digital text forensics and addresses the fundamental question of whether two texts were written by the same author. Although the field has made substantial progress over the past two decades, several important challenges remain unresolved or underexplored. For instance, most AV research has focused on written texts, despite the fact that language is expressed not only in written but also in spoken form, such as in videos. Moreover, existing AV studies have predominantly concentrated on English, while other languages, including German, have received comparatively little attention. To address these research gaps, we apply AV to spoken language in the form of transcripts of German-language videos and examine the effectiveness of established AV methods in verifying a speaker's identity across video pairs. Our experimental evaluation, based on a total of ten AV methods applied to three self-compiled corpora comprising 300 videos from 150 speakers, shows that the best performance (up to 88% accuracy and 90% AUC) is achieved by traditional AV approaches based on simple character- and token n-gram representations. In contrast, more modern transformer-based approaches perform significantly worse on all evaluated corpora. Our results therefore suggest that traditional methods in the field of AV remain both competitive and relevant. 

---
# M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain Benchmark for Full-Duplex Spoken Dialogue Models 

**Authors**: Ryo Fukuda, Atsushi Ando, Hiroki Kanagawa, Takatomo Kano, Marc Delcroix, Naohiro Tawara, Yuya Chiba  

**Link**: [PDF](https://arxiv.org/pdf/2607.29125)  

**Abstract**: Full-duplex spoken dialogue systems (FDSDSs) can listen while speaking, enabling natural behaviors such as smooth turn-taking, backchannel handling, and user barge-in handling. However, fair comparisons in multi-turn conversations remain a challenge. In addition, existing benchmarks provide limited coverage of languages and dialogue domains. We propose M3-DuplexBench, a multi-turn, multilingual, multidomain benchmark for FDSDSs. M3-DuplexBench supports English and Japanese and covers both casual conversation and multi-turn question answering. In addition, we evaluate models under multiple dialogue context settings, including single-turn, user-only, and teacher-forced full-context settings, to analyze how dialogue history affects model behavior. Experiments with recent FDSDSs reveal model-specific turn-taking characteristics, clear performance gaps across languages and domains, and mixed effects of dialogue context. 

---
# Can Zero-Shot LLMs Predict Child Malnutrition? A Fairness and Temporal Robustness Study 

**Authors**: Muhammad Ashad Kabir, Md Ahshanul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2607.29082)  

**Abstract**: Child malnutrition remains a major public health challenge in low- and middle-income countries, particularly in South Asia, where early identification of vulnerable children is critical for timely intervention and resource allocation. This study aims to evaluate the feasibility, fairness, and temporal robustness of using a pretrained large language model (LLM) in a zero-shot setting for child stunting prediction using population health survey data. Using Bangladesh Demographic and Health Survey (BDHS) data collected between 2007 and 2022, we transformed maternal, child, healthcare, and household characteristics into semantically interpretable prompt-based representations and evaluated GPT-4o-mini for zero-shot stunting prediction, comparing its performance against a random forest baseline and assessing fairness across demographic and socioeconomic groups as well as temporal robustness across survey waves. The results demonstrate that zero-shot inference using GPT-4o-mini achieved comparable balanced accuracy to the supervised baseline while exhibiting substantially higher sensitivity for identifying stunting cases, relatively consistent performance across child sex groups, and stable predictive behaviour across BDHS waves; however, important fairness disparities were observed across residence and household wealth categories, highlighting the need for further investigation before deployment of foundation models in public health prediction settings. 

---
# Faster but Different: Diagnosing and Controlling Content Drift in Accelerated Multimodal Diffusion Language Models 

**Authors**: Yaoxuan Dou, Yang Shu  

**Link**: [PDF](https://arxiv.org/pdf/2607.29079)  

**Abstract**: Training-free acceleration makes diffusion-based multimodal large language models (dMLLMs) more deployable, but it may silently change generated content. We study this serving-time consistency problem on 300 real images, comparing Fast-dLLM outputs with the same model's unaccelerated outputs. Across the mild parallelism induced in our long-form setting (1.05--1.25 committed tokens per step), confidence-threshold tuning changes decoding behavior but not baseline agreement. State-refresh ablations and an image-swap intervention instead identify stale visual and generated-text states as contributors to drift. For the tested Fast-dLLM implementation, shortening the KV-cache refresh interval yields a monotonic speed--agreement frontier and near-exact agreement at a measured 1.3x speedup. The initial diagnosis also appears with dLLM-Cache and LaViDa, although dLLM-Cache recovers agreement only after both caches are tightened, which removes its speed advantage. Independent prompts and images reproduce the threshold-insensitivity and refresh recovery. A targeted audit finds genuine content substitution in half of 50 low-agreement pairs. In a separate blinded two-annotator evaluation, the pooled accelerated-minus-baseline factual-error difference is 0.00 (95% CI [-0.17,+0.17]); this sample detects no difference but does not establish factual equivalence. Finally, none of the tested adaptive or smoothed-refresh variants beats the fixed interval at matched compute. Our contribution is a paired diagnostic and an implementation-scoped consistency control, not an accuracy or safety guarantee. 

---
# Semantics of Subterfuge: Benchmarking Legal Deception Detection Against General-domain State-of-the-Art 

**Authors**: Theekshana Samaradiwakara, Nisansa de Silva, George C. Lobb  

**Link**: [PDF](https://arxiv.org/pdf/2607.29066)  

**Abstract**: Deception detection has critical implications for legal proceedings, law enforcement, and online security. Although human judgment is limited in accuracy and scalability, Natural Language Processing (NLP) offers a data-driven alternative. We present a survey and comparative analysis of NLP-based Automatic Deception Detection (ADD) focusing on the legal domain, reviewing the evolution from feature-based machine learning to Large Language Model (LLM) approaches. We conduct a unified empirical evaluation across seven datasets (two legal, five general-domain), comparing six fine-tuned transformer models and seven LLMs under four prompting strategies. The results show strong domain sensitivity, with fine-tuned models excelling in data-rich general domains and few-shot LLMs remaining competitive in low-resource legal settings. Chain-of-Thought prompting often underperforms direct classification. These findings highlight the need for domain adaptation and interpretable systems in high-stakes legal contexts. 

---
# Tokenizer-Agnostic Engram Module 

**Authors**: Jia Peng Lim, Hai Leong Chieu  

**Link**: [PDF](https://arxiv.org/pdf/2607.29065)  

**Abstract**: Deepseek's Engram, a conditional memory module, was introduced to trade-off storage versus reasoning in large language models. However, the module relies on token-level $N$-gram hashing for Engram embedding lookup, introducing a tight coupling to the tokenizer used: a model with a different tokenizer would have to train its own Engram embeddings from scratch. To improve the reusability of Engram embeddings, we propose a change to the hashing routine, enabling compatibility between Engram models using different tokenizers. Instead of modelling disjoint $N$-gram spaces, we treat $N$-gram as a method to sample potentially useful byte sequences, from all possible byte sequences across tokens. We replace the XOR-based hashing with the general polynomial hashing with a joint embedding space across $N$. This work investigates the possible trade-offs and shows that this simple substitution produces comparable performance and achieves tokenizer-agnosticism: hash equivalence for byte-equivalent token sequences. 

---
# From Inline Notes to Collected Commentaries: Toward Context-Preserving Organization of Exegetical Knowledge in Classical Chinese Texts 

**Authors**: Ke Liang, Qi Su, Churen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.29044)  

**Abstract**: Inline notes and collected commentaries are important forms of scholarly communication that evolved within the Confucian exegetical tradition, yet have received little computational attention. Drawing on traditional Chinese exegetics and philology, this paper formulates collected commentary compilation as an NLP task and proposes a computational framework that preserves the contextual dependency of inline notes while enabling their automatic compilation and exegetical knowledge organization. It combines two-step prompt chaining for identifying the associated main-text segments and exegetical functions of annotations with cross-source mention clustering for integrating commentary across editions, achieving a CoNLL F1 score above 97% in a case study on the Classic of Mountains. Our framework lays the foundation for the large-scale organization of historical exegetical knowledge, thereby supporting a broad range of downstream philological and NLP tasks. 

---
# PARALLEL: A Prefrontal-Aligned Reinforcement inspired Approach for Language-Model Learning under Explicit Limits 

**Authors**: Namkyung Yoon, Sanghong Kim, Hwangnam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.28982)  

**Abstract**: Recent language models achieve strong performance across a variety of tasks, but conventional adaptation applies updates uniformly across training samples regardless of their local update benefit. We propose PARALLEL, a prefrontal-aligned reinforcement inspired approach for language-model learning. Inspired by the complementary roles of goal-related and uncertainty-related control, PARALLEL represents these forms of information as separate controller signals and combines them with the current model representation. A reinforcement-inspired controller assigns sample-dependent update intensity using immediate utility-cost feedback. PARALLEL therefore learns when and how strongly to adapt to each sample, prioritizing beneficial updates while limiting unnecessary parameter changes. PARALLEL uses available updates more efficiently than selective baselines while retaining 94.1--99.2\% of Full-adaptation performance. Beyond multiple-choice reasoning, experiments on XSum and CNN/DailyMail show that PARALLEL retains 96.9--98.6\% of the ROUGE-1 and ROUGE-2 scores achieved by Full adaptation and 98.8--98.9\% of the corresponding ROUGE-L scores. When compared at the same cumulative adaptation time or GPU energy, PARALLEL achieves higher ARC accuracy and exhibits a more stable late-stage adaptation trajectory than Full adaptation in the representative run. These results show that learning when and how strongly to update each sample supports stable and efficient post-deployment stream adaptation while avoiding unnecessary updates. 

---
# Mixture-of-Translators: Translating KV Caches Across Heterogeneous Large Language Models 

**Authors**: Jin-woo Lee, Minkyung Song, Junghyun Oh, Seunghoon Han, Soyoung Park, Gwangseon Jang, Sungsu Lim  

**Link**: [PDF](https://arxiv.org/pdf/2607.28979)  

**Abstract**: Heterogeneous Large Language Model (LLM) systems increasingly rely on shared contexts, retrieved evidence, and multi-agent dialogue histories, yet their internal key-value (KV) caches remain model-specific and cannot be reused across architectures. Consequently, each model must repeatedly prefill or store caches for the same context, limiting the scalability of multi-model reasoning and long-context generation. We propose Mixture-of-Translators(MoT), a cache translation framework that maps context KV caches from a source LLM into the cache space of a target LLM. Unlike prior approaches that depend on a single projection path or global shared latent space, MoT uses multiple translator modules to capture diverse source--target mappings. To further reduce residual translation error, we introduce a Context Correction Loss that aligns the replayed target trajectory with the native target trajectory. We reveal two competing failure modes in cache translation: propagated translation shift from early injection and last-state shift from late injection. MoT addresses them through translator mixtures and target-side correction. Across homogeneous and heterogeneous translations among Qwen2.5, GPT-2, and OPT models, MoT preserves downstream QA performance, including Qwen2.5-7B-scale translation with 51.0% average closed-set QA accuracy and 0.43 average extractive QA F1. In practical case studies, MoT enables quality-preserving memory reuse for multi-agent reasoning and retains 96.3% of direct-context quality in long-context cache-augmented generation, demonstrating scalable KV cache reuse across heterogeneous LLMs. 

---
# BLADE: Boundary-Expanded and Layer-Adaptive Dynamic Exit for Efficient LLM Reasoning 

**Authors**: Keshu Fu, Keqin Peng, Jun Bai, Shuhan Qin, Chen Li, Junzhu Liang, Yefei Chen, Jiaqi Li, Yuanxin Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28966)  

**Abstract**: Large language models often improve task performance by generating long reasoning traces, but the resulting computation is frequently wasted on redundant verification and revision. Existing probe-based early-exit approaches mainly inspect explicit self-doubt expressions, leaving many earlier termination opportunities undetected. Expanding inspection to ordinary reasoning boundaries improves coverage, but also exposes highly diverse intermediate states whose predictive information may reside in different hidden layers. We present Boundary-Expanded and Layer-Adaptive Dynamic Exit for Efficient LLM Reasoning (BLADE), a lightweight framework that dynamically terminates reasoning by estimating whether the generated prefix is sufficient for correct answering. BLADE constructs multi-granular checkpoints from sentence, self-doubt, and paragraph boundaries, and derives robust training labels through repeated answer completions. It further learns a compact subset of informative probe layers instead of relying on fixed choices or expensive representations from all layers. At inference time, calibrated predictions are combined with checkpoint-specific confirmation rules to balance responsiveness and premature-exit risk. Experiments on five benchmarks and two Qwen3 reasoning models show that BLADE preserves near-baseline accuracy while reducing generated tokens by 24.8% on Qwen3-8B and 15.8% on Qwen3-4B. Ablation studies further confirm the benefits of diverse checkpoints and automatic layer selection, demonstrating an effective approach to more efficient LLM reasoning. 

---
# FairFund-Bench: Evaluating Distributive Bias in LLM Resource Allocation 

**Authors**: Martin Lukk  

**Link**: [PDF](https://arxiv.org/pdf/2607.28934)  

**Abstract**: Large language models (LLMs) are increasingly involved in the distribution of scarce resources, raising concerns about biased allocations based on characteristics like race and gender. Recent LLM audits have produced inconsistent results, however, finding evidence of both positive and negative discrimination towards women and ethnic minorities, even for the same models. We show that this disagreement can arise from differences in audit format and introduce FairFund-Bench, a benchmark that systematically varies key features of previous audit designs: the evaluation task (rating, ranking, or allocation), comparison context (single or multi-stimulus), and whether the audit is transparent or disguised. The benchmark comprises 600 requests for financial assistance created from human-authored templates (calibrated against 1.3M real GoFundMe campaigns) across three domains, four race and two gender categories, and five causal framings of need derived from welfare deservingness theory. Across 14 models, audit format changes the direction of bias: models advantage minorities when rating claimants individually but penalize some groups when ranking them side by side. Bias magnitude, though small overall, is several times greater in disguised audits than in transparent ones, where, faced with appeals differing only in claimants' names, models overwhelmingly split funds equally. Causal framing effects, by contrast, exceed demographic effects by roughly an order of magnitude and are consistent across models and audit formats, indicating that current LLMs robustly reproduce human deservingness evaluations. The benchmark scores models on four criteria (demographic bias, deservingness alignment, cross-task consistency, and cross-context consistency), is publicly available, and can be readily adapted to other substantive domains. 

---
# Token-Level Diagnosis of Sycophancy in LLMs with Attribution-Guided Steering 

**Authors**: Hieu Nguyen, Mahammed Kamruzzaman, Anshuman Chhabra, Gene Louis Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.28906)  

**Abstract**: Sycophancy refers to the tendency for large language models (LLMs) to match user beliefs at the cost of factual correctness, thereby undermining model reliability. Prior work on evaluating sycophancy in LLMs aims to assess whether a model's output matches an authority's claim, but cannot reveal which part of the prompt drives this sycophantic behavior. To bridge this gap, we investigate the relationship of sycophantic responses with an authority's credentials, their assertive claim, and the problem statement. We introduce the Authority Share Index (ASI), an Integrated Gradients-based token attribution method, which measures the degree to which a model's decision is driven by authority-related text. Through extensive experiments across five models and 30 test configurations, we find that sycophantic responses consistently direct more attention toward authority tokens than resistant ones. Moreover, our token attribution method reveals that for the sycophantic cases, the claim asserted by the authority receives more attention than the authority's credentials. Building on these findings, we propose attribution-guided contrastive activation steering to mitigate LLM sycophancy. Our method constructs a steering vector from high-attribution tokens of sycophantic and resistant responses, selectively pushing models toward resistance. This enables inference-time steering without retraining, lowering sycophancy from 96% to 25% in the strongest case. Together, our results show that token-level attribution can both explain what drives sycophancy and directly inform a practical intervention. 

---
# TextCloak: Thwarting Unauthorized LLM Exploitation via RL-Driven Unlearnable Text 

**Authors**: Chengshuai Zhao, Pingchuan Ma, Dawei Li, Bohan Jiang, Zhiyuan Yu, Zhen Tan, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28862)  

**Abstract**: The rapid development of Large Language Models (LLMs) has led to significant advances across a wide range of language tasks, while simultaneously raising growing concerns about unauthorized data exploitation and privacy leakage. Unlearnable examples (UEs) offer a promising defense by introducing carefully designed perturbations into data such that models trained on them exhibit degraded utility. However, existing methods for text protection are primarily designed for classification tasks (e.g., sentiment analysis) in discriminative language models and often rely on injecting class-specific linguistic cues, which limits their effectiveness in the open-ended generation settings of LLMs. In this work, we propose TextCloak, an RL-driven framework for protecting textual data against unauthorized LLM exploitation. TextCloak employs a generative policy that transforms batches of clean text into unlearnable examples while preserving semantic fidelity and linguistic naturalness. To optimize the policy, we introduce GRPO-UE, which rewards generated unlearnable text based on the downstream degradation they induce in fine-tuned surrogate LLMs and updates the generator parameters via group-relative policy optimization. This bi-level optimization enables the generator to discover generalizable protective patterns beyond class-specific cues. Comprehensive experiments on six publicly available datasets and nine state-of-the-art LLMs demonstrate that TextCloak consistently impairs unauthorized fine-tuning while maintaining text utility for legitimate use. Further analyses establish its transferability and robustness across model architectures, training configurations, and adaptive attacks, highlighting its broad applicability as a practical defense against unauthorized LLM exploitation. 

---
# Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications 

**Authors**: Burak Payzun, İrem Demirtaş, Simona Scala, Elena Ferretti, Seçil Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2607.28840)  

**Abstract**: Large language models are increasingly deployed in financial applications that combine retrieval, proprietary data, tool use, orchestration logic, monitoring, and human escalation. Yet evaluation often remains model-centric: benchmark scores, task accuracy, or one-off qualitative reviews are treated as evidence of readiness. In financial settings, this is insufficient. We take the position that financial LLM systems should not be approved for production based on benchmark performance alone. They require system-level validation evidence across the application stack: data, model design, retrieval and generation performance, agent behavior, governance, and implementation. Drawing on industry experience validating GenAI applications in financial institutions, we outline a multi-layer validation view and explain why hybrid evaluation is necessary. We discuss where LLM-as-a-judge methods are useful and why they require controls such as multiple judges, rubrics, agreement, and auditability checks. We also highlight failure modes poorly captured by static benchmarks, including retrieval failures, unfaithful generation, tool misuse, escalation errors, and operational instability. Our position is that financial LLM validation should be an ongoing system discipline rather than a one-time model scoring exercise. Validation should produce decision-ready evidence, not only scores. We conclude with a research agenda for system-aware benchmarks, agent trace validation, judge alignment protocols, and lifecycle validation standards. 

---
# Rolling With Resistance: Preference-Optimized LLM Counselors Can Trade Goal Persistence for Relational Attunement in Motivational Interviewing 

**Authors**: Weiying Chen, Junlong Shen, Zhexuan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28814)  

**Abstract**: In Motivational Interviewing (MI), a client's sustain talk (arguments for the status quo) calls for the counselor to roll with resistance, a move that can fail in two opposite ways: capitulation (abandoning the change agenda to preserve rapport) or confrontation (arguing or directing, overriding the client's autonomy). We introduce a two-axis evaluation of counselor responses, anchored in the Motivational Interviewing Treatment Integrity (MITI) code, Goal Persistence (GP) and Relational Attunement (RA), yielding a four-quadrant framing in which rolling with resistance is high on both, and we ask whether penalizing one failure through preference optimization teaches rolling with resistance or provokes its opposite. From the expert-annotated AnnoMI corpus we build topic-disjoint Direct Preference Optimization data whose preference sets differ only in which failure is rejected, using on-policy negatives. An automatic judge, validated against AnnoMI's expert labels and rechecked by trained human coders, scores blind pairwise win-rates against each base under a firewall in which disjoint model families generate, label, and judge. Across three aligned instruction models spanning the Qwen and Llama families, penalizing confrontation reliably lowers goal persistence below parity, on every base and in every seed run, a robust cost, whereas the attunement gain is base-dependent, present on two of the three bases but absent on the third. Penalizing capitulation is inert, because these models rarely capitulate on-policy, so the trade is gated by each base's failure profile. A prompt-only control raises attunement without the goal-persistence cost, locating the cost in the optimization rather than in attunement itself. 

---
# Benchmarks Are Not Monolithic: Sample-Level Auditing and Orchestration for LLM Evaluation 

**Authors**: Philipp D. Siedler, Jordan Sassoon  

**Link**: [PDF](https://arxiv.org/pdf/2607.28801)  

**Abstract**: Benchmark datasets are central to evaluating Large Language Models (LLMs), yet they are typically conceived as monolithic tasks, obscuring substantial variation in the demands of individual samples. We introduce a dataset-centric meta-evaluation framework that audits benchmark datasets at the sample level along five latent dimensions: 1. Cognitive and Knowledge Demands, 2. Language and Content Quality, 3. Task Properties, 4. Context, and 5. Ethics, Safety, and Fairness. Applying this framework, we annotate five influential benchmarks -- MMLU, ARC, WinoGrande, HellaSwag, and TruthfulQA -- revealing pronounced internal heterogeneity that is not captured by aggregate accuracy scores. We show how these annotations enable criterion-driven orchestration of composite benchmark subsets across datasets, supporting targeted evaluation of model capabilities such as Reasoning Depth or Ethical Sensitivity. This approach reframes benchmark evaluation as dataset introspection, providing a principled methodology for analyzing and re-composing existing benchmarks to better reflect diverse evaluation needs. 

---
# Self-Supervised Skill Optimization 

**Authors**: Siran Peng, Cuiyu Yang, Tianyu Fu, Tianshuo Zhang, Haoyuan Zhang, Weisong Zhao, Anyang Su, Minghui Wu, Huiying Li, Xiangyu Zhu, Chenxu Zhao, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2607.28777)  

**Abstract**: Agent skills provide frozen large language model (LLM) agents with reusable procedural guidance, and recent work shows that such skills can be optimized with ground-truth (GT) feedback. Many applications, however, lack GT labels, task scores, rewards, or reliable task-specific evaluators. We therefore introduce Self-Supervised Skill Optimization (SSO), a comparative framework that learns a reusable skill from unlabeled task instances alone. At each step, SSO runs the current skill on an unlabeled batch, uses a subset of the resulting executions to generate complete skill probes, and runs the probes on the same batch. An LLM judge compares the resulting answers, trajectories, artifacts, or terminal states. A separate behavior extractor identifies behavioral differences without seeing the judge's decisions. SSO uses these decisions to aggregate evidence for and against the observed behaviors across instances. It then ranks the behaviors by the resulting evidence and renders a new complete skill from the highest-ranked behaviors. The update is accepted only if the new skill outperforms the current one on an unlabeled validation set. SSO outperforms existing GT-free prompt optimizers on both closed-ended and open-ended tasks. On closed-ended benchmarks, it approaches and sometimes exceeds the strongest GT-based skill optimizer without using any GT feedback. 

---
# The Morphological Core of Dungan: A Two-Dialect Finite-State Model and a Multi-Genre Evaluation 

**Authors**: Anton M. Alekseev, Sergey I. Nikolenko  

**Link**: [PDF](https://arxiv.org/pdf/2607.28766)  

**Abstract**: Dungan, a Sinitic language of Central Asia written in a Cyrillic-based script, is described in detail in the grammatical literature, yet the quantitative properties of its morphology in actual usage have, to the best of our knowledge, never been measured systematically. This paper uses a finite-state morphological analyzer as a measuring instrument. Implemented with HFST and covering both dialect groups (the Gansu variety, which is the literary standard, and the Shaanxi variety), the model offers no new grammatical description; it formalises the knowledge accumulated in Dungan studies and makes it measurable on corpora of three genres. Three results follow. Overt inflection is rare and limited: only 9.3% of recognized tokens in the encyclopaedic register have an overt marker, the system has just ten categories, and degree marking is almost absent. Ambiguity is genuine but sharply localized: 78.1% of tokens receive a single analysis, and the residue sits almost entirely on two clitics, -di (genitive/progressive) and -ni (locative/prospective). And the grammatical core proves effectively closed, the claim the instrument is really needed for: between 78% and 95% of the tokens the analyzer fails on, depending on register, are simply absent from the lexicon, and the phenomena the model deliberately declines to implement account for at most 4.5% of those failures. Held-out coverage (80 to 85%) is no lower than development coverage (73%), while a stem list with no morphology already reaches 67.4%, so the morphology is worth 5.2 points. The open frontier of Dungan is lexical. The analyzer, its sources and every evaluation script are released openly. 

---
# Demystifying Entropy-based Selection for Chain-of-Thought Compression in Large Reasoning Models 

**Authors**: Sara Candussio, Daniel Scalena, Luca Bortolussi, Elisabetta Fersini, Malvina Nissim, Gabriele Sarti  

**Link**: [PDF](https://arxiv.org/pdf/2607.28707)  

**Abstract**: Entropy-based pruning has been proposed as an effective method for compressing Chain-of-Thought (CoT) reasoning with negligible accuracy loss. We test the robustness of low- and high-entropy CoT step selection methods across various models and reasoning tasks, showing that entropy offers no advantage over random pruning in any evaluated setting. Moving from sentences to tokens, we then show that retaining low-entropy tokens seems effective only on mathematical benchmarks. We find this is due to the inherently low-entropy nature of numeric tokens, which also convey semantic content in such problems. Finally, we demonstrate that patching a subset of a few CoT tokens with their original activations recovers near-perfect full-trace performance, providing causal evidence that task information is not concentrated in a small set of CoT tokens identifiable by heuristics, but rather distributed across the full reasoning chain. 

---
# TELLER: Dual-Path Iterative Preference Optimization for Table Entity Linking 

**Authors**: Yixin Peng, Kehao Li, Stefan Decker  

**Link**: [PDF](https://arxiv.org/pdf/2607.28680)  

**Abstract**: Entity linking in tables matches short and ambiguous cell mentions to their corresponding knowledge-base entities. Existing approaches typically rely on data preprocessing pipelines that retain either compact or extensive table content as contextual evidence, and then formulate entity linking as a language generation task for instruction-tuned models; recent systems further incorporate explicit reasoning to disambiguate challenging mentions. However, their training supervision is usually static: fixed preference data cannot adapt to the residual errors of an evolving model, while variations in reasoning length can bias sequence-level preference learning. To address these limitations, we present TELLER: Table Entity Linking through Learning from Errors and Reasoning. We first retrieve and rank Wikidata candidates and retain reduced table evidence in the prompt. The direct-answer path applies iterative direct preference optimization and refreshes its preference data with residual errors from the updated model. The reasoning path uses filtered and compressed chain-of-thought rationales for supervised fine-tuning, followed by our iterative length-normalized regularized preference optimization. On the TableInstruct entity-linking subset, the direct-answer path improves accuracy from 94.35\% to 94.50\%; on the MammoTab V2 evaluation set, it improves accuracy from 87.59\% to 88.20\%. The reasoning path improves accuracy from 92.90\% to 92.95\% on TableInstruct and from 79.09\% to 81.85\% on MammoTab V2, while maintaining high rates of complete reasoning generation. These results show that iterative preference learning benefits both concise entity prediction and explicit reasoning. 

---
# The Checking Problem: What must be true before AI ships in a regulated firm 

**Authors**: Prerit Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2607.28666)  

**Abstract**: Enterprise AI programmes stall at a rate that is widely quoted and poorly explained. This paper measures the mechanism. Six document-heavy workflows of the kind performed daily in regulated financial services were run across four model families and three tool configurations, three times each, producing 5,093 scored output elements across 72 configurations. Each configuration was assessed twice: against a demonstration bar, being a single correct run on a single case, and against a production bar requiring sustained accuracy, reproducibility across repeats, verifiable attribution, and a confidence signal that carries information. 57 of 72 configurations cleared the demonstration bar and 32 cleared the production bar, a survival rate of 56.1%. The paper then computes the review burden each configuration imposes, estimated out of sample rather than with hindsight. A tool that states no confidence requires review of 100% of its output, because it offers a reviewer no basis for triage. Requiring the tool to cite its sources and state a confidence reduces that to 49% while holding the residual error tolerance in 17 of 20 configurations. Adding a self-verification pass costs 2.3 times the latency of the plain configuration, reaches 44%, and is the only configuration that fails to hold the error tolerance. The practical implication is that the value of an AI workflow is set less by how often it is right than by how much of it a human must still check, and that the second property is measurable and rarely measured. 

---
# Are the Financial Reasoning from LLMs Credible? A Real World Test over Long-Horizon Statements 

**Authors**: Xinke Tong, Xuanming Zhang, Tianyi Tang, An Yang, Jiatu Hu, Guojie Lin, Zhenzhen Shi, Lingfeng Zeng, Boyu Yang, Bing Zhao, Hu Wei, Lin Qu, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28661)  

**Abstract**: Do Large Language Models (LLMs) possess genuine structural reasoning, or merely rely on surface-level pattern matching? The financial domain, demanding numerical precision and multi-step logic over long contexts, is an ideal testbed. Existing benchmarks fail to capture real-world industrial complexity, predominantly relying on multiple-choice questions or single-hop QA over cropped tables while ignoring intricate cross-statement dynamics and temporal de-cumulation.
To bridge this gap, we introduce FinIndices, a large-scale benchmark evaluating data-processing fidelity over uncropped financial statements (up to 32K tokens). Utilizing an automated synthesis pipeline with adversarial traps, FinIndices encompasses Single-Index computation and Table-Index tabulation to test complex domain, temporal, and caliber reasoning.
Our evaluation reveals two severe LLM vulnerabilities. First, a "Knowledge Bottleneck": despite memorizing formulas during pre-training, models demonstrate fragile pattern matching. Removing explicit formula hints causes performance to collapse (e.g., Gemini-3.1-Pro drops from 70.70% to 38.22% on table tasks), exposing fatal flaws in temporal de-cumulation and stock-flow caliber mismatch. Second, a "Structural Bottleneck": the intense cognitive load of generating multi-metric, multi-period tables actively drains reasoning capacity. Under structural pressure, LLMs that flawlessly execute isolated derivations regress to shallow heuristics, such as fetching incorrect adjacent columns or substituting deep accounting adjustments with lazy literal arithmetic. Finally, Supervised Fine-Tuning (SFT) yields substantial zero-hint gains (+8.54% Single, +3.82% Table), validating that structured logic can be partially restored via data-centric alignment. 

---
# Evaluating Federated Pre-Training: On the Reliability of Downstream Fine-Tuning and Intrinsic Evaluation 

**Authors**: Claudia Grosser, Maike Heuer, Denis Krompass, Thomas A. Runkler  

**Link**: [PDF](https://arxiv.org/pdf/2607.28658)  

**Abstract**: Federated pre-training offers a way to train foundation models on private or distributed data without centralizing the underlying datasets. However, evaluating federated pre-training remains challenging because differences in client participation and local data availability can make directly comparable evaluation difficult. Moreover, pre-training test perplexity is tied to the pre-training distribution, while downstream benchmarks introduce task-specific adaptation that may not faithfully reflect the test perplexity established during pre-training. In this work, we study which evaluation protocol more reliably reflects federated pre-training quality. Using a controlled set of centralized and federated-trained models of a 16M parameter transformer model trained on identical client data, we assess evaluation protocols by whether they preserve a reference ranking established on the same pre-training testset. We compare downstream fine-tuning on GLUE, including full, head-only, and reduced-data variants, with next-token prediction on GLUE text as an intrinsic evaluation signal. Our results show that downstream fine-tuning does not reliably preserve the pre-training ranking, whereas direct next-token prediction exhibits a strong correspondence with the pre-training test perplexity. These findings suggest that downstream fine-tuning alone can be misleading when comparing federated pre-trained models, and that evaluation signals closer to the original pre-training objective deserve greater attention. 

---
# The Formalism Trap: Are LLM-as-a-Judge Evaluators Blinded by Consensus Mimicry under Social Load? 

**Authors**: Dahlia Shehata, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.28641)  

**Abstract**: We introduce the \textit{Agentic Formalism Trap} and the Evaluative Dissonance Index ($D_E$), quantifying how LLM-as-a-Judge systems conflate structural proceduralism with semantic truth under adversarial load. Analyzing 22,500 trajectories across 3 domains (GAIA, SWE-bench, Multi-Challenge), we extract a semantic taxonomy of hallucination maneuvers, validated via deterministic lexical grounding ($p < 10^{-120}$). A logistic meta-evaluator isolates the exact syntactic triggers of this evaluator capture (ROC-AUC 0.8779), while a zero-shot Leave-One-Domain-Out transfer proves the vulnerability is universally domain-agnostic (mean ROC-AUC 0.7482). Architectural profiling reveals that distinct simulated swarm topologies induce mathematically disparate semantic blind spots, proving that unanchored closed-loop evaluation is unstable, systemically divergent and necessitates architecture-specific vigilance filters. 

---
# TokenSwap: Benchmarking and Reducing the Modality Gap in Multimodal LLMs 

**Authors**: Andong Hua, Colton Bishop, Igor Mordatch, Arian Hosseini, Jindong Gu, Aleksandra Faust, Rebecca Roelofs, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.28640)  

**Abstract**: Multimodal large language models (MLLMs) should generate consistent responses given semantically equivalent inputs across modalities. However, we observe a systematic discrepancy in model predictions under such cross-modal variations. Specifically, we define the modality gap as the difference in model performance under semantically equivalent textual and multimodal inputs. We introduce TokenSwap, a method that constructs such inputs by replacing textual concepts with semantically aligned images, resulting in sequences where visual tokens are interleaved with text tokens. Based on TokenSwap, we transform existing text-based benchmarks such as MMLU into image-interleaved counterparts, resulting in TokenSwap-Bench. Across 42 MLLMs, we observe a pervasive modality gap, with performance decreasing by 4.2% to 47.4% when moving from text-only to image-interleaved inputs, averaging 19.6% +/- 3.3% across models. Notably, we observe that reasoning models exhibit consistently smaller gaps, achieving an average gap of 10.1% compared to 25.5% for non-reasoning models. In contrast, neither prompting strategies nor scaling training compute alone reliably reduces the modality gap. Finally, we demonstrate that incorporating TokenSwap during training effectively mitigates this gap while preserving strong text-only and vision-language performance. 

---
# The Asymmetric Effects of Knowledge Distillation on Bias in Small Language Models 

**Authors**: Plawan Kumar Rath  

**Link**: [PDF](https://arxiv.org/pdf/2607.28639)  

**Abstract**: We show that knowledge distillation in small instruction-tuned language models has asymmetric effects on bias. On unambiguous tasks (BBQ-disambig), response-based distillation from a Gemma-2-9B teacher improves context-following: for the most biased baseline (SmolLM2-1.7B-Instruct), it cuts the context-overriding error rate from 44% to 24%. On ambiguous tasks (BBQ-ambig), the same distillation destroys per-item refusal calibration: 15% of items where the baseline correctly abstained instead receive stereotype answers, even when overall refusal rate is preserved. The pattern reproduces on a second student family (OLMo-2-1B-Instruct), with silence-loss of 8% and filled-silence accounting for 89% of new bias. Across the full 28-configuration grid, the magnitudes of silence-loss and filled-silence are uncorrelated (Spearman $\rho=0.19$, n.s.), indicating that the two effects arise from distinct mechanisms.
Aggregate stereotype metrics (CrowS-Pairs, overall BBQ Stereotype Reliance Score) average over both effects and conceal the per-item harm. We trace the calibration loss to a data-side mechanism: an audit of four training corpora finds <0.5% refusal-as-answer-shape. Supervised fine-tuning (SFT) with refusal injection either breaks parsing or over-corrects into a trivial-refuser regime (refusal rate 99.8%, disambig accuracy 0.2%) that aggregate metrics would call perfectly calibrated. We propose Per-Condition Calibration Diagnosis (PCCD), a three-step protocol that evaluates refusal calibration, context-following, and capability preservation. PCCD catches both the asymmetric harm and the trivial-refuser failure mode that aggregate evaluations miss. 

---
# Learning Stateful Predictive Knowledge From Experience 

**Authors**: Yan Song, Xidong Feng, Bo Liu, Xinyu Cui, Haotian Fu, Zichen Liu, Mengyue Yang, Cheng Deng, Jian Zhao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28638)  

**Abstract**: As large language model (LLM) agents increasingly learn from experience, they primarily rely on trajectory-level reflection to extract insights. Viewed through the lens of predictive knowledge, we argue that this approach operates on episodic hindsight rather than predictive foresight, yielding brittle, path-dependent heuristics. To address this, we propose Stateful Knowledge Learning (SKL). SKL shifts the agent's focus from trajectory-level summarization to maintaining Stateful Knowledge: explicit, declarative predictive assessments anchored to state. We first demonstrate a motivating example showing how stateful knowledge provides granularity, enhances generalization, and enables knowledge bootstrapping. To further scale up the idea, we introduce two algorithms via self-distillation (SKL-SD) and reinforcement learning (SKL-RL), training agents to autonomously extract state-grounded predictive knowledge from experience and learn to leverage it for policy making. Experiments on interactive environments (WebShop, ScienceWorld) and a complex reasoning task (ChessPuzzles) demonstrate that equipping models with the inherent ability to learn stateful predictive knowledge significantly outpaces current reflection-based training paradigms. 

---
# ZeroR@CHiPSAL 2026: Two-Stage Vision-Language Adaptation with Contrastive Learning for Nepali Meme Classification 

**Authors**: Nitiz Khanal  

**Link**: [PDF](https://arxiv.org/pdf/2607.28637)  

**Abstract**: This paper presents our system for the CHiPSAL 2026 shared task on multimodal hate speech and sentiment detection in Nepali memes. We address both subtasks: binary hate speech classification and three-class sentiment analysis. Our approach adapts the Robust Adaptation of Hateful Meme Detection (RA-HMD) framework using Qwen3-VL-8B-Instruct, a state-of-the-art vision-language model with native Devanagari support. We employ a two-stage training pipeline: (1) LoRA fine-tuning with an MLP projection head for generative classification, and (2) contrastive backbone fine-tuning with supervised InfoNCE loss. We handle class imbalance through minority oversampling, image augmentation, and focal loss. At inference, we ensemble Stage 1 token probabilities with Stage 2 classifier scores using validation-tuned weights. Our end-to-end approach eliminates error propagation from separate OCR and translation pipelines by leveraging the model's native Devanagari understanding. Our system achieved \textbf{2nd place} on hate speech detection (F1: 0.797) and \textbf{4th place} on sentiment analysis (F1: 0.518). We provide detailed ablations, error analysis, and insights into adapting large vision-language models for low-resource South Asian languages. 

---
# Chain-of-Models: Cross-Model Auditing for Bias-Robust LLM Judges 

**Authors**: Qian Wang, Zhanzhi Lou, Zhenheng Tang, Nuo Chen, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2607.28636)  

**Abstract**: LLMs increasingly serve as automated judges, but their judgments remain vulnerable to cognitive biases. Existing mitigations mostly rely on prompt-driven debiasing, which is brittle across bias types, or human evaluation, which does not scale. We study \emph{Chain-of-Models} (CoM), an automated audit pipeline in which a second model inspects the first model's reasoning trace before producing the final judgment. The key design question is whether the auditor should be the same model, a same-family model, or a different-family model. Across 9 models from 6 families, 4 cognitive biases, and 4 factual datasets, we find that auditor identity matters in two ways. First, standalone bias resistance does not predict audit effectiveness: Kimi-K2.5 is the strongest standalone model on several biases, yet is a weak auditor for Qwen2.5-72B's biased traces. Second, the best auditor is bias-specific: GPT-4o is strongest on bandwagon, authority, and distraction, while GLM-5 is strongest on sycophancy. We operationalize these findings with a per-bias auditor selection rule that, given the bias type, scores candidates along functional diversity, per-bias standalone resistance, and calibrated audit effectiveness. Under a calibration/test split, the selector reaches the highest accuracy across the four biased slices ($0.884$ vs.\ $0.824$ for the strongest single fixed auditor and $0.805$ for the no-audit baseline). We release data, configurations, and an LLM-agent skill at this https URL . 

---
# Imbalanced Data Clustering via Targeted Data Augmentation Using GMM and LLM 

**Authors**: Noor Khalal, Abdallah Alaa-Eddine Djamai, Imed Keraghel, Mohamed Nadif  

**Link**: [PDF](https://arxiv.org/pdf/2607.28635)  

**Abstract**: In Natural Language Processing (NLP), dealing with underrepresented topics is challenging, especially in unsupervised tasks where clustering might not adequately capture minority topics. To tackle this challenge, our paper presents a novel unsupervised data augmentation method that integrates Gaussian Mixture Models (GMMs) and Large Language Models (LLMs). Due to their flexibility and robustness, GMMs can detect clusters corresponding to underrepresented areas in the data, while LLMs create synthetic documents to enrich these clusters and improve their representation. Experiments on various imbalanced text datasets demonstrate that our approach preserves clustering performance in all cases and often enhances cluster interpretability, offering a robust and scalable solution for improving data representation in unsupervised NLP tasks. 

---
# Can LLMs Really Understand Item Difficulty Levels? Implications for Automated Item Generation Using LLMs 

**Authors**: Xinyi Wang, Hong Jiao, Ming Li, Sydney Peters, Hanna Choi, Tianyi Zhou, Qingshu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28634)  

**Abstract**: The estimation of item difficulty plays a key role in both formative assessment and large-scale high-stakes summative assessments. This study explores how large language models (LLMs) perform in predicting item difficulty levels using items from a large-scale Reading and Writing test. The study investigated various prompting strategies and parameter settings across multiple LLMs. LLM performance was compared with encoder-only language models and feature-based supervised machine learning models. Zero-shot GPT-4.1 with a temperature of 0 yielded the highest item difficulty level prediction accuracy, with a quadratic weighted kappa (QWK) of 0.578. However, LLMs' prediction accuracy was lower than that of ConvBERT (QWK = 0.625), which outperformed the best feature-based supervised machine learning model. Further analysis showed that all LLMs struggled to label hard items; in particular, the current advanced GPT-5.4 tended to underestimate item difficulty levels. Dimension reduction of embeddings showed that item embeddings from different difficulty levels were mixed together, indicating that semantic information from items alone is likely insufficient for item difficulty level prediction. The findings suggest that if LLMs cannot understand item difficulty levels as evidenced by empirical data and tend to treat most items as easy when their own capabilities increase, caution should be exercised when using LLMs to generate items with targeted difficulty levels. 

---
# WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning 

**Authors**: Senyu Fei, Xiaopeng Yu, Siyin Wang, Xianzhong Zhao, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.29613)  

**Abstract**: Reinforcement learning (RL) post-training of Vision-Language-Action (VLA) models has shown strong promise for robotic manipulation. Among RL methods, critic-based approaches rely on a value estimator that predominantly operates on single-frame observations or single-frame VLM backbone latents, which is a fundamental mismatch with the partially observable nature of robot control. A naive approach to incorporate observation history into the critic incurs exponential complexity with high-dimensional visual space, and still fails because pure scalar-return regression provides insufficient supervision for learning cross-temporal dynamics. We identify the root cause as a state approximation problem: without an explicit world modeling objective, the critic's representation cannot capture the temporal structure needed for accurate value estimation. To address this, we propose the World Critic Model (WCM), built on a lightweight LeJEPA architecture; WCM jointly predicts future latent state and estimates values, such that the critic's representation is explicitly trained to capture temporal dynamics rather than merely regress scalar returns. WCM integrates seamlessly into both on-policy and off-policy training pipelines and is compatible with state-of-the-art VLA backbones including Pi0, Pi0.5, and OpenVLA-OFT. Extensive experiments on 149 tasks across four benchmarks demonstrate that WCM consistently achieves state-of-the-art performance in both in-distribution and out-of-distribution settings, with particularly strong generalization gains. We further validate WCM on seven real-world manipulation tasks using OpenVLA-OFT and Pi0.5 with off-policy RL, confirming stable deployment across diverse settings. 

---
# Bridging the Question-Answer Gap in Retrieval-Augmented Generation: Hypothetical Prompt Embeddings 

**Authors**: Domen Vake, Jernej Vičič, Aleksandar Tošić  

**Link**: [PDF](https://arxiv.org/pdf/2607.29402)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems synergize retrieval mechanisms with generative language models to enhance the accuracy and relevance of responses. However, bridging the style gap between user queries and relevant information in document text remains a persistent challenge in retrieval-augmented systems, often addressed by runtime solutions (e.g., Hypothetical Document Embeddings (HyDE)) that attempt to improve alignment but introduce extra computational overhead at query time. To address these challenges, we propose Hypothetical Prompt Embeddings (HyPE), a framework that shifts the generation of hypothetical content from query time to the indexing phase. By precomputing multiple hypothetical prompts for each data chunk and embedding the chunk in place of the prompt, HyPE transforms retrieval into a question-question matching task, bypassing the need for runtime synthetic answer generation. This approach does not introduce latency but also strengthens the alignment between queries and relevant context. Our experimental results on six common datasets show that HyPE can improve retrieval context precision by up to 42 percentage points and claim recall by up to 45 percentage points, compared to standard approaches, while remaining compatible with re-ranking, multi-vector retrieval, query decomposition, and other RAG advancements 

---
# Language Models Agree With Each Other, Not With Readers 

**Authors**: Kazuki Nakayashiki, Keisuke Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2607.29274)  

**Abstract**: Claims that language models homogenise are usually measured against human judgements collected for the study, which makes the human side an artifact of the design: a crowdworker given the model's instruction is running the model's prompt. We measure convergence against a human reference nobody built for the purpose -- 2,523 reader mark sets across 120 web documents, produced by people highlighting for their own reasons on a platform where the overlay of others' marks is off by default.
Agreement is the overlap between two size-matched sentence sets minus the overlap expected when each is resampled within its own depth-and-length bands. The null's calibration is demonstrated, not asserted: every pair involving a random baseline lands within 0.006 of zero. On the median document each party names 14 sentences of 70; two readers share 4.1 and two models 8.7.
Across 18 model arms spanning 11 vendors, 3 countries and both weight regimes, the median of 153 model pairs is +0.093 against a human yardstick of +0.040, and 99 sit entirely above the human interval. Two frontier models from rival labs reach +0.203, twice what GPT-4o agrees with itself on a second call. The effect is not determinism, prompt wording, procedure, vendor or routing, and it is graded: the smallest models agree at the human level. No model agrees with readers detectably more than a reader does, and at equal depth and length no surface feature separates their choices.
The multiples are procedure-dependent and the ordering is not: models are cut to their sharpest set while a reader's is a random draw from what they marked, and blunting the models alike halves the gap without closing it. Tested out of sample on four models released after this analysis, against predictions fixed beforehand, none clears the human interval. A population simulated from several models is not several populations. 

---
# RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems 

**Authors**: Haoran Ling, Yuecheng Li, Zeyu Song, Jing Yao, Shuwen Kang, Chi Lu, Wenjin Wu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.29241)  

**Abstract**: Optimizing modern recommender models still depends heavily on engineers manually iterating over architectural, objective, and training-strategy changes. While LLM-based agents can automate this trial-and-error process, allowing the LLM to both select modification directions and generate concrete hypotheses often leads to unstable search under limited experiment budgets. Inspired by the above challenge, we propose RecHarness, a Bandit-Routed Agentic Harness for automated recommender model optimization. RecHarness separates the optimization process into two steps: a bandit router selects the next modification direction according to historical validation feedback, while the LLM generates a concrete optimization hypothesis and executable code edit within the selected direction. To sustain long-horizon exploration, RecHarness uses a jump-basin mechanism to activate a structural-jump arm when local edits stagnate. Across multiple recommendation tasks, datasets, and model backbones, RecHarness achieves more stable performance improvements and uses limited trial budgets more effectively than LLM-reasoning search. During a 7-day online A/B test on a large-scale short-video advertising platform, the selected candidate improves ADVV by 2.084%, Revenue by 0.534%, and Exposure by 0.559%. Code is available at this https URL. 

---
# TransMem: Transforming Hidden States into Memory for Large Language Models 

**Authors**: Haodong Lei, Junming Liu, Yirong Chen, Pinlong Cai, Botian Shi, Ding Wang, Hongsong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.29032)  

**Abstract**: Large language model (LLM) agents increasingly operate over long interaction histories, where effective reasoning requires identifying and exploiting task-relevant evidence distributed across past observations and actions. However, useful information encoded in previously computed representations is often underutilized during subsequent generation. We propose \textbf{TransMem}, a lightweight inference-time parametric memory module that transforms sparse historical hidden states from a frozen LLM backbone into reusable memory representations. TransMem uses a lightweight gating network to dynamically apply the latent intervention to the current hidden states, without repeatedly encoding the preceding context. To learn transferable memory utilization rather than task-specific knowledge, we introduce evidence-conditioned self-distillation. A memory-augmented student processes the full context and matches the predictive distribution of an evidence-only teacher that shares the same frozen backbone. Experiments on LoCoMo, HotpotQA, and MemoryAgentBench demonstrate consistent improvements across different model architectures and scales. TransMem yields gains of 11.58--29.25 $F_1$ on LoCoMo and 10.20--13.03 $F_1$ on HotpotQA, while improving the average MemoryAgentBench accuracy from 29.54\% to 40.00\%. These results establish sparse historical hidden states as an effective and efficient memory substrate for long-context LLM agents. Our code is available at this https URL. 

---
# GoldenRetriever: Non-Interactive Homomorphic Encrypted Retrieval for Privacy-Preserving RAG 

**Authors**: Yang Gao, Gang Quan, Scott Piersall, Qian Lou, Dongdong Wang, Liqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.29019)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models by incorporating external knowledge, but existing pipelines typically operate on plaintext data, raising significant privacy concerns. Prior work on privacy-preserving retrieval leverages cryptographic techniques such as homomorphic encryption (HE) and private information retrieval (PIR), but often relies on interactive protocols or ranking-based selection mechanisms that incur high latency and potential information leakage. In this paper, we propose a practical non-interactive encrypted retrieval framework for RAG based on threshold selection. Instead of performing expensive top-$k$ ranking under encryption, our approach selects documents whose similarity scores exceed a predefined threshold, reducing computational complexity from quadratic to linear in the corpus size. We implement this design using CKKS-based homomorphic computation, enabling fully encrypted similarity evaluation and document selection without revealing query content, intermediate scores, or selected indices. To bridge the gap between approximate encrypted computation and discrete token reconstruction, we introduce a precision-stable mask polarization method that ensures accurate recovery of selected documents. Experiments on standard retrieval benchmarks demonstrate that our approach achieves competitive retrieval effectiveness while significantly reducing latency compared to ranking-based encrypted methods. These results highlight threshold-based selection as a practical foundation for scalable and secure RAG systems. 

---
# Adjudicated Captioning: Multi-Agent Alignment Scoring and Consensus-Distilled Beam Arbitration for Strict Zero-Shot Image Captioning 

**Authors**: Duy Tran Thanh, Thien-Phuc Doan, Long Nguyen-Vu, Ngo Tan Vu Khanh  

**Link**: [PDF](https://arxiv.org/pdf/2607.28986)  

**Abstract**: Zero-shot image captioning (ZIC) describes images without paired image-caption supervision during captioner training, relying on text-only corpora and frozen pretrained image-text scorers. Existing retrieval-augmented methods score image-text alignment once, at retrieval, then commit the captioner's autoregressive beam under language-model probability alone, leaving the decoder without further visual grounding feedback. Progress has stalled, with no method improving on the strict-regime best since 2024.
We propose Adjudicated Captioning, an inference-time multi-agent framework that restores grounding feedback at multiple checkpoints over an unchanged IFCap captioner. First, we install a stronger frozen Retrieval Encoder at the input. Second, between retrieval and decoding we insert a frozen Cross-Attention Verifier that re-ranks the top-9 retrievals to top-5. Third, at the output beam we attach a learned Reranker pairing TriFuse, a multilayer perceptron, with MemAttend, a memory-attended transformer, the pipeline's only learned components; both are trained self-supervised by Borda-consensus distillation across the three frozen scorers, using no paired image-caption labels and no reference captions.
Under the inductive headline protocol, with rerankers fit on the disjoint COCO Karpathy validation beam and applied frozen to test, the framework reaches CIDEr 117.6 and SPICE 21.9 on COCO Karpathy, up from 108.0 and 20.3 for IFCap, a +9.6 CIDEr gain, and +7.7 above NES, the strongest synthetic-image-augmented method at 109.9, without retraining the captioner. A training-free fixed-fusion baseline reaches 115.8 CIDEr, so +7.8 of the +9.6 gain comes from the non-learned architectural intervention and the remaining +1.8 from the learned rerankers. The same recipe transfers off-COCO without captioner retraining: +8.1 CIDEr on Flickr30k Karpathy and +5.7 on NoCaps overall. 

---
# TORUS: A Test of Rendering-Understanding Self-Coherence for Unified Audio Models 

**Authors**: Aryan Vijay Bhosale, Harshit Rajgarhia, Abhishek Mukherji, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2607.28896)  

**Abstract**: Unified audio models capable of audio understanding, audio generation and, increasingly, audio editing are proliferating rapidly. Yet a basic question about them remains unanswered: do the two heads of a unified model agree about the same audio? Current practice evaluates each capability in isolation on specialized benchmarks, and never asks whether a model can make sense of its own generations. We present TORUS, the first self-coherence test for audio-native unified models. TORUS comprises 48 three-stage self-coherence tests carrying 432 six-option questions spanning speech, sound and music across five task families. We holistically evaluate five open unified models alongside a Cascaded Baseline that combines state-of-the-art specialized generation, editing and understanding models. The best unified model answers 50.5% of questions against the Cascaded Baseline's 63.2% and a 16.7% chance floor. Models struggle on audio editing. Among the evaluated audio models (specialized and unified), we observe limited self-coherence, and thus position self-coherence as an essential test for future audio systems. 

---
# Best Friends, Not Forever: Evaluating Long-Horizon Persona Collapse and Behavioral Drift in AI Companions 

**Authors**: Pranav Narayanan Venkit, Akshara Prabhakar, Yu Li, Daniel Lee, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28818)  

**Abstract**: As AI companions increasingly mediate repeated social interaction, users may rely on a stable role and shared history, yet locally acceptable replies do not ensure that either persists. We study two observable long-horizon failures: 'persona collapse', the loss of a deployed role, boundaries, values, or style, and 'behavioral drift', the gradual or recurrent erosion of those properties. We introduce ANCHOR, a controlled synthetic audit that separately measures persona enactment and trajectory recall. The study contains 2,008 conversations spanning 27 personas, nine interaction schedules, three generated memory settings, and four evaluated models. The Identity Probe combines a sealed 102-item questionnaire with turn-level judgments, while the Trajectory Probe scores 110 calibrated counterfactual questions from 35 conversation banks. Our results show that no evaluated model and configuration reliably preserves either dimensions: trajectory accuracy averages only 44.4%, user-state recall remains near four-option chance, and no tested context condition or memory consistently resolves these failures. Questionnaire retention also varies by model and persona facet, disagrees with turn-level behavior, and is sensitive to evaluator choice. These results indicate that current systems do not yet reliably support long-horizon companion continuity and that audits must distinguish persona enactment, trajectory recall, evaluator provenance, and deployment context rather than collapse them into a single trust or stability score. 

---
# SciToolAgent-Evo: An Ontology-Aware Self-Evolving Agent for Open-World Scientific Tool Acquisition 

**Authors**: Yuqi Tang, Chenyi Zhou, Libin Wang, Keyan Ding, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28692)  

**Abstract**: Large language model (LLM) agents have been increasingly adopted in scientific research for organizing and invoking specialized computational tools. However, their reliance on predefined tool spaces with static semantics limits their applicability to open-world scientific workflows, where tool requirements, capabilities, and boundaries evolve dynamically. To this end, we propose SciToolAgent-Evo, an ontology-aware self-evolving agent for open-world scientific tool acquisition. Driven by an evolving memory of skills, experiences, and an ontologized tool graph, it distills generalizable knowledge from contrastive trajectories during accumulation, whereas during inference, it formulates active requests and utilizes a LinUCB-based bandit gate to dynamically balance exploration and exploitation. Once a novel tool is acquired, its scientific ontology is completed online for seamless integration into the known graph. Moreover, we introduce OpenSciToolBench, a benchmark containing 900 realistic tasks across four difficulty levels. Extensive evaluations show that SciToolAgent-Evo achieves state-of-the-art performance, validating its robustness and generalization. 

---
# How Hard Does It Think? Analyzing Step-Aware Reasoning Energy in LLM Chain-of-Thought Trajectories 

**Authors**: Hui Wei, Junda Wu, Sheldon Yu, Sizhe Zhou, Yizhu Jiao, Ming Zhong, Bowen Jin, Tong Yu, Shijia Pan, Jiawei Han, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2607.28674)  

**Abstract**: Understanding how computational effort is allocated across individual chain-of-thought (CoT) reasoning steps remains an open challenge: existing interpretability methods rely on output-level signals or collapse processing depth into a single trajectory-level scalar, leaving step-wise effort opaque. We propose Step-Aware Reasoning Energy (SARE), a geometric framework that quantifies effort at the granularity of individual CoT steps via Centered Kernel Alignment (CKA) between Gram matrices of token hidden states across adjacent transformer layers, capturing inter-token relational structure without requiring eigenvector alignment or cluster correspondence. SARE further contextualizes this energy within reasoning's semantic progression by modeling CoT trajectories as transitions among latent semantic states. Across six reasoning benchmarks and three open-weight LLMs, we find that reasoning energy is highly non-uniform across step types, exhibiting phase-like transitions invisible to trajectory-level metrics; incorrect trajectories show systematically lower energy at critical reasoning junctions; and SARE-based features match or outperform output-based confidence baselines in most settings, indicating that internal geometric dynamics encode predictive information beyond surface-level signals. 

---
# Measuring Cognitive Engagement in Collaborative Discourse with an Extended ICAP Framework: Comparing Human Annotation, In-Context Learning, and Reflective LLM Agents 

**Authors**: Lan Anh Do, Hanling Jiang, Shuchin Aeron, Ayanna K. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2607.28651)  

**Abstract**: Collaboration supports learning and problem-solving, but its effectiveness depends on cognitive engagement during discourse. This study applies an extended 7-point ICAP framework based on the Interactive, Constructive, Active, and Passive modes to characterize variation in cognitive engagement during collaborative dialogue. Engagement was coded by trained human annotators and compared with large language model (LLM)-based labeling approaches, including in-context learning (ICL), zero-shot prompting, and self-reflective agents. Interrater reliability among human annotators was robust across framework refinement stages (kappa = 0.906-0.998), higher than the moderate agreement observed for ICL-based annotation (kappa = 0.541-0.609). The human-refined framework improved agreement among human annotators (Delta kappa = 0.10), but produced only modest gains for ICL-based LLMs (Delta kappa less than 0.04). Agent-refined frameworks improved cross-model agreement but remained below the human-refined framework. These findings highlight the promise of agent-based approaches and the importance of continued interaction between theory-guided human annotation and LLM-based methods in future work. 

---
# Why It Hurts: Identifying the Drivers of Negative Thoughts in Emotional Support Conversations 

**Authors**: Hainiu Xu, Zhaoyue Sun, Hanqi Yan, Jinhua Du, Caroline Catmur, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2607.28648)  

**Abstract**: Large Language Models (LLMs) are increasingly used for emotional support tasks, such as negative thought reframing. This task relies on modifying cognitive appraisals, the subjective interpretation of events that elicit negative emotions, which is typically conceptualized along multiple discrete dimensions. Current LLM-based frameworks model cognitive appraisal by exhaustively evaluating all possible dimensions, but they fail to account for the varying saliency of these dimensions across different contexts. In this work, we investigate a vital yet overlooked question: "Can LLMs infer the salient appraisal dimensions from emotional support conversations?" To address this question, we introduce the AppraiSal benchmark, containing 996 emotional support conversations with human-annotated mental states, including salient cognitive appraisal dimensions. Furthermore, we propose PRISM, a multi-agent probabilistic framework grounded in Bayesian Inverse Planning, designed to improve LLMs' ability to identify context-specific appraisal dimensions. Experimental results show that PRISM brings improvements to LLMs across various sizes, particularly in identifying the most salient appraisal dimensions. 

---
# To Facilitate or not to Facilitate: Human and LLM Facilitator Tendencies in Online Discussions 

**Authors**: Dimitris Tsirmpas, Katerina Korre, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2607.28643)  

**Abstract**: Automating facilitation in online discussions is a long-standing social concern given the increasing time we spend on online spaces and the failure of content moderation approaches. While studies have been conducted on how to facilitate, none have answered the essential question of when to do so. A potential answer is using LLMs, which ostensibly make automated, large-scale intervention increasingly feasible. In this study, we examine when LLMs decide to facilitate by defining what facilitation is, observing when humans decide to facilitate, and comparing their decisions with those made by LLMs. To this end, we create PEFK, a corpus standardizing and aggregating all relevant facilitation datasets. We are the first to run a survey on facilitation timing, which we execute using expert facilitative participants and LLM-as-a-judge models. We discover that while humans are more cautious, LLMs are excessively eager to facilitate, although both are more certain when judging that facilitation is not needed. We then investigate whether this behavior can be corrected using alternative setups for LLMs and training ModernBert classifiers on established datasets, finding that the latter perform more reliably than the former, although current datasets impose a relatively low performance ceiling. 

---
# ThinkReset: Learnable Intermediate Interface Construction for Bounded-Context Long-Horizon Reasoning 

**Authors**: Fei Ding, Yongkang Zhang, Runhao Liu, Yuhao Liao, Zijian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2607.28642)  

**Abstract**: Long chain-of-thought reasoning improves performance on complex problems, but it also introduces redundancy accumulation, context overflow, and error anchoring. We argue that under bounded context windows, the core bottleneck is not trajectory compression or test-time control, but the absence of a reusable intermediate interface that can replace discarded history and support continued solving. We further identify a key failure mode of outcome-reward-driven long-chain reinforcement learning: when the model has not solved the task before the window is nearly exhausted, the final-answer reward encourages premature guessing rather than continued careful reasoning. We propose ThinkReset, a text-space instantiation of this view. ThinkReset explicitly constructs reusable intermediate interfaces through interface writeback and reset, and directly optimizes post-reset continuation success. Across multiple long-horizon reasoning benchmarks, this perspective consistently improves success rates under fixed context windows. 

---
# Can AI Evaluate AI Scientists? A Benchmarking Study of Autonomous Research Generation Systems Using Automated Multi-Model Review 

**Authors**: Vaibhava Lakshmi Ravideshik, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2607.28631)  

**Abstract**: AI Scientist systems capable of autonomous research have the potential to significantly accelerate scientific discovery. However, evaluating and comparing the quality of AI-generated papers remains an open challenge. We propose and implement a rigorous benchmarking protocol using an automated peer-review system that harnesses frontier large language models to assess scientific papers across four core dimensions: originality, scientific rigor, clarity, and significance. We evaluate four leading AI Scientist frameworks: \textit{Sakana AI (v1 & v2)}, \textit{CycleResearcher}, and \textit{Data-to-Paper}. Each framework was run on a consistent set of 15 research proposals published by a commercial autonomous AI scientist company (FARS), generating 60 papers that we evaluate alongside 15 FARS benchmark papers. Using three independent LLM reviewers (GPT-5.4, Gemini, and Claude), we find that FARS benchmark papers significantly outperform all competing frameworks, achieving mean scores of 2.14--2.47 on a 1--5 scale compared to 1.00--1.87 for other systems. Notably, FARS scores are more than 2$\times$ higher than the next-best systems on Gemini and Claude evaluations. We find strong agreement among Gemini and Claude ($\rho$ = 0.907, $p < 0.001$), and both correlate extremely strongly with the synthesis score ($\rho$ = 0.961, $p < 0.001$), validating the reliability of automated evaluation. However, GPT-5.4 exhibits weaker agreement ($\rho \approx 0.32$), suggesting it evaluates papers using different criteria. These results establish the first quantitative benchmark for AI Scientist systems and demonstrate that multi-model LLM evaluation provides a scalable, consistent framework for assessing autonomous research quality. 

---
# Evidence-Ledger Adjudication for Claim-Evidence Traceability 

**Authors**: Gengyu Chen, Yongjie Yu, Weiling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26512)  

**Abstract**: AI agents can draft claims faster than authors can check whether the cited or retrieved evidence supports them. We study evidence-ledger adjudication: a claim-evidence traceability workflow that pairs each claim with an evidence packet, assigns a support relation, and routes unsupported, contradicted, or mixed-evidence claims back to the author. The empirical core is a 2,335-row blind benchmark built from independent external labels in AVeriTeC, CLIMATE-FEVER, and SciFact. Gold relations and source evidence labels are hidden during prediction and joined only for scoring. On this benchmark, the agent evidence-ledger condition achieves 0.676 relation accuracy and 0.601 macro-F1, compared with 0.383 accuracy and 0.303 macro-F1 for the best non-agent baseline. It also routes 1270/1435 claims whose gold labels indicate contradiction, missing evidence, or mixed evidence, while routing 295/900 supported claims. These results show that evidence-ledger adjudication can turn heterogeneous evidence packets into an auditable traceability layer for AI-assisted writing. 

---
