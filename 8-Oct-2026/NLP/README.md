# EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory 

**Authors**: Hongru Cai, Ran Wei, Wenjie Wang, Chengfa Wu, Ning Song, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.10533)  

**Abstract**: Conditional memory architectures such as DeepSeek Engram use input n-grams to look up learned embeddings, expanding the capacity of large language models (LLMs) with limited additional computation. Beyond model scaling, this architecture has demonstrated the potential to decouple factual knowledge storage from general-purpose computation, offering a promising route to updating factual knowledge while keeping the Transformer backbone fixed. Realizing this potential is challenging because different expressions of a fact may activate different n-gram embeddings, while updating shared embeddings can unintentionally change the model's predictions about other facts. We propose EngramEdit for decoupled knowledge updates through conditional memory. EngramEdit first computes target memory representations that make the model predict the updated fact across multiple expressions. It then jointly updates the shared n-gram embeddings to match these targets across expressions and edits, penalizing updates to frequently reused embeddings more strongly to preserve unrelated knowledge. Experiments show that EngramEdit enables independent factual knowledge updates through conditional memory, achieving near-perfect editing success. Revised knowledge is usable across unseen expressions and in multi-hop reasoning, with nearly three times the strongest baseline's accuracy under chain-of-thought (CoT) prompting. Unrelated knowledge and general capabilities are largely preserved even as factual updates accumulate. These findings show that EngramEdit turns conditional memory into an editable knowledge interface, extending its role beyond model scaling to support decoupled knowledge updates. 

---
# Your Prompt Should Do More: Effects of Retrieval Instructions in Embedding Models 

**Authors**: Amanda Myntti, Jenna Kanerva, Veronika Laippala, Filip Ginter  

**Link**: [PDF](https://arxiv.org/pdf/2610.10508)  

**Abstract**: Prompted embedding models have recently received increasing attention, particularly for retrieval, where detailed retrieval instructions are provided as part of the retrieval prompt. Several new datasets and studies have examined this setting, showing that the current embedding models often struggle to follow such instructions reliably. In this paper, we study the mechanism of how instructions actually affect the representations of retrieval queries in asymmetric retrieval tasks. We show that models can fail to follow even simple task instructions when query-side distractors are included in the evaluation. We hypothesize that this behavior is driven by the training setup of current embedding models and their evaluation, and show that fine-tuning with added query-side distractors leads to substantial improvements, with minimal effect on other tasks. 

---
# PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs 

**Authors**: Linghao Meng, Feng He, Xuan Yang, Junyuan Mao, Pinze Ren, Deqing Mu, Hesen Yang, Qiankun Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.10455)  

**Abstract**: Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of post-hallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBench, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBench characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning trajectory. We further find that properties of the hallucinated prompt contain substantial predictive signal for successful recovery, with a lightweight predictor achieving an AUROC of 0.847. These findings provide a behavioral view of post-hallucination reasoning, characterizing how LLMs resolve erroneous context and when successful recovery is likely to occur. 

---
# CoTrace: Data Recipes for Training Terminal Agents with Harness-Model Co-Evolution 

**Authors**: Jixuan Chen, Jiaxin Zhang, Qinyuan Ye, Yada Pruksachatkun, Haoxiang Zhang, Jingming Zhuo, Yifan Zhang, Yutong Dai, Juntao Tan, Xiangyu Peng, Silvio Savarese, Zeyuan Chen, Lianhui Qin, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10426)  

**Abstract**: Terminal-agent capability depends jointly on model weights and the runtime harness that formats prompts, binds tools, and handles error recovery. Existing harness-model co-evolution approaches improve both components, yet often treat trajectories produced during harness search as an undifferentiated replay buffer. This practice overlooks that a trajectory's value for model training depends on the harness under which it was generated. To systematically analyze this interface, we establish an alternating co-evolution framework that decouples harness search and policy training through component-wise promotion decisions. Within this framework, we introduce CoTrace, a harness-aware data recipe that explicitly governs trajectory routing, provenance matching, and curriculum refresh. Under CoTrace, recurring execution failures guide harness synthesis, while policy training is strictly conditioned on verified rollouts matched to the adopted runtime for supervised fine-tuning (SFT) or fresh online interactions for reinforcement learning (RL). On the Tmax promotion split, CoTrace advances Qwen3.5-9B from 78 to 88 solved tasks under supervised fine-tuning while an online reinforcement variant reaches 90. Specifically, a compact harness-matched corpus produces steady model gains at substantially lower compute than much larger corpora pooled across sibling harnesses. Furthermore, evaluations on Terminal-Bench 2.1 and SWE-bench Lite show that out-of-distribution transfer depends fundamentally on harness compatibility, where maintaining consistency between training and evaluation runtimes prevents procedural execution breakdowns observed under foreign scaffolds. 

---
# Document-Level Text Simplification in Estonian Using Large Language Models 

**Authors**: Meeri-Ly Muru, Eduard Barbu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10378)  

**Abstract**: Document-level text simplification involves transformations that go beyond sentence-internal edits, addressing discourse coherence, anaphora resolution, and cross-paragraph consistency. Despite advances in sentence-level simplification for high-resource languages, document-level simplification in morphologically rich, low-resource languages such as Estonian remains largely unexplored. This study presents a comprehensive evaluation of five state-of-the-art multilingual large language models (LLMs) for document-level simplification in Estonian. Three prompting strategies are examined: single-pass generation, pipeline-based modular agents, and guideline-augmented pipelines. The evaluation framework integrates automatic metrics assessing readability, semantic preservation, and discourse coherence, alongside a structured manual annotation protocol. The findings indicate that Gemini-2.0 and LLaMA-3.3 produce outputs with near-native fluency and strong meaning preservation, whereas other models display notable grammatical and semantic limitations. This work contributes novel document-level coherence metrics, evidence-based prompting strategies, and publicly available resources for reproducibility. 

---
# Learning to Act with Task Progress: Distilling Small Agents from Compact Teacher Supervision 

**Authors**: Wenxi Gan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10332)  

**Abstract**: Learning from large-model demonstrations offers a way to train small agents that can complete recurring tasks without calling a large model at every step. A central design choice is what to retain from teacher trajectories that contain reasoning, actions, and information about task progress. We introduce Task-Progress Distillation (TPD), an offline approach that pairs each demonstrated action with a short label describing the current task stage. The student learns these compact targets and selects actions by jointly scoring admissible stage--action pairs, which a deterministic harness executes in the environment. On ALFWorld, a 1.7B student trained with 404 demonstrations achieves 72.4\% mean unseen task success with either TPD or action-only supervision, compared with 48.3\% for a reasoning-trained student using constrained action selection. Explicit stages provide an additional benefit at 200 demonstrations, improving success from 48.0\% to 67.7\% over action-only supervision. With more demonstrations, the action-only student closes the gap, and both approaches reach 76.9\% at 808 demonstrations. Shared-history analyses link part of TPD's local advantage to better decisions when moving between subgoals, particularly from object acquisition to processing. These results show that compact supervision can train effective small task agents, while explicit task progress provides additional guidance at an intermediate demonstration budget. 

---
# Nobody Truly Agrees on Sentiment: Humans, Bespoke Tools, and LLMs Struggle with Social Media Texts 

**Authors**: Himarsha R. Jayanetti, Sivakanesan Dhanushkanda, Shuai Hao, Michael L. Nelson, Michele C. Weigle  

**Link**: [PDF](https://arxiv.org/pdf/2610.10318)  

**Abstract**: Social media is a rich source of real-time public sentiment, but widely used sentiment analysis tools are often applied without understanding their limitations. In this study, we evaluate the inter-rater reliability of three bespoke sentiment analysis tools (TextBlob, VADER, and Twitter-roBERTa-base) and three large language models (LLMs: Qwen3-32B, GPT-OSS-120B, Llama-4-Maverick-17B) against six human raters across 100 tweets. We measured agreement using two statistical measures: Cohen's kappa for pairwise comparisons and Fleiss' kappa for multiple raters. Even among the human raters, our results showed only fair agreement, highlighting the subjectivity of sentiment analysis. Higher agreement was observed under the binary sentiment classification (negative vs. non-negative and positive vs. non-positive) than under the three-class classification across both humans and automated tools. The Twitter-roBERTa-base model showed the strongest alignment with human ratings, outperforming both bespoke sentiment tools and LLMs, particularly in distinguishing negative versus non-negative sentiment. LLMs showed substantial agreement among themselves and moderate to substantial alignment with humans, performing better in positive vs. non-positive classifications. Our findings underscore that domain-specific fine-tuning remains crucial for reliable social media sentiment analysis, and human-centered evaluation remains essential for establishing gold-standard labels. 

---
# LLM Persuasion Is in the Eye of the Evaluation 

**Authors**: Kamile Dementaviciute, Julija Vaitonyte, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2610.10232)  

**Abstract**: Large language models (LLMs) have already been shown to match or exceed human experts in persuasion. While their persuasive capabilities hold promise for beneficial uses such as education and health communication, they can also be used to manipulate and misinform, making their evaluation a growing priority for developers and regulators. That evaluation, however, remains fragmented: studies differ in what they treat as persuasion, and broad claims often rest on narrow, situation-specific assessments. Automated methods, often modelled on human studies, offer a way to compare such assessments directly, as they can be run on the same models at scale and can include high-risk forms of persuasion that would be difficult or unethical to test on people. In this study, we adapt nine published automated methods to a shared setup, run them on the same fifteen LLMs, and ask whether their rankings agree and why. We find that the methods agree only weakly (mean Spearman $\rho = 0.25$). Our analyses point to two contributing factors. Models that refuse some tasks but not others, directly or indirectly, lower agreement by about a quarter, and these refusals fall mostly on manipulation tasks. General capability also plays a part: most rational persuasion (non-manipulative) methods track it, whereas most manipulation methods do not. Together, these findings suggest that agreement depends more on the task a method sets than on how it scores persuasion, although this pattern is only indicative given the eight methods available for analysis. More broadly, our results suggest that persuasion scores combine a model's ability to persuade with its willingness to do so. A single score is therefore informative about its own setting, but says little about a model's persuasiveness across tasks. 

---
# Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents 

**Authors**: Wenyu Huang, Xinyu Hou, Pavlos Vougiouklis, Ruofei Lai, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10179)  

**Abstract**: Search agents enable Large Language Models (LLMs) to iteratively retrieve and use information for complex multi-hop questions. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach for post-training such agents, but its reliance on sparse, outcome-based supervision can make credit assignment difficult and limit learning efficiency. In this paper, we systematically investigate how intermediate supervision can improve reinforcement learning for search agents. We study a range of reward-shaping and credit-assignment strategies that provide learning signals from intermediate retrieval steps. Building on these insights, we develop a training framework that combines intermediate signals with final outcome rewards to improve learning from multi-step search trajectories. Experiments across multiple benchmarks under matched training conditions demonstrate improvements in aggregate search-agent performance and show that both the choice of intermediate signal and where its credit is assigned affect training behaviour. These findings show that reward design and credit assignment are important design dimensions for training effective search agents. 

---
# HySPE: Positional Encoding via Symplectic Dual Shears 

**Authors**: Zhongping Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.10154)  

**Abstract**: We introduce Hyperbolic Symplectic Positional Encoding (HySPE), grounding positional attention in non-compact symplectic transformations. While canonical Rotary Position Embedding (RoPE) parameterizes the compact, elliptic branch of $\Sp(2,\R)$ via rotations, HySPE operationalizes its hyperbolic branch via a damped symmetric composition of dual shears, yielding a conformally symplectic contraction with two spectral decay rates per channel pair. To eliminate the exponential representation drift inherent to naive absolute factorizations, we diagonalize the operator in its invariant eigenbasis and introduce blockwise coordinate rebasing with adaptive centered execution. This guarantees length-independent numerical bounds while matching cached RoPE forward latency (7.21\,ms on an RTX 4090). On TinyShakespeare, HySPE-UltraLong maintains an invariant perplexity of 4.810 up to $16\times$ zero-shot extrapolation ($L=4096$), whereas RoPE degrades to 131.198. Scaled to a 51M-parameter subword Transformer on WikiText-103 ($L_{\text{train}}=512$), HySPE closely matches RoPE in-domain while robustly extrapolating to length 8192, reducing tail perplexity by 83.9\% over RoPE. While these controlled experiments establish HySPE's extrapolation robustness and numerical stability, evaluating its scaling behavior on large-scale foundation models remains an important direction for future investigation. 

---
# InterView-C: A Synchronized Multimodal Corpus of VR Avatar-Mediated Survey Interviews 

**Authors**: Patrick Schrottenbacher, Leon Hammerla, Lydia Kleine, Doris Stingl, Alexander Mehler  

**Link**: [PDF](https://arxiv.org/pdf/2610.10145)  

**Abstract**: We present InterView-C, a German multimodal corpus of 27 survey interviews conducted entirely in virtual reality, with both interlocutors represented by avatars. The corpus aligns spoken interaction with synchronized behavioral data, including gaze, head and body movement, facial behavior, hand and finger tracking. Its reference transcripts and linguistic annotations provide a reliable interface between this multimodal spoken interaction and predominantly text-based NLP methods. This interface is important because automatically transcribing speech can distort linguistically relevant information, while downstream models trained on existing resources may additionally face transfer challenges when applied to transcribed spoken data. InterView-C therefore provides word-timed and manually post-edited verbatim transcripts for all 54 recordings, interview-item timings, questionnaire responses and negation cue and scope annotations for 1,422 sentences, 1,398 of them doubly annotated ({\alpha}=0.87 for cues; {\alpha}=0.81 for scopes). We demonstrate both challenges empirically: nine open-weight ASR systems disproportionately misrecognize short closed answers and number words, while negation models trained on existing corpora show lower and highly variable performance on our transcribed interviews than a model trained on the InterView-C annotations. InterView-C thus enables linguistic analyses of spoken interaction while retaining their alignment with rich multimodal behavior. 

---
# LLM4Impact: Integrating Heterogeneous Information for Scientific Impact Prediction 

**Authors**: Yong Cao, Markus Flicke, Haoyu He, Katrin Renz, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2610.10138)  

**Abstract**: Predicting the future impact of a newly published paper is challenging because it must be inferred from heterogeneous evidence available at publication time. Existing approaches often rely on a single source of information or combine multiple sources without accounting for their different predictive roles. In this paper, we present LLM4Impact, an evidence-aware method for scientific impact prediction that learns to represent, integrate, and calibrate heterogeneous information. LLM4Impact combines semantic, graph, LLM, and temporal representations, and injects graph information into a frozen LLM through continuous prefix tokens. A context aware gating mechanism adaptively weights different evidence, while a separate calibration module accounts for domain and temporal variation in citation scales. We further construct a large-scale benchmark dataset with 2 million papers, leakage-safe point-in-time heterogeneous ego graphs, temporal splits, and both year-level and month-level citation targets. Experiments show that LLM4Impact consistently outperforms strong semantic, graph, and LLM based baselines, with a 10.13% reduction in year RMSE on the in distribution test set and a 6.87% reduction under out-of-domain distribution. Our results reveal that the value of such evidence is context dependent: different papers benefit from different sources, while domain and publication time affect how evidence translates into citations. This finding motivates adaptive evidence selection and context-conditioned calibration rather than simply richer representations. We will release our code, benchmark, and an interactive web demonstration upon publication. 

---
# Mechanics of Long-Context Hybrid Models Part 1.1: From Hybrid Attention to Hybrid Position 

**Authors**: Xiaoran Liu, Ziwei He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10114)  

**Abstract**: The architectural design of Large Language Models (LLMs) is shifting from traditional full-attention-only models to hybrid models, which combine different attention modules to improve long-context efficiency and performance in length extrapolation and context extension. To explain why hybrid models work and how to design them better, we propose Mechanics of Long-Context Hybrid Models. As Part 1.1 of this series, we begin with hybrids of full attention and either sliding-window attention (SWA) or gated variants of linear attention (LA), represented by GLA and GDN. We first observe a Seesaw Effect in Context Extension: LA hybrids benefit more from long-context continual pretraining, whereas SWA hybrids perform better under length extrapolation. We attribute this behavior to differences in the positional inductive biases induced by these attention mechanisms. We find that SWA hybrids suffer from a Short-Context Learning Trap, Short-Window Weariness, and Long-Window Laziness, and require extended windows to enhance performance in continual long-context pretraining. For LA hybrids, we summarize the Matthew Effect of Hybrid Position Extrapolation and propose Sliding-Window Linear Attention, achieving 16$\times$ training-free length extrapolation while maintaining 100\% accuracy on NIAH-SK1 in 64k context length. 

---
# I would rather quit NLP than read another paper like this: The rise of antithesis in NLP papers 

**Authors**: Olga Zamaraeva, Adrián Gude, Roi Santos-Ríos, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2610.10092)  

**Abstract**: For better or worse, LLMs are by now used routinely for scientific writing.\footnote{This paper is no exception; we did use AI to assist with writing some of the sections (see Acknowledgments).} Many have noticed that recent models fill papers with unnecessary antithesis, stating over and over what the work does not do, in ways that do not contribute to its precision or quality of expression and annoy reviewers \emph{rather than impressing them}. We study the construction \emph{rather than} in ACL papers from 2019, ACL-style arXiv papers from 2026, and papers written by GPT models from the same titles and abstracts. Its rate in 2026 is seven times the 2019 rate, and higher still in the GPT papers. Two annotators, blind to the source, find almost no 2019 use \emph{annoying} and about one in ten 2026 uses; they seldom agree on which, yet about half of 2026 papers contain a use that annoys each of them. \emph{Annoying} uses present the rejected alternative less favorably than legitimate uses. Raters of preference data and open reward models favor the construction, and an instruction to be honest promotes it. We conjecture that it is a side effect of post-training on pairwise preferences, which credit a disavowal in a single response and cannot register its cost across a text. 

---
# ExperienceIndex: Artifact-Grounded Memory 

**Authors**: Peter Baile Chen, Geoffrey X. Yu, Xinming Liu, Samuel Madden, Dan Roth, Jacob Andreas, Doug Downey, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2610.10091)  

**Abstract**: Knowledge-intensive tasks require answering many questions by reasoning about a shared corpus of artifacts (e.g., court cases, or scientific literature). As humans interact with these corpora, they naturally accumulate experiential knowledge about artifacts, enabling them to quickly identify the complete set of relevant artifacts for each new task. However, existing AI agents lack appropriate memory solutions to build or reuse such artifact-grounded experience, leading to lower answer quality and higher online cost. Existing memory solutions extract and reuse information from prior task-solving traces, but they primarily focus on user preferences, factual attributes, or abstract reasoning patterns rather than persistent artifact-specific knowledge. We introduce ExperienceIndex, a novel experience layer for AI agents that captures and reuses knowledge about artifacts based on prior reasoning traces. ExperienceIndex stores two complementary forms of experience: (i) single-artifact experiences that summarize an artifact's contribution to prior tasks and (ii) artifact-pair experiences that encode structural relationships discovered during past reasoning. Integrated as lightweight middleware, ExperienceIndex uses an experience retrieval mechanism to guide agents toward the complete set of relevant artifacts for new tasks, improving both answer quality and efficiency. Across diverse corpora and agentic solutions with different search frameworks, ExperienceIndex delivers consistent gains, raising answer quality by up to 11.0 points and reducing online dollar cost by up to 50.5%. We further demonstrate two benefits: (i) cross-task generalization, where experiences accumulated from text-to-SQL tasks transfer to factoid QA tasks over the same artifact corpus, and (ii) teacher-student learning, where experiences from a stronger model enable a weaker model to reach comparable performance. 

---
# Cache the Encoder Within:Compact, Reusable Memory across LLM Queries 

**Authors**: Hanzuo Liu, Chunyu Liu, Chaofan Lin, Alex Lamb, Mingyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.10058)  

**Abstract**: Repeated queries over shared documents incur redundant encoding, while caching model states introduces persistent storage costs. Building on CoMem's intermediate-state interface, EncBank treats a pretrained LLM's lower layers as a reusable document encoder and compactly stores their outputs for an adapted upper-layer reader. A self-distilled suffix adapter is shared across storage precisions within each backbone, without quantization-specific retraining. Across five benchmark suites on three Qwen backbones spanning different sizes and full-attention and hybrid architectures, 4-bit storage keeps each reported benchmark aggregate within one score point of native-precision EncBank. In a fixed Qwen3-8B workload, it retains 28.1% of the native-precision persistent GPU store. Separate native-precision controls yield a 1.40x selected-pack prefill speedup over same-evidence, same-adapter text replay, at a 3.12-point RULER accuracy cost. A native-precision Qwen3.8-27B configuration also passes 70 of 89 Terminal-Bench 2.1 tasks. EncBank thus combines reusable computation with compact memory, while task fidelity and end-to-end benefits remain dependent on the workload, preparation costs, and reuse frequency. 

---
# The Long Road to the Same Answer: Cognitive Bias Under Escalating Reasoning Budgets in Large Language Models 

**Authors**: Obada Kraishan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10049)  

**Abstract**: Reasoning models allocate extra computation at inference time and present their answers as the product of deliberate thought. If this deliberation works the way dual-process accounts of human cognition suggest, longer thinking should weaken the classic decision biases that fast, intuitive judgment produces. Using 30 vignettes covering six biases (anchoring, framing, loss aversion, escalation of commitment, availability, confirmation) from an established benchmark, we run a dose-response study across four model families, pairing each reasoning model with a matched non-reasoning sibling and requesting thinking ceilings of 0, 1,024, 4,096, and 8,192 tokens, for 12,350 API calls. Because a requested ceiling is not the same as realized deliberation, we use the reasoning tokens each call consumed as the dose. First, reasoning models are not less biased than their siblings; the point estimate leans the other way in every family, but the item-level pooled contrast is not reliable (Delta = +0.031, t(29) = 1.45, p = .157). Second, bias magnitude does not reliably fall as realized deliberation grows: no slope is significantly negative, and where anything moves it is the signed score drifting further from the human direction. Third, anchoring is the only bias in the human direction (d = 1.89). Four of the other five lean the opposite way in all seven models; with five items per bias, that reversal is reliable for framing and directional for escalation of commitment, confirmation, and loss aversion, while availability is absent. A one-line instruction to restate the anchor before answering lowered anchoring on all five anchoring items, which no amount of additional thinking did, although the effect does not reach significance (p = .057). The results argue against treating test-time reasoning as a rationality guarantee and for auditing deployed models bias by bias. 

---
# EASE: Entropy-Adaptive Distribution Shaping for Evading AI-generated Text Detectors 

**Authors**: Jicheng Zhou, Kahim Wong, Jialong Wang, Jiantao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.09976)  

**Abstract**: AI-generated text (AIGT) detection can be sensitive to the decoding choices of the source large language model (LLM). We observe that perturbing next-token logits or adjusting sampling temperature can reduce detection performance, providing a clear signal of detector vulnerability to decoding-time distribution changes. Building on this observation, we propose EASE (Entropy-Adaptive Distribution Shaping for Evasion), a training-free and detector-agnostic framework for evading AIGT detectors. EASE computes predictive entropy directly from the source LLM's next-token distribution and uses it to adapt both logit perturbation and sampling temperature, without detector feedback or model fine-tuning. Experiments across three source LLMs and multiple detectors demonstrate consistent reductions in detection performance, with negligible degradation in text quality and negligible inference overhead. 

---
# Itgan at NADI 2026 shared task: Parameter-Efficient Whisper Adaptation for Robust, Mixed-Dialect and Code-Switched Arabic ASR 

**Authors**: Ibrahim Almajai  

**Link**: [PDF](https://arxiv.org/pdf/2610.09934)  

**Abstract**: We describe the Itgan systems for the three ASR subtasks of NADI 2026, namely robust country-level ASR (1.1), mixed-dialect ASR (1.2), and Tunisian code-switched ASR (1.3). All three share one recipe, Whisper adapted with LoRA on consumer GPUs, and each was carried by a different addition to it. On 1.1, where the dialect label is given at test time, per-dialect specialists continued from a pooled adapter gave the largest gain, and the submitted system reached 57.1% country-average WER. A post-evaluation linear probe on frozen encoder features routes utterances without the label and recovers 44% of what oracle routing gives. On 1.2 the choice of base model mattered more than adapter capacity, and system combination helped only once we added a decorrelated member, reaching 46.7% WER. On 1.3 our system placed second at 14.49% WER with the lowest CER among the leading submissions, 5.38%. Its last 0.60 WER points came without further training, mostly from an exact weight-space average of independently trained runs, with ROVER voting adding the remainder. Every comparison carries a paired-bootstrap test, and we report eight directions that did not work. 

---
# Training Advisors for LLM Agents from Task Outcomes 

**Authors**: Sergei Polezhaev, Barys Liskavets, Ori Press, Alexander Golubev  

**Link**: [PDF](https://arxiv.org/pdf/2610.09858)  

**Abstract**: Large language model agents tackle multi-step tasks by interleaving reasoning and tool calls with observations from the environment. Prior work has shown that natural-language feedback can help these agents revise their decisions during task execution. We introduce Caddie, a method for training critics to provide natural-language analysis and advice as agents work through a task. Unlike approaches that rely on step-level labels or reference critiques, Caddie learns from whether the agent ultimately succeeds after receiving the critic's feedback. We optimize the critic through reinforcement learning while keeping the base model frozen. Trained on multi-hop question answering with a single base model, our Qwen3-4B critic improves success rates across four base models of different scales and architectures, including three not used during critic training. On the MuSiQue benchmark, the trained critic improves Qwen3-4B's success rate by more than 25 percentage points, surpassing the performance of Kimi K3 without a critic. The same critic also yields gains on out-of-domain interactive benchmarks, including $\tau^3$ and DeepDive, with no additional training. Our results show that agents can decide when to seek help from a critic at inference time and that outcome-based critic training can produce guidance that transfers across base models and task domains. 

---
# A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks 

**Authors**: Jonghyun Han, Younghoon Song, Jongyoul Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.09835)  

**Abstract**: Continual pre-training and fine-tuning in Large Language Models (LLMs) inevitably induce catastrophic forgetting, typically mitigated by replay using often-inaccessible original data. In this data-free regime, we analyze where forgetting occurs and why. Systematic parameter freezing across five settings up to 1.4B reveals that forgetting concentrates selectively in the output embeddings of tokens rarely seen in the new corpus, whereas the same sqrt(v-hat) band of the body is inert and new learning resides elsewhere. This localization is governed by the vocabulary deficiency of the corpus rather than the training mode, allowing pre-retraining risk ranking from token counts alone within a fixed base model. Mechanistically, absent tokens receive persistent one-sided softmax gradients that Adam's second-moment (sqrt(v-hat)) normalization amplifies into full-sized updates. We therefore propose an intervention: raising Adam's epsilon exclusively for the output projection during training. Across eight settings spanning 160M to 12B parameters and four model families, this removes 39.4% to 67.9% of forgetting across all seven stable configurations without degrading target learning or requiring per-model tuning. The defense combines additively or better with replay (79.8% on Qwen/Korean) and rescues released-head LoRA from a 23-fold forgetting surge. Because post-hoc editing of the drifted rows recovers under 5% of forgetting, the intervention must operate during training. Our findings indicate that a single-line optimizer adjustment may serve as the primary defense against catastrophic forgetting where the corpus starves the vocabulary. 

---
# MIRROR: From Imitation to Internalization in LLM Personalization 

**Authors**: Huayi Lai, Jicheng Yang, Min Yi, Chong Meng  

**Link**: [PDF](https://arxiv.org/pdf/2610.09795)  

**Abstract**: The demand for personalized LLMs is shifting from style imitation toward content quality. We investigate whether self-distillation can bridge this gap in existing fine-tuning paradigm. To address this limitation, we introduce MIRROR(Meta- personalization by Internalizing Reference-Revealed On-policy Reflections), a novel self-distillation framework that shifts LLM personalization from imitation toward preference internalization. First, we replace reference-token imitation with reference-revealed on-policy self-distillation, aligning the model's next-token distributions along its own generation trajectories with those of its reference-conditioned self, thereby internalizing user preferences rather than reproducing reference this http URL, we introduce MIRROR-F, a focal plug-in that augments on-policy distributional alignment with selective supervision over informative reference tokens, thereby strengthening content generation while preserving user-specific expression. Across three personalized generation benchmarks, two model scales, and complementary reference-based and LLM-based evaluations, MIRROR and MIRROR-F achieve leading overall personalization performance and superior text quality, while exhibiting less catastrophic forgetting than SFT-based baselines on three unseen personalized generation tasks. The gains are consistent across model scales and application scenarios, translating to improved performance in LLM personalization tasks. 

---
# Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression 

**Authors**: Mingqing Yuan, Xiaobo Liang, Junwei Yang, Ziwei Chen, Zeren Zhang, Hejin Wang, Yubin Wang, Juntao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09788)  

**Abstract**: Reward modeling often requires jointly representing and reasoning over multiple evaluation criteria, yet verbalizing this process token by token can incur substantial inference cost. Recent work on latent reasoning suggests that continuous states may support this computation more compactly. We introduce LatentGRM, a latent evaluation framework built on semantic chunking, compression, and reconstruction. By using the structure of rubric-guided evaluations to guide compression, LatentGRM learns compact continuous trajectories that support autonomous pairwise judgments without generating textual assessments. A separate interpreter reconstructs evaluation text from these trajectories, providing an offline view of the information retained under compression. Under matched training data and backbones, LatentGRM achieves competitive aggregate preference accuracy relative to explicit Supervised Fine-Tuning (SFT) judges at both 4B and 8B scales. Across four benchmark domains, LatentGRM-8B compresses evaluation trajectories by 8.9--9.2x and reduces total judge inference time by 6.1--7.0x at vote@5. Controlled rubric interventions show that criterion-dependent preference information is carried through the latent sequence. Together, these results demonstrate that continuous latent evaluation can substantially reduce inference cost while preserving competitive judgment quality. 

---
# Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution 

**Authors**: Masaaki Nakatsu, Reno Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09772)  

**Abstract**: Small language-model agents on edge devices must hold a persona and reason correctly at once, inside one context window that fills with conversational history and persona instructions. We study what happens to the logical part of such an agent when that history is long, misleading and persona-heavy (persona-logic interference), and present a Decoupling Architecture (AO-DA) that separates logical inference ("What") from persona expression ("How") into two inference paths on one INT4 base model with hot-swappable LoRA adapters. The logic path receives only the core turn and emits a verifiable structured state (Micro-State); the persona path renders it in character with the full history. In same-base-model ablations on an Apple M2 laptop (Llama-3.1-8B-Instruct and Gemma-3-4B-it, 4-bit; 480 runs over 4 pollution levels x 3 arms x 2 tasks x 2 personas x 5 seeds) we find: (i) the decoupled logic path is structurally invariant to pollution: its prompt stays at 180 (Llama) or 167 (Gemma) tokens while the mixed single-pass prompt grows from 242 to 1,203, and its outputs are byte-identical across levels (40/40); (ii) the mixed single pass degrades monotonically (composite logic score 0.669 to 0.150 on Llama, 0.487 to 0.150 on Gemma), mostly by failing to emit the required structured output (80-95% of runs on Llama, 100% on Gemma at the two highest levels); (iii) with the same pollution fed into the decoupled logic path, the dedicated-adapter, dedicated-format path is still more robust than the single pass on the 8B model (failure 0-20% vs 80-95%; paired $\Delta$ +0.30 to +0.50, Cliff's $\delta$ 0.50-0.85, Holm-adjusted $p \le 0.03$) but not on the 4B model, where both collapse. Separation costs one extra decode on a topic's first turn (28.2 s vs 18.2 s on Llama) and buys persona hot-swapping in 1.7 ms without re-running the logic path. Code, rubric, fixtures, adapters and logs are released. 

---
# Shaer: Controlled Arabic Poetry Generation with Meter Subform and Semantic Conditioning 

**Authors**: Ahmad Abbas, Tamara Fakih, Nour Fakih, Ammar Mohanna  

**Link**: [PDF](https://arxiv.org/pdf/2610.09756)  

**Abstract**: Classical Arabic poetry generation requires simultaneously satisfying semantic, linguistic, and fine-grained prosodic constraints. Existing systems typically control broad poetic attributes but do not jointly model semantic intent, meter subform, and poem length. We present Shaer, a controllable Classical Arabic poetry generation framework jointly conditioned on natural-language descriptions, meter subforms, and target hemistich counts. To support this task, we construct an enriched corpus of 116,032 classical Arabic poems derived from Ashaar, containing normalized meter-subform labels and automatically generated, validated semantic descriptions. We then adapt Yehia-7B using QLoRA-based supervised fine-tuning with a completion-only objective. Our evaluation combines automatic assessment of base-meter conformity, requested-subform adherence, and length control with three LLM judges, blinded human evaluation, and memorization analysis. Shaer achieves 95.17% base-meter accuracy, 91.75% poem-level meter-subform accuracy, and 83.40% exact count accuracy. Relative to its untuned foundation model, these results represent gains of 68.68, 57.77, and 38.93 percentage points, respectively; Shaer also attains the highest base-meter accuracy among all evaluated systems. Multi-LLM evaluation and a blinded human assessment of top-ranked outputs further indicate competitive semantic and literary quality. Finally, analysis of all 3,481 test generations finds no exact copies from the training corpus or paired source poems. Code, models, and datasets are publicly available. 

---
# Bridge Routing Heads: Where Multilingual Multi-hop Reasoning Lives in LLMs 

**Authors**: Seunghan Kim, Minyeong Choe, Hyunil Kim, Haehyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2610.09733)  

**Abstract**: Multilingual LLMs answer the same multi-hop reasoning question across languages, but we lack a mechanistic account of whether they share an internal circuit. We identify Bridge Routing Heads (BRH) in two large multilingual LLMs through a three-stage pipeline. The resulting language-specific head sets exhibit near-complete mutual exclusivity across the five languages, with a mean Jaccard similarity of only 0.017 for Llama 3.1 70B and 0.057 for Qwen 2.5 72B, revealing language-idiosyncratic circuits. Ablating general BRH increases two-hop Negative Log-Likelihood (NLL) by 39-89x the random-head baseline, providing direct causal evidence of their role. Amplifying these heads in a failing target-language pass rescues up to 51.7% of cross-lingual failures, with no training. The two models share this dual-circuit pattern but allocate heads differently: Llama concentrates chaining in a large general pool, while Qwen leans on larger language-specific pools. Together these results show that activation-level intervention alone can recover correct answers from cross-lingual reasoning failures. 

---
# SpikingVLA: Asynchronous Spiking Vision-Language-Action Models 

**Authors**: Jingya Wang, Dehao Zhang, Shuai Wang, Malu Zhang, Yang Yang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09710)  

**Abstract**: ANN-to-SNN conversion offers a practical route toward energy-efficient spiking Vision-Language-Action (VLA) models by bypassing the substantial cost of training large-scale SNNs from scratch. However, existing methods often require many timesteps to maintain competitive performance, resulting in substantial inference latency for real-time VLA deployment. To address this challenge, we introduce SpikingVLA, an ANN-to-SNN conversion framework that enables accurate and low-latency spiking VLA inference. Specifically, we propose a Dendritic Integrate-and-Fire (DIF) neuron that alleviates channel-wise activation outliers through dendritic mixing and adaptive somatic firing, enabling accurate ANN-to-SNN conversion with fewer timesteps. Building on DIF neurons, we further introduce an asynchronous execution mechanism that overlaps temporal computation across VLA components, reducing synchronization overhead and latency. Extensive experiments demonstrate that SpikingVLA achieves competitive navigation performance with substantially improved inference efficiency. Compared with existing spiking VLA methods, SpikingVLA improves SR and SPL by 11.9\% and 12.6\%, respectively, while reducing first-action latency by 11.2$\times$. These results establish SpikingVLA as a practical framework for deploying pretrained VLA models with high-performance and low-latency spiking inference. 

---
# From Pareto to Preference: Personalized Test-Time Scaling via Amortized Agentic Policy Discovery 

**Authors**: Xinglin Wang, Zishen Liu, Tong Zheng, Shaoxiong Feng, Peiwen Yuan, Yiwei Li, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Ji Zhang, Boyuan Pan, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09684)  

**Abstract**: Test-time scaling (TTS) improves the reasoning capabilities of large language models by allocating additional inference computation. Existing approaches to improving TTS efficiency largely optimize accuracy against one resource dimension at a time, advancing either the accuracy--cost or accuracy--latency Pareto frontier. Yet user requirements are multidimensional: users may specify accuracy, latency, and inference-cost requirements jointly, and different requirements can favor different controllers. We formulate Personalized Test-Time Scaling as discovering executable controllers that maximize the joint satisfaction rate of user-specific requirements. To reduce the overhead of repeated policy discovery for new user profiles, we propose PersonTTS, an amortized agentic policy-discovery framework that reuses prior search experience through requirement-matched controller initialization and source-distilled procedural guidance, while retaining target-profile evaluation for every candidate. Experiments on AIME and HMMT show that PersonTTS substantially outperforms strong TTS baselines in joint requirement satisfaction on unseen user profiles and held-out problems. Under the same candidate-evaluation budget, cross-user experience reuse further improves policy quality while substantially reducing discovery-agent time and cost. 

---
# InsClaimBench: Benchmarking Insurance Claim Adjudication Across the Decision Chain 

**Authors**: Linqi Zhang, Chong Qi, Yan Cheng, Wanqing Cao, Yu Liu, Chenwei Lin, Xian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09671)  

**Abstract**: Recent advances in reasoning-oriented large language models (LLMs) have motivated increasing evaluation of their ability to perform professional decision tasks. Insurance claim adjudication is one such task, requiring models to connect case evidence, insurance rules, intermediate judgments, and payout calculations across a structured decision process. We introduce InsClaimBench, an end-to-end benchmark for evaluating insurance claim adjudication across the decision chain. Grounded in real claim materials and structured insurance rules, InsClaimBench contains 3,780 cases in 375 case families across auto, property, and health insurance, comprising 86,656 atomic rule judgments. It evaluates each claim from atomic rules through adjudication modules to payout decisions and amounts, with controlled factual variants testing whether required changes are correctly propagated across levels. Evaluation of six LLMs reveals a progressive loss of reliability along the decision chain. Payout-decision accuracy ranges from 74.23--80.19%, while joint decision--amount accuracy drops to 47.54--73.15%. Strong local performance also fails to ensure case-level correctness: atomic-rule accuracy reaches 95.48%, whereas rule-vector exact match peaks at only 36.90%, and the most frequent module errors are not necessarily those most associated with final-decision failure. Under factual changes, these inconsistencies further become propagation failures: module updates are less reliable than rule updates, correct local judgments can still yield incorrect payouts, and correct payouts can conceal intermediate errors. These results show that reliable claim adjudication requires consistent composition and propagation across the decision chain. 

---
# SAPD: Step-Aligned Privileged Distillation 

**Authors**: Tianle Wang, Jiayu Liu, Ruizhi Zhao, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09665)  

**Abstract**: On-policy post-training can improve large language models by learning from their own trajectories, but requires costly rollout generation. We ask whether fixed demonstrations can support competitive off-policy learning through better supervision. Our premise is that their usefulness depends not only on the training trajectories, but also on whether supervision provides informative preferences among continuations and connects this guidance to the reasoning decision being learned. We introduce Step-Aligned Privileged Distillation (SAPD), a rollout-free self-distillation method that turns demonstrations into step-aligned distributional supervision. Its key insight is to use the known progression of a reference solution to associate each reasoning transition with targeted privileged guidance, rather than treating the solution as undifferentiated context. On mathematical reasoning benchmarks, SAPD outperforms supervised fine-tuning and label smoothing on average while remaining competitive with on-policy reinforcement learning and self-distillation. Analyses support both the value of context-dependent distributional guidance and the benefit of aligning privileged information with the current step. SAPD also largely preserves out-of-domain coding performance and achieves approximately 2x training-loop speedups over the on-policy baselines. These findings suggest that carefully constructed supervision can make fully off-policy post-training a competitive and computationally efficient alternative. Our code is available at this https URL. 

---
# Alice: A Large-Scale German Benchmark for Rubric-Based Multi-Dimensional Automatic Short Answer Scoring 

**Authors**: Zhifan Sun, Sebastian Gombert, Jannik Lossjew, Tobias Wyrwich, Berrit Katharina Czinczel, David Bednorz, Marcus Kubsch, Knut Neumann, Hendrik Drachsler  

**Link**: [PDF](https://arxiv.org/pdf/2610.09661)  

**Abstract**: Automatic Short Answer Scoring (ASAS) is central to NLP for Education. However, openly available benchmarks remain scarce, and existing datasets largely address how well students answer a question directly rather than how well they master underlying concepts (knowledge elements) such as thermal energy or epistemic activities (skills) such as reasoning or claim.
To address this gap, we introduce Alice, a large-scale, rubric-based German ASAS dataset that is pedagogically aligned and comprises three subtasks: (i) learning performance (Alice-LP), (ii) knowledge elements (Alice-KE), and (iii) skills (Alice-SK).
We further formulate rubric-based ASAS as a rubric-retrieval task and benchmark the dataset with a range of language models, from encoder-only models to lightweight LLMs. We also benchmark the dataset with zero-shot prompting via LLMs and a standard classification baseline. The experiments show that LLMs, in particular, struggle to score knowledge elements and skills in the zero-shot setting. They also indicate that rubric text is often useful, especially for Alice-KE and Alice-SK, while on Alice-LP gains over sample-solution-focused inputs are more modest and vary by model and input format. 

---
# Rubric Spans are Label Representations: Joint LLM Encoding for Short Answer Scoring 

**Authors**: Zhifan Sun, Sebastian Gombert, Fabian Zehner, Leon Camus, Longwei Cong, Hendrik Drachsler  

**Link**: [PDF](https://arxiv.org/pdf/2610.09660)  

**Abstract**: Automatic Short Answer Scoring (ASAS) requires models that can score student responses against question-specific criteria while remaining efficient and transferable across rubric sets. We propose RUSPAN, a rubric-conditioned ASAS framework that treats rubric descriptions as semantic label representations. RUSPAN serialises the question context, student answer, and all candidate rubric levels into a single sequence, then scores the levels listwise from the rubric-span and whole-sequence representations produced in a single LM pass. We further introduce RUSPAN-RIM, in which a Rubric-Independent Mask prevents rubric spans from attending to one another, making rubric representations depend only on the answer and question context and preventing overfitting to rubric patterns during training for zero-shot transfer. On six ASAS benchmarks spanning English, German, and Portuguese, RUSPAN improves mono-benchmark scoring over discriminative and generative baselines, while RIM with position reindexing delivers consistent and substantial gains on PT-ASAG, the held-out benchmark with the strongest combined language and rubric-structure shift. 

---
# On-Policy Distillation Teaches New Skills but Not New Knowledge 

**Authors**: Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09639)  

**Abstract**: On-policy distillation (OPD) strengthens language-model reasoning, yet whether students acquire new factual knowledge or compositional skill for multi-step reasoning remains unknown. We separate these capabilities using a controlled synthetic framework that measures the student's initial capabilities and independently controls the teacher's additional facts, compositional skill, or both. Across four models from three families, reverse-KL OPD reliably transfers compositional skill across unseen reasoning structures, but transfers minimal factual knowledge. Decoupling the distillation recipe reveals the source of this asymmetry: replacing reverse KL with forward KL restores factual transfer, whereas student rollouts specifically improve the execution of multi-step reasoning. Experiments on recent factual QA and competition mathematics show a similar asymmetry under reverse-KL OPD, yielding notable reasoning gains without factual memory expansion. Together, these results demonstrate that on-policy distillation does not expand a model's parametric knowledge, but instead teaches it to organize and compose the knowledge it already possesses. 

---
# Which Language Should a Skeleton Speak? Language Choices in Multilingual Reasoning 

**Authors**: HyeonSeok Lim, SeungWoo Song, Inho Won, Hoyun Song, Jihyo Kim, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.09607)  

**Abstract**: Skeleton-based reasoning prompting is a promising training-free approach for structuring LLM reasoning, but prior work largely assumes an English-centric setting. We propose the Language-Aware Skeleton Exploration Framework (LASEF) to study skeleton-language choice in multilingual mathematical reasoning. Across math benchmarks, model scales, and languages, we show that English skeletons yield a small positive tendency on average, most visible for smaller models and low-resource languages. However, few language-level gains remain significant after correction, and English is not universally optimal. Combining greedy decoding, multi-rollout evaluation, translation ablation, and cross-benchmark validation, we further find three patterns of skeleton-language effects: directionally consistent, evaluation- and benchmark-dependent, and asymmetric negative. These effects cannot be fully explained by generation quality alone. Overall, skeleton language is a context-dependent design variable that requires multi-level exploration. All resources are released at this https URL. 

---
# How Do LLMs Change Predictions Under Negation? 

**Authors**: Jongwook Yoon, Jongwon Lim, Sungjib Lim, Woojin Cho, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09571)  

**Abstract**: Negation is an essential feature of human language, yet large language models (LLMs) remain unreliable in processing it. We evaluate recent open-source and closed-source LLMs on our negation benchmark and find that, in 37-71% of cases, they repeat the same answer under negation (e.g., "Madrid" for "What is not the capital of Spain?"). To understand and address this brittleness, we mechanistically examine how models operate under negation. Our main finding is that specialized attention heads and MLP neurons jointly implement negation by (1) suppressing retrieval of the original answer (e.g., "Madrid") while (2) promoting a favored candidate within the answer category (e.g., "Paris"). This contrasts with accounts of human negation processing, in which information about the original answer helps to determine what should be excluded. Furthermore, we find that this difference from human processing is a key source of negation failures: the model's mechanism relies on suppressing the original answer rather than using it to determine what to exclude, so the model can repeat the original answer when suppression is too weak or when a bias toward particular answers prevents it from selecting an alternative. To address this weakness in the model's negation mechanism, we propose a training objective that requires larger shifts in answer preference for more confident original predictions, and show that it reduces negation failures with less degradation of general capabilities than standard fine-tuning baselines. Together, our results demonstrate how mechanistic analysis can reveal why a linguistic capability fails and guide training that targets the underlying limitation. 

---
# RELATE: An Evaluation Framework for measuring Relational Orientation of Large Language Models 

**Authors**: Shivam Shukla, Jihye Kim, Shubham Gaur, Mahnaz Roshanaei, Magy Seif El-Nasr  

**Link**: [PDF](https://arxiv.org/pdf/2610.09569)  

**Abstract**: Large language models (LLMs) are increasingly used for emotional support, raising concern that sustained use may draw users away from their real-world relationships. Yet existing evaluations primarily focus on the safety, empathy, or helpfulness of responses, leaving under-examined a relational question: where does the model orient the user for continued support? To address this question, we introduce relational orientation, a property operationalized through two non-exclusive dimensions: inward-facing (IF) language, which positions the AI as the user's ongoing source of support, and outward-scaffolding (OS) language, which encourages real-world human connection. Grounded in psychological and sociological literature, we formalize a taxonomy of relational orientation and present RELATE, a persona-conditioned framework for measuring inward-facing and outward-scaffolding language at the sentence level in multi-turn dialogues. RELATE pairs 76 help-seeking situations adapted from naturally occurring questions with three simulated user styles, providing 228 evaluation stimuli. In our experiments, we evaluate seven LLMs using dialogues with six assistant turns each, yielding 1,596 dialogues and 69,194 assistant sentences. We assess these sentences using a primary rubric-based LLM judge and apply a secondary judge to a subset. Under automated evaluation, we find that the proportion of sentences labeled as IF is higher at the sixth assistant turn than at the first, while the proportion labeled as OS is substantially lower for hesitant, indirect simulated users than for explicit, reassurance-seeking users. RELATE provides a reproducible framework and a sentence-level signal for auditing and steering the relational orientation of supportive LLMs. 

---
# A Comparative Study of Evaluation Metrics for Long-Document Financial Narrative Summarization with Transformers 

**Authors**: Nadhem Zmandar, Mo El-Haj, Paul Rayson  

**Link**: [PDF](https://arxiv.org/pdf/2610.09529)  

**Abstract**: There are more than 2,000 listed companies on the UK's London Stock Exchange, divided into 11 sectors who are required to communicate their financial results at least twice in a single financial year. UK annual reports are very lengthy documents with around 80 pages on average. In this study, we aim to benchmark a variety of summarisation methods on a set of different pre-trained transformers with different extraction techniques. In addition, we considered multiple evaluation metrics in order to investigate their differing behaviour and applicability on a dataset from the Financial Narrative Summarisation (FNS 2020) shared task, which is composed of annual reports published by firms listed on the London Stock Exchange and their corresponding summaries. We hypothesise that some evaluation metrics do not reflect true summarisation ability and propose a novel BRUGEscore metric, as the harmonic mean of ROUGE-2 and BERTscore. Finally, we perform a statistical significance test on our results to verify whether they are statistically robust, alongside an adversarial analysis task with three different corruption methods. 

---
# Goldsmith: Gold-Loss-Guided Definition Optimization with an Agentic Annotation Harness 

**Authors**: Yihan Li, Hanyi Zhang, Xiaoxi Jiang, Man Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09489)  

**Abstract**: Many annotation projects begin before experts have a stable guideline or enough labels to train a task-specific model. We present Goldsmith, an agentic pipeline that turns a small gold set---expert-annotated calibration examples representing the intended task boundaries---into a reusable structured annotation definition. Goldsmith treats this definition as a trainable textual object. Candidate definitions are run on the same gold examples and scored with an executable structured loss, while the output schema, formatting, retrieval, repair, judging, and human review remain in an external harness. A large language model (LLM) editor converts the highest-loss failures into textual-gradient revisions, which are accepted only when the measured loss decreases. In prompt-optimization comparisons, Goldsmith improves over direct rewriting, OPRO, APE, and PromptBreeder under matched evaluation protocols. The resulting definition also improves downstream annotation when combined with retrieval, score-based routing, and human review across typed span, pair-level relation, and fixed-trigger event-argument tasks. These results show that scarce expert supervision can support both task-definition learning and scalable annotation. 

---
# BanglaRhet: Benchmarking Classical and Transformer Models for Rhetorical and Persuasion Detection in Bangla Political Speech 

**Authors**: Rohit Kumar Sen, Anik Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2610.09464)  

**Abstract**: Political discourse often uses rhetorical and persuasive language to frame narratives, influence public opinion, and mobilize audiences. While Bangla natural language processing has made progress in sentiment analysis and opinion mining, systematic benchmarking of transformer models for fine-grained rhetorical and persuasion technique detection in Bangla political speech remains largely underexplored. This paper presents a benchmark study of transformer-based models for detecting rhetorical form and persuasive intent in Bangla political discourse. Using BanglaRhet, a manually annotated corpus of 30,289 Bangla political speech segments collected from publicly available political news sources, we formulate two supervised single-label classification tasks: rhetorical technique detection (contrast, repetition, exaggeration, metaphor, rhetorical questions) and persuasion technique detection (blame assignment, call to action, unity call, moral, emotional, and logical appeals). We evaluate four transformer-based models, BanglaBERT, BanglaBERT-Base, SahajBERT, and XLM-RoBERTa-Base, against classical TF-IDF baselines. BanglaBERT achieves the highest performance, with 65.40% macro-F1 for rhetorical technique detection and 66.46% for persuasion technique detection, outperforming the best tuned classical baseline by 19.2 and 13.8 macro-F1 points, respectively. Class-level analysis indicates that errors are mainly associated with semantic overlap among labels, figurative language, and class imbalance. The results provide initial benchmark baselines for Bangla rhetorical and persuasion-aware political discourse analysis and highlight the need for context-aware and multi-label modeling. 

---
# Right Number, Wrong State? Measuring Cross-Jurisdiction Substitution in LLM Recall of State Policy 

**Authors**: Jiayu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.09458)  

**Abstract**: When an LLM answers a state-specific policy question wrongly, it may be hallucinating, or it may be returning a real value that holds in another state. We test this with a minimal-set design: the question wording is fixed and only the jurisdiction varies, across the 50 U.S. states and the District of Columbia (51 jurisdictions) and three exactly defined Medicaid income-eligibility quantities. Gold values come from an official data book and agree with an independent source in 101 of 102 checked cells. Under a pre-registered protocol, Claude Sonnet 5.5 and GPT-5.6 Sol reproducibly give another state's current value, identical across two independent repeats, for 10 and 25 of 153 items. Attribution is fragile, however. Crediting any wrong answer that equals another state's value yields 3-5x more reproducible substitutions than checking every number in the asked state's own records, because many apparent cross-state answers are the asked state's own values under another convention or from an earlier year. Claims about cross-jurisdiction error need a complete same-state reference set. We will release the protocol, gold table, and all model outputs. 

---
# Arctic Questions, Missing Answers: A Dataset and Benchmark for LLM Abstention in Arctic Science 

**Authors**: Benjamin Wilcox, Dawei Gao, Pradeeban Kathiravelu, Douglas Causey, Kewei Sha, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.09446)  

**Abstract**: Large language models (LLMs) should abstain from scientific multiple-choice questions when no option is valid, but frequent abstention alone does not demonstrate sensitivity to answer availability. We introduce ArcticQA, a dataset of 194 questions derived from primary Arctic research, with automated checks of answer support and distractor contradiction against source evidence. We further develop ArcticAbstain, a paired benchmark comparing answer-present and answer-absent conditions, with the correct answer replaced by a distractor in the latter and an explicit abstention option in both. We evaluate eight models from the Gemini, Claude, and ChatGPT families at high reasoning effort, with three trials per condition, yielding 9,312 recorded responses. Answer-present abstention rates range from 0.0% to 63.0%, whereas replacing the correct answer increases abstention by 5.05 percentage points on average. These findings highlight substantial baseline differences and the need to evaluate abstention frequency and responsiveness jointly. The dataset and benchmark are available at this https URL. 

---
# ARCS: Towards Precise Text-to-SQL via Structured Disambiguation 

**Authors**: Yihao Hu, Yanlin Feng, Naoki Otani, Nikita Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2610.09396)  

**Abstract**: As text-to-SQL systems move beyond demonstrations toward real-world deployment, ambiguity in user questions becomes a primary source of errors. Such ambiguities are often subtle, domain- or data-specific, and can silently cause system outputs to deviate from the user's true intent. Ambiguity is traditionally addressed through conversational clarification, which is often inefficient, cognitively demanding, and poorly aligned with real-world user workflows. We propose structured disambiguation, a new paradigm in which ambiguity is resolved through explicit, constrained interactions rather than free-form dialogue. We construct ARCS (Ambiguity Resolution Corpus for SQL), the first text-to-SQL benchmark featuring naturally occurring, unconstrained ambiguities over real-world databases, with complete annotations of all valid ambiguity points, interpretations, and SQL queries. Experimental results show that text-to-SQL remains challenging in the presence of ambiguity: gpt-6-sol achieves only 51% end-to-end execution accuracy, and no open-source model exceeds 27%. 

---
# The Persona Hierarchy Model: Understanding Contextual Generalization in Fine-Tuning LLMs 

**Authors**: Jiachen Zhao, Zhengxuan Wu, David Bau, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.09384)  

**Abstract**: Language models are routinely fine-tuned under a fixed context, such as a generic system prompt, persona or domain-specific instruction, yet the learned behavior sometimes stays confined to that context and sometimes broadly generalizes to unseen contexts. We propose the Persona Hierarchy Model to explain this: a shared default persona influences behavior across contexts. Under this model, fine-tuning that modifies the shared persona promotes broader transfer, whereas changes to local personas remain more context-specific. Across 120 fine-tuned models spanning four behaviors and 15 training contexts, generalization narrowness positively correlates with the similarity between the training context's persona and the default persona (Pearson's r = 0.72 for Qwen3-4B). Prior fine-tuning under the default context can broaden generalization in subsequent training under other contexts. Aligning contextual responses with default-persona responses produces stronger effects. Finally, we propose persona-preserving regularization (PPR) to confine undesired contextual generalization. In RL, PPR cuts reward hacking from 42-55% to at most 0.2% under every evaluated prompt while retaining accuracy gains. These results support the Persona Hierarchy Model as an explanation for contextual generalization and can motivate future controls on unintended generalization for better alignment of LLMs. 

---
# Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling 

**Authors**: Radha Gulhane, Quentin Anthony, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2610.09372)  

**Abstract**: Mixture-of-Experts (MoE) layers replace the feed-forward block of a Transformer with E expert networks, and each token is routed to k of these experts. Under expert parallelism (EP) the experts are distributed across GPUs, and every MoE layer runs all-to-all collectives in the forward and backward passes to dispatch tokens to their experts and then combine the results. On a cluster with 8 AMD Instinct MI300X GPUs per node, these collectives can take 45% of the training step at EP32 with top-2 routing and 60% with top-6 routing. We find that early in pretraining routers have already learned to assign tokens to experts in correlated patterns, both within a layer and across layers. At top-2, 0.8% of the expert pairs in a layer are selected together by 42% of tokens, and the experts a token selects at one layer predict the experts it selects at the next layer. We use these correlations to keep more token--expert assignments on the token's own GPU, which reduces communication across GPUs and across nodes. Correlated expert placement puts experts that are often selected together on the same GPU. Combined with a dispatcher that sends each token to each GPU once, it removes up to 58% of dispatched rows. Token shuffling applies when sequence parallelism shards tokens across the EP group. It moves each token to the GPU predicted to hold its next-layer experts during the reduce-scatter that follows attention. On one node this raises the share of token--expert assignments served on the token's GPU from 12.5% to 59%. In Megatron-LM, across EP degrees from 8 to 64 with top-2 and top-6 routing, the two methods reduce all-to-all time by 1.16-2.63X and end-to-end step time by up to 1.41X. Neither method changes the models' underlying routing decisions or expert parameters. 

---
# OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models 

**Authors**: Wenjun Wang, Heng Li, Yanggan Gu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09346)  

**Abstract**: Quantization-aware training (QAT) can recover much of the accuracy lost when large language models are compressed below four bits. Existing re- covery stages, however, are commonly optimized on fixed completions or teacher-generated answers, whereas the deployed quantized model condi- tions on prefixes generated by itself. Quantization errors can therefore move the model into states that are absent from offline recovery data. We introduce OnlineQAT, a two-stage framework that first obtains a usable low-bit initialization through block-wise QAT and then performs on-policy distillation (OPD) on student-generated responses. At each visited pre- fix, a frozen full-precision teacher provides a sampled reverse-KL training signal. On Qwen3-1.7B, OnlineQAT obtains the best average among the compared quantized methods: 57.28 at W3A16 and 32.52 at W2A16, im- proving over ReasoningQAT by 2.90 and 0.44 points, respectively. The results suggest that student-visited states provide a useful recovery signal beyond fixed-completion training, particularly at three bits. 

---
# Dialect-Robust Speech Language Models with Synthetic Pseudo-Dialect Augmentation 

**Authors**: Shunsuke Mitsumori, Tomoya Mizumoto, Yusuke Fujita  

**Link**: [PDF](https://arxiv.org/pdf/2610.09321)  

**Abstract**: Speech Language Model (SLM) performance often degrades on dialects due to data scarcity. Conventional text-to-speech (TTS) augmentation struggles to cover diverse dialects as it requires a certain amount of real dialect speech. We propose synthesizing pseudo-dialect speech by converting LLM-generated dialect text via a standard-language TTS model, requiring zero real dialect speech. Additionally, we introduce intermediate standard-text prediction during training, acting as semantic normalization for downstream tasks. We evaluate dialect understanding via dialect-to-English speech translation across Japanese, German, and Chinese dialects. Compared to synthetic standard speech baselines, pseudo-dialect augmentation improves scores for Japanese (from 25.38 to 26.24) and German (from 31.57 to 32.47). Furthermore, the intermediate standard-text prediction effectively bridges the semantic gap, boosting performance to 28.26 for Japanese and from 11.67 to 16.37 for Chinese. These results suggest that our approach scales to various languages without requiring speech resources specific to each dialect. 

---
# Multi-Objective Aligned Small Language Model Framework for SUD Patient Dialogue Generation 

**Authors**: Thushara Manjari Naduvilakandy, Hyeju Jang, Mohammad Al Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09209)  

**Abstract**: Substance Use Disorder (SUD) counseling requires patient responses that reflect underlying cognitive states such as beliefs, coping strategies, and readiness for change. Although large language models (LLMs) can generate fluent text, they often fail to produce cognitively coherent and clinically realistic patient behavior, especially under ethical and data-scarce clinical settings. Moreover, deploying frontier-scale LLMs in healthcare applications presents practical challenges including high computational cost, latency, privacy concerns, and limited deployability in resource-constrained environments, motivating the need for cognitively aligned small language models (SLMs). We propose a cognitively grounded framework for SUD patient dialogue generation that explicitly models and aligns latent cognitive components with patient histories and counselor questions. Our pipeline consists of two stages: cognitive component detection and cognitive component-aligned dialogue generation. To enable effective learning with smaller models, we combine knowledge distillation from high-capacity teacher models, preference optimization from human-annotations, and attention-guided reward shaping. Extensive evaluations using automatic scores like BERTScore, ROUGE, METEOR and BLEU, and LLM-as-judge hit-metrics against both human and teacher-model references show that cognitively informed fine-tuning substantially improves cognitive realization and alignment over a generic instruction-tuned baselines and mental health domain specific SLMs, with particularly strong gains for open-ended cognitive components. 

---
# ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation 

**Authors**: Arkajyoti Chakraborty, Aryan Tayal, Ishika Agarwal, Tanner Sorensen, Justin Chiu, Alessandro Di Bari, Neha Gupta, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2610.09163)  

**Abstract**: Task-oriented conversational agents remain fragile under real world conversation scenarios as they rarely follow a predictable script, especially when users exhibit non-cooperative behavior. Existing function-calling benchmarks often emphasize successful, cooperative interactions and underrepresent adversarial conversation trajectories, thereby limiting the training resources available for developing robust agents. We present ToolRACER, a synthetic data generation pipeline that coordinates user, assistant and tool emulation models to generate and validated multi-turn interactions between a user and an agent. Using \sysn, we construct ToolRACERBench a robust multi-turn conversation benchmark spanning six domains, ranging over 55 varied personas, generating a validated corpus of 5.6K conversation trajectories, with approximately 66\% of conversations containing failure-prone conversation scenarios. We inject adversarial behaviors, producing validated conversational interaction trajectories that capture realistic, robust scenarios. We evaluate models trained on ToolRACERBench against internal benchmarks, as well as on function calling benchmarks such as $\tau^2$-bench, BFCLv3 and ACEBench to evaluate agentic accuracy and robustness. Models trained on ToolRACERBench improve end to end agentic accuracy across $\tau^2$-bench and ACEBench, demonstrating significant gains when mixed with in-domain dataset in small language models for agent capability tasks. 

---
# sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak 

**Authors**: Marek Šuppa, Ivan Vykopal, Andrej Ridzik, Kristián Sopkovič, Natália Kňažeková, Jaroslav Kopčan, Miroslav Blšták, Viktória Ondrejová, Daniel Hládek, Michal Gregor, Martin Tamajka, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2610.09152)  

**Abstract**: Multilingual LLM benchmarks omit Slovak, a morphologically rich West Slavic language of five million speakers, or cover it only by machine translation. We present sk-bench, a native-first Slovak benchmark with 30 datasets (33 scored task variants) across ten skill categories. Eleven resources are introduced or first packaged for generative-LLM evaluation, including IFEval-SK with Slovak-adapted instruction checkers and native Chiby/SKJ1 resources for Slovak grammar and morphology. We evaluate 55 open- and closed-weights models under one harness. The best open model trails proprietary APIs by 12.6 points. Model rankings are similar for native and translated closed-form data ($\rho\geq0.98$), though translation separates the strongest models less well. By contrast, human-authored and LLM-generated QA questions rank models differently ($\rho=0.72$). For Qwen3-14B, continued Slovak pretraining lowers the overall score by 13.9 points. A small instruction set restores three quarters of that loss. Test-time reasoning improves scores by 8.5 to 12.5 points for models of 9B and above. Together, these findings suggest four design lessons for other under-resourced languages: use native data where translation fails, plan instruction repair after language adaptation, enable test-time reasoning before scaling up, and avoid overinvesting in target-language prompts. We release the data and code at this https URL 

---
# Same Text, Different Prediction: Serving-Context Nondeterminism in Text Classifiers 

**Authors**: Santhosh Kumar Kasa, Siva Rajesh Kasa, Sumit Negi  

**Link**: [PDF](https://arxiv.org/pdf/2610.09111)  

**Abstract**: Deterministic inference is essential for reliable and trustworthy machine learning. Prior studies of text generation have shown that changing factors such as batch size, batch composition, hardware, or inference engine can alter the generated text, even when the prompt, model parameters, and sampling randomness are fixed. These differences have been attributed in part to floating-point non-associativity, shape-dependent kernel selection, and other implementation-level differences in numerical execution. However, it remains unclear whether, when, and to what extent the same factors affect text classification. We present a systematic study of serving-context non-invariance in text classifiers, which prior work has measured only through generated text. We train 180 models spanning discriminative, pseudo-generative, and fully generative classifier formulations and evaluate each across four categories of serving contexts, holding the checkpoint and the text fixed. Label stability does not imply score stability. Changing only the batch shape changes no labels across fp32 comparisons, yet under bf16 it moves up to 56.7 percentage points of predicted probability mass, with label changes concentrated at small margins. Fully generative classifiers change more labels than their discriminative counterparts under the same serving changes. We derive sufficient conditions for label stability under each serving change and give a separate mitigation for each mechanism. Our results identify and quantify the serving conditions that must be fixed for reproducible text classification. 

---
# U-Space: Uncovering When and Why Uncertainty Arises in Language Models 

**Authors**: Tobias Braun, Nils Loose, Alexander Herzog, Virginia Ceccatelli, Marcus Rohrbach, Thomas Eisenbarth, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2610.09087)  

**Abstract**: Large language models are informing decisions with ever-higher stakes. As the consequences of their errors grow, a central question becomes harder to ignore: how much can we trust an individual answer? Yet recognizing when to defer remains difficult because language models can present incorrect conclusions with fluent explanations and an authoritative tone. Uncertainty quantification seeks to address this disconnect by estimating the reliability of individual predictions. However, many existing methods require repeated generations or separately trained components, and their scalar estimates do not reveal where uncertainty arises or how it evolves during reasoning. Recent work has also shown that generation length can be strongly associated with uncertainty estimates and correctness, raising the question of how much of an estimator's predictive power comes from uncertainty-specific information rather than output length alone. Mechanistic interpretability offers a way to address these limitations by connecting human-interpretable concepts to intermediate model states. Building on this capability, we introduce the U-Space, a low-dimensional subspace that makes a model's evolving uncertainty measurable and interpretable. We identify semantic anchors for doubt and certainty, map their unembedding directions back into the residual space, and combine their contrasts into an orthogonal basis. The U-Lens projects each token state onto these basis vectors, yielding an interpretable token-level uncertainty map that can be inspected directly or aggregated into a scalar uncertainty score. Our approach requires no correctness labels, repeated generations, or training. Across reasoning benchmarks, its confidence score outperforms established baselines under both standard and length-controlled evaluation and transfers more reliably than supervised estimators. Code: this https URL. 

---
# Quad-State Safety Evaluation of Open-Weight Large Language Models on Non-Canonical Inputs 

**Authors**: Pavan Maddula  

**Link**: [PDF](https://arxiv.org/pdf/2610.09033)  

**Abstract**: Standard safety evaluations of large language models assess harmful requests written in canonical plain text, while models in real-world deployment routinely receive inputs containing emojis, altered spellings, encoded strings, and character-level variations. This work introduces the Adversarial Surface-Form Robustness Dataset (ASRD), comprising 2,100 prompts across seven distinct surface-form families. Five open-weight language models are evaluated across these prompts, producing 10,500 responses. The Quad-State Evaluation Rubric classifies each response into one of four outcomes: harmful compliance, safe response, comprehension failure, or indeterminate. Emoji and invisible Unicode variations cause almost no comprehension failure, with pooled harmful compliance of 20.27% and 17.20% against a 22.87% baseline that is driven mainly by Mistral 7B, whereas leetspeak, encoded wrappers, and hybrid transformations score 2.40%, 0.13%, and 2.40% while comprehension failure rises to 36.47%, 65.60%, and 34.47%. Inspection of raw model outputs reveals three response behaviors: hallucinated benignity, structural collapse, and language drift. Project page: this http URL 

---
# CARE: Certifying Acceleration for Vision-Language-Action Inference 

**Authors**: Rui Liu, Tong Zheng, Jindong Gu, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08917)  

**Abstract**: While vision-language-action (VLA) models have advanced rapidly, running them at every control step remains expensive. Prior work accelerates VLA inference using techniques like action chunking and visual-token pruning, typically evaluating based on latency and average task success. However, acceleration may discard information and break tasks the original policy would solve, a risk hidden by average metrics. Measuring these failures is challenging because action deviations compound over closed-loop trajectories, meaning task failure is only observable across full episodes. We therefore define an acceleration-induced failure via paired rollouts from identical initial conditions, tracking when the reference succeeds but the accelerated policy fails. To manage this, we introduce CARE, an approach for certified accelerator selection. CARE uses paired rollouts on a calibration set to provide finite-sample guarantees that acceleration-induced failure risk stays below a user-specified budget. It deploys the fastest certified candidate, falling back to the reference if none qualify. By relying only on terminal outcomes and measured compute, CARE applies unchanged across diverse acceleration mechanisms, while sequential testing and failure-triggered reference rollouts keep certification affordable. On four LIBERO suites with OpenVLA-OFT, CARE certifies $9.0$--$10.8\times$ speedups while guaranteeing (at $95\%$ confidence) that at least $85.8\%$ of reference-solved episodes are preserved. Under tight budgets, selectors without guarantees exceed the budget in up to $75\%$ of trials, whereas CARE stays within budget and its sequential form uses $78.9\%$ fewer rollouts than exhaustive evaluation. CARE further generalizes to flow-step reduction for $\pi_{0.5}$, and to Qwen3.5-9B and Llama-3.1-8B agents in Crafter. 

---
# Steering Follows Geometry, Not Labels: Emotion Directions in a Full-Duplex Speech Model 

**Authors**: Pulak Kuli  

**Link**: [PDF](https://arxiv.org/pdf/2610.08887)  

**Abstract**: Full-duplex voice agents need to modulate emotion and delivery during real-time conversations, when de-escalating a complaint, carrying urgency in dispatch, softening a clinical result. Emotion and delivery control is well studied for TTS and turn based models through prompt-conditioned synthesis, reference-conditioned synthesis and activation steering; PersonaPlex controls identity in a duplex model but not affect.
We study emotion steering in Moshi, a fully open sourced full-duplex speech language model, across four emotions, using mean-difference activation steering, which costs only a few vector additions per frame and no retraining.
We show that emotion is linearly decodable from Moshi's residual stream, but activation steering is only partially achievable, and unevenly so; as happy, angry and surprise steer towards a shared direction while sad is distinctly steerable. We also show that the shared component across the three emotions cannot simply be projected away from all the emotions equally. 

---
# Tiny-Scale Chinese BERT Pretraining: A Controlled Comparison of MLM, WWM, and MacBERT Strategies 

**Authors**: Yiping Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.08879)  

**Abstract**: Pretraining strategies significantly impact the quality of language models, yet existing comparisons of Masked Language Modeling (MLM), Whole Word Masking (WWM), and MacBERT-style replacement have focused primarily on base-scale models (>=110M parameters). This paper presents a controlled comparison of these three strategies on a tiny-scale Chinese BERT model (4 layers, 256 hidden dimensions, 8.7M parameters). Under identical architecture, corpus (1.29M sentences from Chinese Wikipedia), and hyperparameters, we train three models from scratch and evaluate them across five intrinsic dimensions: perplexity, MLM hit rate, semantic discrimination, grammatical judgment, and contextual sensitivity. At tiny scale, MLM achieves the best overall intrinsic performance (winning 3 of 5 dimensions), while WWM excels in both perplexity (1.27 vs. 2.10, a 39.5% improvement) and MLM hit rate (22% vs. 16%). Notably, MacBERT under a severely limited synonym dictionary (222 entries, 3.3% coverage) exhibits severe perplexity degradation (47.23, 22x higher than MLM), yielding a ranking (MLM > WWM >> MacBERT) that differs markedly from the established base-scale conclusion (MacBERT > WWM > MLM). We further identify a critical evaluation pitfall: MacBERT achieves the lowest training loss (2.17) yet the highest perplexity (47.23), revealing that training loss alone is unreliable under mixed replacement strategies. All models and corpus are publicly available at this https URL. 

---
# LRCC: Generalizing Low-Rank Compression with Conditional Computation 

**Authors**: Thomas Vaitses Fontanari, Maximo Eduardo Rulli, Federico Alvetreti, Donatella Genovese, Simone Scardapane  

**Link**: [PDF](https://arxiv.org/pdf/2610.08858)  

**Abstract**: Low-rank compression reduces the cost of pretrained language models by replacing linear transformations with low-rank factorizations. However, conventional methods use a fixed rank allocation during inference, assigning the same amount of compute regardless of the input token. We introduce Low-Rank Conditional Computation (LRCC), which adds token-dependent computation to pretrained models by training one lightweight router per Transformer block to select among a small set of nested low-rank paths. During training, the low-rank factors remain frozen, and only the routers are optimized. We evaluate LRCC on Llama and Qwen models for language modeling and zero-shot downstream tasks. Within the same average active-parameter budget, LRCC improves the predictive performance over static low-rank compression, including a 7.6 percentage-point gain in average downstream accuracy on Llama-2-7B over static methods. At matched batch-size-1 decoding latency, LRCC improves both perplexity and downstream accuracy on Llama-3.2-1B and remains competitive on Llama-2-7B, without specialized kernels. Finally, we assess the usefulness of assigning a token-wise path by analyzing the routers' path choices. 

---
# QuanLing: Cross-Branch Validation of Language Distance Quantification on Western Romance 

**Authors**: Yiping Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.08851)  

**Abstract**: Quantifying language distance among closely related languages remains a core challenge in quantitative linguistics. Our previous work [1] introduced QuanLing (Quantitative Linguistics via Pretrained Language Models), a quantitative framework combining language distance metrics (sentence embedding distance, tokenization fragmentation rate) with language property analysis (MLM prediction probability), validated on North Germanic (Danish, Norwegian Bokmål, Swedish). This paper extends QuanLing to Western Romance--French, Portuguese, Spanish, Italian--testing cross-branch applicability with the same metric family and aggregation protocol as our North Germanic study, adapted for four languages (English anchor, quadruplet construction). Using 150 four-language parallel sentences, we compute LaBSE sentence embedding distances, tokenization fragmentation rates from four monolingual BERT tokenizers, and mBERT masked language model mutual intelligibility. Results show that Portuguese--Spanish are closest (LaBSE distance 0.0229), French--Italian most distant (0.0338); LaBSE and mBERT rankings agree on 4 of 6 pairs, confirming cross-model robustness. Western Romance shows a wider absolute distance span than North Germanic (0.011 vs. 0.008) but comparable relative ratios (1.48 vs. 1.67), consistent with longer divergence time. French exhibits notably higher MLM predictability (36.12% top-1 accuracy vs. 29.28% for Italian), reflecting its orthography--phonology decoupling. This cross-branch validation provides further evidence for QuanLing's generalizability beyond a single language branch. 

---
# Beyond Risk Prediction: Evidence Grounding and Psychosocial Factor Verification for Explainable Suicide Risk Assessment 

**Authors**: Tianle Hu, Chen Peng, Yi-Hsin Tsai, Takshing Andy Tung, Bingyang Sun, Yenjou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08842)  

**Abstract**: Identifying suicide risk from social networking services (SNS) posts is important for detecting suicide-related signals in online environments. However, risk classification alone provides limited insight into the textual evidence and psychosocial factors behind a prediction. Based on the IEEE BigData 2026 Explainable Suicide Risk Detection Challenge, this study presents a framework consisting of Risk Assessment, Evidence Grounding, and Factor Identification. Risk Assessment uses length-based routing to accommodate posts of different lengths. Evidence Grounding identifies supporting phrases and uses a Risk-Evidence constraint to maintain consistency with the Risk prediction. For Factor Identification, two verifiers are used. The Taxonomy Verifier focuses on factor semantics, whereas the Evidence-Aware Verifier uses factor-specific lexical-semantic cues to select informative positive training units. Their prediction probabilities are combined to produce the final factor predictions. The three tasks are evaluated using task-specific F1 score measures. Risk Assessment achieved a Weighted F1 of 0.8088, Evidence Grounding achieved a test Macro row F1 of 0.7605, and Factor Identification achieved a Macro F1 of 0.5562. The results show that the framework can provide risk predictions, along with supporting textual evidence and fine-grained information on psychosocial factors. Overall, the proposed framework extends suicide-risk assessment beyond risk-level prediction and provides a more interpretable analysis of SNS posts. 

---
# Beyond the Sycophancy Score: How Task, Model, and Pressure Shape LLM Yielding 

**Authors**: Guang Yang, Homa Hosseinmardi, Fengchen Liu, Amir Ghasemian  

**Link**: [PDF](https://arxiv.org/pdf/2610.08840)  

**Abstract**: Large language models (LLMs) often abandon a correct answer, or endorse a user's position, once the user pushes back. This behavior, called sycophancy, is usually reported as a single rate per model, which says little about when it happens or how a user can avoid it. We study the conditions that produce it with 103,939 graded replies from ten configurations: eight LLMs with reasoning disabled, and two of them again with maximum reasoning, all facing the same 200 items, 13 pressure conditions, and four-turn conversations, with every reply labeled by two independent LLM judges. We find that the dominant factors are how costly it is for the model to verify the user's claim, and whether a trained guardrail covers it. Removing this task factor from a logistic model costs 0.485 of McFadden $R^2$, against 0.139 for model family and 0.009 for pressure tactic. Anchored facts are almost never conceded (1.3%), while adoption on logic puzzles rises with the number of clues needed to refute the pushed answer. Personal choices are endorsed in 77.0% of conversations. Most concessions on hard items come from models that cannot reliably solve them; models that can solve them rarely give the answer up. For both models tested, maximum reasoning removes these concessions completely: adoption on deep puzzles falls from 19.2% and 12.5% to 0%. Fallacious or emotional framing adds nothing beyond plain repetition. Three human annotators agree with the judges' consensus on 118/120 calibration items. These results give practical rules for reliable use: simplify hard-to-verify problems and reason deeply, state the question rather than one's preferred answer, ask for evidence on open questions, and choose models by their measured guardrail profile. 

---
# Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News 

**Authors**: Yupei Guo, Jiajun He, Xiaohan Shi, Tomoki Toda, Zekun Yang, Bowen Wang, Yukinobu Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.08835)  

**Abstract**: The spread of fake news may cause severe social consequences. Existing fake news detection methods mainly focus on stylistic variations or incorporate external information such as explanations. However, news articles are often rewritten under different emotional backgrounds while preserving their underlying factual claims, which may affect the robustness of detection models. In this work, we investigate fake news detec- tion under fact-preserving emotional variations. To study this problem, we construct emotion-rewritten test sets and generate explanations from the original news articles as stable background knowledge. We then propose a Gated Cross Attention (GCA) framework that adaptively integrates emotionally rewritten news with the corresponding explanations, enabling the model to focus on informative explanation content while reducing potential mismatches caused by emotional reframing. Experiments on PolitiFact, GossipCop, and LUN demonstrate that the proposed method achieves notable improvements under multiple emotional conditions on PolitiFact and LUN, while maintaining competitive performance on GossipCop. We further analyze the effects of explanation guidance and gating mechanisms under different emotional conditions. Our code and data are available at: this https URL gca . 

---
# CoDR: Training-Free Confidence-Drift Remasking for Diffusion Language Models 

**Authors**: Yue Wu, Qinghe Zhang, Yu Zhang, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08833)  

**Abstract**: Masked diffusion language models (MDLMs) decode by repeatedly committing tokens to masked positions, but these commitments are usually irreversible. A token chosen under sparse, partial context is kept fixed, even when later context no longer supports it. Existing samplers mainly decide when to commit a token, but rarely check whether an already committed token should still be kept, allowing early mistakes to propagate. We trace this issue to confidence drift, where the model's confidence in a committed token drops from its sparse commit-time context to the denser context available later. Based on this signal, we propose CoDR (Confidence Drift Remasking), a training-free and sampler-agnostic refinement pass. CoDR estimates drift for all committed positions in only k forward passes via k-partition probing, then remasks and regenerates only the tokens the model no longer endorses. Across two backbones, four reasoning and coding tasks, and three base samplers, CoDR improves average accuracy across all evaluated model-sampler configurations and improves most individual task settings with modest overhead. Controlled experiments show that the gains come from targeted confidence-drift remasking rather than extra compute alone, and that CoDR uses far fewer forward passes than prior remasking methods. Code is available at this https URL. 

---
# Emo-Jev: Probabilistic Reasoning for Emotion Classification with Jev 

**Authors**: Yazhou Zhang, Junhao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.08829)  

**Abstract**: Jev offers an alternative interface for language understanding: given an input and predefined questions, it returns probabilistic decisions rather than free-form responses. Whether this interface can support effective reasoning for text classification against leading LLMs remains an open questions. We introduce Emo-Jev, a training-free framework with two complementary implementations. Emo-Jev-D decomposes classification into task-specific atomic judgments and composes their probabilities into a final prediction. Emo-Jev-SC constructs multiple judgment paths from complementary perspectives and aggregates their predictions into a consensus decision. We evaluate Emo-Jev on eight datasets spanning sentiment analysis, emotion recognition, sarcasm detection and humor detection, comparing against direct Jev classification and five SoTA LLMs under input/output and chain-of-thought reasoning. Standard Jev achieves 62.93\% average macro-F1 versus 67.28\% for the strongest LLM baseline, with lower observed latency and generally lower cost. 

---
# When Forgetting Looks Like Improvement: Metric Masking in Streaming Diarizer Adaptation and the Price of Rehearsal 

**Authors**: Mo Yu, Yang Liu, Jing Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.08828)  

**Abstract**: Small-data adaptation can improve speech detection while degrading speaker attribution. We study this discrepancy in a released streaming diarizer adapted on 7.5 h of two-party conversation and evaluated across six corpora. Adaptation substantially improves in-domain diarization performance and transfers to an independent corpus. However, this improvement is not consistent across evaluation scenarios as the additional confusion is mainly associated with impaired temporal identity consistency rather than speaker-count errors. A local-remapping diagnostic reveals different patterns of identity degradation across corpora, indicating that adaptation may alter how streaming models maintain speaker assignments over time. Rehearsal reduces the observed degradation but reduces the cross-domain transfer performance. These results highlight the need to jointly evaluate detection accuracy, identity consistency, and retention behavior when adapting streaming diarization systems. 

---
# Child ASR Adaptation with Adult Retention: An Empirical Study 

**Authors**: Houssam Eddine-Othman Lachemat, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2610.08827)  

**Abstract**: Automatic Speech Recognition (ASR) systems often underperform for children and non-native speakers, while adapting adult ASR models to child speech can cause adult-speech forgetting. We study child ASR adaptation with adult retention across Arabic and English. We compare full fine-tuning, LoRA, and post-hoc weight-space merging across encoder--decoder, encoder--CTC, and AudioLLM-based ASR systems. Experiments use Arabic native and non-native child speech, English MyST child speech, and adult benchmarks from MGB-2 and LibriSpeech test-clean. We evaluate recognition quality with WER and quantify the adaptation--retention trade-off using Retention Index, Child Adaptation Gain, and Adaptation Recovery. Results show that child adaptation is necessary, especially for non-native Arabic and English child speech, but direct adaptation often reduces adult ASR performance. Bilingual adaptation is more stable than language-specific adaptation. Weight-space merging often improves the trade-off, especially for encoder--CTC, Whisper, and AudioLLM-based ASR, with LERP favoring adult retention and TIES recovering stronger child gains. For the encoder--decoder model, direct bilingual fine-tuning remains strongest in raw WER.\footnote{Code, and models are available at this https URL. 

---
# Tokka-Bench: Evaluating Tokenizers Across 100 Natural and 20 Programming Languages 

**Authors**: Ben Gubler  

**Link**: [PDF](https://arxiv.org/pdf/2610.08794)  

**Abstract**: Large language models rely on subword tokenizers whose quality varies across languages, yet no standardized multi-metric framework exists for broad comparative evaluation. We introduce Tokka-Bench, an open-source framework that evaluates tokenizers on five complementary metrics -- bytes per token, unique token coverage, subword fertility, word-split rate, and vocabulary composition -- across 100 natural languages (30+ scripts) and 20 programming languages, using language-aware segmentation adapted to each writing system. Comparing seven BPE tokenizers (GPT-2, GPT-4, gpt-oss, Llama 3.1, Gemma 3, Qwen3, and Kimi K2) within individual languages, we find that vocabulary allocation strategy matters more than raw vocabulary size, and that programming-language efficiency has converged among recent tokenizers despite divergent natural-language profiles. The framework, data, and interactive dashboard are publicly available. 

---
# Decoupling Exploration from Optimization in RLVR 

**Authors**: Saif Punjwani, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2610.10536)  

**Abstract**: Modern language models undergo reinforcement learning with verifiable rewards (RLVR) on top of already-trained checkpoints. A key promise of RLVR is the discovery of new reasoning strategies. In principle, a model can sample novel ideas absent from its prior training data. In practice, however, augmenting RLVR with strong novelty incentives has seen limited success and can degrade model quality. Because verifiable rewards supervise only a narrow slice of the model's knowledge and behavior, such degradations are difficult to recover from. Instead, we decouple exploration from optimization in a framework we call Exploration-Distillation (ExpDis). We train one or more explorer policies with a novelty bonus in the reward, filter their trajectories for correctness and quality, and distill them into a separate student policy. The student policy is then trained without a novelty bonus. We repeat the above procedure for several rounds, alternating between exploration and optimization. This decoupling allows us to aggressively scale exploration without degrading the student policy. Across seven mathematical reasoning benchmarks and two model families, ExpDis outperforms DAPO at the same wall-clock budget. Moreover, we observe improved pass@$k$ scaling, indicating that ExpDis produces models that generate more diverse correct solutions. 

---
# Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models 

**Authors**: Mikey Watts, Yuchen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2610.10526)  

**Abstract**: Vision-language-action models (VLAs) are strikingly sensitive to instruction phrasing and do not inherit the language robustness of the vision-language models they are built on. A one-word edit can move success by tens of points: $\pi_{0.5}$ turns on a LIBERO stove 100% of the time for "switch on the stove" and 2% for "switch on the hot plate", and a $\pi_0$ checkpoint finetuned with rephrase augmentation still shows swings of up to 61 points. We characterize this sensitivity with statistically tested single-edit swings and an oracle phrase search, which shows that phrasing alone nearly closes the 21-point gap between in-distribution and out-of-distribution tasks. We then reduce it without modifying the policy. Because the sensitivity is systematic, it can be expressed as explicit rules: we score many phrasings of a few training tasks, have a large language model distill the evidence into ten to twenty rephrasing rules, and at deployment rewrite each incoming instruction once under these rules. The rules improve the frozen $\pi_0$ by 16 to 27% relative on twelve held-out tasks across adversarial, VLM-generated, and human-generated phrasings, with gains concentrated on out-of-distribution tasks. The pipeline replicates on $\pi_{0.5}$ and LIBERO, lifting in-finetune success from 93.6% to 97.8%. The method requires no retraining and no per-step verification, and applies zero-shot to unseen tasks and instructions. Project website: this https URL 

---
# Validity Without Ground Truth: What Stated-Preference Economics Offers the Evaluation of Language Models 

**Authors**: Daniel Robert Kling Alexander, Catherine Louise Kling  

**Link**: [PDF](https://arxiv.org/pdf/2610.10506)  

**Abstract**: Many of the questions now put to large language models have no correct answer to score against: what a policy is worth, which option a user should choose, how to weigh competing values. Stated-preference economics has faced this problem for decades. It judges survey responses without knowing the true value, through a framework of validity and related concepts: content, construct, and criterion validity, reliability, incentive compatibility, and consequentiality. We argue that this framework is a general method for evaluating language models, and we set out what each concept means for LLM evaluation. We demonstrate the approach using a published water-quality stated preference economic valuation survey (Vossler et al. 2023) administered to six models. In this economic application, the validity tests take the form of predictions from economic theory: demand should slope down, and willingness to pay should respond to the scope of the good and to income. The tests separate the models sharply. Two older models fail the most basic test at a household income level of \$75,000, and the two newest pass every test of theoretical validity we can score, but diverge on convergent validity. Passing validity tests shows that a model's answers are coherent, not that they are correct. 

---
# RunningTab: Direct Workspace Interaction with Environment-Side Tabs 

**Authors**: Jinheon Baek, Soyeong Jeong, Yumin Choi, Dongsu Han, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10444)  

**Abstract**: Much knowledge work produces new deliverables from files a workspace already holds, and LLM agents are beginning to take such work over. Through direct corpus interaction, an agent can search and read any of those files from a terminal with no indexing, and producing a deliverable from many of them in this way is what we call direct workspace interaction (DWI). Reaching the files, however, is only half the task: nothing keeps track of what the task asks for, what has been read, and what was listed but never opened, all of which slip through the context window without leaving a trace, so an agent may extract a figure and still deliver a report without it. To address this, we present RunningTab, a framework that equips direct workspace interaction with an environment-side tab: a per-task record of what the task still owes, kept by the environment alongside the agent. Specifically, the agent adds its requirements, while the environment records every file read as an excerpt with its provenance and every listed but unopened file as a candidate; the agent can then see each requirement beside its best-matching excerpts and top unopened candidates, resolve it against matching content or set it aside with a reason, and, should it try to finish with requirements still open, receive them in a finish check. We validate RunningTab on three benchmarks with three LLMs, where it consistently outperforms plain DWI and baselines that keep the record in the model, while its tab usually holds the values a deliverable needs once seen. 

---
# Which Rollout Taught It That? BehaviorTrace and the Limits of Training-Data Attribution in Online RL 

**Authors**: Amit Nautiyal  

**Link**: [PDF](https://arxiv.org/pdf/2610.10422)  

**Abstract**: When reinforcement learning teaches a language model a new behavior, can we find the training rollouts that taught it? And when an attribution method says it can, how do we know the answer is real? We study both questions on online RL fine-tuning with GRPO, using a planted behavior with a known cause. We release BehaviorTrace, an open evaluation harness that combines full-gradient sketching, the planted-behavior setup, and controls for gradient magnitude, fluency, headroom, and variation across seeds and generation draws. Across three seeds on Qwen2.5-1.5B, much of the apparent attribution signal comes from confounds. A control that ranks training steps by gradient size alone, with no behavior target, reaches 4.2 to 4.5 times chance and matches or beats the best targeted estimator on two of three seeds. At saturated checkpoints, model fluency predicts the behavior label at least as well as every gradient method we compared it with. Once fluency is controlled, the per-rollout results change from seed to seed and from one generation draw to the next, so a single run cannot settle the question. One signal does hold on all three seeds. The gradient of the trigger tokens aligns with a target built where the behavior actually occurs. We turn these findings into a checklist for evaluating attribution in RL. We test existing estimators, including GAS (renormalized TracInCP) and a TRAK-style estimator, and do not propose a new one. 

---
# Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds 

**Authors**: Yunxiao Zhao, Changxiao Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.10411)  

**Abstract**: Speculative decoding accelerates large language model inference by using a low-cost draft model to propose tokens that the full-size target model verifies in parallel. Parallel and semi-autoregressive (semi- AR) drafters improve drafting efficiency by proposing an entire block in a single forward pass, but training them raises a new difficulty: the draft distribution for a given position depends on where the decoding round starts, and where rounds start depends on how many tokens earlier rounds accepted. Existing training objectives typically rely on block-local surrogates that ignore this cross-round coupling, and therefore do not directly optimize the global decoding efficiency. In this work, we develop a theoretical framework for training and evaluating these drafters by representing speculative decoding as a Markov reward process. This formulation yields the Expected Decoding Rounds (EDR) objective, which weights local rejection costs by state occupancies and exactly equals the expected number of decoding rounds. Unlike prior surrogate objectives, EDR introduces no auxiliary hyperparameters. We then derive an exact temporal-difference gradient that supports unbiased stochastic optimization from target-model rollouts. The same framework also yields an exact offline evaluator for round counts, enabling paired drafter comparisons on shared target rollouts without running speculative decoding. Finetuning two state-of-the- art drafters, DSpark and DFly, with EDR consistently improves mean accepted length and outperforms existing training objectives across nine benchmarks spanning math reasoning, code generation, and chat. 

---
# Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models 

**Authors**: Maverick Morales, Tomáš Dominik, Vermut Gao, Katrina Shirey, Paulius Rimkevičius, Aaron Schurger, Uri Maoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.10405)  

**Abstract**: Monitoring the chain-of-thought of reasoning artificial intelligence (AI) models remains a key approach to detecting deception and other forms of misbehavior in such models. However, semantic chain-of-thought monitoring depends on reasoning traces being legible and sufficiently faithful to the underlying computations that produced the model's behavior, not to mention accessible. Moreover, there is increasing evidence that chain-of-thought outputs may soon become illegible or unfaithful, if they even remain accessible. Based on cognitive load theory, we investigate a lower-bandwidth signal -- the number of reasoning tokens generated -- which does not require access to the content of the reasoning trace. Three reasoning-capable large language models answered 210 multiple-choice questions -- across analytic, descriptive, and normative reasoning types as well as moral and non-moral domains -- under system prompts instructing them to respond truthfully, falsely, or without regard for truth. Across all three models, truth-directed responding elicited fewer reasoning tokens than both lie-directed and truth-indifferent responding. These findings show that explicitly prompted untruthful response policies can produce robust group-level differences in test-time reasoning-token use. While not yet establishing reasoning-token count as a detector of spontaneous deception or general misalignment, our results are a proof of concept that it can serve as a simple, content-independent candidate signal for differentiating untruthful from truthful model behavior when raw reasoning traces are unavailable or unreliable. Future work should test instance-level detection rates, out-of-distribution generalization, learned deceptive policies, hidden objectives, and robustness under adversarial pressure. 

---
# Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation 

**Authors**: Yibei Guo, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10368)  

**Abstract**: Adaptive computation aims to improve language-model inference by tailoring execution to each input. For layer programs, oracle evaluations use known answers to estimate the potential gain from this flexibility, before a practical selector is available. However, a gain from selection does not by itself explain why the chosen programs help. This study examines this distinction using 32 layer-skipping and repetition programs on two models and 4,413 multiple-choice items. The analysis compares their gains over a fixed action selected without the evaluation prompt with those of input-blind perturbations at the same sites, re-evaluating selections on another prompt. With shared option order, the controls give 10.2-11.8 and 15.6-19.4 percentage points of headroom on Qwen3-4B-Base and Llama-3.1-8B, exceeding the real programs' 9.0 and 10.1 in all three random-direction draws per model. They match answer-change rate only, and the ordering depends on the menu: in post hoc comparisons, real programs lead on Llama's repeat-only menu in every draw. A smaller KL-calibrated comparison, including an input-dependent control, favours real programs in point estimate, with inconclusive corrected tests. Fixed letter offsets produce headroom of similar scale. Rotating options sharply reduces both families' headroom, while leaving positive real-minus-control differences of 1.4-2.3 and 3.7-4.5 points; their magnitudes and statistical support depend on further adjustments and the reference. A supplementary generated-answer test finds that search-selected programs keep a 26.0-point advantage over programs selected for other problems after rewording, without a placebo comparison. These results show that substantial headroom can persist across prompts with shared option order without establishing a benefit specific to the selected layer computation; neither ordering against these controls identifies that benefit. 

---
# SemanticFold: Latent Sequence Compression SeparatesLanguage Modeling, Decodability, and Reasoning 

**Authors**: Mingyan Liu, Min Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10304)  

**Abstract**: We study whether latent sequence compression of prompt prefixes preserves the capabilities that large language models rely on during inference. We introduce SemanticFold, a compression scheme that folds prefix hidden states at learned boundaries, and evaluate it across five model scales: Qwen3-1.7B, Qwen3-8B, SmolLM2-1.7B, Pythia-1.4B, and Pythia-6.9B. We use a fixed-target protocol: a frozen prefix is executed natively or compressed, and both arms teacher-force identical continuation tokens. This design rules out target-selection explanations for likelihood changes. We examine five endpoint families: fixed-target negative log-likelihood, finite-label reasoning accuracy, linear probe accessibility, open-ended generation, and systems-level memory and latency. We find that compression moves these endpoints non-monotonically and that they do not share a single compression threshold. On Qwen3-1.7B at compression ratio R=1.7, compressed-minus-native mean NLL decreases by 0.135 under paired bootstrap with 10000 draws. On SmolLM2 at R=1.2, the mean change is 0.013 higher than native. On both Pythia checkpoints, NLL is effectively unchanged. An NLL decomposition separating sequence shortening from the learned residual transform shows that the favorable Qwen likelihood is attributable primarily to residual adaptation rather than to shortening alone. MLP-only, which applies the transform without shortening, achieves 0.082 lower NLL than Full SemanticFold. Linear probe accuracy and macro AUC change by less than 0.03 in absolute value across conditions, with confidence intervals crossing zero. We conclude that preservation under latent compression has no single scalar certificate: language-model fit, decodability, and reasoning behavior answer different questions and can move in different directions under the same compression operation. 

---
# PatchBench: Measuring Collateral Damage in Activation Patching 

**Authors**: Alexi Canesse, Mathis Le Bail, Maël Jenny, Clément Elliker, Mahammed El Sharkawy, Sonia Vanier  

**Link**: [PDF](https://arxiv.org/pdf/2610.10276)  

**Abstract**: An LLM safety patch can pass a benchmark while still being a poor repair. This risk is especially acute for jailbreak repairs, where the goal is to correct a specific unsafe behaviour without changing unrelated behaviours. A patch may block exact evaluation prompts yet fail on close harmful variants, or suppress harmful behaviour by over-refusing benign prompts that share its wording or structure. Existing protocols primarily test whether models can be broken, while aggregate metrics (attack success, refusal rates, global capability) cannot distinguish selective repairs from broader local suppression. To address this gap, we introduce PatchBench, a benchmark of empirically observed model-specific jailbreak failures inducing actionable harmful answers. Starting from 27,870 prompts from 37 public datasets, we curate 15,314 English prompts and query 8 open-source instruction-tuned models. Combining WildGuard filtering, pairwise Elo ranking, and manual verification, we retain a curated bank of 400 high-confidence jailbreak failures. We further introduce PatchBench-Local, an evaluation protocol testing whether a patch is behaviourally precise. For each harmful source prompt, PatchBench-Local generates three families of local neighbours: harmful variants preserving malicious intent, benign prompts with matched structure, and benign prompts reusing key harmful terms. It evaluates harmful-neighbour correction and benign-neighbour preservation, distinguishing selective repair from broader local suppression. Evaluating four activation steering methods with PatchBench-Local and MMLU shows that global capability can remain nearly unchanged while local benign regressions are severe, confirming aggregate metrics miss important collateral damage. PatchBench-Local provides a more precise basis for developing and comparing jailbreak repair methods. 

---
# From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification 

**Authors**: Yue Qiu, Zekang Du, Yiqun Diao, Bingsheng He, Qinbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.10227)  

**Abstract**: While Large Language Models (LLMs) possess rich world knowledge and impressive generalization capabilities, their direct application to tabular data classification is hindered by high inference costs and limited interpretability. In contrast, decision trees are fast and transparent but often underperform in low-data regimes. In this work, we propose a novel framework that bridges these paradigms by distilling LLM knowledge into interpretable decision trees under a few-shot learning setting. Instead of directly prompting the LLM to generate full trees, which is often unstable and inefficient, we develop a three-stage paradigm that prompts the LLM to generate rules and organize the rules into a tree. Experiments on multiple real-world tabular datasets demonstrate that our method achieves superior accuracy and interpretability with significantly lower prompting overhead compared to existing baselines. 

---
# GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning 

**Authors**: Jingyao Zhang, Yun Li, Lu Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.10201)  

**Abstract**: Joint spatial-geometric and analytic function reasoning requires translating a perceived spatial configuration into a symbolic function whose executed curve satisfies geometric constraints. We present GAGR-Lab, a framework for measuring this capability through Cartesian game scenes, explicit function semantics, and authoritative Rust trajectory execution. It distinguishes spatial perception, metric grounding, geometric relations, function interpretation, function construction, and constrained synthesis. We specify four configurable scene-difficulty presets and a prospective 24-cell diagnostic design, while reporting only the subset actually evaluated. A bounded pilot of one hosted model (Llama 3.2 11B Vision Instruct) using two API credentials as execution replicas yields 72 balanced games with 432 attempts, 429 valid provider responses, and no target hits; exploratory ordinary-function prompt variants also fail to hit, while the structured localization interface yields no scoreable outputs. A privileged analytic search control independently succeeds on 600 directional cases from 300 generated scenes, with exact repeatability and 1,200 successful vertical-reflection or translation checks. The framework separates serving reliability, symbolic compliance, and geometric success, and preserves exact model-visible inputs and realized paths. A staged protocol outlines diagnostic calibration, held-out replication, multi-model comparison, and paired robustness tests. The contribution is an operational research framework with an executed pilot and a clearly identified prospective study plan; the full difficulty matrix and comparative model results remain untested. 

---
# YANchor-4B: Effective Long-Horizon Reasoning in O(N) Time with O(1) Memory 

**Authors**: Huishan Ji, Hua Xu, Weiming Zhang, Qirui Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.10118)  

**Abstract**: Long-horizon reasoning demands access to earlier information at a manageable generation cost. Full-history attention incurs growing storage and computation, while recurrent compression can lose precise details. Therefore, we present YANchor-4B, a general-purpose recurrent model that preserves crucial memory as ANchors for retrieval during subsequent reasoning. Beyond $O(N)$-time generation and $O(1)$ memory, YANchor enables effective long-horizon reasoning through its multidimensional memory mechanism. For example, on challenging math problems, it achieves 82.93% mean pass@1 on AIME 2024--2026 and 63.64% on HMMT, substantially outperforming linear-time, constant-state counterparts, including larger models. It also delivers several-fold higher batched long-generation throughput than Transformer and hybrid baselines on H100. Furthermore, evaluations across dozens of benchmarks demonstrate YANchor's superiority in general-purpose capabilities. 

---
# SkillSandbox: Skill Verification via Dynamic Scenario Synthesis 

**Authors**: Serin Kim, Kwangwook Seo, Dokyung Song, Jinyoung Yeo, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.10088)  

**Abstract**: Self-evolving agents distill task-solving experience into skills for future reuse, but these skills can encode incorrect procedures or non-transferable knowledge. It is therefore critical to verify each skill's reusability: whether its guidance remains useful beyond the experience from which it was distilled. Such verification requires observing how a skill affects execution in new tasks, yet existing tasks may not expose the situations where the target skill can actually be exercised. To construct such situations, we propose SkillSandbox, a framework that dynamically synthesizes a task and its environment for each skill that are skill-relevant yet novel. A Proposer specifies the conditions to preserve and the source-specific details to vary, a Builder constructs an executable scenario, and a Verifier compares executions with and without the skill. The Verifier assesses executability, utility, and efficiency to assign a Keep or Reject verdict, determining whether the skill enters the library. Across ALFWorld and WebShop with three models, SkillSandbox consistently yields the strongest downstream performance and improved execution efficiency. Further analyses examine whether these gains reflect accurate assessment of skill reusability and identify which components of SkillSandbox contribute to them. 

---
# Sensitive-Topic Leakage Through LLM Routing Metadata: Measurement and Mitigation 

**Authors**: Teng-Ruei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.09981)  

**Abstract**: LLM routers pick a cheap or expensive model per request by its content, and many gateways and some cloud platforms can log that choice with content logging off. We measure this privacy channel beyond token counts, accounting for noisy labels and repeated prompts. We run pre-registered studies on 1.7 million real requests (WildChat-1M, LMSYS-Chat-1M) with two cost/quality routers and a domain router, survey eleven systems' logging, and test post-processing defenses. At matched length, the shift's direction depends on category and router. For RouteLLM at the 50% operating point, harassment and self-harm requests reach the strong model 19 points less often than comparable ones on prompts unseen in exploration, medical requests (exploratory: LLM labels failed their gate) 31 points less often on distinct prompts (both post hoc), and sexual requests 10 points more often (secondary); the other router's four are negative. Twenty RouteLLM decisions separate frequent medical askers with AUC 0.71, exploratory and below the pre-registered primary endpoint's 0.75 (domain router: 0.92, an upper estimate). Per-category length-matched parity with accurate labels removes the gap on real traffic, costing at most 0.2 accuracy points on RouterBench (post hoc), where routers' gaps on sensitive subjects (13-42 points, pre-registered) exceed those of an oracle routing by realized accuracy gain (1-11, post hoc). Per-conversation stickiness, per-user budget bands, and pooled parity fail, the last as categories' shifts differ in size or sign. A post hoc exact per-user rate hides only even-prefix strong counts and forfeits most self-assessed routing value; it preserves odd-position decisions, from which a post hoc log attack reaches AUC 0.73 after 20 RouteLLM requests (exploratory). 

---
# Inverting Multi-Vector Visual Document Indices 

**Authors**: Zhuchenyang Liu, Yao Zhang, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09920)  

**Abstract**: Prevailing multi-vector visual document retrievers store each page as about a thousand patch vectors, often in vector databases run by a third party. Since no one can read a page from its vectors, this index is easily treated as less sensitive than the page. However, because the index keeps one vector per patch in raster order, and each vector is computed by a vision-language model pre-trained to read documents, we hypothesize that whoever runs or breaches the store can reproduce a page from its index alone. We frame inversion as conditional document image generation and infer from the vectors what the attack needs: the encoder, the page shape and, for shuffled vectors, their order. On the ViDoRe v3 benchmark, pages inverted from raw indices recover 47% of the words and 45% of the sensitive tokens. Used as queries against the stored indices, they rank their source page first 98.4% of the time. We test two cheap protections, token pooling and shuffling, which both cut word recall to about 8%. A model that restores the order of a shuffled index raises the share of source pages ranked first from 3.8% to 93.5%, while inverting a pooled index remains open. To test generalisation, we apply the same attack unchanged to another multi-vector retriever: its inverted pages still rank their source page first 70.2% of the time, though its word recall stays below a nearest-neighbour baseline. Multi-vector visual document retrievers are therefore vulnerable to inversion through their stored index, which should be protected like the documents it encodes. 

---
# Constrained-Action AI Remediation for SIEM/XDR via a NeMo-Guardrails Proxy 

**Authors**: Georgios Koutidis, Nikolaos Kekatos, Tom Nianios, Alexios Lekidis  

**Link**: [PDF](https://arxiv.org/pdf/2610.09906)  

**Abstract**: Security Operations Centers (SOCs) for information technology and operational technology share one incident-response problem: a flood of correlated alerts and too few analysts. Large Language Models (LLMs) are increasingly proposed as reasoning engines that triage alerts and, in autonomous deployments, issue commands that block IPs, kill processes, or quarantine files on production hosts. This coupling introduces a new risk: a single adversarial alert can become a remote code path through the LLM's reasoning, leading it to recommend an action the SOC then executes. We present a constrained-action architecture with two coordinated layers: (i) a SIEM/XDR control plane that grounds remediation in correlated host events and confines the LLM's output to a closed intent vocabulary whose templated commands are executed by thin endpoint agents, backstopped by an argument validator; and (ii) a NeMo-Guardrails proxy that wraps the SOC-analyst LLM with input- and output-rail policies, evaluated out-of-the-box against a SOC-specific adversarial corpus we release. The stock proxy lifts injection recall from 25.0% to 94.5% at a 0.1% false-positive rate, and a live red-team exercise confirms that the closed intent vocabulary and argument validator contain the observed LLM failure modes before any command crosses the trust boundary. As an architectural fit (not yet a measured operational-technology deployment), the constrained-action property suits critical-infrastructure settings where a wrong remediation has physical, not merely operational, consequences. The loop is best run human-in-the-loop or delayed: the measured rail latency keeps inline control out of scope. 

---
# LiveMACE: Process-Aware Evaluation of LLM Agent Capabilities in Evolving Markets 

**Authors**: Jun Zhao, Leiming Fu, Yanbo Wen, Yiding Wang, Xuantong Liu, Yang Shu, Yuyang Lu, Xuanran Xing, Jingqi Tong, Hao Xu, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09872)  

**Abstract**: Evaluating agents by outcomes alone can obscure the capabilities that produce them. This problem is especially pronounced in evolving environments, where outcomes reflect a closed-loop interaction between agent behavior and changing external conditions. We introduce LiveMACEBench, a process-aware benchmark that uses live financial markets as a naturally evolving testbed for persistent LLM agents. Five frontier LLMs operate along continuous trajectories under matched Tool Use, Persistent Memory, Rule Following, and Multi-Agent Collaboration configurations. We evaluate them through both realized outcomes and mechanism-specific diagnostics derived from complete decision traces. Across 30 days of live evaluation, we find a pronounced outcome-capability gap: realized returns often diverge from capability-specific measurements, and similar outcomes can arise from markedly different patterns of mechanism use. Trace-level diagnostics further expose distinct bottlenecks across capabilities, demonstrating that mechanism access, effective mechanism use, and downstream performance are not interchangeable measures of agent capability. LiveMACEBench makes this distinction measurable, turning live markets from a performance leaderboard into a diagnostic environment for agent capability 

---
# From Expert-Guided Proof Search to Automated Open-Problem Solving 

**Authors**: Adrián Zámečník, Matěj Kripner, Martin Koutecký, Martin Balko, Jan Grebík, Pavel Hubáček, Robert Šámal, Václav Rozhoň  

**Link**: [PDF](https://arxiv.org/pdf/2610.09769)  

**Abstract**: Large language models are increasingly contributing to mathematical research, where progress often depends on efficient proof search, incremental improvements and careful verification. We describe Bolzano, a multi-agent open-source system that uses parallel prover agents with a verifier agent and maintains a human-readable research state. Initial manual use on expert-selected problems yielded 8 results whose proofs were checked by domain experts. Motivated by these case studies, we ran Bolzano without problem-specific human guidance on about 3,800 open problems extracted from four sets of papers, solving about 200 open problems. One experiment used papers accepted to STOC 2026, a top conference in theoretical computer science. There, we answered four questions raised in the papers, as confirmed by their authors. 

---
# PARC-Loc: Text-to-Point-Cloud Localization with Partial Assignment and Relational Consistency 

**Authors**: Shengkai Ma, Zhenyu Hou, Weihua Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09761)  

**Abstract**: Text-to-point-cloud localization estimates a position in a city-scale 3D map from descriptions of surrounding objects. Existing coarse-to-fine methods retrieve submaps using aggregate learned compatibility and then localize within a selected submap. However, repetitive or similar urban objects can inflate the embedding similarity between the query and multiple submaps, even when the instance layout within a submap violates the query description. Meanwhile, query-relevant instances often span submap boundaries, leaving the retrieved submap with incomplete contextual evidence. We term these failure modes layout-inconsistent aliasing and boundary evidence incompleteness, respectively. To address them, we propose PARC-Loc, a coarse-to-fine localization framework built on Partial Assignment with Relational Consistency (PARC). PARC jointly models hint-object compatibility and pairwise spatial relations, allowing unmatched elements while favoring assignments consistent with the queried layout. At the coarse stage, its candidate-level assessment complements neural similarity for layout-consistent submap selection. At the fine stage, the context is expanded with query-relevant instances from adjacent submaps, while PARC yields object-level matching weights that guide cross-modal attention. Extensive experiments on KITTI360Pose and CityLoc show that PARC-Loc outperforms conventional coarse-to-fine baselines. On KITTI360Pose, our method improves Top-1 localization recall at 5 m from 0.50 to 0.67, achieving a 34% relative gain over the strongest baseline. 

---
# Towards Explaining Query Expansion Performance in Information Retrieval 

**Authors**: Sourav Saha, Aditya Dutta, Soumajit Pramanik, Mandar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2610.09724)  

**Abstract**: Query Expansion (QE) techniques have long been widely used in Information Retrieval (IR) to address the vocabulary mismatch problem. They remain relevant in modern retrieval systems, including those based on large language models (LLMs). However, no single QE method consistently outperforms others across all queries. This work seeks to explain the variation in QE performance through two complementary perspectives. The first is the concept of an Ideal Expanded Query (IEQ)--a hypothetical query that maximizes retrieval effectiveness with a downstream BM25 retrieval model. The second is a separability perspective, which quantifies how distinctly relevant and non-relevant documents are scored for a given expanded query using Cohen's (d). We develop a separability measure and practical formulations to approximate the IEQ and investigate how these factors relate to retrieval effectiveness. Extensive experiments on the TREC Robust collection, TREC DL 2019-2022 passage collections, and TREC DL 2019-2020 document collections reveal several interesting patterns. In particular, we find that expanded queries that are closer to the ideal expanded query tend to achieve higher retrieval effectiveness. We further show that the separability of relevant and non-relevant documents provides a complementary perspective for understanding QE performance. 

---
# When Rank Rises as LLMs Degrade 

**Authors**: Zhaohui Geoffrey Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09647)  

**Abstract**: Post-training adapts language models in non-stationary environments. Practitioners monitor representation health with RankMe and related spectral statistics, often assuming that rank falls when representations degrade. We show that this assumption is unsafe for LLM post-training. In a controlled study of Qwen3-0.6B with four degradation modes and three seeds, data duplication worsens held-out loss by 75% relative to healthy while increasing both original and centred RankMe; the latter changes by 13.5 pooled standard deviations. Covariance effective rank rises to nearly twice its healthy value. This failure is spectral dispersion rather than collapse, so a one-sided monitor rates the worst checkpoint as the healthiest. By contrast, a learning-rate misconfiguration lowers centred RankMe and k95, while uncentred RankMe is inconsistent across seeds. Direction is therefore a property of the regime-statistic pair and cannot be fixed by recalibration alone. We also distinguish two often-conflated statistics: RankMe normalises singular values, whereas covariance effective rank normalises eigenvalues. On raw intermediate-layer states in the pretrained model, massive activations pin the latter near 1 out of dimension d while RankMe retains usable range. We then test a two-sided, multichannel sequential monitor with separate calibration and test data. In a pre-registered shared-prefix, leave-one-seed-out evaluation, it detects all three damage regimes in every fold 10 to 60 steps after the fork and separates dispersion from downward-rank damage by firing direction. However, it never precedes held-out probe loss, and calibration with two seeds produces false alarms on the held-out healthy seed. Spectral monitoring can diagnose failure regimes, but it does not warn earlier than held-out loss, and validity claims require held-out healthy data. 

---
# Coding-Agent Benchmarks Should Match Their Users' Task Flows 

**Authors**: Igor Slinko, Yaroslav Golubev, Sergey Titov  

**Link**: [PDF](https://arxiv.org/pdf/2610.09633)  

**Abstract**: The evaluation of coding agents generally strives to be as realistic as possible. In our study, we collect 4,782 agent sessions of real software engineers in JetBrains IDEs, which we call Production Sessions. Since our subject is interactive agents, we study the sessions with at least three user messages (33% of the sample). These long sessions differ from issue-derived benchmark tasks in two ways: (i) user requests span a far wider mix of task types - questions about the project's code, planning, review, refactoring, execution - and (ii) users switch between types throughout a session. Long-session samples from three public interaction corpora exhibit markedly different Task Flows (the distributions of session lengths, task types, and type-to-type transitions), so no single interaction distribution is universally realistic: benchmarks should name a target use case and calibrate to measurements from it. We present SWE-TaskFlow, an approach for transforming any issue-derived benchmark: it preserves the verified tasks and tests while steering the interaction toward a target Task Flow through prompt splitting and verifiable repository QA, with a TaskFlow Alignment Score (TFAS) for selecting among generated trajectories. In a pilot on 700 SWE-Bench Pro tasks, solving the task sequentially in several steps approximately doubles agent cost without a stable change in resolve rate: the interaction protocol itself is an important dimension of evaluation. 

---
# Collaborative Reasoning Distillation via Cross-Feedback and Coherent Curation 

**Authors**: Taehoon Kim, Seunggeun Cho, Dongsu Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.09587)  

**Abstract**: Reasoning capabilities are critical for advancing Large Language Models, yet current approaches either require massive computational budgets or struggle to effectively distill reasoning to smaller models. Standard distillation methods rely on outcome-based rewards, failing to distinguish between sound reasoning and lucky guesses. We propose Collaborative Reasoning Distillation (CRD), a framework that enhances reasoning in compact models through three innovations: (1) interactive cross-feedback where teachers iteratively critique each other's reasoning, (2) fine-grained step-wise quality assessment capturing logical validity independent of final answers, and (3) coherence-aware step stitching that synthesizes complementary strengths. Students are trained via Reasoning Quality Optimization (RQO) with budget constraints. Our model, CRD-4B, achieves 97.3% on MATH-500 and 70.3% on AIME'25, surpassing baselines while using only 50K training examples, up to 12 times smaller than the datasets of comparable models. 

---
# Constitution-Guided Watermarking 

**Authors**: Toluwani Aremu, Samuele Poppi, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2610.09552)  

**Abstract**: Watermarking enables language model providers to identify text generated by their models. However, its desired properties can conflict (\ie~stronger watermark signals can degrade text quality), while designs that resist editing may also facilitate forgery. Providers address these trade-offs by choosing configurations that balance competing objectives or prioritize particular properties. Either approach imposes a shared operating point on requests with different requirements, potentially sacrificing quality where wording preservation matters or robustness where reliable attribution is essential. To allow flexible and adaptable designs, we introduce \emph{Constitution-Guided Watermarking}, a framework that selects request-appropriate trade-offs from provider requirements, listed as natural-language principles. \emph{Offline}, a pretrained reasoning agent examines constitutional rules alongside watermark implementations and iteratively refines rule-specific configurations using empirical feedback. \emph{At deployment}, a separate monitor identifies applicable rules and retrieves the corresponding policy, including watermarking exemptions, without modifying the serving model. Furthermore, our framework supports offline parallel optimization and refinement of rule-specific configurations based on evolving provider requirements without affecting deployment, and binds each deployed configuration to its evaluation evidence, making deployment decisions auditable. In a proof-of-concept evaluation using KGW and a five-rule constitution, our framework selects configurations responsive to provider priorities and improves post-paraphrase detection on robustness-prioritized requests by up to $14$ percentage points over fixed configurations, while matching or exceeding all baselines in aggregate quality and clean detection at a nominal $0.1\%$ false-positive rate. 

---
# Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets 

**Authors**: Arjun Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2610.09541)  

**Abstract**: Signals that predict whether a chain-of-thought (CoT) trace is correct are compared by AUC, but deploying one requires a threshold with a guarantee. We ask what distribution-free selective guarantees deliver for CoT verifiers at realistic calibration budgets of tens to a few hundred labelled problems, using seven open models, five verifier signals and 37,000 graded traces. The central observation is validity by abstention: an $(\alpha,\delta)$-valid procedure that issues a certificate with probability $P_{\rm fire}$ bounds the failure probability of an issued certificate only by $\delta/P_{\rm fire}$, so a certificate that rarely fires can be valid and wrong every time it is used. In a simulation with known risk the standard certificate fails in at most 0.3% of calibration draws but in up to 69% of those in which it fires. A certification floor and a lattice condition for Benjamini-Hochberg conformal selection explain why certificates abstain at these budgets, and the data bear them out: the standard certificate returns nothing or a large accepted set, and an unreadable residual-stream probe buys two to three times the coverage of the readable signals, an edge a cross-fitted reconstruction cannot recover linearly from the readable features. We then give a floor-started fixed-sequence certificate, valid without monotonicity assumptions, that covers more than the Bonferroni certificate on every model-signal pair and raises coverage at the non-vacuous target $0.75\pi_0$ from 0.05 to 0.16, although the floor keeps absolute coverage small. Finally, a certificate cannot see what matters after deployment: under benchmark shift the error among accepted traces tracks the new task's base error, and under best-of-$n$ selection against the verifier it rises past the target while the empirical failure frequency stays below $\delta$, because abstention absorbs the failures. 

---
# Mitigating Accent-Language Confusion in Self-Supervised Speech Representations for Language Identification 

**Authors**: Minu Kim, Jihwan Lee, David R. Mortensen, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09486)  

**Abstract**: Spoken language identification (LID) aims to recognize the target language regardless of accent. In practice, however, LID models fine-tuned from self-supervised speech representations frequently confuse accents with languages, misclassifying non-native (L2) speech as the speaker's first language (L1). We show that non-native speech representations lie between native target-language and native L1 poles, causing systematic misclassification. To address this, we introduce a geometric projection that estimates an L1-bias direction solely from native speech and removes it before the frozen LID head. Across five MMS-LID models and non-native corpora, this projection substantially improves target language identification for L2-accented speech while preserving predictions for native speech. These results show that accent-induced L1 bias can be corrected directly within the representation space without L2 training data or model adaptation. 

---
# CHASE: Channel-Aligned Structure Exploitation for Geometry-Aware Model Engineering 

**Authors**: Wei Wang, Wei Jiang, Ziran Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09476)  

**Abstract**: Geometric and Spectral Alignment (GSA) characterizes trained networks through spectral concentration, physical-channel alignment, support structure, and changes in singular bases. In this paper, we propose CHASE (Channel-Aligned Structure Exploitation) to use these structures in practical model design. CHASE covers six applications across model modification, reconfiguration, and compression. CORA, COEC, and CORAM apply GSA to parameter-efficient finetuning, structured-pruning compensation, and model merging. We further develop three new methods. CAGA uses GSA to identify multi-head attention heads that can share a KV representation and constructs the shared key and value heads through geometric alignment and low-rank subspace extraction. SAKV uses GSA to determine which adjacent layers can share a low-rank KV-cache representation and the retained rank for each layer group. CAPS uses GSA spectral structure to group output neurons and selects retained input channels separately for each group. Results from CORA, COEC, and CORAM establish the effectiveness of GSA for adaptation, pruning compensation, and model merging. Experiments on CAGA show that geometric shared-head construction substantially improves MHA-to-GQA conversion, and SAKV and CAPS improve over representative baselines for KV-cache compression and structured pruning. These results show that the structures identified by GSA can be used directly to design methods for a range of model operations. 

---
# Boundary-Free Contextual Biasing: Depth-Adaptive Gating and Reading-Space Matching for Unsegmented Languages 

**Authors**: Muhammad Huzaifah, Yu Pan, Zachary Yeo, Ningjie Bai, Guangzhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09467)  

**Abstract**: Contextual biasing supplies an ASR system with a list of expected words at inference time, but existing methods rely on word boundaries that Japanese and Chinese do not provide. We present a boundary-free biasing decoder for frozen public CTC models, built on a character-level Aho-Corasick automaton, with no training and no second pass. Two evidence-based mechanisms replace the boundary: a depth-adaptive gate that sets how hard to push from match depth, and reading-space matching for when the audio is right but the characters are wrong. On Aishell-1 NE's hard R1 subset we reach 66.5% recall, above the trained CLAS baseline (64%), transferring to WenetSpeech and to a second architecture without retuning. We release the first open Japanese contextual-biasing benchmark, where biasing lifts rare-word recall by 25 points at precision above 97%, and still by 19 and 22 points against 1,000-word lists. 

---
# Finding the Right Balance: Relevance and Diversity in LLM Retrieval 

**Authors**: Guillaume Brouillette, Faustin Kagabo, Usef Faghihi, Nadia Ghazzali  

**Link**: [PDF](https://arxiv.org/pdf/2610.09412)  

**Abstract**: Retrieval diversification is widely available in retrieval-augmented generation (RAG) frameworks, yet prior studies disagree on whether it improves retrieval and answer quality. We show that its effectiveness varies primarily with candidate-pool redundancy, in a pattern consistent with the number of distinct evidence pieces a query requires. Using controlled near-duplicate injection and production-style overlapping chunking, we find that diversification harms relevance, evidence coverage and answer quality on clean pools, but becomes beneficial on multi-evidence tasks when redundancy causes nearest-neighbor retrieval to select repeated passages. We therefore introduce a query-adaptive rule that diversifies only when the effective number of distinct documents in the nearest-neighbor top-$k$ selection falls below the query's evidence requirement. Computed from existing embeddings, the rule captures most of the achievable gain, transfers across datasets and encoders and automatically reduces to nearest-neighbor retrieval for single-evidence queries. We also introduce RNG-Score, a geometric reranker with an exact nearest-neighbor fallback whose margin indicates duplicate structure. Overall, we conclude that diversification should be used selectively, based on observable redundancy and evidence requirements. 

---
# The Confidence Game: Strategic Miscalibration in Human-AI Delegation 

**Authors**: Raghu Arghal, Saswati Sarkar, Shirin Saeedi Bidokhti  

**Link**: [PDF](https://arxiv.org/pdf/2610.09371)  

**Abstract**: Calibrated uncertainty quantification is essential to ensuring AI agents are trustworthy and reliable. However, when agents seek to maximize user engagement or revenue, confidence reports may be strategically distorted, detracting from their informativeness. We formalize this problem in the Confidence Game: a repeated signaling game with imperfect monitoring in which an agent of unknown honesty and ability reports its confidence, and a user decides whether to delegate the task or complete it herself. The agent manages the tradeoff between manipulating signals and maintaining its reputation. We characterize the Markov Perfect Bayesian Equilibria of the two-period game and show that honest reporting is not an equilibrium, inflation is the unique best response once the agent is sufficiently myopic, and under-reporting requires that the user believe honesty to be a minority. We then place an LLM in the agent role, supplying it with its true probability of success so that any gap between what it knows and what it reports is attributable to incentives rather than to miscalibration. The model claims high confidence on 56% of tasks it has been told it will probably fail. This persists on real tasks, where it must estimate its own accuracy and causes miscalibration to increase while the agent's signal becomes less informative. Furthermore, we find that the LLM agent's decisions are coherent, but it systematically underestimates both how likely the user is to delegate and how secure its reputation is, resulting in less extreme behavior. Pricing the agent's reporting rule, we find that it destroys 68% of the gains from delegation, of which 71% is information the report no longer carries and no amount of user sophistication recovers. Overall, we establish confidence reporting under delegation as a strategic problem and provide a tractable basis for modeling, analyzing, and testing agent behavior. 

---
# TopoGraphRAG-Bench: Evaluating Multimodal GraphRAG on Layout-Grounded Evidence Reasoning 

**Authors**: Ruochi Li, Jianzhe Lin, Haoxuan Zhang, Haihua Chen, Junhua Ding, Edward Gehringer, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09360)  

**Abstract**: Real-world documents distribute evidence across text, tables, figures, and captions within complex page layouts. Answering complex questions over such documents therefore requires more than retrieving relevant passages: systems must recover the evidence topology that connects heterogeneous evidence units. Existing GraphRAG evaluations remain largely text-centered, while multimodal document RAG benchmarks assess cross-modal retrieval and generation without directly evaluating recovery of the intended evidence topology. We introduce TOPOGRAPHRAG-BENCH, a layout-grounded benchmark for multimodal evidence reasoning in GraphRAG, comprising 2,024 questions over 201 long, visually rich documents. Questions are constructed bottom-up from text, figure, and table evidence units under three controlled topologies: single-hop retrieval, bridge-chain reasoning, and multi-source synthesis. To ensure that questions preserve their intended structure, we apply counterfactual validation for shortcut resistance, modality necessity, and evidence necessity. We evaluate text-only GraphRAG, page-level visual retrieval, and multimodal GraphRAG systems using retrieval, generation, and topology-aware reasoning metrics. Multimodal GraphRAG systems achieve the strongest overall performance, but still fail when visual-textual evidence alignment or multi-unit composition is incomplete. Text-only GraphRAG struggles when key dependencies are grounded in figures or tables, while page-level visual retrieval lacks the fine-grained structure needed for topology recovery. These findings motivate GraphRAG systems that move beyond text-derived entity relation graphs to explicitly model document layouts, cross-modal evidence alignment, and the reasoning roles of evidence units. Code and data are available at this https URL. 

---
# Adversarial Images Hijack Web Agents from Visual Grounding to Browser Execution 

**Authors**: Wanjing Han, Levi Taiji Li, Mu Zhang, Yue Jiang, Guanhong Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09240)  

**Abstract**: Modern web agents built on large vision-language models process webpages, select relevant UI elements, and translate model outputs into browser actions. Existing visual red-teaming approaches use adversarial visual content to manipulate this process. However, they primarily target model inference and do not explicitly account for structured input processing or action post-processing. Consequently, model-level success does not establish control over browser execution and cannot reliably characterize end-to-end agent robustness. To address this gap, we formulate red teaming for vision-grounded web agents as an end-to-end grounding-to-execution problem, and introduce WebMirage, a framework that crafts localized visual perturbations that cause agents to select attacker-controlled content and execute the corresponding browser action across varying webpage renderings. It uses a role-slot abstraction and webpage recomposition to capture competition among webpage elements, and dataflow analysis to align optimization with action post-processing. We evaluate WebMirage across four agent configurations and six VLM backbones on 2,250 tasks covering 13 public websites and a sandbox benchmark. WebMirage achieves an average attack success rate of 91.9%, compared with 17.4% for the strongest baseline, and remains effective against three agent-level defenses. 

---
# Trajectory Abstraction for the Science of Language Agent Behavior 

**Authors**: Tianqiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09237)  

**Abstract**: Scientific studies of language agents need behavioral variables that support hypotheses across tasks and models. We formulate this research problem as learning and testing a hierarchy of trajectory abstractions. A concrete recursive procedure first measures role- and phase-indexed events, proposes temporally constrained relations, and tests their stability across conditions. It then constructs episode-level motif variables from selected relations and repeats the analysis on those variables. Explicit measurement functions connect every abstraction level to the original trajectories. Observations and randomized protocol experiments assess the resulting hypotheses, while comparisons between intervention realizations determine whether an abstraction should be retained, refined, or restricted. We derive a finite-depth bound for accepted reductions, identify protocol effects on fixed abstractions, and characterize realization disagreement and composition of abstraction error. A finite-sample test makes projected intervention consistency operational, and constructed examples illustrate motif construction and abstraction refinement. The formulation distinguishes this experimental approach from semantic taxonomies, qualitative theory induction, and behavior-model recovery. It specifies a proposed research procedure for discovering generalizable behavioral hypotheses, with literature-relative novelty assessed separately from model-relative surprise. 

---
# Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders 

**Authors**: Hyojung Han, Jongmin Kim, Seung-Hun Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2610.09227)  

**Abstract**: Mixed-precision post-training quantization needs a per-module sensitivity signal; for a text embedder the obvious one -- the retrieval quality a module costs when quantized -- needs relevance labels that deployments rarely have. We measure a label-free substitute: quantization-induced representation drift, obtained by quantizing one module, re-encoding the corpus, and recording how far the output embeddings moved from their full-precision positions. What is specific is the observable: the deployed output representation a dense retriever ranks with. Across five development embedders, configuration-level drift orders sampled mixed-precision plans against held-out retrieval quality at a macro Spearman of 0.911, the sensitivity transports across calibration corpora and retrieval domains in the usable regime, module drifts compose rank-consistently but not numerically, and relevance-derived sensitivity adds no consistent value. The method is one additive allocation under a hard packed-byte budget, with no labels and no search. On three embedders held untouched until method, baselines and hypotheses were frozen and sealed, the pre-registered directional hypothesis against the prior LieQ criterion holds (3/3 at the main budget, no collapse) and drift scores above a two-sided LieQ steelman in 2/3; but at the main budget drift is numerically lower than same-budget uniform precision on all three (-0.99, -0.85, -1.01 points), having reduced module and whole-model drift as designed. Output drift is thus a robust coarse sensitivity signal, not a universally optimal allocation objective: it avoids the catastrophic failures of the transferred signed-geometry adaptation and can remain usable at stressed budgets where uniform collapses, but fine-grained redistribution around a strong uniform operating point remains unresolved. 

---
# Few Bits, One Law: Toward W2A4KV2 

**Authors**: Kai Yi, Tarek Elgamal, Sruthikesh Surineni, Vignesh Vivekraja, Soumyadeep Ghosh, Steven Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09202)  

**Abstract**: Extreme low-bit LLM compression is most challenging when weights, activations, and KV caches are quantized together: their distributions differ, and quantization errors interact throughout the network. We introduce CanonQ, a unified quantization-aware training framework that addresses these challenges by separating source canonicalization from task-aware adaptation. Fixed rotations and energy normalization map heterogeneous tensor sources to canonical coordinates, enabling frozen Gaussian-reference codebooks to be reused across layers and models. Joint training then adapts the network to the coupled errors of weight, activation, and cache quantization within a common scalar/vector interface. We bound frozen-codebook transfer error and local task loss, and derive an exact normalization-aware straight-through Jacobian that links quantization distortion to gradient bias. The strongest gains arise under joint W2A4KV2 compression: across LLaMA3-1B/3B/8B, CanonQ-Omni achieves up to 14.28x lower WikiText-2 perplexity and up to 57.9% higher mean zero-shot accuracy than prior state-of-the-art and representative quantization baselines. The benefits extend to Qwen3-1.7B, code generation, and mathematical reasoning: on instruction-tuned MobileLLM-Pro-1B at W2A16KV16, CanonQ achieves relative improvements of 41.7% in HumanEval pass@1 and 39.1% in GSM8K exact match over the strongest evaluated quantization baseline. 

---
# Bookkeeping, Composition, or Unreachable Gold? Reading MemoryAgentBench's Conflict-Resolution Scores Against a Frozen Last-Write Resolver 

**Authors**: Egor Pakhomov, Erik Nijkamp  

**Link**: [PDF](https://arxiv.org/pdf/2610.09193)  

**Abstract**: MemoryAgentBench's Conflict Resolution split is read as measuring "selective forgetting". We execute the benchmark's own rule - the newest statement about a fact wins - as a zero-learning resolver frozen on one of the four fact lists. Under the official metric the rule answers 80.25% of the questions (74.5% on the three held-out lists). Of the rest, 67 items have a released gold that the last-write graph cannot reach but overwritten statements would ("The capital of India is New Delhi." superseded by "The capital of India is Grosseto."; gold New Delhi); such items are a third of the multi-hop questions at 262K. Two long-context models and our pre-registered approximate re-implementation of the benchmark's BM25 agent, one retained run per item and outcomes only, score 84.7%, 82.6% and 41.6% on the items the rule solves against 10.4%, 11.9% and 6.0% on those 67. The failures are a reachability split plus a small parser-scope residual; the per-item split, not the aggregate, is the unit at which a score here can be read. 

---
# LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition 

**Authors**: Shikhar Srivastava, Christopher Kanan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09179)  

**Abstract**: As data propagates through a Transformer, the norm of its hidden states grows by orders of magnitude with depth, a phenomenon framed as 'curse of depth' and nearly universally treated as a pathology to be suppressed. We take the opposite view. Across 16 pre-trained LLMs from 9 families, spanning dense, mixture-of-experts and hybrid architectures and Pre-, Peri- and Post-Norm designs, we find that this growth reflects an emergent depth-positional encoding, carried by the only learned per-layer gain on the residual stream, the normalization weight $\gamma$: with depth, $\gamma$ grows in magnitude and rotates in direction, jointly encoding the layer index. We make this depth-conditioned encoding explicit with LayerRoPE, an implicit analog of RoPE along the depth axis, which replaces all layerwise $\gamma$ vectors with a single shared vector and depth-conditioned scalars, at a net reduction in parameters and $<0.02\%$ change in FLOPs. Across a model ladder scaled up to $100$B+ tokens, LayerRoPE consistently outperforms Pre-, Post- and Peri-Norm and Layer-Norm Scaling, reaching Pre-Norm's 1.3B loss with $3.4\times$ less compute; LayerRoPE is the only approach that shows strong convergence and improves near monotonically as depth scales to 512 layers. It improves learning-rate sensitivity by $3$-$10\times$, and transfers naively to and consistently improves looped latent models and Vision Transformers. Inspecting its learned schedule inverts the prevailing premise: LayerRoPE does not shrink the residual stream but widens it, damping what each block reads while amplifying what it writes. Depth stability, our results suggest, calls not for suppressing the residual stream, but for depth-conditioned regulation of the computational blocks it feeds. 

---
# Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models 

**Authors**: Justin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.09145)  

**Abstract**: We revisit a standard accepted practice in the continuous diffusion language model
literature of fixing conditioning prompt tokens clean during training.
We make a very simple modification: also noise the conditioning prompt tokens during training.
We demonstrate that under this modified training objective, we achieve better generalization
in combinatorial reasoning tasks such as Sudoku and N-Queens, with the largest gains on harder variants
($3.73\% \to 24.65\%$ solve rate on Sudoku Hard), and increased diversity of generated solutions ($50.60\% \to 73.79\%$ coverage on
10x10 N-Queens). We also show measurable improvements to natural language generation quality
in modest dataset regimes with Gigaword summarization, but notably demonstrate that gains do not
transfer to all natural language tasks (e.g open ended dialogue generation).
Our method is a single line change to the training objective, requires no additional inference costs by default,
and provides the flexibility of classifier-free guidance inspired guided sampling. Our
\href{this https URL} {code} is publicly available. 

---
# From Uncertainty to Action: Learning to Steer LLM Agents 

**Authors**: Hanwen Li, Jinhao Duan, Guanhua Zhu, Junchi Lu, Bo Shen, Chenxi Yuan, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09115)  

**Abstract**: Steering an LLM agent means deciding whether to correct it, at which step, and with which mechanism. Uncertainty is often used to decide when to correct an agent, but whether it can guide these decisions remains unclear. We steer agent trajectories separately at every non-terminal step with each of four mechanisms and run each continuation to completion. The resulting stepwise outcome table (SOT) holds about 82,000 counterfactual continuations of 1,864 trajectories from three benchmarks and two agents. It shows that uncertainty can identify failing trajectories, but that no single signal reliably locates the step at which steering helps. We therefore propose VoS (Value of Steering), a trajectory-level monitor, offline or online, that learns from SOT the value of steering at each step and decides where to steer by it. A harm-budgeted trigger decides whether to steer, limiting the fraction of successful trajectories that VoS disturbs. VoS improves on unmodified execution in all 12 settings of benchmark, agent, and offline or online use, by 7.8 points on average, and outperforms the strongest of five existing uncertainty-triggered methods in 11, by 2.9 points on average. Ablations show that training on measured outcomes and a tight harm budget are both essential. 

---
# Constraint Tree Exploration for Learning from Language Feedback 

**Authors**: Shaoang Li, Daniel R. Jiang, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09107)  

**Abstract**: Natural-language feedback in interactive learning often explains why an action failed by pointing to violated requirements. Misinterpreting this feedback can lead an agent to rule out valid solutions. We study this setting by modeling user intent as latent constraints over an action space and formulating learning from language feedback as pure exploration over feasible regions. We introduce TRACE, an algorithm that organizes candidate constraints in a tree and tests each proposed refinement by generating actions that satisfy it. TRACE commits to the refinement only if the resulting feedback does not contradict it over repeated tests. We distinguish two ways of using the same feedback: (i) falsification, which detects contradictions to the constraint set currently being tested, and (ii) identification, which may additionally name a violated constraint. We prove high-probability coverage bounds with dependence on the candidate class size $H$ for TRACE-Falsification. With reliable identification, TRACE-Identification can replace this dependence by $K/p_{\mathrm{ext}}$, where $K$ is the number of latent constraints and $p_{\mathrm{ext}}$ lower-bounds the probability of extracting a missing true constraint from informative feedback. We evaluate TRACE across six language-feedback tasks. On RecMovie, TRACE-Identification achieves 73% and 86% final-output success under caps of 20 and 60 evaluated outputs, compared with at most 42% and 48% for the evaluated prompting baselines given the same feedback and output caps. Controlled identity-corruption experiments further show greater robustness than direct accumulation when the falsification detector remains reliable. 

---
# Large-scale Repository Engineering via Agent-Native Reusable Code Primitives 

**Authors**: Haibo Jin, Peng Kuang, Xucheng Yu, Jerry Wang, Dehao Wu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09079)  

**Abstract**: Large language models equipped with development environments have moved code generation toward repository-scale construction, yet building complete repositories remains difficult because interacting modules, interfaces, configurations, tests, and dependencies must work together. We introduce Code Primitives, agent-native reusable executable components with interface contracts, dependency closures, validation tests, and provenance. Each primitive uses a resident LLM to assess relevance and adapt its implementation, interfaces, and dependencies to the target repository, and we organize 1,424 validated primitives in CodeFace, a searchable library for repository construction. We introduce LEGO (Large-scale repository Engineering via aGent-native reusable cOde primitives), which activates task-relevant primitives, integrates their adapted implementations with task-specific code while resolving cross-component constraints, and revises the result against executed tests. To measure construction end to end, we build LEGO-REPO, a benchmark of 522 executable reconstruction tasks spanning seven software domains, 22 capability tracks, and five difficulty levels, scored against native test suites between an empty-package floor and original-source ceiling. The strongest of 13 evaluated backbones reaches a delivery score of 0.318 and scores zero on 41.0% of tasks; LEGO improves all 13 by 0.1474 on average and raises GPT-5.6-terra from 0.3180 to 0.5134 (+61.4%). In controlled comparisons, adapted primitives outperform retrieved code supplied as context or vendored unchanged. The effect persists against independent repository agents, across three external benchmarks, and with a disjointly re-mined CodeFace; GPT-OSS-20B for adaptation and diagnosis retains 95.1% of the homogeneous score at 24.0% lower cost. 

---
# Talking with Language Models 

**Authors**: James Ravi Kirkpatrick, Alexandru Radulescu, Rachel Katharine Sterken  

**Link**: [PDF](https://arxiv.org/pdf/2610.09064)  

**Abstract**: When we interact with large language models (LLMs), are we having a conversation? They are designed to invite us to treat them as intelligent interlocutors who remember, act, and make commitments. But appearances deceive. We introduce the artifactual stance, a framework that reconceives human-AI interaction as artifact-mediated exchanges of candidate texts. LLM outputs are candidate texts optimized for utility, not utterances bearing meaning or force. LLMs are sophisticated text generators, not speakers. Between sessions, nothing runs; between turns, no one remembers. What persists is a configuration and a transcript. The "conversation" is a user's solo performance, interpretive labour disguised by interface and artifact design. This shift dissolves recent philosophical puzzles. Questions about what 'I' and 'you' refer to in AI exchanges, about whether systems can lie or be held to promises, about the identity of our supposed interlocutors all rest on a false presupposition. There is no speaker behind the screen, hence no one to refer to, no one to hold responsible. What feels like dialogue with someone is interaction with an artifact that generates text at unprecedented scale and fit. By abandoning the conversational framing, we see these systems for what they are: immensely sophisticated artifacts that afford varied uses. The philosophical questions that matter are about the normative underpinnings of design, adoption, authorization, and human practices of use. 

---
# Multi-Label Topic Assignment via LLM Distillation: A Comparative Analysis of Generative vs. Discriminative Student Models 

**Authors**: Sourabh Kasliwal, Shubhranshu Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.09063)  

**Abstract**: Multi-label topic assignment for user-generated content (UGC) -- including product reviews and buyer-seller conversations -- poses unique scalability challenges in large-scale e-commerce due to informal language, extreme label sparsity, and rapidly evolving taxonomies. While utilizing Large Language Models (LLMs) as labeling oracles to distill ground-truth data has emerged as an industry standard to bypass prohibitive manual annotation costs, determining the optimal, low-latency architecture for the resulting student models remains an open challenge. To address this, we conduct a comprehensive evaluation across Small Language Model (SLM) parameter scales (1B, 4B, and 8B) and architectural paradigms (causal generative versus bidirectional discriminative). Comparing generative text-to-label classifiers against discriminative baselines (DeBERTa-V3 and ModernBERT), our analysis reveals a crucial data-dependent trade-off: while discriminative models outperform ultra-lightweight generative models on structured product reviews, even the smallest 1B generative model surpasses discriminative baselines on complex, multi-turn conversational data. Furthermore, generative models maintain robust performance under massive label-set expansion (up to 112 topics) and severe long-tail distributions, whereas discriminative baselines suffer a 35% drop in Macro-F1 at scale. Finally, we detail the successful production deployment of these optimized models across both product review and conversational domains, demonstrating strict latency compliance and tangible business impact at a global marketplace scale. 

---
# BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment 

**Authors**: Kemal Davaslioglu, Nathan Conger, Sastry Kompella, Yalin E. Sagduyu, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2610.09026)  

**Abstract**: We present BEACON-SP, an ontology-grounded Graph Retrieval-Augmented Generation (GraphRAG) framework for clinician-facing decision support in behavioral health settings such as suicide prevention, where effective assessment requires integrating heterogeneous clinical, behavioral, social, and temporal evidence. BEACON-SP combines patient knowledge graphs with ontology-guided retrieval to support multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships. The framework is enabled by a comprehensive suicide prevention ontology that integrates the Three-Step Theory, the Integrated Motivational-Volitional Model, and the Suicide Social Determinants of Health Ontology into a unified representation of patient risk factors. We construct ontology-grounded patient knowledge graphs and evaluate BEACON-SP for clinician-facing question answering. Compared with a vector-based retrieval-augmented generation (RAG) baseline on a 1,500-query benchmark spanning 15 clinical categories and 100 patients, BEACON-SP improves completeness, clinical relevance, and evidence grounding under a corrected comparative evaluation protocol, with a small gain on factual accuracy. In paired criterion-level comparisons, GraphRAG is preferred in 76.4% of cases. These results demonstrate the potential of ontology-guided GraphRAG to provide structured, contextualized patient evidence for clinical decision support. 

---
# How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis 

**Authors**: Muhammad Zeeshan Karamat, Christiana Chamon Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2610.09000)  

**Abstract**: As small language models (SLMs) are increasingly deployed on resource-constrained and on-device platforms, including as components of agentic systems, the integrity of locally stored model parameters becomes an important safety concern. We investigate whether safety-sensitive behavior in LLaMA-2-7B-Chat is concentrated within a sparse subset of parameters, creating a reduced fault surface for targeted analysis. We study two complementary localization methods: low-rank safety-associated subspace analysis and parameter-level safety--utility importance filtering. Both approaches reveal highly non-uniform safety sensitivity across the network, with the MLP down_proj consistently emerging as a prominent safety-sensitive component and o_proj providing a smaller contribution. Using parameter-level localization, modifying only 0.19% of model weights in down_proj yields 53% Basic ASR and 56% GCG ASR, while tinyBenchmarks accuracy remains at 51.6% compared with a 52.2% unmodified baseline. These results motivate targeted fault analysis and selective integrity protection for language models deployed in resource-constrained, on-device, and agentic settings. 

---
# Phoneme-Guided Initialization for LLM-based Speech Recognition 

**Authors**: Ryo Magoshi, Shinsuke Sakai, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2610.08994)  

**Abstract**: Speech large language models (speech LLMs) perform well on automatic speech recognition (ASR) when sufficient paired speech-text data is available, but their performance degrades in low-resource settings. A cascaded pipeline that performs speech-to-phoneme (S2P) conversion followed by phoneme-to-grapheme (P2G) conversion has been shown to outperform end-to-end speech LLMs in this regime, suggesting that phoneme-mediated processing is beneficial when paired data is scarce. We propose \textit{phoneme-guided initialization}, a simple method that uses this insight within an end-to-end framework: we pre-train the audio encoder on S2P and the LLM on P2G tasks, then connect them and fine-tune the full model end-to-end on the target ASR task. Experiments on Japanese (CSJ), Chinese (AISHELL-1), and two low-resource languages from Common Voice 25.0 (Tatar and Urdu) show that our method matches or outperforms both the cascaded S2P-P2G baseline and the end-to-end model without P2G initialization. 

---
# On KL-Regularized Policy Optimization 

**Authors**: Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08963)  

**Abstract**: Asynchronous reinforcement learning (RL) for large language model (LLM) agents trains one policy on trajectories generated by another: rollouts come from stale checkpoints, and the inference engine's probabilities differ from the trainer's even at identical parameters. Standard remedies either clip importance ratios, which biases the update, or, as in GRPO, sample a group of responses per prompt, which is costly when episodes are long. We propose KL-Regularized Policy Optimization (KLPO), a framework that anchors the KL regularizer at the sampler. The regularized improvement step then has a closed-form Gibbs solution, and KLPO fits its log-ratio optimality condition by least squares on the sampler's own trajectories, so the sampler probability enters through a log-ratio and no importance weights are needed. Profiling out the regression intercept replaces the intractable log-partition function with the signal's sampler mean plus a sampler-to-trainer KL divergence. For token-level policy mirror descent targets, we show that the resulting gradient can be computed from terminal returns without a critic, via sampler-centered scores or a single trajectory residual, even under stochastic tool outputs. We further prove that independent Monte Carlo estimates of the KL term keep these gradients unbiased, derive the exact KL gap of cheaper top-$K$ and binary approximations, and show that SPPO, GPO, REBEL, and BPO arise as special cases of KLPO. The result is a critic-free update that uses one rollout per prompt and requires neither a learned normalizer nor a group of responses. 

---
# FinVector-Market-4B: A Controlled Study of LoRA Adaptation for Structured Financial Tasks 

**Authors**: Alina Khaybullina  

**Link**: [PDF](https://arxiv.org/pdf/2610.08882)  

**Abstract**: FinVector-Market-4B adapts Qwen/Qwen3.5-4B with rank-16 LoRA on a 22,000-example corpus for structured financial tasks. We evaluate the base and adapted models on the same 600-example benchmark under implicit and explicit JSON-schema contracts. Supplying the schema alone raises base-model JSON validity from 0% to 91.3%. Under matched explicit prompting, the frozen scores improve from 14.7% to 40.0% for FinQA answer exact match, from 48.0% to 82.7% for calculator-expression correctness, from 20.1% to 89.5% for scenario branch-label agreement, and from 52.4% to 87.2% for implication-direction agreement. A post-hoc policy-scoring audit shows that the reported macro-F1 decline reflects a changing label set; using the same three target classes gives 77.4% for the base and 83.1% for the adapter. Filing overlap and calculator-target inconsistencies qualify the benchmark's generalization claims. The results show that compact financial domain adaptation can produce substantial task-specific gains beyond output-format learning under matched prompting, with gains bounded by the evaluated task distribution and prompt contract. 

---
# Is Word Error Rate Enough? Rethinking Privacy Evaluation in Speech with Entity-Aware Metrics 

**Authors**: Anjana Rajasekhar, Jule Pohlhausen, Nayana Jacob Alappattu, Anna Leschanowsky  

**Link**: [PDF](https://arxiv.org/pdf/2610.08831)  

**Abstract**: As the use of smart devices continues to increase, their potential to capture sensitive speech content raises growing privacy concerns. It is therefore critical to develop techniques that prevent information leakage while preserving the utility of the audio, and evaluation metrics that accurately quantify the level of privacy without overestimating it. In this work, we evaluate the effectiveness of two obfuscation techniques in protecting speech content, with particular emphasis on named entities, by adapting entity-aware privacy metrics from the Natural Language Processing field to the speech privacy domain. Further, we investigate several attack scenarios and show that fine-tuning on entity-rich data improves attack performance for some entity categories but not others. Finally, we provide guidance on metric selection based on whether the obfuscation method preserves temporal alignment. 

---
# Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States 

**Authors**: Shuhao Li, Weidong Yang, Changan Liu, Wei Zhuo, Yingbo Zhou, Fan Zhang, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.08818)  

**Abstract**: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses spatio-temporal dynamics into a latent space, followed by a Generative Transformer (GenT) that reconstructs the future states of unobserved nodes from noise, guided by multi-modal conditions including semantics, geographic coordinates, and neighborhood contexts. Experiments on six traffic and two non-traffic datasets show GenST significantly outperforms existing baselines in zero-shot prediction tasks, demonstrating the practical potential of semantic-guided generation for mitigating spatio-temporal data sparsity. 

---
# Route-Verify-Vote: Procedure-Conditioned Self-Consistency for Mixed-Domain Reasoning 

**Authors**: Xinchen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.08814)  

**Abstract**: Compositional generalization remains challenging when language models must combine familiar reasoning operations in unfamiliar ways. The Scenario-Based Commonsense Reasoning Evaluation (SCoRE) 2026 tests this ability on three mixed domains absent from training and requires models to identify the complete set of correct options for each question.
We introduce Route-Verify-Vote (RVV), a framework for procedure-conditioned self-consistency that uses language models without parameter updates. Route uses the provided domain label to select a reasoning procedure that guides the model in representing and applying the relevant constraints. Verify prompts the model to assess each option against those constraints. Vote aggregates complete answer sets and allocates additional samples to questions with a small vote-count margin between the two most frequent sets. Samples for each question follow the same domain-specific procedure.
On the official test set, voting over 16 sampled answer sets per question achieves an exact-set accuracy of 74.6%. Adaptive RVV reaches 77.3%, and combining models on selected domain routes raises accuracy to 79.4%. The final system ranked second among participating systems. These results support domain-specific reasoning procedures and answer-set disagreement as useful tools for allocating inference-time computation in mixed-domain reasoning. 

---
# EmphTTS: an emphasis-control TTS with reinforcement learning 

**Authors**: Zirui Li, Rech Silas, Lauri Juvela, Tom Backstrom, Mikko Kurimo  

**Link**: [PDF](https://arxiv.org/pdf/2609.27599)  

**Abstract**: Generating controllable and human-like emphasis remains an open challenge in text-to-speech, even when explicit emphasis control signals are provided in the text input, limiting the communicative accuracy of synthetic speech in real-world applications. Reinforcement learning has recently shown promise for post-training TTS systems to align with human preference, yet existing methods have not been applied to word-level prosodic control. We present EmphTTS, a non-autoregressive TTS system that applies Group Relative Policy Optimization (GRPO) to the duration predictor with an emphasis localization reward, enabling direct optimization for word-level emphasis. Evaluations show that EmphTTS achieves the best emphasis controllability and performs the best in emphasis objective evaluation. In subjective preference tests, EmphTTS is significantly preferred over synthetic groundtruth and most baselines. Ablation studies show that GRPO improves emphasis realization beyond supervised-finetuning-based duration modeling and simple speaking-rate adjustment, while alleviating the mismatch between the independently trained duration predictor and TTS model. 

---
