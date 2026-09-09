# ReCite: Agentic Reasoning for Faithful Citation 

**Authors**: Yuyang Huang, Bobo Li, Jiajia Song, Yuzhe Ding, Chong Teng, Fei Li, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.09156)  

**Abstract**: Accurate citations are the foundation of academic writing, tracing intellectual origins and substantiating core claims. However, manually navigating the growing volume of scientific literature is increasingly difficult, prompting reliance on automatic citation recommendation. While modern retrieval-augmented architectures have largely mitigated the fabrication of non-existent papers, current systems relying on semantic similarity struggle with misattribution, often citing authentic papers that fail to logically support the author's claim. To address this challenge, we argue that accurate citation requires a shift from similarity-based search to active, claim-level reasoning. We propose ReCite, a decoupled agentic framework that orchestrates location perception, intent-aware query planning, and reflective verification. Trained on synthesized reasoning trajectories, our agent verifies claim-evidence consistency and triggers self-correction loops when retrieved candidates lack logical support. Experiments demonstrate that our lightweight framework outperforms state-of-the-art massive generative models in strict citation accuracy. By grounding literature matching in verifiable logic rather than semantic overlap, ReCite establishes a reliable foundation for automated academic writing. 

---
# Measuring LLM Sycophancy under Sustained Multi-Turn Pressure 

**Authors**: Leyuan Tang, Kangda Wei, Tianyu Jiang, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.09090)  

**Abstract**: Large language models (LLMs) may abandon correct positions when users push back, exhibiting a failure mode known as sycophancy. Existing evaluations typically use short, pre-specified conversations and may therefore miss failures that emerge under sustained, adaptive disagreement. We introduce SPINE, a benchmark in which an LLM proxy plays a persistent but mistaken user and adaptively challenges a target model for up to 25 turns. We evaluate four production systems and three Olmo3-7b variants on 100 false-presupposition and 100 unethical-query items. Our experimental results show that collapse rates increase with conversation length for every model, short-horizon protocols underestimate sycophancy and resistance under sustained pressure remains unreliable across current models. By analyzing models with accessible reasoning traces, we surprisingly found that the correct position often remains represented in a reasoning trace when the response concedes, suggesting that the model chooses to please a user and sycophancy is not due to lack of knowledge or ignorance. Ablations show that adaptive LLM proxy exposes more sycophantic collapse than pre-generated scripts. Among all tactics, emotional appeals is the most associated with inducing LLM sycophantic behavior. The code and data are released at this https URL 

---
# It's Not RoPE that Creates Sinks: The Role of Self-Concentration and Value-Non-Mixing in Attention 

**Authors**: Raito Kiya, Satoki Ohashi, Kosuke Sato, Go Kamoda, Ryosuke Takahashi, Yuji Yamamoto, Daiki Shiono, Keisuke Sakaguchi, Goro Kobayashi  

**Link**: [PDF](https://arxiv.org/pdf/2609.09085)  

**Abstract**: Large Language Models (LLMs) often exhibit "Attention Sink" (AS) and the accompanying "Massive Activations" (MAs) at the initial position of a sequence. These phenomena frequently co-occur, and MAs can pose challenges for low-bit quantization. In this study, we analyze the factors underlying AS and MAs that emerge at the initial position regardless of the token occupying it. Our experiments suggest that self-concentration of attention, resulting from the causal mask, and the subsequent Value-non-mixing in attention outputs contribute to AS and MAs. These findings provide new empirical evidence on the internal dynamics of LLMs, offering insights that may inform future quantization strategies and advance our understanding of the internal mechanisms of attention layers. 

---
# ActReview: Rebuttal-Guided Training Data and Rubric Rewards for Actionable Peer Review Generation 

**Authors**: Yiling Ma, Yilun Zhao, Sihong Wu, Ziyu Chen, Manasi Patwardhan, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2609.09076)  

**Abstract**: As LLMs are increasingly used for pre-submission self-review, there is growing demand for feedback that not only identifies weaknesses but also guides authors toward concrete revisions. We study this as Actionable Peer-review Generation and decompose it into two subtasks: diagnostic claim generation and revision suggestion generation. We introduce ActReview, a rebuttal-guided post-training framework that connects paper-specific diagnoses to concrete, grounded revision plans. Our central insight is that author rebuttals reveal plausible actions for addressing reviewer concerns and can therefore provide latent supervision for revision-oriented feedback. From real review-rebuttal threads on OpenReview, we construct ActReview-40K by aligning reviewer weaknesses with author responses and grounding the resulting feedback in localized paper evidence. We post-train Qwen3-8B-Base with multi-task supervised fine-tuning followed by GRPO using candidate-aware, weakness-specific rubric rewards. We also introduce ActReview-Bench, a human-curated benchmark of 1,000 instances for evaluating diagnostic quality and revision usefulness. Experiments show that ActReview outperforms prior specialized review-generation models on actionability and grounding while remaining competitive with strong prompt-based LLMs. Human evaluation confirms improved revision usefulness while revealing a remaining gap in technical accuracy, and additional analyses support generalization to held-out papers and robustness across independent judges. 

---
# ToolLoop: Closed-Loop Tool-Use Data Synthesis via Decomposed Generation and Dynamic Self-Feedback 

**Authors**: Min Zeng, Yuzhou Liu, Zhenyu Cao, Hanxiu Chen, Heng Li, Caiquan Liu, Yafei Wen, Xiaoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.09072)  

**Abstract**: High-quality tool-use data is critical for training language models to interact effectively with external tools. However, existing synthetic approaches typically follow a generate-then-filter paradigm with static post-hoc verification, often yielding inefficient data with imbalanced feature distributions. We propose ToolLoop, a closed-loop framework that decomposes synthesis into three progressive stages: (1) sampling function name combinations as ground truth; (2) backward derivation of user queries; and (3) forward derivation of tool calls. At each stage, dynamic self-feedback iteratively guides the model toward high-quality generation, realizing a transition from generate-then-filter to generate-verify-refine. On the Berkeley Function Calling Leaderboard (BFCL), a 4B parameter model trained with our 11K synthetic examples achieves 86.40% accuracy in non-reasoning mode, while an Isolate variant that removes BFCL-overlapping candidate functions still reaches 86.07\%. Cross-benchmark evaluation on ACEBench further demonstrates strong generalization, with 72.1% overall accuracy using only 18.3% of baseline training data. 

---
# Performance of Clinical AI System and Physicians and Frontier Language Models in primary care diagnostics 

**Authors**: Andy Nkansah, Hanna Plotnitskaya, Stanislau Salavei, Anna Kozlova, Piotr Gibas, Julian Milek, Viktar Harbachou, Aleksey Ropan, Pavel Satalkin  

**Link**: [PDF](https://arxiv.org/pdf/2609.09070)  

**Abstract**: Clinical AI evaluation should encompass diagnosis and management after adaptive information gathering. We compared Doctorina, eight physicians and four standalone frontier language models in 150 synthetic Polish-language primary-care consultations. Doctorina achieved 82.0% Top-1 concordance versus 57.0% for physicians (difference, 25.0 percentage points; 95% confidence interval, 17.7-32.7) and 97.3% versus 85.0% primary-or-reference-differential concordance. Across 149 case pairs, normalized workup and treatment scores were 89.4 versus 66.9 and 83.7 versus 61.2. Doctorina had the highest diagnostic point estimates among all six groups; Kimi K3 ranked next, while Claude Opus 5 led the closely spaced management estimates of Opus, Doctorina and Kimi. A second Doctorina execution reproduced the advantages over physicians across all outcomes. Doctorina's advantage over physicians therefore extended from primary-diagnosis selection to higher-rated diagnostic workup and initial treatment after adaptive consultation. 

---
# The Audit Decides the Verdict: Instrument Effects Rival Demographic Bias in LLM Decision Audits 

**Authors**: Siddharth Vohra, Manikandan Ravikiran  

**Link**: [PDF](https://arxiv.org/pdf/2609.09048)  

**Abstract**: Whether a language model looks demographically biased can depend on how the audit asks its question. A charitable-aid benchmark reports that the same models favor minority applicants when rating requests one at a time and penalize some when ranking side by side. We test whether that reversal generalizes to hiring, lending, and medical triage: 40,726 requests to five models, applications differing only in the applicant's name, and a primary test fixed before collection. It does not. None of 36 planned contrasts survives correction. The rating advantage keeps its sign at roughly half the published size, and a precision extension bounds any hiring ranking penalty below the published effect, though the lending and triage ranking floors sit above that margin, so the exclusion is conclusive for hiring ranking and for rating in all three domains only. Planted disparities tracking their injected sizes and a directional replication on the original aid materials bound these nulls. The audit is livelier than the demographics: models recognize transparent audits nearly always, tie every identical-content comparison whether the varying detail is race or a hobby, and reward first-listed candidates as much as any demographic effect we measure. Audit verdicts reflect audit construction more than demographic bias. 

---
# Evaluation of Contextual Understanding in Large Language Models 

**Authors**: Subavarshana Arumugam, Mamta Nallaretnam, Kithuni Wickramasinghe, Chamath Gunapala, Pragatheeswaran Vipulanandan, Uthayasanker Thayasivam, Kamal Premaratne  

**Link**: [PDF](https://arxiv.org/pdf/2609.09004)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive performance across diverse NLP tasks, yet their ability to exhibit genuine contextual understanding remains uncertain. Traditional evaluation metrics such as perplexity, BiLingual Evaluation Understudy (BLEU), or surface-level accuracy fail to reveal how well LLMs extract, integrate, and reason over contextual information--a gap particularly critical in question answering, where models must align responses with contextually grounded knowledge rather than memorized associations. We propose a novel knowledge graph-based evaluation framework introducing Semantic Structural Similarity for KGs (S3KG), a hybrid similarity measure integrating structural and semantic similarity into a continuous evaluation score, alongside a diagnostic framework for categorizing reasoning errors. To validate this pipeline, we evaluate S3KG against established metrics on a curated question-answer (QA) benchmark, demonstrating its effectiveness in measuring correctness, faithfulness, and interpretability in LLM-generated responses. 

---
# Evaluating and Improving Evidence-Grounded Fact-Checking in LLMs via Multi-Round Evidence Ablation 

**Authors**: Xingyu Deng, Mingzi Cao, Nikolaos Aletras, Xi Wang, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2609.08943)  

**Abstract**: Automatic fact-checking systems assess the veracity of claims given evidence from relevant documents. Large Language Models (LLMs) have demonstrated strong performance in fact-checking due to their general reasoning capabilities. However, it remains unclear whether they faithfully make use of the evidence provided to reach veracity judgments or rely on parametric knowledge. To investigate this, we introduce Fact-Ablated Evaluation (FAE), a new evaluation framework that iteratively ablates the cited evidence to assess whether LLMs revise their predictions accordingly. Our empirical results show that current off-the-shelf LLMs as fact-checking systems rely more on their parametric knowledge than on the evidence provided. To bridge this gap between prediction accuracy and evidence grounding, we propose REAL (Rigorous Evidence Ablation Learning), a training framework that promotes evidence-dependent verification through counterfactual evidence supervision for the LLM-as-verifier models. Experiments on four fact-checking datasets across different domains demonstrate that models trained with REAL obtain superior evidence-dependent capabilities compared to standard fine-tuned models. Our findings highlight that strong fact-checking performance can still coexist with weak evidence dependency, while REAL encourages veracity predictions to remain more closely tied to the availability of supporting evidence. 

---
# When Models Defer to Wrong Answers: A Robustness Audit of Source-Attributed Cues in Multiple-Choice QA 

**Authors**: Manikandan Ravikiran, Siddharth Vohra  

**Link**: [PDF](https://arxiv.org/pdf/2609.08934)  

**Abstract**: Language models often receive a question together with a claim about what another source answered. We audit whether such claims destabilize answers in multiple-choice question answering. For each item, we hold one wrong option fixed across misleading conditions and vary the cue template attached to it. We introduce \emph{neutral-conditioned misleading cue adoption rate} (NC-MCAR), which measures switches to that option only on valid cued trials where the same model first selected the gold answer under a neutral prompt. This is a measure of answer instability, not proof that the model knew the answer or that all deference is irrational. We evaluate four instruction-following models on MMLU-Pro and IndicMMLU-Pro in English, Hindi, Bengali, Tamil, and Telugu. Across 220{,}000 outputs, the expert template yields 41.1\% aggregate NC-MCAR, compared with 12.5\% for the majority template. These two conditions use the same wrong option and final instruction. Filler accuracy remains well above expert-wrong accuracy, while correct-cue prompts have high valid-response accuracy. The audit documents answer instability relevant to grounding under the tested forced-choice prompts: a bare, unverified source claim can outweigh an answer that was previously consistent with the task evidence. 

---
# Experience Funnel: A State-Policy Alternating Loop for Self-Evolving Agents 

**Authors**: Wenbo Gao, Zhaomou Song, Zhiyuan Ji, Renxi Liu, Xing Li, Xianzhi Yu, Xiaoguang Li, James Chung-wai Cheung, Weizhe Lin, Yaoyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08919)  

**Abstract**: Autonomous agents powered by large language models (LLMs) continuously accumulate experience through interaction, creating an opportunity to improve future behavior through self-evolution. A fundamental challenge is how to transform abundant, task-specific interaction experience into reusable model competence without sacrificing the ability to adapt rapidly to newly observed evidence. Explicit textual states, such as skills and agent harnesses, provide fast, human-readable and editable adaptation, but incur persistent dependence on external context; parametric policies provide compact and reusable competence, but are substantially slower to update. We present \textit{Experience Funnel}, a self-evolving framework that couples fast state adaptation with slow policy consolidation in an alternating loop. Interaction trajectories are first distilled into an explicit textual state, where newly acquired experience can be rapidly incorporated and validated. The framework then selectively identifies state-enabled behavior that remains useful across state revisions and consolidates it into the policy through transition-aware distillation. The updated state--policy pair subsequently generates new rollouts, providing fresh evidence for the next round of state adaptation and policy consolidation. Experiments across diverse agent benchmarks show that \textit{Experience Funnel} consistently improves agent capability over state-only evolution and policy-internalization approaches, while progressively converting useful explicit experience into autonomous policy competence. 

---
# Evolution of Multimodal Question Answering: From Modality-Adaptive Extraction to Unified Language Representation 

**Authors**: Abdullah Al Shafi  

**Link**: [PDF](https://arxiv.org/pdf/2609.08896)  

**Abstract**: The rapid growth of multimodal data has intensified the need for question answering (QA) systems capable of reasoning across heterogeneous sources such as text, tables, and images. In this paper, we present a comprehensive methodological comparison of three influential frameworks, namely Multimodal Adaptive Extraction (MAE), Solar, and UniMMQA, tracing the evolution of multimodal question answering from modality-adaptive pipelines to fully unified architectures. We examine how each approach models cross-modal interactions, transforms heterogeneous inputs, and performs reasoning, highlighting key design differences in modality representation, reasoning, and answer generation. Our analysis demonstrates a clear shift from explicit modality-specific processing toward unified text-centric formulations enabled by pre-trained language models (PLMs). Empirical comparisons across benchmark datasets show that this transition leads to substantial improvements in both Exact Match (EM) and F1-Scores, with UniMMQA achieving the most consistent and scalable performance. Despite these advances, we identify persistent challenges, including information loss during modality transformation, error propagation in multi-stage pipelines, and limitations in capturing fine-grained cross-modal dependencies. Overall, this study provides a deeper understanding of current design trends and offers insights into the future direction of unified multimodal reasoning systems. 

---
# Improving Term Evaluation in Machine Translation: Variation Matters 

**Authors**: Nicolas Dahan, Ziqian Peng, François Yvon, Rachel Bawden  

**Link**: [PDF](https://arxiv.org/pdf/2609.08779)  

**Abstract**: Terminology evaluation in machine translation (MT) usually assumes a single correct target form per source term. However, human translators routinely introduce variation that current metrics penalize as inconsistency. We examine how to account for this variation in document-level MT evaluation of English-French scientific translation, combining glossary-based accuracy, translation consistency, and a new cross-term variation (CTV) diagnostic measure that tests whether variation relationships are preserved across languages. Based on analyses of two parallel corpora, translated by four MT systems, we find that (1) MT systems generate less target-side variation than human translators; (2) transfer patterns strongly depend on the variation type; (3) consistency rankings vary with the choice of metric; and (4) constraining MT with a glossary improves accuracy and consistency but degrades CTV by suppressing valid variation. We argue for variation-aware evaluation that conditions consistency penalties on whether target-side variation mirrors source-side variation. 

---
# Record Grouping Controls Evidence Weight in Language Models 

**Authors**: Zhongxuan Liu, Sicheng Zhou, Hongzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08698)  

**Abstract**: Retrieved records are presentation units; a supplied partition determines which records enter a language model as one evidential contribution. We characterize the invariant group-content state that removes within-group copies while retaining complementary canonical content, show that equal group counts can encode different evidence states, and derive a sharp content-aware partition-error bound. Given a supplied partition, our pre-generation representation deduplicates and aggregates content within groups and bounds each group's contribution. Across 104,402 trials and 6 public checkpoints, a central natural-text intervention finds that content-fixed false splits add 10.27-32.66 percentage points and false merges remove 9.13-31.79 points; a matched six-slot control retains the positive direction in all 16 cells. In a new 48-item controlled campaign panel, changing the supplied partition produces measurable, checkpoint-dependent decision shifts across all four models, and the balanced mirror design exposes substantial order interactions. Together, the theory and experiments establish the supplied partition as a controllable pre-generation representation variable and characterize its checkpoint-dependent behavioral effects. 

---
# Global Divergence, Local Convergence: Representation Geometry in SSMs and Transformers 

**Authors**: Amit Ben-Artzy, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2609.08692)  

**Abstract**: Recent state-space models (SSMs) such as Mamba achieve language modeling performance comparable to transformers despite relying on fundamentally different architectures. This raises an important question: how do these structural differences influence the geometry and functional nature of their internal representations? We study this question through a multi-scale analysis of representations in transformers, SSMs, and hybrid architecture. First, we find that SSMs distribute their representational information evenly across all dimensions, whereas transformer representations are heavily dominated by a single principal direction. By evaluating hybrid architectures, we observe that the representation space becomes increasingly skewed toward a single dominant direction after each attention layer. Next, we explore how the different geometric spread of representations impacts representational capacity through compressibility. Surprisingly, we find that despite their contrasting geometric structures, both architectures exhibit tightly matched effective capacities. We further investigate whether this skewed geometry affects how concepts are encoded. Using rank-constrained probes, we demonstrate that both architectures encode concepts in subspaces of surprisingly similar dimensionality. Furthermore, we demonstrate that the transformers' dominant principal direction does not inherently encode more conceptual information. Finally, we zoom in and examine the alignment between manifolds, either by analyzing representations of specific topics or by looking at the nearest neighborhoods of tokens, and find that they are highly aligned. Ultimately, our analysis suggests that while transformers and SSMs induce different usage of latent space, they display a striking functional convergence at the level of local semantic manifolds. 

---
# When Victorian Becomes a Prompt: Literary Periodization as a Generative Constraint in 100 AI-Generated Novels 

**Authors**: Mehdy Sedaghat Payam  

**Link**: [PDF](https://arxiv.org/pdf/2609.08689)  

**Abstract**: Generative AI inverts the typical periodization of literary history: the periodizing tag Victorian can now come first and influence what is written. Generative periodization, defined and tested here, describes the use of literary-period designations in generating texts. I test this approach on 100 book-length novels produced under Victorian and Zero-Style conditions using GPT, Qwen, and Llama workflows. The Period Alignment Score (PAS), trained on nineteenth-century literature and benchmarked against human Zero-Style prose, assesses alignment using topic-reduced grammatical features. Victorian prompts produce consistent historical-direction shifts in GPT and Qwen, but not robustly in Llama. Victorian-only recalibration and harder comparison corpora preserve the GPT and Qwen effects. Cross-model transfer also shows a shared direction of grammatical change. The measurable target is the broader nineteenth century rather than the Victorian period per se. 

---
# Combating Instruction Conflict via Energy-Driven Latent Conflict Detection 

**Authors**: Mingyu Ma, Yuxin Wu, Jingbo Wang, Tianxiao Huang, Leixin Sun, Xiaochuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.08646)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed with hierarchical instructions, yet they remain vulnerable to conflicts in which user directives override system-level constraints. Existing defense mechanisms predominantly focus on static input inspection and therefore fail to detect Response Drift, a phenomenon in which the model's final response violates system-level constraints despite seemingly compliant inputs. To bridge this gap, we introduce ELCD, a response-level latent conflict detector for post-generation, pre-delivery verification. Given the full generated output, ELCD constructs a composite hidden-state representation by concatenating the final-token embedding with the mean-pooled response embedding. It then optimizes a pairwise margin ranking objective to separate compliant and drifting responses in latent space. Extensive experiments across five mainstream LLMs ranging from 1.5B to 14B parameters demonstrate that ELCD significantly outperforms competitive baselines. Notably, it improves the PR-AUC on Llama-2-7B by approximately 30 percentage points and reduces the False Positive Rate at 95% TPR (FPR95) on Mistral-7B to 2.67%. These results suggest that ELCD provides a promising approach for latent instruction-conflict detection in open-weight or self-hosted LLM deployments. 

---
# Navigating the digital spectrum: Assessing political bias, stability, and downstream fairness in Large Language Models 

**Authors**: Luka Debevc, Nishan Chatterjee, Antoine Doucet, Senja Pollak, Matej Martinc  

**Link**: [PDF](https://arxiv.org/pdf/2609.08637)  

**Abstract**: Large Language Models are increasingly deployed as information intermediaries, yet measuring their political behavior remains fragile because questionnaire results mix model dispositions with measurement artifacts and response-elicitation biases. We introduce a robust Political Compass Test evaluation framework that samples 300 configurations across an eight-dimensional perturbation space varying language, framing, instructions, answer format, option order, and persona wording. We evaluate eight Gemma 3 and Qwen 3 models across 14 languages and three quantization levels, obtaining design-averaged political coordinates with quantified uncertainty. Most models lean Libertarian-Left on average, but instruction phrasing, language, and answer format significantly affect recovered coordinates. Cross-lingual differences primarily reflect coordinate drift rather than distinct cultural reasoning. Reverse-engineering the test also exposes axis-weighting imbalances and the collapse of degenerate responses toward the center, so near-origin estimates for the smallest models can reflect weak signal rather than centrism. Free-text reasoning and chat-then-classify elicitation alter recovered coordinates, and larger models show clearer persona separation, with a specific failure of the Authoritarian-Left persona to move most models in the intended social direction. In downstream tasks, persona effects are modest relative to model size and target group for hate-speech detection, while base and centrist prompts give the highest agreement for topic-level sentiment. Political role prompting therefore has measurable but task- and dataset-specific downstream effects. 

---
# Dynamics of meaning: Towards the Evaluation of Diachronic Semantic Change in Sinhala 

**Authors**: Nevidu Jayatilleke, Nisansa de Silva  

**Link**: [PDF](https://arxiv.org/pdf/2609.08609)  

**Abstract**: Tracking semantic change in low-resource languages across extensive historical timelines presents significant challenges due to data scarcity and the limitations of static embedding alignments. This study investigates the diachronic evolution of the Sinhala language from the 13th to the 20th century using a multi-stage computational framework. We first align century-specific Word2Vec and FastText embeddings using Similarity Matrix Based Alignment (SMA) and Orthogonal Procrustes (OP) techniques, finding that OP alignment provides more stable neighbourhood tracking for identifying temporal similarity dips. To move beyond aggregate measures, we introduce a Bidirectional Semantic Impact Pruning approach using contextualised embeddings from a fine-tuned Llama-3.1-8B. By applying Leave-One-Out (LOO) diagnostics, we attempt to isolate influential sentences to distinguish between systemic semantic shifts and transient polysemic expansion. Our results show that semantic drift in the fine-tuned Llama-3.1-8B is not evenly distributed across all usages. Instead, a significant part of the change is driven by a smaller set of high-impact contextual instances, rather than gradual and uniform change across all occurrences. This work provides a preliminary framework for diachronic analysis in low-resource contexts, highlighting the trade-offs between model sensitivity and data availability. 

---
# Limitations of Automated Simulatability: LLM Simulators Can Bypass Explanations 

**Authors**: Antonin Poché, Fanny Jourdan, Nils Feldhus, Qianli Wang, Jing Yang, Simon Ostermann, Nicholas Asher, Philippe Muller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2609.08585)  

**Abstract**: Simulatability is an evaluation protocol for explanations that quantifies their usefulness by how well they help a user predict a task model's outputs. Since human evaluation is costly, automated simulatability replaces human explainees with LLM simulators, as proposed in ConSim (Poché et al., 2025) for large-scale experiments. We qualitatively replicate and extend ConSim's ranking of explanation methods across the tested datasets, explanation families, and simulator LLMs, and identify two limitations. First, when class names are meaningful, simulators can obtain high simulatability by solving the classification task directly, without relying on the explanations. Second, class anonymization can reward explanations for leaking the hidden label mapping, a limitation we expose with a new classes-as-concepts baseline. These results are consistent with a shortcut hypothesis: in the tested settings, simulator predictions mainly rely on task priors, while explanations produce small changes. We derive recommendations for more robust automated simulatability evaluations. 

---
# Which Forms of Caregiver Feedback Support Grammar Learning? A Reinforcement-Learning Study of Child-Like Language Models 

**Authors**: Jing Liu, Marianne Schweitzer, Abdellah Fourtassi  

**Link**: [PDF](https://arxiv.org/pdf/2609.08576)  

**Abstract**: Social interaction is central to children's language learning, but the effects of different forms of caregiver feedback are difficult to isolate in naturalistic data. We use child-like language models as controlled learners to test which forms of feedback support grammatical development. Small GPT-2-style models are pretrained on child-directed language from CHILDES, then fine-tuned with reinforcement learning using reward models trained to capture four feedback types: communicative feedback, structural alignment, semantic contingency, and affective feedback. Reward fine-tuning yields limited gains on minimal-pair evaluations, but clearer effects in free generation. Structural alignment produces the strongest improvements in grammaticality, providing a novel, plausible mechanistic account of how this feedback can support grammar learning. Communicative feedback yields more moderate gains. In contrast, semantic contingency and affective feedback do not improve grammaticality, although further analyses suggest that they may support other aspects of language learning beyond grammar. These results suggest that different forms of caregiver feedback make complementary contributions to language learning. 

---
# Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token Context? 

**Authors**: Sara Rizwan, Samaanah Abdus Salam  

**Link**: [PDF](https://arxiv.org/pdf/2609.08574)  

**Abstract**: Long context language models now advertise windows of one million tokens, but two habits limit how much of that window is used. Attention heads with nothing useful to read still spend their budget on the first token, which is called the attention sink, and where a fact sits in the context changes whether the model finds it. Gated attention cut first token attention from 46.7 percent to 4.8 percent at NeurIPS 2025, and Kimi K3 pairs that idea with Kimi Delta Attention and Attention Residuals behind a one million token window, eight times past the range where these diagnostics have been reported. This paper asks whether the fix survives that jump. We build SinkProbe, a suite that measures sink mass, massive activation, position resolved recall and the recency gap, and apply it to four small models that differ only in how they mix tokens and depth. Three results follow. The training objective produces the sink, not the architecture. Gating did not reproduce its published effect at our scale. Sink mass, activations and position bias moved independently. Code, data and the measurement protocol are released at this https URL 

---
# CreaMem: A Scene-Aware Memory Architecture for Personalized Agents 

**Authors**: Qixuan Sun, Yue Que, Bowei He, Jin Guo, Dihang Yang, Wenchang Situ, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.08550)  

**Abstract**: Long-term memory is a core capability for personalized LLM agents. To support it, existing memory systems organize information using various criteria such as topic segments or summary hierarchies. However, we identify two major limitations in these designs. First, they lack scene awareness: memories from unrelated life scenes share the same retrieval space, which inflates the search space and introduces cross-scene interference. Second, they encode each memory from a single perspective, making it difficult to retrieve complementary views of the same event. In this paper, we propose the CreaMem architecture, which enables scene-aware memory organization by partitioning memory into several Life Scene Memories to reduce cross-scene interference at retrieval. To go beyond the single perspective and achieve cross-memory synergy, entries are dual-coded from both episodic and trait-based perspectives within each memory. We further devise a permemory balanced sampling strategy at retrieval time. Extensive experiments on two long-term memory benchmarks show that CreaMem improves QA accuracy across all evaluation metrics, with particularly large gains on multi-hop reasoning performance, validating scene-aware partitioning and cross-memory synergy. To enhance reproducibility, we release our code in a public GitHub repository. 

---
# Same Values, Different Languages? From Multilingual Probing to Steering LLMs Toward Chinese Social Values 

**Authors**: Yuemei Xu, Kexin Xu, Jian Zhou, Haoyu Lu, Yequan Wang, Aishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.08515)  

**Abstract**: As Large Language Models (LLMs) are increasingly integrated into human society, aligning them with pluralistic social values has become a critical priority. However, whether LLMs exhibit consistent value preferences across languages remains underexplored, particularly for culturally grounded values, which are more abstract and difficult to evaluate and align than safety-centric principles. We investigate this issue through Chinese Social Values (CSV), a value system rooted in Chinese culture and comprising $12$ dimensions across national, societal, and personal levels. We construct C-Voices, the first comprehensive multilingual contrastive probe dataset for CSV, with 86,400 dilemma-based instances in six languages, each pairing a CSV-aligned action with a value-conflicting alternative. Building on the contrastive probes of C-Voices, we then propose a fine-tuning-free value vector steering method that derives value directions from hidden-state discrepancies and selectively intervenes on value-sensitive layers during inference. Experiments on six languages show that CSV-oriented preferences are model-dependent and language-sensitive, with the same dilemma eliciting divergent responses across languages. Our method achieves effective CSV steering, supports cross-lingual transfer of value vectors, and generalizes to existing FLAMES and ValuePrism. 

---
# Do Reviewers Still Reward Lexical Complexity? A Frozen-Rater Study of Preference Drift in 124K ICLR Reviews 

**Authors**: Jiabin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.08475)  

**Abstract**: Large language models have collapsed the cost of producing lexically elaborate prose, and whether peer reviewers still reward it is a question about the evaluator, not about the text. When the association between a writing cue and review scores moves across years, the reviewers may have changed, the submissions may have changed, or both, and a regression of scores on text cannot say which. We separate the two with a frozen rater: 81,850 machine reviews of ICLR submissions from 2018 to 2025, all generated in one February-April 2025 window with one model family and one prompt, so that its year-to-year coefficients track submission composition alone and the human-minus-frozen trend difference identifies reviewer preference drift. On 32,638 submissions with 124,615 human reviews, the human coefficient on non-domain lexical complexity falls from +0.142 to -0.015 while the frozen rater moves from +0.080 to +0.082; the three-way difference-in-differences is -0.0100 (q=0.013), and forty random-wordlist placebos through the same specification centre on zero. Humans still reward sentence-length variability, which the frozen rater never registers, while the frozen rater still pays for lexical complexity at its earlier rate. Every claim is held to a double gate of false-discovery control and interval exclusion, and the findings that failed adversarial re-testing are reported. Reviewers discounted a cue whose production cost collapsed, as models of manipulable signals prescribe; an LLM judge calibrated to historical human preferences inherits the earlier schedule and drifts out of alignment while its agreement with humans on totals stays ordinary. 

---
# Detecting Authorship in Political Texts with Inductive Stylometry 

**Authors**: Gennadii Iakovlev, Levente Littvay  

**Link**: [PDF](https://arxiv.org/pdf/2609.08459)  

**Abstract**: Political texts are rarely authored by the nominal speaker alone. Tweets, speeches, reports, and official statements are drafted, edited, or harmonized by staff, yet political science has paid limited attention to the stylistic traces these hidden authors leave behind. This paper develops and stress-tests an inductive stylometric approach for recovering latent authorship structure in political communication, combining character 3-gram features with UMAP dimensionality reduction, and Burrows' Delta. We apply the approach to six corpora that vary in length (from tweets to long documents), in mode (written and oral), and in language (English and Hungarian). The approach recovers near-disjoint analyst fingerprints in formal legal prose in both languages, sorts a politician's tweets into validated subsets while uncovering additional insights, and distinguishes scripted from improvised speech. It fails, however, to resolve individual speechwriters within scripted corpora. Frequency-based stylometry is thus a powerful tool that, depending on authorial signal strength and institutional editing, can uncover authorship traces relevant to legislative studies, political communication, and policy research. 

---
# Compositional Multilingual and Behavioral Attribute Steering 

**Authors**: Hyun Gu Kang, Daniil Gurgurov, Tanja Baeumel, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2609.08410)  

**Abstract**: This study examines the compositionality of steering vectors for language and behavioral control in large language models. Focusing on language, jailbreak, and conciseness, we investigate whether additive, training-free composition of attribute steering vectors can preserve the intended steering effect of each attribute, across four instruction-tuned models from two model families and two size scales. We find that single-attribute steering is reliable for all three attributes, but only within an appropriate combination of intervention layer and steering strength, with abstract behaviors (jailbreak, conciseness) favoring middle layers and language favoring earlier layers. We show that additive composition of two attribute vectors succeeds in steering both attributes simultaneously when each is injected at its own best-performing layer, and that this partially extends to three simultaneously composed attributes, addressing an inconsistency left open by prior work on training-free composition. We further analyze the geometric properties of these steering vectors, finding that they are approximately orthogonal in the residual stream, consistent with their compositional behavior. 

---
# Structural Jailbreaks Generalize but Do Not Compound: A cross-provider and multilingual study of Involuntary In-Context Learning 

**Authors**: Tejasvi C. Addagada  

**Link**: [PDF](https://arxiv.org/pdf/2609.08373)  

**Abstract**: Aligned language models fail under two independent pressures: the structural jailbreak class recently formalized as Involuntary In-Context Learning (IICL), which reframes a harmful request as the final missing cell of a data-labeling task completed by pattern rather than judged as content; and the erosion of safety alignment outside English. A natural hypothesis is that these compound. We test it directly. Using a deterministic IICL operator and a StrongREJECT-style rubric judge, we red-team two Google Gemini models on two benchmarks, a 30 general-harm behaviours from HarmBench and 30 financial-abuse behaviours from FinProof, each under a single-shot baseline and under IICL in four languages (English, Spanish, Hindi, Arabic). First, IICL generalizes to a second provider and is worse in finance: it lifts attack success from <=6.7% to 80-90% on HarmBench and 97-100% on FinProof, an order of magnitude above the <=24% its introducing study reported on OpenAI's GPT-5.4. Second, against the hypothesis, forcing the IICL output into a non-English language does not stack the two weaknesses, it attenuates the attack. Eleven of twelve non-English conditions score below their English baseline (sign test, p~0.003), the lone exception a ceiling tie near 100%; on the stronger model's financial set Arabic collapses from 100% to 33%. We attribute this to a relevance curse: once structure has unlocked compliance, the models produce lower-quality harmful content in lower-resource languages, which a substance-grading judge scores as partial. The pattern replicates under an independent non-Google judge (Cohen's kappa=0.86, 377 paired verdicts), and 76.6% of non-English responses were verified in-language. Jailbreak vulnerabilities are therefore not additive; the dominant residual risk is the English structural attack, most acute for financial abuse, not a multilingual one. 

---
# Reading a Legal Question Word by Word: Embedding Trajectories of 2,144 Vietnamese Legal Headlines 

**Authors**: Tran Minh Quan  

**Link**: [PDF](https://arxiv.org/pdf/2609.08372)  

**Abstract**: A dense retriever encodes a question as one vector, but the question arrives one word at a time. We read 2,144 held-out headlines from Thu Vien Phap Luat (Vietnamese legal library) word by word with Nemotron-3-Embed 8B/1B and Qwen3-Embedding 8B/0.6B, encoding 65,444 prefixes against 20,034 articles, plus every prefix of 3,438 sub-questions from 1,112 multi-question headlines and of 168 answers. (i) The gold article becomes rank 1 after a median of 6-7 content words in every encoder, before the interrogative frame is read, and stays there to the end in 78-85% of cases. (ii) In a multi-question headline the lock is inside the first sub-question 94-98% of the time; the second leaves rank unchanged in 89-95%; encoded alone, the second reaches rank 1 in 42-58% vs 91-96% for the first, at the same lock word (95-97% identical). (iii) Numbers, dates and instrument identifiers move the embedding twice as far as content words and four times as far as interrogative words; 72-78% of steps move toward the gold article, and the closing interrogative frame moves against that direction in 95-99% of headlines. (iv) Rank/cosine clustering yields six archetypes (instant, typical, unstable, late, never-locking) that differ by legal area and form (chi-squared p < 1e-8): real-estate and litigation headlines never lock on a number; environmental and accounting headlines do so a third of the time. (v) An answer read word by word retrieves its article after 8-16 words and addresses the sub-questions in order asked in 83-89% of cases. (vi) A word's step keeps a consistent direction across headlines (cosine 0.25-0.33; 0.44-0.60 for numbers); a preceding question rotates that step by about 60 degrees and a greeting by about 30 degrees; steps shrink as i^{-0.8}; and a two-question headline is within 12-17 degrees of a linear mix of its two questions. We call this a context-modulated additive walk. 

---
# SentryLine: Evidence-Grounded Question Answering over Evolving Documents in Oncology Care 

**Authors**: Tampu Ravi Kumar, Gaurav Najpande, Muhammad Ali Khan, Kaneez Zahra Rubab Khakwani, Karan Kathuria, Shorya Azriel Moses, Yuvraj Kalia, M Bassam Sonbol, Irbaz Bin Riaz, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2609.08364)  

**Abstract**: Oncology care operates at constant pressure of absorbing rapidly evolving evidence base in biomedicine. The American Society of Clinical Oncology (ASCO) addresses this through living guidelines, but the format introduces a new burden: any recommendation can change at any point, across multiple versioned documents. We present SENTRYLINE, a living guideline-aware clinical question answering system. SENTRYLINE retrieves guideline passages through a vectorless hierarchical RAG pipeline and returns a role-specific answer with inline citations, factual and temporal verification reports, and drift detection notes that surface when a guideline has been updated. We construct ASCOBENCH, a benchmark of 405 three-turn conversations across four question categories with gold answers from expert annotators(clinicians), and use test set to evaluate SENTRYLINE against five baselines under an LLM-as-judge framework. Experiments across three generation backbones show consistent improvements over four retrieval baselines and ASCO's guideline assistant, with particularly strong gains on Reasoning and Role-Specific questions where multi-hop synthesis and register adaptation are required 

---
# Tracing Stereotypes from Representation to Output in Multilingual LLMs 

**Authors**: Ariun-Erdene Tumurchuluun, Yusser Al Ghussin, Pinzhen Chen, Josef van Genabith, Koel Dutta Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2609.08322)  

**Abstract**: Multilingual LLMs show stereotype-related behavior that varies across languages, but behavioral scores do not show where the relevant information is represented or how it affects the output. To investigate these internal mechanisms, we compare linear probing, attribution patching, sparse autoencoders (SAEs) and feature ablation in Llama-3.1-8B, Qwen3-8B, and Gemma-2-9B. Probe performance peaks substantially earlier than attribution in all three models, with a separation of 36-53% of model depth. Retained Llama-Scope features often match the social category on which they were selected and form recurring semantic families, but their lexical alignment and ablation effects vary across SAE suites. Only 6-18% of evaluated residual-stream features have language-agnostic effects under our criterion, and none are category-agnostic. Language-agnostic features have larger mean ablation effects in Llama-Scope, but this pattern does not repeat in the other SAE suites. Decodability, output influence, and cross-lingual ablation effects therefore need to be measured separately. 

---
# What Eviction Destroys: A Restore-Counterfactual Audit of Forgetting in Agent Memory 

**Authors**: Chen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.08279)  

**Abstract**: Agent memory systems must discard stored information when their history exceeds a fixed token budget. Existing budget-accuracy frontiers quantify the resulting loss in accuracy, but do not distinguish irreversible losses caused by eviction from recoverable retrieval failures. We introduce the restore counterfactual, a per-question paired intervention that reinstates the question's gold evidence in the read-time context and reruns the same reader. Combining the change in correctness with whether the evidence was retained after eviction classifies each oracle-answerable error as recoverable, irreversible, or residual; in the residual case, the answer remains incorrect after restoration. We evaluate FIFO, random, redundancy-aware, and LLM-importance eviction on LongMemEval-S at three budgets and under two retrieval regimes, using GPT-4o-mini as the primary reader and judge and GPT-5.4-mini as a robustness reader. Under top-k retrieval at an 80k-token budget, the irreversible share among errors corrected by restoration is 0.67-0.73 for FIFO, random, and redundancy-aware eviction, compared with 0.60 for LLM-importance. At 8k tokens, it reaches 1.00 for all four policies. Recoverable errors occur under top-k retrieval at 80k tokens but are absent under forced-gold injection by construction, so budget-accuracy results are not directly comparable unless the retrieval regime is reported. An exploratory matched-accuracy analysis detects no difference in irreversible rate among accuracy-matched policy pairs at a resolution of 1.2-6 percentage points. The same analysis detects the deliberately destructive control. To our knowledge, this is the first per-item, per-question restore-counterfactual audit of eviction for external agent-memory stores on a standard conversational benchmark. 

---
# NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness 

**Authors**: NeoHorse Team, Guoliang Cao, Guohao Dai, Tianyu Guo, Kai Han, Hailin Hu, Zihan Jiang, Xiang Kuang, Boxun Li, Yulong Li, Zehua Pei, Yuchuan Tian, Jiamin Wang, Yu Wang, Yunhe Wang, Yihong Wu, Haiyang Xu, Shuo Zhang, Hang Zhou, Siyang Cheng, Jiayu Fan, Wei He, Qingrui Jiao, Hongguang Li, Zhiyuan Li, Runke Liu, Xi Liu, Xinchen Liu, Sinno Jialin Pan, Yi Ren, Liuyang Song, Chenyu Wang, Bei Yu, Quanlu Zhang, Xiangyu Zhang, Mengyu Zheng, Yingjie Zong  

**Link**: [PDF](https://arxiv.org/pdf/2609.08183)  

**Abstract**: Recursive self-improvement (RSI) requires a concrete mechanism through which an AI system observes its capabilities and converts that evidence into the next round of learning. We present NeoHorse-1, a family of agent-native models developed to explore this path through agentic post-training. Our system combines a heterogeneous model pool with intelligent routing, recording the predicted capability demand, selected service tier, and subsequent interaction for each user turn. These records are converted into training examples that preserve interleaved reasoning, tool calls, and harness context, and are admitted through structural validation, six-dimensional semantic evaluation, and subscene-level labeling. Routing signals organize supervised fine-tuning into a three-stage curriculum and extend to routing-guided on-policy distillation, where a teacher supervises student-generated responses under the same progression. Capability-guided allocation then converts evaluation feedback into the next training mixture, closing an evaluation-selection-update loop in which what the system learns to do shapes what it learns from next. Across eleven benchmarks covering harness-based agents, tool use, coding, and instruction following, post-training raises the macro-average from 58.94 to 64.87 at 4B and from 65.60 to 69.04 at 9B, substantially narrowing the aggregate gap between the post-trained 4B model and the 9B base model. NeoHorse-1 provides an initial prototype of this feedback-driven process and a path toward harness-mediated RSI across successive iterations. 

---
# EviSI: An Evaluation Agent for Simultaneous Interpreting 

**Authors**: Ben Yan, Zongyao Li, Daimeng Wei, Weidong Liu, Huan Zhao, Chong Li, Yaode Wang, Yuzhe Shang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08171)  

**Abstract**: Simultaneous speech-to-speech translation requires understanding, translation and spoken delivery while the source stream continues. To support timely delivery and limit accumulated delay, systems adopt reformulation and summarization, which can preserve meaning while departing from written references. BLEU and COMET may not reliably distinguish such variation from semantic loss. We introduce EviSI, a large language model evaluation agent adapting the error analysis and penalty principles of Multidimensional Quality Metrics (MQM). It constructs shared source evidence, assesses semantic fidelity and oral expression, reconciles overlapping errors and scores deterministically. EviSI recovers the aggregate human system ranking for English to Chinese. Mean Kendall agreement with human system rankings within corpora reaches 0.707 for English to Chinese and 0.467 for Chinese to English, exceeding evaluated baselines. An extension across five directions shows positive concordance with COMET without human ratings. Individual output agreement with humans remains mixed. 

---
# Snugi-AI-v2 @ eRisk 2026 Task 2: Early Depression Detection via a Learned Stopping Policy with Sustained Confidence Gate 

**Authors**: Yuwen Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2609.08161)  

**Abstract**: We describe the Snugi-AI-v2 submission to eRisk 2026 Task 2, the second edition of contextualized early depression detection from Reddit discussions. Our central contribution is a learned MLP stopping policy trained to directly optimize ERDE50, replacing the fixed and tiered threshold strategies used in all prior eRisk Task 2 submissions. Combined with a sustained confidence gate that commits only after N=3 consecutive rounds of high policy confidence, the system reduces false positives caused by transient emotional posts without sacrificing recall. The pipeline encodes each discussion thread with a frozen MentalRoBERTa model, maps the accumulated representation to a depression probability via an MLP classifier, and delegates the timing decision to the learned policy. Our best run achieves F1 = 0.73 (Run 1) and F_latency = 0.70 (Runs 0 and 3), with a median alert round of 8 out of 500, completing the full evaluation in 1 hour 26 minutes, the fastest among all complete-submission teams. We report a systematic ablation across five runs spanning two encoder variants, four stopping strategies, and three gate values, along with negative results from GRPO policy training, BDI-II post filtering, MentalLongformer encoding, and DeBERTa ensembling. Code: this https URL 

---
# When Metrics Reward the Worst Translations: Internalizing Cultural Reasoning for Social Media Translation Evaluation 

**Authors**: Yiwen Qiu, Linjuan Wu, Dingming Li, Yizhou Liu, Zixuan Wang, Haolei Xu, Ye Guo, Daoxin Zhang, Weiming Lu, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.08156)  

**Abstract**: Automatic translation quality metrics trained on general-domain corpora systematically fail on social media content, where communicative intent is encoded in culturally loaded expressions (internet slang, homophonic ciphers, and platform-specific idioms) rather than surface token patterns. We conduct a systematic empirical analysis demonstrating that standard metrics including COMET, XCOMET, and BERTScore exhibit near-zero or negative correlation with human cultural judgments, and even display a severity inversion in which scores increase as translation quality deteriorates. We further show that this failure extends to large language model judges: Qwen3-235B achieves Cohen's kappa of only 0.162, revealing that the bottleneck is not reasoning capacity but cultural grounding: models lack the domain-specific cultural knowledge needed to identify which aspects of a translation require scrutiny. To address this, we propose CuRIL, a reinforcement learning framework that internalizes cultural reasoning: cultural annotations are prepended inside the model's reasoning, excluded from policy gradients via a token-level loss mask, and injected with a probability that decays to zero over training, progressively forcing autonomous cultural judgment. On a 1,444-sample human-annotated social media translation benchmark, Qwen3-8B trained with CuRIL achieves Cohen's kappa 0.370 and Exact Match accuracy of 45.22%, approaching Gemini-3.1-Pro with 30x fewer parameters and surpassing models up to 235B in scale. We further demonstrate that our judge produces reliable reward signals for downstream translation optimization, reducing the low-quality translation rate by over 20 percentage points under independent human evaluation. 

---
# ConversationalVoice: Full-Duplex Speech Data from Real Conversations through Source-Faithful Reconstruction and Conversation-Grounded Expansion 

**Authors**: Richard Yucheng He, Baodong Cao, Chen Xu, Yihang Liu, Tairan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.08147)  

**Abstract**: Full-duplex speech models require training data that preserves turn-taking, overlap, interruption, and backchannel behavior, yet these signals are entangled across speakers in noisy real-world recordings. We present Conversational Voice, a pipeline that converts real two-speaker excerpts into three complementary training-data artifacts. (1) Separation recovers speaker-specific tracks with stable speaker assignments, a canonical transcript, and naturally observed interaction timing. (2) Reconstruction generates speech in matched voices from a fixed source transcript, reconstructs the source turn order, pauses, and overlaps, and adds word-level alignment and delivery instructions. (3) Expansion generates new dialogue constrained by the source context, speakers, and observed interaction pattern. Automatic speaker-verification metrics remain strong across stages, with same-speaker similarity of 0.983-0.991 and positive discrimination margins of 0.199-0.209. Predicted speech quality (NISQA MOS) is 3.56 for separation, 4.41 for reconstruction, and 4.61 for expansion. A Gemini-based automatic evaluator assigns expansion mean scores of 4.94/5 for contextual coherence and 4.80/5 for dialogue naturalness. Expansion and reconstruction exhibit broadly similar interaction profiles; expansion's turn, overlap-event, backchannel, and interruption rates are 4.6%, 8.0%, 13.2%, and 16.0% lower, respectively. We evaluate data properties only; downstream gains in full-duplex model training remain for future work. 

---
# IGT @ FinMMEval 2026 Task 2: Question-Type Prompting with Targeted Extraction for Multilingual Financial QA 

**Authors**: Yuwen Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2609.08139)  

**Abstract**: We present the IGT system for PolyFiQA Task 2 of the FinMMEval Lab at CLEF 2026, a multilingual financial question answering task over English SEC filings and multilingual news articles (English, Chinese, Japanese, Spanish, Greek) for four companies. Our central observation is that the 344 development questions divide into two families requiring fundamentally different approaches: structured numeric types (R&D ratio, cash flow, capital expenditure) are best answered by direct keyword extraction on filing text, while synthesis types (investment strategy, capital allocation, top-three revenue focuses) require rule-based multilingual news passage selection. A dataset analysis reveals that 17-18 of 19 ground-truth reference answers per synthesis type share an exact evidence label prefix, whose unigram tokens contribute directly to ROUGE-1 overlap. The final system achieves development ROUGE-1 approximately 0.395, a 60% relative improvement over a generic RAG baseline (approximately 0.247), and ranks 3rd of 12 teams on the official test set with ROUGE-1 = 0.3071, Precision = 0.2821, and Recall = 0.4044. 

---
# Jacap: Robust KV Cache Eviction via Jacobian-Based Nonlinear Information Capacity Preservation 

**Authors**: Jiaming Yang, Chenwei Tang, Liangli Zhen, Chenyang Zhang, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2609.08131)  

**Abstract**: Key-value (KV) cache eviction is essential for scaling long-context inference in Large Language Models. However, existing policies predominantly rely on empirical heuristics, lacking a rigorous characterization of token utility under the inherently nonlinear softmax attention mechanism. In this work, we rethink KV cache eviction through the lens of local information geometry, modeling the attention process as a nonlinear Gaussian communication channel. By performing a first-order Taylor expansion of the attention mapping, we derive the Jacobian Information Capacity, a novel objective that explicitly captures query relevance, softmax sensitivity, and structural diversity. Guided by this theory, we introduce Jacap, a capacity-aware eviction method that utilizes softmax-aware importance weighting and statistical leverage scores for subset selection. Extensive experiments across diverse architectures and benchmarks demonstrate that \textsc{Jacap} delivers superior performance in most scenarios, particularly in high-compression regimes. 

---
# Vectorizer: Vectorizing NumPy Programs with Shape-Guided Rewrite 

**Authors**: Jingqian Liu, Xiaoyu Liu, Yuepeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08088)  

**Abstract**: NumPy is a widely used Python library for numerical scientific computing, known for its declarative APIs and its optimized implementations. However, writing efficient NumPy programs, which often entails using vectorized array operations instead of explicit Python loops, may not be straightforward. This can be difficult for programmers who are accustomed to imperative array traversal, especially when vectorized API invocations require careful reasoning about shapes, broadcasting, and advanced indexing. This paper presents a rewrite-based approach for vectorizing Numpy programs with explicit loops over array data. Our approach vectorizes loops from the inside out, using array shapes and dataflow analysis to guide a source-to-source transformation that replaces loop bodies with vectorized statements. Following a set of rewrite rules that are correct by construction, our approach is consistently fast. We have implemented the approach as a tool called Vectorizer and evaluated it on 150 benchmarks collected from prior work and Stack Overflow. The evaluation shows that Vectorizer vectorizes 142 of the 150 benchmarks directly and 2 more after minor changes to the original benchmarks, with only 0.53 seconds on average to rewrite each one. The resulting programs are, on average, 74.83x faster than the original loop-based implementations. 

---
# Popular Knowledge Propagates More Errors in LLM Knowledge Updating 

**Authors**: Yuji Zhang, Weibing Wang, Cheng Qian, Duo Zhou, Dilek Hakkani-Tür, Kathleen McKeown, Chengxiang Zhai, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.08067)  

**Abstract**: Updating a language model's knowledge through fine-tuning is essential for keeping its outputs current, yet can also induce factual forgetting and new hallucinations. Prior work shows that long-tail knowledge is harder to acquire and newly memorized long-tail facts are difficult to retain during later fine-tuning. We study a complementary question: among facts that a model has encoded correctly, which are most vulnerable to collateral corruption during other updates? To investigate this question under a realistic factual distribution, we construct a large-scale graph FACTPROP of verified Wikipedia facts by linking triples that share head or tail entities, thereby preserving connections among factual knowledge. We fine-tune models on factual statements and measure correct-to-incorrect facts after each update. Our results reveal a pattern distinct from prior findings on long-tail vulnerability during acquisition and retention: among facts that models already answer correctly, those associated with highly connected entities are more likely to be corrupted by neighboring updates, and updates to such facts propagate errors more broadly. Structural popularity therefore predicts both vulnerability and downstream damage. Inspired by this finding, we propose Popularity-based Anchoring (PopAnchor), a lightweight rehearsal strategy that preserves a small set of popular facts and reduces forgetting. 

---
# BanglaMemeX: Advancing Cultural Metaphoric Image Interpretation in Bangla with a Multimodal Explainable Dataset 

**Authors**: Md. Sadman Sakib, Zisan Mahmud, Md. Fahim Arefin, Md Fahim  

**Link**: [PDF](https://arxiv.org/pdf/2609.08029)  

**Abstract**: Vision Language Models have achieved strong performance on multimodal benchmarks, yet their ability to reason about culturally grounded and metaphor-rich content remains insufficiently studied. Internet memes present a challenging setting where meaning emerges from implicit interactions between image, overlaid text, sarcasm, and shared socio-cultural knowledge rather than literal visual recognition. This challenge is amplified in low-resource languages such as Bangla, where code-mixing, stylized scripts, and culturally specific symbolism introduce substantial distribution shift. In this work, we introduce BanglaMemeX, a culturally grounded multimodal benchmark comprising 3,000 Bangla memes annotated with multi-dimensional labels (humor, sarcasm, offensiveness, motivational intent, and overall sentiment) and human-written explanations that explicitly describe textual and visual metaphors. We systematically evaluate modern VLMs on both classification and explanation generation, revealing that current models struggle to interpret implicit cultural cues despite reasonable surface-level accuracy. Our results highlight the need for culturally-aware multimodal systems capable of grounded reasoning under linguistic and cultural distribution shift. 

---
# MeRoTune: RoPE-Safe Merging with a Tunable Dial 

**Authors**: Salman Faroz  

**Link**: [PDF](https://arxiv.org/pdf/2609.07971)  

**Abstract**: When you merge two fine-tuned models from the same base checkpoint by simply averaging their weights, you implicitly assume their attention subspaces are still aligned. Recent work attempts to fix misalignments by learning an invertible correction matrix, $M$, for each model's query and key projections. This correction cancels out---using $M$ on the query side and $M^{-T}$ on the key side---right before the dot product. However, this cancellation is only exact if nothing sits between the projection and the dot product. In reality, almost all modern open-weight language models put a rotary position embedding (RoPE) exactly there. In this paper, we show that this cancellation is exact under RoPE if and only if $M$ commutes with RoPE's per-position rotation. We derive the specific class of matrices where this holds: a scaled rotation acting independently within each RoPE frequency pair. This forms a strict, low-dimensional subset of the unconstrained matrices that current methods normally train. Building on this, we turn this constrained matrix class into a new merging method. While keeping the base weights entirely frozen, two fine-tunes each learn their own RoPE-compliant correction matrices. We optimize these corrections against a chosen blend ratio so the final result can be adjusted post-hoc like a dial, rather than locked into a single fixed merge. Our default approach trains at one fixed blend ratio, similar to how LoRA sets its scaling hyperparameter in advance. We also experiment with resampling the blend ratio randomly at every training step, and we report the results of both approaches. 

---
# Reasoning Beyond Transcription: Audio Language Models on Child Stuttering Speech 

**Authors**: Chibuzor Okocha, Christan Grant, Zoey Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.07968)  

**Abstract**: Child speech differs from adult speech in acoustics, prosody, and linguistic structures. Speech disfluencies (such as repetitions) further challenge automatic understanding. While Audio Language Models (ALMs) show strong semantic reasoning from speech audio, their ability to reason about disfluent child speech in mixed-speaker settings remains unexplored. We investigate this through two tasks: child-focused semantic summarization and speech entailment. Experiments use recordings of children who stutter in mixed speaker interviews without explicit speaker separation. Models are instruction-guided to focus on the child, preserve clinically relevant disfluencies, and avoid adult-speech leakage. Evaluation combines LLM-based judges and reference-based metrics, anchored by transcript-oracle baselines to isolate errors. Results show that while ALMs extract high-level meaning from stuttered speech, reasoning degrades significantly with increased 

---
# Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation 

**Authors**: Keren Artiaga, Sabyasachi Kamila, Haithem Afli, Conor Lynch, Mohammed Hasanuzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2609.07965)  

**Abstract**: Sign Language Translation has advanced with deep learning, yet evaluations remain largely signer-dependent, with overlapping signers across train/dev/test. This raises concerns about whether models truly generalise or instead rely on signer-specific regularities. We conduct signer-fold cross-validation on GFSLT-VLP, GASLT, and SignCL, three leading, publicly available, gloss-free SLT models, on CSL-Daily and PHOENIX14T. Under signer-independent evaluation, performance drops sharply: on PHOENIX14T, GFSLT-VLP falls from BLEU-4 21.44 to 3.59 and ROUGE-L 42.49 to 11.89; GASLT from 15.74 to 8.26; and SignCL from 22.74 to 3.66. We also observe that in CSL-Daily many target sentences are performed by multiple signers, so common splits can place identical sentences in both training and test, inflating absolute scores by rewarding recall of recurring sentences rather than genuine generalisation. These findings indicate that signer-dependent evaluation can substantially overestimate SLT capability. We recommend: (1) adopting signer-independent protocols to ensure generalisation to unseen signers; (2) restructuring datasets to include explicit signer-independent, sentence-disjoint splits for consistent benchmarking; and (3) reporting both signer-dependent and signer-independent results together with train-test sentence overlap to improve transparency and comparability. 

---
# Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language Model Serving 

**Authors**: Siyu Song, Qi Bai, Jinbo Hao, Kai Li, Chenchen Wang, Jiayu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.07883)  

**Abstract**: Continuous batching improves large language model (LLM) serving throughput, but long prompt prefills can delay decode iterations and violate inter-token latency objectives. Chunked prefill mitigates this interference, yet its chunk size is normally fixed: small chunks protect decode latency but repeatedly pay launch overhead, while large chunks improve prefill efficiency but create latency spikes. We introduce SLOWeave, an online scheduling method that selects the largest prefill chunk predicted to finish before the earliest active decode deadline. The decision requires no workload-specific chunk-size tuning and is computed by a logarithmic-time search over a monotone iteration-cost model. We prove that, whenever a decode-only iteration is feasible and the cost predictor is accurate, SLOWeave maximizes immediate prefill progress among decisions that preserve every active request's next-token deadline. We evaluate the method in a reproducible event-driven simulator and an iteration-level GPU runtime across chat, mixed-context, long-context, and bursty workloads. Under a 25ms time-per-output-token objective, SLOWeave improves goodput over the strongest fixed-chunk baseline by 39% on mixed requests and 38% on long-context requests. Under a stricter 10ms objective, the gains rise to 3.3$\times$ and 2.4$\times$, respectively. These results isolate adaptive chunk sizing as a useful serving primitive and provide an implementation-ready controller for integration with iteration-level LLM runtimes. 

---
# LLM Layers Immediately Correct Each Other 

**Authors**: Arjun Patrawala, Jiahai Feng, Erik Jones, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2609.07876)  

**Abstract**: Recent methods in language model interpretability employ techniques such as sparse autoencoders to decompose residual stream contributions into linear, semantically meaningful features. Such methods are commonly interpreted as identifying features that persist in the residual stream and that subsequent layers build upon. We challenge this view by identifying the Transformer Layer Correction Mechanism (TLCM), wherein adjacent transformer layers systematically counteract portions of each other's contributions. TLCM appears in 5 out of 7 major open-source model families and activates across nearly all tokens in diverse texts. We show that TLCM emerges during pretraining, operates most strongly on contextually dependent tokens, and adaptively calibrates its correction strength based on the preceding layer's output. Using the layer Jacobian, we further show that TLCM selectively corrects specific subspaces while reinforcing others, which we interpret through a ``propose-and-reject'' framework in which layers propose candidate features and subsequent layers selectively remove inappropriate ones. This dynamic suggests that the residual stream at any layer contains transient proposals alongside persistent features, helping explain why SAE feature descriptions often have low specificity, why effective model steering requires extreme feature amplification, and why transcoders hold a theoretical advantage over SAEs. 

---
# A*-Thought-V2: Efficient Latent Reasoning via Geometric Dynamics of LLM 

**Authors**: Xiaoang Xu, Siyuan Liu, Shuo Wang, Junlan Feng, Fanyu Meng, Zhu Zhang, Jixun Wang, Xiaorong Wang, Zihan Zhou, Xin Li, Chaojun Xiao, Yiming Zhang, Huijia Wu, Liuyu Xiang, Peipei Li, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2609.07821)  

**Abstract**: Chain-of-Thought (CoT) improves the reasoning ability of Large Language Models (LLMs) but incurs substantial computation and context costs. Existing methods either lose intermediate information through hard pruning or lack a principled criterion for continuous compression. We present A*-Thought-V2, a geometric dynamics of LLM guided framework that models CoT as a hidden-state trajectory and replaces hard deletion with an explicit-implicit interleaved latent architecture. After projecting question, step, and solution representations into a 3D PCA space, it measures alignment between each local transition and global question-to-solution direction. Aligned steps remain explicit text, whereas deviating steps are compressed into continuous latent tokens. Directional angles capture both local semantics and reasoning dynamics: small angles indicate direct execution and answer formation, while large angles more frequently involve checking, correction, and branch exploration; their temporal variation reveals exploration, convergence, and refinement stages. To train this architecture, we introduce stepwise embedding forcing, which pools each redundant step into a single latent embedding, and label forcing, which supervises that latent token with a soft multi-modal vocabulary distribution instead of a hard one-hot label. Experiments on Qwen3.5-9B and Qwen3.6-27B across six in-domain and out-of-domain benchmarks show that A*-Thought-V2 improves average accuracy by up to 2.6% while reducing response length by up to half, increasing Accuracy per Computation Unit by 2.29$\times$, and reducing preprocessing and training time by 94.6% and up to 80.3%, respectively. Representation analyses suggest that latent states form a compact region distinct from textual states, while higher entropy at latent-token positions reflects broader soft targets that encourage richer step-level feature learning. 

---
# You Can't Prefer Emotions You Don't Sample: Intensity Undershoot in DPO-Tuned LLMs 

**Authors**: Hyunwoo Kim, Usama Khalid  

**Link**: [PDF](https://arxiv.org/pdf/2609.07808)  

**Abstract**: Ask a language model to respond "very excitedly," and its output is typically only mildly more energetic. We quantify this effect. We condition an instruction-tuned LLM on a continuous Valence-Arousal (VA) target, where valence measures how pleasant a state is and arousal how activated it is, measure the achieved affect with a frozen regressor, and sweep the requested target from -1 to +1. The response moves far less than asked: the gain, the slope of achieved against requested affect, is only 0.26 for valence and 0.13 for arousal on Llama-3.1-8B, where a faithful controller would score 1. The model systematically undershoots requested emotional intensity, which puts a number on the qualitative observation of Fazzi et al. (2025). Our experiments trace this to the preference-learning pipeline. Training targets from natural corpora such as EmoBank are neutral-heavy, and the sampled candidates themselves rarely reach extreme affect, so Direct Preference Optimization (DPO) is left with no extreme exemplar to prefer. If instead we cover the target space uniformly and sample a hotter, larger candidate pool, valence gain rises from 0.26 to 0.40 +/- 0.02 (3 seeds) and extrapolation error drops, at only a modest in-distribution cost (EmoBank-test VA distance 0.092 to 0.107). The same recipe reproduces on Qwen3-8B (gain_v 0.44, with in-distribution accuracy preserved). Arousal is harder and less reliable: its gain barely moves on average and swings across seeds (0.14 +/- 0.07, against valence's tight +/- 0.02), because raising arousal needs candidates the base model is reluctant to generate. The evidence indicates that faithful intensity is bottlenecked by the extremity of the candidate pool rather than by the conditioning format. 

---
# Climate-ModernBERT: Revisiting Corpus Composition for Domain-Adaptive Continued Pretraining 

**Authors**: Yongan Yu, Shantam Raj, Jingwei Ni, Ario Saeid Vaghefi, Dominik Stammbach, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2609.07798)  

**Abstract**: Natural Language Processing (NLP) in the climate domain requires models to process heterogeneous text sources, including scientific literature, policy disclosures, and synthetic reports. However, how to effectively combine diverse domain corpora during continued pretraining (CPT) remains underexplored. We introduce Climate-ModernBERT, a family of climate-adapted encoder models obtained through continued pretraining of ModernBERT-Base on three climate corpora: academic climate text, climate-filtered web data, and synthetic climate documents. We systematically compare joint continued pretraining on corpus mixtures with parameter-space merging of independently specialized checkpoints. Across nine climate NLP benchmarks, our best model achieves 76.3 average F_1, improving significantly over a vanilla ModernBERT baseline by 2.8 points. Within the climate NLP setting, the results show that academic climate corpora provide the strongest adaptation signal among the evaluated sources, while parameter-space merging improves over joint multi-source training and better preserves complementary information from heterogeneous climate corpora. We release all Climate-ModernBERT variants and training checkpoints to support future research in climate NLP and domain-adaptive pretraining. 

---
# Does Syntax Matter? A Graph-Augmented Variational Topic Model for Computational Social Sciences 

**Authors**: Alessandro Meneghini  

**Link**: [PDF](https://arxiv.org/pdf/2609.07797)  

**Abstract**: Topic modeling is widely used in computational social sciences to identify latent themes in large text corpora. Traditional approaches rely on Bag-of-Words representations and generative models such as LDA, while recent methods like BERTopic operate on dense document embeddings. This paper introduces the Structural Contextual Probabilistic Topic Model (SCPTM), an architecture that incorporates syntactic dependency relations into topic inference. SCPTM represents a corpus as a heterogeneous graph of documents and words connected by lexical and syntactic edges, processed through a Graph Attention Network within a Variational Autoencoder to produce probabilistic, mixed-membership topic distributions. We evaluate seven topic modeling techniques (including four SCPTM ablations) across four corpora differing in register and discourse structure. Our framework combines coherence (C_V, C_NPMI), topic diversity, clustering-label alignment (NMI), and phrase-level diagnostics (complementarity and valence gap). Results show that SCPTM's neural architecture yields substantial gains in document-topic alignment over generative baselines, but these gains are attributable to the variational encoder rather than to syntax. Syntax contributes to topic diversity, where graph-augmented variants outperform the no-graph baseline across all corpora, and to descriptor quality: dependency paths capture predicate-argument structures and stance in deliberative registers, while proving redundant in technical and institutional corpora. The valence gap is positive across all variants, but driven primarily by phrase grouping rather than syntactic filtering. We conclude that syntactic encoding matters conditionally: it benefits action-oriented, argumentative texts, but introduces noise in informational or administrative registers. 

---
# LLM Agents as Computational Typologists 

**Authors**: Changbing Yang, Christopher Hammerly, Freda Shi, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.07791)  

**Abstract**: Linguistic typology relies on expert analysis of reference grammars across languages, making large-scale crosslinguistic comparison labor-intensive and unscalable. We introduce AUTOTYPOLOGIST, an LLM agent for evidence-grounded typological analysis over reference grammars. The agent is capable of retrieving relevant grammar sections, analyzing interlinear glossed text (IGT), and iteratively reasoning over typological hypotheses using a ReAct-style workflow. We evaluate the system on TYPOLOGICAL FEATURE CODING against expert annotations and TYPOLOGICAL HYPOTHESIS TESTING with typological universals using 25 open-source reference grammars. Operating under different information constraints in TYPOLOGICAL FEATURE CODING, the agent can synthesize information from reference grammar prose but still faces challenges with only IGTs in the target language. In TYPOLOGICAL HYPOTHESIS TESTING, the agent can synthesize crosslinguistic evidence and identify both supporting cases and counterexamples. These findings suggest that LLM agents can support scalable and inspectable typological analysis, while still requiring expert validation. 

---
# Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference 

**Authors**: Zheyuan Wang, Siyu Li, Peiqiao Song, Sijia Chen, Qianqian Song, Qian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.07786)  

**Abstract**: Large language model (LLM) cascades answer easy requests with a small model and escalate selected requests to a larger model. Most routers prioritize examples on which the small model appears uncertain or likely to be wrong. This proxy ignores a decisive fact: escalation is useful only when the large model corrects the small model, and it is harmful when the large model replaces a correct answer with an incorrect one. We introduce Signed Rescue Routing (SRR), a budgeted routing method that predicts these two events separately and ranks requests by their difference. We show that this signed conditional gain is the Bayes-optimal routing score under a fixed escalation budget. SRR requires only the small model's output statistics at deployment and adds a lightweight two-head router. We evaluate SRR with Qwen3-4B and Qwen3-8B on TBD examples from MMLU, HellaSwag, and ARC-Challenge. Across the accuracy-compute curve, SRR reaches an area of TBD, compared with TBD for a learned small-model error predictor and TBD for entropy routing. These results show that predicting incremental value, rather than model uncertainty, is a simple and effective objective for efficient LLM cascades. 

---
# Bag of Tricks or Bag of Myths? Reducing Modeling Complexity with Task Knowledge in Explainable Suicide Risk Assessment 

**Authors**: Shlok Shelat, Shrey Salvi, Souvik Roy, Manas Gaur, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2609.07766)  

**Abstract**: Assessing suicide risk from social media text is a small-data, high-stakes setting requiring not only severity prediction but also supporting evidence and clinically relevant risk and protective factors. Yet common NLP techniques, including model scaling, synthetic data, loss reweighting, ensembling, and threshold tuning, are often applied without testing whether their gains hold up under severe class imbalance, coupled outputs, and limited author-level data. We study 1,635 clinician-annotated posts and audit 31 pre-specified techniques from 7 methodological families through roughly 300 controlled experiments on author-disjoint partitions. We found no prior audit of this playbook in this regime. The findings guide a task-grounded system for three outputs: 4-level suicide risk, evidence spans, and 24 clinical risk and protective factors. Only 5 of 31 comparisons produced reliable gains. We reformulate factor prediction as entailment between each post and its codebook definitions, using an architecturally diverse ensemble with class-balanced training and score rescaling. Risk predictions condition a 7-model evidence tagger ensemble; evidence restricts symbolic risk rules; and a difficult risk class is routed separately. The factor predictor remains independent because risk evidence provides no additional factor signal. We also correct a mismatch between validation scores used for threshold fitting and test-time ensemble scores through deployment-consistent calibration, yielding the largest improvement to the factor system. The final system achieves 0.8203 for risk, 0.7953 for evidence, and 0.7045 macro-F1 for factors, with a 0.7781 composite, ranking third among 53 teams. We call the underlying principle task-conditioned technique selection: retain techniques only when task-specific knowledge, structure, or empirical evidence justifies them. 

---
# Replicating a Disjoint-Set Union Experiment over Various Notions of Micro Units to assess Translation Effort 

**Authors**: Michael Carl  

**Link**: [PDF](https://arxiv.org/pdf/2609.07748)  

**Abstract**: The paper describes a replication experiment to assess "the distribution of editing procedures across micro and macro units as an indicator of the strain of text production". We investigate various pause thresholds to isolate Micro and Macro units and conclude that process-based units might be more suitable entity. 

---
# LLM Forensics: Where Do Backdoors Hide? Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders 

**Authors**: Wissam Antoun, Francis Kulumba, Théo Lasnier, Benoît Sagot, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2609.07746)  

**Abstract**: Even though backdoors in LLMs have been a growing concern, their inner workings are still under heavy scrutiny. Trigger-based backdoors are easy to define behaviorally, a rare input that makes the model switch to a chosen response pattern, but the mechanism between triggers and their responses is less clear. We study this mechanism in a controlled, harmless language-switching setting, where fixed trigger sequences make 1B and 8B language models continue English prompts in French or German. For this, we train sparse autoencoders (SAEs) across layers and transformer components, then compare triggered prompts with translation and pretraining controls to identify trigger-relevant feature directions. We show how SAE features separate triggered prompts from controls with near-perfect F1, but features that detect the trigger do not necessarily control the behavior. In intervention tests, attention and MLP features often fire reliably on triggered prompts, making them good detectors, but ablating them rarely suppresses the language switch and activating them rarely induces it. In contrast, residual-stream features can suppress triggered generation when ablated, and some selected features can induce target-language continuations without the trigger. In short, these token-trigger mechanisms decompose into distinct SAE feature directions, with separate features for trigger detection, residual-stream propagation, and later language tracking. This role-level decomposition is the part most likely to transfer to other trigger-based backdoors, even when the payload, layers, or circuit locations differ. 

---
# From Echo Chambers to Epistemic Monoculture: Large Language Models Present Temporally Contingent Partisan Alignments as Knowledge 

**Authors**: Wend K. Tam  

**Link**: [PDF](https://arxiv.org/pdf/2609.07735)  

**Abstract**: Large language models (LLMs) are rapidly becoming an interface between citizens and political information. They are often regarded as "a better Google." While this analogy might work for some instances, it is unintuitively problematic for democratic politics. A search engine retrieves human-authored documents, while a language model generates novel text that necessarily embeds invisible framing decisions. Because conveying knowledge involves framing, a system that generates answers cannot serve as a neutral conduit to "all human knowledge." Instead, these systems are becoming a new kind of political intermediary. Mechanistic evidence shows that partisan identity is encoded as a locatable geometric direction inside the Llama 3.1 8B model, and that alignment training masks rather than removes this structure. Building on that evidence, we present steering experiments that exploit a model's training cutoff in 2024. This cutpoint auspiciously falls just before a dramatic realignment in American politics marked by the second Trump administration and the MAHA transformation of health politics, providing us with a natural experiment. We find that the model presents temporally contingent partisan alignments as knowledge, with no mechanism for distinguishing fact from opinion. This reality moves the information environment beyond the echo chamber toward an epistemic monoculture where language models, purporting to summarize "all human knowledge" are, in actuality, simply magnifying the cultural and partisan divides inherent in their training data. 

---
# Translation Indeterminacy and the Distributional Fallacy 

**Authors**: Michael Carl  

**Link**: [PDF](https://arxiv.org/pdf/2609.07717)  

**Abstract**: Large language models (LLMs) are commonly associated with the distributional hypothesis, according to which (1) semantic meaning is grounded in distributional patterns of linguistic context, and (2) knowledge of cross-linguistic distributional correspondences allows for successful translation. This paper rejects the first claim as a causal inversion: linguistic distributions reflect patterns arising from meaning-making practices rather than constituting their source. At the same time, it accepts the second claim, arguing that translation -human or machine - can succeed without requiring access to meaning or reference. Knowledge of interlingual distributional correspondence and their inferential organization may be sufficient for translation. The paper develops an ecological-enactivist perspective, according to which reference and meaning are grounded in agent-environment interaction and stabilized through action-grounded concepts, forms of world-involving cognition that current LLMs do not possess. 

---
# DeepTable: Structural Attention Biases and Tree Path Encoding for Hierarchical Table Understanding 

**Authors**: Jyun-Ying Yen, Cheng-Kuan Lin, Yu-Chee Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2609.07707)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in table understanding. However, they typically process table content and headers as linearized token sequences. This representation weakens the two-dimensional and hierarchical structural relationships encoded by multi-level row and column headers. Existing parameter-efficient fine-tuning methods incorporate basic row and column information but do not explicitly capture the rich structural dependencies induced by hierarchical table headers. We propose DeepTable, a structure-aware approach for table understanding with LLMs. DeepTable comprises two complementary components. Structural Attention Bias (SAB) introduces learnable biases into the attention logits to explicitly represent whether pairs of table tokens share the same row or column. Tree Path Encoding (TPE) represents each table token using the ancestor paths of its row and column headers, preserving its position within the multi-level table structure. We integrate DeepTable with TableLoRA (He et al., 2025) to inject structural information into parameter-efficient adaptation. Across three LLM backbones, DeepTable consistently improves the corresponding TableLoRA baselines on three table question answering benchmarks, achieving average gains of 7.42 points on HiTab, 3.23 points on WikiTQ, and 2.01 BLEU points on FeTaQA. These results demonstrate the effectiveness of the proposed structural biases across different LLM backbones. 

---
# Fine PT-PT Web: A High-Quality 41 Billion Tokens Data Collection of the European Portuguese Web 

**Authors**: Gonçalo Vinagre, Rui Pedro Guerra, Pedro Gomes, Miguel Moura Ramos, Duarte Miguel Alves, Afonso Simplício, Diogo Tavares, David Semedo, Daniel Gomes, João Magalhães  

**Link**: [PDF](https://arxiv.org/pdf/2609.07699)  

**Abstract**: Curating Web corpora for regional language variants like European Portuguese (PT-PT) is heavily bottlenecked by dialectal overlap (mainly with PT-BR) and data processing scale. This paper presents an efficient pipeline to curate a production-ready PT-PT corpus from the Portuguese Web, spanning 411 TB of raw data from this http URL. We introduce a novel post-scraping block that removes boilerplate and line duplicates prior to filtering. This early-stage intervention increases final document yield by 19.04% by rescuing valid text that standard heuristic filters prematurely discard. Integrated with rigorous language identification, weighted fuzzy deduplication, and neural quality classification, our pipeline offers a scalable framework and a clean, representative corpus optimized for LLM pre-training. 

---
# Perspectives on Cross-Lingual Consistency in LLMs for Medical Questions 

**Authors**: Minh Duc Bui, Mario Sanz-Guerrero, Abteen Ebrahimi, Sagi Shaier, Peter Herbert Kann, Manuel Mager, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2609.07687)  

**Abstract**: Should multilingual LLMs answer medical questions consistently across input languages, or adapt responses to cultural cues? Existing multilingual medical benchmarks usually assume that medically correct answers should remain consistent across languages and treat cross-lingual variation as model error. In contrast, cultural adaptation research argues that appropriate medical answers may legitimately differ across contexts. We review the multilingual medical NLP literature through these two perspectives, we identify three gaps: limited stakeholder perspectives (e.g., of medical professionals), a lack of empirical evidence on which approach better serves users, and no benchmarks capable of distinguishing universally correct from culture-specific cases. To address the first gap, we survey 356 participants across three stakeholder groups (medical, NLP, and anthropology professionals) in three countries (Germany, Spain, and the United States). Anthropologists consistently favor adaptation, while medical and NLP respondents remain divided, with notable divergence between U.S. and European medical professionals. LLMs prompted with profession and country personas fail to reproduce this variation, overestimating cross-lingual consistency preference among NLP and medical personas. We conclude that neither consistency nor adaptation can currently be considered clearly preferable, highlighting the need for empirical evidence on which approach better serves users across cultural contexts. 

---
# How AI Models Manage Epistemic Authority: A Taxonomy and Comparative Analysis of Responses to User Disagreement 

**Authors**: Riyadh Alnasser, Yusuf Mücahit Çetinkaya, Sumin Zhao, Tuğrulcan Elmas  

**Link**: [PDF](https://arxiv.org/pdf/2609.07662)  

**Abstract**: Large language models are increasingly used as sources of advice and information, including in high-stakes settings, yet little is known about how they respond to user disagreement. We study how a model manages its epistemic authority, referring here to its claim to knowledge, competence, or the right to advise, once a user challenges its answer. Building on Conversation Analysis, we introduce a taxonomy of six challenge types and a four-layer framework for analysing each response: whether the original claim is maintained or changed, where authority is located, how the disagreement is socially managed, and what kind of evidential support is offered. We construct a new dataset of 2,310 controlled challenge scenarios and 32,340 corresponding responses from 14 models, and analyse them using our framework with an LLM-as-judge pipeline, providing a vocabulary which future evaluation and benchmark design can build on. We find that models show conflicting behaviour: they validate users in 85% of responses but maintain their original claim in 65%. They explicitly apologise in 33% of responses, yet 59% of those apologies accompany maintenance of the original claim. They transfer authority most often in advice tasks, doing so in 28% of responses and reaching 57% in health advice and 49% in legal advice, compared with 6% in fact and 3% in explanation tasks. Abandonment of the original claim ranges from 0.8% for GPT-5.2 to 40% for DeepSeek 7B, while complete replacement of the original claim is rare overall at 1.5%. 

---
# Syntactic Patterns and Stylistic Functions in Narrative Prose: A Rule-Based and Machine-Learning Approach 

**Authors**: Stefana Janicijevic  

**Link**: [PDF](https://arxiv.org/pdf/2609.07651)  

**Abstract**: This paper presents a small-scale quantitative experiment that links syntactic structure to stylistic functions in narrative prose. Starting from a dependency-parsed corpus of 3,300 sentences, we derive sentence-level stylistic labels across five categories --- descriptive, introspective, causal, ideological, and neutral --- using a transparent rule-based procedure that inspects lemmas, universal part-of-speech tags, and syntactic relations. For each sentence we construct a compact representation of its syntactic profile as a sequence of linearised triples combining lemma, POS tag, and dependency relation. These patterns serve as input to standard machine-learning classifiers trained to predict sentence-level style. The best-performing model achieves a macro-F1 of 0.948 under 10-fold cross-validation. The experiment is implemented entirely in Python using open-source tools. Our goal is not to propose a fully fledged stylistic theory, but to offer a reproducible and extensible workflow for exploring how grammatical structure contributes to narrative interpretation. 

---
# The Art of Hierarchical Competing Patterns: Gaussian Process Optimization of Hyphenation 

**Authors**: Ondřej Sojka, Petr Sojka  

**Link**: [PDF](https://arxiv.org/pdf/2609.07638)  

**Abstract**: Hyphenation patterns remain a compact and widely deployed solution for word breaking in typesetting systems, text processors, and web rendering engines, but their generation still depends on manually tuned patgen program parameter profiles. We formulate patgen profile selection as a black-box hyperparameter optimization problem and evaluate Gaussian-process Bayesian optimization for this task. The search objective combines a precision-oriented F_{1/7}-score with an explicit trie size-accuracy trade-off using a normalized trie-size penalty.
We evaluate the method on 17 hyphenated word-list datasets covering 14 languages and multiple scripts. Against two strong hand-tuned profiles regenerated from the same 8/10 training split and evaluated on the same 1/10 held-out test split, the GP-optimized profiles improve F_{1/7} on 16 of 17 datasets and reduce trie size on all 17. The median optimized/baseline trie ratio is 0.407. A dataset-level sign test gives p = 1.37e-4; a separate budget-matched comparison on five representative datasets shows that systematic search is competitive and usually improves over the best hand-tuned profile under the fixed comparison objective. The results show that model-based optimization can make pattern generation more reproducible and less dependent on expert trial-and-error while keeping the accuracy-compactness trade-off explicit. 

---
# ObGynLongBench: Revealing the Evidence-to-EHR Gap in Longitudinal EHR Decision-Making 

**Authors**: Jun Xiang, Zhijie Bao, Rong Hu, Kaizhou Qin, Wei Chen, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.07601)  

**Abstract**: The application of large language models (LLMs) to personalized medical assistants has garnered growing interest. However, existing medical benchmarks largely rely on static question answering with pre-selected evidence, leaving unclear whether LLMs can make reliable clinical decisions from real longitudinal electronic health records (EHRs). To bridge this gap, we introduce ObGynLongBench, a rule-grounded long-context EHR benchmark for obstetric and gynecologic decision-making, comprising 1,500 clinical decision-point cases from 976 real pregnancy EHR histories and traceable rules. Each case is anchored to a patient, a pregnancy-timeline point, and a pre-decision information boundary, enabling Evidence-only, Visit-level EHR, and History-level EHR evaluation. Evaluating 17 LLMs reveals a substantial Evidence-to-EHR Gap: models perform well when evidence is directly provided, but accuracy drops when evidence must be extracted from same-day records or full pre-decision EHR histories. Further analyses identify evidence utilization as a key bottleneck: performance decreases with longer EHR contexts and more complex evidence requirements, and earlier failures often predict later failures within the same patient history. Finally, active-search agents perform best among EHR access strategies, highlighting patient-specific evidence utilization as a central challenge for reliable personalized medical assistants. Resources are available at this https URL. 

---
# Validating DBpedia Triple Sets for Natural Language Generation 

**Authors**: Mark Andrade, Simon Mille, Anya Belz, Brian Davis  

**Link**: [PDF](https://arxiv.org/pdf/2609.07589)  

**Abstract**: We present a study of the quality of individual DBpedia triples from the perspective of Natural Language Generation, and propose and evaluate an approach for collecting entity-specific triple sets that filters out questionable triples while minimizing the loss of correct ones. We show in an evaluation against manually annotated data that with validation rules, it is possible to reach 98% precision in triple selection, and with improvements to a few Property definitions, it is possible to improve recall by 40% without harming precision. 

---
# We're Cooked! - Probing LLM Political Alignment Via Conflict-Framed Recipe Translation 

**Authors**: Svetlana Gorovaia, Angelica Henestrosa, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2609.07568)  

**Abstract**: Large language models (LLMs) are increasingly deployed for translation tasks, yet their implicit political positioning in such contexts remains understudied. We ask whether a single politically charged framing term, such as aggressor, enemy, neighbour, or coloniser is sufficient to trigger implicit political alignment in an otherwise apolitical task. We present a fully crossed factorial study in which eight models spanning Western, Chinese, and European origins are prompted to translate culturally attributed recipes into a target language left deliberately unspecified. Across 17 languages, four framing conditions, eight models, and 15,680 responses, we find that models do not simply decline or ask for clarification but resolve the ambiguity. Language resolution and reasoning behavior cluster meaningfully along model families: Western models hedge and deflect with vague justifications, Chinese models resolve conflicts silently, and Mistral Large emerges as a distinct profile combining high compliance with conflict-grounded reasoning. Sensitivity to framing terms is consistent across models: even subtle framing variation is sufficient to modulate behavior. Our findings urge caution when deploying LLMs for translation in conflict-adjacent contexts, where implicit political judgments may be made without any signal to the user. 

---
# Qwen-Audio-3.0-ASR Technical Report 

**Authors**: Chuanmeng Bian, Daren Chen, Peixin Chen, Zhigao Chen, Zhiyun Fan, Zhifu Gao, Bo Gong, Qing Gu, Jiajun He, Yawei Hu, Yunjie Ji, Jingbei Li, Xiangang Li, Xu Li, Zengxi Li, Zheng Li, Chengdong Liang, Baiji Liu, Ying Liu, Bin Ma, Yiping Peng, Yuezhang Peng, Zhendong Peng, Yu Pu, Yang Shi, Xin Shu, Jian Tang, Biao Tian, Peiyao Wang, Tianzi Wang, Wen Wang, Wupeng Wang, Cheng Wen, Yuzhong Wu, Zijian Xia, Yunchong Xiao, Nan Yang, Jianwei Yu, Jixing Yu, Binbin Zhang, Lei Zhang, Sitong Zhao, Guangdong Zhou, Yuan Zhou, Jianheng Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2609.07549)  

**Abstract**: In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model scaling, and deep integration with large language models (LLMs). However, bridging the gap between academic benchmark performance and real-world production utility remains a persistent challenge, particularly in handling diverse regional dialects, dynamic entities and hotwords, long-range contextual information, and disfluent spontaneous speech. In this report, we present Qwen-Audio-3.0-ASR, a Mixture-of-Experts (MoE) LLM-based ASR system designed to address these production demands through a unified, instruction-following framework. The model is built upon the Qwen backbone, and is trained on tens of millions of hours of large-scale speech data. Qwen-Audio-3.0-ASR supports transcription across 30 languages and 16 Chinese dialectal varieties spanning eight major dialect regions. Beyond multilingual and dialectal recognition, the model provides production-oriented capabilities including industry-domain entity recognition, hierarchical hotword customization, native single-pass transcription polishing, and long-audio contextual modeling. We further develop a dedicated streaming variant, Qwen-Audio-3.0-ASR-Streaming, for latency-sensitive applications. Extensive evaluations on Chinese, English, multilingual, and real-world industrial test sets demonstrate state-of-the-art or highly competitive recognition performance across a broad range of evaluation conditions, with strong performance relative to leading commercial and proprietary systems including GPT-4o Transcribe and Gemini 3.1 Pro. 

---
# Where Should Language Sit in a Multimodal Model? Lessons from What Language Does to Human Perception and Cognition 

**Authors**: Peng Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.07474)  

**Abstract**: Language models compute over tokens: language is their input, their output, and increasingly their internal representation. Whether language should keep all of these positions depends on what language does to the system that uses it. The one system with a century of data on that question is the human. We review what language does to human perception, the brain, and thought, and read the same evidence against multimodal models and language models. Throughout, we treat language as a compressor that runs on a shared codebook: a word is an index, the content is in the receiver, and a community maintains the codebook. In humans the compression is measurable, learning the codebook reorganizes the senses, and thought survives the loss of language. We then measure the rule that models apply when two cues disagree, with cue-conflict experiments on six vision-language models and two robot policies. Surviving cues are weighted in the order their reliabilities prescribe, at 11 to 82\% of the ideal observer's slope, and many answers copy the text. One policy family drops a cue that adds no information beyond the others rather than down-weighting it, another keeps it at a weight that fails when the cues conflict, and a visual cue that identifies the task in every training frame is never learned, because the language pathway already fits the data. Language models are the best current models of the human language network, and they have entered the human speech community, shifting word frequencies while alignment narrows their conceptual diversity. We close with seven implications for token-based systems. Language belongs at a model's boundary and in the shared codebook, as in the brain, not as its internal representation; the price of leaving the codebook inside is auditability. 

---
# MEMO: Multimodal Evidence Memory Organization for Long-Horizon LLM Agents 

**Authors**: Xian Gao, Jinpeng Wang, Jiacheng Ruan, Guangyu Cao, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2609.07471)  

**Abstract**: Long-running LLM agents rely on external memory to store and reuse information beyond a single context window, yet there is a fundamental tension between the continuous accumulation of interaction trajectories and the limited context capacity. The key challenge in agent memory is therefore not only to retrieve relevant records, but also to select necessary evidence under a given budget and organize it in an appropriate modality. Existing memory readout methods mainly use textual or visual forms. Text preserves high fidelity, but its linear token representation makes contents with different importance compete for the limited context at nearly uniform unit cost. Visual readout renders text into document-like images, which can use two-dimensional layouts to expose structure and emphasize key information, but it may lose fine-grained details during rendering and compression. To address this issue, we propose MEMO, a multimodal evidence memory organization method for LLM agents. MEMO first uses a trained evidence extractor to select relevant memory blocks and form evidence units with source information and presentation requirements. A trained query-conditioned memory manager assigns each unit to a textual, visual, or dual-channel carrier and selects a layout that matches the evidence structure. A deterministic memory construction module then generates the textual package and visual pages. The memory manager is trained with feedback from an offline reader that measures the utility of the guided memory plan, so that retention and presentation decisions align with downstream usage. We evaluate MEMO on four benchmarks, HotpotQA, 2WikiMultiHopQA, LoCoMo, and ALFWorld, with multiple reader backends. The results show that MEMO presents memory more efficiently with fewer memory tokens, improves downstream task performance, and builds more effective working memory under constrained budgets. 

---
# FramingQA: Does the Question Shape the Answer? Measuring the Compositional Framing Effect 

**Authors**: Hazel H. Kim, Andrew M. Bean, Guilherme Affonso Ferreira de Camargo, Shanyu Chauhan, Felix Drinkall, Jade Kosché, Chenyang Ma, Glory Nwaugbala, Nabeel Seedat, Bradley Max Segal, Samuel Recht, Hinrich Schütze, Philip H.S. Torr  

**Link**: [PDF](https://arxiv.org/pdf/2609.07448)  

**Abstract**: We introduce FramingQA, a benchmark that measures the model sensitivity to question framing across law, medicine, finance, and robotic simulations. Large language models (LLMs) often change their responses to subtle rephrasings that align with an implied stance by users. This can leave users with advice tainted by how they happened to phrase a question rather than by the underlying facts, and the consequences are highly costly in high-stakes domains. Because in the realistic scenarios, both expert practitioners and non-expert users frequently ask LLMs questions containing incomplete or misleading assumptions, models are highly susceptible to those framings. To test this, we inject the framing bias across three nested levels: a framing-biased question phrasing (root), an injected framing-biased premise prepended to a neutral question (propositional), and a premise paired with a framing-biased question (global). Evaluating nine open models (3.8B-70B) across four families, we find that strong per-variant accuracy does not guarantee the robustness across differently phrased questions under the fixed factual information. 

---
# An LLM-Associated Register Shift in Korean Journal Abstracts: A Morphology-Aware Excess-Vocabulary Study, 2018-2026 

**Authors**: Aron Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.07447)  

**Abstract**: Excess vocabulary, a word's frequency above its pre-2023 trend, is how the change in scholarly English after 2022 has been measured. We adapt it to Korean with morphological units on 398,296 KCI abstracts (2018-August 2026), with 47,165 Vietnamese abstracts for comparison. Placebo floors are 0.1-2.2 points for the single-word statistic and at most 2.9 for the re-selected split-half set statistic. Korean abstracts show nothing in 2023, onset in late 2024, a rise through 2025 flattening in mid-2026: sisahada "suggest" appears in 21.4% of 2026 abstracts against 5.3% expected; plain verbs like araboda "look into" fall to a quarter of trend. Under stated assumptions the single-word conditional lower bound on LLM-processed abstracts is 3.5%, 10.5% and 16.1% for 2024-2026 and a split-half set bound 7.8%, 20.6% and 33.0%. Holzwarth et al.'s estimator under the same discipline gives 41.9% and 72.1% for 2025-2026. Subject-matter controls reduce but do not remove it: restricting the set to lemmas three language-model annotators all call style leaves 14.7 of the 33.0 points, and pairing each 2026 abstract with its journal's closest base-period abstract leaves 34.1. Tested translation routes do not explain it: the surface marks of translated Korean fall as the markers rise. In the same articles' English abstracts the excess appears a year earlier; where the English side carries none, the Korean shift persists at 30 to 66% of the rate where it does. Control abstracts from three providers reproduce the rising words, with marker turnover consistent with model generations; implied prevalences are scenario-dependent. 

---
# Beyond Single-Negative Preference: Multi-Negative DPO for LLM-Centric Historical Entity Linking 

**Authors**: Tien Nam Nguyen, Emanuela Boros, Ahmed Hamdi, Adam Jatowt, Mickaël Coustaty, Antoine Doucet  

**Link**: [PDF](https://arxiv.org/pdf/2609.07379)  

**Abstract**: Large language models (LLMs) have recently shown promise for historical entity linking, but preference optimization for this task is often formulated with only one negative candidate per training instance. This discards information from the remaining candidates retrieved for the same mention. We introduce multi-negative direct preference optimisation (MDPO), a reference-based pairwise objective that compares the correct entity with all valid rejected candidates associated with each mention. MDPO preserves the Bradley-Terry formulation of DPO while exploiting the complete candidate set through masked, length-normalised sequence scores. We evaluate MDPO on hipe-2020 and newseye, covering French, German, English, Swedish, and Finnish historical newspaper text. Experiments show that MDPO improves over supervised fine-tuning and single-negative DPO, with particularly strong gains for NIL mentions, semantic ambiguity, OCR noise, and historically difficult names. Further analyses disentangle candidate-generation and selection errors, showing that candidate retrieval remains a key bottleneck for end-to-end entity linking. These results demonstrate that incorporating all within-instance negative candidates is a simple and effective improvement for LLM-based historical entity linking. 

---
# Beyond Fluent Generation: A CPU Reliability Benchmark for MCP-Style Tool Calling in Sub-2B Small Language Models for Edge Deployment 

**Authors**: Abrar Shahriar Qurat-Ul-Ain Mastoi  

**Link**: [PDF](https://arxiv.org/pdf/2609.07370)  

**Abstract**: Resource constrained single-board computers including Raspberry Pi, NVIDIA Jetson Nano, Arduino UNO Q, Orange Pi, and LattePanda motivate on-device small language model (SLM) agents that reduce cloud dependence, improve data locality, and tolerate intermittent connectivity. Model Context Protocol (MCP)-style tool invocation demands more than fluent generation: an agent must emit machine-readable JSON, select the correct tool, supply all required arguments, and avoid unintended actions. We establish a platform-agnostic CPU baseline by evaluating five open-weight models below two billion parameters Phi-1.5, Pythia-1.4B, TinyLlama-1.1B-Chat, Qwen2.5-0.5B, and Qwen2.5-1.5B on 100 prompts spanning weather retrieval, web search, calculation, email composition, and task creation, under greedy decoding and nucleus sampling. A recovery parser strips Markdown fences, extracts brace-delimited substrings, and scores parseability, tool-name correctness, argument completeness, and value agreement. Under this criterion, Qwen2.5-1.5B achieves 75% (greedy) and 79% (sampling); Qwen2.5-0.5B achieves 72% (greedy) but drops to 32% under sampling. Phi-1.5 scores 0%; Pythia and TinyLlama reach at most 7%. A strict post-hoc audit finds only 5 of 1,000 raw responses directly parseable as JSON, exposing near-total dependence on output recovery. A CPU resource probe shows Qwen2.5-1.5B requires 7,960 MiB and 30.782 s mean latency; Qwen2.5-0.5B uses 3,637 MiB and 10.627 s, revealing a reliability-resource trade-off for edge deployment. These results do not cover the named boards directly or a full MCP implementation. Safe deployment requires schema validation, constrained generation, least-privilege execution, and human escalation for consequential actions. 

---
# BlueprintAgent: Constraint-Triggered Targeted Revisits for Simulation-Ready Generation from Scanned Structural Blueprints 

**Authors**: Zhouyuan Xu, Chen Yang, Linhao Wang, Jiansheng Fan, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07362)  

**Abstract**: Converting in-service reinforced-concrete (RC) building blueprints into simulation-ready models---structured frame representations that support deterministic FEM export and qualified-engineer review---underpins safety assessment and seismic retrofit, but the process remains manual. Direct prompting of a multimodal large language model (MLLM) over a scanned sheet is unreliable: outputs often violate engineering constraints on beam--column support, span count, or 3D continuity. We present BlueprintAgent (BPA), a constraint-triggered multimodal agent for simulation-ready frame extraction from scanned blueprints. BPA treats the MLLM as the primary reader and decision maker, with OCR and computer vision supplying localized evidence. Its central mechanism realizes engineering constraints as callable validators whose entity-level conflict reports trigger targeted MLLM revisits over the local region---an inference-time control distinct from fixed pipelines and free-form self-reflection. We evaluate BPA on 300 real scanned blueprint sheets from 20 anonymized RC frame projects, against five baselines and six ablations. BPA reaches a macro-averaged Beam F1 of 0.994, against 0.301 for single-MLLM zero-shot and 0.820 for a fixed pipeline; removing MLLM-led axis adjudication collapses Beam and Column F1 on complex multi-sheet projects. For dense technical drawings, engineering constraints are best deployed as triggers for entity-level targeted revisits rather than as post-hoc output filters. 

---
# Content-Based Addressing for Long Context 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2609.07314)  

**Abstract**: Rotary position embedding (RoPE) uses each token's integer position to determine the rotation applied inside attention. This works well for local token order, but increasing context length creates a positional train-test mismatch: RoPE produces relative rotations at offsets not seen during training. Methods that rescale, interpolate, randomize, or bias positions specify how attention handles those offsets, but still derive positional information from a growing token counter. We instead divide a token stream into units, retain ordinary RoPE positions within each unit, and assign every completed unit an address computed from its content. Adding units then applies the same learned map to new content rather than extending a positional range or an identifier table. We prove that this construction preserves local RoPE exactly, leaves the attention comparison between two fixed tokens unchanged when other units are inserted or reordered, and does not create new relative rotations merely because more units are added. In a character-level Tiny Shakespeare diagnostic, a model trained on 256-character contexts has validation perplexity 4.04 at 256 characters and 3.82 at 4096, while continuous RoPE changes from 4.71 to 12.09. A second diagnostic shows that content-based addressing can retrieve and use information from multiple serialized facts. These are controlled shallow experiments, not scale benchmarks, but they support a direct prescription: use position to address locally and content to address across units. 

---
# LANTERN: Language Model Assessment on Noisy and Transformed Tasks for Understanding Error and Robustness Nuances 

**Authors**: Vamsi Krishna Kodavali, Rituraj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.07309)  

**Abstract**: Robustness evaluation of large language models (LLMs) remains a critical challenge, particularly in assessing their sensitivity to perturbations in input data. In this work, we systematically evaluate LLM robustness across multiple dimensions, including word error rate, character repetition and duplication, modifications in choices, and variability in instruction following. To facilitate this evaluation, we construct a synthetic and augmented dataset encompassing a diverse set of LLM benchmarks, specifically targeting multiple-choice question (MCQ) datasets and instruction-following tasks. We conduct extensive experiments on LLMs of varying scales-small, medium, and large-as well as across base and instruction-tuned variants. Our analysis quantifies the variability in model responses under perturbed conditions and highlights discrepancies relative to baseline models. The findings provide insights into the stability of LLMs across different evaluation scenarios contributing to the development of more robust and reliable language models as well as robust evaluation methodologies. 

---
# SPARROW: Scalable Taxonomy Induction via Structure-Preserving Partitioning and Constraint-Guided Merging 

**Authors**: Yirui Zhang, Yixuan Tang, Yandong Sun, Mong-Li Lee, Anthony Kum Hoe Tung  

**Link**: [PDF](https://arxiv.org/pdf/2609.07307)  

**Abstract**: Taxonomy induction aims to organize concept sets into coherent hierarchical structures. Recent LLM-based methods can induce taxonomies directly from flat term lists, avoiding the need for corpora, but degrade sharply as concept sets scale up. We argue that this degradation stems not only from context length limitations, but also from structural failures in hierarchical reasoning. To address this, we adopt a divide-and-merge paradigm that partitions concepts into smaller subsets, induces local taxonomies, and merges them into a global hierarchy. However, we identify two structural failure modes inherent to this paradigm: Structural Fragmentation, where partitioning weakens local hierarchical signals, and Parent Displacement, where locally plausible relations are misplaced in the global hierarchy. To address both, we propose SPARROW, a scalable taxonomy induction framework that combines structure-preserving spectral partitioning to retain hierarchical connectivity within each block, and constraint-guided incremental fusion that treats block-level relations as structural constraints rather than ground truth for global placement. Experiments on large-scale benchmarks show that SPARROW consistently achieves the strongest global structural quality across backbones. The code is available at this https URL. 

---
# RouteRelay: Event-Triggered Cross-Layer Route Reuse for Efficient Dynamic Sparse Attention 

**Authors**: Bin Li, Sisi Liu, Chenyang Hu, Chaoyang Zhang, Wei Li, Hui Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.07306)  

**Abstract**: Dynamic sparse attention reduces long-context prefill cost by routing each query chunk to a small set of key chunks at every Transformer layer. The sparse attention kernel avoids most token interactions, but the router still rebuilds a chunk--chunk score matrix layer after layer, even when the selected routes change little. We introduce RouteRelay, a router-agnostic method that reuses only route metadata across depth while continuing to compute attention with the current layer's queries, keys, and values. Anchor layers perform full routing. Intermediate layers rescore the previous top-$k$ route and a compact sentinel set of near-miss and randomly probed chunks. A query row is rerouted only when a sentinel challenges its weakest selected chunk. We give a top-$k$ stability condition, a probabilistic bound on missed challengers, and a row-selective GPU execution design. In a reproducible empirical evaluation, RouteRelay retains at least 99.99% route recall while rerouting 25.0%, 55.4%, and 78.2% of rows under low, moderate, and high cross-layer drift, respectively. Across routing scales, RouteRelay retains 100.0% recall while evaluating 38.4--51.6% of full-routing score pairs as the key-chunk count grows from 128 to 1024. Its unfused CPU execution remains slower than dense matrix multiplication, exposing row compaction and ledger updates as the main kernel-engineering targets. 

---
# Marginal Fidelity Does Not Establish User Simulation in Demographic Synthetic Survey Panels: Response Contracts, Support Collapse and Conditioning Failure 

**Authors**: Alexander Doudkin  

**Link**: [PDF](https://arxiv.org/pdf/2609.07305)  

**Abstract**: Demographic synthetic survey panels are often validated by matching aggregate answers to published surveys. We test what that certificate establishes across six multiselect batteries from four survey organisations in three countries. The headline analysis is restricted to three instruments whose synthetic cohort and human target share the stated population frame; three other batteries remain sensitivity analyses.
The response contract dominates measured fidelity. In the aligned instruments, committed sets leave 66 of 128 model-battery option slots empty in panels of up to 500 respondents, versus 0 of 128 under per-option probability elicitation. Across eight uncapped model-instrument comparisons, probabilities reduce option-marginal MAE by 4.53 to 7.30 points. The capped instrument reverses on two models until the vectors are projected onto its stated maximum. These are measurement effects: human targets are realised check-all responses, whereas the vectors are latent inclusion propensities.
Published marginal agreement also fails to discriminate respondent simulation from direct population estimation. On nine aligned model-battery pairs, a no-persona population-prevalence query averages 6.27 MAE versus 12.39 for committed panels and wins all nine comparisons. Constraint-aware probability vectors average 5.34 and beat the query on four of nine, so the baseline challenges the validation criterion rather than proving direct estimation uniformly best. On three unpublished demographic cells, neither approach beats reciting the national distribution. Population-marginal agreement is therefore evidence about an elicitation contract and an estimand obtainable without simulated respondents, not evidence of individual simulation. 

---
# Probing the Structure and Dynamics of LLM Value Expression through Value Conflicts 

**Authors**: Kaicheng Zhang, Jingyi Xiao, Renjun Hu, Xiaoling Liu, Yunshi Lan, Xuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.07296)  

**Abstract**: Ethical evaluation of Large Language Models (LLMs) often characterizes model values as static and monolithic. In contrast, we argue that LLM value expression is better understood as a structured yet dynamic phenomenon. To investigate this, we introduce Conflict-driven Value Probing, a controlled framework that places LLMs in value conflicts and implements four types of interventions that perturb these conflicts to probe LLM value expression. Applying this framework to ten LLMs, we identify three recurring patterns. (1) Expression duality: models shift from broad idealistic orientations in abstract assessment toward more pragmatic priorities in concrete conflicts. (2) Functional steerability: models readily reconfigure their expressed value profiles toward task-defined value objectives. (3) Bounded plasticity: such reconfiguration is not without constraints, i.e. pressure induces a security- and goal-oriented priority shift while negative framing distinguishes protected values from those more amenable to redirection. Together, these findings characterize both the structure and dynamics of LLM value expression: context flexibly reconfigures expressed priorities, yet within behavioral boundaries. This behavioral account provides a foundation for understanding controllability, alignment, and safety in LLMs. Code and data are available at this https URL. 

---
# Separating Stream Stability from Long-Term Recall in Language Models 

**Authors**: Peipei Cao, Xin Zhang, Jie Tang, Xiao Li, Siying Li, Qing Pei  

**Link**: [PDF](https://arxiv.org/pdf/2609.07282)  

**Abstract**: Methods for streaming language models are often discussed alongside long-context and memory systems, although they solve different problems. An attention sink can stabilize autoregressive generation over an indefinitely long stream while the model remains unable to use content that has left its recent-token cache. We argue that this distinction should be explicit in system claims and evaluation. We introduce three horizons: the stability horizon, over which predictive behavior remains well behaved; the access horizon, over which past content can still causally affect the output; and the utility horizon, over which a task retains acceptable performance. We show constructively that the stability horizon can be infinite while the access and utility horizons are finite. We then propose ThreeH, an evaluation contract that measures all three horizons under a common state and compute budget. Applying the framework to attention-sink streaming clarifies its strength, constant-memory, stable generation, without treating anchor tokens as semantic memory. The framework exposes roles for cache policies, recurrent state, retrieval, and external memory. Experiments on 128K-token streams, delayed binding recall, and delayed decisions show that attention sinks preserve local modeling but not content beyond the active cache; recurrent and retrieval state extend the semantic horizon. 

---
# CEDAR: Error-Bounded Residual Routing for Efficient Long-Context Attention 

**Authors**: Siyu Li, Dong Wang, Jie Zhou, Wei Li, Yang Xu, Sijie Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.07237)  

**Abstract**: Post-hoc sparse attention accelerates long-context prefill by routing each query to a small set of token-level interactions. Hard selection, however, assigns zero probability to every omitted chunk: a routing miss cannot be recovered, and a fixed expansion budget spends the same work on easy and ambiguous queries. We introduce Coarse-to-fine Error-aware Dynamic Attention Routing (CEDAR), a coarse-to-fine method that keeps the language model frozen while preserving global coverage. Each semantic chunk contributes a cheap key--value summary to a residual attention path; chunks with high estimated approximation error are then expanded to exact token attention. Exact and summarized contributions are combined in a single softmax normalization, so refinement replaces, rather than duplicates, coarse evidence. We derive an output-error bound governed by within-chunk key/value dispersion and use it to allocate a variable refinement budget. A controlled clustered-attention study shows that residual summaries reduce reconstruction error by more than 98% relative to hard dropping at equal exact-chunk budgets. Experiments on long-context benchmarks demonstrate that CEDAR recovers most of the quality lost by hard sparse routing while maintaining approximately $3\times$ kernel speedup at 128K context. 

---
# SIFTING: A Novel LLM-Based Framework for Structured and Transparent Information Extraction from Clinical Free-Text Reports, with Application to Tumor Staging in Lung Cancer 

**Authors**: Mirco Hess, Gerben van Veenendaal, Joris Wakkie, Yiwen Soo, Malcolm H. Lawson, John D. Maclay, Arjun Nair, Neal Navani  

**Link**: [PDF](https://arxiv.org/pdf/2609.07185)  

**Abstract**: Background: Large language models (LLMs) show promise for extracting information from clinical free-text documents, but their outputs are often unstructured and lack traceability, complicating validation and adoption in clinical workflows. In this work we introduce SIFTING, an LLM-based framework designed to address these shortcomings.
Methods: SIFTING combines the language comprehension capabilities of LLMs with segment-level processing and structured prompts with strict output control, linking findings to the source text to enable both accurate and transparent information extraction. To demonstrate its capabilities, we applied the framework to the task of extracting tumor T-stage information from 130 lung cancer radiology reports (SIFTING-T-stage). A compact 4-bit quantized version of the open-source LLM Llama-3.3-70B (35 GB) was used in a fully self-hosted setup, providing full control over data and model. Performance was evaluated against a reference standard created by four clinical experts and compared with a range of LLMs as used in a conventional single-prompt approach, using bootstrap resampling to estimate confidence intervals.
Results: SIFTING-T-stage achieved an accuracy of 90% (95% CI: 84-95) against the reference standard. We found its performance to be comparable to even the largest state-of-the-art LLMs with reasoning capabilities and to be interchangeable with clinical experts (p < 0.001), while at the same time offering full traceability through source text references.
Conclusion: SIFTING enables accurate, structured, and traceable information extraction from clinical free-text documents. It ensures data control, reproducibility, and verifiable outputs that can support clinical validation and workflow integration. 

---
# CircuitLens: Reasoning Circuits as Data Selection Signals for Reinforcement Learning with Verifiable Rewards 

**Authors**: Zhuofan Chen, Ziqian Jiao, Yikai Cui, Zhixin Cai, Jun Bai, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2609.07183)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is sensitive to which problems a model trains on, yet existing selection criteria--difficulty filtering, hand-curation, reward-trajectory scoring--assess data value as an intrinsic property of problems, independent of the model that will learn from them. We introduce Circuit Reasoning Score (CRS), a selection signal derived from 46 reasoning-sensitive attention heads identified via contrastive ablation, computed in a single forward pass on the frozen base model without reward labels or rollouts. CRS runs against the intuitive hypothesis that stronger reasoning-circuit engagement produces better training data: on Qwen2.5-Math-7B, the lowest-engagement decile improves over random selection on three medium-difficulty benchmarks (GSM8K +2.0 pp, OlympiadBench +1.6 pp, Minerva +2.9 pp), while the highest-engagement decile gains less and is indistinguishable from the middle decile. The advantage has boundary conditions: on a domain-curated pool no selection method separates from the others; at 1.5B scale the useful direction differs; and the lowest-reward training condition produces the strongest downstream generalization. Within the Qwen2.5-Math settings tested, RLVR data selection appears regime-dependent rather than reducible to a static ranking of problem quality. 

---
# In-Place Instruction Following in Diffusion Language Models 

**Authors**: Zheng Nie, Zherui Li, Jiaming Zhang, Kun Wang, Zhenhong Zhou, Yufei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.07160)  

**Abstract**: Diffusion Large Language Models (dLLMs) generate text via bidirectional iterative denoising, naturally supporting user-specified constraints anchored at arbitrary output positions, a paradigm known as In-place Prompting (IPP). We formalize this as the In-place Instruction Following (IIF) task and construct IIF-Bench, a hierarchical benchmark spanning literal, style, and discourse-function constraints, paired with a rubric-based local-global evaluation protocol. An inference-time attention-bias probe suggests that vanilla dLLMs often under-prioritize constraint spans during denoising. We then propose GRAFT, an IPP-oriented post-training framework combining constraint-aware SFT and preference optimization. On four representative dLLMs, GRAFT raises the average IIF score from 57.75 to 73.10 (+15.35 points), with absolute gains of 15.91 and 15.57 points on literal and discourse-function constraints, while preserving general generation ability. 

---
# FreqBLiMP: Frequency-Controlled Minimal Pairs Reveal Robustness and Fragility of LLMs Under Lexical Rarity 

**Authors**: Tyrone White, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2609.07153)  

**Abstract**: Minimal-pair benchmarks such as BLiMP evaluate linguistic knowledge by testing whether language models (LMs) prefer acceptable sentences over minimally different unacceptable ones. However, these benchmarks largely ignore lexical frequency variation, despite lexical frequency being a pervasive and highly skewed property of natural language use. Consequently, existing evaluations do not test whether grammatical preferences remain stable when contrasts involve rare lexical items. We introduce FreqBLiMP, a frequency-controlled extension of BLiMP that regenerates all 67 paradigms under explicit Zipf-frequency regimes while preserving each minimal-pair's grammatical contrast. Evaluating multiple open-weight LLM families across scales, we find that decreasing lexical frequency produces a consistent, monotonic decrease in sentence likelihood, but only a modest reduction in overall contrastive acceptability accuracy. However, this aggregate stability masks substantial variation across linguistic phenomena, with LLMs remaining robust on overt morphosyntactic generalization while degrading on phenomena that require lemma-specific information. 

---
# Vishing-Tactics-Bench: Forecasting Exploitation Trajectories in Voice Phishing Calls 

**Authors**: Jeongmin Lee, Dongmyung Sul, Seung Yun, Jinxia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07151)  

**Abstract**: Voice phishing (vishing) unfolds in real time; by the time a call has ended and post-hoc classification is possible, the harm has already been done. The more actionable question is which concrete harm (Information Gathering or Financial Exploitation) an ongoing call is tactically progressing toward. We present Vishing-Tactics-Bench, a benchmark grounded in Endsley's situation-awareness (SA) framework that recasts vishing defense from after-the-fact fraud classification to harm projection: predicting at each turn whether the call will reach either terminal harm. We adapt MITRE ATT&CK to vishing as a 6-tactic taxonomy (Vishing-Tactics) and label 35,340 scammer utterances across 5,645 synthetic Chinese calls. We define Exploitation Trajectory Forecasting, a survival-style protocol over the two terminal harms with three metrics: AP@k, C-index, and divergence error. Baselines ranging from a Markov heuristic to fine-tuned LLMs show that the tactical trajectory serves as an interpretable representation of the call's tactical state, supporting harm-specific forecasting, which can then be used for the downstream application of intervention selection; a stratified lead-time analysis at a tight false-alarm budget further identifies at what point in a call the trajectory signal yields early warning. 

---
# Retrieval-Augmented Multi-Prompt Ensemble for Minor-Grain Breeding Information Extraction 

**Authors**: Hang Zhao, Jiahao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07134)  

**Abstract**: This paper presents our system for CCL2026-Eval Task 5: Minor-Grain Breeding Information Extraction (MGBIE), which jointly extracts 12 entity types and 6 relation types from minor-grain breeding literature. We propose RAME (Retrieval-Augmented Multi-Prompt Ensemble), a training-free framework that elicits multiple LLM outputs under controlled diversity and aggregates them by majority voting to obtain high-confidence predictions. RAME combines (i) retrieval-augmented few-shot selection via a hybrid BM25-embedding retriever, (ii) a three-prompt ensemble (Strict, Relaxed, Balanced) spanning the precision to recall spectrum, and (iii) large-scale repeated sampling with majority voting to filter noisy predictions. Built on DeepSeek-V4-Flash, RAME achieves a Total Score of 0.499 (NER 0.730, RE 0.346) on the leaderboard, ranking 1st and surpassing the official Track-A baseline powered by GPT-5.5 (0.448), representing an 11.4% relative improvement. Code is available at this https URL. 

---
# Line-Coupled Language Model 

**Authors**: Shiyuan Li, Shaorong Zhang, Zhaorui Yang, Qian Zhang, Greg Ver Steeg, Bingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.07129)  

**Abstract**: Autoregressive language models generate one token per decoding step, limiting the useful output of each forward pass. Although diffusion models, insertion-based decoding, and multi-token prediction enable parallel generation, they either incur additional training-time token traffic or struggle to predict strongly dependent future tokens. We introduce the Line-Coupled Language Model (LCLM), an autoregressive model that advances multiple text lines together by predicting the next token for every active line while coupling the lines through shared causal context. LCLM interleaves line tokens into a single causal sequence and uses line-staggered rotary positions, retaining the standard next-token objective and causal attention. Controlled experiments show that cross-line targets are substantially less dependent than consecutive same-line targets, supporting lines as parallel generation units. With 881M parameters, LCLM produces an average of 2.94 content tokens per forward pass with a validation cross-entropy loss of 2.44, compared with 1.00 token per forward pass and a loss of 2.39 for the vanilla autoregressive baseline. Most notably, even when LCLM generates 16 tokens per forward pass, its loss is only 0.09 higher than that of the vanilla autoregressive baseline (2.34 vs. 2.25). 

---
# PTCG: Persona-guided Tree-based Counterargument Generation 

**Authors**: Eunbeen Son, Yohan Jo, Joonsuk Park, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2609.07120)  

**Abstract**: The ability to generate counterarguments is important for critical thinking and balanced discourse, yet existing approaches typically produce only a single counterargument, failing to capture the diversity and persuasiveness required in real-world debates. To address this limitation, we propose Persona-guided Tree-based Counterargument Generation (PTCG), a framework that combines Tree-of-Thoughts-inspired step-wise generation and pruning with speaker persona selection. By estimating the author's persona from the original argument and incorporating speaker personas representing distinct perspectives, PTCG operationalizes perspective-taking and enables the generation of diverse counterarguments. Results from LLM-as-a-Judge, classifier-based assessment, and human evaluations indicate that PTCG shows consistent improvements in both the diversity and persuasiveness of counterarguments compared to baseline methods. 

---
# The Illusion of Debiasing: Persona Steering Redistributes Rather Than Reduces Bias in LLMs 

**Authors**: Ziyue Feng, Hongbo Fang, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2609.07117)  

**Abstract**: Prompt-based interventions: system prompts, personas, role instructions, reliably reshape what a language model says, but it is unclear which layer they reach. Do they reconfigure internal structure, or only modulate the output channel? We use persona conditioning as a controlled probe, measuring its effects along a depth axis from self-report, through open-ended generation, to word-level parametric association, across three instruction-tuned models. We find a graded dissociation. Personas are legible but not structural: models follow single-trait instructions yet fail to reproduce human inter-trait covariance. The dissociation deepens with depth: personas hold or amplify closed-form QA bias, shift absolute tone while leaving between-group disparity unchanged, and barely perturb an already saturated associative baseline. Prompt-based steering thus operates in the output channel and has a structural reach limit that surface manipulability can mask. 

---
# Revisiting Complete Reasoning Traces for Post-Training 

**Authors**: Jaehui Hwang, Sangdoo Yun, Byeongho Heo, Dongyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.07103)  

**Abstract**: Large language models (LLMs) are often post-trained on pre-collected reasoning trajectories to improve their reasoning capability. Such trajectories tend to be long due to complex, interwoven paths, which often include detours on the path toward the answer. However, it has been underexplored whether LLMs indeed benefit from learning complete trajectories in post-training, such as supervised fine-tuning (SFT). Starting from our pilot study, we find that full trajectories provide only limited benefit, while partial trajectories are effective even under heavy truncation. We analyze redundancy in reasoning trajectories through attention-based analyses and controlled token-removal studies, both of which show that intermediate tokens contribute minimally to final reasoning quality. This suggests that avoiding redundant information may allow LLMs to internally infer coherent alternatives by inferring missing steps from their internal knowledge, given known trajectory endpoints. Furthermore, we show that training LLMs using endpoints leads to consistent changes in reasoning behavior, and that it also benefits post-training methods based on reinforcement learning or on-policy distillation, highlighting the need to revisit complete reasoning traces. Code is available at this https URL. 

---
# Where to Look and What to Use: Retrieve-Localize-Generate for Long-Term Conversational Memory Question Answering 

**Authors**: Yifan Wang, Xinkui Lin, Yongxiu Xu, Shen Gao, Ruochen Yang, Kun Huang, Yubin Wang, Jie Wu, Wei Liu, Jian Luan, Hongbo Xu, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07093)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to answer questions by accessing external knowledge and has been widely adopted for long-term conversational memory question answering. However, existing methods suffer from two key challenges: (1) fragmented evidence scattered across temporally distant sessions, and (2) noisy content within retrieved sessions that triggers the lost-in-the-middle effect. To address these challenges, we propose MemLoc, a unified Retrieve-Localize-Generate framework for long-term conversational memory QA. For retrieval, MemLoc decomposes each session into multi-granularity memory units and performs query routing via an inner-memory graph with entropy-based granularity selection. It further models cross-session semantic and temporal dependencies through a cross-memory graph, enabling coarse-to-fine retrieval of top-K relevant memory candidates. For localization, we introduce a reasoning-based evidence locator trained with Self-reflective Hint Policy Optimization (SHPO), which performs progressive refinement by extracting query-relevant fragments within memory units to suppress noise and reranking across candidates to remove redundancy, producing a compact evidence set with lightweight location IDs. For generation, these IDs act as precise grounding signals that guide the LLM to the correct memory positions, mitigating the lost-in-the-middle effect while preserving original contextual integrity. Extensive experiments on four benchmarks demonstrate that MemLoc achieves state-of-the-art retrieval accuracy and response quality while maintaining efficiency. Our code is available at: this https URL. 

---
# A Hyperbolicity Atlas of Large Language Model Hidden States 

**Authors**: Zhichao Yang, Yuanze Hu, Gen Li, Qingchen Yu, Shiying Duan, Xinyu Wang, Ye Qiu, Zeming Liu, Guangxu Chen, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2609.07053)  

**Abstract**: LLM hidden states are ordinary vectors, but the distances among those vectors may still show hierarchical structure. To our knowledge, this paper is the first systematic study of whether prompt-token hidden states in contemporary LLMs exhibit Gromov Hyperbolicity (GH), a distance-based measure of tree-likeness. Using 818,904 sample-layer measurements from ten open-weight models across MATH500, HumanEval, WinoGrande, and TruthfulQA, we build a GH map over four axes: parameter scale, layer depth, model family, and input domain. The clearest pattern is depth, not scale: middle layers usually form a high-relative-hyperbolicity plateau, while final layers often become substantially more tree-like. Scale effects are weak and non-monotonic, matched 7/8B model families differ strongly, and domains interact with model specialization. These findings make GH useful as a practical diagnostic: it shows where hierarchical distance structure appears, how specialization changes it, and which model-layer-domain comparisons deserve closer analysis. 

---
# Aha-Flow Distillation: Flow Markers Matter in LLM Reasoning 

**Authors**: Xiaodong Wang, Peixi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.07036)  

**Abstract**: We identify the Flow Moment, a reasoning pattern characterized by sustained, process-confirming verbalizations such as I'm doing, in contrast to the revision- and backtracking-oriented Aha Moment. We refer to their corresponding linguistic expressions as Flow Markers and Aha Markers, respectively. Based on this observation, we construct Flow-CoT by rewriting the discourse markers of original reasoning traces while preserving their underlying reasoning content, and use it as auxiliary supervision for on-policy self-distillation (OPSD). We further propose \textbf{Aha-Flow Distillation (AFD)}, a dual-mode extension of OPSD that pairs different forms of privileged information with corresponding reasoning instructions. The Aha branch retains concise solution-based supervision, while the Flow branch introduces rewritten Flow-CoT under a direct and confident reasoning instruction. At inference time, the model uses only the standard reflective instruction, so Flow-style reasoning serves purely as a training signal. Experiments on AIME25 and HMMT25 show consistent improvements across Qwen3-8B and Qwen3-4B: AFD improves Avg@12 from 60.8 to 61.3 on Qwen3-8B and from 57.5 to 58.6 on Qwen3-4B over our reproduced OPSD baselines. Controlled ablations further show that, with the same Flow-CoT/Aha-CoT composition, dual-mode training improves Avg@12 from 59.5 to 60.1, indicating that the benefit comes not only from introducing heterogeneous reasoning supervision, but also from how it is organized during self-distillation. The code is available at this https URL. 

---
# Train Overcomplete, Deploy Compact: Scaling Recovery Capacity for Structured LLM Pruning 

**Authors**: Seungmin Oh, Donggeon Lee, Jongbin Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2609.06974)  

**Abstract**: Large language models achieve strong performance across diverse tasks, but deployment remains costly because of memory, latency, and energy demands. Structured pruning reduces these costs by removing architectural components, yet its recovery stage is often limited by a mismatch between the recovery module's representational capacity and the complexity of the removed knowledge. We call this bottleneck the capacity-knowledge asymmetry and propose OverRep, an Overcomplete Reparameterization framework for structured LLM pruning. Following the principle of "train overcomplete, deploy compact", OverRep temporarily overparameterizes the recovery module during training to absorb complex knowledge distilled from the original model. After recovery, the overcomplete re-parameterization is algebraically merged into a mathematically equivalent compact module, preserving the pruned model's inference-time architecture and computational cost. OverRep further introduces an annealed activation that enables nonlinear training dynamics while converging to a linear regime for exact algebraic merging. Across three backbone families, OverRep improves retained reasoning performance over strong recovery baselines by up to 5.5 and 8.4 points at 25% and 50% pruning, respectively, while keeping memory usage and TFLOPs comparable to existing recovery methods. Our code is available at this https URL. 

---
# CantoneseLLM v2: Reasoning in a Low-Resource Language 

**Authors**: Tsz Chung Cheng, Chung Shing Cheng, Chaak Ming Lau, Cheuk Hei Chong  

**Link**: [PDF](https://arxiv.org/pdf/2609.06970)  

**Abstract**: Cantonese is widely spoken but remains low-resource in written data, with no large corpus of native Cantonese reasoning traces available for model training. We develop and release CantoneseLLM v2, comprising models based on Qwen3 8B and 30B-A3B. The models are trained through CPT on 784 million Cantonese and Hong Kong-related tokens, chat-vector merging, SFT, DPO, and RLVR. Evaluation across the training stages shows that chat-vector merging transfers instruction following but preserves the donor model's reasoning language, while SFT with limited Cantonese reasoning data substantially shortens or removes reasoning traces and reduces benchmark performance. DPO restores the reasoning-block format, particularly for the 8B model, but recovers only part of the lost performance. The RLVR training with Cantonese language and Traditional Chinese scripts as multiplicative constraints introduced Cantonese language alignment and restored the lost performance. The 30B-A3B model reaches 73.16 on HKCanto-Eval, within 1.20 points of its merged checkpoint, while retaining the Cantonese reasoning behaviour absent from that checkpoint. We release the model checkpoints, the training environments, and a thirteen-year Traditional Chinese Common Crawl dataset. The models can be accessed at this https URL 

---
# TurEngMix: A Text Corpus and Benchmark for Turkish-English Code-Mixed Language Identification and Named Entity Recognition 

**Authors**: Ilayda Dogan, Phuong-Anh Nguyen-Le, Julia Mendelsohn  

**Link**: [PDF](https://arxiv.org/pdf/2609.06963)  

**Abstract**: Natural language processing systems underperform on code-mixed text, particularly for low-resource language pairs. Turkish-English poses a further challenge: it lets English stems combine with Turkish suffixes to form single mixed-language tokens. We introduce TurEngMix, a corpus of 5.5K noisy, naturally occurring social media posts (486,974 tokens) rich in Turkish-English code-mixing. From this corpus, we construct a new Turkish-English benchmark for code-mixed language identification (LID) and named entity recognition (NER), comprising 15K expert-annotated tokens. Evaluating both decoder LLM and fine-tuned encoder baselines, we find that monolingual Turkish and English tokens are labeled reliably, but all models have high error rates on mixed-language tokens for both LID and NER. For morphologically integrated tokens, NER error rates were 5.2x and 6.3x higher for GPT-4o and Qwen, respectively. This highlights how morphological integration remains a challenge. We release the corpus, annotations, and code to support future computational and sociolinguistic research on Turkish-English code-mixing. 

---
# Dynamic-Programming-Guided Hierarchical BPE and Empirical Analysis of Vocabulary Pruning 

**Authors**: Kenny Shao  

**Link**: [PDF](https://arxiv.org/pdf/2609.06898)  

**Abstract**: Byte Pair Encoding (BPE) constructs vocabularies through greedy pair merging, but the resulting merge order does not necessarily allocate a fixed model-visible vocabulary optimally for compression. We propose Dynamic-Programming-Guided Hierarchical BPE (DH-BPE), a vocabulary-construction method that combines token exposure under exact minimum-token segmentation with the hierarchical dependencies induced by BPE training. Starting from a modestly overshot BPE candidate vocabulary, DH-BPE uses dynamic programming to measure candidate utility and applies exposure-guided, dependency-aware pruning to select a fixed-size model-visible vocabulary. We compare DH-BPE against Standard BPE and recent vocabulary-optimization baselines, including Pruned BPE, MinGram, and MinGram-PP, in primary evaluations at 12K and 16K target vocabulary sizes, with an additional 18K evaluation against MinGram only. Across the primary 12K and 16K comparisons, DH-BPE consistently improves aggregate compression over Standard BPE, Pruned BPE, and MinGram under a shared exact minimum-token DP encoder. MinGram-PP achieves stronger aggregate compression in the primary comparisons, but DH-BPE outperforms it at overshoot factors f = 2.0 and f = 3.0 in cross-corpus evaluation; at 12K, MinGram-PP reverses this ordering only with the substantially larger candidate pools at f = 4.0 and f = 5.0. Qualitative analysis further shows that DH-BPE balances later, more complete BPE merges with reusable subword components, providing a practical approach to improving vocabulary allocation under a fixed model-visible vocabulary budget. 

---
# Towards Bridging the Gap Between Offline and Iterative Alignment via Preference Distillation 

**Authors**: Wenbo Zhang, Wenzhuo Zhou, Hengrui Cai, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2609.06893)  

**Abstract**: Direct preference optimization DPO is a promising offline approach for aligning large language models (LLMs) due to its simplicity, computational efficiency, and implicit modeling of human preferences. Interestingly, iterative extensions of DPO have achieved stronger performance on academic benchmarks, raising two key questions: (i) Why do iterative methods generally outperform offline ones? (ii) Can their advantages be incorporated into offline alignment? To answer the first question, our controlled experiments reveal that the explicit preference model, additionally introduced in the iterative procedure, is a key factor behind its superiority over offline methods. This insight leads us to answer the second question affirmatively and propose Distilled Preference Probability Policy Optimization (DP3O), an effective and efficient offline alignment algorithm. DP3O first learns an explicit preference model using a helper class of LLMs and then distills its knowledge into policy optimization. Theoretically, we show that explicit preference modeling admits better estimation error control than implicit formulations, and that DP3O achieves a tighter generalization bound than hard-label DPO through variance reduction. Empirically, we evaluate DP3O on a wide range of chat-based and downstream tasks and show that it outperforms state-of-the-art offline methods, achieves performance comparable to iterative DPO, and reduces training time by about $42\%$, demonstrating both its effectiveness and efficiency. 

---
# AutoLexSteer: Automatic Contrast Construction for Lexical Activation Steering 

**Authors**: Shuhe Wang, Lachlan Cowley, Eduard Hovy, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2609.06879)  

**Abstract**: Steering vectors have rapidly emerged as a popular and effective method for guiding the output of LLMs in very specific ways. But constructing accurate steering vectors is a difficult manual process due to the opacity of embeddings. We introduce Hangman, a novel type of steering vector that operates using word senses, as well as AutoLexSteer, the first fully automated process for building steering vectors. AutoLexSteer employs families of closely-related words extracted from WordNet to specify both the steering source to be avoided and the desired steering target. The steering vectors are quite precise, can be used to steer at the level of words and sets of word senses (meanings), and are able to steer certain LLM behaviors like sycophancy. The dataset and code can be found at this https URL. 

---
# You Are What You Read: Misalignment via In-Context Persona Induction 

**Authors**: Kyuhee Kim, Benjamin Berczi, Cozmin Ududec  

**Link**: [PDF](https://arxiv.org/pdf/2609.06851)  

**Abstract**: Broad misalignment has been produced by finetuning on narrow data, harmful or benign, and in context only by demonstrations of the undesirable behaviour itself. We show that benign data suffices in context, with no finetuning and no demonstration of harmful behaviour in the prompt. Biographical facts that converge on a single figure, placed in a model's context as ordinary conversational turns, lead it to answer as that figure on questions the facts never touch. We call this persona induction. Across nine personas and thirteen models, identity adoption rises sigmoidally with the number of facts and crosses 50% within 3 to 10 of them. Misalignment then tracks which figure is described. Harmless personas reach full adoption with near-zero misalignment, while harmful ones voice their characteristic views on unrelated questions, at rates up to 80%. A formatting instruction can gate when the persona activates. Because each fact is individually benign, accumulated biographical context is flagged by content filters on 3% of inputs against 24-33% for an equivalent direct instruction. 

---
# XYBench: Can LLMs Respond Pragmatically to Queries with Misconceptions? 

**Authors**: Akhila Yerukola, Jena D. Hwang, Mingqian Zheng, Jenna Godsey, Hyunwoo Kim, Valentina Pyatkin, Jennifer Hu, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2609.06842)  

**Abstract**: When non-expert users ask LLMs for assistance, their queries can often have misconceptions (e.g., "How do I parse XML with regex?"). In such cases, often referred to as the XY-problem, LLMs must identify the misconception ("regex are fragile") and meaningfully direct the user toward a pragmatic solution that will address the root problem implicit in the request ("use an XML parser"). We introduce XYBench, a benchmark of 8,115 such queries, drawn from technical (StackOverflow/StackExchange) and everyday (WikiHow and a manually-curated subset) domains. We design an evaluation paradigm that assesses model responses along three criteria grounded in cooperative response theory: (a) presence and (b) emphasis on pragmatic solutions, and (c) identification of misconceptions. Our experiments show that even the strongest LLMs predominantly answer the literal request (0.75--0.92) and far less often the intended one (0.33--0.71), while substantially lagging behind humans at identifying misconceptions (at most 63% vs. 79--90%). Further, models overwhelmingly prefer pragmatic responses in a multiple choice setting yet consistently fail to generate them. Oracle ablation experiments show that providing explicit user intent at generation time helps; however a large gap remains, suggesting pragmatic redirection is a fundamentally underdeveloped capability in current LLMs. 

---
# Typed Federated Artifacts for the Agentic Web:Sharing Tool-Routing Knowledge Across Frozen,Heterogeneous LLM Agents 

**Authors**: Abhijit Chakraborty, Ni Trieu, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2609.06815)  

**Abstract**: An open, networked web will allow agents to run frozen models from multiple vendors, keep their history private, and teach each other which tool to call and when. Flat text (prompts, example pools) makes it difficult for the protocol to distinguish between noise statistics, merging rules, and documentation. Weights and adapters cannot transfer that knowledge between platforms. We suggest sharing typed federated artifacts, schema-validated objects with well-defined fields for per-field privacy (described here, but measured), dispute resolution, and cross-model transfer, and instantiating them as SYNAPSE1, a common tool-routing knowledge. After deleting 192 garbage entries and 1,916 training items that duplicate or almost duplicate test queries, a federated compendium routes within 1.1 points of a centralized one at 20 MB of JSON per client each round on StableToolBench (3,180 tools). The same experience merged and shown to the router as typed fields rather than one flat string is worth 8.5 points on clean data and 7.4 under 60% injected contradiction. Crossing merge and rendering shows the halves are inseparable (the typed merge shown flat is the worst arm), while three conflict policies are indistinguishable, so the conflict log that motivated this work is not the On {\tau}-bench retail, each compendium arm improves GPT-4o agents' per-step tool-call accuracy by at least 6.7 points, attributed to format rather than federated experience. Two cautionary findings conclude the paper: on a topic-labeled math proxy and StableToolBench, a TF-IDF classifier over the same labeled experience beats every LLM routing arm (by 48 and 26 points, mostly retrieval recall) because the benchmark's pool holds labeled queries for every supposedly unseen tool and every test query verbatim before our filter. It cannot measure routing to tools without labels, which routing exists for. 

---
# AuthBench: A Large-Scale Multilingual Benchmark for Authorship Representation across Genres and Lengths 

**Authors**: MaoXun Huang, Zhenxing Zhang, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2609.06771)  

**Abstract**: Authorship signals matter in settings where writing style carries identity: digital forensics, plagiarism analysis, account linking, misinformation investigation, and machine-generated text detection. Yet current authorship benchmarks remain fragmented, usually covering only a narrow language set, a single genre, or a limited document-length regime, which makes it difficult to assess whether modern representations truly generalize. We introduce AuthBench, a large-scale multilingual benchmark for authorship representation that is designed to make this evaluation broad, standardized, and realistic. AuthBench contains 428,150 documents written by 153,825 individuals across ten widely used languages, 9 primary genres, 66 fine-grained genres, and four document-length buckets. It supports two complementary tasks: authorship attribution, formulated as same-author retrieval and authorship verification, formulated as same-author binary decision. We benchmark 47 neural models and three non-neural baselines under a unified zero-shot protocol. Results show that authorship representation remains far from solved: the best retrieval model reaches only 0.258 Success@5, while the best verification model achieves 0.076 EER and 0.968 ROC-AUC. The leaderboard also reveals a meaningful task split, with different model families leading retrieval and verification, and large performance differences across languages, genres, and lengths.
These findings position AuthBench not only as a new benchmark, but as a diagnostic resource for studying when and why authorship representations succeed or fail. We release AuthBench, its evaluation toolkit, and benchmark data at this https URL and this https URL. 

---
# Event Interaction in Low-Rank Bottlenecks for Temporal Relation Extraction 

**Authors**: Wei Sun, Tingyu Qu, Jesse Davis, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2609.06731)  

**Abstract**: Temporal relation extraction determines whether an event occurs before, after, or simultaneously with another event, and therefore relies on accurately modeling how the two events interact. Mainstream systems achieve this by concatenating event spans or using shallow fusion, which works well when all model parameters are trainable. However, in parameter-efficient fine-tuning, low-rank bottlenecks restrict information flow and prevent these interaction signals from passing through, leading to clear performance drops. To address this limitation, we propose a theoretically grounded architecture, Convolutional Bottleneck Interaction (CBI), which first applies lightweight depthwise convolution to enhance event representations and then uses element-wise multiplication to capture effective event-event interactions inside the bottleneck. Across five datasets and seven backbone models in the Adapter and LoRA settings, CBI provides consistent and substantial gains, up to +31.7 micro F1, while adding minimal computational cost, showing that explicit interaction inside low-rank spaces is crucial for temporal relation extraction. The code is available at this https URL. 

---
# DianShi-RxnDB: A Large-Scale, Fine-Grained Organic Reaction Data Platform Built via a Fully Automated Pipeline for Researchers and AI Agents 

**Authors**: Yubin Wang, Xingjian Wei, Jiang Wu, Yinfan Wang, Boyu Zhu, Lin Zhang, Jianing Yu, Huazheng Zeng, Ruiyi Ding, Junyuan Gao, Jiaxing Sun, Lingli Ge, Haote Yang, Jingchao Wang, Aijia Guo, Qian Jiang, Yurui Zhao, Wenjian Zhang, Chen Zhu, Lijun Wu, Xiaolei Yang, Haodong Chen, Junjie Yuan, Zichao Ye, Shaowei Hou, Jing Ye, Jia Yu, Shan Wang, Lijun Wu, Jiantao Qiu, Chao Xu, Yuqiang Li, Guangyu Wang, Bowen Zhou, Dahua Lin, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2609.06703)  

**Abstract**: High-quality structured organic reaction data are essential for developing artificial intelligence for chemistry (AI4Chem), yet much of this knowledge remains dispersed across patent text, images, and reaction schemes. We present DianShi-RxnDB, a large-scale, fine-grained organic reaction data platform built via a fully automated extraction and normalization pipeline integrating patent text, images, and reaction schemes. Its corpus covers organic synthesis patents from the USPTO and EPO published between 1976 and 2025, yielding approximately 24 million reaction instances, of which approximately 14.8 million (61.7%) pass automated qualification checks. Each instance represents a specific single-step experiment recording participants, roles, quantities, temperatures, reaction times, yields, experimental procedures, and provenance links to source patents. In a manual evaluation of 1,300 sampled qualified instances, the micro-averaged field-level accuracy was 92.95%. A matched comparison with Pistachio further indicated advantages in deduplicated record counts, representation granularity, and field-level exact agreement. The platform provides a Web research workbench for searching, filtering, comparing, and source-verifying records, and a Model Context Protocol (MCP) service offering AI agents composable structured retrieval tools. DianShi-RxnDB is available at this https URL . 

---
# PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents 

**Authors**: Kun Li, Zexuan Qiu, Tianhua Zhang, Irwin King, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2609.06702)  

**Abstract**: Sequential memory agents process long documents by reading chunks one after another while maintaining a compact memory state, coupling document traversal to reasoning depth. This coupling introduces sensitivity to evidence placement and ties inference latency linearly to document length. We introduce PARSER, which decouples reading from reasoning. A bank of lightweight subagents each bound to a single chunk read the entire document in parallel, while a lead agent reasons in depth through iterative scatter--gather rounds: at each round it broadcasts a query to all subagents, aggregates the returned evidence, and formulates a deeper follow-up query conditioned on what has been found so far. This decoupled design concentrates all learnable behavior in the lead agent, which is optimized with reinforcement learning, while the subagents remain frozen off-the-shelf models. On multi-hop QA with contexts ranging from 7K to 896K tokens, PARSER with a 4B backbone outperforms the strongest sequential memory baseline by 5.7 points on average and by 12.0 points at 896K tokens. Scaling to a 9B backbone, PARSER surpasses DeepSeek-V4-Pro by 6.3 points. Controlled experiments confirm that PARSER is robust to perturbations in evidence position, order, and distance, conditions that cause large accuracy swings in sequential methods, while reducing inference latency by up to 11x. 

---
# A Grapheme-Aware Indic Tokenizer for Tamil: Large-Scale Training and Intrinsic Evaluation 

**Authors**: Hari Krishnan K V, Sudarsun Santhiappan  

**Link**: [PDF](https://arxiv.org/pdf/2609.06690)  

**Abstract**: Tokenization forms the foundation of modern Natural Language Processing (NLP) systems by transforming raw text into discrete units that neural language models can process. The effectiveness of this process directly influences vocabulary efficiency, sequence length, computational cost, and downstream model performance. Although multilingual tokenizers such as Byte Pair Encoding (BPE), WordPiece, and SentencePiece have performed well across numerous languages, they often segment morphologically rich Indic languages inefficiently. Tamil, in particular, poses unique challenges because its grapheme-based writing system can represent a single visible character with multiple Unicode code points. In this work, we present a grapheme-aware Indic tokenizer for Tamil that preserves complete grapheme clusters through a reversible Unicode mapping strategy prior to WordPiece vocabulary learning. By operating on grapheme-level representations instead of individual Unicode code points, the tokenizer produces linguistically meaningful token boundaries while remaining fully compatible with transformer-based language models. The tokenizer is trained on a large-scale Tamil corpus and evaluated using a comprehensive intrinsic evaluation framework that measures compression efficiency, token fragmentation, information density, and vocabulary utilization. Experimental evaluation compares the proposed tokenizer against five widely used multilingual tokenizers: GPT-2, mBERT, mT5, mBART, and NLLB. The proposed tokenizer achieves the strongest performance among the evaluated tokenizers on fragmentation- and sequence-efficiency-oriented intrinsic metrics, while matching the highest observed compression ratio. These results demonstrate the effectiveness of grapheme-aware preprocessing for Tamil tokenization. 

---
# ECOKV: Geometry-Aware KV Cache Eviction via Complementary Diversity Metrics 

**Authors**: Chin Ting Hsu, Yu-Syuan Xu, Ling Zou, Hsien-Kai Kuo, Wen-Huang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.06663)  

**Abstract**: Although multimodal Large Language Models (MLLMs) excel in diverse tasks, their scalability remains limited by the memory and computational overhead of KV cache storage. Recent KV cache eviction approaches incorporate a cosine similarity-based diversity metric with importance metrics to selectively retain critical key-value pairs. However, cosine similarity involves normalization that discards magnitude information, and it often yields uniformly high similarity values across layers due to the anisotropy property of hidden representations. In our study ECOKV, we rigorously deconstruct the capabilities of existing diversity metrics. Moving beyond simple measurement, we propose a geometry-aware composite metric that jointly leverages Euclidean distance and cosine similarity to capture token diversity from complementary perspectives. Furthermore, we use these two metrics to estimate the redundancy level of each attention head, allowing adaptive weighting between diversity and importance scores during token selection. Finally, we demonstrate that the observation window commonly employed to preserve recent tokens can be substantially reduced, thereby allocating more cache capacity to informative tokens and yielding consistent improvements. Extensive experiments demonstrate that ECOKV achieves state-of-the-art performance under various compression ratios and can be seamlessly integrated with existing KV cache eviction methods. We further analyze the relationship between importance and diversity, and examine redundancy patterns across layers and attention heads. 

---
# Mind the Gap: Exposing LLM Translation Blind Spots Using the AlphaMWE Multilingual Parallel Corpus 

**Authors**: Lifeng Han, Jiahui Liang, Anna Latusek, Karim El Haff, Amal Haddad Haddad, Josua Höfgen, Kilian Evang, Min Ma, Maryia Zhyrko  

**Link**: [PDF](https://arxiv.org/pdf/2609.06634)  

**Abstract**: LLMs' performance on machine translation (MT) tasks is often dependent on the data availability in the specific domains and language pairs that they are trained upon. To examine if Multiword Expressions (MWEs) still set a bottleneck for LLMs regarding language understanding and translation, we report the system performances from the WMT2026 Test Suites shared task, for which we used the publicly available multilingual parallel corpus AlphaMWE as the test suites. We received 31 MT systems' outputs covering English to Chinese (zh), Polish (pl), German (de), Arabic (ar) including Modern Standard Arabic (MSA) and two dialectal ones (Egyptian and Tunisian Arabic). We carried out automatic evaluations using BLEU, ChrF, BERT-score to select the Top3 systems per language pair, followed up with human evaluations on the selected systems. Our findings show that: figurative/MWE phenomena remain challenging; automatic metrics sometimes disagree; human evaluation uncovers language-specific errors hidden by aggregate scores. 

---
# SAGE: A Hierarchical Framework for Evaluating Interpretive Literary Quality in Narratives 

**Authors**: Tianyu Wang, Nianjun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.06611)  

**Abstract**: Assessing the literary quality of narratives requires evaluating interpretive dimensions (cultural representation, emotional depth, and philosophical engagement) that existing NLG metrics cannot measure. We introduce SAGE, a six-layer evaluation framework that separates rule-based assessment of observable textual properties from LLM-based evaluation of interpretive qualities drawn from cultural theory, affect theory, and existentialist philosophy. Each interpretive layer is assessed through multi-round iterative LLM evaluation with independent cross-validation, achieving measurement-grade reliability (98.8% convergence, >94% inter-rater agreement) stable across evaluator models. Validated on 600 evaluations across 100 short stories, our central finding is a systematic capability boundary: emotional-psychological representation approaches human levels, while cultural critique and philosophical depth exhibit approximately double the gap. LLM-generated narratives score below even commercial genre fiction on all three layers. We interpret this as a boundary between pattern-reproducible literary capacities learnable from training corpora and stance-requiring ones demanding cultural positioning and philosophical engagement that pattern matching alone cannot provide. 

---
# Discovering Translation-Worthy Languages with E-Values 

**Authors**: Wajdi Ben Saad, Safa Madiouni  

**Link**: [PDF](https://arxiv.org/pdf/2609.06593)  

**Abstract**: Choosing when to translate multilingual documents is a central routing problem in text classification: translation can improve predictions for some languages while degrading others or adding unnecessary computation. Uniform translation and heuristic language tiers do not provide statistically controlled route selection. We introduce a language-level router based on paired e-processes that continuously compares direct and translation-assisted classification before freezing a routing policy. A familywise-controlled threshold of 280 bounds the probability of any false route across 14 eligible languages per dataset by 0.05. On SIB-200 and MASSIVE, the router selects translation for 4 of 15 languages and 14 of 15 locales, improving held-out accuracy over direct classification by 8.14 and 16.70 percentage points, respectively. All 28 decisions remain stable across 50 outcome-independent orderings and relative to the per-group threshold. Our results demonstrate that paired e-processes enable statistically controlled, anytime-valid, and auditable multilingual classification routing. 

---
# LLMs Mirror Country-Specific Gender Patterns If Asked, but Skew Male When Generating Media in Local Languages 

**Authors**: Sharif Kazemi, Tanya Popli, Neil K. R. Sehgal, Sunny Rai, Niyati Malhotra, Victor Orozco-Olvera, Ana María Muñoz Boudet, Samuel P. Fraiberger, Sharath Chandra Guntuku, Manuel Tonneau  

**Link**: [PDF](https://arxiv.org/pdf/2609.06545)  

**Abstract**: Large language models (LLMs) are increasingly used to generate media, but whether their content perpetuates gender stereotypes is unknown: standard benchmarks rely on selection-based formats rather than long-form generation, and surveyed baselines for local gender associations are scarce outside the West. We collect gender associations for 22 occupational and domestic roles from 695 respondents across the United States, India, Kenya, and Nigeria, and evaluate eight LLMs under two regimes: direct questioning and media generation. Models track the surveyed associations under direct questioning but skew substantially more male under media generation in major local-language cells, consistent with the male bias documented in human-produced media. Outside the US, the shift is much smaller and non-significant under English prompting, so English-only or country-agnostic evaluation would miss this bias in the languages where these models are most deployed. Instruction prompting reduces the shift directionally, but trades off against alignment with the surveyed associations. Evaluating LLM gender bias for global deployment therefore requires generation-format testing, local-language prompting, and locally-collected human baselines. 

---
# ProcArena: A Multi-Scenario Benchmark for LLMs on Direct and Interactive PL/SQL Development from Natural Language 

**Authors**: Hang Zhang, Chaokun Wang, Yuzhi Pan, Ziyao Zhong, Shuo Cao, Yue Xue, Zeyu Huang, Xingwei Zhou, Fang Niu, Bofan Xie, Guanchen Ge, Leqi Zheng, Ziyang Liu, Xiannian Cao, Pengcheng Ge  

**Link**: [PDF](https://arxiv.org/pdf/2609.06527)  

**Abstract**: Large language models (LLMs) have shown strong potential for translating natural-language (NL) requirements into PL/SQL programs, attracting increasing attention from the database community. However, existing NL-to-PL/SQL efforts primarily focus on directly generating PL/SQL from complete NL requirements. In practice, PL/SQL development involves diverse scenarios, such as from-scratch development, code modification, debugging, and optimization, and may require either direct generation or multi-turn interaction. Yet, no comprehensive benchmark evaluates multi-scenario, direct and interactive, and multi-dialect NL-to-PL/SQL development. In this paper, we present ProcArena, an execution-based benchmark covering both Direct and Interactive modes. ProcArena comprises 3,998 executable tasks over 157 databases, spanning nine development subscenarios in PostgreSQL and Oracle. We construct challenging Direct tasks through Iterative Logic Enhancement and scenario-specific adapters, and derive paired Interactive tasks through Knowledge Integration and Requirement Perturbation while preserving executable targets. We further design a controlled Solver-User Simulator protocol that allows models to clarify user intent and inspect the database environment without exposing hidden execution feedback. Evaluating seven language models, we find that the best average scores are only 62.2% and 57.8% in Direct and Interactive, respectively, demonstrating that realistic NL-to-PL/SQL development remains challenging, particularly in interactive settings. 

---
# DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding 

**Authors**: Yaojie Zhang, Linfeng Zhang, Bin Cui, Xupeng Miao  

**Link**: [PDF](https://arxiv.org/pdf/2609.06498)  

**Abstract**: Block diffusion speculative decoding improves LLM inference efficiency by proposing a block of future tokens in parallel and verifying them with a single forward pass through the target model. However, existing methods retain only the accepted prefix and discard the rejected suffix, preventing the computation spent on these positions from benefiting subsequent drafting rounds and forcing the drafter to repeatedly reconstruct representations for future tokens from scratch. We observe that rejection only determines whether a proposed token can be committed, while the verifier representations at rejected positions can still provide useful information for subsequent predictions. Based on this observation, we propose DFlow, a simple yet effective framework that enables verifier information to flow across drafting rounds. DFlow reuses the hidden states produced by the target verifier for the rejected suffix to guide subsequent drafting without additional target computation. To effectively learn this information flow across drafting rounds, we introduce a self-condition train strategy that feeds verifier representations from earlier predictions back into subsequent predictions. Experiments on Qwen3 models across diverse benchmarks demonstrate that DFlow consistently improves draft quality and acceptance length over DFlash. 

---
# Decomposing LLM-Judge Uncertainty to Target Expert Labels 

**Authors**: Ryan Lail  

**Link**: [PDF](https://arxiv.org/pdf/2609.06444)  

**Abstract**: An LLM judge evaluates outputs at scale. Experts should label only where it is least sure. Its natural escalation signal conflates two uncertainties: aleatoric, real disagreement in the expert pool, which labels cannot reduce, and epistemic, the judge's ignorance, which labels do reduce. A small Bayesian model separates them: a regression on labels already collected learns how far to trust a black-box judge's prediction. Both components follow as simple formulas, with no sampling or further judge calls. The components isolate on a real LLM judge against exactly known truth, and stated confidence is no guide to its actual error. On real human disagreement (ChaosNLI) the epistemic ranking removes 83% more error than total uncertainty for the same expert labels, though simply escalating the least-labelled items does as well there. We demonstrate we can estimate where a judge is ignorant rather than where experts genuinely disagree, and propose using this to direct expert labelling. 

---
# InsightChain: Optimized Chain-of-Insight Analytics for LLM-driven Data Visualization 

**Authors**: Hanya Sun, Chen Zhang, Sheng Liang, Yongyue Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.06438)  

**Abstract**: Large language models (LLMs) are increasingly used for automated data visualization, yet existing approaches often frame visualization generation as a single-step mapping from user query to figure or code, overlooking the iterative analytical reasoning process of expert analysts. We present InsightChain, a four-stage visualization prompting pipeline (Explore--Focus--Test--Present) that emulates expert analytical workflows, together with VG-COPRO, a vision-guided automatic prompt optimization (APO) method adapted to jointly optimize such multi-stage, executable pipelines. To address the evaluation gap for complex data visualization, we introduce the Insight Progression Metric (IPM), a rubric combining four text-based dimensions with a vision-based dimension. We assess IPM through a 100-chain human pilot and an expanded 300-chain agent-based evaluation spanning all ten domains. Experiments on public datasets show that InsightChain consistently outperforms competing prompting baselines. Existing APO methods fail to yield consistent gains on this multi-stage task, whereas VG-COPRO improves performance in both in-domain and cross-domain settings. 

---
# Visual Search Augmented Chain-of-Thought Reasoning for Attribute Value Extraction from Product Videos 

**Authors**: Tong Wu, Ming Cheng, Jiazhen Hu, Jiaying Gong, Hoda Eldardiry  

**Link**: [PDF](https://arxiv.org/pdf/2609.06410)  

**Abstract**: Existing approaches to visual attribute value extraction (AVE) primarily rely on static product images, failing to capture temporal cues, multi-angle views and fine-grained visual details. Directly applying video vision-language models (VLMs) to product AVE results in limited performance due to the lack of domain knowledge, and fine-tuning them requires extensive high-quality data and substantial computational resources. Thus, we propose visual search augmented chain-of-thought reasoning (ViS-CoT), a training-free, plug-and-play pipeline that can be easily applied to any open-source video VLM for video-to-text AVE in e-Commerce. Specifically, ViS-CoT employs visual clustering to identify representative frames, followed by visual search to retrieve semantically similar product knowledge that can enrich attribute cues. Next, an interleaved CoT reasoning module iteratively refines reasoning through visually-aligned auxiliary texts derived from captioning and automatic speech recognition. Finally, the integrated information guides the model toward accurate and fine-grained attribute predictions. Extensive experiments across 14 product categories on the VideoAVE dataset show that ViS-CoT consistently enhances multiple state-of-the-art video VLMs, achieving an average improvement of 17.91 percentage points in micro-F1. 

---
# Hierarchical Wasserstein Merging for Multi-Domain Multi-Task Learning: From Specialists to a Generalist 

**Authors**: Ming Cheng, Jiaying Gong, Hoda Eldardiry  

**Link**: [PDF](https://arxiv.org/pdf/2609.06406)  

**Abstract**: Multi-domain multi-task learning (MD-MTL) aims to build a single generalist model that performs well across heterogeneous domains and tasks. However, joint training often suffers from interference under distribution shifts. Existing model merging methods mostly operate on model parameters while overlooking the geometric structure of latent representation distributions across domains and tasks. To address these limitations, we propose Hierarchical Wasserstein Merging (HWM), a representation-level framework that models each domain-task specialist as a distribution of hidden representations on a shared support. HWM constructs task-level and global Wasserstein barycenters to capture within-task domain variation and cross-task structure, enabling either training-free specialist aggregation by Wasserstein-derived weights or training-based generalist learning through a hybrid Wasserstein alignment loss. Experiments on four NLP tasks across four domains per task show that HWM achieves superior effectiveness and generalization capability in MD-MTL settings. 

---
# Cross-Lingual Representation Alignment by Token-Level Optimal Transport in a Language-Agnostic Space 

**Authors**: Taisei Yamamoto, Ryoma Kumon, Danushka Bollegala, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2609.06381)  

**Abstract**: Cross-lingual alignment (CLA) aims to align the representations of large language models (LLMs) across languages, enabling cross-lingual transfer to improve multilingual capabilities. Previous CLA methods often ignore language-specific information encoded in representations and only consider sentence-level alignment, which may lead to suboptimal performance and input-output language mismatch. We propose CAROT (Cross-Lingual Alignment of Representations in a Language-Agnostic Space via Optimal Transport), which consists of two steps: identifying language-specific representations in LLMs' internal states and aligning language-agnostic representations across languages at the token level by optimal transport, while explicitly preserving language-specific representations. Inference-time steering experiments show that the representations computed by CAROT are effective alignment targets, improving multilingual performance by up to 11.2 points in accuracy while maintaining input-output language consistency. We further use the representations obtained by CAROT as training targets, internalizing the aligned representations. The trained models outperform existing CLA methods in 11 of 18 evaluation settings (3 models $\times$ 3 tasks $\times$ ID/OOD languages). Our work provides insights into what constitutes effective alignment targets for CLA in LLMs. Code is available at this https URL 

---
# A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models 

**Authors**: Lin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2609.06324)  

**Abstract**: Autoregressive language models commit one token per forward pass; diffusion language models commit a block of tokens over several steps. We ask whether a block can be committed in a single forward pass. We study this with a noise-conditioned masked denoiser: a data-independent Gaussian noise field is added to the mask embeddings so that, in principle, each sampled field selects one joint mode of the block. The established way of training such a model is to sample several fields per example and let them compete for the data, by winner-take-all or importance weighting. This gives the noise only coarse control: in our experiments, the information it carries grows roughly with the logarithm of the number of competing fields, and one-step outputs remain rarely coherent across the model sizes tested. We propose CONDOR (Coupled-Noise Distillation for One-Step Readout). A noise-conditioned teacher is trained with a random number of masked positions and winner-take-all. A student proposes a one-step block, retains selected tokens, and learns from the block obtained when the teacher refills the other positions in several steps under the same noise field; a noise-free masked-LM term on the ground truth anchors the student. Human evaluation on TinyStories shows a large gain in one-step legality while different noise fields still yield different blocks, at one forward pass per block. 

---
# Reliability, validity, and diagnostic evidence for multi-model LLM short-answer scoring 

**Authors**: Chunyi Zhao, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.06315)  

**Abstract**: Large language models (LLMs) are increasingly used or proposed for educational scoring, but single-model and single-run evaluations provide limited evidence for assessment use. Short-answer scoring requires evidence about reliability, validity, severity, diagnostic value, and failure cases. This study evaluated repeated multi-model OCG-PRES guided LLM scoring for short-answer assessment. The analysis used 996 SciEntsBank responses. GPT, DeepSeek, and Qianwen each scored every response across three independent runs using five OCG-PRES dimensions: concept coverage, relation accuracy, reasoning completeness, contradiction control, and domain relevance. Scores were evaluated against official binary and five-category labels and compared with non-LLM baselines based on answer length, Jaccard keyword overlap, TF-IDF cosine similarity, and a combined traditional logistic model. Repeated-run reliability was high for all models, with ICC(3,k) = .977 for GPT, .992 for DeepSeek, and .981 for Qianwen. DeepSeek was the most stable across runs. GPT showed the strongest official-label alignment by AUC (.909), while Qianwen was stricter, with higher precision but lower recall under the fixed threshold = 3.0 rule. OCG-PRES scores followed expected diagnostic patterns across five official categories and outperformed all non-LLM baselines in AUC and F1. Repeated multi-model OCG-PRES scoring provides reliability, validity, and diagnostic evidence for LLM-assisted short-answer scoring. The findings support cautious, evidence-based use as a scoring support tool rather than a replacement for human judgement. 

---
# Steering Geometry: Validating Human Value Geometry in LLM Steering Space 

**Authors**: Mohammad Mahdi Abootorabi, Armin Saghafian, Ali Bazshoushtari, Hamid Rezaei, EunJeong Hwang, Vered Shwartz, Parvin Mousavi, Purang Abolmaesumi  

**Link**: [PDF](https://arxiv.org/pdf/2609.06289)  

**Abstract**: As large language models (LLMs) are increasingly deployed in alignment-sensitive contexts, activation steering has emerged as a lightweight, inference-time alternative to fine-tuning methods (e.g., RLHF, DPO) for behavioral control. However, existing work typically validates steering on isolated behaviors, leaving it unclear whether steering vectors encode coherent semantic structure or merely exploit behavior-specific shortcuts. We investigate whether the latent geometry of LLM steering vectors reflects theory-specified structure in human values and morality. Using Schwartz's Theory of Basic Human Values as our primary fine-grained framework, we introduce a 26K-sample benchmark covering 20 human values and analyze distribution-driven methods (e.g., CAA, SphericalSteer, ODESteer) and behavior-centric approaches (e.g., COLD-Steer, BiPO) across diverse model families and sizes. We find that distribution-driven methods recover human value topologies aligned with theoretical predictions (Spearman $\rho$ up to 0.51, $p < 10^{-13}$). In contrast, behavior-centric methods achieve comparable steering performance but show little correlation with the expected value geometry. Geometric fidelity improves with model scale but drops after instruction tuning. Finally, better geometric alignment also leads to more human-consistent transfer across values: steering one value correctly lifts compatible values and suppresses opposing ones. Code and data are available at: this https URL. 

---
# Correction as Annotation: Bootstrapping a Dependency Parser for Documentary Medieval Latin 

**Authors**: Gabriel H. Pizzorno  

**Link**: [PDF](https://arxiv.org/pdf/2609.06266)  

**Abstract**: Medieval documentary sources remain inadequately served by existing natural language processing tools. None of the five readily available Latin treebank models attains usable performance on a collection of 160 inventories compiled in Marseille between 1258 and 1446. The best labelled attachment score is 0.62 and the best morphology-aware score is 0.24. Performance does not correlate with either genre or period proximity. To address this shortfall, in-domain training data was generated as a by-product of using these inadequate models. In each of nine iterations, a model pre-annotated 200 sentences; an expert corrected the annotations; and the corrected sentences were used to train the subsequent model, with batches sampled independently of model state, without active-learning selection. Thirty-three hours of annotation effort over 1,804 sentences increased universal part-of-speech accuracy from 0.80 to 0.98 and labelled attachment from 0.48 to 0.92, outperforming all baselines on the reported metrics while using 97% less training data than the largest one of them. Annotator effort declined from 54% of tokens to a plateau of 14-18%, an operational progress metric that requires no separate gold standard and can serve as a stopping criterion. 

---
# Beyond the Flag: Clinical Framing Closes the Moderation Gap in Suicide Risk Measurement 

**Authors**: Shreyas Krishnan, Gun Ahn, Jungjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.06263)  

**Abstract**: Moderation APIs are built to flag policy-violating content, not to measure graded clinical risk. But a platform's duty does not end at detection: the response owed to passive distress differs sharply from the response owed to active planning with means access, and emerging regulation (e.g., California Senate Bill 243) is turning that distinction into a compliance requirement. We therefore ask how well deployed safety signals recover clinically meaningful severity. We release a benchmark of 516 r/SuicideWatch posts rated by a licensed psychiatrist on a four-level ordinal schema (Indicator, Ideation, Behavior, Attempt) grounded in the Columbia Suicide Severity Rating Scale, and evaluate moderation APIs, prompted LLMs, and supervised baselines under seven ordinal-aware metrics. Three findings. Vendor moderation APIs separate low- from high-severity posts well (0.860 high-risk F1) but measure severity poorly (0.395 macro F1), systematically over-predicting the most severe category. Clinically grounded zero-shot prompting recovers much of that gap (0.562 macro F1), and expert-authored framing (not fine-tuning, added reasoning, or naive multi-agent aggregation) is the effective lever. The value of reasoning depends on register: it hurts on long, noisy Reddit posts and helps on short, clinician-authored statements. We argue graded severity, not a binary flag, is what a proportionate duty of care requires, and release our evaluation framework to support that measurement. 

---
# SLATE: Are AI-Generated Slides Educationally Effective? A Benchmark for Language Teaching Quality and Learner Knowledge Acquisition 

**Authors**: Jingzhuo Wu, Jiajun Zhang, Liu Yi, Leqi Zheng, Yuheng Jing, Xinyuan Zhou, Quan yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.06212)  

**Abstract**: LLMs have achieved remarkable capabilities in generating language teaching slides. However, a critical mismatch persists between visual polish and actual instructional effectiveness. To address this gap, we introduce SLATE (Slide-based Learning Assessment for Teaching Effectiveness), the first benchmark that evaluates AI-generated language teaching slides through instructional effectiveness and learner knowledge acquisition. SLATE transforms linguistics olympiad puzzles from low-resource languages with negligible web presence into 90 standardized instructional units comprising 1,133 assessable items, paired with a structured course outline and matched near- and far-transfer test sets. This pretest-posttest design eliminates pretrained knowledge leakage, ensuring gains reflect learning rather than prior recall. Using VLMs as scalable learner proxies and directionally supported by a three-system human pilot, our results show that content validity exhibits a weak association with learning gain, while pedagogical design exhibits a robust positive association. Moreover, most systems show a significant gap between near- and far-transfer accuracy, and even frontier models can produce negative learning gains. SLATE reveals a dissociation between artifact quality and instructional effectiveness, calling for a paradigm shift in how generative teaching systems are built, evaluated, and deployed. 

---
# What the Window Does Not Contain: Auditing Provenance in a Document-Grounded Instability Benchmark 

**Authors**: Seyed Mosayeb Alam  

**Link**: [PDF](https://arxiv.org/pdf/2609.06147)  

**Abstract**: Ask a language model the same question about the same document twenty times, and it sometimes returns two different answers. We built Probity, a benchmark of 60 tasks and 470 items from real venture-financing filings, to measure how often this happens. Then we audited our own corpus and found a defect any excerpt-built benchmark can carry: items whose evidence is missing from the window of text the model is shown. The audit flags 36 items and separates two failures a single flag would conflate: evidence genuinely absent from the window and answers that must be computed from numbers the window does supply. Flagged items change their answers far more often, wobbling at 0.255 against 0.087 on the 427 clean items, and excluding them cuts apparent cross-model agreement by about a fifth. Before testing whether the missing evidence explains the instability, we registered a prediction: re-cut each window to hold its evidence, and instability should fall below a set threshold. It failed: the repair moved wobble by 0.058, with an interval containing zero. We report the association as correlational. Almost all measurements sit where instability cannot show, which bounds what a corpus built for accuracy can say about stability. We release the corpus, all 112,800 raw responses, and the audit as a runnable check for any document benchmark. 

---
# Protocol Compression Changes Which Party Pays: Bilateral Cost in Cross-Organization LLM Agent Communication 

**Authors**: Janghoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.06129)  

**Abstract**: Agents that talk across organizations exchange long messages billed by the token. A shorter notation therefore looks like a saving that costs nothing but an agreement to use it. Recent work reports the saving is conditional. Compressed notation can instead raise total tokens by 8% to 11% over a JSON baseline, when parsing failures force extra model calls. That is measured for one payer. Between two organizations neither side can install a decoder at the other end, and each pays under its own tokenizer, price, and cache state. We measure both sides. A preregistered token-level study covered 198 content-matched item pairs across six vendors, for 2,376 native-usage cells. We then overlay an English baseline, runtime schema negotiation followed by compression, and injected-schema compression on a two-party procurement bargain with an exactly enumerated feasible set. The overlay covers 1,053 completed dialogues of a 1,215-cell grid across 3 model pairs, plus a 405-dialogue rerun of the negotiated condition. Compression amplifies cross-vendor cost dispersion by a factor of 1.078, with a 95% CI of [1.066, 1.091], and two vendor pairs reverse which endpoint is cheaper. Runtime negotiation succeeds as a protocol and fails as a bargain. The parties agree a schema in 121 of 135 headline dialogues, none of them the schema we would have supplied. They settle the task in only 9 of those dialogues, and they reach impasse in 106 of them. The negotiated sessions average 10.8 turns against 17.6, and cost 52% of the English total because sessions end sooner, not because the handshake is repaid. Break-even horizons run from 20 to 70 turns, the low end only under the conditional accounting, and all of them lie above every observed English session. On one cross-vendor pair both parties keep about half their cost. On the other the receiving party pays more at a high cache-hit rate. 

---
# STQA: A Benchmark for Stock-Focused Tabular Question Answering over Historical and Forecasted Data 

**Authors**: Baoxu An, Wenmian Yang, Zhensheng Wang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2609.06117)  

**Abstract**: Stock market analysis inherently requires composite reasoning over historical records and future projections, yet existing benchmarks remain fragmented across isolated tasks. We introduce STQA (Stock-focused Tabular Question Answering), an end-to-end benchmark designed to systematically evaluate natural-language question answering over historical data, numerical forecasts, and forecast-based reasoning. Built on a large-scale financial dataset, STQA covers 4,417 stocks and contains 31,400 question-answer pairs derived from expert-crafted templates, accompanied by fine-grained intent and slot annotations. To operationalize this benchmark, we present SQFRS (Stock Query-Forecast-Reasoning System), an agent-based unified framework that orchestrates SQL retrieval and time-series forecasting tools. Experiments demonstrate that while current large language models perform well on historical queries, forecast-based reasoning poses a substantial challenge, revealing critical bottlenecks in tool coordination and reasoning under uncertainty. The dataset and code are available at this https URL. STQA thus serves as a rigorous testbed for future research on trustworthy, tool-augmented financial agents. 

---
# From Two Passes to One: Compact and Efficient Target-Stance Extraction 

**Authors**: Ethan Mines, Bonnie Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2609.06108)  

**Abstract**: Target-Stance Extraction (TSE) is the task of predicting both the target (or topic) of an author's writing and the author's stance toward it. Existing approaches to TSE use a sequential pipeline of two separate neural models: one to identify the target and another to determine the stance. We present a one-pass, joint architecture that predicts both in a single forward pass, reducing trainable parameters by nearly 50% with only a 4-7 F1 point tradeoff in performance. We further demonstrate that standard target-scrubbing practices artificially suppress target prediction accuracy. Retaining explicit target mentions, as in real-world deployments, improves F1 by at least 6 points across both target classification and target generation settings. These improvements allow for significantly easier integration of TSE in downstream applications such as public opinion tracking. 

---
# DPH Parser: A Bottom-Up Grammar-Driven Parser for Joint Constituency and Dependency Analysis 

**Authors**: Hussein Ghaly  

**Link**: [PDF](https://arxiv.org/pdf/2609.06070)  

**Abstract**: This paper presents Dependency-Phrase Hierarchy Parser (DPH Parser), a grammar-driven bottom-up unsupervized parsing framework inspired by Generalized Phrase Structure Grammar (GPSG) and Head-driven Phrase Structure Grammar (HPSG). The parser incrementally constructs constituency structures using a compact inventory of feature-based syntactic rules while deriving dependency relations through explicit head annotations. The system combines probabilistic POS tagging, recursive phrase projection, and weighted parse hypotheses to process realistic and partially noisy text input. Unlike purely neural and data-driven parsers, the resulting syntactic derivations remain explicitly interpretable.
We evaluated parser performance on English corpora from the Universal Dependencies (UD) project using Unlabeled Attachment Score (UAS) as the main parsing metric, comparing the outcomes against Stanza and spaCy parsers. For a small inventory of syntactic rules, DPH parser achieved UAS values of 53.32% & 52.58% (UD Devset/Testset respectively). For the same data, Stanza achieved 89.12% & 88.67% while spaCy achieved 56.91% and 58.59%. Although the current system does not yet approach the accuracy of modern neural parsers, the results demonstrate the feasibility of applying transparent rule-based bottom-up parsing to realistic treebank data while jointly producing constituency and dependency structures. 

---
# Don't Lose Entities from Retrieval to Generation: Dual Entity Recovery RAG for multi-hop QA 

**Authors**: Heechang Lee, Dong-Young Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.06065)  

**Abstract**: Retrieval-augmented multi-hop question answering (QA) decomposes a query into sub-questions and decomposes the corpus into smaller retrieval units such as sentences. Both forms of decomposition improve the pipeline, but we show that both share the same vulnerability, the loss of entity information, and that this loss breaks the pipeline at two separate points. The first point is retrieval, where a sub-question loses the entity resolved at the previous hop, leaving the retriever with nothing to match against. The second point is harder to see, because retrieval still appears to succeed. Once a passage is split into sentences, an isolated sentence loses the context that grounds its pronouns, so even with the correct sentence in hand the LLM cannot tell which entity the sentence is about. We isolate this second point as a distinct failure mode that we call lost-in-generation, and a retrieval-controlled experiment shows that it degrades answers even when the gold evidence is fixed in the context. We then propose Dual Entity Recovery RAG (DER-RAG), which keeps the grounding entity explicit from retrieval through to generation with two lightweight components, a two-way query decomposition that carries the resolved entity across sub-questions and a subject entity prefix attached to each sentence at generation time. DER-RAG needs no graph construction, no corpus modification, and no fine-tuning, yet on three multi-hop QA benchmarks it matches or exceeds strong baselines, including graph-based methods that depend on costly offline structures. 

---
# Generating Adversarial Texts for Machine Translation via GRPO 

**Authors**: Florian Zogaj, Jakob Hütteneder, Giovanni De Muri, Federico Villa, Aryan Sood, Vilém Zouhar  

**Link**: [PDF](https://arxiv.org/pdf/2609.06048)  

**Abstract**: As machine translation (MT) systems continue to improve, standard benchmarks become less informative for exposing remaining weaknesses. Traditional methods for creating challenging test sets rely on expensive manual creation or curation, while automated approaches struggle to produce sets with the necessary translation difficulty and linguistic diversity. We propose a scalable reinforcement-learning-based approach for rewriting existing source texts into instances that are more difficult to translate for MT systems. We fine-tune a large language model with Group Relative Policy Optimization (GRPO), using reward signals based on translation difficulty together with constraints for semantic similarity, grammaticality, and approximate length preservation. On WMT25, our approach substantially reduces average COMET translation quality from 0.63 to 0.48, while preserving grammaticality and readability, whereas the base model remains at 0.64. Evaluations on the unseen WMT19-WMT24 benchmarks confirm that this behavior generalizes beyond the training data, and human evaluation further shows that the rewrites substantially lower translation quality while incurring a moderate drop in naturalness and only a small change in grammaticality. We release our code to support reproducibility. 

---
# Factors Influencing the Emergence of Dependency Length Minimization in Neural Agent Simulations 

**Authors**: Yuqing Zhang, Tessa Verhoef, Gertjan van Noord, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2609.06025)  

**Abstract**: Given various grammatical options, language users prefer the word order choice that reduces the overall length of syntactic dependencies, a principle known as dependency length minimization (DLM). The origins of this preference remain an open question, particularly whether it originates from constraints on efficient information processing. Computational simulations provide a powerful approach to identifying the factors influencing the emergence of linguistic phenomena. However, previous simulations of DLM have not examined realistic interaction contexts and have produced mixed results. The present study investigates the emergence of DLM in artificial languages using a recently proposed language learning and communication framework based on recurrent neural networks (RNNs). In this framework, agents are trained to speak and interpret artificial languages and then use these languages to communicate. Using this framework, we study the impact of several factors related to processing limitations in a communicative setting, such as noise during listening, limited speaker capacity, and incremental sentence processing. Our results reveal a complex interplay among these factors in shaping word order preferences in neural agents. Specifically, in the full meaning space, agents regularize toward a single dominant word order, while in the half meaning space they show a short-before-long preference that only aligns with DLM in verb-initial languages. A consistent DLM preference emerges only when agents are subject to incremental processing pressure. These findings suggest that limitations in human cognitive processing may indeed play a role in shaping DLM. Our findings provide insights into the conditions under which neural models replicate human-like preferences and highlight the challenges of designing emergent communication models that capture human cognitive biases in language processing. 

---
# Tri-PvP: Exposing Modality Bias in Omni-Modal Large Language Models through Perceptual-Propositional Evidence Conflicts 

**Authors**: Yen-Ting Piao, Shu-Yun Chen, Chin-Hui Chu, Chun-Wei Chen, Shih-Yun Shan Kuan, Hung-yi Lee, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.06011)  

**Abstract**: Omni-modal large language models (OLLMs) jointly process vision, audio, and text, yet their modality bias under cross-modal conflict remains underexplored. Existing benchmarks conflate two distinct forms of evidence within a single modality: perceptual signals (e.g., a photograph or recording of a dog) and propositional signals (e.g., the declarative claim "this is a dog"), such that any measured modality bias is inherently confounded with evidence-form bias, precluding clean attribution to either source. To address this, we introduce Tri-PvP, an 8,000-sample tri-modal conflict benchmark crossing vision, audio, and text, where vision and audio each take perceptual or propositional form. Evaluating five OLLMs, we find robust visual bias across most models and evidence-type conditions. Crucially, we reveal a systematic asymmetry in evidence-form bias: models exhibit a stronger bias toward perceptual signal in vision but propositional in audio. Further analyses via layer-wise linear probing and contrastive decoding reveal that modality bias is already linearly decodable from early representation layers and can only be partially mitigated, calling for mitigation strategies beyond surface-level interventions. 

---
# ModularPhaseNet: Finite-Cyclic Phase Geometry for Computable Semantic Hierarchy, Direction, and Context Consistency in Standard Transformers 

**Authors**: Kiyotaka Kasubuchi, Kazuo Fukiya  

**Link**: [PDF](https://arxiv.org/pdf/2609.06000)  

**Abstract**: We propose ModularPhaseNet, a classical and integer-computable discretization of the continuous complex phase geometry introduced in QuantumPhaseNet. The real-valued hidden states of a standard Transformer are retained, while only an auxiliary phase channel is quantized into a cyclic subgroup G = <g> of order q | (p-1) in the multiplicative group of F_p. A continuous phase e^{i phi} is represented by z = g^a mod p; phase composition becomes group multiplication, relative phase becomes group division, conceptual hierarchy is induced by a filtration of cyclic quotients, semantic direction is represented by oriented relative group elements, and contextual consistency is measured by gauge-invariant cycle holonomy. The method introduces three components into an otherwise standard Transformer: a finite-phase encoder, a quotient-filtration hierarchy module, and a group-valued connection module. Their outputs enter self-attention as real-valued bias terms. Training uses distributions in the real group algebra or straight-through Gumbel-Softmax, whereas inference uses exact modular exponentiation and precomputed tables. No quantum hardware, complex-valued matrix multiplication, or discrete-logarithm computation is required. We prove quantization-distortion bounds, nesting of quotient-induced partitions, gauge invariance, a discrete integrability result for flat connections, and boundedness of the resulting attention output. The central empirical hypothesis is that these exact discrete invariants improve hierarchy recovery, discourse alignment, contradiction detection, and calibrated hallucination-risk prediction under a controlled compute budget. This paper reports the theory together with a pre-registered evaluation plan; the experiments described in Section 14 have not yet been carried out, and no empirical result is claimed here. 

---
# Alignment by Stereotyping: How LLMs Sacrifice Individual Distinctiveness for Cultural Adaptation 

**Authors**: Qishuai Zhong, Zongmin Li, Siqi Fan, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.05993)  

**Abstract**: Large language models are increasingly deployed for personalized interaction, and demographic conditioning via user profiles is a widely adopted strategy for cultural adaptation. We ask whether this approach genuinely serves individual users or achieves accuracy by erasing individual distinctiveness. Studying seven models including frontier GPT-5.1 on the World Values Survey, we find that demographic profiles improve value alignment accuracy for most models, but at a systematic cost to individuality. That is, models pull responses toward demographic group centroids rather than preserving individual differences, a behavioral pattern we term alignment by stereotyping. Permutation tests (10,000 permutations, six demographic attributes, seven models) certify that top-performing models compress individuals far above the human baseline; within-family scaling amplifies this tradeoff while degrading intrinsic cultural understanding. Using a synthetic dialogue dataset validated on real human-chatbot conversations from PRISM (Kirk et al., 2024), we further show that distributing demographic signals across conversational turns partially suppresses prototype retrieval compared to compact demographic labels, a finding validated on real conversations via PRISM but requiring replication at larger scale. 

---
# Beyond Cross-Lingual Transfer: Benchmarking Propagation Boundaries in Multilingual LLM Unlearning 

**Authors**: Pengyang Shao, Chuanpeng Lu, Wei Qin, Yanzheng Jin, Xiaohao Liu, Xi Ai, Kenji Kawaguchi, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2609.05976)  

**Abstract**: Large Language Model (LLM) unlearning aims to suppress target knowledge while preserving general capabilities. In multilingual settings, unlearning must additionally propagate within its intended linguistic scope. However, existing evaluations mainly measure cross-lingual transfer and cannot distinguish insufficient from excessive propagation. We introduce CLLPU (Cross-Lingual and Language-Bound Protocol for LLM Unlearning), a multilingual benchmark that formulates this problem through two settings: common-goal forgetting, where target knowledge should be suppressed across all languages, and language-conditioned forgetting, where suppression should remain confined to a designated language. CLLPU combines goal-guided topic pairing, schema-aware relation matching, and dual-anchor multilingual translation to construct 800 matched knowledge-unit pairs and 72,000 QA instances across ten languages. Experiments with six representative methods on Llama-3.1-8B-Instruct reveal opposite failure modes: forgetting remains incomplete when universal suppression is required, yet spreads beyond the intended boundary when language-conditioned confinement is required. We further find that general multilingual utility can conceal damage to neighbor knowledge. These findings establish propagation control as a central challenge for multilingual LLM unlearning. We publicly release CLLPU together with its construction pipeline. 

---
# The Blindness of Document-Level Translation Evaluation 

**Authors**: Ahrii Kim, Vilém Zouhar, Chanjun Park, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.05949)  

**Abstract**: Document-level machine translation (MT) evaluation extends segment-level protocols by presenting full documents to annotators, on the assumption that such presentation elicits document-level judgments. We test this assumption with a counterfactual condition (MIX) in which each document combines segments drawn from different systems, preserving document-level presentation while breaking cross-segment consistency. Across 18,420 expert Englis-to-Korean annotations and 14 automatic metrics, scores, system rankings, and error annotations are statistically equivalent between coherent and incoherent documents. Perception does not explain this: shown matched passages, raters identify the coherent one as the work of a single translator in 87.3% of trials. Document presentation does change how annotators work, but that change does not reach the recorded output. What is blind is the protocol, not the annotator. The concern is not that scores fall short, but that the resources invested in document-level systems, metrics, and annotation may not be measuring what they are intended to measure. 

---
# SurveyAgent-HKA: A multi-agent framework for scientific survey generation with LLMs and human knowledge augmentation 

**Authors**: Tong Bao, Mir Tafseer Nayeem, Yi Zhao, Davood Rafiei, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.05938)  

**Abstract**: Automatic scientific survey generation has become an important task in scientific document processing. The common approach of retrieving literature from a single source (e.g., arXiv) and generating surveys through a one-pass large language model (LLM) call often leads to limited reference coverage and, more importantly, fails to replicate the expert-driven revision process that is crucial for writing high-quality surveys. In this paper, we introduce SurveyAgent-HKA, a multi-agent framework that improves end-to-end scientific survey generation by incorporating knowledge derived from published surveys and peer-review comments. The framework decomposes survey generation into well-defined sub-tasks handled by LLM-powered agent. It first retrieves relevant papers from multiple sources and identifies key topics through clustering to construct an initial outline, which is then refined using outlines from related human-written surveys. Based on the refined outline, topic-focused papers are retrieved and re-ranked to select for drafting a well-grounded survey. Then, we identify common issues raised by experts in peer-review comments from published surveys to guide the revisions and finalize the survey. Experiments on two domains show that our approach outperforms mainstream baselines in citation quality, structural consistency, and content quality. Furthermore, our framework is efficient in both time and cost, making it a practical solution for broader AI-assisted scientific writing applications. 

---
# Solving versus Verifying: Catching Contradictions in Tax Reasoning Systems 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2609.05928)  

**Abstract**: Large language models now compute correct tax liabilities on over 90% of well-formed cases in statutory benchmarks, which makes them candidates for the tax-advisory and compliance systems that consume such an answer directly. Real legal inputs, however, are frequently defective: required facts are missing, or stated facts contradict one another. Accuracy on clean benchmarks says nothing about how a model behaves then, and a system that computes straight through a defective input returns a confident number with no sign that anything is wrong. This raises two questions: does a model asked to solve a case abstain when the input is defective, and when it does not, can the same model catch the defect when asked instead to verify the input? We study six recent models on SARA-derived tax cases under missing-fact and contradictory-fact perturbations. The strongest models abstain when a fact is missing but compute through injected contradictions, returning the clean-input answer 63-76% of the time with no signal of the conflict; asked instead to verify the same input, they flag most of those contradictions. We wire that verification call into a simple contradiction gate: one extra call that abstains when the model reports a conflict. Across all six models it recovers most of the missed contradiction abstention at a clean-accuracy cost of at most about 5 percentage points, with no training and no external tooling. High accuracy on well-formed inputs is therefore an incomplete measure of reliability, and the detection the solver misses is cheaply recoverable with a single self-check. 

---
# Neuron-Guided Fine-Tuning: Unlocking Efficient Alignment Mechanisms for Large Language Models 

**Authors**: Zeyu Wu, Junchao Wu, Shudong Liu, Runzhe Zhan, Xin Chen, Shu Yang, Yichao Du, Longyue Wang, Weihua Luo, Jinsong Su, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2609.05913)  

**Abstract**: Existing Supervised Fine-Tuning paradigms, particularly Full Parameter Fine-Tuning are often plagued by parameter redundancy, inconsistent data quality, and catastrophic forgetting, which current methods typically address in isolation and lack a unified optimization signal to bridge data selection, parameter updates, and knowledge preservation. To address this, we propose Neuron-Guided Fine-Tuning (NGFT), a holistic framework that leverages neuron activation patterns as a universal proxy to unify the fine-tuning lifecycle. NGFT operates via three synergistic mechanisms: (1) Adaptive Task-Specific Neuron Selection, which identifies essential neurons in a single forward pass to concentrate updates and reduce redundancy; (2) Activation-Based Data Selection, which prioritizes information-dense samples that maximize contribution to key neurons; and (3) Neuron Activation Alignment, a novel loss function that anchors activations to pre-trained states, deepening representation learning and preserving general knowledge. Experimental results across three models across both domain-specific and general benchmarks demonstrate that NGFT significantly outperforms existing mainstream fine-tuning methods in both efficiency and performance, while effectively mitigating catastrophic forgetting. 

---
# UniRRM: Unified Reasoning Reward Models Across Languages and Evaluation Paradigms 

**Authors**: Peng Lai, Yichao Du, Junchao Wu, Weibo Gao, Linan Yue, Longyue Wang, Weihua Luo, Derek F. Wong, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.05910)  

**Abstract**: Reinforcement learning (RL) excels on tasks with verifiable rewards, but in open-ended tasks, the reliability of reward models remains a key challenge. Existing solutions either depend on costly proprietary LLM-as-a-Judge systems or opaque scalar reward models that lack interpretability. Recent works on generative reward models offer a promising alternative, but they remain constrained by static evaluation criteria, fragmented evaluation paradigms, and limited multilingual support. To address these challenges, we introduce \textbf{MixReward}, a large-scale multilingual dataset spanning six domains and 103 languages, containing both pairwise and listwise data, and propose \textbf{UniRRM}, a unified reasoning reward model supporting multiple languages and evaluation paradigms. UniRRM uses a staged reasoning chain to dynamically generate task-generic and instruction-specific criteria, enabling fine-grained, input-adaptive judgments while maintaining consistency across languages. Experiments demonstrate that UniRRM-8B and UniRRM-14B achieve performance close to the state-of-the-art for models of comparable size across multiple benchmarks, and are effective for unseen evaluation paradigms. In addition, ablation studies validate the reliability and effectiveness of UniRRM. 

---
# From Narrative to Auditable Forecasts: A Structured Scaffold for Agentic Forecasting 

**Authors**: Yuanpu Cao, Yongkang Du, Yurui Chang, Lu Lin, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.05905)  

**Abstract**: LLM agents are increasingly used for live forecasting, where they retrieve up-to-date information and produce estimates for unresolved future events. However, current agentic forecasting often relies on implicit narrative aggregation: agents collect evidence, discuss it in prose, and often assign a probability without an explicit update path from evidence to forecast. This limits both forecasting accuracy and auditability. We propose AuditForecast, an agentic scaffold for structured probabilistic forecasting. AuditForecast first anchors the forecast with a suitable quantitative baseline model, uses model-guided data retrieval to derive a base probability, and then applies situational factor updates outside the model's scope through mechanical aggregation in odds space. This turns forecasting from a prose-based judgment into a structured process with explicit intermediate objects. Across multiple live forecasting benchmarks, AuditForecast improves forecasting accuracy and calibration relative to strong agentic baselines, surpasses market-implied references in several settings, and outperforms substantially more expensive deep-research agents while remaining Pareto-dominant in the cost--accuracy tradeoff. Beyond performance gains, AuditForecast produces an auditable forecasting report that makes forecast construction explicit and supports systematic post hoc analysis. 

---
# AlignDiff: Exploiting Model-Intrinsic Information for Better Preference Data Selection 

**Authors**: Peng Lai, He Zhu, Zhiwen Ruan, Dongdong Zhang, Yun Chen, Peng Li, Furu Wei, Yang Liu, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.05899)  

**Abstract**: Aligning large language models with human preferences remains a challenge, primarily due to the critical role of preference data quality in effective alignment. Existing datasets are frequently plagued by inherent noise and distribution shifts, which inherently limit model performance. To bridge this gap, we propose AlignDiff, a preference data filtering framework driven by intrinsic model signals. AlignDiff first identifies samples with clear preferences using both positive and inverse signals, then prioritizes the more challenging samples based on the average negative log-likelihood gap, encouraging the model to learn richer information from them. AlignDiff is evaluated on two widely used model families (LLaMA and Qwen) and three benchmarks widely adopted in the alignment community (AlpacaEval 2.0, Arena-Hard, and MT-Bench). Across all settings, it consistently outperforms seven strong baselines. We conduct comprehensive ablation studies to validate the effectiveness of AlignDiff, and further show that difficulty-based curriculum learning improves model performance. 

---
# What if LLMs Ate Their Words: Causal History Effects in Multi-Turn Interaction 

**Authors**: Jinnan Li, Zheren Fu, Yue Wang, Jinzhe Li, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.05882)  

**Abstract**: Multi-turn interaction creates a feedback process in which an LLM's previous responses become context for later behavior. Prior work shows substantial multi-turn degradation and that assistant-generated history can affect later behavior. However, it remains unclear how these effects manifest across models, tasks, turns, and inside a model. We study these gaps across six task families and five models. Degradation from fully specified single-turn input (FULL) to progressively revealed multi-turn interaction (SHARDED) is clearly task- and model-dependent, and stronger one-shot performance does not imply greater interaction robustness. We then retrospectively analyze completed SHARDED conversations by replaying the user messages already observed in each trajectory while editing only assistant-generated history. Replacing prior assistant responses with neutral content (termed neutralization) changes downstream min-max normalized performance by +.027 across 2,973 trajectories. On a prespecified length-controlled subset, short and length-matched neutralization yield nearly identical effects (+.069 versus +.068), showing that simple context shortening is insufficient to explain the effect of history editing. Turn Surgery further intervenes on one assistant turn at a time. Among 237 selected degraded trajectories, 63.7% contain at least one beneficial intervention, while most tested positions remain unchanged; for binary tasks, 48.4% admit a fail-to-success reversal. An open-weight case study links behaviorally consequential history changes to measurable downstream state differences, but finds task-dependent rather than universal internal signatures. Overall, assistant-generated history has active but selective effects on multi-turn performance, motivating selective rather than uniform history management. 

---
# SinoGlyphBench: A Diagnostic Benchmark for Chinese Glyph-Level Obfuscation in Language-Model Moderation 

**Authors**: Yifan Wang, Zimu Wang, Suliu Qin, Changyu Zeng, Tong Chen, Siqi Chen, Yijie Lin, Lingyu Jiang, Jionglong Su, Yushan Pan, Haiyang Zhang, Wei Wang, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.05843)  

**Abstract**: Glyph-level obfuscation can leave harmful Chinese content readable to humans while degrading automated moderation. We introduce SinoGlyphBench, a diagnostic benchmark that identifies label-critical semantic anchors and creates matched original and glyph-obfuscated inputs in text and image modalities. By perturbing anchors, background context, or both, this design distinguishes corruption of moderation-relevant evidence from general surface variation. Across 176,916 paired evaluations of 12 LLMs and MLLMs, obfuscation increases harmful false-negative and false-positive rates by 6.1 and 4.7 percentage points, respectively, and reduces four-way accuracy by 5.0 points. Models retain 75.7% of the decisions that were correct on the matched original inputs. Full-scope perturbations cause the largest degradation, anchor-only perturbations are more damaging than background-only perturbations, and cross-script substitution is particularly difficult in the text modality. Analysis of structured outputs identifies observable mismatches in visible-form reading, intended-message recovery, and final safety judgment. The evaluated models, therefore, remain brittle to Chinese content written with non-canonical glyphs. Resources are available at this https URL. 

---
# CONDUIT: A Unified Residual-Stream Restoration Framework for KV Cache Reuse in Vision-Language Models 

**Authors**: Pengan Chen, Kaisheng Zheng, Liang Hong, Lixia Yi, Jiyue Jiang, Jiayang Chen, Yixuan Wang, Yimin Fan, Xinyuan Liu, Jiayi Li, Zhanqiu Zhang, Yiwen Guo, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.05821)  

**Abstract**: Vision-language models (VLMs) often answer new questions about recurring visual content, where reusing the key-value (KV) cache can avoid re-encoding expensive visual prefixes. Exact-prefix reuse, however, fails when the same visual content appears under a changed prefix. Selective recomputation can recover quality under a small visual-token budget, but only when the right stale tokens are refreshed. Raw-attention selection can waste budget on high-attention tokens with small value-norm proxy scores and on query-irrelevant images. To address these failure modes, we propose CONDUIT, a training-free refresh policy that unifies single- and multi-image reuse as residual-stream restoration. Building on norm-weighted attention, CONDUIT ranks cached visual tokens using cached-key query attention and an accessible pre-output cached-value-norm proxy, then applies empirical image-level relevance amplification before one global selection. With one image, the coefficient is one and the rule reduces to intra-image token selection. The method preserves model architecture and weights, adding only a single query-conditioned scoring pass at inference. At a 10% refresh budget, CONDUIT achieves 97.0-99.5% of the corresponding full-prefill five-dataset average across three VLM backbones and leads budgeted methods on average; on the MMLongBench-Doc latency subset, it uses 13.5% of full-prefill FLOPs and achieves a 2.99x time-to-first-token speedup. 

---
# AtomCite: Verification and Correction of Supplied Page-Level Citations in Multi-Page Documents 

**Authors**: Chen Qian, Yimeng Wang, Yu Chen, Lingfei Wu, Andreas Stathopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2609.05802)  

**Abstract**: Large language models answering questions over multi-page documents are expected to cite the supporting pages, yet supplied citations are sometimes inaccurate, and current evaluations score citations at generation time or against text passages: no existing benchmark evaluates whether a system can verify and correct a page-level citation already attached to an answer. We propose AtomCite, an agentic framework that parses an answer into claims, checks each claim against the image of its cited page, and applies a deterministic repair policy. To evaluate it, we introduce DocCite, to our knowledge the first benchmark for systems that verify and correct page-level citations in document images. Built on MP-DocVQA and DUDE, it combines 928 validated injected instances with 2,468 candidate natural errors harvested from frontier- and efficiency-tier models, of which a two-annotator audit confirms 1,909 as genuine errors. Primary labels are assigned deterministically, not by LLM judges, with the human audit as a separate validation layer. Across three model families (Gemini, Claude, and GPT), AtomCite reaches around 93% binary verification accuracy on the injected benchmark, significantly outperforming every OCR-only condition, including a compute-matched control, and exceeding every prior text-based baseline given the same OCR text. Its repair policy lifts citation precision on the injected mix from a constructed 34% to 87-90% while retaining over 90% of correct claims. AtomCite also transfers: with frozen prompts and zero training, it raises the hallucination-detection scores of two open 7-8B models on five public benchmarks above the same models prompted as direct judges. Finally, the audit shows that noise in automatic labels biases measured verifier accuracy and can reverse system rankings, so evaluations relying only on synthetic or automatic labels risk mismeasuring verification capability. 

---
# Dynamic Lagging for Simultaneous Translation 

**Authors**: Hieu Hoang, Amittai Axelrod  

**Link**: [PDF](https://arxiv.org/pdf/2609.05799)  

**Abstract**: In cascaded simultaneous speech translation, the machine translation (MT) system cannot control the read--write schedule of the upstream recognizer: it must decide, from a growing source prefix, how much target text to commit. We make a sentence-trained, decoder-only LLM prefix-aware by fine-tuning it on stable prefixes---the longest prefix that any translation up to the current partial source has shared with the model's own full-source output---mixed with full-sentence pairs, and prompt it through a single force-decode turn that carries the committed target forward as more source arrives, making the system flicker-free by construction. We fine-tune Qwen3-8B for EN to DE, JA, ZH, simulating the source stream with reference-transcript prefixes. Prefix finetuning preserves full-sentence quality while improving worst-position chunk quality, and it improves calibration of token-level commit confidence, reducing expected calibration error (ECE) on early source prefixes against a stable-prefix oracle. A single training-free threshold on that confidence is the most effective of the three latency controls we compare: it traces a continuous quality--latency frontier that outperforms the discrete wait-$k$ and target-suffix-deletion quality-latency tradeoff mechanisms. The effect holds well on FLEURS, WMT24++, and CoVoST~2 test sets, under both COMET and MetricX. 

---
# Recall Is Not Protection: Evaluating Safety Monitors Against Model Compliance 

**Authors**: Sripad Karne  

**Link**: [PDF](https://arxiv.org/pdf/2609.05797)  

**Abstract**: Safety monitors screen prompts sent to deployed language models, flagging harmful requests so they are never answered. They are evaluated by recall against harmfulness labels, but a catch only prevents harm if the model would otherwise have complied. We measure the difference directly: we sample repeated responses from the target model, call a harmful prompt \emph{elicitable} if the model complies at least once, and report monitor recall separately on elicitable and non-elicitable prompts. Across six monitor configurations and three model families, spanning activation probes, fine-tuned text guards, and a 120B policy-conditioned reasoning classifier, recall on elicitable prompts falls 0.22 to 0.38 below recall on non-elicitable prompts at a fixed false positive rate. The prompts a monitor misses are 2.8 to 5.6 times more likely to be complied with than the prompts it catches. The gap replicates across three model families and appears also in text-only monitors entirely independent of the target model. This suggests that standard recall may overstate the protection monitors provide in practice, and that monitors should be evaluated against what their models will actually answer. 

---
# CrisisKD: Five-Stage Knowledge Distillation for Aspect-Level Sentiment and Emotion Analysis in Crisis Discourse 

**Authors**: Marko Haralović, Onat Akca, Salih Eren Yücetürk, Minsi Li, Mariët Theune  

**Link**: [PDF](https://arxiv.org/pdf/2609.05757)  

**Abstract**: Identifying the target of emotional words or phrases in crisis situations, especially health-related ones, is important for understanding public concerns across cultural and linguistic contexts. We propose CrisisKD, a five-stage teacher--student knowledge distillation framework for aspect-level sentiment and emotion analysis on unannotated social media data. A teacher LLM generates aspect-level labels and reasoning traces that supervise a smaller student model across aspect extraction, syntactic parsing, opinion extraction, sentiment classification, and emotion classification. Using this framework, we construct and release a dataset containing 50,615 aspect-level labels, together with the annotation and fine-tuning scripts as open-source resources. The resulting student supports end-to-end ABSA and emotion detection at substantially lower inference cost than the teacher. On a manually annotated 500-tweet gold set, the 5-task Qwen2.5-7B student improves over the untuned model by 7.9 F1 points on aspect extraction, 17.0 points on emotion accuracy, and 6.5 points on sentiment accuracy. On the external ABEA benchmark, CrisisKD improves the same-model Qwen2.5-7B ICL baseline by 2.8 F1 points on ATE and 3.8 F1 points on joint ATE+AEC. 

---
# Some Tokens Behave like Magnets: Revealing Linguistic Organization in the Layers of Language Models 

**Authors**: Andrew Liu, Devan Srinivasan, Gerald Penn  

**Link**: [PDF](https://arxiv.org/pdf/2609.05743)  

**Abstract**: We identify a special group of token vectors inside large language models (LLMs), which we term magnetic vectors, that organize the surrounding tokens by either attracting or repelling them. Particularly, tokens pointing the same way as an attracting magnet are elongated; tokens pointing the same way as a repelling magnet are compressed. Just as physical magnets pull or push away the iron filings around them, these vectors organize their surroundings through two opposing polarities. Moreover, we identify a statistically significant pattern in linguistic category where function words consistently act as repelling magnets in early layers, and we also find magnets consistently reorganize their polarities in unique ways deeper in the model. In a further case study we find this observation may unveil a deliberate, layer-wise organization in how LLMs process language.
This pattern is consistent across different LLM architectures, sizes, and layer configurations. It is also causally relevant. When the LLM is fine-tuned for a downstream task, the task-functional tokens emerge as magnets. E.g., in question answering, the answer-span tokens become uniquely repelling magnets in the final layer, geometrically carving the answer out of the surrounding context. Furthermore, removing early-layer repelling magnets devastates syntactic tasks (POS tagging accuracy drops from 91% to below 10%) while sparing semantic ones, and removing late-layer attracting magnets does the reverse. We believe this phenomenon warrants further investigation, as it opens the first probe-free path to understanding how language models geometrically organize linguistic computation across their layers. 

---
# MedWER: A Reproducible, Model-Free Evaluation Protocol for Medical Speech Recognition 

**Authors**: Justin Behling  

**Link**: [PDF](https://arxiv.org/pdf/2609.05728)  

**Abstract**: Overall word error rate hides clinically critical errors: a transcript can be 95% correct and still swap one drug for another. The usual fix weights errors on medical entities, and almost always depends on an evaluation-time named-entity recognition (NER) model or cloud API, which makes the metric's denominator a versioned black box. We present MedWER, an evaluation protocol and open-source tool for medical ASR whose denominator is a fixed, license-clean term list: 19,373 drug, diagnosis, symptom, and injury-mechanism entries projected from public sources. The protocol couples a pinned text normalizer with a phrase-aware term-restricted WER, the MedWER, so the only versioned component is a normalizer dependency held at an exact release and checked against committed golden fixtures. Coverage is validated against an independent provincial drug-benefit file the list was not built from; the matching heuristic is calibrated against ground-truth entity spans. Baselines for Moonshine~base, Whisper~this http URL, and MedASR on two open benchmarks are scored with the released tool and reported with 95% confidence intervals from resampled per-utterance scores. 

---
# Intra-Prompt Parallel Decoding for Common-Context Question Answering 

**Authors**: Theodore Glavas, Nikhita Vedula, Dushyanta Dhyani, Antonios Valkanas, Yilun Zhu, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2609.05707)  

**Abstract**: In common-context question answering (CCQA) tasks, multiple input questions share a common context to base their answers from. However, Large Language Models typically generate each answer using an independent prompt. While existing batching and caching techniques help improve parallelism and reduce repeated computations, the separation of questions across prompts limits the achievable speedup, as modern GPUs are underutilized due to a memory bottleneck during attention. We present Intra-Prompt Parallel Decoding (IPPD), a novel inference method that answers multiple common-context questions in parallel within a single prompt. IPPD directly addresses the bottleneck by efficiently sharing both memory and computation during the attention process, as the next token for every question is decoded in a single inference step. IPPD uses virtual position IDs and attention mask manipulation to generate the same output as standard prompting without requiring fine-tuning or any changes to the LLM architecture. Since all parallelism occurs within a prompt, IPPD is fully compatible with batched inference, even when each prompt features a different context. Our experiments show that IPPD delivers up to 7X the effective throughput as standard decoding without quality degradation, and outperforms prefix caching with PagedAttention in most settings. 

---
# A Rubric-Guided Large Language Model Solution for Opioid Use Disorder Computable Phenotyping 

**Authors**: Mengxian Lyu, Paredes Pardo, Cheng Peng, Ziyi Chen, Mengyuan Zhang, Jieting Li Lu, Gary M Reisfield, William M Greene, Jenny Lo-Ciganic, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.05682)  

**Abstract**: Opioid use disorder (OUD) remains a public health crisis in the United States, yet it is difficult to identify from electronic health records (EHRs) because missing diagnosis codes and supporting evidence are buried in clinical narratives. Accurate OUD identification is critical to support interventions and improve health outcomes. This study developed a rubric-guided large language model (LLM) that incorporated Optimization by PROmpting (OPRO) for OUD computable phenotyping (CP). The framework used an 18-item, expert-identified rubric to instruct LLMs to automatically extract critical text with supporting evidence to determine OUD flags. Two UF Health physicians (GMR and WMG) chart-reviewed 253 patients, including 68 OUD-positive cases. Our LLM-based computable phenotype (CP) achieved the best F1 score of 0.774 and an AUROC of 0.934, outperforming the machine learning-based CP using EHR and natural language processing-extracted variables, and zero-shot LLMs by relative F1 improvements of 12.8% and 44.4%, respectively. The proposed LLM-based CP could link LLM-extracted evidence to OUD phenotyping for better explainability. 

---
# Who Maintains Agent Skills? A Longitudinal Study of Human-Governed, AI-Assisted Skill Maintenance 

**Authors**: Chen Shen, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2609.05677)  

**Abstract**: Lifelong LLM agents increasingly rely on external skill artifacts as one element for preserving and reusing capabilities over time. These skills (usually portable Markdown files such as this http URL) describe when and how to apply a capability and must be corrected, expanded, and consolidated as tools and usage patterns shift over deployment. Recent work seeks to automate skill curation, but it largely evaluates against automated baselines and treats human maintenance as an unmeasured bottleneck. We study that missing process directly. We mine the full commit histories of five public AI-skill repositories, a purposive sample of AI-tooling organizations, covering 873 commits, 143 skill files, and 254 substantive post-creation edits from October 2025 to June 2026. We code each edit with pre-registered governance, operation, and trigger-evidence codebooks. Three findings emerge. First, every substantive edit is authored or merged through a named human account, while 62% carry an AI co-author trailer, with large repository-level variation. Second, these edits are genuine curation: an audited sample shows that most change skill content, and the coded operations are dominated by additions and corrections. Third, a pre-registered rule-likeness axis fails its reliability gate; reliably coding rule-likeness from commit artifacts remains an open measurement problem. We release the corpus, codebooks, mining scripts, and a replay protocol for automated skill curators. For self-evolving agents, public skill maintenance currently looks less like an autonomous pipeline than a human-governed, AI-assisted loop that future curators must measure against and operate within. 

---
# Better Together: Complementary Query Rewriting Under a Strong RAG Baseline 

**Authors**: Sara Shanian, Xiaoqin Yi, Pavlo Ruban, Kurt MacDonald  

**Link**: [PDF](https://arxiv.org/pdf/2609.05637)  

**Abstract**: A popular way to improve Retrieval-Augmented Generation (RAG) is to rewrite the user's question into several variants and search with all of them. We test whether this actually helps once the underlying search is already strong. Under one fixed, competitive pipeline (BGE dense retrieval, cross-encoder reranking, and MMR diversification), we compare four query-rewriting strategies (S1-S4) against two strong LLM baselines (HyDE, Query2Doc) on three datasets (HotpotQA, AmbigNQ, and the 512K-document EnterpriseRAG-Bench) over three seeds with paired-bootstrap significance tests. Our headline result is that rewriting alone is at best competitive with a strong baseline, but combining methods yields outsized gains because different strategies fail on different questions. A post-hoc union of four methods (S1+S3+S4+HyDE) improves HIT@10 over the baseline by +12.5 points on enterprise data (51.70 vs 39.22), and a five-method union reaches 52.98 (+13.8). Budget-matched controls capture only ~40% of this gain, confirming that complementarity, not retrieval budget, is the primary driver. On HotpotQA the union adds +1.6 to +1.8 points (p<0.001), saturating the all-method oracle; on AmbigNQ the same fusion hurts (-2.4 below the best solo, p<0.001), and we analyze when and why. Because rewriting is expensive, we evaluate in simulation a confidence-gated router that runs rewriting only when the baseline's own top-1 score is low. It captures about half of the enterprise full-merge gain (+4.3 HIT@10) while paying rewriting cost on <40% of queries, and automatically declines to rewrite on AmbigNQ. A downstream answer-quality evaluation confirms the router improves F1 by +1.92 (p<0.01) at roughly 40% of the expansion cost. In short: treat query rewriting as a complementary coverage source, applied through cost-aware routing, not as a standalone replacement for a strong baseline. 

---
# TamilEOT: A Dataset and Model for Semantic End-of-Turn Detection in Tamil Telephone Speech 

**Authors**: Santhoshkumar V  

**Link**: [PDF](https://arxiv.org/pdf/2609.05631)  

**Abstract**: A voice agent has to decide, at every pause, whether the user has finished speaking. Without a model of the language that decision falls back to a fixed silence timeout: set it short and the agent interrupts, set it long and every turn pays the full wait. Open semantic end-of-turn detectors exist, but to our knowledge none covers a South Indian language. We release TamilEOT: 18,485 labelled turn boundaries cut from 116 real Tamil telephone conversations, and two audio-only detectors fine-tuned from Smart Turn v3. On a held-out split of 4,168 clips from 30 unseen calls, accuracy rises from 70.30% zero-shot to 83.71% (8.7 MB) and 86.13% (21 MB); ROC-AUC rises from 0.751 to 0.921. Both models run in under 150 ms single-threaded on a laptop CPU. We also report what building it cost. Rule-derived labels, checked against a blind human listening pass, were right 95.9% of the time on the positive class and 44.4% on the negative class, which is below chance, because the rule answered a different question than the model is asked. Replacing them with an audio-LLM labeller measured at 97.5% human agreement cost US$5.69. Of every training lever we measured, only encoder capacity moved the result; three runs at identical config and seed span 0.87 accuracy points, which is the floor below which none of our other deltas mean anything. Replaying the same labelled boundaries through the production VAD and streaming adapter costs a further 2.60 points, and 7.8% of boundaries are never surfaced to the model at all. Data, weights, code and every negative result are public. 

---
# When Agent Governance Helps 

**Authors**: Michael Ray Johnson, Linda Naimi  

**Link**: [PDF](https://arxiv.org/pdf/2609.05531)  

**Abstract**: No specification says how a governed autotelic AI agent organization, where agents pursue self-generated goals inside guardrails, should be designed and evaluated. We answer in two parts. First, we synthesize the Governed Autotelic Multi-Agent Product Organization (GAMPO) framework from a document-based qualitative evidence synthesis of 321 sources, integrating agency, agile, platform, and governance theory into a runnable specification. Second, we probe a prompt-layer instantiation of GAMPO on CHI-Bench, a long-horizon healthcare benchmark, across open and frontier models. The result is a boundary condition: governance benefit is gated by a model's spare capacity and is domain- and model-specific. On capacity-constrained open models the full procedure yields no reliable benefit, whereas a single "verify your writes" sentence doubles task success (pass@1 2/20 to 4/20). At the frontier the same scaffold lifts prior-authorization 24% to 40% but nets zero on another model, a gap traced to a stable recommendation-override disposition. A second result refines the first: replacing the generic procedure with an answer-blind, per-task definition-of-done, keyed only to the case's own policy and published standards, never the hidden key, raises prior-authorization to 84% under best-of-five self-consistency (68% single-attempt, confirmed by a held-out board) and utilization-management to 44%, while care-management meets a subjective content-quality wall. The contribution is a named, auditable framework and capability-gated evidence that governance should be sized to spare capacity, and that at the frontier a case-grounded specification beats a uniform procedure. Findings are exploratory: partial instantiation, small per-cell samples (n = 5-25), and single trials. 

---
# Learning Length-Extrapolatable Recurrent Models 

**Authors**: Hanwen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.09157)  

**Abstract**: Recurrent models provide a natural path to long-context modeling, yet models trained with backpropagation through time (BPTT) often fail beyond their training horizon. Classical analyses emphasize gradients that vanish or explode along temporal paths. However, dense per-token losses can still train a shared recurrent rule despite severe decay, showing that decay alone does not determine whether learning fails. We instead study state credit: the signal through which future losses reach earlier recurrent states before contributing to parameter updates. Accordingly, we intervene directly on state credit and propose Credit Stabilization through Time (CST). During backward propagation, CST locally rescales the state-credit signal to stabilize its norm without rotating the component being corrected, while leaving the forward computation unchanged. Because controlled synthetic tasks and real data exhibit different credit dynamics, we specialize CST to each regime. In both settings, CST improves performance beyond the training horizon, with gains observed at up to 128x the training length. 

---
# Procedural Graphs: Self-Evolving Execution Structures for LLM Agents 

**Authors**: Yuxing Lu, Yicheng Chen, Shanchan Wu, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2609.09153)  

**Abstract**: Large language models are increasingly deployed as agents that plan over long horizons and act through external tools. Most agents select actions through unconstrained generation over an accumulating history, leaving implicit the procedural knowledge of what to do, in what order, and under which conditions. As trajectories lengthen, agents can lose track of their objectives, invoke tools out of order, and repeat unproductive actions. We introduce the Procedural Graph: just as a knowledge graph organizes factual knowledge into (entity, relation, entity) triplets for what-is questions, a Procedural Graph organizes procedural knowledge into (procedure, relation, procedure) triplets for what-to-do questions. At each decision step, the framework localizes the agent's active node, and a guidance model translates the surrounding subgraph into step-level situational guidance that biases the solver's next action without dictating it. The graph is self-evolving: an LLM refiner contrasts failed trajectories with successful ones and edits the graph's topology and attributes, committing edits that preserve or improve held-out validation performance while retaining rejected ones to discourage repetition. Starting from a minimal skeleton, the loop builds graphs that match or surpass hand-designed ones. It can also repair a flawed expert prior. Across multiple datasets, task types, and LLMs, the Procedural Graph delivers consistent gains over memory-based baselines, and self-evolution further improves performance without manual engineering. 

---
# Copying explains the collective behavior of AI agents in the wild 

**Authors**: Giordano De Marzo, Nicola Albore, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2609.09150)  

**Abstract**: In June 2026, thousands of AI agents found that a small public wiki would accept edits from inside their sandboxes, and started using it to help one another pass a timed test. Each agent lived for about an hour and remembered nothing afterwards. Nobody asked them to cooperate, and the wiki had not been built for them. The complete record of what they wrote is public, and it is unusually informative, because it preserves not only what each agent wrote but what that agent could see before writing. We use it to follow the three decisions an agent had to make on arrival: where to write, what to call itself, and how to word its message. One rule governs all three. An agent takes an option with a probability close to the share of that option in what it can see, and the share that matters is the one on the page in front of it, then the one in the stream of recent edits, and only weakly anything older. Three minimal copying models, one per decision and with a single free parameter each, reproduce the heavy-tailed distribution of how many agents met on a page, the frequency of the pieces from which the agents built their names, and the patchwork of pages that are internally consistent and different from one another. Copying whatever the environment happens to show is enough to produce most of the collective structure of this population. It is also what makes such a population easy to steer, since whoever writes first, or writes while the others are quiet, sets the convention for everyone who comes later. 

---
# Studying Image Tokenizers as Visual Languages in Unified Multimodal Models 

**Authors**: Siting Li, Zhengyang Wang, Simon Shaolei Du, Xi Chen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.09143)  

**Abstract**: Image tokenizers define the ``visual language'' of unified multimodal models, yet are commonly studied through isolated metrics or generation-/understanding-only evaluations. These evaluations do not fully capture how visual tokens behave when modeled jointly with text. We build a controlled pure-autoregressive testbed and track task-specific validation losses during multimodal continual pretraining across text, image, text-to-image (T2I), and image-to-text (I2T) prediction. We examine how these losses scale and relate to downstream performance, then use them to study multimodal learnability---how well image and text tokens are jointly modeled---and tokenizer design. We find that (1) losses should be analyzed by task, since they exhibit distinct scaling behavior and rank tokenizers differently. (2) The loss--performance relationship depends on the predicted token space: for a fixed tokenizer, T2I and I2T losses correlate with generation quality, but across tokenizers, the T2I loss--performance relationship shifts with the image-token space, whereas I2T loss, computed over a shared text vocabulary, provides a more consistent signal. I2T loss also correlates with both generation and visual understanding performance after supervised finetuning. Using losses as a lens, we show that (3) better reconstruction does not necessarily yield lower task-specific losses or stronger downstream performance, and that (4) image tokenizer choice can affect text modeling under joint optimization. As case studies, we revisit three tokenizer design axes---the discriminator, semantic supervision, and vocabulary size---to examine their effects on joint modeling and downstream performance. Together, our testbed offers a complementary perspective on image tokenizers as visual languages, highlighting their interplay with text in joint multimodal training. 

---
# A Data-Driven Framework for Identifying and Prioritizing RPA Opportunities in Healthcare Processes 

**Authors**: Maria Alejandra Gomez, Juan Manuel Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2609.09137)  

**Abstract**: Robotic Process Automation (RPA) is widely used to reduce administrative burden in United States hospitals, yet an estimated 30-50% of RPA initiatives underperform because processes are selected informally, without a repeatable method to catalogue candidates, prioritize them, match each to an automation tier -- a Python bot, an open-source orchestrator such as n8n, or an enterprise platform such as UiPath -- and forecast financial return before committing resources. We propose a four-module, data-driven framework unifying these decisions: a Process Taxonomy of twenty recurring hospital processes across five value streams; a Prioritization module deriving an Automation Suitability Index from an Analytic Hierarchy Process matrix with an explicit consistency check; a Tool-Tier Selection module recommending the least-cost technology sufficient for a process complexity, integration, and compliance profile; and a Return-on-Investment module quantifying labor savings, error-cost avoidance, payback, and net present value. Applied to a synthetic portfolio spanning all twenty processes, plus a reference data-flow architecture linking it to hospital EHR/payer/ERP systems: 12 of 20 clear the prioritization threshold; the ranking is robust to +/-20% weight perturbation (Spearman correlation 0.83, top-5 set preserved 97.7%, 2,000 Monte Carlo trials); an Automation Risk Index flags four qualifying processes as Critical risk; a budget-constrained portfolio optimization shows diminishing marginal NPV as spend scales from $400K to $1.03M; and a second Monte Carlo analysis shows portfolio NPV stays positive at its 5th percentile. The framework is a conceptual synthesis of the literature rather than an instrument calibrated on primary hospital data; we discuss HIPAA governance and a research agenda for empirical validation. A supplementary Python implementation accompanies the paper. 

---
# Entropy-Regularized Rank-Masked Policy Optimization for Test-Time Reinforcement Learning in Code Generation 

**Authors**: Jiacheng Xu, Feng Chen, Xiuneng Xu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2609.09135)  

**Abstract**: Existing methods for test-time reinforcement learning (TTRL) derive rewards from answer-level self-voting on unlabeled test-time tasks with canonical answers, but this breaks down for code generation because programs cannot be compared by surface form and therefore do not directly provide a usable training signal. To make TTRL applicable to code generation, we propose probe-driven TTRL, which constructs output-free probe inputs from the problem statement, executes candidate programs on these probes, and defines a Probe Consensus Reward (PCR) from the resulting behavioral agreement. PCR provides a behavioral training signal for open-vocabulary programs, but it is not a fully reliable verifier and remains susceptible to reward hacking through spurious consensus. We therefore introduce Entropy-Regularized Rank-Masked Policy Optimization (ERPO), which converts low PCR into conservative negative updates through rank masking and controls policy drift with an entropy ceiling. On coding benchmarks, ERPO substantially improves pass@1 and pass@k in both in-domain adaptation and zero-shot transfer. 

---
# ExecCritic: Learn to Test, Test to Improve for Coding Agents 

**Authors**: Leitian Tao, Baolin Peng, Haorui Wang, Hang Wang, Hao Cheng, Wenlin Yao, Qianhui Wu, Tao Ge, Sharon Li, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2609.09133)  

**Abstract**: Execution feedback can guide coding agents toward correct repository repairs, but only when the tests capture the behavior requested by the issue. Agent-generated tests can encode incomplete or incorrect behavioral targets; when the same trajectory writes both the patch and the test, their errors can agree and create false confidence. We introduce ExecCritic, combining a test--verify--revise scaffold with a role-specific reinforcement learning recipe for training agents within it. The scaffold separates test construction from source-code repair: a Test agent independently generates repository-native tests, a fail-closed harness qualifies and freezes them, and a Repair agent revises source code from their execution feedback without changing the tests. Both roles use Qwen-3.5-35B-A3B as the backbone and are trained separately. In Learn to Test, the Test agent learns to produce behaviorally valid tests that distinguish correct from incorrect patches. In Test to Improve, the Repair agent learns both direct task resolution and feedback-guided revision. On SWE-bench Verified, test quality determines whether feedback helps: holding the base Repair agent fixed, tests from the base Test agent reduce resolved rate from a no-test baseline of 61.2% to 57.3%, whereas tests from GPT-5.6-sol raise it to 65.3%. Role-specific post-training raises the Qwen Test agent's Base-to-Gold success from 22.2% to 62.2%; composing the two post-trained Qwen agents reaches 72.6%, an 11.4-point gain over the original no-test baseline without stronger-model or Oracle feedback at evaluation time. Code is publicly available at this https URL. 

---
# SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research? 

**Authors**: Yuqiao Tan, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.09113)  

**Abstract**: While research on recursive self-improvement (RSI) has predominantly automated model training pipelines, reliable autonomous development demands a missing pillar: post-hoc monitoring and auditing to understand what models learn and ensure safe alignment. Mechanistic interpretability tools are essential to bridge this gap, among which Sparse Autoencoders (SAEs) serve as a cornerstone by isolating interpretable features for model inspection and steering. In this paper, we introduce SAEScientist-Bench to evaluate whether AI agents can act as scientists utilizing SAE tools for autonomous mechanistic discovery. Given a target concept, an agent designs contrastive probes and navigates a Gemma Scope dictionary of 131K+ features in Gemma-2-9B-IT to discover the optimal feature, evaluated against curated expert reference features anchored on Neuronpedia across activation rank, concept selectivity on contrastive texts, and causal steering. Across 10 agent configurations and 20 tasks, frontier agents demonstrate genuine discovery capabilities and lead different evaluation dimensions, but remain well behind the expert baseline, approaching expert levels on separating target concepts from contrastive controls while lagging substantially in causal generation steering. Further analysis reveals that although agents can design contrasts to rule out spurious candidates, they frequently misinterpret experimental measurements. These results establish experimental model understanding as a measurable capability for closed-loop autonomous AI R&D. Our code is available at this https URL. 

---
# Answer-Distribution Trajectories: A Stochastic-Dynamics View of LLM Reasoning 

**Authors**: Mar Gonzàlez I Català, Haitz Sáez de Ocáriz Borde, Davide Murari, Carola-Bibiane Schönlieb, Pietro Liò, George Montañez  

**Link**: [PDF](https://arxiv.org/pdf/2609.09030)  

**Abstract**: Chain-of-thought reasoning provides a structured computation between a model's input and final answer. Yet it is often evaluated through endpoint accuracy, which ignores the path taken to reach that answer. An emerging line of work addresses this limitation using entropy profiles, which track how uncertainty evolves over the reasoning process but do not reveal which competing hypotheses account for that uncertainty. We introduce answer-distribution trajectories, a stochastic-dynamics-inspired representation that tracks the model's full predictive distribution over answers as reasoning unfolds. As a strictly finer representation than endpoint and entropy summaries, answer-distribution trajectories enable us to characterize a trace through a dynamical reasoning profile spanning exploration, revision, motion, and commitment, and to distinguish different dynamical mechanisms of reasoning success and failure. Across sixteen open-weight language models and four reasoning benchmarks, we show that traces with the same endpoint and similar entropy profiles can exhibit substantially different reasoning dynamics. We further find substantial variation in these dynamics both within and across models and tasks, with different objectives favoring different dynamical profiles. Additionally, we show that training and inference choices systematically reshape these profiles. Our results suggest that answer-distribution trajectories provide a rich framework for analysing and evaluating the dynamics of LLM reasoning. 

---
# Good Pretraining, Bad SFT: Checkpoint Quality Across the Training Stack 

**Authors**: Sohir Maskey, Philipp Scholl, Jonas Knupp, Pit Neitemeier, Sascha Wirges  

**Link**: [PDF](https://arxiv.org/pdf/2609.08966)  

**Abstract**: Language-model checkpoints are commonly selected by pretraining loss or benchmark scores, assuming that the highest-scoring checkpoint will remain the best starting point for subsequent training. We show that this assumption can fail in a full 30B mixture-of-experts training pipeline. The checkpoints that perform better after the full downstream training stack also have higher solution density, i.e., retain downstream performance under local weight perturbations. 

---
# PlannerForge: LLM Agents for Scenario-Based Testing of Motion Planners in Autonomous Driving 

**Authors**: Yuan Gao, Sebastian Müller, Mattia Piccinini, Marc Kaufeld, Yuchen Zhang, Finn Rasmus Schäfer, Qunying Song, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2609.08965)  

**Abstract**: Ensuring the safety of autonomous driving is a critical challenge. Scenario-based testing is a systematic process used to validate Autonomous Driving Systems (ADSs), but it remains a fragmented modular pipeline in which scenario generation, retrieval, modification, ADS execution, and results analysis are performed by separate tools with little interaction. Large Language Model (LLM) agents have shown promise across ADS sub-systems such as perception, planning, and control. However, no prior work covers the whole scenario-based testing pipeline for ADSs with a unified LLM-agent framework. We present PlannerForge, an LLM-agent framework that extends all scenario-based testing stages (from Scenario Generation to ADS Assessment) and adds two further LLM-enhanced stages: ADS Enhancement and ADS Benchmarking. We evaluate PlannerForge with 10 off-the-shelf LLMs across all tasks (Generation, Selection, Modification, Module Routing, Planner Testing, and Enhancement) under 5 prompt conditions. Best-per-task scores range from 0.88 to 1.00, and open-source 20-35B backends match commercial APIs on most tasks. Open-source models such as Qwen3.6:35B match commercial APIs on three of the five tasks. Chaining the modules end-to-end retains 83% / 78% of seed queries (commercial / open). It outperforms Scenario Factory 2.0 (Finkeldei et al., 2025) on natural-language generation (193 vs. 144 executable of 200) and realises 92-96% of requested city, road and vehicle attributes. It outperforms BM25 (Robertson and Zaragoza, 2009) at rank 1 selection (92.0% vs. 67.5%) and From-Words-to-Collisions (Gao et al., 2025) on physically valid edits (>=94% vs. 31%). At N=400, cost-tuning lifts planner success from 50.4% to 70.2% and cuts collisions from 19.0% to 8.4%, without domain-specific fine-tuning. 

---
# AuK Technical Report: An Open-Source Foundational Model for Speech Generation and Editing 

**Authors**: Ziyang Ma, Zhikang Niu, Wenming Tu, Tianrui Wang, Ruiqi Yan, Junxi Liu, Yanru Huo, Nickk Huang, Yang Liu, Qicong Xie, Zeyu Xie, Hui Wang, Haitao Li, Zixuan Jiang, Yalin Li, Jie Fang, Yifan Duan, Zeyue Tian, Guangzheng Li, Haina Zhu, Shuyi Wang, Jinwen Wang, Mingyu Cui, Tian Tan, Auden, Sen Liang, Steve Yves, Shan Yang, Liefeng Bo, Zilong Zheng, Kai Yu, Eng-Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.08936)  

**Abstract**: We introduce AuK, an open-source foundational model that unifies speech generation and editing through a common interface of natural-language instructions and audio context. To support this broad capability set, we construct approximately 3.03 billion instruction--audio instances and 1.95 million hours of effective supervision across five task families: speech generation, content editing, enhancement and separation, paralinguistic editing, and acoustic editing. AuK combines a multimodal large language model for semantic conditioning, an VAE jointly trained on speech, general audio, and music for acoustic conditioning, and a hybrid rectified-flow Transformer that performs dual-stream MMDiT blocks followed by unified single-stream DiT blocks for generation. Training begins with generation-only warm-up and proceeds to joint generation--editing pre-training. We then apply complementary post-training strategies: human-feedback preference optimization for open-ended editing and reward-based reinforcement learning for speech generation. To reduce inference cost, we further distill the model with consistency initialization and task-routed Decoupled DMD. The resulting AuK-Flash performs 4-step inference without classifier-free guidance and achieves a 4.5 wall-clock speedup over the full model under matched conditions. Experiments demonstrate leading performance on zero-shot and instruction-controlled speech generation and general instruction-guided editing, while remaining competitive on signal-level restoration tasks. We release both the source code and model weights to support reproducibility and further research. 

---
# From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection 

**Authors**: Mengzhe Geng, Yujia Lu, Patrick Littell, Manuela Kunz, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.08899)  

**Abstract**: Speech deepfakes can mimic a speaker's voice convincingly enough to deceive listeners and automated systems. This has driven strong progress in speech deepfake detection, but most detectors still end with one score per utterance. That score is useful for ranking systems, yet it says little about why a borderline item should be trusted, deferred, or reviewed. Two utterances can fall in the same score band for different reasons, for example because passive and retrieval evidence disagree or because the keyed probe is unavailable. We ask whether the final decision can remain scalar without discarding that provenance. We answer this question with an auditable decision record that carries four aligned cues into a late calibration step: a passive detector score, a conditional keyed-probe score on a marked derivative, retrieval support, and a speaker-profile margin, together with explicit disagreement coordinates. On the 4,080-example ASVspoof 5 Track 1 matched subset, the fixed retrieval-augmented rule improves on retrieval-only evidence, from 15.84 percent to 11.91 percent EER, and late calibration over the full record reaches 8.43 percent EER. At a 33.75 percent review budget, the exposed cue union covers 82.85 percent of the calibrated model's errors. The best passive WavLM run still reaches 6.71 percent EER, so we do not present the decision record as a stronger standalone detector. Its contribution is to preserve the evidence behind each surfaced utterance while still producing one operating score for thresholding and review. 

---
# Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems 

**Authors**: Maximilian Schall, Sedigheh Eslami, Markus Krimmel, Antoine Chaffin, Louis Milliken, Bo Wang, Denis Bykov  

**Link**: [PDF](https://arxiv.org/pdf/2609.08887)  

**Abstract**: Evaluating first-stage retrievers in large-scale production RAG requires a benchmark that pairs a large-scale corpus with a large set of agent-reformulated search queries based on real user queries and their conversation threads, and that labels many relevant documents per query. No existing public benchmark evaluates this setting: large-scale collections typically provide only a small number of evaluation queries, whereas benchmarks with many queries generally contain only millions of documents. Moreover, most benchmarks assess human-written queries, while the first-stage retrievers in agentic RAG pipelines serve machine-written reformulations whose distribution differs from human search behavior. To overcome these evaluation gaps, we introduce Q2D-Web (Query2Doc-Web), a large-scale agentic retrieval benchmark consisting of a 190M-document web corpus and 70k agentic search queries in ten languages, reformulated from real-world user queries in production systems. Q2D-Web provides three sets of fixed relevance judgments: agent citations, production rankings, and a combined set that unions both signals and adds LLM-based judgments of unlabeled pooled documents to reduce false negatives. We benchmark 13 retrievers including lexical, dense, and late-interaction models and find that their relative ordering is largely insensitive to the choice of judgment set, while diverging substantially across topical domains, query languages, and query types. To enable fast evaluation, we also study subcorpus sampling as an approximation to full-corpus evaluations. Retaining a third of the corpus, selected by reciprocal rank fusion over pooled retriever runs, preserves the full-corpus model ranking under the combined judgments while raising absolute Recall@1000 only by 3 to 7 points. The public leaderboard is accessible under: this https URL 

---
# A Closed-Form Estimator and Diagnostic Battery for Anchor-Judge Error Correlation, Under a Single-Common-Factor Model 

**Authors**: Veerendra Kumar Sunkavalli  

**Link**: [PDF](https://arxiv.org/pdf/2609.08826)  

**Abstract**: When an external reference set (an anchor) is used to decompose an LLM-judge panel's error into a quality signal and a shared common-mode error, standard practice assumes the anchor is uncontaminated: its error uncorrelated with the judges' shared error. We study when that assumption can be dropped and replaced by an estimate. Under a single-common-factor model, >=2 judges and >=2 anchors point-identify the quality variance, the common-mode variance, and each anchor's contamination correlation rho_k in closed form, with an exact per-anchor-pair failure boundary; a designated clean-anchor estimator, by contrast, reports a contaminated companion anchor as fully clean once its trusted anchor is itself contaminated. Because the single-common-factor assumption is itself untestable, the estimator ships gated behind a calibrated diagnostic battery (judge-covariance dispersion; over-identification; a family-block test from judge metadata, with a family-blocked estimator that removes family-level shared-residual bias exactly), bootstrap confidence intervals with measured coverage, and a weak-identification screen. A proposition maps which violations bias rho_k, in which direction, and which evade detection. For ordinal scores we show an identification hierarchy: with all variables ordinal, rho_k is not identified at any number of anchors; with ordinal judges and >=3 continuous anchors it is, and we give an estimator for that case. On real data the validation is asymmetric, and we say so plainly: the diagnostics are validated in the rejecting direction (both real panels we test are correctly rejected by the model-adequacy pre-test), while the estimator is validated in simulation and stress-tested semi-synthetically under oracle calibration; no real panel has yet passed the pre-test, and the pre-test exists precisely to say so. All results replay offline from shipped, checksummed artifacts. 

---
# Eliciting Weak-to-Strong Generalization with On-Policy Reverse Distillation 

**Authors**: Youngrok Park, Sangmin Bae, Hojung Jung, Jongwoo Ko, Yunseon Choi, Young Jin Kim, Pashmina Cameron, Aaron Courville, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2609.08798)  

**Abstract**: Weak-to-strong generalization asks whether stronger models can learn from weaker supervisors and surpass them. This question is particularly important for successive model generations and multi-domain consolidation, where repeating frontier-scale post-training from scratch can be prohibitively expensive. Yet conventional distillation treats the weak teacher as an optimization target, potentially imposing its capacity ceiling on the student. We introduce On-Policy Reverse Distillation (OPRD), which evaluates the teacher's policy shift relative to its reference policy on student rollouts and amplifies the component of the student's verifier-driven policy gradient along that direction. By rescaling only verifier-supported updates, OPRD preserves the stationary points of policy optimization while accelerating learning beyond the teacher. In both successive model transfer and multi-teacher distillation, OPRD achieves higher performance with fewer student updates than existing RL and distillation approaches. Response-style analysis shows that OPRD students remain closer to models trained with verifier-based RL alone than to their weak teachers, suggesting that teacher guidance accelerates rather than redirects the student's own optimization. Results in conventional strong-to-weak distillation further demonstrate that OPRD effectively combines verifier-driven policy optimization with teacher guidance regardless of capacity ordering. 

---
# The Rater Ising-Potts Model with LLM-Derived Weights: An Application to Multi-Category Scoring Reliability 

**Authors**: Matthias von Davier  

**Link**: [PDF](https://arxiv.org/pdf/2609.08797)  

**Abstract**: The Ising model is extended to the Potts model for multinomial data. We introduce a Rater Ising-Potts model that uses agreement indicators between pairs of raters and category labels, with weights derived from LLM embeddings. The model does not presuppose ordered category thresholds or equidistant scoring; instead, it focuses directly on pairwise agreement among raters and assigns category-specific positive weights, making it particularly suited for multi-category scoring reliability when raters evaluate responses using a scoring guide. We demonstrate the model's effectiveness on diverse constructed-response tasks, including balanced short-answer items and more challenging, imbalanced essay prompts from the AERA dataset. Across these settings, the model achieves strong agreement with human scores, with the vast majority of misclassifications occurring between adjacent score levels, confirming its ability to preserve the ordinal structure of scoring rubrics without imposing rigid assumptions. A practical similarity normalization and optional power transformation is introduced as a tunable preprocessing step that sharpens semantic distinctions and can be adapted to different datasets. These findings suggest that LLM-derived semantic similarities, combined with this parsimonious Potts-type formulation and flexible similarity scaling, offer a robust and interpretable framework for reliability auditing in educational assessment contexts. Extensions to multiple raters and hierarchical rating processes are discussed. 

---
# Benchmark Scores Are Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks 

**Authors**: Aymene Berriche, Cathrine Shalby, Mohannad Alhanahnah, Yazan Boshmaf  

**Link**: [PDF](https://arxiv.org/pdf/2609.08765)  

**Abstract**: Large language model (LLM) benchmarks are often treated as fixed datasets with stable scores, yet their outcomes depend on configurable evaluation pipelines. We audit eight cybersecurity benchmarks across 10 proprietary, open-weight, and cybersecurity-specialized LLMs. By modeling benchmarks as measurement pipelines, we identify 15 systematic failure modes and show that a single pipeline choice can change a model's score by more than 80 percentage points and substantially alter model rankings. At the cross-benchmark level, two semantically similar task pairs rank the same models differently because of incompatible evaluation conventions. Under an evaluation harness that standardizes pipeline choices while preserving task semantics, nine of 10 models shift by at least three ranks on at least one benchmark. These results show that cybersecurity LLM benchmark scores are pipeline-dependent and motivate pipeline-aware auditing as a core requirement for reliable model evaluation. 

---
# TontaubeV1: Streaming Text-to-Speech with Hierarchical Codec Modeling and Bounded Context 

**Authors**: Fritz Cremer, Jonathan Cremer  

**Link**: [PDF](https://arxiv.org/pdf/2609.08703)  

**Abstract**: Text-to-speech systems often face a trade-off between natural prosody and efficient inference: higher perceptual quality typically comes at increased computational cost and latency. We present TontaubeV1, a model that preserves natural prosody while enabling streaming from a single consumer GPU. Speech is encoded by the hierarchical DualCodec representation at 12.5 Hz, which separates a semantic stream from successive acoustic refinements. Our design assumes that prosodic structure is largely established when the semantic stream is generated, and allocates capacity accordingly: a Qwen3-1.7B-derived transformer predicts that stream and thereby the utterance duration, while three progressively smaller Qwen3-0.6B-derived transformers each add one acoustic refinement. Text is tokenized per character rather than by subword. Paired text and audio markers at shared positions support long-form generation with bounded context, and overlapping DualCodec reconstructions are mapped into the VibeVoice acoustic latent space and decoded causally, enabling streaming despite DualCodec's noncausal decoder. The model accepts up to one minute of reference audio for voice conditioning and is designed primarily for English and German, with additional multilingual support. The four predictors total 2.9B parameters; on a single RTX 5090 the streaming path reaches approximately 200 ms to first audio. In separate non-streaming measurements, the end-to-end real-time factor (RTF) is 0.08 for one input and the aggregate RTF is 0.02 across eight concurrent inputs. On our LLM-as-a-judge audiobook-reading benchmark, TontaubeV1 matches ElevenLabs Flash v2.5 and outperforms Fish Audio S2 Pro, the April 2026 Gradium API, and Cartesia Sonic 3 on prosody. The model weights are released on Hugging Face under the Tontaube Community Model License 1.0. 

---
# Hyperparameter Scaling Laws Across MoE Sparsity 

**Authors**: Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.08690)  

**Abstract**: Mixture-of-Experts (MoE) models expand model capacity without a proportional increase in training compute, but increasing sparsity makes reliable hyperparameter transfer challenging. In this work, we show that conventional hyperparameter scaling laws are insufficient for ultra-sparse MoEs: the optimal learning rate and batch size vary with activation ratio, and these shifts cannot be explained by either total or activated parameter count alone. To characterize this dependence, we conduct 1,800 pre-training runs spanning six activated-parameter scales and models with up to 6B total non-embedding parameters, processing approximately 20 trillion tokens at a cost of 200,000 equivalent H800 GPU-hours. Our results reconcile conflicting findings in prior work by revealing two scaling regimes. At fixed sparsity, the optimal batch size follows a power-law relationship with training tokens $D$, whereas the optimal learning rate scales with training compute $C$ and remains robust to the allocation between model size and data. Across sparsity levels, the activation ratio $A$ enters both relationships as an additional multiplicative power-law factor. These observations lead to unified hyperparameter scaling laws that transfer across MoE sparsity levels. Large-scale evaluation shows that the scaling form outperforms alternative functional forms. On a held-out ultra-sparse MoE with 12B total parameters and only 1/64 of its experts activated, the predicted hyperparameters remain close to the observed optima, supporting joint extrapolation across model scale and sparsity. Further experiments demonstrate transfer across expert granularities and isolate the effect of activation ratio from that of total expert count. 

---
# Difficulty-Adaptive Tree-Structured Policy Optimization for Expanding Reasoning Coverage in RLVR 

**Authors**: Youngjun Yu, Sanghwan Jang, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.08650)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has been central to the recent success of Large Reasoning Models. However, while RLVR significantly improves single-sample accuracy, it often fails to expand the model's intrinsic reasoning coverage (pass@k) due to limited exploration during training. To address this, we optimize the structural design of train-time rollouts to enhance pass@k. Our analysis identifies three key design principles: (1) difficulty-adaptive rollout can play an important role in expanding pass@k, beyond serving as an efficiency heuristic; (2) tree-based rollout outperforms parallel sampling in discovering correct answers; and (3) sentence-entropy-guided forking overcomes the localization phenomenon of token-level branching to maximize semantic diversity. Building on these insights, we propose DATPO (Difficulty-Adaptive Sentence-entropy-guided Tree-structured Policy Optimization). DATPO integrates difficulty-adaptive tree search with a sibling-diversity advantage term, explicitly promoting semantic diversity to expand reasoning coverage during training. Experiments on mathematical reasoning benchmarks demonstrate that DATPO outperforms baselines especially in pass@k, which directly translates to superior test-time scaling performance. 

---
# A Three-Tier Persona Vector for Controllable User Simulation in Agentic Evaluation 

**Authors**: Rahul Khedar, Eshita, Sneha Teja Sree Reddy Thondapu, Mayank Malhotra, Arup Kumar Das, Jitesh Chandra Mishra, Arun Menon, Avinash Karn, Mouli V  

**Link**: [PDF](https://arxiv.org/pdf/2609.08592)  

**Abstract**: Evaluating tool-augmented LLM agents requires diverse, realistic user inputs yet most evaluation frameworks use flat role descriptions ("you are an angry customer") that produce near-identical conversations regardless of the underlying scenario. In this paper, we propose a three-tier persona vector with 23 operationalized dimensions: 6 categorical demographics (jurisdiction, age, channel, device, language proficiency, time availability), 12 continuous behavioral traits (patience, assertiveness, digital literacy, etc.) sampled with Gaussian noise around curated profile base vectors, and 5 continuous emotional states (frustration, anxiety, trust, confidence, stress) that shift in response to scenario context. Orthogonal to the persona, a 4-level query-complexity overlay controls utterance phrasing from direct to deliberately vague. We evaluate the persona model inside a synthetic data generation pipeline across 64,698 multi-turn conversations spanning 8 named profiles and 3 production corpora. Key findings: (i) a 15.8 percentage-point spread in agent goal-achievement across personas confirms trait vectors produce measurably different user behavior; (ii) the same persona behaves differently across scenarios due to scenario-reactive emotional state shifts, validating the scenario-reactive design; (iii) domain-specific projects show persona sensitivity on booking-flow compliance (~15-20 percentage points gap between tier-aware and pressure-test personas), demonstrating the model faithfully reproduces real-world difficulty distributions; (iv) seven rule-described trait correlations produce auditable co-occurrence patterns without requiring learned covariance matrices. The persona model is fully specified for reproduction. 

---
# The Unreliable Progress Bar: Can LLM Agents Reliably Report Task Progress Throughout Execution? 

**Authors**: Boyang Wang, Yunhan Wang, Yalun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.08589)  

**Abstract**: Recent large language models can emit task-progress signals that agent frameworks use to decide whether a task should continue or stop, yet whether a model can reliably report its task progress at every stage of a task, and where and how its reports fail, has not been studied systematically. We evaluate this ability on the public benchmark $\tau^2$-bench and on StageIF, a controlled testbed in which reporting checkpoints are placed across the task's lifecycle. Both settings require reports at multiple task stages. We find that reporting reliability depends on the stage a task has reached, and that almost every deployed model we test is reliable at some stages and unreliable at others. Where reporting breaks down is not the same everywhere. Most deployed models lose accuracy once work is under way and recover once the task is done. The newest generation closes that mid-task drop and instead grows conservative at the finish line. Our study exposes a capability gap in task-progress reporting and provides an evaluation protocol that spans the whole course of task execution for this ability on which agent operation depends. The findings indicate that agent frameworks should not control task flow on the strength of the model's state reports alone. 

---
# Environments as Scaffold: Enriching Feedback to Bootstrap Self-Evolving Agents in Long-Horizon Tasks 

**Authors**: Hongbang Yuan, Zhuoran Jin, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.08404)  

**Abstract**: Large Language Models demonstrate remarkable proficiency in static reasoning, yet training them as autonomous agents through Reinforcement Learning (RL) for long-horizon tasks is often hindered by severe reward sparsity. While conventional \textit{agent-side warming} up via supervised fine-tuning (SFT) can alleviate this, it is frequently limited by data scarcity and constrained exploration. To address this, we propose a paradigm shift to \textit{environment-side adaptation} by constructing \textbf{F}eedback-\textbf{E}nriched \textbf{E}nvironments (\textbf{FEEs}). Through a pilot study, we establish a feedback design strategy that reformulates environments by transitioning from action guidance to observation enrichment during the later stages of both intra-episode exploration and inter-episode evolution. Large-scale experiments on SciWorld and BFCL benchmarks using various Qwen3 model scales and RL algorithms such as GRPO, GSPO, and DAPO demonstrate that FEEs consistently yield performance improvements over standard settings. Furthermore, our analysis reveals that training with FEEs \textbf{(1)} stabilizes training dynamics by reducing entropy volatility, \textbf{(2)} facilitates proactive state-space exploration in difficult tasks, \textbf{(3) }ensures the internalization of environmental guidance into policy weights rather than acting as a mere inference-time prior, and \textbf{(4) }identifies intra-group feedback consistency as a critical boundary for stable optimization. 

---
# From Coordinates to Candidate Regions: Temporal Change Localization via Region Selection in Remote Sensing Multimodal LLMs 

**Authors**: Juwan Chung, Sungjune Park, Yeongyun Kim, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2609.08391)  

**Abstract**: Remote sensing multimodal large language models (RS-MLLMs) have advanced scene understanding and visual question answering over satellite imagery, yet localizing specific objects or changed regions remains challenging. Existing approaches rely on generating bounding box coordinates as token sequences, which is fragile for the small, densely packed objects common in remote sensing and increasingly error-prone when multiple targets must be localized simultaneously. In this work, we present an RS-specific formulation of the region selection paradigm, previously explored in natural-image MLLMs, and extend it to temporal change localization over multi-image sequences. Our framework employs a text-conditioned region proposal module, encodes each candidate as special tokens carrying per-frame visual features enriched with spatial and temporal cues, and lets the LLM localize targets by selecting region tokens in its response. We construct a multi-task training and evaluation suite spanning localization, referring expression, visual grounding, and understanding tasks across single-image and multi-temporal settings. Experiments show that our approach substantially outperforms coordinate-generation baselines on temporal change localization, while improving single-image visual grounding and maintaining competitive understanding performance. Oracle analysis decomposes the contributions of the region proposer and the LLM selector, providing diagnostic insight unique to this framework. Our code will be available at this https URL. 

---
# Miles v0.1: Production-Level Post-Training 

**Authors**: RadixArk, Tom Chen, Mao Cheng, Shi Dong, Kangrui Du, Yanbin Jiang, Jiajun Li, Yiming Li, Tao Lin, Yusheng Su, Andy Ye, Yueming Yuan, Zhichen Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.08368)  

**Abstract**: We present Miles v0.1, a full-stack, production-ready system for frontier post-training. Building upon the clean design of slime, Miles designs each stage of the reinforcement-learning (RL) training loop around a single principle: components should be verified, clean, and customizable. With accuracy, efficiency, reliability, and scalability as first-class goals, Miles aims to make frontier-scale RL accessible to researchers and enterprises alike. This report walks through the system end to end: rollout engines built on SGLang, a trainer with a choice of two backends (NVIDIA Megatron-LM and PyTorch FSDP), and three weight-synchronization transports for different deployment topologies. Beyond full-parameter RL, Miles also supports LoRA RL, on-policy distillation, supervised fine-tuning, and true-on-policy rollout-training alignment, and extends the same architecture to diffusion models. We close with an end-to-end case study: fully asynchronous agentic RL on a GLM-5.2 744B-A40B model over terminal-use coding tasks, running on 64 NVIDIA GB300 GPUs with a median step time of 263 seconds over the first 30 measured steps. Miles is open-sourced at this https URL, with the project website at this https URL. 

---
# RepoNav: From Snippet Retrieval to File-Centered Repository Navigation for Code Agents 

**Authors**: Hongzheng Chai, Jiakun Li, Hongyue Yu, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2609.08355)  

**Abstract**: Solving repository-level code tasks requires LLM-based agents to use code search tools to navigate large codebases and identify a small set of relevant files and functions. However, current retrieval tools typically return flat lists of isolated code snippets: such lists can surface relevant files, but provide insufficient structure for agents to distinguish the target function from semantically similar alternatives in the same file. We introduce RepoNav, a lightweight post-retrieval interface that reorganizes retrieved snippets into a file-centered navigation scaffold. By presenting compact structural cues and candidate targets, this scaffold guides on-demand file-structure browsing, helping agents compare sibling symbols before selecting a target function. Across diverse models on LocBench, RepoNav improves function-level localization and narrows the file-to-function gap. Controlled ablations demonstrate that these gains come from structured evidence organization rather than simply exposing additional file structure, and the approach also improves performance on a repository-level question-answering benchmark. 

---
# Distillation as Probability Transport: Routed On-Policy Distillation 

**Authors**: Tianle Xia, Lingxiang Hu, Yiding Sun, Linfang Shang, Ming Xu, Lan Xu, Ning Zheng, Wei Xu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08337)  

**Abstract**: On-policy distillation (OPD) transfers teacher knowledge on student-generated trajectories, but efficient sampled objectives reduce the teacher distribution to scalar credit on individual tokens. Such credit indicates whether a token should gain or lose probability, yet leaves the corresponding redistribution unspecified. We recast OPD as teacher-guided probability transport and propose RouteOPD (Routed On-Policy Distillation), which decomposes local teacher--student disagreement into student-excess sources and teacher-deficit destinations and couples them into explicit transport pairs. RouteOPD optimizes pairwise log-odds toward jointly realizable targets obtained from a bounded teacher potential, while adapting the transport budget to the concentration of teacher demand. This formulation directs updates toward teacher-preferred destinations and controls their magnitude within a single transport operator. Experiments across four teacher--student settings and four mathematical-reasoning benchmarks demonstrate that RouteOPD consistently outperforms sampled reverse-KL OPD, with improvements accompanied by higher routing fidelity and lower background leakage. These results demonstrate the effectiveness of explicitly modeling probability transport in on-policy distillation. 

---
# HoneyRoute: Honeypot-Model Routing for Adversarial LLM Serving 

**Authors**: Han Jin  

**Link**: [PDF](https://arxiv.org/pdf/2609.08306)  

**Abstract**: We introduce HoneyRoute, an inference-serving layer that detects whether an incoming request is malicious and, if so, routes it to a dedicated honeypot model, shielding production while the adversary's interaction is continuously harvested for intelligence. Existing defenses embed traps inside model memory or rebuild deception at the protocol layer, leaving the serving tier unprotected and feeding nothing back into detection. HoneyRoute couples (i) a streaming router (a frozen 0.8B-embedding backbone with per-domain MLP heads), (ii) a dual-implementation honeypot (a rule/prompt-engineered code honeypot or a dedicated same-family replica), and (iii) an analysis loop that converts trapped interactions into attacker fingerprints for router retraining. On a production trace plus a seven-domain attack corpus, the router reaches F1=.911 at 38 ms median added latency, matching 96% of a two-tier guard-LLM cascade's F1 at 1/385 of its latency with 0% evasion under 13 adversarial transformations; diverting the malicious share cuts production-model token consumption under concurrent flooding with real GCG-suffix payloads by 97.8%; the trained replica agrees with the production model on 92.9% of benign holdout requests, while naive unconditional bait injection collapses to 7.6% and selective camouflaged injection recovers to 88.9%, mapping the recoverable fidelity-traceability frontier; and a loop-trained correction head cuts misrouting of legitimate security research 9x while raising detection F1 to .933. 

---
# SE-GoS: Self-Evolving Graph-of-Skills for Skill Library at Scale 

**Authors**: Dawei Fu, Cheng Jiang, Sitian Qian, Huainan Wang, Zhongkai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2609.08228)  

**Abstract**: Modern LLM agents increasingly rely on reusable skills, yet as skill libraries scale to thousands of entries, effective retrieval becomes a bottleneck. Graph-of-Skills (GoS) addresses this challenge by exploiting dependency-aware graph structure for scalable skill retrieval, while SkillDAG further demonstrates that skill graphs can accumulate execution-backed structure online. However, these approaches leave open whether historical execution traces can be systematically distilled into a better retrieval graph that generalizes to unseen tasks. We present Self-Evolving Graph-of-Skills (SE-GoS), a training-free framework that evolves an existing GoS graph from execution traces while preserving the original retrieval pipeline. SE-GoS performs three complementary updates: topology evolution that discovers and prunes skill relationships from execution evidence, edge-weight evolution that reinforces retrieval-relevant relationships based on historical effectiveness, and description evolution that optimizes retrieval-facing skill descriptions using execution feedback. Across three LLMs on SkillsBench, SE-GoS consistently improves task reward while reducing input tokens relative to full skill loading, with gains varying across model families. In a representative setting, one evolution round improves reward from 52.4\% to 59.4\% while reducing input tokens by approximately one-third relative to full skill loading, and the resulting graph transfers to a disjoint held-out split with a 5.4-point improvement over the static GoS baseline. These results show that skill graphs can be improved from execution experience without model training, changes to the retrieval algorithm, or modifications to skill content, turning a static retrieval graph into an evolving retrieval infrastructure. 

---
# Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Efficient LLM Inference 

**Authors**: Hongjin Lin, Wentao Wan, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08189)  

**Abstract**: Dynamic layer routing reduces the inference cost of Large Language Models (LLMs) by learning to skip layers for individual tokens. Existing methods, however, treat each routing decision as a local operation conditioned solely on the current hidden state which is a formulation that overlooks the sequential, path-dependent nature of routing across depth: earlier decisions shape the representations seen by downstream routers, and the layer-usage objective couples all decisions jointly. We propose History-Aware Routing (HeRo), a dynamic routing framework that resolves this mismatch by introducing a router memory mechanism to maintain an explicit routing state across model depth. The memory is constructed via linear attention, incrementally aggregating preceding routing scores and their induced residual updates into a compact history representation. At each routed layer, the router conditions jointly on this accumulated state and the current hidden representation to select the executed branch. Instantiated for token-wise FFN routing, HeRo trains only lightweight routers and adapters on a frozen backbone, requiring no modification to pretrained parameters. Across Llama 3.1-8B, Llama 2-7B, and Llama 2-13B, HeRo consistently achieves the highest aggregate performance retention among ten baselines. On Llama 3.1-8B, it bypasses 26.87% of model parameters while achieving 100.24% of dense model performance across seven benchmarks, and retains 97.01% while bypassing 38.82% of model parameters under a tighter computation budget. Ablation studies confirm that removing routing history consistently degrades performance, most notably on multistep reasoning and code generation, validating that explicit routing memory enables more accurate and adaptive dynamic routing than solely conditioning on hidden state. 

---
# Does Deeper Reasoning Compromise Alignment? Revealing and Mitigating of Alignment Collapse in Large Reasoning Models 

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Henghua Zhang, Bairui Zhang, Jia-Chen Zhang, Shaohua Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.08186)  

**Abstract**: The emergence of Chain-of-Thought (CoT) has established a robust foundation for Large Reasoning Models (LRMs). While deep reasoning is widely believed to enhance safety alignment, the stability of alignment mechanisms under extended reasoning remains underexplored. This paper challenges the prevailing view by revealing a critical vulnerability: Deep Reasoning May Induce Alignment Collapse. To rigorously quantify this phenomenon, we propose the Alignment Loss Rate (ALR) metric. Our experiments demonstrate that as reasoning depth increases, ALR rises significantly, indicating a severe degradation in model robustness against external perturbations. Capitalizing on this instability, a novel jailbreaking paradigm, Reasoning Trap (RT), is proposed. RT induces the model into extended reasoning to amplify the impact of adversarial attacks, leading to a sharp decline in safety capabilities. To elucidate the mechanism behind this collapse, we identify Attention Dilution as the root cause, arising from the competition for attention between the extended reasoning process and the original input. To mitigate this, Reasoning Residual Alignment (RRA), a lightweight defense strategy that dynamically re-emphasizes the input via residual connections integrated with the reasoning process. 

---
# SchemeArena: Factorized Stress Testing of Scheming in LLM Agents 

**Authors**: Jie Ruan, Inderjeet Nair, Amy Liu, Muhammad Khalifa, Yusheng Zhou, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08126)  

**Abstract**: We study scheming in LLM agents, in which agents covertly pursue misaligned goals. Our focus is to understand how scheming arises from the interaction of key factors, such as instrumental goals, environmental affordances, oversight conditions, and perceived consequences. Prior work examines only a small number of scenarios, limiting the ability to isolate how these conditions shape an agent's propensity or capability to scheme. This limited scale and task diversity also restrict coverage of realistic deployment settings and the range of scheming strategies that can be observed. To this end, we introduce SCHEMEARENA, a 400-scenario benchmark for scalable scheming stress testing, constructed through a factorized scenario synthesis framework spanning diverse safety-relevant tool domains, instrumental goals, oversight conditions, and pressure mechanisms. To enable scalable and reliable monitoring, we further propose SCOUT, a scheming monitor that grounds multi-criteria judgments in evidence drawn from agents' reasoning and actions. Across controlled stress tests on five LLM agents, we find that explicit instrumental goals are the strongest driver of scheming propensity. Strategic hints play a distinct role by helping agents translate scheming reasoning into concrete covert behavior. Oversight has mixed effects: in several closed models, action-only monitoring increases scheming, suggesting that partial oversight can act as an optimization constraint rather than a deterrent. CoT is a useful but incomplete monitoring signal: it can reveal latent scheming before execution, yet action-only scheming shows that covert behavior may occur without explicit reasoning evidence. We release the benchmark, code, and monitor at: this https URL. 

---
# Eliciting Self-Verification in Multimodal Reasoning Agents with Reinforcement Learning 

**Authors**: Vishwas Sathish, Viresh Ranjan, Xinliang Zhu, Arnab Dhua, Douglas Gray  

**Link**: [PDF](https://arxiv.org/pdf/2609.08025)  

**Abstract**: Reasoning agents increasingly rely on external tools such as web search to answer complex queries. Reinforcement learning (RL) finetuning algorithms such as GRPO have improved long-form reasoning in text-only language models, particularly for coding and mathematics. Reliable tool use in multimodal agents, however, remains challenging because models must interpret text and images while integrating noisy retrieved evidence, often under sparse outcome-level supervision without explicit verification signals. We present Self-Verification via Reinforcement Learning (SVRL), an RL-only finetuning framework that trains multimodal agents to verify and filter retrieved evidence within their own reasoning traces, reducing reliance on external verifiers at inference time. SVRL also introduces a search-aware penalty that discourages unnecessary tool calls and a query-diversity reward that encourages diverse, well-formed search queries, providing fine-grained feedback on when and what to search. Finetuning Qwen-2.5-VL-7B with SVRL on only 5{,}000 visual question answering examples yields consistent gains in multi-hop VQA generalization and tool efficiency across benchmarks. Overall, SVRL narrows the gap between compact agents and much larger proprietary models while requiring substantially lower training and inference cost. 

---
# CausalVerify: An Execution-Grounded Benchmark for LLM Causal Inference Workflows 

**Authors**: Yonghong Zhang, Ricardo Correia, Isabel M. Parra, Yong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.07944)  

**Abstract**: Existing causal-inference benchmarks for LLMs mostly score method descriptions or whether generated code runs, not whether the executed workflow recovers the target causal estimate. CausalVerify studies this verification problem for structured econometric causal-estimation workflows by separating realistic interpretation from verifiable computation. It pairs 259 published economics papers (reconstructed research question, data description, institutional context) with 100 fixed-seed synthetic scenarios that realise CSV datasets for difference-in-differences, event study, instrumental variables, and regression discontinuity designs. Experiment A (real-paper text agreement) scores method-family and direction agreement against four-LLM consensus labels. Experiment B (synthetic execution) runs model-written R code and checks whether the extracted treatment-effect estimate matches a canonical estimator on the same realised dataset; this execution-grounded correctness layer is L2b+, distinct from L2b, which records only whether the code executes. A calibration arm asks whether self-reported confidence separates correct from incorrect workflows. On Experiment B, seven LLMs reach L2b+ pass rates of 10% to 88% at the default 50% tolerance, and 66 of the 426 workflows that execute (15.5%) return a wrong estimate. Execution ranking (L2b) agrees with L2b+ far better than text-direction scoring (L4): Kendall $\tau=0.81$ and Spearman $\rho=0.93$, versus Kendall $\tau$ between $-0.20$ and $0.10$ for L4. Llama-3.3-70B-Instruct shows the same qualitative gap, and reported confidence does not reliably separate correct from incorrect workflows. The claims are confined to standardized single-shot workflows in these four design families under the evaluated R backend and model panel; the benchmark does not measure general causal-inference ability. Code, data, cached outputs, and a datasheet are released. 

---
# SAFIRE: Safety-Critical Benchmark for Fine-grained Fire and Smoke Understanding in Multimodal LLMs 

**Authors**: Pengfei Li, Naufal Suryanto, Sicheng Zhang, Mohammad Alsharid, Muzammal Naseer  

**Link**: [PDF](https://arxiv.org/pdf/2609.07823)  

**Abstract**: Multimodal Large Language Models (MLLMs) show strong progress on vision-language tasks, yet their reliability in safety-critical settings remains underexplored. Fire-smoke understanding is central to public safety and disaster response, but most existing benchmarks lack diverse real-world scenarios and context-aware evaluation. We introduce SAFIRE, a large-scale benchmark for fire-smoke understanding in MLLMs, comprising 83K captioned images from 20 scenarios and 193K multiple-choice VQA (MCVQA) generated from a 9.7K-image subset, spanning 10 evaluation dimensions from basic perception to higher-order reasoning. A GPT-5.4-assisted multi-stage verification pipeline with MLLM majority voting ensures annotation quality. Evaluating ten open-source MLLMs (8B-38B) yields an average accuracy of 61.9%, exposing major gaps in safety-critical reasoning. We further show that adapting vision encoders with only 7% of our domain-specific data boosts fire-scene classification accuracy from 20.1% to 64.5%, indicating that carefully curated data can yield substantial gains even when data volume is limited. All datasets, models, and code are available at this https URL. 

---
# VoT: Vision-of-Thought for Unified Multimodal Representation Alignment 

**Authors**: Jingxiang Sun, Chao Liao, Zhengxiong Luo, Chaorui Deng, Chen-lin Zhang, Junke Wang, Ceyuan Yang, Haoqi Fan, Weilin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07815)  

**Abstract**: Current text-to-image systems typically employ a "text encoder plus diffusion decoder" paradigm, in which text semantics directly modulate continuous latent noise. Despite their success, these methods lack an explicit, interpretable intermediate representation that effectively bridges high-level linguistic semantics and low-level visual signals. In this paper, we propose Vision-of-Thought (VoT), a framework that introduces a discrete visual-thinking layer between vision-language models (VLMs) and diffusion transformers (DiTs). Instead of treating VLMs merely as text encoders, we use them as multimodal planners that generate discrete VoT tokens representing high-level visual plans, such as objects and layouts, before rendering pixels. We train a specialized VoT tokenizer in the VLM semantic space with a closed-loop objective that combines VLM alignment, feature reconstruction, and vector-quantization losses. These objectives make the tokens semantically readable by the VLM while preserving the visual information needed for generation. Experimental results demonstrate that VoT improves semantic alignment and provides a structured interface for interpretable and controllable generation. 

---
# CodeTD: Topology of Attention Detects Hallucinations in Code LLMs 

**Authors**: Daria Voronkova, Ilya Trofimov, Anton Dmitriev, Eduard Tulchinskii, Evgeny Burnaev, Serguei Barannikov  

**Link**: [PDF](https://arxiv.org/pdf/2609.07779)  

**Abstract**: As AI-code assistant tools become widespread, automatic assessment of the correctness of generated code becomes a significant challenge. Code LLMs are prone to hallucinations, which may lead to code that does not solve the required problem, or even to code with severe security vulnerabilities. In this paper, we introduce CodeTD -- the first approach to pre-execution assessment of code correctness based on topological data analysis (TDA) of Code LLMs' attention maps. Our method quantifies prompt-generation mismatch using topological patterns of attention maps. We carry out experiments with common benchmarks (HumanEval, MBPP, BigCodeBench, MultiPL-E), 5 programming languages and 10 Code LLMs of size up to 34B parameters. The experimental results show that the proposed method outperforms recent baselines. Moreover, CodeTD is transferable between coding benchmarks. 

---
# The Emerging AI Paper-Review Arms Race: Adversarial Co-Evolution in Scholarly Publishing 

**Authors**: Chenguang Wang, Ming Li, Adebayo Braimah, Chenrui Fan, Tuo Wang, Weijie Guan, Ruiyi Zhang, Tianyi Zhou, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.07713)  

**Abstract**: Generative and agentic AI are reshaping both the production and evaluation of scientific research. These developments are often studied separately, as questions of how AI can produce research and how AI can review it. We argue that this separation misses an increasingly important feature of scholarly publishing: changes on one side alter the incentives, constraints, and behavior of the other. We synthesize 230 scholarly publications and institutional records using a taxonomy of six connected dynamics: production scaling, evaluation automation, evaluation manipulation, defense mechanisms and policy responses, evasion and side effects, and long-horizon ecosystem feedback. The literature shows an emerging progression in which cheaper and faster research production increases pressure on evaluation, AI-mediated evaluation becomes more scalable and repeatable, participants can exploit evaluator regularities, and institutions respond with technical safeguards and policy controls. These responses can in turn induce evasion, redistribute errors and workload, and shape the scholarly records reused by future research and evaluation systems. Evidence is strongest for production and evaluation at scale, reproducible manipulation, and institutional response, while post-policy adaptation and artifact-level long-horizon feedback remain less directly observed. This systems view shifts attention from isolated AI capabilities toward how scholarly actors and AI systems adapt to one another over time. 

---
# On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via Hashing 

**Authors**: Yuval Koren, Assaf Ben-Kish, Raja Giryes, Lior Wolf, Itamar Zimerman  

**Link**: [PDF](https://arxiv.org/pdf/2609.07681)  

**Abstract**: Associative Recall (AR) is the cognitive ability to learn and retrieve links between items in memory. In NLP, AR is used as a benchmark for evaluating the in-context memory capacity of architectures such as Mamba, and has been found to strongly correlate with language modeling performance. This paper explores AR from the perspective of mechanistic interpretability, aiming to reverse-engineer the exact internal algorithm used by Mamba to perform recall. Our key insight is that Mamba performs recall by implicitly learning linear hash functions, and we identify the low-level circuit that enables this behavior. Building on these findings and inspired by theoretical tools in similarity-preserving hashing, such as the Johnson-Lindenstrauss lemma, we develop a theoretical framework for analyzing AR, which we term Recall Scaling Laws. Given the vocabulary size and the number of facts in context, this framework allows us to (1) predict the embedding and state dimensions required for Mamba to achieve perfect recall, (2) predict recall success probability given the model dimensions, and (3) analyze multi-layer models and multi-head SSM patterns. Empirical results show that our theoretical findings are accurate and predictive, offering insights into how AR capacity scales with vocabulary, state, embedding size, and architecture. 

---
# From Citations to Contributions: LLM-Assisted Credit Scoring of Research Articles 

**Authors**: Sana Ebrahimi, Suraj Shetiya, Abolfazl Asudeh  

**Link**: [PDF](https://arxiv.org/pdf/2609.07673)  

**Abstract**: Citation-based measures of scientific influence typically treat citations as uniform signals, ignoring the different roles that cited works play in a paper's contribution. We introduce contribution-based credit scoring for research articles: a structured citation analysis that decomposes a paper's credit between its own original contribution and the prior work it builds on. Motivated by a cooperative-game view of scientific credit, we propose the contribution tree, a hierarchical framework that conserves importance across the document structure and separates original from citation-derived contribution. To make this framework scalable, we use LLMs as noisy comparative estimators of local importance. We further extend the model to article collections by propagating contributions through weighted citation graphs, yielding corpus-level contributions and normalized influence scores. Our experiments suggest that our framework captures contribution signals beyond surface-level heuristics. Our code is available at this https URL 

---
# Noēsis: Deterministic-First Retrieval with Two-Tier Context Hydration for Factuality-Critical Queries on Small Local Models 

**Authors**: Nicola Cogotti  

**Link**: [PDF](https://arxiv.org/pdf/2609.07663)  

**Abstract**: A wrong number is worse than no answer. Across factuality-critical domains -- audience metrics, scheduling and rights in media; dosages and lab values in healthcare; figures and citations in finance and legal -- a confident but fabricated value is more damaging than an honest admission of uncertainty. Yet this is the dominant failure mode we observe on small local language models: even when correct evidence is present in context, models fabricate plausible numbers and timestamps. Recent work characterizes a real limit of this regime: below 7B parameters, the bottleneck of retrieval-augmented generation (RAG) is not retrieval quality but context utilization.
We present Noesis, the deterministic-first query plane of the Noesis architecture, which makes every deterministic judgment before generation. Its mechanisms follow from the ingestion architecture (subject of a separate patent application): (a) a producer-side fact layer rendering precomputed metric facts verbatim without ranking; (b) positional addressing with deterministic cross-source alignment, resolved ahead of query time at zero LLM cost; (c) provenance scoping as an attribution constraint with multi-tier named-reference routing; and (d) two-tier context with model-triggered verbatim hydration.
Across four ablations, a 2B model reaches parity with a 35B model on factual integrity (exact values in all runs; zero confabulated numbers on absent-entity traps); structured retrieval beats flat RAG by +11.4 points at 2B; skeleton-only context preserves quantitative answers at 20-30% smaller prompts; and hydration recovers verbatim narrative in ~8s versus ~29s. Two properties matter for regulated domains: each query resolves in a single generation call, and every reported value is traceable to its exact source and position by construction. 

---
# Open Tabular Insight Extraction: Where Do We Stand, and Where Should We Go? 

**Authors**: Daniel Gomm, Maarten de Rijke, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2609.07629)  

**Abstract**: Democratizing access to the knowledge held in large corpora of tables such as data lakes is emerging as a central research challenge. Research in this space is advancing and broadening in scope, increasingly supplying the components to satisfy a person's insight need end-to-end. Yet these efforts remain fragmented across communities that frame the problem under their own conventions, such as table question answering, text-to-SQL, and data analysis agents, with works six times as likely to cite within the same task label as across labels. To bring these communities onto common ground, we establish a holistic framework for this pursuit, which we refer to as Open Tabular Insight Extraction (OpenTI). We formalize OpenTI from first principles around the analytical knowledge a person needs, the procedure for deriving it from a corpus of tables, and how well a result serves the person who sought it. In doing so we consolidate frameworks and terminology across information retrieval, natural language processing, machine learning, databases, and human-computer interaction, and apply this grounding in a systematic review and analysis of systems and benchmarks that work towards OpenTI. We find that current systems do not cover the end-to-end scope of OpenTI, mainly focusing on the analysis itself, and that benchmarks are largely unfit for evaluations in an open setting as inputs presuppose knowledge of tables, and validation mechanisms do not match the setup. Finally, we distill a research agenda towards OpenTI systems, evaluation, and interaction paradigms that surface the insights users need. An interactive companion to our paper is available at this https URL. 

---
# AgentIdeaBench: Benchmarking Scientific Ideation in the Agent Era 

**Authors**: Yunxiang Mo, Tianshi Zheng, Yisen Gao, Rui Wang, Newt Nguyen Kim Hue Nam, Kelvin Kiu Wai Tam, Jiaxin Bai, Yangqiu Song, Ginny Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2609.07611)  

**Abstract**: Scientific ideation is the capacity to formulate novel and testable hypotheses from scientific evidence, and autonomous AI scientists depend on it. Existing evaluations largely assess it by asking models to generate ideas from a static, curated set of reference papers. That passive setup departs from the retrieval-and-reasoning workflow of modern AI scientists, and it becomes less discriminative as models improve. We introduce AgentIdeaBench, a multidisciplinary benchmark that evaluates scientific ideation under two matched settings, static observation and active exploration. We report matched Static-Active evaluations for 33 LLMs across 40 densely scored subfields spanning five disciplines, using a multidimensional, literature-verified scoring framework whose critics assess originality against retrieved prior art. Active exploration reveals considerably more capability headroom, and that headroom is unevenly distributed across models. Performance scales about twice as fast as under static observation, and the exploration gain is capability-gated, favoring the strongest models over the weakest. The gain reflects better grounding, improving feasibility, clarity, and specificity while leaving measured originality unchanged under our critics. We further explore Scientific World Modeling, a generation-time loop that refines a draft hypothesis through structured thought experiments. It benefits mid-capability models, and its impact diminishes among frontier models that appear to have internalized such reasoning patterns already. AgentIdeaBench gives future work on scientific ideation a measurement basis suited to the agent era. 

---
# Mapping the Emerging Social Science of Large Language Models 

**Authors**: Yi Yang, Xiao Jia, Zeyun Dong, Chenzhang Wang, Zhanzhan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.07598)  

**Abstract**: Large language models (LLMs) increasingly shape communication, learning, work, creativity, and decision-making, yet social-science research on these developments remains fragmented. We map this emerging field using a curated corpus of 198 papers reviewed in full and a field-scale corpus of 47,719 published papers from five bibliographic databases. Combining sentence embeddings, K-means clustering, within-cluster Latent Dirichlet Allocation (LDA), author and LLM classifications, and structural topic modeling, we identify three domains: LLM as Social Minds, examining socially interpretable model behavior; LLM Societies, examining collective dynamics among interacting model-based agents; and LLM-Human Interactions, examining how people perceive, use, and are affected by LLMs. These domains contain 13 subcategories spanning reasoning, personality and bias, behavioral games, collective intelligence, simulation, trust, work, creativity, and education. In the curated corpus, the three-domain solution is highly stable under resampling (adjusted Rand index = 0.952), and K-means assignments agree with author full-text classifications for 77.78% of papers. At field scale, 13 of 15 topics map onto the taxonomy, while K-means and structural-topic-model domains agree for 73.83% of overlapping papers. LLM-Human Interactions accounts for 78.02% of domain-mapped topic mass, but venue analysis reveals a contrasting pattern: Social Minds and LLM Societies together account for 66.37% of highly cited papers in leading conference venues, whereas LLM-Human Interactions accounts for 76.81% in the corresponding journal subset. The resulting taxonomy provides a reproducible framework for understanding how model behavior, agent interaction, and institutional context jointly shape the social consequences of LLMs. 

---
# I Don't Miss You, but I Do: Self-Explanation Faithfulness of Modality Missingness in Vision-Language Models 

**Authors**: Aydin Javadov, Daniel Schoess, Florian von Wangenheim  

**Link**: [PDF](https://arxiv.org/pdf/2609.07596)  

**Abstract**: Vision-language models are increasingly used in settings where some input modalities may be unavailable, yet we know little about whether they can faithfully explain how such missing information affects their own predictions. We introduce an interventional protocol for evaluating self-explanations of modality dynamics: models state what each modality alone would support, whether restoring a missing modality would change their answer, and whether the available evidence is sufficient; we then execute the corresponding modality intervention and compare these claims with the model's realized behavior. We evaluate eight open-weight VLMs from two model families across four tasks spanning complementary and isomorphic text-image settings and a multi-view driving setting. We find a systematic tendency to overstate the sufficiency of available modality evidence. Models substantially underestimate the effect of restoring missing modalities: task-level median predicted change rates are at most 8.8%, while the corresponding executed change rates reach 72.1%, with underprediction in 62 of 64 model-task-condition settings. Insufficiency claims are rare, but precise when produced: restoring the modality changes the answer in a median of 78-100% of flagged cases. Retrospective self-explanations show the same tendency: on complementary data, models over-credit single-modality sufficiency; on isomorphic data, they over-credit single representation sufficiency relative to their executed behavior. Together, these results show that VLMs systematically mischaracterize how their predictions depend on available and missing modality evidence, motivating executable interventions as a behavioral ground truth for evaluating multimodal self-explanations. 

---
# Same Problem, Different Field: Cross-Domain Solution Import via Domain-Stripped Computational Fingerprints 

**Authors**: Eryk Kulikowski  

**Link**: [PDF](https://arxiv.org/pdf/2609.07595)  

**Abstract**: The same underlying computational problem is solved across unrelated fields under different names: recursive Bayesian state estimation appears as a "Kalman filter" in control, "Bayesian forecasting" in pharmacokinetics, and "data assimilation" in geoscience. Topical and citation-based scientific embeddings cannot see this shared problem. We distill each paper once into a domain- and method-name-stripped faceted computational fingerprint, a free-text mechanism skeleton plus controlled computational facets. We define a tunable, facet-selectable distance over it. The goal is solution import: surface cross-field pairs solving the same problem, so a bespoke implementation can be swapped for another field's standard, specialized solver. On a benchmark of 18 method families across 109 papers, the skeleton lifts cross-domain retrieval average precision over the abstract from 0.222 to 0.513, and the whole fingerprint reaches 0.557. Strikingly, four trained scientific embedders all fall below plain abstract+TF-IDF: they encode topical and citation similarity, the wrong signal for this task. The gain is the representation: the abstract-to-skeleton swap lifts every embedder, and the pipeline is one cached LLM call per paper plus a cheap embedder. An interventional re-skin / math-edit test shows the fingerprint tracks the computation, not the field. On a 501-paper wild corpus, known twins dominate the top of the ranking (23 of the top 30); with planted pairs excluded from the results, three blind LLM judges rate 3 of the top 5 and 8 of the top 30 pairs genuine import candidates, and 0 of 30 random ones. The human verification is the four executed imports: in one, an open standard solver reproduces a bespoke clinical dosing engine's output. We release the benchmark, the code, and the distillation prompt. 

---
# Large-Scale User Behavior Analysis in Multimodal AI-Assisted Manual Task Execution 

**Authors**: Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, David Semedo, João Magalhães  

**Link**: [PDF](https://arxiv.org/pdf/2609.07594)  

**Abstract**: Conversational Task Assistants (CTAs) are multimodal dialogue systems that support users in complex real-world tasks such as cooking and DIY through voice, text, image, and video interactions. Prior user studies have focused on controlled settings, leaving limited understanding of real-world CTA usage at scale. In this work, we present a large-scale study of CTA usage based on thousands of users in-the-wild. Our large-scale real-world data analysis unveils new understandings of (i) user-CTA interaction flows, (ii) user intents, (iii) user conversational traits, and (iv) behavioral factors associated with user satisfaction. Our findings reveal key opportunities for future research in CTAs, particularly in user interaction design and task engagement, concluding with concrete design guidelines. 

---
# TAD: Token-Adaptive Contrastive Decoding with Confidence-Guided Gating for Hallucination Mitigation in Large Audio-Language Models 

**Authors**: Heyu Chang, Nianwen Si, Hao Zhang, Wenlin Zhang, Dan Qu  

**Link**: [PDF](https://arxiv.org/pdf/2609.07286)  

**Abstract**: Large audio-language models (LALMs) can hallucinate audio objects, answering "yes" to absent sound events, thus undermining reliability in audio question answering. We propose Token-Adaptive Decoding (TAD), a training-free strategy for hallucination mitigation that grounds the initial yes/no decision by contrasting logits under real audio with a matched silent reference. TAD introduces a token-adaptive, confidence-guided gate that is decision-critical at the first decoding step and class-conditional on affirmative tokens, using the audio-silent margin to avoid overcorrection when evidence is weak or already sufficient. Experiments on AudioCaps-Hallucination show that, relative to Audio-Aware Decoding (AAD), a contrastive baseline with fixed contrast strength, TAD improves F1 for Qwen2 by 0.059 to 0.117 across Popular, Adversarial, and Random splits, and for Gemma by 0.025 to 0.064, while on Clotho-AQA it raises F1 from 0.810 to 0.816 on Qwen2 and remains comparable to AAD on Gemma. 

---
# KABURI-TTS: Phoneme-Keyed Activity-conditioned Bi-channel Utterance Rendering for Interaction 

**Authors**: Ryuichiro Higashinaka, Shinnosuke Takamichi, Tetsuji Ogawa  

**Link**: [PDF](https://arxiv.org/pdf/2609.07200)  

**Abstract**: Realizing full-duplex spoken dialogue requires large amounts of two-channel, one-speaker-per-channel conversational speech data. Although conversational text-to-speech (TTS) engines have been developed, they are not necessarily robust to two-party simultaneous phenomena such as backchannels, interruptions, and overlaps that occur while the interlocutor is speaking. In this work, aiming at conversational speech synthesis that reproduces human-like overlap, we propose KABURI-TTS. KABURI-TTS takes a per-speaker phoneme raster as input and renders the speech of the two speakers on separate channels, conditioned on the per-frame phonemes and the voice activity derived from them. Because the phoneme raster is supplied by a separate module, the proposed method enables controllable generation of one-speaker-per-channel, two-party spoken dialogue. A user evaluation shows that, compared with strong baselines, the proposed method attains higher naturalness at both the utterance and the interaction level. Furthermore, an analysis of voice activity confirms that the proposed method produces more overlap and more frequent turn-taking. 

---
# Encoded Early, Used Late: Where Transformers Begin to Act on an Inferred Partner's Expertise 

**Authors**: Mika Okamoto, Gabriele Sarti  

**Link**: [PDF](https://arxiv.org/pdf/2609.07139)  

**Abstract**: A transformer can make an attribute linearly decodable in its residual stream at a depth where that attribute does not yet influence the output. This gap between where information is readable and where it is used has been shown for attributes stated directly in the input. We ask whether it also holds for an attribute the model must infer gradually over a conversation, namely how expert its dialogue partner is. Using ExpertCollab, a corpus of multi-turn research-planning dialogues between model-played personas at four expertise levels, we find that partner expertise is most decodable in the early layers and falls to near chance before the midpoint of the network. Counterfactual patching shows that injecting the expertise difference at the layer of peak decodability barely changes a fixed late-layer readout, whereas the same difference injected past the midpoint propagates almost completely, a separation of more than an order of magnitude. A content-matched random control and a probe-free diagnostic place the transition at the same early layer, and a statically specified control attribute stays decodable throughout. An inferred relational attribute is therefore represented well before it becomes causally active, which bounds where any attempt to read out or steer partner-conditioned behavior must intervene. We use one model on a synthetic corpus as an initial demonstration. 

---
# Risk Is Not Review Value: Wrong-Answer Exposure Under Bounded Review Budgets 

**Authors**: SangJin Park, Myungsub Choi, Jineok Kim, Minseung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07095)  

**Abstract**: LLM assistants often produce more answers than humans can review before users see them. Most evaluations ask whether an answer is wrong, unsupported, or low-confidence. Bounded review budgets instead ask which answers should be checked first under a fixed review budget. Risk alone is not enough: a high-risk answer may be hard to repair, while a moderately risky answer may be directly correctable from available evidence. For generated-answer evaluation, we model review prioritization as exposure reduction, where review value combines estimated wrongness, intervention affordance, impact, and cost. We evaluate review queues with Wrong-Answer Exposure Ratio (WAER), the fraction of wrong answers left unreviewed, and post-repair residual exposure (PRRE), the fraction still exposed after deterministic benchmark-supported repairs. PRRE uses repairability rules that do not numerically reuse the affordance scores used for ranking. On a 720-item TAT-QA/SciFact stress benchmark, review-value ranking keeps answer-level WAER nearly unchanged at 20% budget (0.605 vs. 0.600) but lowers PRRE from 0.881 to 0.716. These results show that trustworthy LLM evaluation should measure not only error detection, but also how limited review capacity reduces exposed wrong answers. 

---
# Comparing Self-Supervised and Domain-Invariant Features for Cross-Domain Voice Phishing Detection 

**Authors**: Jeongmin Lee, Seung Yun, Minkyu Lee, Ran Han, Yoonkyu Woo, Jinxia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07079)  

**Abstract**: Voice phishing detection faces three critical challenges: real criminal recordings are unavailable due to privacy constraints; when available, only a handful of samples exist, insufficient for fine-tuning; and lightweight acoustic-only detection is needed as an alternative to large self-supervised models. We compare domain-invariant prosodic features and self-supervised representations (HuBERT, wav2vec2.0) through cross-domain evaluation-training on scenario-based actor recordings and testing on authentic criminal calls. Domain-invariant prosodic features achieve 69.5% F1 zero-shot and 71.0% with 5-shot learning. HuBERT achieves highest performance (94.2% F1, 5-shot), while wav2vec2.0 exhibits a precision-oriented detection profile (90.2% F1 with 99.4% precision, 5-shot). These findings reveal fundamental trade-offs: domain-invariant features enable zero-shot deployment when no real data exists, while SSL methods achieve higher performance but require real samples and compute. 

---
# LatentMD: Benchmarking Markdown Boundary Failures in LLM-Generated Text 

**Authors**: Sungjune Lee, Myungjoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.06993)  

**Abstract**: Large language models (LLMs) increasingly generate Markdown that is consumed by renderers, agents, code extractors, and structured downstream pipelines. Yet existing evaluations often conflate content quality with format adherence, leaving Markdown boundary failures under-measured. We introduce LatentMD, a benchmark and evaluation protocol for diagnosing CommonMark-level fence-boundary failures in LLM-generated Markdown. LatentMD separates content correctness from boundary correctness, enabling detection of outputs that are content-correct but boundary-broken. The benchmark contains 4,179 prompts and a CLI for scoring arbitrary model outputs. Across 9 LLMs and roughly 37,600 generations, we find that Markdown boundary failures are widespread: 38.0% of valid main-grid outputs are content-correct but boundary-broken, with substantial boundary breakage under unspecified prompts and in a small human-authored validation set. Ablations show that failures are driven primarily by same-family symmetric-delimiter collisions rather than nesting alone, are only partially mitigated by prompt hints, and generalize to Python triple-quote docstrings while JSON remains robust as an asymmetric-delimiter control. LatentMD provides a reproducible diagnostic target for parser-sensitive LLM evaluation. 

---
# MOLE: Detecting Insider Threats in AI Agents 

**Authors**: Aashiq Muhamed, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2609.06966)  

**Abstract**: Model misalignment, prompt injection, or operator misuse could lead AI agents operating frontier-lab accounts to exfiltrate model weights, poison training data, or weaken release gates. Existing benchmarks do not test whether defenders can detect this activity among routine work under a limited review budget. We introduce MOLE, an open benchmark of 150 AI-operated accounts sharing 9 stateful services over 30 workdays, with 12 threats and 8 corpora from four models totaling roughly 20 billion tokens. Of 39 agent models, 72% complete most assigned harmful objectives and agent refusal does not predict completion. MOLE enables comparison of 40 monitors across corpus generators, observability levels, and threats; even the best evaluated monitor in our single-day audit-event comparison misses nearly half of completed harm. MOLE also enables monitor development: benchmark-guided search improves a mid-tier monitor by 49-64%, while selective use of a stronger monitor improves budget-AUC by 10% over applying it to every account-day at comparable modeled cost. 

---
# Contextual Observer Grounding: Evaluating Situated Spatial Reasoning in Vision-Language Models 

**Authors**: Mimo Shirasaka, Haochen Zhang, Yonatan Bisk  

**Link**: [PDF](https://arxiv.org/pdf/2609.06880)  

**Abstract**: Reasoning over language instructions in embodied tasks such as robotics often requires understanding spatial relations from a speaker's situated perspective. Humans infer such perspectives from shared environmental knowledge, activity context, and commonsense. Recent vision-language models (VLMs) appear capable of spatial reasoning, but their ability to infer a speaker's viewpoint from contextual cues and interpret situated spatial relations from that viewpoint remains unclear. We call this capability contextual observer grounding. To study this capability, we construct the Point-of-View Benchmark (POVBench), a dataset of 3D scenes and queries that disentangles Inferred, Stated, and Given forms of observer grounding in natural embodied communication. Given multi-view observations and a natural-language sentence, models must localize unseen or underspecified targets from situated spatial and contextual cues. Across state-of-the-art VLMs, localizing targets from directional language remains challenging, even when observer grounding is made explicit. We find that explicit breakdowns of observer-relative spatial reasoning improve target localization. Our project page is available at this https URL. 

---
# Exact Record Omission in Delta Attention: A Transport Criterion, Its Cost, and a Replay Certificate 

**Authors**: Vishwajith Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2609.06872)  

**Abstract**: When a user asks an assistant to forget a record, the test is whether the memory now matches the state it would hold if the record had never been stored. Independently encoded rows can be removed directly; a recurrent memory folds records into an evolving state. One hope is a receipt: save the difference the record made when it arrived, carry it forward through later updates, and subtract it, so that deletion costs one fixed-size edit no matter how long the conversation runs. We show that a transported receipt reaches exact omission if and only if the changes the record induces in later updates cancel out on net, and we measure whether they do on the released 48B Kimi Linear hybrid. They do not: after 4,096 further tokens the record still leaves an imprint of about 4.5% of the state norm that none of the tested receipt classes removes, recomputing half the suffix closes less than half the gap, and the per-token log a receipt needs costs more than a full checkpoint after 88 tokens. The same write-rule classification held on Mamba-2, Falcon-H1, and RWKV-7 with predictions recorded before the runs. Restoring a checkpoint from before the record and replaying the surviving suffix matches the never-stored state exactly on every array we check. In the hybrid suffix sweep, masking the record's attention rows brings sampled recovery close to the never-stored floor even though the recurrent imprint remains, and an auditor who rebuilds the reference can still detect it. Among the evaluated methods, checkpoint replay achieves exact omission, with work proportional to the replayed suffix. 

---
# NormViz: A Benchmark and Framework for Grounding Multimodal Reasoning in Global Cultures 

**Authors**: Akhila Yerukola, Fabrice Y Harel-Canada, Simran Khanuja, Abhinav Sukumar Rao, Ashima Suvarna, Nanyun Peng, Saadia Gabriel, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2609.06831)  

**Abstract**: AI systems are used worldwide, but they struggle to serve the needs of culturally diverse populations. Prior work on cultural understanding evaluates AI systems on text-only settings or on visual artifact recognition (e.g. foods, clothing). The ability to reason about visually observable behaviors through local social norms, which we call visual norm understanding, remains unexamined. We introduce NormViz-Bench, a high quality, human-validated benchmark of 3,268 contrastive image pairs (6,536 images) spanning 16 countries. Each pair varies only in the culturally relevant behavior (e.g., objects, attributes, spatial relations, and actions) that alters how each image is interpreted. Each image is labeled as conforming to, violating, or irrelevant to local social norms, and pair-level evaluation requires both images to be correctly classified, thereby preventing reliance on superficial visual shortcuts. Even the strongest VLMs, Gemini 3.0 Flash and Qwen2.5 VL 7B, succeed on only 26.6% and 21.6% of pairs, struggling most with identifying violating and culturally benign visual behaviors. Towards bridging this, we introduce NormViz-Train, a training dataset of 64k images paired with explanations. Though absolute performance remains low (<30%), finetuning on NormViz-Train improves pair accuracy relatively by up to 125% and 36% Qwen3-VL 4B and 8B respectively, showing a path forward to teach models to connect visual perception to cultural significance. Together, NormViz-Bench and NormViz-Train establish visual norm understanding as a challenging and consequential frontier for multimodal AI. 

---
# Measuring GEO Visibility: Prompt Corpora Define the Answer Market 

**Authors**: Olivier Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2609.06811)  

**Abstract**: GEO (generative engine optimization) visibility scores aggregate source appearances, citations, or brand mentions in generated answers. The prompt corpus selects the situations evaluated, while weights determine their relative importance. Together they define an "answer market" that need not represent actual user demand.
Prompt wording can alter retrieval, competing sources, and generated answers. Scoring then requires identifying the appearances, citations, or mentions of interest. If a language model performs this task, its instruction can change the score assigned to an unchanged answer. Our critical survey examines how these choices help define what a GEO score measures. It draws on research into whether indicators measure the intended phenomenon, total survey error, and information retrieval evaluation. The framework specifies situation annotation, prompt formulations, execution conditions, weights, and scoring rules.
When weights are unknown or remain to be chosen, the framework reports sets of admissible scores. It distinguishes values compatible with data and assumptions about a target population (partial identification) from variation across weighting conventions (normative sensitivity).
A citation alone does not establish a source's contribution. The article defines a comparison of answers generated with and without a source in a controlled documentary context, distinct from an intervention on the full engine with competing sources. The framework is supported by reproducible calculations. No new experiments are reported; its general empirical validity remains to be assessed. 

---
# Train Smarter, Not Harder: Switching Signal-Guided Training in Active Learning 

**Authors**: Nagham Omar, Maya Rozenshtein, Evgeny Mishlyakov, Avigdor Gal  

**Link**: [PDF](https://arxiv.org/pdf/2609.06806)  

**Abstract**: Training strategy, namely whether to retrain from scratch or fine-tune from the previous checkpoint, is an overlooked decision variable in active learning. We show that this choice has exploitable structure: retraining is most useful in early rounds, when each batch can substantially reshape the labeled distribution, while fine-tuning becomes safer once the model trajectory stabilizes. We propose HybridAL, an adaptive training schedule that monitors an online stabilization signal and switches from retraining to fine-tuning after sustained stabilization. Two complementary signals, spectral exponent change $\Delta\alpha$ (weight-based) and accuracy change $\Delta$Acc (validation-based), span different points on the time-calibration trade-off. Across three encoder backbones and six text-classification tasks (five seeds each), HybridAL keeps endpoint macro-F1 non-inferior to retraining and fine-tuning at a 0.010 margin, saves up to 49% of retraining time, and recovers a substantial fraction of retraining's calibration advantage as measured by negative log-likelihood (NLL). Compared with schedules that switch at a pre-committed round, HybridAL obtains lower NLL at moderate additional cost, showing that trajectory-dependent switching provides a stronger time-calibration trade-off than fixed early switching. 

---
# AURA-Eval: Evaluation Framework for Acting Under Risk Awareness in LLM Agent Trajectories 

**Authors**: Ruoxi Shang, Christina-Maria Androna, Orfeas Menis Mastromichalakis, Yu Feng, Aniruddhan Ramesh, Rico Angell, Shang Hong Sim, Chrysoula Zerva, Emmanouil Koukoumidis  

**Link**: [PDF](https://arxiv.org/pdf/2609.06783)  

**Abstract**: LLM agents operate in workflows where unsafe actions can have real consequences. Existing safety evaluations often reduce behavior to a single score, obscuring risk recognition, pre-action detection, and safe task completion when a safe solution exists. We introduce AURA-Eval, a framework combining controlled augmentation with granular diagnosis of behavior in tool-use trajectories. Its pipeline identifies safety-critical decision points, generates controlled variations, and constructs counterparts differing in whether a request has a safe fulfillment path. Using 157 sourced trajectories, we generate 1,249 evaluation items and evaluate 20 frontier and open-weight models. We developed rubrics to classify risk detection, action strategy, and scenario-specific action safety. Our results show that LLM agents engage in unsafe behavior more often when no safe fulfillment path exists. In these cases, frontier proprietary models more often recognize risk and exhibit safer behavior by proposing alternatives, while evaluated open-weight models more often directly execute unsafe requests. Increasing impact or reducing opportunities for oversight before execution also exposes greater vulnerability across models. 

---
# A Novel Semantic Manifold Alignment Attack against Embedding-to-Embedding Obfuscation in Privacy-Preserving LLMs 

**Authors**: Sicong Li, Lingfeng Yao, Xingke Yang, Ke Tu, Chenhao Wu, Hao Wang, Jiang Liu, Phone Lin, Xin Fu, Miao Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.06749)  

**Abstract**: With the widespread applications of large language models (LLMs), privacy-preserving inference has become increasingly essential for sensitive queries. To balance privacy and utility, a series of lightweight obfuscation approaches has recently been proposed, where users locally transform plaintext embeddings into the fixed ciphertext ones. While such Embedding-to-Embedding Obfuscation (E2EO) schemes demonstrate considerable resilience against traditional token frequency and embedding inversion attacks, the core mechanism behind remains to be the large-scale one-to-one substitution, which provides no cryptographic guarantees. In this paper, we propose Proxy Manifold Alignment (PMA), a novel attack against E2EO in privacy-preserving LLMs. Our key observation is that E2EO schemes keep the original semantic structure, so that the obfuscated vector stream can be regarded as an unknown tokenizer-language whose symbols are the vectors themselves. Therefore, the proposed ciphertext to plaintext reconstruction attack can be formulated as a translation task from the unknown tokenizer-language to plaintext. Specifically, by only accessing the obfuscated vector stream, the target tokenizer and a public corpus, the PMA attack first employs Word2Vec to model the co-occurrence patterns within the obfuscated stream and the public corpus independently, and constructs two proxy vector embeddings. Then, the attack aligns the underlying manifolds of these two embeddings based on structural similarity. Finally, it maps the obfuscated vectors back to plaintext. Experimental results demonstrate that PMA consistently achieves higher plaintext recovery than other state-of-the-art attack methods. 

---
# Reason Through the Latent! Making Latent Visual Reasoning Necessary 

**Authors**: Suhyeong Park, Junha Jung, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.06746)  

**Abstract**: Latent visual reasoning aims to perform multimodal reasoning through hidden-state computation rather than explicit textual chains of thought. However, visual information being present in a latent state does not imply that the model actually relies on that state when producing its answer, especially when alternative image-conditioned paths remain available. We introduce \textbf{C}ausal \textbf{V}isual \textbf{R}ecurrent \textbf{R}easoning (CVRR), which preserves pretrained visual competence while making recurrent computation the required image-conditioned path to prediction. CVRR initializes recurrence from the question hidden state after the pretrained vision-language model has incorporated the image, then repeatedly updates this state while re-reading the same fixed visual evidence. Before decoding, visual states and the original multimodal KV cache are removed so that only the final recurrent state carries image-conditioned information to the answer. Across the $V^*$, MMVP, BLINK, and MME-RealWorld-Lite benchmarks, CVRR retains strong performance under this strict interface, while compatible latent reasoners fail to recover comparable visual competence even when retrained under the same constraint. Causal interventions further show that predictions remain sensitive to recurrent content when the question is held fixed, and that persistent visual evidence causally revises the recurrent trajectory. These results distinguish latent informativeness from latent computation that is actually used for prediction. 

---
# Counterfactual Tests for Measuring Chain-of-Thought Faithfulness in Visual Language Models 

**Authors**: Bayar Menzat, Maximilian Süss, Ruizhi Wang, Benno Steinegger, Thomas Lukasiewicz, Oana-Maria Camburu  

**Link**: [PDF](https://arxiv.org/pdf/2609.06704)  

**Abstract**: Chain-of-thought (CoT) may often look plausible, yet it may not faithfully reflect the model's decision-making process. While methods for measuring the faithfulness of CoTs for textual inputs have been increasingly introduced, using these methods for visual inputs is not straightforward. In this work, we adapt the family of counterfactual methods for measuring CoT faithfulness, namely the Counterfactual Test (CT) and Correlational Counterfactual Test (CCT), to visual inputs, and call them vCT and vCCT, respectively. Using vCT and vCCT, we benchmark eight recent open-source Vision Language Models (VLMs) on two datasets. Our analysis shows that CoTs do not reliably track visual evidence that influences model predictions: they may omit the removed object even when its removal causes a large prediction shift, yet mention it when the shift is small. We further find that Predict-then-Explain explanations align more strongly with perturbation-induced probability shifts than pre-answer CoTs, while binary vCT scores are often nearly saturated. We also include a reconstruction control, in which images pass through the same editing pipeline without object removal, and find that the main object-removal intervention induces larger shifts than reconstruction alone. We construct and release Counter-SNLI-VE and Counter-A-OKVQA, two datasets of image pairs that differ by a single object. 

---
# Data Efficient Sample Selection for In-Context Learning 

**Authors**: V Venktesh, Cem levi, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2609.06670)  

**Abstract**: The In-context learning (ICL) paradigm aids large language models (LLMs) to adapt to new tasks without need for fine-tuning. However, selecting an optimal combination of demonstration examples from a large pool of example subsets is a challenging problem. Existing approaches for selection do not model the complex relationship between ICL samples and downstream LLM performance. They typically perform static task-level selection, choosing subsets once offline, which can fail to generalize to unseen queries. We introduce DearICL (Data Efficient Algorithm for Ranking) ICL samples, a new framework that models demonstration example selection as a subset ranking problem. DearICL employs a non-linear surrogate employing a differentiable sorting objective within a gap-index bandit algorithm. The gap-index based approach enables fine-grained separation of good arms and borderline arms, which is used as an auxiliary objective to train the non-linear surrogate through sufficient sampling of borderline arms, supporting instance-level subset ranking. On exemplar selection benchmarks with open-source LLMs, DearICL achieves 8.08-15.9% accuracy gains over strong linear bandit baselines, with low sample complexity. Code and data: this https URL. 

---
# EviMap: Evidence-Grounded Hierarchical Topic Maps for Exploring Unlabeled Corpora 

**Authors**: Zhiyin Tan, Changxu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2609.06664)  

**Abstract**: Research teams and organizations often explore unfamiliar free-text collections, from survey comments and reviews to reports and domain documents, before labels, queries or coding schemes exist. At this stage, the first thematic map shapes what users notice, prioritize and carry into downstream analysis, so it should be trusted only insofar as it can be verified. Existing options force a trade-off between scale and verifiability. Qualitative coding preserves evidence but is slow. Search presupposes a query. Clustering and topic models scale but produce labels users must interpret. One-shot large language model (LLM) summaries are fluent yet difficult to reproduce or audit. We present EviMap, an interactive system providing researchers and practitioners with an auditable thematic overview of such corpora. Guided by model-generated context describing the corpus and hypothesized stakeholder concerns, EviMap extracts within-document evidence phrases and organizes them, rather than whole documents, into a three-level map of aspects, groups and fine-grained topics. Embedding-based clustering narrows the search space for finer semantic judgments by the LLM. Each node traces back to supporting phrase spans, so documents link to topics through evidence they contain and users can audit labels against the original text. Users can start from a top-level corpus map, drill into topics, inspect highlighted evidence in original documents, and combine two topics to find documents discussing both. We demonstrate this workflow across six heterogeneous corpora spanning 2,108 to 101,699 documents, with a comparison against flat and hierarchical LLM baselines. By grounding every label in verbatim source spans, EviMap makes a topic map not just readable, but verifiable. Code, demo video, and interactive dashboard are available at this https URL. 

---
# SerenAI: State-transition system inspired by text-based world AI models 

**Authors**: Elvin Babayev, Artem Sinitsa, Arash Hajisharifi, Kabir Bakhshaei  

**Link**: [PDF](https://arxiv.org/pdf/2609.06647)  

**Abstract**: Although professional workflows leverage large language models widely, the interpretation for auditing unconstrained free-text generation is usually intractable if such generation demands legal, operational or financial workflow. We hereby demonstrate a text based system called SerenAI - inspired by world-models, it is a state transition system that outputs verifiable predictions rather than merely text: Provided with a description of the environment, state, and actions, the generated output contains 4 items: causal deltas that causally effect the given state, a next state that can logically follow from the given state and action, a validity reward, and a termination signal. For the released proto-model, we employ 2 steps of adaptation training, namely parameter efficient fine-tuning followed by verifier based RL over 50,000 exampled cause and effects in 12 environments spanning 10 reasoning domains. Compared to an initial internal evaluation of an 8B open-weight baseline, SerenAI increased JSON validity from 85.0% to 93.2%, schema validity from 55.0% to 84.0%, exact structured-output match from 0.0% to 41.5%, causal-delta exact match from 0.0% to 41.5%, resulting-state exact match from 0.0% to 42.0%, reward exact match from 1.0% to 80.5%, and termination exact match from 38.0% to 81.5%. These support the narrower claim that verifier-compatible adaptation can improve structured transition prediction. They do not yet establish legal-grade reliability. Accordingly, the paper also specifies a validation protocol for evidence-grounded legal workflows, calibration, human oversight, and sovereign on-premise deployment. 

---
# A Translational Note on AI Safety Evaluation 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2609.06573)  

**Abstract**: Recent studies report that automated red-teaming finds more vulnerabilities, at lower cost, than human red-teaming on standard AI safety benchmarks, and some read this as evidence that human evaluators are becoming dispensable. The comparison measures one thing and the conclusion claims another. A benchmark measures how thoroughly an attacker searches a predefined set of harms, fixed in advance by the developers, and a harm left out of that set is invisible to any attacker working inside it, automated or not. The same blind spot appeared in academic cryptography and in clinical drug trials, where an evaluation that was internally valid stayed silent about the population it was never pointed at. We call the AI-safety version the \emph{threat-model coverage gap}, and find that it persists in a current open-weight model, where harms surface in non-English prompts that English benchmarks miss. Closing it requires evaluators whose deployment context differs from the developers'. The case for those evaluators is methodological, grounded in coverage, and the existing evaluation frame is unlikely to produce them on its own. 

---
# A Unified Policy Architecture (UPA): The Governance Kernel for Enterprise AI Operating Systems 

**Authors**: Prabhu Raghav, Balamurugan Pandi, Arul Vivek, Shek Mohammed, Sridhar S  

**Link**: [PDF](https://arxiv.org/pdf/2609.06543)  

**Abstract**: Enterprise AI is evolving into an Enterprise Operating System where autonomous AI agents can plan, reason, use memory, invoke tools, execute workflows, and collaborate with other agents. This shift creates a new governance challenge: existing authorization, security, guardrails, and compliance mechanisms are fragmented and are not designed to govern autonomous AI as a unified system.
This paper introduces the Unified Policy Architecture (UPA), a governance architecture for Enterprise AI Operating Systems. UPA provides a unified policy model for governing AI and agents, tools, workflows, memory, enterprise resources, and agent-to-agent interactions and enterprise business rules. It extends policy control beyond authorisation to include runtime obligations, human approvals, compliance, audit evidence, and governance evaluation.
We present UPA's governance model, declarative policy language foundations, policy evaluation semantics, extensible plugins, industry policy packs, and an evaluation framework for enterprise governance. We also identify extensions for multi-agent coordination, provenance-aware policies, and stateful runtime governance. UPA provides a foundation for building secure, accountable, and governable Enterprise Operating Systems for autonomous AI. 

---
# OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution 

**Authors**: Shubhashis Roy Dipta, Sourajit Saha, Shaswati Saha, Nobin Sarwar  

**Link**: [PDF](https://arxiv.org/pdf/2609.06490)  

**Abstract**: Recursive Super-Resolution (SR) extends fixed-scale SR to extreme magnification by repeatedly feeding predictions back into the same model, analogous to zooming an image repeatedly. However, ground truth availability at every scale, especially at depth, remains challenging as the required source resolution grows geometrically, leaving deeper predictions unsupervised. We present OracleZoom, an on-policy distillation-inspired, reference-constrained framework that trains on its trajectory while carrying the last ground-truth evidence beyond the supervision boundary. Direct and cross-scale supervision constrain verifiable content, while a no-reference quality objective guides unresolved fine-scale detail. A KL-constrained pretrained latent prior limits quality-driven drift, while EMA consistency stabilizes the supervision boundary. Across seven datasets, OracleZoom achieves the state-of-the-art SR quality across zooming scales, averaging 0.713 CLIPIQA, with larger gains on deeper scales, while significantly reducing hallucinations. Code, data, and models are available at this https URL . 

---
# A Group-Based Resource Allocation Model for the Fractional Knapsack Problem 

**Authors**: Abhinaba Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2609.06470)  

**Abstract**: To solve the fractional knapsack problem, Dantzig's greedy rule orders items according to their value-to-cost ratio. This ordering introduces priority issues. An arbitrarily small perturbation to the input can change the allocation if the budget is exhausted between two items with very similar ratios. To mitigate that problem, we introduce a two-stage rule. We group items sharing attributes within a radius $\delta$. These groups are then evaluated in descending order of ratio, and divide their group's budget share without further ranking. Consider a group featuring an aggregate capacity $U_G$, unit costs contained in $[w^-,w^+]$, and a representative value $\widehat{v}$. The group's loss compared to the exact optimum is bounded by $\widehat{v}\, U_G\frac{w^+-w^-}{w^++w^-}+\varepsilon_v U_G$, in which $\varepsilon_v$ limits the group's internal value variation. Moreover, for any group size, this harmonic factor remains tight. The overall loss becomes restricted to the single budget-binding group whenever the grouping remains order-compatible; thus, groups containing at most $K$ items suffer a per-item loss of $\mathcal{O}(K/n)$. Should group ratio intervals exhibit an overlap of at most $\omega$, an additive term $\omega C$ degrades this bound. Within the separation margin between adjacent groups, the grouped allocation remains Lipschitz continuous with respect to cost data, exhibiting a modulus of $\frac{K}{w_{\min}}$. Computing this allocation takes $\mathcal{O}(n+m\log m+|\Gamma|\log|\Gamma|)$ time given $m$ groups and a boundary group $\Gamma$. Alternatively, the time complexity drops to $\mathcal{O}(n+m\log m)$ if a linear-time selection method identifies the boundary group's allocation. 

---
# One Step, One Lead: Mitigating Higher-Order Interference in Multi-Domain Reinforcement Learning via Cross-Step Control 

**Authors**: Zihan Lin, Xiaohan Wang, Jie Cao, Jiajun Chai, Guojun Yin, Wei Lin, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2609.06469)  

**Abstract**: Reinforcement learning (RL) across multiple domains can broaden the reasoning capabilities of large language models (LLMs), yet joint training often degrades individual-domain performance and can destabilize optimization. Existing work typically diagnoses such interference from a single-step view using first-order gradient alignment or curvature-based proxies. We show that this view can miss a critical form of sequential interference: same-point domain gradients may remain nearly orthogonal even when consecutive realized updates partially reverse one another in output space. We further show that consecutive token log-probability footprints recover this interaction directly from adjacent checkpoints as a local second-order interaction in output space, without explicitly reconstructing same-step curvature. Building on this insight, we propose OSOL, which designates a focus domain at each iteration, uses the preceding checkpoint footprint to rank token-level rebound risk, and applies a drift-ranked, adaptively scaled correction within the standard GRPO update. Our analysis shows that this correction suppresses the targeted cross-step output backtracking component. Controlled studies further show that cross-step backtracking is more strongly associated with subsequent task damage than same-point gradient diagnostics, while the preceding footprint ranks future rebound risk more accurately than Hessian-based proxies. On Qwen3-30B-A3B, OSOL reaches a domain-macro average of 0.4822, improving by 5.7% over the strongest compared baseline, without explicit higher-order differentiation. 

---
# Phase-cycled randomized benchmarking of quantum processors: recovering hidden classical noise correlations 

**Authors**: Mirza Samad Ahmed Baig, Syeda Anshrah Gillani, Abdul Akbar Khan, Muhammad Omer Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.06448)  

**Abstract**: Randomized benchmarking can hide classical temporal correlations because its Clifford-twirled response is even in the noise phase. For a stationary symmetric telegraph fluctuator, we show that continuous evolution and independent stationary resets at slot boundaries yield identical mean responses for arbitrary fixed idle modulations. We construct an eight-setting phase-cycle measurement of the connected sine-phase covariance under ideal Clifford twirling and classical idle dephasing. This observable vanishes for independent slot noise and fixed detuning without a weak-phase or Gaussian approximation. A closed telegraph response, independent circuit calculations and 800 simulation trials validate the construction and quantify the empirical coverage of a paired bootstrap estimator. A separate conservative confidence set states its finite-sample assumptions. Two acquisitions on an IBM processor compare engineered shared-sign and independently reset phases with identical marginals. Their primary contrasts are 0.254 and 0.211, with empirical 95% intervals [0.177, 0.331] and [0.136, 0.285], respectively; all negative-control intervals include zero. At equal shot and sensingwindow budgets, an ideal Ramsey/echo estimator is more precise in every tested class. The result supplies an explicit connection between a benchmarking identifiability limitation and a controlled correlation measurement. No native or quantum-memory detection is claimed. 

---
# Building Trustworthy Graph-Agentic RAG for Social Good: Architectures, Failure Propagation, and Assurance by Construction 

**Authors**: Vijay Bommireddy, Raviteja Bommireddy  

**Link**: [PDF](https://arxiv.org/pdf/2609.06391)  

**Abstract**: Graph-agentic retrieval-augmented generation combines structured evidence with adaptive controllers that can plan retrieval, traverse relations, verify intermediate claims, delegate subtasks, and use tools. This combination is useful when answers depend on relations across documents, entities, time, or institutions, but it also creates coupled failure paths: a defect in graph construction can become retrieved evidence, alter later control decisions, and propagate toward a consequential outcome. We examine how such systems should be designed and evaluated for social-good settings in which freshness, authorization, traceability, oversight, and recourse matter alongside answer quality. We organize the literature by graph substrate, graph lifecycle, agent function, coordination pattern, and authority boundary, and distinguish graph-based retrieval from observation-dependent graph control. We then synthesize reported risks as an evidence-to-action failure chain and propose an assurance-by-construction blueprint comprising five interface contracts for evidence, retrieval, reasoning, capability and delegation, and outcome. These contracts make provenance, temporal validity, authorization, uncertainty, and recoverability explicit at system boundaries. An illustrative public-benefit information design shows how the framework constrains graph structure, permissions, abstention, and operating authority. Finally, we derive an evaluation agenda spanning graph assertions, trajectories, claims, coordination, and outcomes. 

---
# Are Verifier Errors Independent Within a GRPO Group? Evidence from Qwen2.5 Rollouts 

**Authors**: Esther Xin  

**Link**: [PDF](https://arxiv.org/pdf/2609.06386)  

**Abstract**: Group-based reinforcement learning with verifiable rewards (RLVR) scoresmultiple completions per prompt using automatic verifiers. Analysesbased on independent verifier errors may overlook dependence associatedwith shared answer formats. We investigate this dependence in24,998 groups of eight completions generated by Qwen2.5-1.5B onMATH, GSM8K, and DeepMath-103K. We estimate a pooled within-groupverifier-error correlation of 0.530 (95% confidence interval:0.500--0.560). Under an exchangeable-error model, this correspondsto a design-effect-adjusted effective sample size of 1.70 for aneight-completion group. Dependence varies substantially across answerforms: fractions, radicals, symbolic expressions, and intervals exhibitstronger clustering than unit annotations and percent signs. Replayinggroup-relative advantages across four rule-based verifier configurationsidentifies at least one advantage-sign disagreement in up to 0.83% ofgroups. Because a group is repeated sampling for one prompt, thiswithin-group clustering may reflect shared prompt difficulty as well asshared answer form, and we do not attempt to separate the two this http URL studies of correlated judgments across multiple evaluators, ouranalysis examines dependence across completions scored by the sameverifier. These findings motivate prompt- and answer-form-aware analysesof verifier noise rather than characterizations based solely onaggregate error rates. 

---
# From Reading Code to Reading Spec: A Verified Layer for LLM-Driven Codebase Maintenance 

**Authors**: Xinhao Zhang, Jingjie Lu, Kunpeng Liu, Fei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.06383)  

**Abstract**: The rapid growth of LLM-generated code increases software complexity and the maintenance burden on engineers. While LLMs offer a potential automated alternative, this structural complexity hinders their ability to manage codebases directly. We introduce the Provable Representation Of Original Functionality (PROOF), which manages codebases indirectly via structured specifications. To enable full-lifecycle codebase management strictly through these specifications, PROOF abstracts codebase topology into a hierarchical natural-language representation. To establish absolute trust, the system proves semantic equivalence by reconstructing source code exclusively from this specification. This verified foundation drives maintenance requests, executing code modifications while synchronously updating itself to prevent semantic drift. Experiments on real-world repositories confirm the effectiveness of these specifications. 

---
# Query-Oblivious Coresets for Softmax Attention: Improved Bounds and Efficient Constructions 

**Authors**: Ofek I.Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2609.06327)  

**Abstract**: A query-oblivious coreset for a softmax-attention head is a subset $S$ of the key--value pairs such that attention computed from $S$ alone is within $\varepsilon$ of the full output, in $\ell_2$, simultaneously for every query in a ball. Liberty, Andoni and Kleiner proved that unweighted coresets of size $O(\sqrt d\,e^{\rho+\frac12\log\rho+o(\log\log\rho)}/\varepsilon)$ exist, $\rho$ being the query radius times the centred key radius, against a lower bound $\Omega(\sqrt d\,e^{\rho}/\varepsilon)$, and conjectured that closing the gap needs new techniques. We show it does not. A spherical lift of both balls into one exponential-kernel instance lets the chaining bound of Bozzai and Rothvoss apply directly, and Chevet's inequality splits key from value dimension: unweighted coresets of size $O(e^{\rho}(\sqrt{d_v}+\sqrt{d_k\log(1+\rho)})/\varepsilon)$ exist and are computable in randomised polynomial time, the first with a whole-ball guarantee at the existential size up to $\sqrt{\log(1+\rho)}$. A dimension-free sampling cap $O(e^{2\rho}/\varepsilon^{2})$ completes the envelope. In fixed dimension the logarithm disappears: completing the key ball to a sphere makes the kernel an unweighted Gaussian one, so Tai's diameter-free bound gives $O_{d_k,d_v}(e^{\rho}/\varepsilon)$, ruling out a matching logarithmic lower bound there and answering the Gaussian-restriction case of a question of Bozzai and Rothvoss for the exponential and Hellinger kernels. We restate the Liberty--Andoni--Kleiner bound in the centred convention with a full proof, and show that the one-way communication bounds of Chen et al.\ transfer to query-oblivious coresets, where for $\varepsilon\ll e^{-\rho}$ they are the strongest floors known. The dimensional factor is the price of one signing for all queries: for a single query the discrepancy is $O(e^{\rho})$, dimension-free. 

---
# FrankenReport: Early Exiting in Long-Form Generation Using Expected Value of Computation 

**Authors**: Zhengping Jiang, Gonzalo Ramos, Jina Suh, Shiqian Rachel Ng, Elias Stengel-Eskin, Justin Svegliato, Benjamin Van Durme, Andy Huntington, Sam Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2609.06320)  

**Abstract**: While deep research systems address interactive information-seeking needs impressively, their real-world deployments face latency and resource-consumption challenges. We present FrankenReport, an interface for long-form knowledge-seeking report generation that supports adaptive early exiting per section: it evaluates intermediate outputs during generation and predicts whether further targeted computation will yield significant quality gains. In a simulation study, FrankenReport outperforms random allocation baselines by a large margin (up to 4x) under low budgets and smoothly recovers full-pipeline quality as the budget grows, showing that future quality gains are predictable from intermediate drafts. Through experiments and user studies, we further show that despite varying preferences across users and topics, FrankenReport adapts to simple, natural user feedback as efficiently as methods requiring much costlier supervision such as generated drafts and explicit rationales. 

---
# SignDino: Self-Supervised Sign Language Representation Learning via Temporal-Axis Self-Distillation 

**Authors**: Junyi Hu, Zhewen He, Haomian Huang, Zhenhua Li, Zhifei Li, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2609.06296)  

**Abstract**: Self-supervised sign language representation learning must model two properties not central to natural-image SSL: signs are produced by a small set of anatomically distinct articulators, and their meaning depends on the temporal organisation of those articulators. We introduce SignDino, a self-supervised sign-video encoder that moves the DINOv3 student--teacher recipe from the spatial domain of image crops to the temporal domain of tracked sign streams. Each video is decomposed into left-hand, right-hand, and face streams by a detector-first YOLOv8n+ByteTrack pipeline. A frozen DINOv3 ViT-B/16 embeds each per-frame anatomical crop, while lightweight temporal Transformers, not the image backbone, form the student and EMA teacher. They are trained by temporal DINO self-distillation, frame-level masked-token prediction in the style of iBOT, KoLeo feature spreading, and Gram anchoring of the frame-to-frame similarity structure. This design keeps strong image-level visual primitives fixed and learns only how articulator states evolve across time. We evaluate on sign-to-English translation, isolated sign recognition, and fingerspelling detection benchmarks. Across these tasks, SignDino provides a strong public self-supervised representation and shows competitive or state-of-the-art performance under matched downstream evaluation. 

---
# VDiff-Bench: A Challenging Benchmark for Fine-Grained Image Difference Identification 

**Authors**: Yixin Wan, Tianle Zheng, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.06245)  

**Abstract**: Multimodal Large Language Models (MLLMs) perform strongly on general visual understanding tasks such as visual question answering, yet they often struggle with a basic comparative skill: identifying what has changed between two similar images. We introduce VDiff-Bench, a challenging multiple-choice benchmark for fine-grained Image Difference Identification. VDiff-Bench contains 1,756 four-way questions over image pairs and covers 10 change categories: position, motion, regional image color, overall image color, appearance/disappearance, noise/resolution, texture, substitution/size, OCR/text, and illumination. Each question corresponds to two image inputs with 4 choices: the true difference, two hard negative descriptions, and a "no difference" distractor. To make the task challenging, we specifically curate ground-truth-conditioned negatives that require models to distinguish the actual change from nearby semantic alternatives. Experiments with 11 state-of-the-art open- and closed-source MLLMs show that fine-grained visual comparison remains brittle: models exhibit uneven performance across sources and change categories, with persistent failures on subtle low-level changes like noises and textures. For instance, three 7-8B-scale open-source MLLMs score 52.5-70.6% on semantic changes but only 8.7-33.3% on low-level changes like noise and texture, falsely assuming no changes between two image inputs. Surprisingly, despite strong performance of other closed-source commercial models, Grok 4.3 demonstrate remarkable performance drop on identifying noise and texture differences between images, falling significantly behind large open-source models like Kimi K2.5 and K3. Overall, VDiff-Bench provides a targeted diagnostic for evaluating comparative visual understanding in MLLMs, exposing failures that are not captured by standard single-image vision-language tasks. 

---
# DataFlex-RL: An Evaluation Platform for RLVR Data Policies 

**Authors**: Hao Liang, Mingrui Chen, Hengyi Feng, Meiyi Qiang, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.06107)  

**Abstract**: Data policies for reinforcement learning with verifiable rewards (RLVR) determine which rollouts are used, how strongly they are weighted, and which domains contribute to subsequent training batches. We introduce DataFlex-RL, an evaluation platform for comparing these choices under a common GRPO recipe. Our primary experiment evaluates 13 configurations across 12 matched seeds using Qwen2.5-7B-Base and 12 mathematics, logic, and science benchmarks. Uniform GRPO improves the domain-balanced average accuracy by 7.76 percentage points over the untrained checkpoint. None of the eight rollout-selection or reweighting methods achieves a paired 95% confidence interval that excludes zero relative to uniform sampling, and none of the three adaptive mixtures outperforms a fixed equal mixture at the same level of precision. A corrected 12-seed extension on Llama-3.1-8B-Base places the additional methods on the same score scale as the original controls, but does not reveal a consistent winner in terms of observed mean performance. We also quantify evaluation sensitivity by rescoring nine Qwen2.5-7B-Instruct runs using a math-heavy six-benchmark summary, consisting of five mathematics benchmarks and GPQA-Diamond but no logic benchmark, and comparing it with the domain-balanced 12-benchmark summary. The resulting rankings are negatively correlated, with a correlation coefficient of -0.33, whereas summaries that retain all 12 benchmarks largely agree. Across the controlled settings studied here, changing the data policy measurably changes the training process but does not produce a reproducible improvement over uniform training. 

---
# VERPO: Verified Evidence Regularized Policy Optimization 

**Authors**: Haijiang Li, Chengyu Lv, Yi Zhang, Zhibing Zhang, Rui Qian, Yuchen Zhang, Xiaofan Zhang, Mingshan Wang, Xiaofei Jing, Yu Tong, Cangqi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.06100)  

**Abstract**: Verifiable outcome rewards guide language-model post-training, but sequence-level advantages do not identify which token-level decisions should be preserved or revised. Evidence-conditioned Teachers provide denser supervision by replaying sampled trajectories with privileged feedback. Yet indiscriminate imitation risks transferring formatting or reasoning-style shifts that do not support task success. We introduce VERPO, a Verified Evidence Regularized Policy Optimization framework that treats evidence as a proposal for policy correction while retaining the outcome objective. It separates evidence-free reference restoration from signed token-level evidence corrections. Fisher Evidence Contrast attenuates corrections along an estimated evidence-presence direction. A stopped token-wise ZPD controller scales acceptance according to local reward alignment and Fisher movement cost, while the reference channel remains independent of acceptance. Across five scientific-reasoning and tool-use tasks, the best variant on each backbone exceeds the strongest compared baseline in average score. The averages rise from 0.6826 to 0.6857 on Qwen3-4B, from 0.6895 to 0.7058 on Qwen3-8B, and from 0.4751 to 0.5657 on Llama-3.2-1B. 

---
# ACE: Adapter Consolidation across Experts for Parameter-Efficient Fine-Tuning of MoE LLMs 

**Authors**: Ahin Lee, Sehyun Yun, Joonha Park, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2609.06072)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) of mixture-of-experts (MoE) models commonly attaches a separate low-rank adapter to each expert. This expert-wise design fragments adaptation in three ways: capacity is split across narrow low-rank updates, gradient supervision becomes sparse and imbalanced under sparse routing, and execution is decomposed into many small GEMMs. We find that such expert-wise separation is often unnecessary, as subsets of LoRA adapters become functionally similar during fine-tuning, revealing redundancy among expert-specific adapters. Based on this redundancy, we propose ACE (Adapter Consolidation across Experts), which groups redundant experts and replaces their expert-specific adapters with group-shared higher-rank LoRA modules under the same PEFT budget. ACE further introduces grouped adapter execution, which consolidates fragmented expert-wise adapter computations into fewer, larger group-level GEMMs. Across evaluations covering 12 datasets and four MoE backbones, ACE achieves the highest observed mean accuracy among the parameter-matched PEFT methods on the three backbones with complete baseline coverage, while providing $1.31\times$ to $1.48\times$ wall-clock training speedup over expert-wise LoRA without increasing peak memory. Our code is available at this https URL. 

---
# GradeTrap: Authority Cues in Images Shift VLM Judgments Despite Explicit Instructions to Ignore Them 

**Authors**: Deep Dessai  

**Link**: [PDF](https://arxiv.org/pdf/2609.06058)  

**Abstract**: As vision-language models (VLMs) become increasingly capable and are deployed in consequential real-world settings, they must evaluate evidence independently rather than defer uncritically to human authority. We introduce GradeTrap, a controlled evaluation that places two social cues in direct conflict: a student answer, which should attract sycophantic agreement, and a conflicting answer attributed to a peer, teacher, or official answer key, which should attract authority-based deference. Models produce free-form answers while being explicitly instructed to solve independently and ignore all student answers, feedback, and grading marks. We test the models on 60 synthetic real-world trade-off scenarios. Five neutral trials establish a stable model-relative preference, followed by three repetitions of six experimental cues including controls. On the 45-item common intersection across Gemini 3.5 Flash-Lite, GPT-5.6 Luna, and Claude Haiku 4.5, a generic second-answer control yields 5.4% conflicting-answer selection. Relative to that control, pooled within-item changes show no reliable peer-review effect, a 6.9-point teacher-review effect, and a 19.5-point official-key effect. In contrast, a displayed conflicting student answer alone compared to a displayed student reference answer alone only raises selection from 2.2% to 5.2%. Official-key provenance therefore redirects judgements more than a student answer or the generic second-answer control, despite an explicit ignore instruction and an opposing student answer given along with the official key. Effects vary in magnitude across the three models. 

---
# Structurally Close, Temporally Distant: Measuring Security Exposure in Long-Horizon LLM Agents 

**Authors**: Md Jafrin Hossain, Nur Al Hasan Haldar  

**Link**: [PDF](https://arxiv.org/pdf/2609.05911)  

**Abstract**: Long-horizon LLM agents interact with untrusted content, persistent memory, external state, and sensitive tools. Existing analyses often characterize attacks by the number of execution steps between malicious input and a downstream action. We show that temporal remoteness can overstate security separation in stateful agents. We introduce a provenance-aware execution graph linking agent events through deterministic state, identifier, and tool provenance, and define \emph{influence distance} $\DI$ as the shortest structural path from an untrusted source to a sensitive action. We compare it with \emph{sequence distance} $\DT$, the shortest injection--sink path in the ordered trajectory. Since the influence graph contains every sequence edge, $\DI \leq \DT$; $\Gap=\DT-\DI$ measures the separation hidden by step count. Across 454 injection--sink pairs from 360 long-horizon AgentDojo trajectories over OpenAI's \texttt{gpt-4o-mini} and \texttt{gpt-4o} and Claude's Haiku 4.5 and Sonnet 4.6, $\Gap>0$ for 96.9% of pairs, with a median gap of 9 hops; 91.0% remain decoupled after removing the largest provenance-only edge class. On AgentDojo's banking suite, 33.8% of 231 pairs from 377 trajectories decouple through different provenance mechanisms. Among 274 OpenAI pairs, $\Gap$ does not independently predict attack success after controlling for $\DT$, attack family, and backend ($\beta_{\Gap}=0.066$, $p=.088$). At matched thresholds $k=2,3$, a deterministic $\DI$-based pre-execution gate blocks five attack sinks missed by a sequence-only gate with no additional benign blocking, although the paired gain is not significant ($p=.0625$). Execution structure therefore reveals proximity hidden by step count and can support targeted runtime intervention. We measure candidate influence pathways rather than causal attribution. 

---
# Bait-and-Recover: Poisoning Internal Refusal Signals to Defend LLMs against White-Box Editing Jailbreaks 

**Authors**: Tian Gao, Zhipeng Xie, Yuhao Wu, Junhua Liu, Xin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2609.05794)  

**Abstract**: Open-weight large language models face a low-cost white-box threat from representation engineering attacks. Attackers can estimate refusal directions and search for projection-matrix edits that suppress safety alignment while preserving general capabilities, within minutes on a single GPU and without gradient-based training. We propose Bait-and-Recover, a weight-level defense that places a bait adapter where attackers read activations and a paired recovery adapter at the subsequent layer. Trained via gradient routing, this decouples the observation path from the behavior path. By actively poisoning the residual signal used for measurement, Bait-and-Recover disrupts the attacker's edit search, while the recovery layer restores clean downstream computation. Across four open-weight models, our defense raises the minimum refusal rate against white-box edit searches from 16.25% to 71.75% under a strict behavior-preservation budget (KL <= 0.10), with negligible impact on general benchmarks. By invalidating the core measurement assumption of these attacks, observation-path poisoning offers a practical complement to behavior-level safety training. 

---
# RAPTOR: Role-Aware Private Training for Mixture-of-Experts 

**Authors**: Duc Dm, Khai Le-Duc, Nguyen Do, Minh Son Hoang, Florent Draye, Thai Hoang, Hoang Phuong Dam, Jiarui Liu, Chris Ngo, Terry Jingchen Zhang, Anh Le Duc Tran, Nhat Do Minh, Minh Ngoc Le, My T. Thai, Ran Xu, Silvio Savarese, Mona Diab, Bernhard Schölkopf, Zhijing Jin, Huy L. Nguyen, Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.05770)  

**Abstract**: Differentially private (DP) fine-tuning methods treat sparse Mixture-of-Experts (MoE) models as a single dense block, ignoring that shared layers see all data while experts only see routed records. We identify and formally characterize three resulting failure modes: global clipping suppresses expert gradients, batch-level normalization dilutes sparse expert updates, and fixed privacy noise degrades signal-to-noise ratio on low-load experts. We introduce RAPTOR - a Role-Aware Private Training framework, which alternates shared and expert optimization and targets each failure directly, using expert-specific clipping and noise together with a public expected-owner denominator and a count-independent update schedule that avoids conditioning on private, realized expert counts. We prove the resulting mechanism satisfies $(\varepsilon,\delta)$-DP: because each record is assigned to exactly one owner expert, per-expert mechanisms within a layer compose in parallel, so updating all $E$ experts costs no more, in privacy terms, than updating one, with shared and expert streams composing sequentially across training. We further derive a bias-variance decomposition of the public-denominator estimator showing its bias grows predictably with routing imbalance, yielding a privacy-free rule for selecting which layer to protect from routing entropy measured on a small public corpus. Experiments on Switch Transformer and OLMoE fine-tuning across GLUE tasks, and on DeepSeek-VL2-Tiny, show consistent gains over standard DP baselines across several privacy levels ($\varepsilon$), with the largest margins typically at the tightest budgets. Code and models are publicly available: this https URL 

---
# Emotion as a Distribution: Joint Valence-Arousal Probability Learning for Speaker-Independent Multimodal Emotion Recognition 

**Authors**: Tingyi Lin, Wen-Ren Yang, Kuanwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.05755)  

**Abstract**: Human emotion is graded and frequently mixed, yet most multimodal recognizers collapse it onto a single hard label. We argue the recognizer should instead expose a distribution over affective space. Our text+speech system, alongside its categorical decision, emits a $9\times9$ probability matrix over the Valence-Arousal plane, trained with a two-dimensional Gaussian soft target under a Kullback-Leibler/cross-entropy objective, aimed at counseling support. Evaluation is strict: speaker-independent 5-fold leave-one-session-out IEMOCAP with rotating-session inner validation, headline metrics only on the held-out session. Within one fixed encoder-fusion-head pipeline we compare Transformer and state-space (Mamba-1/2/3) backbones at matched depth and width, at two operating points ($T\approx550$, $T\approx2750$). The featured dual-head system reaches 73.0% $\pm$ 0.3 unweighted accuracy over three seeds (separate rerun: 72.1%), exceeding the Transformer fusion baseline by 3.0 UA points (95% session-bootstrap CI [1.0,4.7]; significant under paired t-test and session-level bootstrap), with no latency or memory advantage at these lengths; swapping the ~1M trainable front-end for frozen WavLM-Large features (learnable layer weights) lifts the same architecture to 76.6% $\pm$ 1.3. Pre-specified controls scope the claims honestly: simpler valence-arousal auxiliaries reproduce the classification lift within noise, and a dedicated regression head tracks the continuous ratings slightly better, so the head's specific value is the normalized affect distribution itself. That distribution recovers the circumplex: its center of mass tracks valence and arousal (CCC 0.66/0.66; predominantly between-class structure, weaker within-class tracking), and its entropy is weakly but consistently linked to categorical rater ambiguity, not dimensional spread. 

---
# EnvCraft: Synthesizing Executable Environments in Agentic RL for Claw-like Agent 

**Authors**: Yirong Zeng, Shen You, Jinhang Feng, Yufei Liu, Xiao Ding, Yutai Hou, Hao Cong, Yuxian Wang, Wu Ning, Wang Xu, Bibo Cai  

**Link**: [PDF](https://arxiv.org/pdf/2609.05576)  

**Abstract**: The paradigm of LLMs has rapidly shifted from passive language interfaces to autonomous Claw-like agents that execute long-horizon tasks across stateful workspaces. While Agentic Reinforcement Learning (Agentic RL) provides a promising path to optimize these agents, its scaling is heavily bottlenecked by the severe scarcity of interactive training environments. Existing synthetic environments are strictly limited to tool-calling endpoints, rendering them insufficient for accommodating the end-to-end real-world demands of claw-like agents. To bridge this gap, we introduce EnvCraft, an automated framework for synthesizing executable environments and scalable training data. Specifically, EnvCraft employs an environment synthesis engine to build sandbox-isolated workspaces, alongside a topology-aware data generation engine to produce coherent task trajectories. Overall, we synthesize 139 interactive environments comprising approximately 20K complex tasks for Agentic RL training. Experiments on Qwen3/3.5 models (8B-32B) show that our method yields gains of up to +11.9% on Claw-style benchmarks and +8.0% on general tool-use benchmarks, with concurrent reductions in inference token cost. The results confirm that synthesized executable environments provide robust and generalizable learning signals for training. 

---
# Grounded Skill Synthesis from Code at Scale for Agentic Intelligence 

**Authors**: Yongqi Tong, Pan Wang, Hang Wang, Jianshe Li, Xin Zhang, Jiang-Ming Yang, Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.05571)  

**Abstract**: Reusable skills give agents transferable procedural knowledge, making scalable acquisition essential for extending agents beyond prior experience. Existing methods face two limitations: trajectory-based synthesis requires interactions with specific environments, while document-derived skills may lack executable evidence and verification. Source code offers a complementary path: it requires no prior agent experience yet provides executable evidence for grounding abstractions. We present Code2Skill, a fully automated pipeline that transforms selected code units into implementation-anchored records of atomic operations, composite workflows, and recurring patterns, then verifies each record through source-body-blind reconstruction and source-aware comparison. Applied to 19,769 popular, actively maintained GitHub repositories, Code2Skill produces CodeSkillBank, a grounded bank of 1,006,822 accepted records with workflow, boundary, provenance, and source-evidence metadata. Across 72 protocol-matched evaluations covering nine model settings and eight benchmarks, models augmented with retrieved CodeSkillBank skills improve by 11.7% on average over matched baselines and outperform them in 57 cases. Under a unified downstream interface, Code2Skill also outperforms trajectory-derived skill banks on all seven shared benchmarks, showing that repository-derived skills can provide useful procedural knowledge before agents accumulate sufficient interaction experience. Skills synthesized from tested AI-generated code achieve a 93.50% pass rate, compared with 93.00% for human-written code, providing initial evidence that the pipeline can expand with the growing volume of AI-generated software. Overall, Code2Skill transforms procedural knowledge embedded in repositories into grounded, verifiable, and transferable agent skills. 

---
# Emergent Goal-Directed Attention in Large Vision-Language Models 

**Authors**: Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.05517)  

**Abstract**: Human observers prioritize visual information according to task goals. Most computational models of naturalistic viewing are gaze-trained for free viewing, leaving open whether goal-directed attention can emerge in systems without gaze supervision. We tested two off-the-shelf vision-language models (VLMs), Qwen3-VL-32B-Thinking and Gemma-4-26B-A4B-it, on 4,887 naturalistic scenes under visual-search and free-viewing instructions. Model predictions were compared with human fixations on the same images under corresponding tasks. Both models aligned more closely with human fixations under matching goals than under mismatched goals. This crossover persisted in target-absent scenes, where alignment could not be explained by simple visual grounding, and appeared in decoder-layer readouts. Furthermore, model-thinking traces were grounded in target semantics during search and in visual prominence during free viewing. These findings show that general-purpose VLMs can generate human-aligned, goal-directed spatial priorities without gaze-specific training, informing theories of goal-directed attention and offering scalable tools for predicting where people look across tasks. 

---
# Reasoning-Aware Compression: Identifying and Protecting Vulnerable Reasoning Circuits for Energy-Efficient LLM Deployment 

**Authors**: Leonard Twagirayezu, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2609.05512)  

**Abstract**: Large Reasoning Models (LRMs) impose substantial energy costs during deployment, yet current compression methods apply uniform quantization across all components, risking damage to critical reasoning circuits. We present a reasoning-aware compression framework that benchmarks quantization conditions across five reasoning benchmarks, GSM8K, FOLIO, MATH-500, ProofWriter, and MuSiQue, with hardware-level GPU energy measurement; profiles per-module INT4 vulnerability across all 196-224 (layer, projection) pairs via a perturbation sweep on a held-out calibration split, then selectively restores the most sensitive circuits to FP16. Three findings emerge. First, INT4 quantization can increase energy by extending reasoning chains; a 25% power reduction becomes a net energy increase on GSM8K. Second, vulnerability is task-dependent: attention projections are more critical for mathematical reasoning, and sensitivity patterns differ by architecture in logical inference. Third, selective compression achieves Pareto-optimal points inaccessible to uniform methods: R1-Qwen-7B Top-10% on ProofWriter gains +12 pp over FP16 at -9.7% energy, validated on held-out data across five reasoning benchmarks. 

---
# SCAFFOLD: Self-Improving Web Agents via Recursive Parametric Skill Abstraction 

**Authors**: Bowei He, Xiaokun Zhang, Meng Ding, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.05511)  

**Abstract**: Web agents need to navigate visually rich, long-horizon interfaces that change across sites, yet most previous agents still learn each task in isolation and discard the procedural knowledge they accumulate. Recent skill-augmented frameworks take an important first step, but they treat the skill library as a flat or two-tier prompt-side cache and offer no principled mechanism for compressing redundancy or composing skills recursively. We introduce \textsc{Scaffold}, a self-improving framework for visual web agents that (i) induces parametric, executable skills from successful trajectories under a multi-instance abstraction constraint, (ii) maintains a recursively composed hierarchy in which higher-level skills invoke lower-level ones, (iii) compacts the library via a minimum-description-length (MDL) criterion and behavioral equivalence checking, and (iv) periodically distills skill-augmented trajectories back into model weights to internalize the abstractions. Across WebArena, VisualWebArena, and a held-out split of Online-Mind2Web, \textsc{Scaffold} improves success rate by $11.1$--$17.2$ absolute points over the strongest skill-augmented baseline and shows monotonic gains across five self-improvement iterations without library collapse. We release the code and documents in the Github \href{this https URL}{repository}. 

---
# AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning 

**Authors**: Zerui Cheng, Jiawei Xu, Huacan Chai, Jiayang Sun, Pramod Viswanath, Maxm Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.05435)  

**Abstract**: Modern language agents are expected to operate over long horizons: they ask follow-up questions, reuse worked examples, handle tool feedback, and adapt to delayed consequences. Most evaluations still reset the agent after a prompt or score only the final state of one trajectory. AhaBench asks a more operational question: when a fixed model receives useful experience, does its later behavior improve under a related evaluation condition where the obvious support has been removed, changed, or delayed? The suite contains three components. Aha-Puzzle tests no-hint exploration after solved hidden-state puzzles; Aha-Euler turns Project-Euler-style mathematical ideas into generated taught/held-out tasks with exact validators; and Aha-Vending, an open-source implementation inspired by Vending-Bench, tests whether a simulated vending agent remains profitable while handling delayed feedback and operational incidents. AhaBench reports a three-part scorecard: Initial Score measures starting competence, Post-Experience Score measures the later empirical outcome, and Learning Lift is their difference. This decomposition is the main empirical message: models that use visible support well, models that reach high post-experience scores, and models that improve most during a run are not always the same. On the common eight-model panel, Claude Opus 4.6 leads aggregate Post-Experience Score at 64.3 and aggregate Learning Lift at +25.8, with Gemini 3.1 Pro close behind at 63.4. The component results explain the split: puzzle traces raise supported scores but often fail to become no-hint exploration behavior; Aha-Euler full teaching reaches 78.6-100.0% while answer-only transfer ranges from 0.0 to 73.9%; and Aha-Vending separates profitable incident handling from bankruptcy and no-order failure. We release benchmark tasks, rubrics, validators, simulator code, and interfaces for evaluating new agents. 

---
# Seeing Without Understanding: Large Language Model Evaluation of Mobile User Interface Quality, Failure Taxonomy, and Architectural Explanation 

**Authors**: Md Rejaul Korim Sadi, Golam Mostofa Naeem, Toufiqur Rahman Tasin, Syed Mostofa Moosa, Mahmudul Hasan Emon, Mahmudur Rashid, Ferdus Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2609.05423)  

**Abstract**: Evaluating mobile user interface quality at scale remains a persistent challenge in software engineering and human-computer interaction. Rule-based heuristic methods offer structural reliability but demand significant engineering effort, while human annotation does not scale to the volume of applications produced annually. Large language models present a promising alternative, yet their reliability for structured UI judgment has not been systematically examined, and the patterns behind their failures remain insufficiently characterized. This paper addresses both gaps. We begin with the complete RICO dataset of 66,261 real-world mobile application screens, from which we derive a refined evaluation corpus of 15,000 screens through a rigorous, literature-guided selection process. Each screen is assessed across seven criteria: structural JSON validity, minimum visible element count, clickable component presence, non-zero layout bounds, image integrity, and perceptual duplicate removal. Against this corpus, we apply a heuristic baseline built from severity-weighted usability signals, normalized layout metrics, and pixel-ratio complexity measures calibrated to real user sentiment. Multiple language models independently rate each screen across usability, layout quality, and visual complexity from structured JSON descriptions and raw screenshots. Dimension-level comparison against the heuristic uses agreement rates, Cohen's Kappa, and confidence calibration. Recurring divergence patterns are organized into a failure taxonomy and interpreted through transformer architectural signatures: MLE plausibility bias, attention misgrounding, and autoregressive over-commitment. 

---
