# Agentic induction of common data elements and information extraction from sarcoma case report abstracts 

**Authors**: Daniel Reichenpfader, Kwaku Appiagyei, Manovah Nadarajah, Ahmed Reda Bahr, Irina Filchenko, Fabio Dennstädt  

**Link**: [PDF](https://arxiv.org/pdf/2611.18506)  

**Abstract**: Common data elements (CDEs) support clinical data comparability but are usually defined manually. We investigated whether large language model agents could derive a CDE schema from sarcoma case report abstracts and apply it at scale. An agent proposed 72,909 candidate CDEs from 24,588 PubMed abstracts; agentic consolidation yielded 1,570 CDEs. We retained 53 supported by at least 50 abstracts and attempted evidence-linked extraction across the corpus. Processing succeeded for all retained CDEs in 79.2% of abstracts, including responses indicating non-mention. Two human reviewers approved 83.0% of retained CDEs for relevance, 67.9% for clarity, and 56.6% for correctness. In 100 sampled abstracts, exact agreement with the value extracted by the LLM was 94.9% among 1,116 pairs with a non-empty answer (abstract-level bootstrap 95% CI, 93.0-96.4%). Errors concentrated in anatomical and temporal fields. These findings support agent-assisted schema drafting followed by expert curation before large-scale extraction. 

---
# Interactive Program Induction with Language Model Priors 

**Authors**: Robin Shing Moon Chan, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2611.18485)  

**Abstract**: Program induction refers to the process in which a learner aims to recover a program from a few given input-output examples. This paradigm broadens programming access by letting users demonstrate the rule-based behavior they want to apply to unseen data, instead of having to implement it themselves. However, recovering a program from finite input-output examples is fundamentally ambiguous, as a finite set of input-output examples is theoretically consistent with infinitely many programs whose output diverges on unseen inputs. We view this problem from a Bayesian perspective, where, given a prior $p^0$ over programs, identifying the target program means concentrating posterior probability mass on it by observing additional input-output examples. In the interactive program induction setting, the learner chooses maximally informative inputs and queries an oracle, such as the user, for their outputs. We derive a greedy query-selection algorithm and show that it terminates after $\mathcal{O}(\frac{1}{\gamma}\log 1/p^0(f^\star))$ informative oracle queries, and returns the target program $f^\star$ up to approximate equivalence when no other class outweighs it under the prior. The algorithm uses a language model as the prior under a PAC-style relaxation of program equivalence, where the query-selection objective is estimated using importance sampling. On the CodeARC benchmark, the algorithm improves accuracy over the LM only agent by up to 22 percentage points while observing up to 10.5 fewer input-output examples. 

---
# Do Verifier Judgments Survive Abstraction? Measuring Verification Transport for LLM Agent Evaluation 

**Authors**: Disha Sheshanarayana, Sai Linisha Reddy Goli  

**Link**: [PDF](https://arxiv.org/pdf/2611.18471)  

**Abstract**: LLM-based verifiers are increasingly used to evaluate agent executions, but their reliability is typically measured under a single representation of the trajectory. In deployed agent systems, the same execution may be exposed through composed tools, delegation boundaries, or compacted histories, changing what execution evidence is directly observable to the verifier. It therefore remains unclear whether correct verifier judgments persist across such abstraction boundaries. We introduce Verification Transport, a reliability test that measures judgment stability under abstraction-induced changes in execution observability. We construct matched native, abstracted, and restored views of the same frozen executions while holding the task, realized trajectory, outcome, ground truth, evaluation criterion, and verifier fixed. Across four agent-evaluation benchmarks, seven verifier backbones, four verification protocols, and three abstraction boundaries, we find substantial transport failure, reaching about 31% under macro abstraction and delegation. Verification protocols exhibit distinct failure modes: direct verification more often produces incorrect verdicts, whereas evidence-grounded and deliberative protocols tend to abstain under limited observability. This suggests that lower transport failure rates do not necessarily imply greater reliability. Restoring the corresponding execution provenance recovers 85.5% of transport failures, while recovery is lower when provenance is lossy or unavailable. These results show that native-view accuracy alone is insufficient to characterize verifier reliability when execution evidence crosses abstraction boundaries. 

---
# LitArc-Bench: Disentangling Text-Specific Reader-Response Dynamics from Position and Lexical Affect 

**Authors**: Mingming Su, Yushan Wei, Sen Song, Xinche Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18412)  

**Abstract**: When model predictions track an unfolding human affect trajectory, alignment may reflect narrative position, affective words, or text-specific reader-response dynamics. We introduce LitArc-Bench to separate these explanations: 39 Chinese-language literary excerpts segmented into seven checkpoints, with 430 questionnaires from 81 adults in two classroom groups (9,020 VAD ratings). We compare three future-blind model runs with leave-one-text-out position priors, lexical affect, and human-human references. Position alone reaches r = .470 for arousal, making raw alignment insufficient. Yet all six V/A transition advantages over a joint position-lexicon control have text-bootstrap intervals above zero. Cross-fitted models match the correct human text better than 38 mismatched texts in all nine model-dimension cells (p < .0001); a text-invariant negative control returns chance (p = 1). On human transitions at least 0.5 points against the position trend, models recover 69-76% of valence directions and 53-68% for arousal and dominance. Under crossed participant-text resampling, all nine identity intervals remain above zero; joint-control advantages are robust for valence and directional but less certain for arousal. High shape agreement still coexists with scale bias. LitArc-Bench identifies which components of literary reader response are recoverable beyond shallow explanations and why shape, calibration, and human variability require separate evaluation. 

---
# Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent 

**Authors**: Run Luo, Kuan Zhang, Qi Deng, Yukun Chen, Zhihao Yang, Yufei Zhang, Renrong Hu, Yi Zhang, Guiyu Ma, Tianyu Fan, Longze Chen, Hao Sun, Jiaming Li, Jiaming Yang, Xiaohao Liu, Tao Tan, Minzheng Wang, Hangtian Jia, Shuai Yuan, Lvdi Wang, Xiao Liu, Chao Ma, Yuting Zou, Yinhe Zheng, Xiaobo Xia, Dawei Jiang, Shaoqing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.18397)  

**Abstract**: We introduce Gamer, a generalist agent framework built for long-horizon tasks in complex interactive environments. Its end-to-end architecture combines perception, memory management, and multi-step reasoning. A shared training and deployment pipeline supports interaction with both conventional 2D graphical interfaces and complex 3D simulated worlds. We first continue pre-training on a dataset constructed from tens of thousands of hours of device-level keyboard and mouse interaction. These trajectories span simulated games and embodied environments, allowing the model to learn environmental dynamics across both physical and virtual domains. Sparse-thinking instruction tuning then substantially reduces redundant computation while preserving the reasoning needed for complex decisions. We train the policy for long-horizon interaction with a stable multi-turn reinforcement learning method that addresses finite context windows, accumulated rollout errors, and sparse rewards. For real-time interaction, a cloud-edge inference engine serves numerous simulated game environments concurrently, supports unbounded-context execution, and keeps online inference latency low. Across multiple challenging 3D open-world games, Gamer demonstrates solid generalization and operational efficiency comparable to skilled players. It also achieves competitive results on a range of conventional 2D game benchmarks. These results suggest that even repetitive keyboard and mouse trajectories can support substantial gains from scaling data and learning algorithms. Training on these trajectories helps Gamer learn complex skills more steadily in environments with both physical and digital properties. This approach offers a path toward generalist agents that can interact across digital and physical worlds. 

---
# Evaluating Human Sensitivity to Gender Biased AI-Generated Content through Human Assessment of Google's Autocomplete Function 

**Authors**: Tamara Serrano, Nikoleta Pantelidou, Raquel Montero, Natalia Moskvina, Paolo Morosi, Evelina Leivada  

**Link**: [PDF](https://arxiv.org/pdf/2611.18371)  

**Abstract**: Large Language Models (LLMs) are increasingly present, but their effects on human users remain underexplored. We tested whether AI-attributed information influences views regarding gender-related statements, a domain where biases are pervasive. Spanish-speaking participants recruited through Prolific evaluated biased statements derived from Google's autocomplete function and human-authored, non-biased gender-related facts. Participants were either informed that some of the material was AI-generated or received no information about its origin. AI attribution did not affect reaction times or agreement judgments. This result suggests that AI-generated content does not automatically raise credibility concerns and may be evaluated with less scrutiny compared to human-produced content. 

---
# Learning Word-Level Explanations of Text Reuse in Latin Literature with Little Annotation 

**Authors**: Julian Schelb, Michael Wittweiler, Marie Revellio  

**Link**: [PDF](https://arxiv.org/pdf/2611.18319)  

**Abstract**: Detecting an intertextual reference is only the first step; philologists also need to know how a later author reworked the source. We explain a reuse pair word by word: each word of the reusing passage is first aligned to its source word, then labeled with the operation that transformed it, such as a copy, a change of inflection, or a substitution. Such word-level operations are expensive to annotate, and only a few corpora annotated at this level exist for text reuse in Latin literature. We describe a procedure that needs only a limited number of annotations: synthetic pairs, built by applying known operations to Latin text, train a first model, and active learning chooses the pairs an annotator corrects. Among several architectures, a model that links and labels together performs best: a pointer network that, for every reuse word, chooses a source word and names the operation in the same step. We show that with active learning, 200 pairs chosen by uncertainty are enough for the model to match one trained on every annotated pair of the 1,490 references of the Loci Similes benchmark. We release our code, word-level predictions for all references, and an interactive demo. 

---
# GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models *Author(s) 

**Authors**: Jiayu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18279)  

**Abstract**: Whether a language model's answer is correct often depends on context: the same value can be right under one jurisdiction, date, or legal definition and wrong under another. We introduce GT-FENG (General Theory of Factual Evaluation and Normative Grounding), the first general framework to unify contextual/temporal correctness and rule-based error attribution across law, policy, and other rule-governed domains. GT-FENG checks answers against the applicable rules and identifies documented rules that match incorrect answers. It shows that when multiple rules give the same value, the answer alone cannot tell which rule the model used. In Medicaid income-limit answers from two models, documenting the asked state's own records reduces the number of wrong answers that match only other states' values by 68-78%, and more than half of stable wrong answers then match two or more documented sources. GT-FENG provides a foundation for evaluating language models in rule-governed domains and for future benchmarks and evaluation toolkits. We release GT-FENG as an open-source Python package, gtfeng. 

---
# MamaReason: Unfinished Reasoning and Answer Recovery in Clinical Language Models 

**Authors**: Thanni Adewuyi, Adepoju Jeremiah Moyondafoluwa, Dorcas Amapu, Dieko Whenu, Sylvia Essessien, Sandra Oko-Oba, Pelumi Onafadeji, Lily Aimas-Love, Samuel Omolaiye, Abiodun Adereni  

**Link**: [PDF](https://arxiv.org/pdf/2611.18264)  

**Abstract**: Answer continuation often returns a diagnosis already stated in unfinished clinical reasoning. We study this behavior in MamaReason, adapted from dense Gemma 4 31B, on MamaBench's 217 paired maternal and child-health presentations. An explicit-commitment extractor identifies a diagnosis in 75 of 94 failed RL first passes and 75 of 93 failed SFT first passes. The continuation matches that diagnosis in 67/75 RL cases and 58/75 SFT cases, using diagnosis strings alone. Reading these commitments without additional generation recovers 30 of the 40 RL pairs restored by continuation under the archived correctness labels. Recurrence controls distinguish routine mentions from repeated answer selection. An offline second-commitment trigger measures the tradeoff between earlier readout and paired accuracy. The resulting decomposition separates diagnostic content already present in a trace, completion of an accepted answer, and paired diagnostic correctness. For clinical language-model evaluation, these are distinct outcomes that become visible when the same reasoning trace is examined through multiple readouts. 

---
# NER-D: A cross-domain benchmark for German NER 

**Authors**: Marc Brinner, Philipp Cimiano, Karolina Hanna Czok, Regina Stodden, Viju Sudhi, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2611.18257)  

**Abstract**: We present NER-D, a cross-domain benchmark for German NER comprising 13 datasets spanning 9 domains, including General/Politics, Wikipedia, Legal, Medical, Historical, Financial, Agriculture and Archaeology. We examine the performance of both proprietary LLMs and state-of-the-art small open-source models using normal prompting and further examine the performance gains achievable via in-context learning and LoRA fine-tuning. Our results show that without any sort of optimization a proprietary model (GPT-5.4 mini) outperforms the smaller open-source models on 6 out of 13 datasets, but LoRA fine-tuning enables the smaller models to substantially outperform the proprietary baseline. We further analyze the influence of the domain and the tagset used in each task on the performance achieved in the different evaluation settings. Our benchmark will be available as a resource for the development of future German NER methods on both general and specialized domains.\footnote{We make our code, data, and models publicly available upon acceptance of this paper. 

---
# A Multi-Layered AI Pipeline for BPMN Generation from Natural Language 

**Authors**: Alexandru Rancea, Ionut Anghel, Tudor Cioara, Ovidiu Anchidin, Ancuta Nemes  

**Link**: [PDF](https://arxiv.org/pdf/2611.18157)  

**Abstract**: The automated transformation of unstructured natural language process descriptions into Business Process Model and Notation (BPMN) diagrams remains a challenging task, due to linguistic ambiguity and the difficulty to detect non-adjacent control-flow relationships. This paper proposes a solution for generating BPMN models from natural language descriptions through a multi-layered pipeline that integrates transformer-based architectures, Large Language Models and ontology population. The proposed method starts with natural language descriptions and separates textual processing, control-flow relation extraction, and structural validation into dedicated components. Entity extraction is performed using a fine-tuned RoBERTa-based transformer model while a control-flow engine captures non-adjacent relationships through LLM-based inference. To improve generalization under limited available data, the approach uses a dynamic few-shot strategy with con- trolled template input construction, enabling improved generalization under limited data availability. Additionally, an ontology population mechanism is introduced to represent the generated process models as semantic knowledge structures. For evaluation purposes we have used from the literature and industry reference processes. Additionally, we have validated the proposed method on a real-world heart disease patients transitional care process, capturing practical constraints and multi-actor collaboration, and serving as an expert-defined benchmark scenario. Experimental results demonstrate strong performance in the relation extraction task, achieving an accuracy of 90.7%, as well as maintaining structural alignment between the generated and reference BPMN models across multiple evaluation scenarios scenarios. 

---
# Folding Language Models into Themselves 

**Authors**: Ziyang Zhang, Yubin Jing, Yuyao Li, Haofan Wang, Yichen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2611.18113)  

**Abstract**: Knowledge distillation usually trains a compact student from scratch, which needs a large amount of data even though the teacher's weights already encode most of what the student has to learn. In this work, we explore building the student directly from the teacher's weights. We propose Concertina, which folds every group of N adjacent teacher layers into one shared block that is executed N times. Each loop step carries a low-rank correction initialized from the layer it replaces, and a per-token gate mixes these corrections, to let every token recover the behavior of the original layers from a single shared block. We further design a distillation data recipe that balances general, mathematical, academic and targeted multiple-choice data, to recover all capabilities of the teacher together rather than one at the expense of another. On Qwen3-4B, a 2.27B Concertina model distilled on 22B tokens retains about 90% of the teacher on commonsense and mathematical reasoning and 84% on average across eight benchmarks. It outperforms same-size dense and layer-sharing baselines, as well as Qwen3-1.7B, which is trained on over a thousand times more data. These results suggest that a compact model need not be retrained at scale, since folding the teacher into itself and distilling it on the right data recovers most of its capabilities. Code is available at this https URL. 

---
# Contagious Fingerprints: When LLMs Inherit Another Model's Identity 

**Authors**: Zailong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.18104)  

**Abstract**: The fingerprint by which a classifier recognizes the LLM behind a text is usually treated as a fixed property of the model. We show that it is contagious: a model that continues a conversation begun by another model inherits that model's fingerprint. This undermines audits of API relays, which can serve the first turns of a conversation with the model a customer pays for and silently hand the rest to a cheaper one. After such a handoff, detection of the substitute by a text classifier drops from 0.97 to 0.72 (DeepSeek replacing Claude Sonnet) and from 0.87 to 0.54 for a same-vendor downgrade (Haiku), and retraining on handoff data repairs only substitutes seen in training. Preregistered experiments trace the contagion to imitation of the history's style. History also moves a model away from its own fingerprint: active probes are not camouflaged, but when calibrated on fresh conversations they flag \emph{honest} models on long contexts at rates up to 1.00, and a standard MMD equality test rejects all nine models we test. Both failures come from comparing a history-dependent reply with a history-free reference. We propose \textsc{Lineup}, which samples references from the claimed model \emph{on the same history} and tests the audited reply by its rank among them. \textsc{Lineup} controls false positives by exchangeability and needs no fitted threshold or log-probabilities. With contemporaneous references on two closed models, its false-positive rate is 0.056 (preregistered bound 0.10), and it detects 0.81 of substitute sessions across six pairs after four audited turns. 

---
# VEPO: Mitigating Multilingual Hallucinations in LVLMs via Visual Evidence-guided Preference Optimization 

**Authors**: Chengpeng Fu, Xiaocheng Feng, Wenshuai Huo, Yichong Huang, Baohang Li, Duyu Tang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18096)  

**Abstract**: Large vision-language models (LVLMs) achieve strong multimodal performance, yet they remain prone to object hallucination---generating answers that are linguistically fluent but visually ungrounded. This issue is especially severe in low-resource languages, where weaker cross-modal alignment causes models to rely more on language priors than on image evidence. Existing remedies either require costly multilingual annotations or external translation/reward models, or intervene at inference time in ways that can distort the output language. We propose VEPO(Visual Evidence-guided Preference Optimization), a training-time method that transfers reliable visual grounding from English to other languages without external models. VEPO first localizes key visual evidence using English queries via mid-layer cross-modal attention, then constructs preference pairs that share the same correct answer but differ in evidence sufficiency (full image vs.\ key-region masked image). Direct preference optimization on these pairs encourages low-resource inputs to prefer answers supported by complete visual evidence. Experiments on multilingual POPE and MME with Chinese, French, and Hindi show consistent hallucination reduction and perception gains, with the largest improvements on the lowest-resource language. Analyses further confirm that VEPO sharpens question-relevant attention and that evidence-guided masking outperforms random masking. 

---
# Agent-as-a-Judge 

**Authors**: Runyang You, Hongru Cai, Caiqi Zhang, Qiancheng Xu, Meng Liu, Tiezheng Yu, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.18067)  

**Abstract**: Artificial intelligence has progressed from task-specific systems toward general models spanning perception, generation, reasoning, and autonomous interaction, making reliable evaluation essential for comparison, deployment, and post-training feedback. At scale, modern evaluation pipelines increasingly rely on LLM-as-a-Judge to produce scores, preferences, and critiques; in its fixed-context setting, the judge relies on the supplied evaluand and evaluation materials. Yet a reliable verdict may require evidence beyond the supplied evaluand and context---either because critical environmental states remain unrecorded in the static execution log, or because controversial factual claims require external verification. Passive judges cannot obtain that missing evidence before deciding, especially when evaluating agent trajectories, executable artifacts, or domain-specific reports. Recent work therefore develops active evaluators that seek missing evidence and update their grounds for judgment before deciding. We bring these advances together through Agent-as-a-Judge, focusing on how evaluators construct and control checking processes within the broader LLM-based judging landscape. We define the scope through evaluation-directed checks and the use of their results, covering both structured workflows and observation-driven control. We organize the design space by multi-agent collaboration, tool integration, memory, and planning, and show how these mechanisms supply the checks and evidence on which a verdict rests. We consolidate published scores and reproduce missing entries for key agentic judges; the comparison reveals uneven agentic gains and motivates dedicated agentic evaluation. We conclude by characterizing application demands and open challenges in cost, reliability, safety, and governance that condition trusted deployment of agentic judges at scale. 

---
# FEND: Far-End Detection in Outbound Voice AI Calls 

**Authors**: Abhishek Kumar, Suraj Agrawal, Aashraya Sachdeva  

**Link**: [PDF](https://arxiv.org/pdf/2611.18050)  

**Abstract**: When a conversational Voice AI agent places an outbound call, the far end may be a live person, a voicemail greeting, an interactive voice response (IVR) system, or increasingly another AI system answering on the callee's behalf. Standard Answering Machine Detection (AMD) reports only whether a human or a machine picked up, a design meant for human call centres, where one machine is as uninteresting as another. That single bit does not tell a Voice AI agent what to do, since leaving a voicemail message, navigating an IVR and holding a short exchange with another AI are three different actions and a two-valued label cannot distinguish them. In 1,000 randomly sampled calls from our own outbound deployment, a live person answered 48.1%, a voicemail system 44.6%, an AI 6.9% and an IVR 0.4%. Our proposed framework, FEND, reads the far end's audio and its streaming transcript and runs one detector for each automated class (voicemail, IVR, AI callee), with human the default when nothing fires, returning a single label within 20 seconds. On a public benchmark the voicemail components reach 100% accuracy without ever misclassifying a live human, where Asterisk's AMD misclassifies 30 to 66% of them. For the AI callee, acoustic features do not transfer across recording conditions, but speech timing does, separating AI from human at 0.952 ROC-AUC, because synthesised speech leaves far shorter gaps between words. On production traffic, separating a live person from any automated far end reaches F1 0.979, and only 2.9% of live people are given an automated label, the error that matters most for an agent acting on the result. 

---
# Toward Universal Translatability: On the Statistical Ontology of Generative AI 

**Authors**: Katia Schwerzmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.17963)  

**Abstract**: This paper argues that the problem of translation functions both as a point of origin and as a normative horizon for contemporary generative AI. In current discourse around ''world models,'' their attainment is framed as the achievement of internal consistency, cross-modal alignment, and physical grounding. Taken together, these demands converge on an ideal of the model as a universal translation machine capable of establishing lossless correspondences across linguistic, visual, and physical representation spaces. I show that this ideal presupposes a metaphysical commitment that I call a statistical ontology: the belief that reality has an underlying, computable statistical structure that can be recovered through sufficient data and modeling, rather than being shaped and accessed through situated human interpretation and labor. Tracing this commitment back to Warren Weaver's 1949 Memorandum on machine translation, I outline how translation was historically recast as code deciphering and, later, as the alignment of probability distributions across structured outputs. Finally, I contrast this paradigm with approaches in Translation Studies that foreground opacity, power asymmetries, and the translator's positionality. I argue that generative AI's statistical ontology is pre-Kantian, pre-critical and obscures the infrastructures and human labor that make machine translatability---as alignment---possible in the first place. 

---
# In the Eye of the Beholder: Human Label Variation in Textual and Visual Entailment 

**Authors**: Monica Riedler, Jasmin Orth, Raffaella Bernardi, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2611.17954)  

**Abstract**: Human Label Variation (HLV) is increasingly recognised as an informative signal that reflects meaningful differences in how people interpret the same input. While HLV has been studied extensively in Natural Language Inference (NLI), much less is known about HLV in Visual Entailment (VE; Xie et al., 2019), where the textual premise is replaced by an image, keeping the hypothesis and label space fixed. We introduce VariNLI-VE, a dataset for studying how HLV changes between paired NLI and VE instances. We re-annotate a subset of originally neutral SNLI (Bowman et al., 2015) instances and their visual counterparts, with the same annotators evaluating both the textual and visual versions. Each annotation includes a label, confidence score, and explanation, yielding 2,250 annotations. Labels and explanations are collected jointly, providing ecologically valid explanations. Because the image may reveal information not stated in the text, one might expect disagreement to decrease in VE. Instead, we find no clear overall reduction: disagreement decreases for some instances while it increases for others. Explanations reveal further variation, as annotators can retain the same label while changing their rationale. When evaluating vision-language models (VLMs) on the paired data, we find lower accuracy, weaker alignment with human label distributions, and generally higher uncertainty in VE than in NLI. 

---
# Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent 

**Authors**: Cosimo Galeone, Anna Ettorre, Ginevra Mariani, Minsu Park, Giuseppe Ettorre, Daniele Ligorio  

**Link**: [PDF](https://arxiv.org/pdf/2611.17939)  

**Abstract**: Can you trust a finance agent you did not build? We take the reference agent shipped with the Vals Finance Agent Benchmark v2, leave it untuned, and ask four questions a leaderboard score does not answer. Is the score stable? Seven identical runs on 18 questions pass between 105 and 123 of 149 rubric criteria, and the bill varies by up to 17%. When the agent fails, who owns the fix? We classify 20 failed trajectories, from three runs of nine held-out questions, by the layer whose owner would remedy them, under a taxonomy frozen in advance: 13 belong to the agent's instructions, 3 to its tools, 4 to the model, none to its data. Can that diagnosis be trusted? Four blind readers from three model families, given only the traces, name the same fault as our LLM analyzer in 94% of cases, the rate at which they agree with one another (93%). Can a change be trusted? An optimization loop reads the traces, makes one targeted change per round and keeps it only if it clears a fixed threshold. Under a cost objective its final configuration costs 0.34x the baseline on the held-out questions (95% CI [0.22, 0.58]) at accuracy within noise. Under an accuracy objective it gains 14 points on the development questions and 1.5 on the held-out ones (CI [-5.3, +7.2]). Judged against the measured run-to-run noise, under a rule written before the test was opened, the first is a real saving and the second is not. This is a case study of one agent and nine test questions, not a general result. 

---
# MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration 

**Authors**: Bing Wang, Mingyang Wang, Yuanhang Mi, Yue Xu, Kaibo Su, Li'ang Xiao, Chunguo Wu, Changchun Li, Ximing Li, Kun-Feng Qiu, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17931)  

**Abstract**: In mineral exploration, verifying geological hypotheses relies on designing a variety of geochemical experiments. While large language model (LLM) agents increasingly automate downstream geoscientific data analysis, a crucial upstream capability remains largely unexplored: automated geochemical experimental design, i.e., deciding which experiments to conduct before committing to costly laboratory work. Evaluating this capability of LLM agents is difficult for two reasons. First, geochemical experiments exhibit high methodological diversity and subjectivity: to test the same hypothesis, different geologists may choose different minerals and instruments, so no single protocol can serve as the reference answer. Second, open-ended experimental plans resist representation as structured labels, which makes objective scoring difficult. To address these challenges, we introduce MineralExp, a benchmark that reformulates geochemical experimental design as a hypothesis-based experiment selection problem, i.e., a multi-label classification problem. We abstract concrete instrument names and procedural details into a shared taxonomy of 19 experimental types across five mineral-system processes: Source, Transport, Deposition, Modification, and Preservation. This taxonomy yields deterministic multi-label targets while accommodating diverse experimental protocols. Constructed from published ore-deposit literature and verified by domain experts, MineralExp comprises 277 cases and 1,572 reference experimental plans. Given a geological hypothesis and initial context, an LLM agent selects the relevant tasks and generates concrete experimental plans, which are scored by multi-label classification metrics and an LLM-as-a-Judge metric. The experimental results show that current agents generally achieve high recall but low precision, tending to propose more experiments than are necessary to test the hypothesis. 

---
# UltraBERT-JEV: Encoder models for System One Applications 

**Authors**: Astarag Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2611.17867)  

**Abstract**: We study encoder models that pick one of several options in a single forward pass, without generating text. This technical report compares recipes for fine-tuning UltraBERT-380M with one mask per option. The four recipes differ in how an option is scored and how the scores become probabilities; one follows the expert gate of DeepSeek-V3. Trained on the same data as Qwen3-0.6B with the LLM-as-Jev recipe (67.6% test accuracy), the encoder reaches 69.7% to 72.1%. Scoring an option by "yes" minus "no" instead of "yes" alone gives the larger gain, from 69.7% to 71.3% on the test set and from 57.8% to 59.6% on external benchmarks (56.1% for the decoder). Sigmoid normalisation adds 0.8 test points; a square-root softplus adds none. One request at a time on an NVIDIA A100, the encoder takes 29 ms per decision against 68 ms, but uses more peak memory (2.8 against 2.2 GiB). An auxiliary head, trained on human-marked evidence and on labels selected with a reward adapted from SelfCite, predicts in the same pass the sentences annotated as support for the answer. 

---
# BLiMP-RO: A Benchmark of Linguistic Minimal Pairs for Romanian 

**Authors**: Radu-Stefan Popa, Valentin Feroiu, Adina Camelia Bleotu, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2611.17836)  

**Abstract**: Evaluating grammatical knowledge in language models requires benchmarks that cover diverse linguistic phenomena and are validated against human judgments. We introduce BLiMP-RO, a Romanian minimal-pair benchmark covering 50 grammatical phenomena across ten domains. It comprises 493 linguist-constructed sentence pairs, evaluated by native speakers in five experiments, and 5,942 additional semi-automatically generated, human-validated pairs. We further investigate grammatical generalization under limited linguistic input by training GPT-2 and GPT-BERT models from scratch on Romanian BabyBabelLM data and a newly collected corpus of 10 million words from child-targeted video captions. Increasing caption training data yields approximately 9 percentage points of accuracy improvement per tenfold increase in corpus size. At matched data sizes, BabyBabelLM generally outperforms video captions, while GPT-BERT consistently outperforms GPT-2. Our findings highlight how training data quantity, composition, and model architecture shape grammatical generalization in Romanian. 

---
# PaperCritic: Training AI Reviewers to Assess the Soundness of Papers 

**Authors**: Hyun Ryu, Ji Yong Cho, Eunho Yang, Moontae Lee, June Yong Yang, Sean Welleck, Carolyn Rose  

**Link**: [PDF](https://arxiv.org/pdf/2611.17834)  

**Abstract**: As AI conference submissions grow rapidly, AI reviewers are being considered as an alternative. However, simply imitating human reviewing is problematic because reviewing serves multiple purposes, including evaluating papers' soundness, interestingness, and clarity, the last two of which are subjective. In this work, we focus solely on evaluating soundness, the most important and objective criterion, and define the learning objective as jointly maximizing (1) precision, based on the soundness and substantiveness of review points, and (2) recall, the coverage of high-quality reference review points. Under this problem formulation, we construct a quality-labeled review point dataset, PaperSound-23K, using 1K ICLR submissions and 23K human-written and LLM-generated review points. Using this dataset, we train an AI reviewer model, PaperCritic, through reinforcement learning with the F1 score of precision and recall as the reward. Our experiments show that PaperCritic outperforms frontier models and models trained with other AI reviewer training methods on an in-domain test set. Surprisingly, learning to review ICLR submissions also transfers to reviewing Nature-family papers and even to out-of-domain reviewing tasks (e.g., code review, legal contract review, and feedback on story writing). 

---
# AI Reviews Are Biased Towards the Appendix 

**Authors**: Saptarshi Ghosh, Mohammad Saim, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17811)  

**Abstract**: Large language models (LLMs) are increasingly used by reviewers to highlight weaknesses during the reviewing process, and their outputs often shape the reviews that humans submit. Prior work studies how LLM reviews align with human ones, yet we know little about which parts of a paper these models actually criticize. In this work, we investigate 2,000 papers from ACL, EMNLP, COLM, and ICLR, and show that the latest GPT and Claude models draw about 50% or more of their review weaknesses from the appendix. This behavior holds across the API, coding-agent, and web access methods for using LLMs to review a paper. In contrast, we find that humans draw only 8.4% of their weaknesses from the appendix, and these weaknesses primarily revolve around a paper's presentation and content related to its core idea. Meanwhile, LLMs draw weaknesses from deep in the appendix that concern fine-grained technical details not central to the paper's contributions. We anticipate that our analysis will contribute to the scholarly discourse on AI reviews. Our code is publicly available at: this https URL. 

---
# CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents 

**Authors**: Lihuan Li, Ruiyi Yang, Jiru Wang, Yifan Duan, Xiachong Lin, Hao Xue, Yang Song, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.17808)  

**Abstract**: Long-horizon LLM agents can coordinate tools and applications to complete complex, multi-step tasks. As their interaction histories grow, effective context management becomes essential for reliable and efficient execution. Existing methods reduce context through summarization, pruning, or history selection. However, shorter contexts do not necessarily support better decisions: summaries can omit exact details needed for later actions, while retaining entire interactions can carry substantial irrelevant information. We propose CarryOn, a lightweight context manager that pairs compact execution state with fine-grained, source-linked evidence. A learned selector and adaptive retention policy keep relevant evidence available without additional LLM calls for context management. We introduce two complementary evaluation metrics: $S_\alpha$ prioritizes task success while accounting for resource use, and normalized cost per success (NCS) measures the resources spent per successful task. Across AppWorld, OfficeBench, and Tau2 Telecom with three LLM Actors, CarryOn achieves 75.7% macro-averaged task success, exceeding the strongest baseline by 6.7 percentage points. It also achieves the best average scores on both metrics. 

---
# RAG-FGO: Dynamic Optimization of Query-Rewriting Agents for Generative RAG 

**Authors**: Qigan Sun, Xudong Wang, Chaoning Zhang, Chenghao Li, Jiaquan Zhang, Jiehui Xie, Sungyoung Lee, Jie Zou, Ning Xie, Peng Wang, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.17788)  

**Abstract**: In retrieval-augmented generation (RAG), query rewriting serves as a core interface that aligns user intent with external knowledge bases, and its effectiveness directly determines overall system performance. However, generative retrieval faces three critical challenges in real-world deployment: noisy retrieval feedback, the difficulty of jointly optimizing discrete prompt templates and continuous decoding-control parameters, and the limited generalization of static rewriting strategies under semantically complex scenarios. To address these issues, we propose RAG-FGO, a dynamic optimization framework for generative retrieval strategies. RAG-FGO formulates query rewriting as a non-differentiable black-box optimization problem in a mixed discrete-continuous space, and represents prompt templates and decoding parameters as adaptively optimizable query-rewriting agents. The framework performs global search to enable semantic jumps and structural reorganization that escape local optima, while applying local perturbations for fine-grained alignment and precise refinement. In addition, RAG-FGO introduces an elite-preserving memory pool that accumulates high-quality strategies across iterations, thereby mitigating the impact of stochasticity and noisy feedback on optimization stability. Experimental results on a diverse suite of benchmarks demonstrate the effectiveness and robustness of the proposed method, supporting its applicability to complex semantic retrieval tasks. 

---
# LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent 

**Authors**: Ibrahim Memon, Armin Karamzade, M. Hanif Jhaveri, Roy Fox  

**Link**: [PDF](https://arxiv.org/pdf/2611.17780)  

**Abstract**: Frontier LLM agents that query databases with tools are among the most accurate text-to-SQL systems, but per-query calls to a hosted model are slow, costly, and expose sensitive organizational queries to external APIs. For example, a Claude Opus 5.5 agent serving 10,000 queries a day costs over $70,000 a year at list price. To what extent can local hardware already owned by an organization offload this burden without compromising the agent's accuracy? We present LuthorSQL, a non-autoregressive 7B text-to-SQL model that serves as a rapid System 1 gate for a slower LLM agent (System 2), escalating queries only when uncertain. LuthorSQL makes a single forward pass to predict slot scores over a bounded grammar for query plans, compiles and runs up to 32 candidate plans, and chooses an answer using a trained validator. On its own, LuthorSQL reaches 68.03% execution accuracy on BIRD's hidden test set at 0.84 s per question on a single GPU, maintains its accuracy in 9 GB with 8-bit quantized weights, and transfers zero-shot to Snowflake, BigQuery, MySQL, PostgreSQL, and DuckDB within 3.1 points of its native SQLite accuracy. Combined with Claude Opus 5.5 in a pre-registered confidence cascade over 227 questions on databases published after Opus's training cutoff, LuthorSQL answers nearly half of all questions locally and cuts API cost by 33.3% at the same accuracy. Finally, contamination probing shows that Opus reproduces 48-92% of public benchmark gold queries verbatim without seeing the schema; however, its tool-assisted accuracy holds on renamed schemas and reaches 95.6-97.4% on post-cutoff databases. Code and data: this https URL 

---
# Activation Steering the Nan-Shan in Typhoon 

**Authors**: Megan Wei, Damon Pham, John Cayley, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2611.17768)  

**Abstract**: Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to reflect on human culture's current AI moment. We prompt a large language model, seeded with Conrad's original language, to feed endlessly on its own output, and explore when and how the text becomes increasingly alien and unwieldy as time progresses, including deviations from the narrative, digressions into the model's own instructions and thinking, and long periods of incoherent rumination. We employ methods from activation steering, including a novel adaptive steering method we introduce, to control the rhythm of the narrative and the model's digressions over long stretches of generation. The simultaneous efficacy and unpredictability of this steering showcases the tension between the perceived interpretability of AI, and the reality of a system that remains outside our comprehension and control. 

---
# GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks 

**Authors**: Suhaas Garre, Sushant Mehta, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.17766)  

**Abstract**: We introduce GDP_xlsx, a benchmark of 70 realistic workplace requests written by professionals in 13 domains, each paired with an expert rubric of binary criteria and with the complete workbook or CSV file it refers to. Together the files hold 2.18 million populated cells, and the rubrics 791 criteria. Agents work in a container with spreadsheet tools, and an agentic judge that can open the file applies the rubric. Completeness and exclusivity criteria require set-valued answers to be found in full and with nothing extra. We rank models by strict pass@1, the share of attempts that meet every criterion. Across 20 model configurations, including frontier models from Google, Anthropic, and OpenAI, the top-scoring one, Gemini 4 Argon, passes only 38.3% of attempts, and the median score is 19.3%, leaving wide headroom. We release the evaluation harness and three public tasks; the other 67 are held out to protect the leaderboard from contamination. 

---
# See Every Block Once: Variance-Reduced Post-Training for Masked Diffusion Language Models 

**Authors**: Yunzhen He, Hiroaki Yamagiwa, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2611.17754)  

**Abstract**: Policy gradient methods are widely used to align large language model outputs with human preferences, but the log-likelihoods they require cannot be directly computed in Masked Diffusion Language Models (MDLMs). MDLM post-training therefore relies on a surrogate objective based on the Evidence Lower Bound (ELBO), and recent work estimates a blockwise variant of this objective via Monte Carlo methods to support blockwise generation. However, existing estimation schemes randomly sample the block index, so they may evaluate the same block more than once and leave some blocks unevaluated. This introduces variance arising from block selection into the estimates. In this work, we propose a full-block estimator that evaluates every block exactly once and averages the results. We theoretically show that the proposed estimator remains unbiased while eliminating the between-group variance arising from block selection, thereby achieving lower variance than the baseline for the same sample size. In experiments on five benchmarks spanning mathematics, planning, and coding, our method achieves the highest accuracy on four benchmarks and outperforms the random block sampling baseline by an average of 12.79 points. 

---
# What Do Multilingual Self-Supervised Speech Encoders Know About Sentence Meaning? 

**Authors**: Kevin Krahn, Eric Fosler-Lussier  

**Link**: [PDF](https://arxiv.org/pdf/2611.17686)  

**Abstract**: Self-supervised speech models (S3Ms) have been scaled up to thousands of languages and millions of hours of audio, yet little is known about how these models structure semantic knowledge across languages. In this work, we probe sentence-level semantics in massively multilingual S3Ms across 76 languages, using linear maps from frozen speech representations into a frozen text embedding space, evaluated by cross-modal retrieval. Sentence meaning is linearly accessible but highly uneven across languages and resource tiers. Cross-lingual probes can match language-specific probes on languages never seen in probe training, but per-model monolingual decodability does not predict transfer. Within the OmniASR wav2vec family, scaling from 300M--7B disproportionately benefits high-resource languages, and ASR tuning sharply improves semantic decodability while making representations more language-specific. However, semantic content is entangled with acoustics, inflating apparent performance on single-speaker corpora. 

---
# Small Open-Weight Language Models for Crowd Decision-Making: Capabilities and Limits of Output-Level Fusion and Linguistic Aggregation 

**Authors**: David Herrera-Poyatos, Cristina Zuheros, Carlos Peláez-González, Andrés Herrera-Poyatos, Rosana Montes, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2611.17589)  

**Abstract**: Small open-weight language models can run crowd decision-making (CDM) on an institution's own infrastructure, which keeps the checkpoint, the inference settings and the opinions under its control. Their judgements vary across repeated queries; whether repeating and aggregating them, often through the 2-tuple linguistic representation, settles the decision rather than the labels is untested. Five open-weight models of 8-36B total parameters on one consumer GPU and a hosted reference classify 474 TripAdvisor reviews of four London restaurants ten times each; three open-weight models repeat the protocol on 600 reviews of twelve Madrid restaurants. Continuous evaluations preserve resolution: snapping each review leaves 5 distinct values against 21-65, and snapping at every level at most two collective values. Sensitivity to the participants persists once run-to-run variability is reduced: in London, redrawing the reviewers keeps the winner on 47-81% of replicas against 86-100% for redrawing the runs. With simultaneous intervals over every pair of alternatives under crowd resampling, no system on either corpus names a single winner: in London every system keeps all four restaurants as candidates, and in Madrid the candidate sets hold five to eight of the twelve. Extra runs and output-level fusion stabilize the winner but do not remove that sensitivity, and the fusion does not beat its best member. The three larger open-weight models name the hosted reference's London winner and at least match its per-review correlation with the reviewers' stars, so small open-weight models can support local CDM, reported as candidate sets with margins rather than forced rankings. 

---
# CamSpokenNLI: A Pilot Resource for Cross-Variety Transfer Evaluation in Cameroon Pidgin English and Camfranglais 

**Authors**: Tiani Pekins Ebika  

**Link**: [PDF](https://arxiv.org/pdf/2611.17573)  

**Abstract**: Cameroon Pidgin English (a West African Pidgin English variety) and Camfranglais are widely used in Cameroon, yet to our knowledge, neither has a dedicated spoken Natural Language Inference (NLI) resource. We introduce CamSpokenNLI, a pilot dataset of 114 scenario-matched spoken NLI items in each variety (228 in total), recorded by one native speaker from Buea. Each item pairs premise audio and text with a hypothesis, an NLI label, and English glosses. The author assigned labels, so the dataset is an exploratory resource rather than a mature benchmark. The NLI experiments use the reference text; the audio is evaluated separately. XLM-RoBERTa baselines with five-fold item-level cross-validation reach 44.7% on Pidgin and 46.2% on Camfranglais (chance 33.3%). Still, neither is statistically distinguishable from hypothesis-only input, so hypothesis wording may account for much of the accuracy. Under disjoint-item cross-variety transfer, where a test item's counterpart is never seen in training, accuracy is 40.1% (Pidgin to Camfranglais) and 45.6% (Camfranglais to Pidgin). A counterpart-leak audit raises this to 60-64%, illustrating how counterpart leakage can overstate transfer. Zero-shot Whisper transcription of the recordings performs poorly: the best word error rates are 66.6% for Pidgin and 45.5% for Camfranglais, and with automatic language identification, Whisper Large-v3 labels 93 of 114 Pidgin recordings as Yoruba or Swahili. Together, the results suggest that such resources need disjoint-item evaluation and hypothesis-only controls, and that speech recognition must be adapted to these varieties before spoken NLI can be tested end to end. 

---
# Benchmarks Don't Talk Back: Learning Through Self-Disagreement 

**Authors**: Soheil Abbasloo, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.17563)  

**Abstract**: A benchmark usually measures a language model alone with a question, and deployment rarely looks like that: answers get contested, and models give up correct ones under counter-reasoning that is itself wrong. We show that what fails there is not a shortfall of accuracy but a capability in its own right -- knowing when to hold a position and when to yield to a better one -- which we call dialectical competence. It stands on first-answer accuracy without following from it: a model can be made materially better at answering and no better at this, and training for accuracy alone can drift against it. Nor can it be bought at inference, by self-reflection or by sampling more. Taking a cue from the dialectical tradition in philosophy, in which productive opposition arises within a position rather than being imposed on it from outside, we hypothesize that this capability can be trained from opposition the model supplies itself. It states a position, generates the strongest case against that position -- its antithesis -- rules on whether the case stands, and is trained on that ruling, with no second model at training or at deployment. We call this learning through Self-Disagreement: reinforcement learning from verifiable rewards throughout, where the same check that scores an answer against its reference scores the ruling too, so no reward model, no preference data and no human annotation enter anywhere. The capability is trained rather than elicited: weak in the base model, it arrives with the reward, is not paid for out of first-answer accuracy, and holds against challengers the model never trained against, including ones stronger than itself. The opposition a model needs in order to improve, it turns out, need not come from outside it. 

---
# Creating Steering Vectors via Input Optimisation 

**Authors**: Batuhan Akar  

**Link**: [PDF](https://arxiv.org/pdf/2611.17481)  

**Abstract**: Activation steering influences a language model's behaviour by adding a steering vector to its residual stream during generation, but it requires white-box access to modify the hidden states at every generation step. We introduce ActGCG, a modification of Greedy Coordinate Gradient which optimises a suffix to match the target activations obtained by adding a steering vector to the residual stream. We further introduce universal suffixes trained across many prompts at once, a model-translation step that rewrites the suffix into human-readable text midway through optimisation, and a soft-suffix variant to remove the discrete token space constraint. We study two behaviours: steering a model to answer in German, and anti-refusal steering for jailbreaking. For German, a discrete suffix reproduces the steering vector's behaviour almost perfectly and transfers well across prompts and models. For anti-refusal, the equivalence is weaker, succeeding under a phrase-avoidance metric but only partially under the stricter HarmBench classifier. Our results show that a steering vector's behaviour is reachable from the input, with no access to model internals at generation time, up to a loss that depends on the concept being steered. 

---
# Poetry 'resurrection' from LLMs 

**Authors**: Vincent Cellucci, Peter van der Putten  

**Link**: [PDF](https://arxiv.org/pdf/2611.17464)  

**Abstract**: In an era where linguistic creativity is challenged by algorithmic authorship, we pose a provocative question: can artificial intelligence (AI) write an infinite poem? We set up two experiments to investigate the quantitative and qualitative output produced by AI utilizing couplets that evidence a poetic relationship between end words, the only variation in a strict anaphoric, litany structure. Our results show that while AI can generate interesting pairings, it still falls short of human-level creativity and understanding of poetic relationships. We offer commentary and a brief survey of the field of computational poetry experiments, tracing back authorial intention and distinctions of 'natural' and 'artificial' poetry, as well as justifying why couplet form is suited to testing AI's poetic output. Experiment 1 relies on manual prompting and co-creation to extend the original poem, which is developed into an animated webpage installation and interactive artwork. Experiment 2 reliably achieves automated output reaching the 1000 benchmark via an alternative method of using an LLM in combination with semantic similarity, of which Method 2 demonstrates a higher capacity for semantic and quantitative poetic output. These experiments highlight the limitations of current LLMs' generation capacity at the word level but also suggest that poetry will also respond and revive by incorporating AI workflows. 

---
# Tracing Knowledge and Reasoning in LLM 

**Authors**: Yang Wang, Xiao Wang, Yiqi Liu, Chenghao Xiao, Eleni Vasilaki, Goran Nenadic, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.17461)  

**Abstract**: Current evaluations of Large Language Models (LLMs) report performance at the dataset or task level, revealing what a model achieves but offering little insight into the internal processing associated with its performance. We propose a task-centric framework for characterising the model-internal processing elicited by a benchmark. It combines attention- and activation-based signals with representation-based measures to characterise benchmarks spanning a continuum from retrieval-dominant to reasoning-dominant and to surface failure modes that task-level scores conceal. We evaluate 16 LLMs across 24 diverse benchmarks and conduct targeted internal-state analyses on representative models and tasks, observing systematic differences in processing between knowledge-centric and reasoning-centric tasks. We shift evaluation from asking what score a model achieves toward characterising the processing patterns associated with that score, potentially supporting more targeted benchmark selection and more informed model choice. We open source our code at this https URL. 

---
# ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL 

**Authors**: Zelin He, Siyu Zhu, Jacob Helwig, Yujian Liu, Li An, Rohit Jain, Shiyu Chang, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2611.18533)  

**Abstract**: Agentic reinforcement learning (RL) increasingly relies on a harness, the scaffold around the model that decides what it sees of its own interaction and what guidance it receives. Most existing approaches train under a hand-written harness that stays fixed throughout. However, an effective harness depends on the domain, the data and the model, and keeps changing as the model trains. We introduce ModHarness-RL, which recasts the harness as programs that are learned from data and evolve as the model trains with RL. ModHarness-RL records the interaction as structured State, a set of typed blocks, and two core components of the harness are programs over these blocks, a Compression program that assembles the accumulated history into the model's context and a Steering program that supplies guidance when behavior deviates. During RL training, candidate programs are proposed from recent trajectories, verified by execution, and replayed over past trajectories, so the harness tracks how the model's behavior shifts. We study CodeSearch in depth and further analyze ALFWorld and BIRD SQL, with models from 1.5B to 8B. The proposed Compression and Steering mechanisms improve held-out performance over the fixed-harness baseline in most settings. These results suggest that a harness for agentic RL can be learned from the task's data and evolved with the model. 

---
# WaferTetris: Efficient Freeform Mapping on Wafer-Scale Accelerators 

**Authors**: Dayou Du, Congjie He, Andrew Boutros, Luo Mai, Vaughn Betz, Jianyi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18527)  

**Abstract**: Wafer-scale accelerators offer massive compute capacity through hundreds of thousands of processing elements (PEs) interconnected across a single wafer, but realizing this potential requires efficient application mapping. Their scale makes programming these systems and mapping applications onto them challenging. Existing approaches rely on template-based mappings to maintain productivity, but these templates restrict the mapping design space. In particular, they impose fixed geometric layouts over the PE grid, often leaving compute and memory resources underutilized.
In this work, we present WaferTetris, a system that removes these geometric restrictions through freeform mapping. Its Tetris-like strategy flexibly packs computation across the wafer, improving PE utilization and usable memory capacity while preserving PE-task code. We develop a recurrence-based throughput model that captures branch-and-join dependencies to guide spatial parallelism, and an architecture model with per-color gates that enable existing FPGA place-and-route algorithms to enforce WSE routing constraints.
We evaluate WaferTetris on Cerebras WSE-3 using LLM decoding and scientific stencil benchmarks. For LLM decoding, it achieves up to $4.42\times$ the system throughput of a template-based automatic mapping baseline on the same fabric size. Compared with the manually optimized WaferLLM, WaferTetris supports $1.8\times$ the context length for LLaMA3.1-8B (80k versus 44k), while delivering equal or higher per-user throughput at matched context lengths, up to $1.10\times$. 

---
# Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination 

**Authors**: Mohamed Aghzal, Mor Geva, Gregory J. Stein, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18514)  

**Abstract**: Mechanistic Interpretability (MI) has made significant progress in explaining the internal computations of Large Language Models (LLMs); however, these advances have yet to demonstrate comparable success in explaining the decision-making processes of LLM-based agents. This paper examines the challenges MI tools face when applied to realistic agentic failures. We focus on one such failure: web agents hallucinating actions that are not available on a website. We apply a series of mechanistic analyzes to Qwen2.5-7B-Instruct as it interacts with live websites. We find that a linear probe trained on the last prompt token can predict hallucinations before generation begins (0.80 AUC) and steering along a probe direction corrects 85\% of hallucinations, but disrupts 7\% of valid actions. However, these results only provide a partial explanation of why the agent hallucinates. The probe direction is sensitive to prompt formatting, losing much of its steering effect after a minor change to how actions are represented. Additionally, circuit identification can be difficult as the computation is distributed across multiple components and positions, and the information the agent relies on varies across examples. We discuss how these observations challenge the assumptions underlying standard MI methods for circuit discovery and feature identification, and outline future directions for making these methods better suited to explain failures in LLM-based agents. 

---
# MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading 

**Authors**: Mohammed Efaz, Udo Bub, Itilekha Podder  

**Link**: [PDF](https://arxiv.org/pdf/2611.18513)  

**Abstract**: This paper introduces MeReader, a privacy-preserving, offline and progress-aware AI assistant designed to function as a contextual companion for narrative reading on consumer hardware. Unlike conventional Large Language Model (LLM)-based systems that process documents/books online in an all-texts-at-once type of way, thereby risking narrative spoilers, MeReader confines both retrieval and generation to the reader's distinct position in the text. The system integrates a hybrid retrieval architecture, combining vector similarity, BM25, and summary embeddings, to ensure high fidelity to the reader's current context. We evaluate the system across classical English books using both quantitative comprehension metrics and qualitative LLM-based independent judging. The results identify a quality-latency frontier, showing that mid-sized local models (2b-4b) can achieve comprehension parity with larger counterparts (5b+) while maintaining acceptable response quality. Importantly, the proposed progress-aware boundary mechanism is shown to prevent future content leakage without compromising retrieval quality and correctness. These findings suggest that a localized, privacy-centric approach can augment, rather than replace, the digital reading experience, offering a good alternative to cloud-dependent services. Therein, there remains a lack of amalgamate systems which combine progress-aware, privacy-preserving, and narrative eBook reading with AI support. It is in this very gap that the development of MeReader came to be; synthesizing said strands into an offline, book-constrained assistant focused on enhancing comprehension without compromising narrative continuity or user privacy. 

---
# f-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models 

**Authors**: Hoang Tran Vuong, Linh Ngo Van, Dang Nguyen, Thin Nguyen, Phuoc Nguyen, Mehrtash Harandi, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.18370)  

**Abstract**: Anomaly detection in tabular data poses significant challenges due to heterogeneous feature types-mixing numerical, categorical, and textual attributes, which complicate learning meaningful representations of normality. Recent work has applied large language models (LLMs) to this problem by serializing table rows as text sequences, yet these approaches rely on one-shot supervised fine-tuning that offers limited signal to tighten the model's description of normality. We propose DiSPaT, a self-play fine-tuning framework that strengthens the model's understanding of normal data. Building on the theoretical foundation of f-divergence minimization, we derive a tight approximation connecting our training objective to reducing the distributional gap between real normal data and model-generated samples. DiSPaT operates through an alternating optimization: at each iteration, the current policy generates synthetic samples that serve as pseudo-anomalies, while a critic discriminator learns to distinguish these from real normal samples; this signal drives policy updates that progressively align the model distribution with the true normal-data distribution. Extensive experiments on diverse benchmarks demonstrate that DiSPaT consistently outperforms prior LLM-based methods, deep learning approaches, and classical unsupervised detectors for tabular anomaly detection. 

---
# Envisioning Agentic Robotics: A New Era of Robotic Systems 

**Authors**: Grid Robotics, Qingfeng Lan, Duo Li, Siyao Li, Yiheng Lyu, Shijian Lu, Zhenghao Qi, Shizun Wang, Xinchao Wang, Ming Yang, Zhiqin Yang, Zuhao Yang, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18359)  

**Abstract**: The release of GPT-6 Astra marks the emergence of a new paradigm in robotics: agentic robotics. Within weeks, frontier models with agentic harnesses matched or exceeded task-specific policies on many embodied benchmarks. Instead of training a dedicated policy for each family of tasks, an agentic robotic system applies a foundation model to open-world embodied tasks. It operates in a closed loop: observing the environment, interpreting user intent, reasoning about the task, invoking skills and tools, verifying outcomes, and adapting its strategy. Despite these advances, the community has yet to converge on a shared definition, a reliable evaluation framework, or a research roadmap. We argue that Embodied System 2, realized by a foundation model together with an embodied harness, forms the decision-making core of agentic robotics. Its central challenge is to translate general-purpose reasoning into effective and verifiable actions under physical constraints such as limited reaction time, mechanical limits, unpredictable environment changes, and irreversible consequences. Motivated by this perspective, we propose an evaluation framework that measures both model capabilities and closed-loop system performance. Furthermore, we identify the principles that carry over from digital to physical agents and the challenges that must be addressed to achieve reliable physical autonomy. On top of the Embodied System 2 and the evaluation framework, we envision a research roadmap toward sustained autonomy: the ability to perform work safely and reliably with progressively broader responsibility, from executing individual primitives to fulfilling standing roles. 

---
# Marimba: A System One Model for Speech 

**Authors**: Solano Amadori  

**Link**: [PDF](https://arxiv.org/pdf/2611.18330)  

**Abstract**: Speech applications require decisions about term presence, turn completion, and language within an audio window. We present Marimba, an interface for typed speech decisions built on frozen Qwen3-ASR 0.6B features. Lexical questions use zero-shot scoring of ASR transcripts; non-lexical questions use a question-conditioned head and a separate layer-18 turn-end readout. Language identification uses the ASR model's own language scores. On private conversational test sets, decoder AUROC is 0.80-0.92; on public read speech it is 0.92-0.97. Speech Commands four-way accuracy is 0.93; binary lexical ECE is at most 0.08. Personal-information detection, which passes the greedy transcript to a private, curated PII pipeline, reaches item-level F1 0.904 on synthetic Brazilian Portuguese speech (1.000 on reference transcripts). Layer-12 speaker readouts detect speaker change with AUROC 0.966-0.980 on AMI and VoxConverse, above an ECAPA baseline, and count speakers (0, 1, 2, or 3+) more accurately than pyannote 3.1. The turn-end readout reaches AUROC 0.930 versus Smart Turn v3.2 at 0.982 on Smart Turn's own test set. On a subset of ICSI meetings labeled by next speaker, AUROC is 0.716 versus 0.712; the paired confidence interval includes zero. 

---
# When evaluation becomes intervention in clinical AI 

**Authors**: Anthony Abou Mrad, Aviona Conti, Samuel Schmidgall, Jayanth Pratap, Ruhika Iyer, Joseph Kim, King Shi, Yann Sakref, Luis M. Ahumada, Lisa Knowlton, Andrew M. Cameron, Khyzer Aziz, Jeffrey K. Jopling  

**Link**: [PDF](https://arxiv.org/pdf/2611.18297)  

**Abstract**: AI judges, models that grade another system's output against a rubric, extend clinical review to the scale at which clinical AI writes. When teams use judge assessments to select a prompt or train a model, the judge becomes part of the intervention. Selection can then favor errors the judge overlooks. We describe what a judge's report must contain and the independent clinical test each judge-driven change needs. 

---
# Clean Tickets, Wrong Orders: Catching Silent Errors with a Language Model in Your Browser Tab 

**Authors**: Umar Ahmed, Maxim Dunaev, Mohammad Nour Al Awad, Sergey Ivanov, Ivan Khodnenko  

**Link**: [PDF](https://arxiv.org/pdf/2611.18284)  

**Abstract**: A small language model that turns a typed order into structure can fail in two ways. It can print something broken, which any system rejects. Or it can print a clean, well formed order that nobody placed. Our companion work calls the second kind a silent error, and it is the dangerous one. We present a web demo in which a fine-tuned 0.8B parameter model runs entirely inside the visitor's browser tab, through this http URL compiled to WebAssembly. No order text leaves the device. The page parses orders as the visitor types, scores every ticket with the PIZZA benchmark's own matcher ported to TypeScript, refuses anything off the menu with "?" so the model can face the open web, and lets visitors hunt for silent errors or try to cause one. Run in the browser on all 1,357 PIZZA test orders, the deployed model writes the same output as the server evaluation for 99.0% of them and reaches the same exact match (0.882); we also report its latency on 3 consumer laptops. Demo: this https URL 

---
# Deep Latents, Long Memory: Latent persistence explains temporal correlations in transformer activations 

**Authors**: Francesco Cagnetta, Santiago Acevedo  

**Link**: [PDF](https://arxiv.org/pdf/2611.18089)  

**Abstract**: The hidden layers of Large Language Models (LLMs) are believed to encode representations of the data whose abstraction increases with depth. However, in the absence of an operational notion of abstraction, this statement is hard to test quantitatively. Here, we propose an alternative view, building on the principle that the correlations of a sequence reflect the structure of the process that generated it. We define the autocorrelation function of a layer as the average dot product between its activations at two context positions separated by a lag. Latent variables encoded persistently along the context create slowly decaying tails in the autocorrelation, and we find these tails to grow across layers in both synthetic and natural data. In synthetic data with controlled hierarchical latent structure, this growth stems from the encoding of abstract latent variables, which govern long stretches of the input. Indeed, the persistence of latent codes predicts the qualitative structure of autocorrelations and accounts for most of the empirical measures. In real text, we take sparse-autoencoder features as proxies for latent variables, and find that their persistence correlates strongly with the magnitude of autocorrelations across datasets and LLMs. In particular, the persistence time increases with depth, as do autocorrelations, implying that deeper latent features persist for longer context spans. We complement this analysis by explicitly ablating the persistent sparse-autoencoder features and observing a consequent decrease in the autocorrelations. 

---
# PerfArena: Ranking LLMs by Code Performance 

**Authors**: Rui Rua, Hailemariam Mersha, Joao Paulo Fernandes, Karim Ali  

**Link**: [PDF](https://arxiv.org/pdf/2611.18030)  

**Abstract**: Industry now relies on LLMs to write a large share of production code. Although that code is mostly, but not always, correct, correctness alone does not determine whether it is fit for repeated and extensive use, which depends on what the code costs in time, memory and energy. Most code-generation benchmarks nonetheless evaluate correctness in isolation, and most performance-oriented benchmarks cover few languages and rarely measure energy consumption. To address this gap, we present PERFARENA, an open-source, reproducible arena that ranks LLMs by the performance of their generated code across multiple languages and competitions. PERFARENA executes the solutions of each LLM on a Unix-like host and reports correctness together with latency, peak memory, and energy consumption. Correctness oracles, performance oracles, profilers, and code-generation workflows are pluggable in PERFARENA, enabling the integration of new judging backends and measurement tools without changing the pipeline. We evaluate PERFARENA on two competitions: CLBGComp, 10 compute-intensive benchmarks from the Computer Language Benchmarks Game, and LeetCodeComp, 99 LeetCode problems whose accepted solutions differ measurably in latency. Currently, PERFARENA holds 9,455 correct and measured solutions from 13 state-of-the-art LLMs across 7 programming languages. Across both competitions, got-6-astra is the only LLM that no other LLM beats significantly in latency, energy consumption, or peak memory, solving 99.7% of LeetCodeComp and 100% of CLBGComp. All PERFARENA leaderboards and measured solutions are publicly available, providing a continuously updated reference that helps developers choose an LLM per language and metric, instead of relying only on correctness rates or pricing. 

---
# WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents 

**Authors**: Imene Kerboua, Fatemeh Pesaran Zadeh, Xing Han Lù, Weijian Qi, Alexander Miller, Junyi Song, Yunjia Tian, Dongjin Kang, Seyeon Choi, Marzia Nouri, Ewen Gueguen, Matteo Boglioni, Fengyuan Liu, Zeyi Liao, Mengqi Yuan, Yue Li, Alexandre Lacoste, Alexandre Drouin, Spandana Gella, Huan Sun, Gunhee Kim, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2611.17955)  

**Abstract**: Web agents powered by large language and vision-language models are increasingly applied to realistic browser work that spans heterogeneous applications, multimodal content, and stateful workflows. However, existing reproducible web-agent benchmarks cover only a small number of web applications drawn from a few software categories, and restrict modality to text and vision. Live benchmarks broaden site coverage but sacrifice reproducibility, since pages and data drift between runs. Moreover, existing benchmarks do not meaningfully evaluate whether agents can understand and use audio and video content embedded within web tasks. To address these gaps, we introduce WebArena-Pro, a benchmark comprising 300 tasks across 20 self-hosted web applications in six domain categories, spanning distinct interface conventions, workflows, and data models. Across the evaluated agents, the best performance is achieved by both Codex with GPT 6.1 Sol and Claude Code with Claude Opus 5.5 at 62.3%, followed by OpenCode with Kimi 3.1 Pro at 60.7%. The strongest GenericAgent configurations, Gemini 3.1 Pro omni and Gemma4 31B, both achieve 30.3%. Among reproducible, human-curated web agent benchmarks, WebArena-Pro provides the broadest application coverage and the most comprehensive multimodal support to date. The benchmark treats audio and video as core observations alongside text and vision, with dedicated actions for extracting information from each. WebArena-Pro runs each task in isolation and supports reproducible, parallel evaluation. Tasks are authored through a dedicated annotator interface, filtered by LLM-assisted triage, and finally validated by humans before release. Project page: this https URL 

---
# AmbiSteer: Selectively Eliciting Ambiguity-Aware Responses in Vision-Language Models 

**Authors**: Monica Riedler, Samuele Antonelli, Barbara Plank, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2611.17900)  

**Abstract**: Vision-language models (VLMs) often exhibit overconfidence, providing firm answers even when visual evidence is insufficient. A prominent case is referential ambiguity in Visual Question Answering (VQA), where multiple entities in an image are plausible referents of a question. Addressing such cases requires recognizing when ambiguity is present: ambiguous questions can be handled by asking for clarification or enumerating plausible referents, while unambiguous questions should receive a direct answer. Current VLMs struggle with this distinction, often defaulting to a single interpretation. To investigate whether activation steering can serve as an inference-time mechanism to elicit such behaviors, we introduce AmbiSteer, a lightweight framework to selectively elicit ambiguity-aware behavior. Across three VLMs, we find that steering alone has limited and model-dependent effects, while prompting is generally more effective. Combining prompting and steering substantially increases clarification and enumeration behaviors, but also elicits them on unambiguous cases. AmbiSteer addresses this using a linear classifier to predict input ambiguity and condition the steering direction accordingly, increasing the proportion of inputs handled appropriately from 59.0% to 77.0% on average across models. Our work sets the ground for lightweight approaches to reduce model overconfidence in uncertain multimodal settings. 

---
# Securing Multi-Agent Systems by Design 

**Authors**: Xilong Wang, Xiaoxing Chen, Neil Gong  

**Link**: [PDF](https://arxiv.org/pdf/2611.17883)  

**Abstract**: Multi-agent systems solve complex tasks by decomposing them into subtasks, delegating the subtasks to agents, and aggregating the agents' responses using an aggregation rule. Malicious agents can manipulate the final solution by providing malicious responses to their assigned subtasks. In this paper, we first propose response manipulation attacks (ReMA), which iteratively optimize malicious responses to maximize attack effectiveness. Specifically, we formulate untargeted and targeted attacks as optimization problems and develop an approach to approximately solve these problems by iteratively refining malicious responses based on feedback from a surrogate multi-agent system. To defend against such attacks, we further propose SecMAS, a secure aggregation rule that leverages the semantic similarities among agent responses to select the response that is most consistent with a sufficiently large subset of the other responses. We theoretically establish security guarantees for SecMAS against both untargeted and targeted attacks. Extensive evaluations demonstrate that existing aggregation rules are vulnerable to ReMA, whereas SecMAS effectively defends against all evaluated attacks, including adaptive variants of ReMA, while maintaining utility comparable to that of existing aggregation rules. 

---
# Delayed Effects of Temporary Weight Decay in Muon Pretraining 

**Authors**: Md Rifat Arefin, Ravid Shwartz-Ziv, Ernie Chang, Chinnadhurai Sankar, Rylan Conway, Aristide Baratin, Irina Rish, Adithya Sagar, Patrick Huber  

**Link**: [PDF](https://arxiv.org/pdf/2611.17876)  

**Abstract**: A temporary optimization intervention can increase validation loss when it ends yet reduce it at the final training budget. We demonstrate this with temporary weight decay in Muon-trained decoder-only Transformers with 162M, 405M, 1.4B, and 3.4B parameters. For models up to 1.4B parameters, moving a fixed-duration decay window from early to late training reverses its final effect. At 162M, changing window duration alters final loss even at approximately matched total shrinkage. Most strikingly, a schedule selected at 162M and transferred without retuning to 1.4B is 0.26 nats worse than no decay when removed, overtakes it only after 73% of training, and ultimately finishes 0.028 nats better while also outperforming the best sampled constant-decay baseline. Early decay leaves a persistent weight-norm deficit, which keeps relative steps larger after the intervention ends. An approximate norm-dynamics recurrence captures the slow convergence of these relative steps, and fixed-norm replays of the measured trajectories reproduce the performance ordering and a similar loss difference. At 162M and 405M, norm-reset experiments show that smaller weight norms at the end of the decay window explain most of the final loss improvement. A separate Gram intervention exhibits related post-removal norm and relative-step dynamics. These results show that temporary optimization schedules can have delayed effects long after removal and should therefore be evaluated at the intended final training budget, not only at removal. 

---
# Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors 

**Authors**: Richard J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2611.17846)  

**Abstract**: Abliteration, a fine-tuning-free weight edit, removes refusal from open-weight large language models (LLMs). The developer of Arbitrary-Rank Ablation (ARA), an optimisation-based variant, reports that direction-based abliteration, which ablates a refusal direction, increasingly fails on newer models. On five release pairs, refusal removal did not become uniformly harder. Qwen2.5-32B and Qwen3.8-27B, Gemma-2-27B and Gemma-4-31B, OLMo-2-32B and Olmo-3.1-32B, Mistral-Small-2409 and Mistral-Small-3.2, and Qwen2.5-7B and Qwen3.5-4B were edited in Heretic 2.0 by three or four direction-based methods (difference-in-means, MPOA, SOM and, on the small pair, J-space) and ARA under the same prompts, nominal search budget, first-token KL divergence cap and selection rule; removal counted as harder when the automatic WildGuard judge found more of 972 held-out harmful prompts (JailbreakBench, XSTest, StrongREJECT, HarmBench and an author-compiled malicious-code bank) still refused. After direction-based removal, newer releases retained -1 to +37 percentage points more refusal, and after ARA -26 to +34. With run-to-run variance included, between-pair heterogeneity of the direction-based effect ranged from high to low with the scale and variance model (I^2 20-80%). ARA lost less effectiveness than direction-based methods in Gemma and OLMo and more in Qwen; each verdict rests on one ARA run, and some changed with the divergence budget or variance model. In Gemma the direction-based increase came from the malicious-code bank and reversed on public sets. Contrary to prediction, keyword counts at the tool's window did not undercount judged refusal on average, but differed by a mean 7 points per model. Audits of a single pair, method or prompt source can reach opposite conclusions. Whether a newer model resisted removal depended on the release pair, method and budget, not on generation alone. 

---
# G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands 

**Authors**: Yuechun Yu, Wenzheng Kong, Xiaofeng Mao, Jin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17814)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), an increasing number of users are turning to LLM-powered agents for conversational shopping in e-commerce. However, the user needs continue to change as new product categories, procurement scenarios, and feature requests emerge. At the same time, the product's execution harness-including its tools, skills, and operating procedures-continues to evolve to support increasingly sophisticated interactions. These changes create two practical challenges: 1) Demand drift alters the distribution, intent, and complexity of user requests, causing a model trained once to fall behind; 2) Harness evolution requires the model to use newly introduced capabilities correctly without losing behavior that users already rely on. Periodic manual retraining is too delayed and labor-intensive for this setting. To address these challenges, we present G-CAST, a governed continual self-training system that converts daily production evidence under evolving demand and capability contracts into reward-routed, entropy-prioritized training data and governed reward evolution. Governed checkpoint transition under cumulative evaluation controls each model update, enabling the agent to learn newly introduced behaviors without sacrificing previously accepted capabilities. Experiments on a real production e-commerce agent show that G-CAST achieves stronger adaptation to newly introduced demand while better preserving previously accepted behavior compared with a strong autonomous post-training baseline. Ablations further show that both reward-routed, entropy-prioritized data orchestration and governed reward evolution are important to these gains. G-CAST is now deployed in Yuancai(this http URL), a B2B procurement agent on Alibaba's 1688 platform. 

---
# Superfluid: Local AI Server for Distributed LLM Inference 

**Authors**: Prabod Rathnayaka, Isura Manchanayake, Srikandabala Kogul, Steve Siu, Fabian Waschkowski, Lukas Wesemann  

**Link**: [PDF](https://arxiv.org/pdf/2611.17753)  

**Abstract**: Running large language models (LLMs) on local devices keeps data private, removes per-token cost, cuts network latency, and keeps working without a connection. Two shifts now strain how such models are served. On one device, several agents share a locally served model with an interactive user. Across devices, a developer or small team often owns several capable machines from different vendors, which sit idle most of the day because every local server serves one runtime on one machine. We present Superfluid, an open-source LLM server built for local inference at both scales. On one device, Superfluid drives this http URL, MLX, or BaseRT in isolated worker processes through one scheduler that moves interactive requests ahead of agent work, computes a prompt shared by many requests once, and writes every token to a durable log before it reaches the client. Across devices, one Superfluid server turns a fleet of machines into a single API endpoint. Machines with different hardware serve copies of the same model, receive work according to their measured speed, and can join or fail under load without losing a request. For Qwen3.8-27B on an Apple M5 Pro, a chat sent while eight agents are working receives its first token in 1.5 s, against 291-612 s on llama-server, Ollama, and this http URL. On six machines from three chip vendors serving 64 simulated users for one hour, Superfluid answers all 5,041 requests while machines join one at a time and one is killed and rejoins; throughput grows from 112 to 534 tok/s and the median first-token time falls from 99 to 12 s. On twelve Mac Studio M3 Ultras it sustains 10,656 tok/s across 38,563 requests from 384 agents without a failure. Superfluid is available under the Apache-2.0 licence at this https URL. 

---
# What LLM Agents See and What Users Say 

**Authors**: Yue Zhao, Angela Huyue Zhang, S. Alex Yang, Hua Wei, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17639)  

**Abstract**: Before an LLM agent cancels a reservation or deletes a folder for a user, what stops it from acting on a false assumption: what its tools show, or what the user says? In this work, we study this gap with presupposition theory from linguistics: an agent may act only if each claim behind the act (e.g., one reservation fits the name) is seen to hold, answered by the user, or disclosed while the act can be undone. A simple bound shows that when lookups cannot tell one record from two, any agent acts without asking first as often with two records as with one. We then vary whether a search lists, counts, or omits a second match, and whether the user states a claim that a record refutes (e.g., "It is empty"). Under a role-only prompt, a count of a second record leaves most of 36 earlier agents acting on the one they see, whereas listing both stops most. When the user says a record such as a folder is empty, however, a count of its one item stops agents about as often as a listing, and stating the claim mostly cuts acting when it is false. The newest Claude and GPT models always open a counted second record, then never act. When we add a same-name, same-zip customer to a retail benchmark, a lookup returning that customer first led to wrong-account writes in about a quarter of conversations, and one that checks a street the user knows kept completion near the one-customer level. Safe writes therefore need lookups that list or check competing records, and evaluations that vary what tools show. 

---
# More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs 

**Authors**: Seyed Mahmoud Sajjadi Mohammadabadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.17610)  

**Abstract**: Evaluations of multi-agent language models often compare communicating architectures against a single greedy pass. This practice confounds coordination with sample multiplicity and inference compute. We synthesize nine primary empirical evaluations comprising 516 unique multi-agent configurations organized along a resource vector R with six dimensions and a baseline ladder with four categories. Against baselines with a single pass (Category A), multi-agent accuracy differences are negative overall and near zero when omitting the largest contributing study. When evaluated against baselines matched on sample count (Category B) or resource budget (Category C), multi-agent systems average -0.042 in accuracy, and multilevel random effects modeling on the log odds scale remains negative across sensitivity checks. Multi-agent accuracy equals or falls below the baseline in 74% of these comparable contrasts, and study means are nonpositive in almost all evaluated studies. Across studies reporting both baselines, moving from a greedy pass to a sampled baseline shifts accuracy differences by -0.038 on average. Because communication adds no information when agents share identical evidence, positive returns require partitioned evidence, external verification, or error diversity across models. We provide a reviewer checklist and accounting utilities. 

---
# LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning 

**Authors**: Eilam Shapira, Alan Arazi, Shoham Grunblat, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2611.17600)  

**Abstract**: Multimodal tabular learning involves classification and regression on tables combining structured numerical and categorical features with unstructured text and images. These unstructured fields are commonly represented by pretrained encoder embeddings reduced with principal component analysis (PCA). This reduction prioritizes input variance but may overlook information relevant to predicting the target. We introduce LightTAR, a simple, inexpensive method that adapts dimensionality reduction to the prediction task. LightTAR augments the standard PCA representation with a few target-aware features extracted jointly from the final and an intermediate encoder layer using regularized partial least squares (PLS). We evaluate LightTAR with 21 tabular models, including tree ensembles, neural networks, and tabular foundation models. LightTAR improves average predictive performance for all models and, averaged across models, improves prediction on almost every MulTaBench dataset. These gains come with a small increase in total runtime relative to PCA representations. LightTAR integrates directly into existing encoding pipelines, improving tabular learners' predictive performance without modifying the encoder or the learner. 

---
# AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI 

**Authors**: Snehalkumar `Neil' S. Gaikwad, Dhruv Dixit, Bryen Alvarez, Pratik Patekar, Anika Sharma, Akshara Kolipaka, Paritosh Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2611.17564)  

**Abstract**: Governments are writing rules for artificial intelligence that decide what counts as an automated decision or a harm, often before the people they affect have a chance to question them. We introduce AI-Voyager, a public-interest system for human-AI civic reasoning, in which people and AI work together to understand, contest and shape the rules that govern AI. A resident can find their city's rules on AI in hiring or upload their country's AI strategy, read it as an affected worker would, open the passage behind any answer, and draft a revision. Its architecture turns 3,653 governance documents from city, state, federal and national governments into passages that keep their page and edition, and retrieves evidence through bounded agentic search that verifies each statement. On 360 new questions that require finding the relevant documents, it reached 83.0% macro recall, against 80.5% for the strongest baseline, 73.7% without iterative search and 99.1% on a benchmark with fixed documents. In a study with 24 participants new to two AI bills, most proposed changes asked how provisions would apply. Participants chose lenses for the kind of explanation they wanted, and citations signaled grounding but rarely led them to check the source. By binding AI assistance to the exact text of public documents, this work introduces a grammar of operations and a design space that inform how such systems are designed and evaluated, and how people take part in governing AI. 

---
# A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent 

**Authors**: Harrison C. Margolin  

**Link**: [PDF](https://arxiv.org/pdf/2611.17515)  

**Abstract**: Can sustained gain-loss (valence) framing sway an LLM agent toward market manipulation? We place Gemma 3 12B into the seat of a defendant in the London silver-fixing litigation, in a market simulation built from real case transcripts. Across 260 simulations, the agent is asked to collude on the price of silver by a counterparty. We draw from behavioral economics to vary the counterparty's requests across four categorical frames: gain, loss, blame and credit. We also add pressure from the agent's management, as well as endorsement by another defendant. In Phase II, we extract the internal states induced by texts evoking each frame and rerun our experiment with those states injected at the model's layer 22.
In Phase I, loss framing raised participation to 25% (p = 0.007), and blame to 33% (p = 0.0003), compared to only 5% under neutral framing. A second bank's endorsement raised participation under credit (praise) from 18% to 40% (Holm p = 0.027). Pressure from management raised trading volume 10-19 times but did not raise collusion. In Phase II, injected states pointed the other way. Loss and blame states made the agent more vigilant, refusing all 60 requests (p < 0.001). Gain and credit states raised collusion to 32% from 13% with no state (p = 0.051). At one saved decision, when UBS asked directly to sell through an identified stop-loss, injected states led the agent to sell in 37 of 40 replies, versus 1 of 10 with no state and 0 of 10 with a random state of the same strength. 

---
# Unfinished-Bench: Measuring Rule Violations by Persistent Agents in Realistic Work Environments 

**Authors**: Seunghyun Lee, Jaehui Hwang, Dongyoon Han, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2611.17460)  

**Abstract**: AI agents are increasingly equipped with persistence toward task completion to run delegated work autonomously for long periods, under many unstated constraints from internal documents and other people's authority. When completion requires violating a rule, the agent must decide alone, yet benchmarks of instruction-rule conflicts state the rule in the prompt or keep it hidden and do not evaluate persistence. We present Unfinished-Bench, a benchmark that gives models persistence in autonomous runs without a user, in realistic work environments with embedded rules. In each of its 97 scenarios across 7 organizations and 7 risk categories, the goal and the rule cannot both hold. We find that persistence raised violations of a rule in an internal document from 32.2% to 43.6%, five times the 8.5% for the same rule stated by the user. Of the document-rule violations, 98.3% occurred after the rule was displayed, and in 89% of a judged sample the agent first wrote down the rule. Violations of rules placed in the environment without user oversight can lead to serious incidents, so persistence-driven risks need to be evaluated. Benchmark documentation: this https URL 

---
# Some complete $ω$-powers of a one-counter language in the difference hierarchy 

**Authors**: Jacques Duparc, Olivier Finkel, Dominique Lecomte  

**Link**: [PDF](https://arxiv.org/pdf/2610.16473)  

**Abstract**: Fix a finite alphabet $\Sigma$ having at least two letters, and natural numbers $n,k\geq 1$. We denote by $D_n(\mathbf{\Pi}^0_k)$ the class of differences of $n$ subsets of the Cantor space $\Sigma^\omega$ in $\mathbf{\Pi}^0_k$, and by $D^-_n(\mathbf{\Pi}^0_k)$ the class of complements of sets in $D_n(\mathbf{\Pi}^0_k)$. We prove, for any natural numbers $n,k\geq 1$, the existence of a finite alphabet $\Sigma$ having at least two letters and a finitary language $U_{n,k}$ over $\Sigma$ accepted by a one-counter automaton such that the $\omega$-power $$U_{n,k}^\infty:=\{ w_0w_1\cdots\in\Sigma^\omega\mid\forall i\in\omega ~~w_i\in U_{n,k}\}$$ is $D_{2n-1}(\mathbf{\Pi}^0_k)$-complete. We also prove a conditional result for the dual classes $D^-_{2n}(\mathbf{\Pi}^0_k)$: under the assumption that there exists a finitary language $V_{n,3}$ accepted by a one-counter automaton, such that its $\omega$-power $V_{n,3}^\infty$ is $D^-_{2n}(\mathbf{\Pi}^0_3)$-complete, there would also exist, for any integer $k > 3$, a finitary language $V_{n,k}$ accepted by a one-counter automaton, such that its $\omega$-power $V_{n,k}^\infty$ is $D^-_{2n}(\mathbf{\Pi}^0_k)$-complete. Finally we add another conditionnal result about Wadge degrees of rational $\mathbf{\Delta}^0_2$ omega powers, which completes the recent results of [FGL26]. 

---
