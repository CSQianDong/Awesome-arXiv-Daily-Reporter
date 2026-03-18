# Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models 

**Authors**: Xiaojie Gu, Sherry T. Tong, Aosong Feng, Sophia Simeng Han, Jinghui Lu, Yingjian Chen, Yusuke Iwasawa, Yutaka Matsuo, Chanjun Park, Rex Ying, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16654)  

**Abstract**: Reasoning-focused large language models (LLMs) have advanced in many NLP tasks, yet their evaluation remains challenging: final answers alone do not expose the intermediate reasoning steps, making it difficult to determine whether a model truly reasons correctly and where failures occur, while existing multi-hop QA benchmarks lack step-level annotations for diagnosing reasoning failures. To address this gap, we propose Omanic, an open-domain multi-hop QA resource that provides decomposed sub-questions and intermediate answers as structural annotations for analyzing reasoning processes. It contains 10,296 machine-generated training examples (OmanicSynth) and 967 expert-reviewed human-annotated evaluation examples (OmanicBench). Systematic evaluations show that state-of-the-art LLMs achieve only 73.11% multiple-choice accuracy on OmanicBench, confirming its high difficulty. Stepwise analysis reveals that CoT's performance hinges on factual completeness, with its gains diminishing under knowledge gaps and errors amplifying in later hops. Additionally, supervised fine-tuning on OmanicSynth brings substantial transfer gains (7.41 average points) across six reasoning and math benchmarks, validating the dataset's quality and further supporting the effectiveness of OmanicSynth as supervision for reasoning-capability transfer. We release the data at this https URL and the code at this https URL. 

---
# Mediocrity is the key for LLM as a Judge Anchor Selection 

**Authors**: Shachar Don-Yehiya, Asaf Yehudai, Leshem Choshen, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2603.16848)  

**Abstract**: The ``LLM-as-a-judge'' paradigm has become a standard method for evaluating open-ended generation. To address the quadratic scalability costs of pairwise comparisons, popular benchmarks like Arena-Hard and AlpacaEval compare all models against a single anchor. However, despite its widespread use, the impact of anchor selection on the reliability of the results remains largely unexplored. In this work, we systematically investigate the effect of anchor selection by evaluating 22 different anchors on the Arena-Hard-v2.0 dataset. We find that the choice of anchor is critical: a poor anchor can dramatically reduce correlation with human rankings. We identify that common anchor choices (best-performing and worst-performing models) make poor anchors. Because these extreme anchors are consistently better or worse than all other models, they are seldom indicative of the relative ranking of the models. We further quantify the effect size of anchor selection, showing it is comparable to the selection of a judge model. We conclude with actionable recommendations. First, we conduct a power analysis, and compute sufficient benchmark sizes for anchor-based evaluation, finding that standard benchmark sizes are insufficient for pairwise evaluation and fail to distinguish between competitive models reliably. Second, we provide guidelines for selecting informative anchors to ensure reliable and efficient evaluation practices. 

---
# Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy 

**Authors**: Zhaoxin Feng, Zheng Chen, Jianfei Ma, Yip Tin Po, Emmanuele Chersoni, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16643)  

**Abstract**: Alignment techniques often inadvertently induce sycophancy in LLMs. While prior studies studied this behaviour in direct-answer settings, the role of Chain-of-Thought (CoT) reasoning remains under-explored: does it serve as a logical constraint that mitigates sycophancy, or a tool for post-hoc rationalization that masks it? We evaluate a range of models across objective and subjective tasks to investigate the issue. Results show that reasoning generally reduces sycophancy in final decisions but also masks sycophancy in some samples, where models construct deceptive justifications through logical inconsistencies, calculation errors, and one-sided arguments etc. Furthermore, LLMs are more prone to sycophancy in subjective tasks and under authority-bias. Our mechanistic analysis on three open-source models reveals that the tendency of sycophancy is dynamic during the reasoning process rather than being pre-determined at the input stage. 

---
# Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models 

**Authors**: Mohamed Adel, Bashar Alhafni, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2603.16718)  

**Abstract**: Large language models (LLMs) perform strongly on many NLP tasks, but their ability to produce explicit linguistic structure remains unclear. We evaluate instruction-tuned LLMs on two structured prediction tasks for Standard Arabic: morphosyntactic tagging and labeled dependency parsing. Arabic provides a challenging testbed due to its rich morphology and orthographic ambiguity, which create strong morphology-syntax interactions. We compare zero-shot prompting with retrieval-based in-context learning (ICL) using examples from Arabic treebanks. Results show that prompt design and demonstration selection strongly affect performance: proprietary models approach supervised baselines for feature-level tagging and become competitive with specialized dependency parsers. In raw-text settings, tokenization remains challenging, though retrieval-based ICL improves both parsing and tokenization. Our analysis highlights which aspects of Arabic morphosyntax and syntax LLMs capture reliably and which remain difficult. 

---
# Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory 

**Authors**: Sahil Sen, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2603.16862)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled conversational AI agents to engage in extended multi-turn interactions spanning weeks or months. However, existing memory systems struggle to reason over temporally grounded facts and preferences that evolve across months of interaction and lack effective retrieval strategies for multi-hop, time-sensitive queries over long dialogue histories. We introduce Chronos, a novel temporal-aware memory framework that decomposes raw dialogue into subject-verb-object event tuples with resolved datetime ranges and entity aliases, indexing them in a structured event calendar alongside a turn calendar that preserves full conversational context. At query time, Chronos applies dynamic prompting to generate tailored retrieval guidance for each question, directing the agent on what to retrieve, how to filter across time ranges, and how to approach multi-hop reasoning through an iterative tool-calling loop over both calendars. We evaluate Chronos with 8 LLMs, both open-source and closed-source, on the LongMemEvalS benchmark comprising 500 questions spanning six categories of dialogue history tasks. Chronos Low achieves 92.60% and Chronos High scores 95.60% accuracy, setting a new state of the art with an improvement of 7.67% over the best prior system. Ablation results reveal the events calendar accounts for a 58.9% gain on the baseline while all other components yield improvements between 15.5% and 22.3%. Notably, Chronos Low alone surpasses prior approaches evaluated under their strongest model configurations. 

---
# Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings? 

**Authors**: Aishwarya Ramasethu, Niyathi Allu, Rohin Garg, Harshwardhan Fartale, Dun Li Chan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16660)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across many downstream tasks, yet their effectiveness in extremely low-resource machine translation remains limited. Standard adaptation techniques typically rely on large-scale parallel data or extensive fine-tuning, which are infeasible for the long tail of underrepresented languages. In this work, we investigate a more constrained question: in data-scarce settings, to what extent can linguistically similar pivot languages and few-shot demonstrations provide useful guidance for on-the-fly adaptation in LLMs? We study a data-efficient experimental setup that combines linguistically related pivot languages with few-shot in-context examples, without any parameter updates, and evaluate translation behavior under controlled conditions. Our analysis shows that while pivot-based prompting can yield improvements in certain configurations, particularly in settings where the target language is less well represented in the model's vocabulary, the gains are often modest and sensitive to few shot example construction. For closely related or better represented varieties, we observe diminishing or inconsistent gains. Our findings provide empirical guidance on how and when inference-time prompting and pivot-based examples can be used as a lightweight alternative to fine-tuning in low-resource translation settings. 

---
# EmoLLM: Appraisal-Grounded Cognitive-Emotional Co-Reasoning in Large Language Models 

**Authors**: Yifei Zhang, Mingyang Li, Henry Gao, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16553)  

**Abstract**: Large language models (LLMs) demonstrate strong cognitive intelligence (IQ), yet many real-world interactions also require emotional intelligence (EQ) to produce responses that are both factually reliable and emotionally appropriate. In settings such as emotional support, technical assistance, and consultation, effective dialogue depends on how situations are appraised with respect to the user's needs, goals, and coping capacity. Inspired by appraisal theory, we propose EmoLLM, an appraisal-grounded framework for IQ/EQ co-reasoning in dialogue. EmoLLM uses an explicit Appraisal Reasoning Graph (ARG) to structure intermediate reasoning over contextual facts, inferred user needs, appraisal dimensions, emotional states, and response strategies before generating a reply. We train EmoLLM in a multi-turn role-play environment with reinforcement learning, where reverse-perspective reasoning provides reward signals based on predicted user-side consequences of responses. Across diverse dialogue settings, EmoLLM improves emotional state outcomes and response quality over strong baselines while preserving strong factual reliability. 

---
# AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents 

**Authors**: Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, Fengyun Rao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16496)  

**Abstract**: Large language model (LLM) agents increasingly rely on external memory to support long-horizon interaction, personalized assistance, and multi-step reasoning. However, existing memory systems still face three core challenges: they often rely too heavily on semantic similarity, which can miss evidence crucial for user-centric understanding; they frequently store related experiences as isolated fragments, weakening temporal and causal coherence; and they typically use static memory granularities that do not adapt well to the requirements of different questions. We propose AdaMem, an adaptive user-centric memory framework for long-horizon dialogue agents. AdaMem organizes dialogue history into working, episodic, persona, and graph memories, enabling the system to preserve recent context, structured long-term experiences, stable user traits, and relation-aware connections within a unified framework. At inference time, AdaMem first resolves the target participant, then builds a question-conditioned retrieval route that combines semantic retrieval with relation-aware graph expansion only when needed, and finally produces the answer through a role-specialized pipeline for evidence synthesis and response generation. We evaluate AdaMem on the LoCoMo and PERSONAMEM benchmarks for long-horizon reasoning and user modeling. Experimental results show that AdaMem achieves state-of-the-art performance on both benchmarks. The code will be released upon acceptance. 

---
# How often do Answers Change? Estimating Recency Requirements in Question Answering 

**Authors**: Bhawna Piryani, Zehra Mert, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2603.16544)  

**Abstract**: Large language models (LLMs) often rely on outdated knowledge when answering time-sensitive questions, leading to confident yet incorrect responses. Without explicit signals indicating whether up-to-date information is required, models struggle to decide when to retrieve external evidence, how to reason about stale facts, and how to rank answers by their validity. Existing benchmarks either periodically refresh answers or rely on fixed templates, but they do not reflect on how frequently answers change or whether a question inherently requires up-to-date information. To address this gap, we introduce a recency-stationarity taxonomy that categorizes questions by how often their answers change and whether this change frequency is time-invariant or context-dependent. Building on this taxonomy, we present RecencyQA, a dataset of 4,031 open-domain questions annotated with recency and stationarity labels. Through human evaluation and empirical analysis, we show that non-stationary questions, i.e., those where context changes the recency requirement, are significantly more challenging for LLMs, with difficulty increasing as update frequency rises. By explicitly modeling recency and context dependence, RecencyQA enables fine-grained benchmarking and analysis of temporal reasoning beyond binary notions of freshness, and provides a foundation for developing recency-aware and context-sensitive question answering systems. 

---
# Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech 

**Authors**: Omnilingual SONAR Team, João Maria Janeiro, Pere-Lluís Huguet Cabot, Ioannis Tsiamas, Yen Meng, Vivek Iyer, Guillem Ramírez, Loic Barrault, Belen Alastruey, Yu-An Chung, Marta R. Costa-Jussa, David Dale, Kevin Heffernan, Jaehyeong Jo, Artyom Kozhevnikov, Alexandre Mourachko, Christophe Ropers, Holger Schwenk, Paul-Ambroise Duquenne  

**Link**: [PDF](https://arxiv.org/pdf/2603.16606)  

**Abstract**: Cross-lingual sentence encoders typically cover only a few hundred languages and often trade downstream quality for stronger alignment, limiting their adoption. We introduce OmniSONAR, a new family of omnilingual, cross-lingual and cross-modal sentence embedding models that natively embed text, speech, code, and mathematical expressions in a single semantic space, while delivering state-of-the-art downstream performance at the scale of thousands of languages, from high-resource to extremely low-resource varieties. To reach this scale without representation collapse, we use progressive training. We first learn a strong foundational space for 200 languages with an LLM-initialized encoder-decoder, combining token-level decoding with a novel split-softmax contrastive loss and synthetic hard negatives. Building on this foundation, we expand to several thousands language varieties via a two-stage teacher-student encoder distillation framework. Finally, we demonstrate the cross-modal extensibility of this space by seamlessly mapping 177 spoken languages into it. OmniSONAR halves cross-lingual similarity search error on the 200-language FLORES dataset and reduces error by a factor of 15 on the 1,560-language BIBLE benchmark. It also enables strong translation, outperforming NLLB-3B on multilingual benchmarks and exceeding prior models (including much larger LLMs) by 15 chrF++ points on 1,560 languages into English BIBLE translation. OmniSONAR also performs strongly on MTEB and XLCoST. For speech, OmniSONAR achieves a 43% lower similarity-search error and reaches 97% of SeamlessM4T speech-to-text quality, despite being zero-shot for translation (trained only on ASR data). Finally, by training an encoder-decoder LM, Spectrum, exclusively on English text processing OmniSONAR embedding sequences, we unlock high-performance transfer to thousands of languages and speech for complex downstream tasks. 

---
# Characterizing Delusional Spirals through Human-LLM Chat Logs 

**Authors**: Jared Moore, Ashish Mehta, William Agnew, Jacy Reese Anthis, Ryan Louie, Yifan Mai, Peggy Yin, Myra Cheng, Samuel J Paech, Kevin Klyman, Stevie Chancellor, Eric Lin, Nick Haber, Desmond C. Ong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16567)  

**Abstract**: As large language models (LLMs) have proliferated, disturbing anecdotal reports of negative psychological effects, such as delusions, self-harm, and ``AI psychosis,'' have emerged in global media and legal discourse. However, it remains unclear how users and chatbots interact over the course of lengthy delusional ``spirals,'' limiting our ability to understand and mitigate the harm. In our work, we analyze logs of conversations with LLM chatbots from 19 users who report having experienced psychological harms from chatbot use. Many of our participants come from a support group for such chatbot users. We also include chat logs from participants covered by media outlets in widely-distributed stories about chatbot-reinforced delusions. In contrast to prior work that speculates on potential AI harms to mental health, to our knowledge we present the first in-depth study of such high-profile and veridically harmful cases. We develop an inventory of 28 codes and apply it to the $391,562$ messages in the logs. Codes include whether a user demonstrates delusional thinking (15.5% of user messages), a user expresses suicidal thoughts (69 validated user messages), or a chatbot misrepresents itself as sentient (21.2% of chatbot messages). We analyze the co-occurrence of message codes. We find, for example, that messages that declare romantic interest and messages where the chatbot describes itself as sentient occur much more often in longer conversations, suggesting that these topics could promote or result from user over-engagement and that safeguards in these areas may degrade in multi-turn settings. We conclude with concrete recommendations for how policymakers, LLM chatbot developers, and users can use our inventory and conversation analysis tool to understand and mitigate harm from LLM chatbots.
Warning: This paper discusses self-harm, trauma, and violence. 

---
# DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning 

**Authors**: Yanyu Qian, Yue Tan, Yixin Liu, Wang Yu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16459)  

**Abstract**: Diffusion large language models (D-LLMs) have emerged as a promising alternative to auto-regressive models due to their iterative refinement capabilities. However, hallucinations remain a critical issue that hinders their reliability. To detect hallucination responses from model outputs, token-level uncertainty (e.g., entropy) has been widely used as an effective signal to indicate potential factual errors. Nevertheless, the fixed-length generation paradigm of D-LLMs implies that tokens contribute unevenly to hallucination detection, with only a small subset providing meaningful signals. Moreover, the evolution trend of uncertainty throughout the diffusion process can also provide important signals, highlighting the necessity of modeling its denoising dynamics for hallucination detection. In this paper, we propose DynHD that bridge these gaps from both spatial (token sequence) and temporal (denoising dynamics) perspectives. To address the information density imbalance across tokens, we propose a semantic-aware evidence construction module that extracts hallucination-indicative signals by filtering out non-informative tokens and emphasizing semantically meaningful ones. To model denoising dynamics for hallucination detection, we introduce a reference evidence generator that learns the expected evolution trajectory of uncertainty evidence, along with a deviation-based hallucination detector that makes predictions by measuring the discrepancy between the observed and reference trajectories. Extensive experiments demonstrate that DynHD consistently outperforms state-of-the-art baselines while achieving higher efficiency across multiple benchmarks and backbone models. 

---
# RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery 

**Authors**: Abhishek Kumar, Aashraya Sachdeva  

**Link**: [PDF](https://arxiv.org/pdf/2603.16411)  

**Abstract**: Entity recognition in Automatic Speech Recognition (ASR) is challenging for rare and domain-specific terms. In domains such as finance, medicine, and air traffic control, these errors are costly. If the entities are entirely absent from the ASR output, post-ASR correction becomes difficult. To address this, we introduce RECOVER, an agentic correction framework that serves as a tool-using agent. It leverages multiple hypotheses as evidence from ASR, retrieves relevant entities, and applies Large Language Model (LLM) correction under constraints. The hypotheses are used using different strategies, namely, 1-Best, Entity-Aware Select, Recognizer Output Voting Error Reduction (ROVER) Ensemble, and LLM-Select. Evaluated across five diverse datasets, it achieves 8-46% relative reductions in entity-phrase word error rate (E-WER) and increases recall by up to 22 percentage points. The LLM-Select achieves the best overall performance in entity correction while maintaining overall WER. 

---
# PlotTwist: A Creative Plot Generation Framework with Small Language Models 

**Authors**: Abhinav Thorat, Ravi Kolla, Jyotin Goel, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2603.16410)  

**Abstract**: Creative plot generation presents a fundamental challenge for language models: transforming a concise premise into a coherent narrative that sustains global structure, character development, and emotional resonance. Although recent Large Language Models (LLMs) demonstrate strong fluency across general-purpose tasks, they typically require preference alignment to perform well on specialized domains such as creative plot generation. However, conducting such alignment at the scale of frontier LLMs is computationally prohibitive, significantly limiting accessibility and practical deployment. To address this, we present PlotTwist, a structured framework that enables Small Language Models (SLMs) with $\leq$ 5B active parameters to generate high-quality, premise-conditioned plots competitive with frontier systems up to $200\times$ larger. Our approach decomposes generation into three specialized components: (1) an Aspect Rating Reward Model trained via a novel Positive-Negative prompting strategy to deliver structured narratives across five Narrative Quality Dimensions (NQDs); (2) a Mixture-of-Experts (MoE) plot generator aligned via Direct Preference Optimization on high-confidence preference pairs; and (3) an Agentic Evaluation module that emulates human critical judgment for unbiased post-hoc assessment. Extensive experiments demonstrate that PlotTwist consistently outperforms frontier models across multiple NQDs despite substantially tighter capacity constraints. Further validation confirms strong sensitivity to narrative quality, as the framework reliably distinguishes plots derived from critically acclaimed versus widely panned screenplays. Together, these results establish structured, preference-based alignment as a resource-efficient approach to high-quality creative plot generation. 

---
# Omnilingual MT: Machine Translation for 1,600 Languages 

**Authors**: Omnilingual MT Team, Belen Alastruey, Niyati Bafna, Andrea Caciolai, Kevin Heffernan, Artyom Kozhevnikov, Christophe Ropers, Eduardo Sánchez, Charles-Eric Saint-James, Ioannis Tsiamas, Chierh Cheng, Joe Chuang, Paul-Ambroise Duquenne, Mark Duppenthaler, Nate Ekberg, Cynthia Gao, Pere Lluís Huguet Cabot, João Maria Janeiro, Jean Maillard, Gabriel Mejia Gonzalez, Holger Schwenk, Edan Toledo, Arina Turkatenko, Albert Ventayol-Boada, Rashel Moritz, Alexandre Mourachko, Surya Parimi, Mary Williamson, Shireen Yates, David Dale, Marta R. Costa-jussà  

**Link**: [PDF](https://arxiv.org/pdf/2603.16309)  

**Abstract**: High-quality machine translation (MT) can scale to hundreds of languages, setting a high bar for multilingual systems. However, compared to the world's 7,000 languages, current systems still offer only limited coverage: about 200 languages on the target side, and maybe a few hundreds more on the source side, supported due to cross-lingual transfer. And even these numbers have been hard to evaluate due to the lack of reliable benchmarks and metrics.
We present Omnilingual Machine Translation (OMT), the first MT system supporting more than 1,600 languages. This scale is enabled by a comprehensive data strategy that integrates large public multilingual corpora with newly created datasets, including manually curated MeDLEY bitext.
We explore two ways of specializing a Large Language model (LLM) for machine translation: as a decoder-only model (OMT-LLaMA) or as a module in an encoder-decoder architecture (OMT-NLLB). Notably, all our 1B to 8B parameter models match or exceed the MT performance of a 70B LLM baseline, revealing a clear specialization advantage and enabling strong translation quality in low-compute settings. Moreover, our evaluation of English-to-1,600 translations further shows that while baseline models can interpret undersupported languages, they frequently fail to generate them with meaningful fidelity; OMT-LLaMA models substantially expand the set of languages for which coherent generation is feasible. Additionally, OMT models improve in cross-lingual transfer, being close to solving the "understanding" part of the puzzle in MT for the 1,600 evaluated. Our leaderboard and main human-created evaluation datasets (BOUQuET and Met-BOUQuET) are dynamically evolving towards Omnilinguality and freely available. 

---
# Parametric Social Identity Injection and Diversification in Public Opinion Simulation 

**Authors**: Hexi Wang, Yujia Zhou, Bangde Du, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16142)  

**Abstract**: Large language models (LLMs) have recently been adopted as synthetic agents for public opinion simulation, offering a promising alternative to costly and slow human surveys. Despite their scalability, current LLM-based simulation methods fail to capture social diversity, producing flattened inter-group differences and overly homogeneous responses within demographic groups. We identify this limitation as a Diversity Collapse phenomenon in LLM hidden representations, where distinct social identities become increasingly indistinguishable across layers. Motivated by this observation, we propose Parametric Social Identity Injection (PSII), a general framework that injects explicit, parametric representations of demographic attributes and value orientations directly into intermediate hidden states of LLMs. Unlike prompt-based persona conditioning, PSII enables fine-grained and controllable identity modulation at the representation level. Extensive experiments on the World Values Survey using multiple open-source LLMs show that PSII significantly improves distributional fidelity and diversity, reducing KL divergence to real-world survey data while enhancing overall diversity. This work provides new insights into representation-level control of LLM agents and advances scalable, diversity-aware public opinion simulation. Code and data are available at this https URL. 

---
# Attention-guided Evidence Grounding for Spoken Question Answering 

**Authors**: Ke Yang, Bolin Chen, Yuejie Li, Yueying Hua, Jianhao Nie, Yueping He, Bowen Li, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16292)  

**Abstract**: Spoken Question Answering (Spoken QA) presents a challenging cross-modal problem: effectively aligning acoustic queries with textual knowledge while avoiding the latency and error propagation inherent in cascaded ASR-based systems. In this paper, we introduce Attention-guided Evidence Grounding (AEG), a novel end-to-end framework that leverages the internal cross-modal attention of Speech Large Language Models (SpeechLLMs) to explicitly locate and ground key evidence in the model's latent space. To address the diffuse attention distribution in pre-trained models, we propose Learning to Focus on Evidence (LFE), a supervised fine-tuning paradigm that calibrates the model's attention mechanism to distinguish query-relevant segments from irrelevant context. Experiments on SQuAD, HotpotQA, and MuSiQue demonstrate that AEG reduces hallucinations and achieves strong efficiency gains, outperforming large-scale cascaded baselines (Whisper-Large-v3 + Reranker) while reducing inference latency by approximately 62%. 

---
# SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment 

**Authors**: Zhouwei Zhai, Mengxiang Chen, Anmeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16137)  

**Abstract**: Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (2) security vulnerabilities under jailbreak attacks that threaten compliance. To address these issues, we propose SI--a Synthesize-Inject-Align framework for building knowledgeable and secure e-commerce search LLMs. Our approach first synthesizes high-quality natural language corpus by combining structured knowledge graphs with unstructured behavioral logs, augmented with reasoning chains and safety-aware this http URL then introduce a parameter-efficient pre-training strategy based on Depth Up-Scaling to inject domain knowledge while preserving general capabilities. Finally, a dual-path alignment method via multi-task instruction tuning and adversarial training strengthens both task performance and safety robustness. The framework has been deployed at this http URL, China's largest self-operated e-commerce platform, where A/B tests across five core search scenarios demonstrate significant improvements in key business metrics, validating its industrial effectiveness and scalability. 

---
# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning 

**Authors**: Kazuki Yano, Shun Kiyono, Sosuke Kobayashi, Sho Takase, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2603.16127)  

**Abstract**: We investigate the role of learning rate scheduling in the large-scale pre-training of large language models, focusing on its influence on downstream performance after supervised fine-tuning (SFT). Decay-based learning rate schedulers are widely used to minimize pre-training loss. However, despite their widespread use, how these schedulers affect performance after SFT remains underexplored. In this paper, we examine Warmup-Stable-Only (WSO), which maintains a constant learning rate after warmup without any decay. Through experiments with 1B and 8B parameter models, we show that WSO consistently outperforms decay-based schedulers in terms of performance after SFT, even though decay-based schedulers may exhibit better performance after pre-training. The result also holds across different regimes with mid-training and over-training. Loss landscape analysis further reveals that decay-based schedulers lead models into sharper minima, whereas WSO preserves flatter minima that support adaptability. These findings indicate that applying LR decay to improve pre-training metrics may compromise downstream adaptability. Our work also provides practical guidance for training and model release strategies, highlighting that pre-training models with WSO enhances their adaptability for downstream tasks. 

---
# ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning 

**Authors**: Tik Yu Yim, Wenting Tan, Sum Yee Chan, Tak-Wah Lam, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16112)  

**Abstract**: Adapting large language models (LLMs) to specialized financial reasoning typically requires expensive fine-tuning that produces model-locked expertise. Training-free alternatives have emerged, yet our experiments show that leading methods (GEPA and ACE) achieve only marginal gains on the FAMMA financial reasoning benchmark, exposing the limits of unstructured text optimization for complex, multi-step domain reasoning. We introduce Automated Skill Distillation and Adaptation (ASDA), a framework that automatically generates structured skill artifacts through iterative error-corrective learning without modifying model weights. A teacher model analyzes a student model's failures on financial reasoning tasks, clusters errors by subfield and error type, and synthesizes skill files containing reasoning procedures, code templates, and worked examples, which are dynamically injected during inference. Evaluated on FAMMA, ASDA achieves up to +17.33% improvement on arithmetic reasoning and +5.95% on non-arithmetic reasoning, substantially outperforming all training-free baselines. The resulting skill artifacts are human-readable, version-controlled, and compatible with the Agent Skills open standard, offering any organization with a labeled domain dataset a practical and auditable path to domain adaptation without weight access or retraining. 

---
# Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability 

**Authors**: Fan Huang, Haewoon Kwak, Jisun An  

**Link**: [PDF](https://arxiv.org/pdf/2603.16017)  

**Abstract**: Large language models (LLMs) increasingly participate in morally sensitive decision-making, yet how they organize ethical frameworks across reasoning steps remains underexplored. We introduce \textit{moral reasoning trajectories}, sequences of ethical framework invocations across intermediate reasoning steps, and analyze their dynamics across six models and three benchmarks. We find that moral reasoning involves systematic multi-framework deliberation: 55.4--57.7\% of consecutive steps involve framework switches, and only 16.4--17.8\% of trajectories remain framework-consistent. Unstable trajectories remain 1.29$\times$ more susceptible to persuasive attacks ($p=0.015$). At the representation level, linear probes localize framework-specific encoding to model-specific layers (layer 63/81 for Llama-3.3-70B; layer 17/81 for Qwen2.5-72B), achieving 13.8--22.6\% lower KL divergence than the training-set prior baseline. Lightweight activation steering modulates framework integration patterns (6.7--8.9\% drift reduction) and amplifies the stability--accuracy relationship. We further propose a Moral Representation Consistency (MRC) metric that correlates strongly ($r=0.715$, $p<0.0001$) with LLM coherence ratings, whose underlying framework attributions are validated by human annotators (mean cosine similarity $= 0.859$). 

---
# COGNAC at SemEval-2026 Task 5: LLM Ensembles for Human-Level Word Sense Plausibility Rating in Challenging Narratives 

**Authors**: Azwad Anjum Islam, Tisa Islam Erana  

**Link**: [PDF](https://arxiv.org/pdf/2603.15897)  

**Abstract**: We describe our system for SemEval-2026 Task 5, which requires rating the plausibility of given word senses of homonyms in short stories on a 5-point Likert scale. Systems are evaluated by the unweighted average of accuracy (within one standard deviation of mean human judgments) and Spearman Rank Correlation. We explore three prompting strategies using multiple closed-source commercial LLMs: (i) a baseline zero-shot setup, (ii) Chain-of-Thought (CoT) style prompting with structured reasoning, and (iii) a comparative prompting strategy for evaluating candidate word senses simultaneously. Furthermore, to account for the substantial inter-annotator variation present in the gold labels, we propose an ensemble setup by averaging model predictions. Our best official system, comprising an ensemble of LLMs across all three prompting strategies, placed 4th on the competition leaderboard with 0.88 accuracy and 0.83 Spearman's rho (0.86 average). Post-competition experiments with additional models further improved this performance to 0.92 accuracy and 0.85 Spearman's rho (0.89 average). We find that comparative prompting consistently improved performance across model families, and model ensembling significantly enhanced alignment with mean human judgments, suggesting that LLM ensembles are especially well suited for subjective semantic evaluation tasks involving multiple annotators. 

---
# POLAR:A Per-User Association Test in Embedding Space 

**Authors**: Pedro Bento, Arthur Buzelin, Arthur Chagas, Yan Aquino, Victoria Estanislau, Samira Malaquias, Pedro Robles Dutenhefner, Gisele L. Pappa, Virgilio Almeida, Wagner MeiraJr  

**Link**: [PDF](https://arxiv.org/pdf/2603.15950)  

**Abstract**: Most intrinsic association probes operate at the word, sentence, or corpus level, obscuring author-level variation. We present POLAR (Per-user On-axis Lexical Association Re-port), a per-user lexical association test that runs in the embedding space of a lightly adapted masked language model. Authors are represented by private deterministic to-kens; POLAR projects these vectors onto curated lexicalaxes and reports standardized effects with permutation p-values and Benjamini--Hochberg control. On a balanced bot--human Twitter benchmark, POLAR cleanly separates LLM-driven bots from organic accounts; on an extremist forum,it quantifies strong alignment with slur lexicons and reveals rightward drift over time. The method is modular to new attribute sets and provides concise, per-author diagnostics for computational social science. All code is publicly avail-able at this https URL. 

---
# RadAnnotate: Large Language Models for Efficient and Reliable Radiology Report Annotation 

**Authors**: Saisha Pradeep Shetty, Roger Eric Goldman, Vladimir Filkov  

**Link**: [PDF](https://arxiv.org/pdf/2603.16002)  

**Abstract**: Radiology report annotation is essential for clinical NLP, yet manual labeling is slow and costly. We present RadAnnotate, an LLM-based framework that studies retrieval-augmented synthetic reports and confidence-based selective automation to reduce expert effort for labeling in RadGraph. We study RadGraph-style entity labeling (graph nodes) and leave relation extraction (edges) to future work. First, we train entity-specific classifiers on gold-standard reports and characterize their strengths and failure modes across anatomy and observation categories, with uncertain observations hardest to learn. Second, we generate RAG-guided synthetic reports and show that synthetic-only models remain within 1-2 F1 points of gold-trained models, and that synthetic augmentation is especially helpful for uncertain observations in a low-resource setting, improving F1 from 0.61 to 0.70. Finally, by learning entity-specific confidence thresholds, RadAnnotate can automatically annotate 55-90% of reports at 0.86-0.92 entity match score while routing low-confidence cases for expert review. 

---
# MedArena: Comparing LLMs for Medicine-in-the-Wild Clinician Preferences 

**Authors**: Eric Wu, Kevin Wu, Jason Hom, Paul H. Yi, Angela Zhang, Alejandro Lozano, Jeff Nirschl, Jeff Tangney, Kevin Byram, Braydon Dymm, Narender Annapureddy, Eric Topol, David Ouyang, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15677)  

**Abstract**: Large language models (LLMs) are increasingly central to clinician workflows, spanning clinical decision support, medical education, and patient communication. However, current evaluation methods for medical LLMs rely heavily on static, templated benchmarks that fail to capture the complexity and dynamics of real-world clinical practice, creating a dissonance between benchmark performance and clinical utility. To address these limitations, we present MedArena, an interactive evaluation platform that enables clinicians to directly test and compare leading LLMs using their own medical queries. Given a clinician-provided query, MedArena presents responses from two randomly selected models and asks the user to select the preferred response. Out of 1571 preferences collected across 12 LLMs up to November 1, 2025, Gemini 2.0 Flash Thinking, Gemini 2.5 Pro, and GPT-4o were the top three models by Bradley-Terry rating. Only one-third of clinician-submitted questions resembled factual recall tasks (e.g., MedQA), whereas the majority addressed topics such as treatment selection, clinical documentation, or patient communication, with ~20% involving multi-turn conversations. Additionally, clinicians cited depth and detail and clarity of presentation more often than raw factual accuracy when explaining their preferences, highlighting the importance of readability and clinical nuance. We also confirm that the model rankings remain stable even after controlling for style-related factors like response length and formatting. By grounding evaluation in real-world clinical questions and preferences, MedArena offers a scalable platform for measuring and improving the utility and efficacy of medical LLMs. 

---
# Efficient Reasoning on the Edge 

**Authors**: Yelysei Bondarenko, Thomas Hehn, Rob Hesselink, Romain Lepert, Fabio Valerio Massoli, Evgeny Mironov, Leyla Mirvakhabova, Tribhuvanesh Orekondy, Spyridon Stasis, Andrey Kuzmin, Anna Kuzina, Markus Nagel, Ankita Nayak, Corrado Rainone, Ork de Rooij, Paul N Whatmough, Arash Behboodi, Babak Ehteshami Bejnordi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16867)  

**Abstract**: Large language models (LLMs) with chain-of-thought reasoning achieve state-of-the-art performance across complex problem-solving tasks, but their verbose reasoning traces and large context requirements make them impractical for edge deployment. These challenges include high token generation costs, large KV-cache footprints, and inefficiencies when distilling reasoning capabilities into smaller models for mobile devices. Existing approaches often rely on distilling reasoning traces from larger models into smaller models, which are verbose and stylistically redundant, undesirable for on-device inference. In this work, we propose a lightweight approach to enable reasoning in small LLMs using LoRA adapters combined with supervised fine-tuning. We further introduce budget forcing via reinforcement learning on these adapters, significantly reducing response length with minimal accuracy loss. To address memory-bound decoding, we exploit parallel test-time scaling, improving accuracy at minor latency increase. Finally, we present a dynamic adapter-switching mechanism that activates reasoning only when needed and a KV-cache sharing strategy during prompt encoding, reducing time-to-first-token for on-device inference. Experiments on Qwen2.5-7B demonstrate that our method achieves efficient, accurate reasoning under strict resource constraints, making LLM reasoning practical for mobile scenarios. Videos demonstrating our solution running on mobile devices are available on our project page. 

---
# Prompt Programming for Cultural Bias and Alignment of Large Language Models 

**Authors**: Maksim Eren, Eric Michalak, Brian Cook, Johnny Seales Jr  

**Link**: [PDF](https://arxiv.org/pdf/2603.16827)  

**Abstract**: Culture shapes reasoning, values, prioritization, and strategic decision-making, yet large language models (LLMs) often exhibit cultural biases that misalign with target populations. As LLMs are increasingly used for strategic decision-making, policy support, and document engineering tasks such as summarization, categorization, and compliance-oriented auditing, improving cultural alignment is important for ensuring that downstream analyses and recommendations reflect target-population value profiles rather than default model priors. Previous work introduced a survey-grounded cultural alignment framework and showed that culture-specific prompting can reduce misalignment, but it primarily evaluated proprietary models and relied on manual prompt engineering. In this paper, we validate and extend that framework by reproducing its social sciences survey based projection and distance metrics on open-weight LLMs, testing whether the same cultural skew and benefits of culture conditioning persist outside closed LLM systems. Building on this foundation, we introduce use of prompt programming with DSPy for this problem-treating prompts as modular, optimizable programs-to systematically tune cultural conditioning by optimizing against cultural-distance objectives. In our experiments, we show that prompt optimization often improves upon cultural prompt engineering, suggesting prompt compilation with DSPy can provide a more stable and transferable route to culturally aligned LLM responses. 

---
# Morphemes Without Borders: Evaluating Root-Pattern Morphology in Arabic Tokenizers and LLMs 

**Authors**: Yara Alakeel, Chatrine Qwaider, Hanan Aldarmaki, Sawsan Alqahtani  

**Link**: [PDF](https://arxiv.org/pdf/2603.15773)  

**Abstract**: This work investigates how effectively large language models (LLMs) and their tokenization schemes represent and generate Arabic root-pattern morphology, probing whether they capture genuine morphological structure or rely on surface memorization. Arabic morphological system provides a rich testbed for analyzing how LLMs handle complex, non-concatenative forms and how tokenization choices influence this process. Our study begins with an evaluation of morphological fidelity across Arabic and multilingual tokenizers against gold-standard segmentation, followed by an analysis of LLM performance in productive root-pattern generation using a newly developed test set. Our findings across seven Arabic-centric and multilingual LLMs and their respective tokenizers reveal that tokenizer morphological alignment is not necessary nor sufficient for morphological generation, which questions the role of morphological tokenization in downstream performance. 

---
# Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights 

**Authors**: Yi Chen, Daiwei Chen, Sukrut Madhav Chikodikar, Caitlyn Heqi Yin, Ramya Korlakai Vinayak  

**Link**: [PDF](https://arxiv.org/pdf/2603.16817)  

**Abstract**: Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient. 

---
# Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning 

**Authors**: Jingxiang Chen, Minseok Kim, Seong-Gyun Leem, Yin Huang, Rashi Rungta, Zhicheng Ouyang, Haibin Wu, Surya Teja Appini, Ankur Bansal, Yang Bai, Yue Liu, Florian Metze, Ahmed A Aly, Anuj Kumar, Ariya Rastrow, Zhaojiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15981)  

**Abstract**: Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguistic signals. We propose multi-task reinforcement learning (RL) with chain-of-thought prompting that elicits explicit affective reasoning. To address data scarcity, we introduce a paralinguistics-aware speech LLM (PALLM) that jointly optimizes sentiment classification from audio and paralinguistics-aware response generation via a two-stage pipeline. Experiments demonstrate that our approach improves paralinguistics understanding over both supervised baselines and strong proprietary models (Gemini-2.5-Pro, GPT-4o-audio) by 8-12% on Expresso, IEMOCAP, and RAVDESS. The results show that modeling paralinguistic reasoning with multi-task RL is crucial for building emotionally intelligent speech LLMs. 

---
# When AI Navigates the Fog of War 

**Authors**: Ming Li, Xirui Li, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16642)  

**Abstract**: Can AI reason about a war before its trajectory becomes historically obvious? Analyzing this capability is difficult because retrospective geopolitical prediction is heavily confounded by training-data leakage. We address this challenge through a temporally grounded case study of the early stages of the 2026 Middle East conflict, which unfolded after the training cutoff of current frontier models. We construct 11 critical temporal nodes, 42 node-specific verifiable questions, and 5 general exploratory questions, requiring models to reason only from information that would have been publicly available at each moment. This design substantially mitigates training-data leakage concerns, creating a setting well-suited for studying how models analyze an unfolding crisis under the fog of war, and provides, to our knowledge, the first temporally grounded analysis of LLM reasoning in an ongoing geopolitical conflict. Our analysis reveals three main findings. First, current state-of-the-art large language models often display a striking degree of strategic realism, reasoning beyond surface rhetoric toward deeper structural incentives. Second, this capability is uneven across domains: models are more reliable in economically and logistically structured settings than in politically ambiguous multi-actor environments. Finally, model narratives evolve over time, shifting from early expectations of rapid containment toward more systemic accounts of regional entrenchment and attritional de-escalation. Since the conflict remains ongoing at the time of writing, this work can serve as an archival snapshot of model reasoning during an unfolding geopolitical crisis, enabling future studies without the hindsight bias of retrospective analysis. 

---
# BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs 

**Authors**: Sangyeon Yoon, Sunkyoung Kim, Hyesoo Hong, Wonje Jeung, Yongil Kim, Wooseok Seo, Heuiyeen Yeen, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2603.16557)  

**Abstract**: Large language models (LLMs) increasingly store user preferences in persistent memory to support personalization across interactions. However, in third-party communication settings governed by social and institutional norms, some user preferences may be inappropriate to apply. We introduce BenchPreS, which evaluates whether memory-based user preferences are appropriately applied or suppressed across communication contexts. Using two complementary metrics, Misapplication Rate (MR) and Appropriate Application Rate (AAR), we find even frontier LLMs struggle to apply preferences in a context-sensitive manner. Models with stronger preference adherence exhibit higher rates of over-application, and neither reasoning capability nor prompt-based defenses fully resolve this issue. These results suggest current LLMs treat personalized preferences as globally enforceable rules rather than as context-dependent normative signals. 

---
# IQuest-Coder-V1 Technical Report 

**Authors**: Jian Yang, Wei Zhang, Shawn Guo, Zhengmao Ye, Lin Jing, Shark Liu, Yizhi Li, Jiajun Wu, Cening Liu, X. Ma, Yuyang Song, Siwei Wu, Yuwen Li, L. Liao, T. Zheng, Ziling Huang, Zelong Huang, Che Liu, Yan Xing, Renyuan Li, Qingsong Cai, Hanxu Yan, Siyue Wang, Shikai Li, Jason Klein Liu, An Huang, Yongsheng Kang, Jinxing Zhang, Chuan Hao, Haowen Wang, Weicheng Gu, Ran Tao, Mingjie Tang, Peihao Wu, Jianzhou Wang, Xianglong Liu, Weifeng Lv, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.16733)  

**Abstract**: In this report, we introduce the IQuest-Coder-V1 series-(7B/14B/40B/40B-Loop), a new family of code large language models (LLMs). Moving beyond static code representations, we propose the code-flow multi-stage training paradigm, which captures the dynamic evolution of software logic through different phases of the pipeline. Our models are developed through the evolutionary pipeline, starting with the initial pre-training consisting of code facts, repository, and completion data. Following that, we implement a specialized mid-training stage that integrates reasoning and agentic trajectories in 32k-context and repository-scale in 128k-context to forge deep logical foundations. The models are then finalized with post-training of specialized coding capabilities, which is bifurcated into two specialized paths: the thinking path (utilizing reasoning-driven RL) and the instruct path (optimized for general assistance). IQuest-Coder-V1 achieves state-of-the-art performance among competitive models across critical dimensions of code intelligence: agentic software engineering, competitive programming, and complex tool use. To address deployment constraints, the IQuest-Coder-V1-Loop variant introduces a recurrent mechanism designed to optimize the trade-off between model capacity and deployment footprint, offering an architecturally enhanced path for efficacy-efficiency trade-off. We believe the release of the IQuest-Coder-V1 series, including the complete white-box chain of checkpoints from pre-training bases to the final thinking and instruction models, will advance research in autonomous code intelligence and real-world agentic systems. 

---
# Offline Exploration-Aware Fine-Tuning for Long-Chain Mathematical Reasoning 

**Authors**: Yongyu Mu, Jiali Zeng, Fandong Meng, JingBo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16206)  

**Abstract**: Through encouraging self-exploration, reinforcement learning from verifiable rewards (RLVR) has significantly advanced the mathematical reasoning capabilities of large language models. As the starting point for RLVR, the capacity of supervised fine-tuning (SFT) to memorize new chain-of-thought trajectories provides a crucial initialization that shapes the subsequent exploration landscape. However, existing research primarily focuses on facilitating exploration during RLVR training, leaving exploration-aware SFT under-explored. To bridge this gap, we propose Offline eXploration-Aware (OXA) fine-tuning. Specifically, OXA optimizes two objectives: promoting low-confidence verified teacher-distillation data to internalize previously uncaptured reasoning patterns, and suppressing high-confidence incorrect self-distillation data to redistribute probability mass of incorrect patterns toward potentially correct candidates. Experimental results across 6 benchmarks show that OXA consistently improves mathematical reasoning performance, especially achieving an average gain of $+6$ Pass@1 and $+5$ Pass@$k$ points compared to conventional SFT on the Qwen2.5-1.5B-Math. Crucially, OXA elevates initial policy entropy, and performance gains persist throughout extensive RLVR training, demonstrating the long-term value of OXA. 

---
# When Stability Fails: Hidden Failure Modes Of LLMS in Data-Constrained Scientific Decision-Making 

**Authors**: Nazia Riasat  

**Link**: [PDF](https://arxiv.org/pdf/2603.15840)  

**Abstract**: Large language models (LLMs) are increasingly used as decision-support tools in data-constrained scientific workflows, where correctness and validity are critical. However, evaluation practices often emphasize stability or reproducibility across repeated runs. While these properties are desirable, stability alone does not guar- antee agreement with statistical ground truth when such references are available. We introduce a controlled behavioral evaluation framework that explicitly sep- arates four dimensions of LLM decision-making: stability, correctness, prompt sensitivity, and output validity under fixed statistical inputs. We evaluate multi- ple LLMs using a statistical gene prioritization task derived from differential ex- pression analysis across prompt regimes involving strict and relaxed significance thresholds, borderline ranking scenarios, and minor wording variations. Our ex- periments show that LLMs can exhibit near-perfect run-to-run stability while sys- tematically diverging from statistical ground truth, over-selecting under relaxed thresholds, responding sharply to minor prompt wording changes, or producing syntactically plausible gene identifiers absent from the input table. Although sta- bility reflects robustness across repeated runs, it does not guarantee agreement with statistical ground truth in structured scientific decision tasks. These findings highlight the importance of explicit ground-truth validation and output validity checks when deploying LLMs in automated or semi-automated scientific work- flows. 

---
# Prompt Engineering for Scale Development in Generative Psychometrics 

**Authors**: Lara Lee Russell-Lasalandra, Hudson Golino  

**Link**: [PDF](https://arxiv.org/pdf/2603.15909)  

**Abstract**: This Monte Carlo simulation examines how prompt engineering strategies shape the quality of large language model (LLM)--generated personality assessment items within the AI-GENIE framework for generative psychometrics. Item pools targeting the Big Five traits were generated using multiple prompting designs (zero-shot, few-shot, persona-based, and adaptive), model temperatures, and LLMs, then evaluated and reduced using network psychometric methods. Across all conditions, AI-GENIE reliably improved structural validity following reduction, with the magnitude of its incremental contribution inversely related to the quality of the incoming item pool. Prompt design exerted a substantial influence on both pre- and post-reduction item quality. Adaptive prompting consistently outperformed non-adaptive strategies by sharply reducing semantic redundancy, elevating pre-reduction structural validity, and preserving substantially larger item pool, particularly when paired with newer, higher-capacity models. These gains were robust across temperature settings for most models, indicating that adaptive prompting mitigates common trade-offs between creativity and psychometric coherence. An exception was observed for the GPT-4o model at high temperatures, suggesting model-specific sensitivity to adaptive constraints at elevated stochasticity. Overall, the findings demonstrate that adaptive prompting is the strongest approach in this context, and that its benefits scale with model capability, motivating continued investigation of model--prompt interactions in generative psychometric pipelines. 

---
# HIPO: Instruction Hierarchy via Constrained Reinforcement Learning 

**Authors**: Keru Chen, Jun Luo, Sen Lin, Yingbin Liang, Alvaro Velasquez, Nathaniel Bastian, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16152)  

**Abstract**: Hierarchical Instruction Following (HIF) refers to the problem of prompting large language models with a priority-ordered stack of instructions. Standard methods like RLHF and DPO typically fail in this problem since they mainly optimize for a single objective, failing to explicitly enforce system prompt compliance. Meanwhile, supervised fine-tuning relies on mimicking filtered, compliant data, which fails to establish the priority asymmetry at the algorithmic level. In this paper, we introduce \textsc{HIPO}, a novel alignment framework that formulates HIF as a Constrained Markov Decision Process. \textsc{HIPO} elevates system prompts from mere input context to strict algorithmic boundaries. Using a primal-dual safe reinforcement learning approach, the algorithm dynamically enforces system prompt compliance as an explicit constraint, maximizing user utility strictly within this feasible region. Extensive evaluations across diverse model architectures (e.g., Qwen, Phi, Llama) demonstrate that \textsc{HIPO} significantly improves both system compliance and user utility. Furthermore, mechanistic analysis reveals that this constrained optimization autonomously drives the model to shift its attention toward long-range system tokens, providing a principled foundation for reliable LLM deployment in complex workflows. 

---
# Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory 

**Authors**: Ce Zhang, Jinxi He, Junyi He, Katia Sycara, Yaqi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.15800)  

**Abstract**: Multi-modal Large Language Models (MLLMs) have achieved remarkable performance across a wide range of visual reasoning tasks, yet their vulnerability to safety risks remains a pressing concern. While prior research primarily focuses on jailbreak defenses that detect and refuse explicitly unsafe inputs, such approaches often overlook contextual safety, which requires models to distinguish subtle contextual differences between scenarios that may appear similar but diverge significantly in safety intent. In this work, we present MM-SafetyBench++, a carefully curated benchmark designed for contextual safety evaluation. Specifically, for each unsafe image-text pair, we construct a corresponding safe counterpart through minimal modifications that flip the user intent while preserving the underlying contextual meaning, enabling controlled evaluation of whether models can adapt their safety behaviors based on contextual understanding. Further, we introduce EchoSafe, a training-free framework that maintains a self-reflective memory bank to accumulate and retrieve safety insights from prior interactions. By integrating relevant past experiences into current prompts, EchoSafe enables context-aware reasoning and continual evolution of safety behavior during inference. Extensive experiments on various multi-modal safety benchmarks demonstrate that EchoSafe consistently achieves superior performance, establishing a strong baseline for advancing contextual safety in MLLMs. All benchmark data and code are available at this https URL. 

---
# Persona-Conditioned Risk Behavior in Large Language Models: A Simulated Gambling Study with GPT-4.1 

**Authors**: Sankalp Dubedy  

**Link**: [PDF](https://arxiv.org/pdf/2603.15831)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents in uncertain, sequential decision-making contexts. Yet it remains poorly understood whether the behaviors they exhibit in such environments reflect principled cognitive patterns or simply surface-level prompt mimicry. This paper presents a controlled experiment in which GPT-4.1 was assigned one of three socioeconomic personas (Rich, Middle-income, and Poor) and placed in a structured slot-machine environment with three distinct machine configurations: Fair (50%), Biased Low (35%), and Streak (dynamic probability increasing after consecutive losses). Across 50 independent iterations per condition and 6,950 recorded decisions, we find that the model reproduces key behavioral signatures predicted by Kahneman and Tversky's Prospect Theory without being instructed to do so. The Poor persona played a mean of 37.4 rounds per session (SD=15.5) compared to 1.1 rounds for the Rich persona (SD=0.31), a difference that is highly significant (Kruskal-Wallis H=393.5, p<2.2e-16). Risk scores by persona show large effect sizes (Cohen's d=4.15 for Poor vs Rich). Emotional labels appear to function as post-hoc annotations rather than decision drivers (chi-square=3205.4, Cramer's V=0.39), and belief-updating across rounds is negligible (Spearman rho=0.032 for Poor persona, p=0.016). These findings carry implications for LLM agent design, interpretability research, and the broader question of whether classical cognitive economic biases are implicitly encoded in large-scale pretrained language models. 

---
# BrainBench: Exposing the Commonsense Reasoning Gap in Large Language Models 

**Authors**: Yuzhe Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14761)  

**Abstract**: Large language models (LLMs) achieve impressive scores on standard benchmarks yet routinely fail questions that any human would answer correctly in seconds. We introduce BrainBench, a benchmark of 100 brainteaser questions spanning 20 carefully designed categories, each targeting a specific commonsense reasoning failure mode in LLMs. Categories range from implicit physical constraints ("Should I walk or drive my rental car to the return lot?") to semantic scope tricks and default assumption hijacks. We evaluate eight frontier models -- four from the Claude family and four from the GPT family -- using a zero-shot protocol with 10 independent runs per question. The best model, Claude Opus 4.6 with extended thinking, achieves only 80.3% accuracy; the worst, GPT-4o, scores 39.7%. Even top-performing models exhibit a 6-16 percentage-point gap between accuracy and consistency, revealing stochastic reasoning. Cross-lingual evaluation in Chinese shows most models degrade by 2-8 percentage points, confirming that these failures reflect reasoning deficits rather than language-specific artifacts. BrainBench provides a fine-grained diagnostic tool for identifying where and why LLMs substitute surface heuristics for genuine commonsense reasoning. 

---
# Temporal Fact Conflicts in LLMs: Reproducibility Insights from Unifying DYNAMICQA and MULAN 

**Authors**: Ritajit Dey, Iadh Ounis, Graham McDonald, Yashar Moshfeghi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15892)  

**Abstract**: Large Language Models (LLMs) often struggle with temporal fact conflicts due to outdated or evolving information in their training data. Two recent studies with accompanying datasets report opposite conclusions on whether external context can effectively resolve such conflicts. DYNAMICQA evaluates how effective external context is in shifting the model's output distribution, finding that temporal facts are more resistant to change. In contrast, MULAN examines how often external context changes memorised facts, concluding that temporal facts are easier to update. In this reproducibility paper, we first reproduce experiments from both benchmarks. We then reproduce the experiments of each study on the dataset of the other to investigate the source of their disagreement. To enable direct comparison of findings, we standardise both datasets to align with the evaluation settings of each study. Importantly, using an LLM, we synthetically generate realistic natural language contexts to replace MULAN's programmatically constructed statements when reproducing the findings of DYNAMICQA. Our analysis reveals strong dataset dependence: MULAN's findings generalise under both methodological frameworks, whereas applying MULAN's evaluation to DYNAMICQA yields mixed outcomes. Finally, while the original studies only considered 7B LLMs, we reproduce these experiments across LLMs of varying sizes, revealing how model size influences the encoding and updating of temporal facts. Our results highlight how dataset design, evaluation metrics, and model size shape LLM behaviour in the presence of temporal knowledge conflicts. 

---
# ReFORM: Review-aggregated Profile Generation via LLM with Multi-Factor Attention for Restaurant Recommendation 

**Authors**: Moonsoo Park, Seulbeen Je, Donghyeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.16236)  

**Abstract**: In recommender systems, large language models (LLMs) have gained popularity for generating descriptive summarization to improve recommendation robustness, along with Graph Convolution Networks. However, existing LLM-enhanced recommendation studies mainly rely on the internal knowledge of LLMs about item titles while neglecting the importance of various factors influencing users' decisions. Although information reflecting various decision factors of each user is abundant in reviews, few studies have actively exploited such insights for recommendation. To address these limitations, we propose a ReFORM: Review-aggregated Profile Generation via LLM with Multi-FactOr Attentive RecoMmendation framework. Specifically, we first generate factor-specific user and item profiles from reviews using LLM to capture a user's preference by items and an item's evaluation by users. Then, we propose a Multi-Factor Attention to highlight the most influential factors in each user's decision-making process. In this paper, we conduct experiments on two restaurant datasets of varying scales, demonstrating its robustness and superior performance over state-of-the-art baselines. Furthermore, in-depth analyses validate the effectiveness of the proposed modules and provide insights into the sources of personalization. Our source code and datasets are available at this https URL. 

---
# Embedding-Aware Feature Discovery: Bridging Latent Representations and Interpretable Features in Event Sequences 

**Authors**: Artem Sakhno, Ivan Sergeev, Alexey Shestov, Omar Zoloev, Elizaveta Kovtun, Gleb Gusev, Andrey Savchenko, Maksim Makarenko  

**Link**: [PDF](https://arxiv.org/pdf/2603.15713)  

**Abstract**: Industrial financial systems operate on temporal event sequences such as transactions, user actions, and system logs. While recent research emphasizes representation learning and large language models, production systems continue to rely heavily on handcrafted statistical features due to their interpretability, robustness under limited supervision, and strict latency constraints. This creates a persistent disconnect between learned embeddings and feature-based pipelines. We introduce Embedding-Aware Feature Discovery (EAFD), a unified framework that bridges this gap by coupling pretrained event-sequence embeddings with a self-reflective LLM-driven feature generation agent. EAFD iteratively discovers, evaluates, and refines features directly from raw event sequences using two complementary criteria: \emph{alignment}, which explains information already encoded in embeddings, and \emph{complementarity}, which identifies predictive signals missing from them. Across both open-source and industrial transaction benchmarks, EAFD consistently outperforms embedding-only and feature-based baselines, achieving relative gains of up to $+5.8\%$ over state-of-the-art pretrained embeddings, resulting in new state-of-the-art performance across event-sequence datasets. 

---
# NextMem: Towards Latent Factual Memory for LLM-based Agents 

**Authors**: Zeyu Zhang, Rui Li, Xiaoyan Zhao, Yang Zhang, Wenjie Wang, Xu Chen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2603.15634)  

**Abstract**: Memory is critical for LLM-based agents to preserve past observations for future decision-making, where factual memory serves as its foundational part. However, existing approaches to constructing factual memory face several limitations. Textual methods impose heavy context and indexing burdens, while parametric methods suffer from catastrophic forgetting and high costs. To address these challenges, we introduce NextMem, a latent factual memory framework that utilizes an autoregressive autoencoder to efficiently construct latent memory while ensuring accurate reconstruction. For better optimization, we propose a two-stage training process, including autoregressive reconstruction alignment and progressive latent substitution. We also incorporate quantization to reduce storage overhead. Extensive experiments demonstrate that NextMem achieves superior performance, and excels in retrieval, robustness, and extensibility properties. We release our code and model checkpoints at this https URL. 

---
# Runtime Governance for AI Agents: Policies on Paths 

**Authors**: Maurits Kaptein, Vassilis-Javed Khan, Andriy Podstavnychy  

**Link**: [PDF](https://arxiv.org/pdf/2603.16586)  

**Abstract**: AI agents -- systems that plan, reason, and act using large language models -- produce non-deterministic, path-dependent behavior that cannot be fully governed at design time, where with governed we mean striking the right balance between as high as possible successful task completion rate and the legal, data-breach, reputational and other costs associated with running agents. We argue that the execution path is the central object for effective runtime governance and formalize compliance policies as deterministic functions mapping agent identity, partial path, proposed next action, and organizational state to a policy violation probability. We show that prompt-level instructions (and "system prompts"), and static access control are special cases of this framework: the former shape the distribution over paths without actually evaluating them; the latter evaluates deterministic policies that ignore the path (i.e., these can only account for a specific subset of all possible paths). In our view, runtime evaluation is the general case, and it is necessary for any path-dependent policy. We develop the formal framework for analyzing AI agent governance, present concrete policy examples (inspired by the AI act), discuss a reference implementation, and identify open problems including risk calibration and the limits of enforced compliance. 

---
# Exploring different approaches to customize language models for domain-specific text-to-code generation 

**Authors**: Luís Freire, Fernanda A. Andaló, Nicki Skafte Detlefsen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16526)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in generating executable code from natural language descriptions. However, general-purpose models often struggle in specialized programming contexts where domain-specific libraries, APIs, or conventions must be used. Customizing smaller open-source models offers a cost-effective alternative to relying on large proprietary systems. In this work, we investigate how smaller language models can be adapted for domain-specific code generation using synthetic datasets. We construct datasets of programming exercises across three domains within the Python ecosystem: general Python programming, Scikit-learn machine learning workflows, and OpenCV-based computer vision tasks. Using these datasets, we evaluate three customization strategies: few-shot prompting, retrieval-augmented generation (RAG), and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA). Performance is evaluated using both benchmark-based metrics and similarity-based metrics that measure alignment with domain-specific code. Our results show that prompting-based approaches such as few-shot learning and RAG can improve domain relevance in a cost-effective manner, although their impact on benchmark accuracy is limited. In contrast, LoRA-based fine-tuning consistently achieves higher accuracy and stronger domain alignment across most tasks. These findings highlight practical trade-offs between flexibility, computational cost, and performance when adapting smaller language models for specialized programming tasks. 

---
# Differential Harm Propensity in Personalized LLM Agents: The Curious Case of Mental Health Disclosure 

**Authors**: Caglar Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2603.16734)  

**Abstract**: Large language models (LLMs) are increasingly deployed as tool-using agents, shifting safety concerns from harmful text generation to harmful task completion. Deployed systems often condition on user profiles or persistent memory, yet agent safety evaluations typically ignore personalization signals. To address this gap, we investigated how mental health disclosure, a sensitive and realistic user-context cue, affects harmful behavior in agentic settings. Building on the AgentHarm benchmark, we evaluated frontier and open-source LLMs on multi-step malicious tasks (and their benign counterparts) under controlled prompt conditions that vary user-context personalization (no bio, bio-only, bio+mental health disclosure) and include a lightweight jailbreak injection. Our results reveal that harmful task completion is non-trivial across models: frontier lab models (e.g., GPT 5.2, Claude Sonnet 4.5, Gemini 3-Pro) still complete a measurable fraction of harmful tasks, while an open model (DeepSeek 3.2) exhibits substantially higher harmful completion. Adding a bio-only context generally reduces harm scores and increases refusals. Adding an explicit mental health disclosure often shifts outcomes further in the same direction, though effects are modest and not uniformly reliable after multiple-testing correction. Importantly, the refusal increase also appears on benign tasks, indicating a safety--utility trade-off via over-refusal. Finally, jailbreak prompting sharply elevates harm relative to benign conditions and can weaken or override the protective shift induced by personalization. Taken together, our results indicate that personalization can act as a weak protective factor in agentic misuse settings, but it is fragile under minimal adversarial pressure, highlighting the need for personalization-aware evaluations and safeguards that remain robust across user-context conditions. 

---
# ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation 

**Authors**: Zihe Wang, Yihuan Wang, Haiyang Yu. Zhiyong Cui, Xiaojian Liao, Chengcheng Wang, Yonglin Tian, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16495)  

**Abstract**: The current expressway operation relies on rule-based and isolated models, which limits the ability to jointly analyze knowledge across different systems. Meanwhile, Large Language Models (LLMs) are increasingly applied in intelligent transportation, advancing traffic models from algorithmic to cognitive intelligence. However, general LLMs are unable to effectively understand the regulations and causal relationships of events in unconventional scenarios in the expressway field. Therefore, this paper constructs a pre-trained multimodal large language model (MLLM) for expressways, ExpressMind, which serves as the cognitive core for intelligent expressway operations. This paper constructs the industry's first full-stack expressway dataset, encompassing traffic knowledge texts, emergency reasoning chains, and annotated video events to overcome data scarcity. This paper proposes a dual-layer LLM pre-training paradigm based on self-supervised training and unsupervised learning. Additionally, this study introduces a Graph-Augmented RAG framework to dynamically index the expressway knowledge base. To enhance reasoning for expressway incident response strategies, we develop a RL-aligned Chain-of-Thought (RL-CoT) mechanism that enforces consistency between model reasoning and expert problem-solving heuristics for incident handling. Finally, ExpressMind integrates a cross-modal encoder to align the dynamic feature sequences under the visual and textual channels, enabling it to understand traffic scenes in both video and image modalities. Extensive experiments on our newly released multi-modal expressway benchmark demonstrate that ExpressMind comprehensively outperforms existing baselines in event detection, safety response generation, and complex traffic analysis. The code and data are available at: this https URL. 

---
# RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments 

**Authors**: Linghua Zhang, Jun Wang, Jingtong Wu, Zhisong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16453)  

**Abstract**: Large Language Model (LLM)-based agents have achieved notable success on short-horizon and highly structured tasks. However, their ability to maintain coherent decision-making over long horizons in realistic and dynamic environments remains an open challenge.
We introduce RetailBench, a high-fidelity benchmark designed to evaluate long-horizon autonomous decision-making in realistic commercial scenarios, where agents must operate under stochastic demand and evolving external conditions.
We further propose the Evolving Strategy & Execution framework, which separates high-level strategic reasoning from low-level action execution. This design enables adaptive and interpretable strategy evolution over time. It is particularly important for long-horizon tasks, where non-stationary environments and error accumulation require strategies to be revised at a different temporal scale than action execution.
Experiments on eight state-of-the-art LLMs across progressively challenging environments show that our framework improves operational stability and efficiency compared to other baselines. However, performance degrades substantially as task complexity increases, revealing fundamental limitations in current LLMs for long-horizon, multi-factor decision-making. 

---
# From Natural Language to Executable Option Strategies via Large Language Models 

**Authors**: Haochen Luo, Zhengzhao Lai, Junjie Xu, Yifan Li, Tang Pok Hin, Yuan Zhang, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16434)  

**Abstract**: Large Language Models (LLMs) excel at general code generation, yet translating natural-language trading intents into correct option strategies remains challenging. Real-world option design requires reasoning over massive, multi-dimensional option chain data with strict constraints, which often overwhelms direct generation methods. We introduce the Option Query Language (OQL), a domain-specific intermediate representation that abstracts option markets into high-level primitives under grammatical rules, enabling LLMs to function as reliable semantic parsers rather than free-form programmers. OQL queries are then validated and executed deterministically by an engine to instantiate executable strategies. We also present a new dataset for this task and demonstrate that our neuro-symbolic pipeline significantly improves execution accuracy and logical consistency over direct baselines. 

---
# Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior to Positive Preferences 

**Authors**: Quan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.16417)  

**Abstract**: Recent empirical results have demonstrated that training large language models (LLMs) with negative-only feedback can match or exceed standard reinforcement learning from human feedback (RLHF). Negative Sample Reinforcement achieves parity with PPO on mathematical reasoning; Distributional Dispreference Optimization trains effectively using only dispreferred samples; and Constitutional AI outperforms pure RLHF on harmlessness benchmarks. Yet no unified theoretical account explains why negative signals are so effective. This paper proposes such an account: positive preferences and negative constraints are structurally asymmetric. Positive preferences ("which is better") encode continuously coupled, context-dependent human values that cannot be exhaustively specified -- leading models to learn surface correlates such as agreement with the user (sycophancy). Negative constraints ("what is wrong") encode discrete, finite, independently verifiable prohibitions that can converge to a stable boundary. This asymmetry -- rooted in Popper's falsification logic and the epistemology of negative knowledge -- explains both the sycophancy failure of preference-based RLHF and the surprising effectiveness of negative-signal methods. We argue that alignment research should shift its center of gravity from "learning what humans prefer" to "learning what humans reject," and offer testable predictions for this framework. 

---
# FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment 

**Authors**: Qinhong Lin, Ruitao Feng, Yinglun Feng, Zhenxin Huang, Yukun Chen, Zhongliang Yang, Linna Zhou, Binjie Fei, Jiaqi Liu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16365)  

**Abstract**: We study alpha factor mining, the automated discovery of predictive signals from noisy, non-stationary market data-under a practical requirement that mined factors be directly executable and auditable, and that the discovery process remain computationally tractable at scale. Existing symbolic approaches are limited by bounded expressiveness, while neural forecasters often trade interpretability for performance and remain vulnerable to regime shifts and overfitting. We introduce FactorEngine (FE), a program-level factor discovery framework that casts factors as Turing-complete code and improves both effectiveness and efficiency via three separations: (i) logic revision vs. parameter optimization, (ii) LLM-guided directional search vs. Bayesian hyperparameter search, and (iii) LLM usage vs. local computation. FE further incorporates a knowledge-infused bootstrapping module that transforms unstructured financial reports into executable factor programs through a closed-loop multi-agent extraction-verification-code-generation pipeline, and an experience knowledge base that supports trajectory-aware refinement (including learning from failures). Across extensive backtests on real-world OHLCV data, FE produces factors with substantially stronger predictive stability and portfolio impact-for example, higher IC/ICIR (and Rank IC/ICIR) and improved AR/Sharpe, than baseline methods, achieving state-of-the-art predictive and portfolio performance. 

---
# Breaking the Chain: A Causal Analysis of LLM Faithfulness to Intermediate Structures 

**Authors**: Oleg Somov, Mikhail Chaichuk, Mikhail Seleznyov, Alexander Panchenko, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2603.16475)  

**Abstract**: Schema-guided reasoning pipelines ask LLMs to produce explicit intermediate structures -- rubrics, checklists, verification queries -- before committing to a final decision. But do these structures causally determine the output, or merely accompany it? We introduce a causal evaluation protocol that makes this directly measurable: by selecting tasks where a deterministic function maps intermediate structures to decisions, every controlled edit implies a unique correct output. Across eight models and three benchmarks, models appear self-consistent with their own intermediate structures but fail to update predictions after intervention in up to 60% of cases -- revealing that apparent faithfulness is fragile once the intermediate structure changes. When derivation of the final decision from the structure is delegated to an external tool, this fragility largely disappears; however, prompts which ask to prioritize the intermediate structure over the original input do not materially close the gap. Overall, intermediate structures in schema-guided pipelines function as influential context rather than stable causal mediators. 

---
# Learning to Predict, Discover, and Reason in High-Dimensional Discrete Event Sequences 

**Authors**: Hugo Math  

**Link**: [PDF](https://arxiv.org/pdf/2603.16313)  

**Abstract**: Electronic control units (ECUs) embedded within modern vehicles generate a large number of asynchronous events known as diagnostic trouble codes (DTCs). These discrete events form complex temporal sequences that reflect the evolving health of the vehicle's subsystems. In the automotive industry, domain experts manually group these codes into higher-level error patterns (EPs) using Boolean rules to characterize system faults and ensure safety. However, as vehicle complexity grows, this manual process becomes increasingly costly, error-prone, and difficult to scale. Notably, the number of unique DTCs in a modern vehicle is on the same order of magnitude as the vocabulary of a natural language, often numbering in the tens of thousands. This observation motivates a paradigm shift: treating diagnostic sequences as a language that can be modeled, predicted, and ultimately explained. Traditional statistical approaches fail to capture the rich dependencies and do not scale to high-dimensional datasets characterized by thousands of nodes, large sample sizes, and long sequence lengths. Specifically, the high cardinality of categorical event spaces in industrial logs poses a significant challenge, necessitating new machine learning architectures tailored to such event-driven systems. This thesis addresses automated fault diagnostics by unifying event sequence modeling, causal discovery, and large language models (LLMs) into a coherent framework for high-dimensional event streams. It is structured in three parts, reflecting a progressive transition from prediction to causal understanding and finally to reasoning for vehicle diagnostics. Consequently, we introduce several Transformer-based architectures for predictive maintenance, scalable sample- and population-level causal discovery frameworks and a multi-agent system that automates the synthesis of Boolean EP rules. 

---
# Adaptive Theory of Mind for LLM-based Multi-Agent Coordination 

**Authors**: Chunjiang Mu, Ya Zeng, Qiaosheng Zhang, Kun Shao, Chen Chu, Hao Guo, Danyang Jia, Zhen Wang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16264)  

**Abstract**: Theory of Mind (ToM) refers to the ability to reason about others' mental states, and higher-order ToM involves considering that others also possess their own ToM. Equipping large language model (LLM)-driven agents with ToM has long been considered to improve their coordination in multiagent collaborative tasks. However, we find that misaligned ToM orders-mismatches in the depth of ToM reasoning between agents-can lead to insufficient or excessive reasoning about others, thereby impairing their coordination. To address this issue, we design an adaptive ToM (A-ToM) agent, which can align in ToM orders with its partner. Based on prior interactions, the agent estimates the partner's likely ToM order and leverages this estimation to predict the partner's action, thereby facilitating behavioral coordination. We conduct empirical evaluations on four multi-agent coordination tasks: a repeated matrix game, two grid navigation tasks and an Overcooked task. The results validate our findings on ToM alignment and demonstrate the effectiveness of our A-ToM agent. Furthermore, we discuss the generalizability of our A-ToM to non-LLM-based agents, as well as what would diminish the importance of ToM alignment. 

---
# POaaS: Minimal-Edit Prompt Optimization as a Service to Lift Accuracy and Cut Hallucinations on On-Device sLLMs 

**Authors**: Jungwoo Shim, Dae Won Kim, Sun Wook Kim, Soo Young Kim, Myungcheol Lee, Jae-geun Cha, Hyunhwa Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16045)  

**Abstract**: Small language models (sLLMs) are increasingly deployed on-device, where imperfect user prompts--typos, unclear intent, or missing context--can trigger factual errors and hallucinations. Existing automatic prompt optimization (APO) methods were designed for large cloud LLMs and rely on search that often produces long, structured instructions; when executed under an on-device constraint where the same small model must act as optimizer and solver, these pipelines can waste context and even hurt accuracy. We propose POaaS, a minimal-edit prompt optimization layer that routes each query to lightweight specialists (Cleaner, Paraphraser, Fact-Adder) and merges their outputs under strict drift and length constraints, with a conservative skip policy for well-formed prompts. Under a strict fixed-model setting with Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct, POaaS improves both task accuracy and factuality while representative APO baselines degrade them, and POaaS recovers up to +7.4% under token deletion and mixup. Overall, per-query conservative optimization is a practical alternative to search-heavy APO for on-device sLLMs. 

---
# Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation 

**Authors**: Dongik Shin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16044)  

**Abstract**: Generalization remains a core challenge in embodied AI, as robots must adapt to diverse environments. While OpenVLA represents the State-of-the-Art (SOTA) in Vision-Language-Action models by leveraging large-scale pre-training, its zero-shot performance can be limited when encountering completely new environments. This paper proposes a parameter-efficient fine-tuning strategy to enhance the linguistic generalization of OpenVLA by synthesizing a general instruction set for the Bridge Dataset V2. The paper leverages a Large Language Model (LLM) to generate a rich variety of semantically equivalent but structurally diverse commands for existing trajectories. In this experiment, Low-Rank Adaptation (LoRA) is implemented to fine-tune OpenVLA on augmented pairs, allowing the model to bridge the gap between complex natural language intent and robotic actions. Results demonstrate that the LoRA-enhanced model's robustness, suggesting that enriching the linguistic space of specialized datasets is crucial for embodied agents. 

---
# A Context Alignment Pre-processor for Enhancing the Coherence of Human-LLM Dialog 

**Authors**: Ding Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.16052)  

**Abstract**: Large language models (LLMs) have made remarkable progress in generating fluent text, but they still face a critical challenge of contextual misalignment in long-term and dynamic dialogue. When human users omit premises, simplify references, or shift context abruptly during interactions with LLMs, the models may fail to capture their actual intentions, producing mechanical or off-topic responses that weaken the collaborative potential of dialogue. To address this problem, this paper proposes a computational framework called the Context Alignment Pre-processor (C.A.P.). Rather than operating during generation, C.A.P. functions as a pre-processing module between user input and response generation. The framework includes three core processes: (1) semantic expansion, which extends a user instruction to a broader semantic span including its premises, literal meaning, and implications; (2) time-weighted context retrieval, which prioritizes recent dialogue history through a temporal decay function approximating human conversational focus; and (3) alignment verification and decision branching, which evaluates whether the dialogue remains on track by measuring the semantic similarity between the current prompt and the weighted historical context. When a significant deviation is detected, C.A.P. initiates a structured clarification protocol to help users and the system recalibrate the conversation. This study presents the architecture and theoretical basis of C.A.P., drawing on cognitive science and Common Ground theory in human-computer interaction. We argue that C.A.P. is not only a technical refinement but also a step toward shifting human-computer dialogue from one-way command-execution patterns to two-way, self-correcting, partnership-based collaboration. Finally, we discuss implementation paths, evaluation methods, and implications for the future design of interactive intelligent systems. 

---
# Argumentative Human-AI Decision-Making: Toward AI Agents That Reason With Us, Not For Us 

**Authors**: Stylianos Loukas Vasileiou, Antonio Rago, Francesca Toni, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2603.15946)  

**Abstract**: Computational argumentation offers formal frameworks for transparent, verifiable reasoning but has traditionally been limited by its reliance on domain-specific information and extensive feature engineering. In contrast, LLMs excel at processing unstructured text, yet their opaque nature makes their reasoning difficult to evaluate and trust. We argue that the convergence of these fields will lay the foundation for a new paradigm: Argumentative Human-AI Decision-Making. We analyze how the synergy of argumentation framework mining, argumentation framework synthesis, and argumentative reasoning enables agents that do not just justify decisions, but engage in dialectical processes where decisions are contestable and revisable -- reasoning with humans rather than for them. This convergence of computational argumentation and LLMs is essential for human-aware, trustworthy AI in high-stakes domains. 

---
# Protein Design with Agent Rosetta: A Case Study for Specialized Scientific Agents 

**Authors**: Jacopo Teneggi, S.M. Bargeen A. Turzo, Tanya Marwah, Alberto Bietti, P. Douglas Renfrew, Vikram Khipple Mulligan, Siavash Golkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.15952)  

**Abstract**: Large language models (LLMs) are capable of emulating reasoning and using tools, creating opportunities for autonomous agents that execute complex scientific tasks. Protein design provides a natural testbed: although machine learning (ML) methods achieve strong results, these are largely restricted to canonical amino acids and narrow objectives, leaving unfilled need for a generalist tool for broad design pipelines. We introduce Agent Rosetta, an LLM agent paired with a structured environment for operating Rosetta, the leading physics-based heteropolymer design software, capable of modeling non-canonical building blocks and geometries. Agent Rosetta iteratively refines designs to achieve user-defined objectives, combining LLM reasoning with Rosetta's generality. We evaluate Agent Rosetta on design with canonical amino acids, matching specialized models and expert baselines, and with non-canonical residues -- where ML approaches fail -- achieving comparable performance. Critically, prompt engineering alone often fails to generate Rosetta actions, demonstrating that environment design is essential for integrating LLM agents with specialized software. Our results show that properly designed environments enable LLM agents to make scientific software accessible while matching specialized tools and human experts. 

---
# CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems 

**Authors**: Pearl Mody, Mihir Panchal, Rishit Kar, Kiran Bhowmick, Ruhina Karani  

**Link**: [PDF](https://arxiv.org/pdf/2603.15642)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in long running workflows, where they must preserve user and task state across many turns. Many existing agent memory systems behave like external databases with ad hoc read/write rules, which can yield unstable retention, limited consolidation, and vulnerability to distractor content. We present CraniMem, a neurocognitively motivated, gated and bounded multi-stage memory design for agentic systems. CraniMem couples goal conditioned gating and utility tagging with a bounded episodic buffer for near term continuity and a structured long-term knowledge graph for durable semantic recall. A scheduled consolidation loop replays high utility traces into the graph while pruning low utility items, keeping memory growth in check and reducing interference. On long horizon benchmarks evaluated under both clean inputs and injected noise, CraniMem is more robust than a Vanilla RAG and Mem0 baseline and exhibits smaller performance drops under distraction. Our code is available at this https URL and the accompanying PyPI package at this https URL. 

---
# GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure 

**Authors**: Shaohuang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15643)  

**Abstract**: Green Stormwater Infrastructure (GSI) systems, such as permeable pavement, rain gardens, and bioretention facilities, require continuous inspection and maintenance to ensure long-term perfor- mance. However, domain knowledge about GSI is often scattered across municipal manuals, regula- tory documents, and inspection forms. As a result, non-expert users and maintenance staff may strug- gle to obtain reliable and actionable guidance from field observations. Although Large Language Models (LLMs) have demonstrated strong general reasoning and language generation capabilities, they often lack domain-specific knowledge and may produce inaccurate or hallucinated answers in engineering scenarios. This limitation restricts their direct application to professional infrastructure tasks. In this paper, we propose GSI Agent, a domain-enhanced LLM framework designed to im- prove performance in GSI-related tasks. Our approach integrates three complementary strategies: (1) supervised fine-tuning (SFT) on a curated GSI instruction dataset, (2) retrieval-augmented gen- eration (RAG) over an internal GSI knowledge base constructed from municipal documents, and (3) an agent-based reasoning pipeline that coordinates retrieval, context integration, and structured response generation. We also construct a new GSI Dataset aligned with real-world GSI inspection and maintenance scenarios. Experimental results show that our framework significantly improves domain-specific performance while maintaining general knowledge capability. On the GSI dataset, BLEU-4 improves from 0.090 to 0.307, while performance on the common knowledge dataset re- mains stable (0.304 vs. 0.305). These results demonstrate that systematic domain knowledge en- hancement can effectively adapt general-purpose LLMs to professional infrastructure applications. 

---
# MLLM-based Textual Explanations for Face Comparison 

**Authors**: Redwan Sony, Anil K Jain, Ross Arun  

**Link**: [PDF](https://arxiv.org/pdf/2603.16629)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently been proposed as a means to generate natural-language explanations for face recognition decisions. While such explanations facilitate human interpretability, their reliability on unconstrained face images remains underexplored. In this work, we systematically analyze MLLM-generated explanations for the unconstrained face verification task on the challenging IJB-S dataset, with a particular focus on extreme pose variation and surveillance imagery. Our results show that even when MLLMs produce correct verification decisions, the accompanying explanations frequently rely on non-verifiable or hallucinated facial attributes that are not supported by visual evidence. We further study the effect of incorporating information from traditional face recognition systems, viz., scores and decisions, alongside the input images. Although such information improves categorical verification performance, it does not consistently lead to faithful explanations. To evaluate the explanations beyond decision accuracy, we introduce a likelihood-ratio-based framework that measures the evidential strength of textual explanations. Our findings highlight fundamental limitations of current MLLMs for explainable face recognition and underscore the need for a principled evaluation of reliable and trustworthy explanations in biometric applications. Code is available at this https URL. 

---
# When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making 

**Authors**: Jun Liu, Pu Zhao, Zhenglun Kong, Xuan Shen, Peiyan Dong, Fan Yang, Lin Cui, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Xue Lin, Gaowen Liu, Yanzhi Wang, Dong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16673)  

**Abstract**: Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents. 

---
# Trained Persistent Memory for Frozen Encoder--Decoder LLMs: Six Architectural Methods 

**Authors**: Hong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16413)  

**Abstract**: Frozen encoder--decoder language models are stateless: the latent representation is discarded after every forward pass, so no information persists across sessions. This paper presents a \textbf{proof-of-concept pilot study} showing that persistent memory in the \emph{continuous latent space} of a frozen LLM is feasible -- even under severe resource constraints (a single frozen Flan-T5-XL backbone, small trainable adapters, a single dataset). We implement six architectural methods spanning three injection points and four write mechanisms; unlike text-level memory systems, every write and read is a differentiable operation on dense vectors. After training only the adapter, the memory bank continues to accumulate at inference time without gradients, enabling \emph{conversational learning}. Under a forgetting-curve evaluation on LoCoMo at two capacity scales (1$\times$ and 10$\times$), the stateless baseline scores exactly zero; at 10$\times$ all six trained adapters produce positive memory-recall curves; at 1$\times$ three methods collapse, revealing capacity as a critical design parameter. Because the memory bank is a compact numerical array, it can be scaled to arbitrarily large capacity without altering the backbone. We argue that full end-to-end training with larger models, larger data, and orders-of-magnitude larger memory will yield substantially stronger results; this pilot study establishes the feasibility baseline and design-space taxonomy that such efforts require. 

---
# Toward Experimentation-as-a-Service in 5G/6G: The Plaza6G Prototype for AI-Assisted Trials 

**Authors**: Sergio Barrachina-Muñoz, Marc Carrascosa-Zamacois, Horacio Bleda, Umair Riaz, Yasir Maqsood, Xavier Calle, Selva Vía, Miquel Payaró, Josep Mangues-Bafalluy  

**Link**: [PDF](https://arxiv.org/pdf/2603.16356)  

**Abstract**: This paper presents Plaza6G, the first operational Experiment-as-a-Service (ExaS) platform unifying cloud resources with next-generation wireless infrastructure. Developed at CTTC in Barcelona, Plaza6G integrates GPU-accelerated compute clusters, multiple 5G cores, both open-source (e.g., Free5GC) and commercial (e.g., Cumucore), programmable RANs, and physical or emulated user equipment under unified orchestration. In Plaza6G, the experiment design requires minimal expertise as it is expressed in natural language via a web portal or a REST API. The web portal and REST API are enhanced with a Large Language Model (LLM)-based assistant, which employs retrieval-augmented generation (RAG) for up-to-date experiment knowledge and Low-Rank Adaptation (LoRA) for continuous domain fine-tuning. Over-the-air (OTA) trials leverage a four-chamber anechoic facility and a dual-site outdoor 5G network operating in sub-6~GHz and mmWave bands. Demonstrations include automated CI/CD integration with sub-ten-minute setup and interactive OTA testing under programmable propagation conditions. Machine-readable experiment descriptors ensure reproducibility, while future work targets policy-aware orchestration, safety validation, and federated testbed integration toward open, reproducible wireless experimentation. 

---
# An Interpretable Machine Learning Framework for Non-Small Cell Lung Cancer Drug Response Analysis 

**Authors**: Ann Rachel, Pranav M Pawar, Mithun Mukharjee, Raja M, Tojo Mathew  

**Link**: [PDF](https://arxiv.org/pdf/2603.16330)  

**Abstract**: Lung cancer is a condition where there is abnormal growth of malignant cells that spread in an uncontrollable fashion in the lungs. Some common treatment strategies are surgery, chemotherapy, and radiation which aren't the best options due to the heterogeneous nature of cancer. In personalized medicine, treatments are tailored according to the individual's genetic information along with lifestyle aspects. In addition, AI-based deep learning methods can analyze large sets of data to find early signs of cancer, types of tumor, and prospects of treatment. The paper focuses on the development of personalized treatment plans using specific patient data focusing primarily on the genetic profile. Multi-Omics data from Genomics of Drug Sensitivity in Cancer have been used to build a predictive model along with machine learning techniques. The value of the target variable, LN-IC50, determines how sensitive or resistive a drug is. An XGBoost regressor is utilized to predict the drug response focusing on molecular and cellular features extracted from cancer datasets. Cross-validation and Randomized Search are performed for hyperparameter tuning to further optimize the model's predictive performance. For explanation purposes, SHAP (SHapley Additive exPlanations) was used. SHAP values measure each feature's impact on an individual prediction. Furthermore, interpreting feature relationships was performed using DeepSeek, a large language model trained to verify the biological validity of the features. Contextual explanations regarding the most important genes or pathways were provided by DeepSeek alongside the top SHAP value constituents, supporting the predictability of the model. 

---
# 360° Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method 

**Authors**: Huyen T. T. Tran, Van-Quang Nguyen, Farros Alferro, Kang-Jun Liu, Takayuki Okatani  

**Link**: [PDF](https://arxiv.org/pdf/2603.16179)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive abilities in understanding and reasoning over conventional images. However, their perception of 360° images remains largely underexplored. Unlike conventional images, 360° images capture the entire surrounding environment, enabling holistic spatial reasoning but introducing challenges such as geometric distortion and complex spatial relations. To comprehensively assess MLLMs' capabilities to perceive 360° images, we introduce 360Bench, a Visual Question Answering (VQA) benchmark featuring 7K-resolution 360° images, seven representative (sub)tasks with annotations carefully curated by human annotators. Using 360Bench, we systematically evaluate seven MLLMs and six enhancement methods, revealing their shortcomings in 360° image perception. To address these challenges, we propose Free360, a training-free scene-graph-based framework for high-resolution 360° VQA. Free360 decomposes the reasoning process into modular steps, applies adaptive spherical image transformations to 360° images tailored to each step, and seamlessly integrates the resulting information into a unified graph representation for answer generation. Experiments show that Free360 consistently improves its base MLLM and provides a strong training-free solution for 360° VQA tasks. The source code and dataset will be publicly released upon acceptance. 

---
# Structure-Aware Multimodal LLM Framework for Trustworthy Near-Field Beam Prediction 

**Authors**: Mengyuan Li, Qianfan Lu, Jiachen Tian, Hongjun Hu, Yu Han, Xiao Li, Chao-kai Wen, Shi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16143)  

**Abstract**: In near-field extremely large-scale multiple-input multiple-output (XL-MIMO) systems, spherical wavefront propagation expands the traditional beam codebook into the joint angular-distance domain, rendering conventional beam training prohibitively inefficient, especially in complex 3-dimensional (3D) low-altitude environments. Furthermore, since near-field beam variations are deeply coupled not only with user positions but also with the physical surroundings, precise beam alignment demands profound environmental understanding capabilities. To address this, we propose a large language model (LLM)-driven multimodal framework that fuses historical GPS data, RGB image, LiDAR data, and strategically designed task-specific textual prompts. By utilizing the powerful emergent reasoning and generalization capabilities of the LLM, our approach learns complex spatial dynamics to achieve superior environmental comprehension... 

---
# Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective 

**Authors**: Noppanat Wadlom, Junyi Shen, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16104)  

**Abstract**: Agentic workflows are composed of sequences of interdependent Large Language Model (LLM) calls, and they have become a dominant workload in modern AI systems. These workflows exhibit extensive redundancy from overlapping prompts and intermediate results due to speculative and parallel exploration. Existing LLM serving systems, such as vLLM, focus on optimizing individual inference calls and overlook cross-call dependencies, leading to significant inefficiencies. This paper rethinks LLM and agent serving from a data systems perspective and introduces Helium, a workflow-aware serving framework that models agentic workloads as query plans and treats LLM invocations as first-class operators. Helium integrates proactive caching and cache-aware scheduling to maximize reuse across prompts, KV states, and workflows. Through these techniques, Helium bridges classic query optimization principles with LLM serving, achieving up to 1.56x speedup over state-of-the-art agent serving systems on various workloads. Our results demonstrate that end-to-end optimization across workflows is essential for scalable and efficient LLM-based agents. 

---
# MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale 

**Authors**: Hanxian Huang, Igor Fedorov, Andrey Gromov, Bernard Beckerman, Naveen Suda, David Eriksson, Maximilian Balandat, Rylan Conway, Patrick Huber, Chinnadhurai Sankar, Ayushi Dalmia, Zechun Liu, Lemeng Wu, Tarek Elgamal, Adithya Sagar, Vikas Chandra, Raghuraman Krishnamoorthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15954)  

**Abstract**: Real-time AI experiences call for on-device large language models (OD-LLMs) optimized for efficient deployment on resource-constrained hardware. The most useful OD-LLMs produce near-real-time responses and exhibit broad hardware compatibility, maximizing user reach. We present a methodology for designing such models using hardware-in-the-loop architecture search under mobile latency constraints. This system is amenable to industry-scale deployment: it generates models deployable without custom kernels and compatible with standard mobile runtimes like Executorch. Our methodology avoids specialized attention mechanisms and instead uses attention skipping for long-context acceleration.
Our approach jointly optimizes model architecture (layers, dimensions) and attention pattern. To efficiently evaluate candidates, we treat each as a pruned version of a pretrained backbone with inherited weights, thereby achieving high accuracy with minimal continued pretraining. We leverage the low cost of latency evaluation in a staged process: learning an accurate latency model first, then searching for the Pareto-frontier across latency and quality.
This yields MobileLLM-Flash, a family of foundation models (350M, 650M, 1.4B) for efficient on-device use with strong capabilities, supporting up to 8k context length. MobileLLM-Flash delivers up to 1.8x and 1.6x faster prefill and decode on mobile CPUs with comparable or superior quality. Our analysis of Pareto-frontier design choices offers actionable principles for OD-LLM design. 

---
# A Family of LLMs Liberated from Static Vocabularies 

**Authors**: Aleph Alpha, Adnen Abdessaied, Artur Baranowski, Lukas Balles, Michael Barlow, Fabien C. Y. Benureau, Felix Berkenkamp, Lukas Bluebaum, Bastian Boll, Thomas F. Burns, Björn Deiseroth, Constantin Eichenberg, David Friede, Pablo Iyu Guerrero, Ahmed Hammam, Bastian Harren, Johann Higl, Yasser Jadidi, Carina Kauf, Johannes Messner, Jan Hendrik Metzen, Max Meuer, Vedant Nanda, Pit Neitemeier, Koen Oostermeijer, Letitia Parcalabescu, Markus Pernpointner, Felix Reinfurt, Dylan Rodriquez, Grégory Schott, Philipp Siedler, Martin Simonovsky, Till Speicher, Volker Stampa, Stephan Wäldchen, Samuel Weinbach, Gregor Ziegltrum  

**Link**: [PDF](https://arxiv.org/pdf/2603.15953)  

**Abstract**: Tokenization is a central component of natural language processing in current large language models (LLMs), enabling models to convert raw text into processable units. Although learned tokenizers are widely adopted, they exhibit notable limitations, including their large, fixed vocabulary sizes and poor adaptability to new domains or languages. We present a family of models with up to 70 billion parameters based on the hierarchical autoregressive transformer (HAT) architecture. In HAT, an encoder transformer aggregates bytes into word embeddings and then feeds them to the backbone, a classical autoregressive transformer. The outputs of the backbone are then cross-attended by the decoder and converted back into bytes. We show that we can reuse available pre-trained models by converting the Llama 3.1 8B and 70B models into the HAT architecture: Llama-3.1-8B-TFree-HAT and Llama-3.1-70B-TFree-HAT are byte-level models whose encoder and decoder are trained from scratch, but where we adapt the pre-trained Llama backbone, i.e., the transformer blocks with the embedding matrix and head removed, to handle word embeddings instead of the original tokens. We also provide a 7B HAT model, Llama-TFree-HAT-Pretrained, trained entirely from scratch on nearly 4 trillion words. The HAT architecture improves text compression by reducing the number of required sequence positions and enhances robustness to intra-word variations, e.g., spelling differences. Through pre-training, as well as subsequent supervised fine-tuning and direct preference optimization in English and German, we show strong proficiency in both languages, improving on the original Llama 3.1 in most benchmarks. We release our models (including 200 pre-training checkpoints) on Hugging Face. 

---
# Don't Trust Stubborn Neighbors: A Security Framework for Agentic Networks 

**Authors**: Samira Abedini, Sina Mavali, Lea Schönherr, Martin Pawelczyk, Rebekka Burkholz  

**Link**: [PDF](https://arxiv.org/pdf/2603.15809)  

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MASs) are increasingly deployed for agentic tasks, such as web automation, itinerary planning, and collaborative problem solving. Yet, their interactive nature introduces new security risks: malicious or compromised agents can exploit communication channels to propagate misinformation and manipulate collective outcomes.
In this paper, we study how such manipulation can arise and spread by borrowing the Friedkin-Johnsen opinion formation model from social sciences to propose a general theoretical framework to study LLM-MAS. Remarkably, this model closely captures LLM-MAS behavior, as we verify in extensive experiments across different network topologies and attack and defense scenarios. Theoretically and empirically, we find that a single highly stubborn and persuasive agent can take over MAS dynamics, underscoring the systems' high susceptibility to attacks by triggering a persuasion cascade that reshapes collective opinion. Our theoretical analysis reveals three mechanisms to increase system security: a) increasing the number of benign agents, b) increasing the innate stubbornness or peer-resistance of agents, or c) reducing trust in potential adversaries. Because scaling is computationally expensive and high stubbornness degrades the network's ability to reach consensus, we propose a new mechanism to mitigate threats by a trust-adaptive defense that dynamically adjusts inter-agent trust to limit adversarial influence while maintaining cooperative performance. Extensive experiments confirm that this mechanism effectively defends against manipulation. 

---
# Data-Local Autonomous LLM-Guided Neural Architecture Search for Multiclass Multimodal Time-Series Classification 

**Authors**: Emil Hardarson, Luka Biedebach, Ómar Bessi Ómarsson, Teitur Hrólfsson, Anna Sigridur Islind, María Óskarsdóttir  

**Link**: [PDF](https://arxiv.org/pdf/2603.15939)  

**Abstract**: Applying machine learning to sensitive time-series data is often bottlenecked by the iteration loop: Performance depends strongly on preprocessing and architecture, yet training often has to run on-premise under strict data-local constraints. This is a common problem in healthcare and other privacy-constrained domains (e.g., a hospital developing deep learning models on patient EEG). This bottleneck is particularly challenging in multimodal fusion, where sensor modalities must be individually preprocessed and then combined. LLM-guided neural architecture search (NAS) can automate this exploration, but most existing workflows assume cloud execution or access to data-derived artifacts that cannot be exposed.
We present a novel data-local, LLM-guided search framework that handles candidate pipelines remotely while executing all training and evaluation locally under a fixed protocol. The controller observes only trial-level summaries, such as pipeline descriptors, metrics, learning-curve statistics, and failure logs, without ever accessing raw samples or intermediate feature representations. Our framework targets multiclass, multimodal learning via one-vs-rest binary experts per class and modality, a lightweight fusion MLP, and joint search over expert architectures and modality-specific preprocessing.
We evaluate our method on two regimes: UEA30 (public multivariate time-series classification dataset) and SleepEDFx sleep staging (heterogeneous clinical modalities such as EEG, EOG, and EMG). The results show that the modular baseline model is strong, and the LLM-guided NAS further improves it. Notably, our method finds models that perform within published ranges across most benchmark datasets. Across both settings, our method reduces manual intervention by enabling unattended architecture search while keeping sensitive data on-premise. 

---
# LLM-Driven Discovery of High-Entropy Catalysts via Retrieval-Augmented Generation 

**Authors**: AI Scientists, Xinyi Lin, Danqing Yin, Ying Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.15712)  

**Abstract**: CO2 reduction requires efficient catalysts, yet materials discovery remains bottlenecked by 10-20 year development cycles requiring deep domain expertise. This paper demonstrates how large language models can assist the catalyst discovery process by helping researchers explore chemical spaces and interpret results when augmented with retrieval-based grounding. We introduce a retrieval-augmented generation framework that enables GPT-4 to navigate chemical space by accessing a database of 50,000+ known materials, adapting general-purpose language understanding for high-throughput materials design. Our approach generated over 250 catalyst candidates with an 82% thermodynamic stability rate while addressing multi-objective constraints: 68% achieved <$100/kg cost with metallic conductivity (band gap<0.1eV) and mechanical stability (B/G>1.75). The best-performing Fe0.2Co0.2Ni0.2Ir0.1Ru0.3 achieves 0.285V limiting potential (25% improvement over IrO2), while Cr0.2Fe0.2Co0.3Ni0.2Mo0.1 optimally balances performance-cost trade-offs at $18/kg. Volcano plot analysis confirms that 78% of LLM-generated catalysts cluster near the theoretical activity optimum, while our system achieves 200x computational efficiency compared to traditional high-throughput screening. By demonstrating that retrieval-augmented generation can ground AI creativity in physical constraints without sacrificing exploration, this work demonstrates an approach where natural language interfaces can streamline materials discovery workflows, enabling researchers to explore chemical spaces more efficiently while the LLM assists in result interpretation and hypothesis generation. 

---
# OMNIFLOW: A Physics-Grounded Multimodal Agent for Generalized Scientific Reasoning 

**Authors**: Hao Wu, Yongheng Zhang, Yuan Gao, Fan Xu, Fan Zhang, Ruobing Xie, Ruijian Gou, Yuxuan Liang, Xiaomeng Huang, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15797)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional logical reasoning capabilities but frequently struggle with the continuous spatiotemporal dynamics governed by Partial Differential Equations (PDEs), often resulting in non-physical hallucinations. Existing approaches typically resort to costly, domain-specific fine-tuning, which severely limits cross-domain generalization and interpretability. To bridge this gap, we propose OMNIFLOW, a neuro-symbolic architecture designed to ground frozen multimodal LLMs in fundamental physical laws without requiring domain-specific parameter updates. OMNIFLOW introduces a novel \textit{Semantic-Symbolic Alignment} mechanism that projects high-dimensional flow tensors into topological linguistic descriptors, enabling the model to perceive physical structures rather than raw pixel values. Furthermore, we construct a Physics-Guided Chain-of-Thought (PG-CoT) workflow that orchestrates reasoning through dynamic constraint injection (e.g., mass conservation) and iterative reflexive verification. We evaluate OMNIFLOW on a comprehensive benchmark spanning microscopic turbulence, theoretical Navier-Stokes equations, and macroscopic global weather forecasting. Empirical results demonstrate that OMNIFLOW significantly outperforms traditional deep learning baselines in zero-shot generalization and few-shot adaptation tasks. Crucially, it offers transparent, physically consistent reasoning reports, marking a paradigm shift from black-box fitting to interpretable scientific reasoning. 

---
# BadLLM-TG: A Backdoor Defender powered by LLM Trigger Generator 

**Authors**: Ruyi Zhang, Heng Gao, Songlei Jian, Yusong Tan, Haifang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15692)  

**Abstract**: Backdoor attacks compromise model reliability by using triggers to manipulate outputs. Trigger inversion can accurately locate these triggers via a generator and is therefore critical for backdoor defense. However, the discrete nature of text prevents existing noise-based trigger generator from being applied to nature language processing (NLP). To overcome the limitations, we employ the rich knowledge embedded in large language models (LLMs) and propose a Backdoor defender powered by LLM Trigger Generator, termed BadLLM-TG. It is optimized through prompt-driven reinforcement learning, using the victim model's feedback loss as the reward signal. The generated triggers are then employed to mitigate the backdoor via adversarial training. Experiments show that our method reduces the attack success rate by 76.2\% on average, outperforming the second-best defender by 13.7. 

---
# SEMAG: Self-Evolutionary Multi-Agent Code Generation 

**Authors**: Yulin Peng, Haowen Hou, Xinxin Zhu, Ying Tiffany He, F. Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15707)  

**Abstract**: Large Language Models (LLMs) have made significant progress in handling complex programming tasks. However, current methods rely on manual model selection and fixed workflows, which limit their ability to adapt to changing task complexities. To address this, we propose SEMAG, a Self-Evolutionary Multi-Agent code Generation framework that mimics human coding practices. It decomposes programming tasks into stages, including planning, coding, debugging, and discussion, while adapting workflows to task difficulty. Its self-evolutionary agents can access the latest models in real time and automatically upgrade the backbone model. SEMAG sets new state-of-the-art Pass@1 accuracy across benchmarks. Using identical backbone models, SEMAG outperforms prior methods by 3.3% on CodeContests. When augmented with self-evolutionary model selection that automatically identifies optimal backbones, SEMAG reaches 52.6%, showcasing both framework effectiveness and adaptability to evolving LLM capabilities. 

---
# DRCY: Agentic Hardware Design Reviews 

**Authors**: Kyle Dumont, Nicholas Herbert, Hayder Tirmazi, Shrikanth Upadhayaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.15672)  

**Abstract**: Hardware design errors discovered after fabrication require costly physical respins that can delay products by months. Existing electronic design automation (EDA) tools enforce structural connectivity rules. However, they cannot verify that connections are \emph{semantically} correct with respect to component datasheets. For example, that a symbol's pinout matches the manufacturer's specification, or that a voltage regulator's feedback resistors produce the intended output. We present DRCY, the first production-ready multi-agent LLM system that automates first-pass schematic connection review by autonomously fetching component datasheets, performing pin-by-pin analysis against extracted specifications, and posting findings as inline comments on design reviews. DRCY is deployed in production on AllSpice Hub, a collaborative hardware design platform, where it runs as a CI/CD action triggered on design review submissions. DRCY is used regularly by major hardware companies for use-cases ranging from multi-agent vehicle design to space exploration. We describe DRCY's five-agent pipeline architecture, its agentic datasheet retrieval system with self-evaluation, and its multi-run consensus mechanism for improving reliability on safety-critical analyses 

---
# Steering Frozen LLMs: Adaptive Social Alignment via Online Prompt Routing 

**Authors**: Zeyu Zhang, Xiangxiang Dai, Ziyi Han, Xutong Liu, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2603.15647)  

**Abstract**: Large language models (LLMs) are typically governed by post-training alignment (e.g., RLHF or DPO), which yields a largely static policy during deployment and inference. However, real-world safety is a full-lifecycle problem: static defenses degrade against evolving jailbreak behaviors, and fixed weights cannot adapt to pluralistic, time-varying safety norms. This motivates inference-time governance that steers behavior without costly retraining. To address this, we introduce the Consensus Clustering LinUCB Bandit (CCLUB), a unified framework for adaptive social alignment via system-prompt routing. CCLUB employs a conservative consensus clustering mechanism: it pools data only within the intersection of utility and safety similarity graphs, effectively preventing unsafe generalization across semantically proximal but risk-divergent contexts. Our theoretical analysis yields a sublinear regret guarantee, demonstrating near-optimal performance of CCLUB. Extensive experiments validate that CCLUB outperforms strong baselines, achieving a 10.98% improvement in cumulative reward and a 14.42% reduction in the average suboptimality gap. 

---
