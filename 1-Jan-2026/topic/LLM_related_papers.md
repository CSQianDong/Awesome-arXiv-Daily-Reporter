# CogRec: A Cognitive Recommender Agent Fusing Large Language Models and Soar for Explainable Recommendation 

**Authors**: Jiaxin Hu, Tao Wang, Bingsan Yang, Hongrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24113)  

**Abstract**: Large Language Models (LLMs) have demonstrated a remarkable capacity in understanding user preferences for recommendation systems. However, they are constrained by several critical challenges, including their inherent "Black-Box" characteristics, susceptibility to knowledge hallucination, and limited online learning capacity. These factors compromise their trustworthiness and adaptability. Conversely, cognitive architectures such as Soar offer structured and interpretable reasoning processes, yet their knowledge acquisition is notoriously laborious. To address these complementary challenges, we propose a novel cognitive recommender agent called CogRec which synergizes the strengths of LLMs with the Soar cognitive architecture. CogRec leverages Soar as its core symbolic reasoning engine and leverages an LLM for knowledge initialization to populate its working memory with production rules. The agent operates on a Perception-Cognition-Action(PCA) cycle. Upon encountering an impasse, it dynamically queries the LLM to obtain a reasoned solution. This solution is subsequently transformed into a new symbolic production rule via Soar's chunking mechanism, thereby enabling robust online learning. This learning paradigm allows the agent to continuously evolve its knowledge base and furnish highly interpretable rationales for its recommendations. Extensive evaluations conducted on three public datasets demonstrate that CogRec demonstrates significant advantages in recommendation accuracy, explainability, and its efficacy in addressing the long-tail problem. 

---
# Encyclo-K: Evaluating LLMs with Dynamically Composed Knowledge Statements 

**Authors**: Yiming Liang, Yizhi Li, Yantao Du, Ge Zhang, Jiayi Zhou, Yuchen Wu, Yinzhu Piao, Denghui Cao, Tong Sun, Ziniu Li, Li Du, Bo Lei, Jiaheng Liu, Chenghua Lin, Zhaoxiang Zhang, Wenhao Huang, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24867)  

**Abstract**: Benchmarks play a crucial role in tracking the rapid advancement of large language models (LLMs) and identifying their capability boundaries. However, existing benchmarks predominantly curate questions at the question level, suffering from three fundamental limitations: vulnerability to data contamination, restriction to single-knowledge-point assessment, and reliance on costly domain expert annotation. We propose Encyclo-K, a statement-based benchmark that rethinks benchmark construction from the ground up. Our key insight is that knowledge statements, not questions, can serve as the unit of curation, and questions can then be constructed from them. We extract standalone knowledge statements from authoritative textbooks and dynamically compose them into evaluation questions through random sampling at test time. This design directly addresses all three limitations: the combinatorial space is too vast to memorize, and model rankings remain stable across dynamically generated question sets, enabling reliable periodic dataset refresh; each question aggregates 8-10 statements for comprehensive multi-knowledge assessment; annotators only verify formatting compliance without requiring domain expertise, substantially reducing annotation costs. Experiments on over 50 LLMs demonstrate that Encyclo-K poses substantial challenges with strong discriminative power. Even the top-performing OpenAI-GPT-5.1 achieves only 62.07% accuracy, and model performance displays a clear gradient distribution--reasoning models span from 16.04% to 62.07%, while chat models range from 9.71% to 50.40%. These results validate the challenges introduced by dynamic evaluation and multi-statement comprehensive understanding. These findings establish Encyclo-K as a scalable framework for dynamic evaluation of LLMs' comprehensive understanding over multiple fine-grained disciplinary knowledge statements. 

---
# Compute-Accuracy Pareto Frontiers for Open-Source Reasoning Large Language Models 

**Authors**: Ákos Prucs, Márton Csutora, Mátyás Antal, Márk Marosi  

**Link**: [PDF](https://arxiv.org/pdf/2512.24776)  

**Abstract**: Large Language Models (LLMs) are demonstrating rapid improvements on complex reasoning benchmarks, particularly when allowed to utilize intermediate reasoning steps before converging on a final solution. However, current literature often overlooks the significant computational burden associated with generating long reasoning sequences. For industrial applications, model selection depends not only on raw accuracy but also on resource constraints and inference costs. In this work, we conduct a test-time-compute aware evaluation of both contemporary and older open-source LLMs, mapping their Pareto frontiers across math- and reasoning-intensive benchmarks. Our findings identify the Mixture of Experts (MoE) architecture as a strong candidate to balance performance and efficiency in our evaluation setting. Furthermore, we trace the trajectory of Pareto efficiency over time to derive an emergent trend regarding accuracy gain per unit of compute. Finally, we demonstrate that there is a saturation point for inference-time compute. Beyond a certain threshold, accuracy gains diminish, indicating that while extended reasoning capabilities are beneficial, they cannot overcome intrinsic model limitations regarding specific complexities. 

---
# Do Large Language Models Know What They Are Capable Of? 

**Authors**: Casey O. Barkan, Sid Black, Oliver Sourbut  

**Link**: [PDF](https://arxiv.org/pdf/2512.24661)  

**Abstract**: We investigate whether large language models (LLMs) can predict whether they will succeed on a given task and whether their predictions improve as they progress through multi-step tasks. We also investigate whether LLMs can learn from in-context experiences to make better decisions about whether to pursue a task in scenarios where failure is costly. All LLMs we tested are overconfident, but most predict their success with better-than-random discriminatory power. We find that newer and larger LLMs generally do not have greater discriminatory power, though Claude models do show such a trend. On multi-step agentic tasks, the overconfidence of several frontier LLMs worsens as they progress through the tasks, and reasoning LLMs perform comparably to or worse than non-reasoning LLMs. With in-context experiences of failure, some but not all LLMs reduce their overconfidence leading to significantly improved decision making, while others do not. Interestingly, all LLMs' decisions are approximately rational given their estimated probabilities of success, yet their overly-optimistic estimates result in poor decision making. These results suggest that current LLM agents are hindered by their lack of awareness of their own capabilities. We discuss the implications of LLMs' awareness of their capabilities for AI misuse and misalignment risks. 

---
# Understanding and Steering the Cognitive Behaviors of Reasoning Models at Test-Time 

**Authors**: Zhenyu Zhang, Xiaoxia Wu, Zhongzhu Zhou, Qingyang Wu, Yineng Zhang, Pragaash Ponnusamy, Harikaran Subbaraj, Jue Wang, Shuaiwen Leon Song, Ben Athiwaratkun  

**Link**: [PDF](https://arxiv.org/pdf/2512.24574)  

**Abstract**: Large Language Models (LLMs) often rely on long chain-of-thought (CoT) reasoning to solve complex tasks. While effective, these trajectories are frequently inefficient, leading to high latency from excessive token generation, or unstable reasoning that alternates between underthinking (shallow, inconsistent steps) and overthinking (repetitive, verbose reasoning). In this work, we study the structure of reasoning trajectories and uncover specialized attention heads that correlate with distinct cognitive behaviors such as verification and backtracking. By lightly intervening on these heads at inference time, we can steer the model away from inefficient modes. Building on this insight, we propose CREST, a training-free method for Cognitive REasoning Steering at Test-time. CREST has two components: (1) an offline calibration step that identifies cognitive heads and derives head-specific steering vectors, and (2) an inference-time procedure that rotates hidden representations to suppress components along those vectors. CREST adaptively suppresses unproductive reasoning behaviors, yielding both higher accuracy and lower computational cost. Across diverse reasoning benchmarks and models, CREST improves accuracy by up to 17.5% while reducing token usage by 37.6%, offering a simple and effective pathway to faster, more reliable LLM reasoning. 

---
# Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models 

**Authors**: Junru Lu, Jiarui Qin, Lingfeng Qiao, Yinghui Li, Xinyi Dai, Bo Ke, Jianfeng He, Ruizhi Qiao, Di Yin, Xing Sun, Yunsheng Wu, Yinsong Liu, Shuangyin Liu, Mingkong Tang, Haodong Lin, Jiayi Kuang, Fanxu Meng, Xiaojuan Tang, Yunjia Xi, Junjie Huang, Haotong Yang, Zhenyi Shen, Yangning Li, Qianwen Zhang, Yifei Yu, Siyu An, Junnan Dong, Qiufeng Wang, Jie Wang, Keyu Chen, Wei Wen, Taian Guo, Zhifeng Shen, Daohai Yu, Jiahao Li, Ke Li, Zongyi Li, Xiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2512.24618)  

**Abstract**: We introduce Youtu-LLM, a lightweight yet powerful language model that harmonizes high computational efficiency with native agentic intelligence. Unlike typical small models that rely on distillation, Youtu-LLM (1.96B) is pre-trained from scratch to systematically cultivate reasoning and planning capabilities. The key technical advancements are as follows: (1) Compact Architecture with Long-Context Support: Built on a dense Multi-Latent Attention (MLA) architecture with a novel STEM-oriented vocabulary, Youtu-LLM supports a 128k context window. This design enables robust long-context reasoning and state tracking within a minimal memory footprint, making it ideal for long-horizon agent and reasoning tasks. (2) Principled "Commonsense-STEM-Agent" Curriculum: We curated a massive corpus of approximately 11T tokens and implemented a multi-stage training strategy. By progressively shifting the pre-training data distribution from general commonsense to complex STEM and agentic tasks, we ensure the model acquires deep cognitive abilities rather than superficial alignment. (3) Scalable Agentic Mid-training: Specifically for the agentic mid-training, we employ diverse data construction schemes to synthesize rich and varied trajectories across math, coding, and tool-use domains. This high-quality data enables the model to internalize planning and reflection behaviors effectively. Extensive evaluations show that Youtu-LLM sets a new state-of-the-art for sub-2B LLMs. On general benchmarks, it achieves competitive performance against larger models, while on agent-specific tasks, it significantly surpasses existing SOTA baselines, demonstrating that lightweight models can possess strong intrinsic agentic capabilities. 

---
# BIOME-Bench: A Benchmark for Biomolecular Interaction Inference and Multi-Omics Pathway Mechanism Elucidation from Scientific Literature 

**Authors**: Sibo Wei, Peng Chen, Lifeng Dong, Yin Luo, Lei Wang, Peng Zhang, Wenpeng Lu, Jianbin Guo, Hongjun Yang, Dajun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2512.24733)  

**Abstract**: Multi-omics studies often rely on pathway enrichment to interpret heterogeneous molecular changes, but pathway enrichment (PE)-based workflows inherit structural limitations of pathway resources, including curation lag, functional redundancy, and limited sensitivity to molecular states and interventions. Although recent work has explored using large language models (LLMs) to improve PE-based interpretation, the lack of a standardized benchmark for end-to-end multi-omics pathway mechanism elucidation has largely confined evaluation to small, manually curated datasets or ad hoc case studies, hindering reproducible progress. To address this issue, we introduce BIOME-Bench, constructed via a rigorous four-stage workflow, to evaluate two core capabilities of LLMs in multi-omics analysis: Biomolecular Interaction Inference and end-to-end Multi-Omics Pathway Mechanism Elucidation. We develop evaluation protocols for both tasks and conduct comprehensive experiments across multiple strong contemporary models. Experimental results demonstrate that existing models still exhibit substantial deficiencies in multi-omics analysis, struggling to reliably distinguish fine-grained biomolecular relation types and to generate faithful, robust pathway-level mechanistic explanations. 

---
# MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes 

**Authors**: Siddhant Agarwal, Adya Dhuler, Polly Ruhnke, Melvin Speisman, Md Shad Akhtar, Shweta Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2512.25015)  

**Abstract**: Over the past years, memes have evolved from being exclusively a medium of humorous exchanges to one that allows users to express a range of emotions freely and easily. With the ever-growing utilization of memes in expressing depressive sentiments, we conduct a study on identifying depressive symptoms exhibited by memes shared by users of online social media platforms. We introduce RESTOREx as a vital resource for detecting depressive symptoms in memes on social media through the Large Language Model (LLM) generated and human-annotated explanations. We introduce MAMAMemeia, a collaborative multi-agent multi-aspect discussion framework grounded in the clinical psychology method of Cognitive Analytic Therapy (CAT) Competencies. MAMAMemeia improves upon the current state-of-the-art by 7.55% in macro-F1 and is established as the new benchmark compared to over 30 methods. 

---
# HaluNet: Multi-Granular Uncertainty Modeling for Efficient Hallucination Detection in LLM Question Answering 

**Authors**: Chaodong Tong, Qi Zhang, Jiayang Gao, Lei Jiang, Yanbing Liu, Nannan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.24562)  

**Abstract**: Large Language Models (LLMs) excel at question answering (QA) but often generate hallucinations, including factual errors or fabricated content. Detecting hallucinations from internal uncertainty signals is attractive due to its scalability and independence from external resources. Existing methods often aim to accurately capture a single type of uncertainty while overlooking the complementarity among different sources, particularly between token-level probability uncertainty and the uncertainty conveyed by internal semantic representations, which provide complementary views on model reliability. We present \textbf{HaluNet}, a lightweight and trainable neural framework that integrates multi granular token level uncertainties by combining semantic embeddings with probabilistic confidence and distributional uncertainty. Its multi branch architecture adaptively fuses what the model knows with the uncertainty expressed in its outputs, enabling efficient one pass hallucination detection. Experiments on SQuAD, TriviaQA, and Natural Questions show that HaluNet delivers strong detection performance and favorable computational efficiency, with or without access to context, highlighting its potential for real time hallucination detection in LLM based QA systems. 

---
# Adaptive Dependency-aware Prompt Optimization Framework for Multi-Step LLM Pipeline 

**Authors**: Minjun Zhao, Xinyu Zhang, Shuai Zhang, Deyang Li, Ruifeng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2512.24933)  

**Abstract**: Multi-step LLM pipelines invoke large language models multiple times in a structured sequence and can effectively solve complex tasks, but their performance heavily depends on the prompts used at each step. Jointly optimizing these prompts is difficult due to missing step-level supervision and inter-step dependencies. Existing end-to-end prompt optimization methods struggle under these conditions and often yield suboptimal or unstable updates. We propose ADOPT, an Adaptive Dependency-aware Prompt Optimization framework for multi-step LLM pipelines. ADOPT explicitly models the dependency between each LLM step and the final task outcome, enabling precise text-gradient estimation analogous to computing analytical derivatives. It decouples textual gradient estimation from gradient updates, reducing multi-prompt optimization to flexible single-prompt optimization steps, and employs a Shapley-based mechanism to adaptively allocate optimization resources. Experiments on real-world datasets and diverse pipeline structures show that ADOPT is effective and robust, consistently outperforming state-of-the-art prompt optimization baselines. 

---
# World model inspired sarcasm reasoning with large language model agents 

**Authors**: Keito Inoshita, Shinnosuke Mizuno  

**Link**: [PDF](https://arxiv.org/pdf/2512.24329)  

**Abstract**: Sarcasm understanding is a challenging problem in natural language processing, as it requires capturing the discrepancy between the surface meaning of an utterance and the speaker's intentions as well as the surrounding social context. Although recent advances in deep learning and Large Language Models (LLMs) have substantially improved performance, most existing approaches still rely on black-box predictions of a single model, making it difficult to structurally explain the cognitive factors underlying sarcasm. Moreover, while sarcasm often emerges as a mismatch between semantic evaluation and normative expectations or intentions, frameworks that explicitly decompose and model these components remain limited. In this work, we reformulate sarcasm understanding as a world model inspired reasoning process and propose World Model inspired SArcasm Reasoning (WM-SAR), which decomposes literal meaning, context, normative expectation, and intention into specialized LLM-based agents. The discrepancy between literal evaluation and normative expectation is explicitly quantified as a deterministic inconsistency score, and together with an intention score, these signals are integrated by a lightweight Logistic Regression model to infer the final sarcasm probability. This design leverages the reasoning capability of LLMs while maintaining an interpretable numerical decision structure. Experiments on representative sarcasm detection benchmarks show that WM-SAR consistently outperforms existing deep learning and LLM-based methods. Ablation studies and case analyses further demonstrate that integrating semantic inconsistency and intention reasoning is essential for effective sarcasm detection, achieving both strong performance and high interpretability. 

---
# Automated Analysis of Sustainability Reports: Using Large Language Models for the Extraction and Prediction of EU Taxonomy-Compliant KPIs 

**Authors**: Jonathan Schmoll, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2512.24289)  

**Abstract**: The manual, resource-intensive process of complying with the EU Taxonomy presents a significant challenge for companies. While Large Language Models (LLMs) offer a path to automation, research is hindered by a lack of public benchmark datasets. To address this gap, we introduce a novel, structured dataset from 190 corporate reports, containing ground-truth economic activities and quantitative Key Performance Indicators (KPIs). We use this dataset to conduct the first systematic evaluation of LLMs on the core compliance workflow. Our results reveal a clear performance gap between qualitative and quantitative tasks. LLMs show moderate success in the qualitative task of identifying economic activities, with a multi-step agentic framework modestly enhancing precision. Conversely, the models comprehensively fail at the quantitative task of predicting financial KPIs in a zero-shot setting. We also discover a paradox, where concise metadata often yields superior performance to full, unstructured reports, and find that model confidence scores are poorly calibrated. We conclude that while LLMs are not ready for full automation, they can serve as powerful assistive tools for human experts. Our dataset provides a public benchmark for future research. 

---
# MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring 

**Authors**: Qipeng Wang, Rui Sheng, Yafei Li, Huamin Qu, Yushi Sun, Min Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.24181)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated significant promise in clinical diagnosis. However, current models struggle to emulate the iterative, diagnostic hypothesis-driven reasoning of real clinical scenarios. Specifically, current LLMs suffer from three critical limitations: (1) generating hallucinated medical content due to weak grounding in verified knowledge, (2) asking redundant or inefficient questions rather than discriminative ones that hinder diagnostic progress, and (3) losing coherence over multi-turn dialogues, leading to contradictory or inconsistent conclusions. To address these challenges, we propose MedKGI, a diagnostic framework grounded in clinical practices. MedKGI integrates a medical knowledge graph (KG) to constrain reasoning to validated medical ontologies, selects questions based on information gain to maximize diagnostic efficiency, and adopts an OSCE-format structured state to maintain consistent evidence tracking across turns. Experiments on clinical benchmarks show that MedKGI outperforms strong LLM baselines in both diagnostic accuracy and inquiry efficiency, improving dialogue efficiency by 30% on average while maintaining state-of-the-art accuracy. 

---
# QianfanHuijin Technical Report: A Novel Multi-Stage Training Paradigm for Finance Industrial LLMs 

**Authors**: Shupeng Li, Weipeng Lu, Linyun Liu, Chen Lin, Shaofei Li, Zhendong Tan, Hanjun Zhong, Yucheng Zeng, Chenghao Zhu, Mengyue Liu, Daxiang Dong, Jianmin Wu, Yunting Xiao, Annan Li, Danyu Liu, Jingnan Zhang, Licen Liu, Dawei Yin, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.24314)  

**Abstract**: Domain-specific enhancement of Large Language Models (LLMs) within the financial context has long been a focal point of industrial application. While previous models such as BloombergGPT and Baichuan-Finance primarily focused on knowledge enhancement, the deepening complexity of financial services has driven a growing demand for models that possess not only domain knowledge but also robust financial reasoning and agentic capabilities. In this paper, we present QianfanHuijin, a financial domain LLM, and propose a generalizable multi-stage training paradigm for industrial model enhancement.
Our approach begins with Continual Pre-training (CPT) on financial corpora to consolidate the knowledge base. This is followed by a fine-grained Post-training pipeline designed with increasing specificity: starting with Financial SFT, progressing to Finance Reasoning RL and Finance Agentic RL, and culminating in General RL aligned with real-world business scenarios. Empirical results demonstrate that QianfanHuijin achieves superior performance across various authoritative financial benchmarks. Furthermore, ablation studies confirm that the targeted Reasoning RL and Agentic RL stages yield significant gains in their respective capabilities. These findings validate our motivation and suggest that this fine-grained, progressive post-training methodology is poised to become a mainstream paradigm for various industrial-enhanced LLMs. 

---
# iCLP: Large Language Model Reasoning with Implicit Cognition Latent Planning 

**Authors**: Sijia Chen, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.24014)  

**Abstract**: Large language models (LLMs), when guided by explicit textual plans, can perform reliable step-by-step reasoning during problem-solving. However, generating accurate and effective textual plans remains challenging due to LLM hallucinations and the high diversity of task-specific questions. To address this, we draw inspiration from human Implicit Cognition (IC), the subconscious process by which decisions are guided by compact, generalized patterns learned from past experiences without requiring explicit verbalization. We propose iCLP, a novel framework that enables LLMs to adaptively generate latent plans (LPs), which are compact encodings of effective reasoning instructions. iCLP first distills explicit plans from existing step-by-step reasoning trajectories. It then learns discrete representations of these plans via a vector-quantized autoencoder coupled with a codebook. Finally, by fine-tuning LLMs on paired latent plans and corresponding reasoning steps, the models learn to perform implicit planning during reasoning. Experimental results on mathematical reasoning and code generation tasks demonstrate that, with iCLP, LLMs can plan in latent space while reasoning in language space. This approach yields significant improvements in both accuracy and efficiency and, crucially, demonstrates strong cross-domain generalization while preserving the interpretability of chain-of-thought reasoning. 

---
# Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models 

**Authors**: Rohit Kumar Salla, Manoj Saravanan, Shrikar Reddy Kota  

**Link**: [PDF](https://arxiv.org/pdf/2512.24058)  

**Abstract**: Large Language Models (LLMs) like LLaMA, Mistral, and Gemma are increasingly used in decision-critical domains such as healthcare, law, and finance, yet their reliability remains uncertain. They often make overconfident errors, degrade under input shifts, and lack clear uncertainty estimates. Existing evaluations are fragmented, addressing only isolated aspects. We introduce the Composite Reliability Score (CRS), a unified framework that integrates calibration, robustness, and uncertainty quantification into a single interpretable metric. Through experiments on ten leading open-source LLMs across five QA datasets, we assess performance under baselines, perturbations, and calibration methods. CRS delivers stable model rankings, uncovers hidden failure modes missed by single metrics, and highlights that the most dependable systems balance accuracy, robustness, and calibrated uncertainty. 

---
# Large Emotional World Model 

**Authors**: Changhao Song, Yazhou Zhang, Hui Gao, Chang Yang, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24149)  

**Abstract**: World Models serve as tools for understanding the current state of the world and predicting its future dynamics, with broad application potential across numerous fields. As a key component of world knowledge, emotion significantly influences human decision-making. While existing Large Language Models (LLMs) have shown preliminary capability in capturing world knowledge, they primarily focus on modeling physical-world regularities and lack systematic exploration of emotional factors. In this paper, we first demonstrate the importance of emotion in understanding the world by showing that removing emotionally relevant information degrades reasoning performance. Inspired by theory of mind, we further propose a Large Emotional World Model (LEWM). Specifically, we construct the Emotion-Why-How (EWH) dataset, which integrates emotion into causal relationships and enables reasoning about why actions occur and how emotions drive future world states. Based on this dataset, LEWM explicitly models emotional states alongside visual observations and actions, allowing the world model to predict both future states and emotional transitions. Experimental results show that LEWM more accurately predicts emotion-driven social behaviors while maintaining comparable performance to general world models on basic tasks. 

---
# Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process 

**Authors**: Zhenyu Zhang, Shujian Zhang, John Lambert, Wenxuan Zhou, Zhangyang Wang, Mingqing Chen, Andrew Hard, Rajiv Mathews, Lun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23988)  

**Abstract**: Despite the growing reasoning capabilities of recent large language models (LLMs), their internal mechanisms during the reasoning process remain underexplored. Prior approaches often rely on human-defined concepts (e.g., overthinking, reflection) at the word level to analyze reasoning in a supervised manner. However, such methods are limited, as it is infeasible to capture the full spectrum of potential reasoning behaviors, many of which are difficult to define in token space. In this work, we propose an unsupervised framework (namely, RISE: Reasoning behavior Interpretability via Sparse auto-Encoder) for discovering reasoning vectors, which we define as directions in the activation space that encode distinct reasoning behaviors. By segmenting chain-of-thought traces into sentence-level 'steps' and training sparse auto-encoders (SAEs) on step-level activations, we uncover disentangled features corresponding to interpretable behaviors such as reflection and backtracking. Visualization and clustering analyses show that these behaviors occupy separable regions in the decoder column space. Moreover, targeted interventions on SAE-derived vectors can controllably amplify or suppress specific reasoning behaviors, altering inference trajectories without retraining. Beyond behavior-specific disentanglement, SAEs capture structural properties such as response length, revealing clusters of long versus short reasoning traces. More interestingly, SAEs enable the discovery of novel behaviors beyond human supervision. We demonstrate the ability to control response confidence by identifying confidence-related vectors in the SAE decoder space. These findings underscore the potential of unsupervised latent discovery for both interpreting and controllably steering reasoning in LLMs. 

---
# Integrating Domain Knowledge for Financial QA: A Multi-Retriever RAG Approach with LLMs 

**Authors**: Yukun Zhang, Stefan Elbl Droguett, Samyak Jain  

**Link**: [PDF](https://arxiv.org/pdf/2512.23848)  

**Abstract**: This research project addresses the errors of financial numerical reasoning Question Answering (QA) tasks due to the lack of domain knowledge in finance. Despite recent advances in Large Language Models (LLMs), financial numerical questions remain challenging because they require specific domain knowledge in finance and complex multi-step numeric reasoning. We implement a multi-retriever Retrieval Augmented Generators (RAG) system to retrieve both external domain knowledge and internal question contexts, and utilize the latest LLM to tackle these tasks. Through comprehensive ablation experiments and error analysis, we find that domain-specific training with the SecBERT encoder significantly contributes to our best neural symbolic model surpassing the FinQA paper's top model, which serves as our baseline. This suggests the potential superior performance of domain-specific training. Furthermore, our best prompt-based LLM generator achieves the state-of-the-art (SOTA) performance with significant improvement (>7%), yet it is still below the human expert performance. This study highlights the trade-off between hallucinations loss and external knowledge gains in smaller models and few-shot examples. For larger models, the gains from external facts typically outweigh the hallucination loss. Finally, our findings confirm the enhanced numerical reasoning capabilities of the latest LLM, optimized for few-shot learning. 

---
# Entropy-Aware Speculative Decoding Toward Improved LLM Reasoning 

**Authors**: Tiancheng Su, Meicong Zhang, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2512.23765)  

**Abstract**: Speculative decoding (SD) accelerates large language model (LLM) reasoning by using a small draft model to generate candidate tokens, which the target LLM either accepts directly or regenerates upon rejection. However, excessive alignment between the draft and target models constrains SD to the performance of the target LLM. To address this limitation, we propose Entropy-Aware Speculative Decoding (EASD), a training-free enhancement. Building on standard SD, EASD incorporates a dynamic entropy-based penalty. At each decoding step, we employ the entropy of the sampling distribution to quantify model uncertainty. When both models exhibit high entropy with substantial overlap among their top-N predictions, the corresponding token is rejected and re-sampled by the target LLM. This penalty prevents low-confidence errors from propagating. By incorporating draft-model verification, EASD enables the possibility of surpassing the target model's inherent performance. Experiments across multiple reasoning benchmarks demonstrate that EASD consistently outperforms existing SD methods and, in most cases, surpasses the target LLM itself. We further prove that the efficiency of EASD is comparable to that of SD. The code can be found in the Supplementary Materials. 

---
# Adversarial Lens: Exploiting Attention Layers to Generate Adversarial Examples for Evaluation 

**Authors**: Kaustubh Dhole  

**Link**: [PDF](https://arxiv.org/pdf/2512.23837)  

**Abstract**: Recent advances in mechanistic interpretability suggest that intermediate attention layers encode token-level hypotheses that are iteratively refined toward the final output. In this work, we exploit this property to generate adversarial examples directly from attention-layer token distributions. Unlike prompt-based or gradient-based attacks, our approach leverages model-internal token predictions, producing perturbations that are both plausible and internally consistent with the model's own generation process. We evaluate whether tokens extracted from intermediate layers can serve as effective adversarial perturbations for downstream evaluation tasks. We conduct experiments on argument quality assessment using the ArgQuality dataset, with LLaMA-3.1-Instruct-8B serving as both the generator and evaluator. Our results show that attention-based adversarial examples lead to measurable drops in evaluation performance while remaining semantically similar to the original inputs. However, we also observe that substitutions drawn from certain layers and token positions can introduce grammatical degradation, limiting their practical effectiveness. Overall, our findings highlight both the promise and current limitations of using intermediate-layer representations as a principled source of adversarial examples for stress-testing LLM-based evaluation pipelines. 

---
# Emergent World Beliefs: Exploring Transformers in Stochastic Games 

**Authors**: Adam Kamel, Tanish Rastogi, Michael Ma, Kailash Ranganathan, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23722)  

**Abstract**: Transformer-based large language models (LLMs) have demonstrated strong reasoning abilities across diverse fields, from solving programming challenges to competing in strategy-intensive games such as chess. Prior work has shown that LLMs can develop emergent world models in games of perfect information, where internal representations correspond to latent states of the environment. In this paper, we extend this line of investigation to domains of incomplete information, focusing on poker as a canonical partially observable Markov decision process (POMDP). We pretrain a GPT-style model on Poker Hand History (PHH) data and probe its internal activations. Our results demonstrate that the model learns both deterministic structure, such as hand ranks, and stochastic features, such as equity, without explicit instruction. Furthermore, by using primarily nonlinear probes, we demonstrated that these representations are decodeable and correlate with theoretical belief states, suggesting that LLMs are learning their own representation of the stochastic environment of Texas Hold'em Poker. 

---
# HarmTransform: Transforming Explicit Harmful Queries into Stealthy via Multi-Agent Debate 

**Authors**: Shenzhe Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23717)  

**Abstract**: Large language models (LLMs) are equipped with safety mechanisms to detect and block harmful queries, yet current alignment approaches primarily focus on overtly dangerous content and overlook more subtle threats. However, users can often disguise harmful intent through covert rephrasing that preserves malicious objectives while appearing benign, which creates a significant gap in existing safety training data. To address this limitation, we introduce HarmTransform, a multi-agent debate framework for systematically transforming harmful queries into stealthier forms while preserving their underlying harmful intent. Our framework leverages iterative critique and refinement among multiple agents to generate high-quality, covert harmful query transformations that can be used to improve future LLM safety alignment. Experiments demonstrate that HarmTransform significantly outperforms standard baselines in producing effective query transformations. At the same time, our analysis reveals that debate acts as a double-edged sword: while it can sharpen transformations and improve stealth, it may also introduce topic shifts and unnecessary complexity. These insights highlight both the promise and the limitations of multi-agent debate for generating comprehensive safety training data. 

---
# Noise-Driven Persona Formation in Reflexive Neural Language Generation 

**Authors**: Toshiyuki Shigemura  

**Link**: [PDF](https://arxiv.org/pdf/2512.23716)  

**Abstract**: This paper introduces the Luca-Noise Reflex Protocol (LN-RP), a computational framework for analyzing noise-driven persona emergence in large language models. By injecting stochastic noise seeds into the initial generation state, we observe nonlinear transitions in linguistic behavior across 152 generation cycles. Our results reveal three stable persona modes with distinct entropy signatures, and demonstrate that external noise sources can reliably induce phase transitions in reflexive generation dynamics. Quantitative evaluation confirms consistent persona retention and significant differences across modes (p < 0.01). The protocol provides a reproducible method for studying reflexive generation, emergent behavior, and longrange linguistic coherence in LLMs. 

---
# CAT: A Metric-Driven Framework for Analyzing the Consistency-Accuracy Relation of LLMs under Controlled Input Variations 

**Authors**: Paulo Cavalin, Cassia Sanctos, Marcelo Grave, Claudio Pinhanez, Yago Primerano  

**Link**: [PDF](https://arxiv.org/pdf/2512.23711)  

**Abstract**: We introduce \textsc{CAT}, a framework designed to evaluate and visualize the \emph{interplay} of \emph{accuracy} and \emph{response consistency} of Large Language Models (LLMs) under controllable input variations, using multiple-choice (MC) benchmarks as a case study. Current evaluation practices primarily focus on model capabilities such as accuracy or benchmark scores and, more recently, measuring consistency is being considered an essential property for deploying LLMs in high-stake, real-world applications. We argue in this paper that although both dimensions should still be evaluated independently, their inter-dependency also need to be considered for a more nuanced evaluation of LLMs. At the core of \textsc{CAT} are the \emph{Consistency-Accuracy Relation (CAR)} curves, which visualize how model accuracy varies with increasing consistency requirements, as defined by the \emph{Minimum-Consistency Accuracy (MCA)} metric. We further propose the \emph{Consistency-Oriented Robustness Estimate (CORE)} index, a global metric that combines the area and shape of the CAR curve to quantify the trade-off between accuracy and consistency. We present a practical demonstration of our framework across a diverse set of generalist and domain-specific LLMs, evaluated on multiple MC benchmarks. We also outline how \textsc{CAT} can be extended beyond MC tasks to support long-form, open-ended evaluations through adaptable scoring functions. 

---
# PyBangla at BLP-2025 Task 2: Enhancing Bangla-to-Python Code Generation with Iterative Self-Correction and Multilingual Agents 

**Authors**: Jahidul Islam, Md Ataullha, Saiful Azad  

**Link**: [PDF](https://arxiv.org/pdf/2512.23713)  

**Abstract**: LLMs excel at code generation from English prompts, but this progress has not extended to low-resource languages. We address Bangla-to-Python code generation by introducing BanglaCodeAct, an agent-based framework that leverages multi-agent prompting and iterative self-correction. Unlike prior approaches relying on task-specific fine-tuning, BanglaCodeAct employs an open-source multilingual LLM within a Thought-Code-Observation loop, enabling dynamic generation, testing, and refinement of code from Bangla instructions. We benchmark several small-parameter open-source LLMs and evaluate their effectiveness on the mHumanEval dataset for Bangla NL2Code. Our results show that Qwen3-8B, when deployed with BanglaCodeAct, achieves the best performance, with pass@1 accuracy of 94.0\% on the development set and 71.6\% on the blind test set. These results establish a new benchmark for Bangla-to-Python translation and highlight the potential of agent-based reasoning for reliable code generation in low-resource languages. Experimental scripts are publicly available at this http URL. 

---
# STED and Consistency Scoring: A Framework for Evaluating LLM Structured Output Reliability 

**Authors**: Guanghui Wang, Jinze Yu, Xing Zhang, Dayuan Jiang, Yin Song, Tomal Deb, Xuefeng Liu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2512.23712)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for structured data generation, yet output consistency remains critical for production applications. We introduce a comprehensive framework for evaluating and improving consistency in LLM-generated structured outputs. Our approach combines: (1) STED (Semantic Tree Edit Distance), a novel similarity metric balancing semantic flexibility with structural strictness when comparing JSON outputs, and (2) a consistency scoring framework aggregating multiple STED measurements across repeated generations to quantify reliability. Through systematic experiments on synthetic datasets with controlled schema, expression, and semantic variations, we demonstrate STED achieves superior performance ($0.86-0.90$ similarity for semantic equivalents, $0.0$ for structural breaks) compared to existing metrics including TED, BERTScore, and DeepDiff. Applying our framework to benchmark six LLMs reveals significant variations: Claude-3.7-Sonnet demonstrates exceptional consistency, maintaining near-perfect structural reliability even at high temperatures ($T=0.9$), while models like Claude-3-Haiku and Nova-Pro exhibit substantial degradation requiring careful tuning. Our framework enables practical applications including targeted model selection for structured tasks, iterative prompt refinement for reproducible results, and diagnostic analysis to identify inconsistency root causes. This work provides theoretical foundations and practical tools for ensuring reliable structured output generation in LLM-based production systems. 

---
# Scaling Open-Ended Reasoning to Predict the Future 

**Authors**: Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2512.25070)  

**Abstract**: High-stakes decision making involves reasoning under uncertainty about the future. In this work, we train language models to make predictions on open-ended forecasting questions. To scale up training data, we synthesize novel forecasting questions from global events reported in daily news, using a fully automated, careful curation recipe. We train the Qwen3 thinking models on our dataset, OpenForesight. To prevent leakage of future information during training and evaluation, we use an offline news corpus, both for data generation and retrieval in our forecasting system. Guided by a small validation set, we show the benefits of retrieval, and an improved reward function for reinforcement learning (RL). Once we obtain our final forecasting system, we perform held-out testing between May to August 2025. Our specialized model, OpenForecaster 8B, matches much larger proprietary models, with our training improving the accuracy, calibration, and consistency of predictions. We find calibration improvements from forecasting training generalize across popular benchmarks. We open-source all our models, code, and data to make research on language model forecasting broadly accessible. 

---
# CPJ: Explainable Agricultural Pest Diagnosis via Caption-Prompt-Judge with LLM-Judged Refinement 

**Authors**: Wentao Zhang, Tao Fang, Lina Lu, Lifei Wang, Weihe Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.24947)  

**Abstract**: Accurate and interpretable crop disease diagnosis is essential for agricultural decision-making, yet existing methods often rely on costly supervised fine-tuning and perform poorly under domain shifts. We propose Caption--Prompt--Judge (CPJ), a training-free few-shot framework that enhances Agri-Pest VQA through structured, interpretable image captions. CPJ employs large vision-language models to generate multi-angle captions, refined iteratively via an LLM-as-Judge module, which then inform a dual-answer VQA process for both recognition and management responses. Evaluated on CDDMBench, CPJ significantly improves performance: using GPT-5-mini captions, GPT-5-Nano achieves \textbf{+22.7} pp in disease classification and \textbf{+19.5} points in QA score over no-caption baselines. The framework provides transparent, evidence-based reasoning, advancing robust and explainable agricultural diagnosis without fine-tuning. Our code and data are publicly available at: this https URL. 

---
# Large language models and the entropy of English 

**Authors**: Colin Scheibner, Lindsay M. Smith, William Bialek  

**Link**: [PDF](https://arxiv.org/pdf/2512.24969)  

**Abstract**: We use large language models (LLMs) to uncover long-ranged structure in English texts from a variety of sources. The conditional entropy or code length in many cases continues to decrease with context length at least to $N\sim 10^4$ characters, implying that there are direct dependencies or interactions across these distances. A corollary is that there are small but significant correlations between characters at these separations, as we show from the data independent of models. The distribution of code lengths reveals an emergent certainty about an increasing fraction of characters at large $N$. Over the course of model training, we observe different dynamics at long and short context lengths, suggesting that long-ranged structure is learned only gradually. Our results constrain efforts to build statistical physics models of LLMs or language itself. 

---
# Iterative Deployment Improves Planning Skills in LLMs 

**Authors**: Augusto B. Corrêa, Yoav Gelberg, Luckeciano C. Melo, Ilia Shumailov, André G. Pereira, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2512.24940)  

**Abstract**: We show that iterative deployment of large language models (LLMs), each fine-tuned on data carefully curated by users from the previous models' deployment, can significantly change the properties of the resultant models. By testing this mechanism on various planning domains, we observe substantial improvements in planning skills, with later models displaying emergent generalization by discovering much longer plans than the initial models. We then provide theoretical analysis showing that iterative deployment effectively implements reinforcement learning (RL) training in the outer-loop (i.e. not as part of intentional model training), with an implicit reward function. The connection to RL has two important implications: first, for the field of AI safety, as the reward function entailed by repeated deployment is not defined explicitly, and could have unexpected implications to the properties of future model deployments. Second, the mechanism highlighted here can be viewed as an alternative training regime to explicit RL, relying on data curation rather than explicit rewards. 

---
# Vibe Coding, Interface Flattening 

**Authors**: Hongrui Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.24939)  

**Abstract**: Large language models are reshaping programming by enabling 'vibe coding': the development of softwares through natural-language interaction with model-driven toolchains. This article argues that vibe coding is best understood as interface flattening, a reconfiguration in which previously distinct modalities (GUI, CLI, and API) appear to converge into a single conversational surface, even as the underlying chain of translation from intention to machinic effect lengthens and thickens. Drawing on Friedrich Kittler's materialist media theory and Alexander Galloway's account of interfaces as sites of protocol control, the paper situates programming as a historically localised interface arrangement rather than an essential relation to computation. Through a materialist reconstruction of the contemporary vibe-coding stack, it shows how remote compute infrastructures, latency and connectivity, structured outputs, function/tool calling, and interoperability standards such as the Model Context Protocol relocate control and meaning-making power to model and protocol providers. The apparent democratisation of technical capability therefore depends on new dependencies and new literacies. By foregrounding the tension between experiential flattening and infrastructural thickening, I demonstrate how LLM-mediated development redistributes symbolic labour/power, obscures responsibility, and privatises competencies previously dispersed across programming communities, contributing a critical lens on the political economy of AI-mediated human-computer interaction. 

---
# Recursive Language Models 

**Authors**: Alex L. Zhang, Tim Kraska, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2512.24601)  

**Abstract**: We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. We find that RLMs successfully handle inputs up to two orders of magnitude beyond model context windows and, even for shorter prompts, dramatically outperform the quality of base LLMs and common long-context scaffolds across four diverse long-context tasks, while having comparable (or cheaper) cost per query. 

---
# More Than Bits: Multi-Envelope Double Binary Factorization for Extreme Quantization 

**Authors**: Yuma Ichikawa, Yoshihiko Fujisawa, Yudai Fujimoto, Akira Sakai, Katsuki Fujisawa  

**Link**: [PDF](https://arxiv.org/pdf/2512.24545)  

**Abstract**: For extreme low-bit quantization of large language models (LLMs), Double Binary Factorization (DBF) is attractive as it enables efficient inference without sacrificing accuracy. However, the scaling parameters of DBF are too restrictive; after factoring out signs, all rank components share the same magnitude profile, resulting in performance saturation. We propose Multi-envelope DBF (MDBF), which retains a shared pair of 1-bit sign bases but replaces the single envelope with a rank-$l$ envelope. By sharing sign matrices among envelope components, MDBF effectively maintains a binary carrier and utilizes the limited memory budget for magnitude expressiveness. We also introduce a closed-form initialization and an alternating refinement method to optimize MDBF. Across the LLaMA and Qwen families, MDBF enhances perplexity and zero-shot accuracy over previous binary formats at matched bits per weight while preserving the same deployment-friendly inference primitive. 

---
# From Building Blocks to Planning: Multi-Step Spatial Reasoning in LLMs with Reinforcement Learning 

**Authors**: Amir Tahmasbi, Sadegh Majidi, Kazem Taram, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2512.24532)  

**Abstract**: Spatial reasoning in large language models (LLMs) has gained increasing attention due to applications in navigation and planning. Despite strong general language capabilities, LLMs still struggle with spatial transformations and multi-step planning in structured environments. We propose a two-stage approach that decomposes spatial reasoning into atomic building blocks and their composition. First, we apply supervised fine-tuning on elementary spatial transformations, such as rotation, translation, and scaling, to equip the model with basic spatial physics. We then freeze this physics-aware model and train lightweight LoRA adapters within the GRPO framework to learn policies that compose these building blocks for multi-step planning in puzzle-based environments, in a closed-loop manner. To support this pipeline, we synthesize an ASCII-art dataset and construct a corresponding ASCII-based reinforcement learning environment. Our method consistently outperforms baselines, including the generic backbone, physics-aware model, and end-to-end RL models, under both Dynamic environments with explicit state updates and Static environments where the model must rely on its internal state across steps. In addition, the proposed approach converges faster and exhibits more stable training compared to end-to-end reinforcement learning from scratch. Finally, we analyze attention patterns to assess whether fine-tuning induces meaningful improvements in spatial understanding. 

---
# AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives 

**Authors**: Yanxi Chen, Wenhui Zhu, Xiwen Chen, Zhipeng Wang, Xin Li, Peijie Qiu, Hao Wang, Xuanzhao Dong, Yujian Xiong, Anderson Schneider, Yuriy Nevmyvaka, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24052)  

**Abstract**: Although Large Audio-Language Models (LALMs) deliver state-of-the-art (SOTA) performance, they frequently suffer from hallucinations, e.g. generating text not grounded in the audio input. We analyze these grounding failures and identify a distinct taxonomy: Event Omission, False Event Identity, Temporal Relation Error, and Quantitative Temporal Error. To address this, we introduce the AHA (Audio Hallucination Alignment) framework. By leveraging counterfactual hard negative mining, our pipeline constructs a high-quality preference dataset that forces models to distinguish strict acoustic evidence from linguistically plausible fabrications. Additionally, we establish AHA-Eval, a diagnostic benchmark designed to rigorously test these fine-grained temporal reasoning capabilities. We apply this data to align Qwen2.5-Omni. The resulting model, Qwen-Audio-AHA, achieves a 13.7% improvement on AHA-Eval. Crucially, this benefit generalizes beyond our diagnostic set. Our model shows substantial gains on public benchmarks, including 1.3% on MMAU-Test and 1.6% on MMAR, outperforming latest SOTA methods. 

---
# Many Minds from One Model: Bayesian Transformers for Population Intelligence 

**Authors**: Diji Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.25063)  

**Abstract**: Despite their scale and success, modern transformers are almost universally trained as single-minded systems: optimization produces one deterministic set of parameters, representing a single functional hypothesis about the data. Motivated by the idea that intelligence emerge from many minds, we propose Population Bayesian Transformers (B-Trans), which transform a standard Large Language Model into a Bayesian Transformer model to supports sampling diverse yet coherent model instances from a single set of pre-trained weights.
B-Trans introduces a Bayesian-motivated posterior proxy by treating the bias-like offsets in normalization layers as stochastic variables with a Gaussian variational approximation, inducing a distribution over model behavior without the cost of training full Bayesian neural networks. Sampling from this proxy yields a set of model instances with diverse behaviors while maintaining general competence. To preserve coherence within each generation, we freeze the sampled noise at the sequence level, enforcing temporal consistency across tokens. B-Trans allows for population-level decision-making, where aggregating predictions across sampled individuals significantly enhances exploration. Experiments across zero-shot generation, Reinforcement Learning with Verifiable Rewards (RLVR), and RL without explicit labels demonstrate that B-Trans effectively leverage the wisdom of crowds, yielding superior semantic diversity while achieving better task performance compared to deterministic baselines. 

---
# Context-aware LLM-based AI Agents for Human-centered Energy Management Systems in Smart Buildings 

**Authors**: Tianzhi He, Farrokh Jazizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2512.25055)  

**Abstract**: This study presents a conceptual framework and a prototype assessment for Large Language Model (LLM)-based Building Energy Management System (BEMS) AI agents to facilitate context-aware energy management in smart buildings through natural language interaction. The proposed framework comprises three modules: perception (sensing), central control (brain), and action (actuation and user interaction), forming a closed feedback loop that captures, analyzes, and interprets energy data to respond intelligently to user queries and manage connected appliances. By leveraging the autonomous data analytics capabilities of LLMs, the BEMS AI agent seeks to offer context-aware insights into energy consumption, cost prediction, and device scheduling, thereby addressing limitations in existing energy management systems. The prototype's performance was evaluated using 120 user queries across four distinct real-world residential energy datasets and different evaluation metrics, including latency, functionality, capability, accuracy, and cost-effectiveness. The generalizability of the framework was demonstrated using ANOVA tests. The results revealed promising performance, measured by response accuracy in device control (86%), memory-related tasks (97%), scheduling and automation (74%), and energy analysis (77%), while more complex cost estimation tasks highlighted areas for improvement with an accuracy of 49%. This benchmarking study moves toward formalizing the assessment of LLM-based BEMS AI agents and identifying future research directions, emphasizing the trade-off between response accuracy and computational efficiency. 

---
# Youtu-Agent: Scaling Agent Productivity with Automated Generation and Hybrid Policy Optimization 

**Authors**: Yuchen Shi, Yuzheng Cai, Siqi Cai, Zihan Xu, Lichao Chen, Yulei Qin, Zhijian Zhou, Xiang Fei, Chaofan Qiu, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, Guocan Cai, Yong Mao, Yunsheng Wu, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.24615)  

**Abstract**: Existing Large Language Model (LLM) agent frameworks face two significant challenges: high configuration costs and static capabilities. Building a high-quality agent often requires extensive manual effort in tool integration and prompt engineering, while deployed agents struggle to adapt to dynamic environments without expensive fine-tuning. To address these issues, we propose \textbf{Youtu-Agent}, a modular framework designed for the automated generation and continuous evolution of LLM agents. Youtu-Agent features a structured configuration system that decouples execution environments, toolkits, and context management, enabling flexible reuse and automated synthesis. We introduce two generation paradigms: a \textbf{Workflow} mode for standard tasks and a \textbf{Meta-Agent} mode for complex, non-standard requirements, capable of automatically generating tool code, prompts, and configurations. Furthermore, Youtu-Agent establishes a hybrid policy optimization system: (1) an \textbf{Agent Practice} module that enables agents to accumulate experience and improve performance through in-context optimization without parameter updates; and (2) an \textbf{Agent RL} module that integrates with distributed training frameworks to enable scalable and stable reinforcement learning of any Youtu-Agents in an end-to-end, large-scale manner. Experiments demonstrate that Youtu-Agent achieves state-of-the-art performance on WebWalkerQA (71.47\%) and GAIA (72.8\%) using open-weight models. Our automated generation pipeline achieves over 81\% tool synthesis success rate, while the Practice module improves performance on AIME 2024/2025 by +2.7\% and +5.4\% respectively. Moreover, our Agent RL training achieves 40\% speedup with steady performance improvement on 7B LLMs, enhancing coding/reasoning and searching capabilities respectively up to 35\% and 21\% on Maths and general/multi-hop QA benchmarks. 

---
# GenZ: Foundational models as latent variable generators within traditional statistical models 

**Authors**: Marko Jojic, Nebojsa Jojic  

**Link**: [PDF](https://arxiv.org/pdf/2512.24834)  

**Abstract**: We present GenZ, a hybrid model that bridges foundational models and statistical modeling through interpretable semantic features. While large language models possess broad domain knowledge, they often fail to capture dataset-specific patterns critical for prediction tasks. Our approach addresses this by discovering semantic feature descriptions through an iterative process that contrasts groups of items identified via statistical modeling errors, rather than relying solely on the foundational model's domain understanding. We formulate this as a generalized EM algorithm that jointly optimizes semantic feature descriptors and statistical model parameters. The method prompts a frozen foundational model to classify items based on discovered features, treating these judgments as noisy observations of latent binary features that predict real-valued targets through learned statistical relationships. We demonstrate the approach on two domains: house price prediction (hedonic regression) and cold-start collaborative filtering for movie recommendations. On house prices, our model achieves 12\% median relative error using discovered semantic features from multimodal listing data, substantially outperforming a GPT-5 baseline (38\% error) that relies on the LLM's general domain knowledge. For Netflix movie embeddings, our model predicts collaborative filtering representations with 0.59 cosine similarity purely from semantic descriptions -- matching the performance that would require approximately 4000 user ratings through traditional collaborative filtering. The discovered features reveal dataset-specific patterns (e.g., architectural details predicting local housing markets, franchise membership predicting user preferences) that diverge from the model's domain knowledge alone. 

---
# BatteryAgent: Synergizing Physics-Informed Interpretation with LLM Reasoning for Intelligent Battery Fault Diagnosis 

**Authors**: Songqi Zhou, Ruixue Liu, Boman Su, Jiazhou Wang, Yixing Wang, Benben Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24686)  

**Abstract**: Fault diagnosis of lithium-ion batteries is critical for system safety. While existing deep learning methods exhibit superior detection accuracy, their "black-box" nature hinders interpretability. Furthermore, restricted by binary classification paradigms, they struggle to provide root cause analysis and maintenance recommendations. To address these limitations, this paper proposes BatteryAgent, a hierarchical framework that integrates physical knowledge features with the reasoning capabilities of Large Language Models (LLMs). The framework comprises three core modules: (1) A Physical Perception Layer that utilizes 10 mechanism-based features derived from electrochemical principles, balancing dimensionality reduction with physical fidelity; (2) A Detection and Attribution Layer that employs Gradient Boosting Decision Trees and SHAP to quantify feature contributions; and (3) A Reasoning and Diagnosis Layer that leverages an LLM as the agent core. This layer constructs a "numerical-semantic" bridge, combining SHAP attributions with a mechanism knowledge base to generate comprehensive reports containing fault types, root cause analysis, and maintenance suggestions. Experimental results demonstrate that BatteryAgent effectively corrects misclassifications on hard boundary samples, achieving an AUROC of 0.986, which significantly outperforms current state-of-the-art methods. Moreover, the framework extends traditional binary detection to multi-type interpretable diagnosis, offering a new paradigm shift from "passive detection" to "intelligent diagnosis" for battery safety management. 

---
# MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use 

**Authors**: Wenrui Liu, Zixiang Liu, Elsie Dai, Wenhan Yu, Lei Yu, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24565)  

**Abstract**: Large Language Models (LLMs) are increasingly serving as autonomous agents, and their utilization of external tools via the Model Context Protocol (MCP) is considered a future trend. Current MCP evaluation sets suffer from issues such as reliance on external MCP services and a lack of difficulty awareness. To address these limitations, we propose MCPAgentBench, a benchmark based on real-world MCP definitions designed to evaluate the tool-use capabilities of agents. We construct a dataset containing authentic tasks and simulated MCP tools. The evaluation employs a dynamic sandbox environment that presents agents with candidate tool lists containing distractors, thereby testing their tool selection and discrimination abilities. Furthermore, we introduce comprehensive metrics to measure both task completion rates and execution efficiency. Experiments conducted on various latest mainstream Large Language Models reveal significant performance differences in handling complex, multi-step tool invocations. All code is open-source at Github. 

---
# Evaluating the Reasoning Abilities of LLMs on Underrepresented Mathematics Competition Problems 

**Authors**: Samuel Golladay, Majid Bani-Yaghoub  

**Link**: [PDF](https://arxiv.org/pdf/2512.24505)  

**Abstract**: Understanding the limitations of Large Language Models, or LLMs, in mathematical reasoning has been the focus of several recent studies. However, the majority of these studies use the same datasets for benchmarking, which limits the generalizability of their findings and may not fully capture the diverse challenges present in mathematical tasks. The purpose of the present study is to analyze the performance of LLMs on underrepresented mathematics competition problems. We prompted three leading LLMs, namely GPT-4o-mini, Gemini-2.0-Flash, and DeepSeek-V3, with the Missouri Collegiate Mathematics Competition problems in the areas of Calculus, Analytic Geometry, and Discrete Mathematics. The LLMs responses were then compared to the known correct solutions in order to determine the accuracy of the LLM for each problem domain. We also analyzed the LLMs reasoning to explore patterns in errors across problem types and models. DeepSeek-V3 has the best performance in all three categories of Calculus, Analytic Geometry, and Discrete Mathematics, both in reasoning and correct final answers. All three LLMs exhibited notably weak performance in Geometry. The majority of errors made by DeepSeek-V3 were attributed to computational and logical mistakes, whereas GPT-4o-mini frequently exhibited logical and approach-related errors. Gemini, on the other hand, tended to struggle with incomplete reasoning and drawing rushed conclusions. In conclusion, evaluating LLMs on underrepresented mathematics competition datasets can provide deeper insights into their distinct error patterns and highlight ongoing challenges in structured reasoning, particularly within the domain of Geometry. 

---
# Reinforcement Learning-Augmented LLM Agents for Collaborative Decision Making and Performance Optimization 

**Authors**: Dong Qiu, Duo Xu, Limengxi Yue  

**Link**: [PDF](https://arxiv.org/pdf/2512.24609)  

**Abstract**: Large Language Models (LLMs) perform well in language tasks but often lack collaborative awareness and struggle to optimize global performance in multi-agent settings. We present a reinforcement learning-augmented LLM agent framework that formulates cooperation as a decentralized partially observable Markov decision process (Dec-POMDP) and adopts centralized training with decentralized execution (CTDE). We introduce Group Relative Policy Optimization (GRPO) to jointly optimize agent policies with access to global signals during training, together with a simplified joint reward that balances task quality, speed, and coordination cost. On collaborative writing and coding benchmarks, our framework delivers a 3x increase in task processing speed over single-agent baselines, 98.7% structural/style consistency in writing, and a 74.6% test pass rate in coding. The approach consistently outperforms strong multi-agent LLM baselines and provides a practical path toward reliable collaboration in complex workflows. 

---
# Group Deliberation Oriented Multi-Agent Conversational Model for Complex Reasoning 

**Authors**: Zheyu Shi, Dong Qiu, Shanlong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.24613)  

**Abstract**: This paper proposes a group deliberation oriented multi-agent conversational model to address the limitations of single large language models in complex reasoning tasks. The model adopts a three-level role division architecture consisting of generation, verification, and integration. An opinion generation agent produces diverse reasoning perspectives, an evidence verification agent retrieves external knowledge and quantifies factual support, and a consistency arbitration agent integrates logically coherent conclusions. A self-game mechanism is introduced to expand multi-path reasoning trajectories, while a retrieval enhancement module dynamically supplements external knowledge. A composite reward function combining factual consistency and logical coherence is designed, and an improved proximal policy optimization strategy is applied for collaborative training. Experimental results show that the proposed model improves multi-hop reasoning accuracy by 16.8 percent on HotpotQA, 14.3 percent on 2WikiMultihopQA, and 19.2 percent on MeetingBank, while improving consistency by 21.5 percent. The model achieves higher reasoning efficiency than mainstream multi-agent approaches, providing an effective and stable solution for complex reasoning tasks. 

---
# SPARK: Search Personalization via Agent-Driven Retrieval and Knowledge-sharing 

**Authors**: Gaurab Chhetri, Subasish Das, Tausif Islam Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2512.24008)  

**Abstract**: Personalized search demands the ability to model users' evolving, multi-dimensional information needs; a challenge for systems constrained by static profiles or monolithic retrieval pipelines. We present SPARK (Search Personalization via Agent-Driven Retrieval and Knowledge-sharing), a framework in which coordinated persona-based large language model (LLM) agents deliver task-specific retrieval and emergent personalization. SPARK formalizes a persona space defined by role, expertise, task context, and domain, and introduces a Persona Coordinator that dynamically interprets incoming queries to activate the most relevant specialized agents. Each agent executes an independent retrieval-augmented generation process, supported by dedicated long- and short-term memory stores and context-aware reasoning modules. Inter-agent collaboration is facilitated through structured communication protocols, including shared memory repositories, iterative debate, and relay-style knowledge transfer. Drawing on principles from cognitive architectures, multi-agent coordination theory, and information retrieval, SPARK models how emergent personalization properties arise from distributed agent behaviors governed by minimal coordination rules. The framework yields testable predictions regarding coordination efficiency, personalization quality, and cognitive load distribution, while incorporating adaptive learning mechanisms for continuous persona refinement. By integrating fine-grained agent specialization with cooperative retrieval, SPARK provides insights for next-generation search systems capable of capturing the complexity, fluidity, and context sensitivity of human information-seeking behavior. 

---
# Align While Search: Belief-Guided Exploratory Inference for World-Grounded Embodied Agents 

**Authors**: Seohui Bae, Jeonghye Kim, Youngchul Sung, Woohyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2512.24461)  

**Abstract**: In this paper, we propose a test-time adaptive agent that performs exploratory inference through posterior-guided belief refinement without relying on gradient-based updates or additional training for LLM agent operating under partial observability. Our agent maintains an external structured belief over the environment state, iteratively updates it via action-conditioned observations, and selects actions by maximizing predicted information gain over the belief space. We estimate information gain using a lightweight LLM-based surrogate and assess world alignment through a novel reward that quantifies the consistency between posterior belief and ground-truth environment configuration. Experiments show that our method outperforms inference-time scaling baselines such as prompt-augmented or retrieval-enhanced LLMs, in aligning with latent world states with significantly lower integration overhead. 

---
# LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm 

**Authors**: Chunhui Wan, Xunan Dai, Zhuo Wang, Minglei Li, Yanpeng Wang, Yinan Mao, Yu Lan, Zhiwen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.24077)  

**Abstract**: The transition from static Large Language Models (LLMs) to self-improving agents is hindered by the lack of structured reasoning in traditional evolutionary approaches. Existing methods often struggle with premature convergence and inefficient exploration in high-dimensional code spaces. To address these challenges, we introduce LoongFlow, a self-evolving agent framework that achieves state-of-the-art solution quality with significantly reduced computational costs. Unlike "blind" mutation operators, LoongFlow integrates LLMs into a cognitive "Plan-Execute-Summarize" (PES) paradigm, effectively mapping the evolutionary search to a reasoning-heavy process. To sustain long-term architectural coherence, we incorporate a hybrid evolutionary memory system. By synergizing Multi-Island models with MAP-Elites and adaptive Boltzmann selection, this system theoretically balances the exploration-exploitation trade-off, maintaining diverse behavioral niches to prevent optimization stagnation. We instantiate LoongFlow with a General Agent for algorithmic discovery and an ML Agent for pipeline optimization. Extensive evaluations on the AlphaEvolve benchmark and Kaggle competitions demonstrate that LoongFlow outperforms leading baselines (e.g., OpenEvolve, ShinkaEvolve) by up to 60% in evolutionary efficiency while discovering superior solutions. LoongFlow marks a substantial step forward in autonomous scientific discovery, enabling the generation of expert-level solutions with reduced computational overhead. 

---
# ROAD: Reflective Optimization via Automated Debugging for Zero-Shot Agent Alignment 

**Authors**: Natchaya Temyingyong, Daman Jain, Neeraj Kumarsahu, Prabhat Kumar, Rachata Phondi, Wachiravit Modecrua, Krittanon Kaewtawee, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2512.24040)  

**Abstract**: Automatic Prompt Optimization (APO) has emerged as a critical technique for enhancing Large Language Model (LLM) performance, yet current state-of-the-art methods typically rely on large, labeled gold-standard development sets to compute fitness scores for evolutionary or Reinforcement Learning (RL) approaches. In real-world software engineering, however, such curated datasets are rarely available during the initial cold start of agent development, where engineers instead face messy production logs and evolving failure modes. We present ROAD (Reflective Optimization via Automated Debugging), a novel framework that bypasses the need for refined datasets by treating optimization as a dynamic debugging investigation rather than a stochastic search. Unlike traditional mutation strategies, ROAD utilizes a specialized multi-agent architecture, comprising an Analyzer for root-cause analysis, an Optimizer for pattern aggregation, and a Coach for strategy integration, to convert unstructured failure logs into robust, structured Decision Tree Protocols. We evaluated ROAD across both a standardized academic benchmark and a live production Knowledge Management engine. Experimental results demonstrate that ROAD is highly sample-efficient, achieving a 5.6 percent increase in success rate (73.6 percent to 79.2 percent) and a 3.8 percent increase in search accuracy within just three automated iterations. Furthermore, on complex reasoning tasks in the retail domain, ROAD improved agent performance by approximately 19 percent relative to the baseline. These findings suggest that mimicking the human engineering loop of failure analysis and patching offers a viable, data-efficient alternative to resource-intensive RL training for deploying reliable LLM agents. 

---
# CASCADE: Cumulative Agentic Skill Creation through Autonomous Development and Evolution 

**Authors**: Xu Huang, Junwu Chen, Yuxing Fei, Zhuohan Li, Philippe Schwaller, Gerbrand Ceder  

**Link**: [PDF](https://arxiv.org/pdf/2512.23880)  

**Abstract**: Large language model (LLM) agents currently depend on predefined tools or brittle tool generation, constraining their capability and adaptability to complex scientific tasks. We introduce CASCADE, a self-evolving agentic framework representing an early instantiation of the transition from "LLM + tool use" to "LLM + skill acquisition". CASCADE enables agents to master complex external tools and codify knowledge through two meta-skills: continuous learning via web search and code extraction, and self-reflection via introspection and knowledge graph exploration, among others. We evaluate CASCADE on SciSkillBench, a benchmark of 116 materials science and chemistry research tasks. CASCADE achieves a 93.3% success rate using GPT-5, compared to 35.4% without evolution mechanisms. We further demonstrate real-world applications in computational analysis, autonomous laboratory experiments, and selective reproduction of published papers. Along with human-agent collaboration and memory consolidation, CASCADE accumulates executable skills that can be shared across agents and scientists, moving toward scalable AI-assisted scientific research. 

---
# A Proof-of-Concept for Explainable Disease Diagnosis Using Large Language Models and Answer Set Programming 

**Authors**: Ioanna Gemou, Evangelos Lamprou  

**Link**: [PDF](https://arxiv.org/pdf/2512.23932)  

**Abstract**: Accurate disease prediction is vital for timely intervention, effective treatment, and reducing medical complications. While symbolic AI has been applied in healthcare, its adoption remains limited due to the effort required for constructing high-quality knowledge bases. This work introduces McCoy, a framework that combines Large Language Models (LLMs) with Answer Set Programming (ASP) to overcome this barrier. McCoy orchestrates an LLM to translate medical literature into ASP code, combines it with patient data, and processes it using an ASP solver to arrive at the final diagnosis. This integration yields a robust, interpretable prediction framework that leverages the strengths of both paradigms. Preliminary results show McCoy has strong performance on small-scale disease diagnosis tasks. 

---
# Vulcan: Instance-Optimal Systems Heuristics Through LLM-Driven Search 

**Authors**: Rohit Dwivedula, Divyanshu Saxena, Sujay Yadalam, Daehyeok Kim, Aditya Akella  

**Link**: [PDF](https://arxiv.org/pdf/2512.25065)  

**Abstract**: Resource-management tasks in modern operating and distributed systems continue to rely primarily on hand-designed heuristics for tasks such as scheduling, caching, or active queue management. Designing performant heuristics is an expensive, time-consuming process that we are forced to continuously go through due to the constant flux of hardware, workloads and environments.
We propose a new alternative: synthesizing instance-optimal heuristics -- specialized for the exact workloads and hardware where they will be deployed -- using code-generating large language models (LLMs). To make this synthesis tractable, Vulcan separates policy and mechanism through LLM-friendly, task-agnostic interfaces. With these interfaces, users specify the inputs and objectives of their desired policy, while Vulcan searches for performant policies via evolutionary search over LLM-generated code. This interface is expressive enough to capture a wide range of system policies, yet sufficiently constrained to allow even small, inexpensive LLMs to generate correct and executable code.
We use Vulcan to synthesize performant heuristics for cache eviction and memory tiering, and find that these heuristics outperform all human-designed state-of-the-art algorithms by upto 69% and 7.9% in performance for each of these tasks respectively. 

---
# AstroReview: An LLM-driven Multi-Agent Framework for Telescope Proposal Peer Review and Refinement 

**Authors**: Yutong Wang, Yunxiang Xiao, Yonglin Tian, Junyong Li, Jing Wang, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2512.24754)  

**Abstract**: Competitive access to modern observatories has intensified as proposal volumes outpace available telescope time, making timely, consistent, and transparent peer review a critical bottleneck for the advancement of astronomy. Automating parts of this process is therefore both scientifically significant and operationally necessary to ensure fair allocation and reproducible decisions at scale. We present AstroReview, an open-source, agent-based framework that automates proposal review in three stages: (i) novelty and scientific merit, (ii) feasibility and expected yield, and (iii) meta-review and reliability verification. Task isolation and explicit reasoning traces curb hallucinations and improve transparency. Without any domain specific fine tuning, AstroReview used in our experiments only for the last stage, correctly identifies genuinely accepted proposals with an accuracy of 87%. The AstroReview in Action module replicates the review and refinement loop; with its integrated Proposal Authoring Agent, the acceptance rate of revised drafts increases by 66% after two iterations, showing that iterative feedback combined with automated meta-review and reliability verification delivers measurable quality gains. Together, these results point to a practical path toward scalable, auditable, and higher throughput proposal review for resource limited facilities. 

---
# Dynamic Large Concept Models: Latent Reasoning in an Adaptive Semantic Space 

**Authors**: Xingwei Qu, Shaowen Wang, Zihao Huang, Kai Hua, Fan Yin, Rui-Jie Zhu, Jundong Zhou, Qiyang Min, Zihao Wang, Yizhi Li, Tianyu Zhang, He Xing, Zheng Zhang, Yuxuan Song, Tianyu Zheng, Zhiyuan Zeng, Chenghua Lin, Ge Zhang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24617)  

**Abstract**: Large Language Models (LLMs) apply uniform computation to all tokens, despite language exhibiting highly non-uniform information density. This token-uniform regime wastes capacity on locally predictable spans while under-allocating computation to semantically critical transitions. We propose $\textbf{Dynamic Large Concept Models (DLCM)}$, a hierarchical language modeling framework that learns semantic boundaries from latent representations and shifts computation from tokens to a compressed concept space where reasoning is more efficient. DLCM discovers variable-length concepts end-to-end without relying on predefined linguistic units. Hierarchical compression fundamentally changes scaling behavior. We introduce the first $\textbf{compression-aware scaling law}$, which disentangles token-level capacity, concept-level reasoning capacity, and compression ratio, enabling principled compute allocation under fixed FLOPs. To stably train this heterogeneous architecture, we further develop a $\textbf{decoupled $\mu$P parametrization}$ that supports zero-shot hyperparameter transfer across widths and compression regimes. At a practical setting ($R=4$, corresponding to an average of four tokens per concept), DLCM reallocates roughly one-third of inference compute into a higher-capacity reasoning backbone, achieving a $\textbf{+2.69$\%$ average improvement}$ across 12 zero-shot benchmarks under matched inference FLOPs. 

---
# Chat-Driven Optimal Management for Virtual Network Services 

**Authors**: Yuya Miyaoka, Masaki Inoue, Kengo Urata, Shigeaki Harada  

**Link**: [PDF](https://arxiv.org/pdf/2512.24614)  

**Abstract**: This paper proposes a chat-driven network management framework that integrates natural language processing (NLP) with optimization-based virtual network allocation, enabling intuitive and reliable reconfiguration of virtual network services. Conventional intent-based networking (IBN) methods depend on statistical language models to interpret user intent but cannot guarantee the feasibility of generated configurations. To overcome this, we develop a two-stage framework consisting of an Interpreter, which extracts intent from natural language prompts using NLP, and an Optimizer, which computes feasible virtual machine (VM) placement and routing via an integer linear programming. In particular, the Interpreter translates user chats into update directions, i.e., whether to increase, decrease, or maintain parameters such as CPU demand and latency bounds, thereby enabling iterative refinement of the network configuration. In this paper, two intent extractors, which are a Sentence-BERT model with support vector machine (SVM) classifiers and a large language model (LLM), are introduced. Experiments in single-user and multi-user settings show that the framework dynamically updates VM placement and routing while preserving feasibility. The LLM-based extractor achieves higher accuracy with fewer labeled samples, whereas the Sentence-BERT with SVM classifiers provides significantly lower latency suitable for real-time operation. These results underscore the effectiveness of combining NLP-driven intent extraction with optimization-based allocation for safe, interpretable, and user-friendly virtual network management. 

---
# Generative AI-enhanced Sector-based Investment Portfolio Construction 

**Authors**: Alina Voronina, Oleksandr Romanko, Ruiwen Cao, Roy H. Kwon, Rafael Mendoza-Arriaga  

**Link**: [PDF](https://arxiv.org/pdf/2512.24526)  

**Abstract**: This paper investigates how Large Language Models (LLMs) from leading providers (OpenAI, Google, Anthropic, DeepSeek, and xAI) can be applied to quantitative sector-based portfolio construction. We use LLMs to identify investable universes of stocks within S&P 500 sector indices and evaluate how their selections perform when combined with classical portfolio optimization methods. Each model was prompted to select and weight 20 stocks per sector, and the resulting portfolios were compared with their respective sector indices across two distinct out-of-sample periods: a stable market phase (January-March 2025) and a volatile phase (April-June 2025).
Our results reveal a strong temporal dependence in LLM portfolio performance. During stable market conditions, LLM-weighted portfolios frequently outperformed sector indices on both cumulative return and risk-adjusted (Sharpe ratio) measures. However, during the volatile period, many LLM portfolios underperformed, suggesting that current models may struggle to adapt to regime shifts or high-volatility environments underrepresented in their training data. Importantly, when LLM-based stock selection is combined with traditional optimization techniques, portfolio outcomes improve in both performance and consistency.
This study contributes one of the first multi-model, cross-provider evaluations of generative AI algorithms in investment management. It highlights that while LLMs can effectively complement quantitative finance by enhancing stock selection and interpretability, their reliability remains market-dependent. The findings underscore the potential of hybrid AI-quantitative frameworks, integrating LLM reasoning with established optimization techniques, to produce more robust and adaptive investment strategies. 

---
# HOLOGRAPH: Active Causal Discovery via Sheaf-Theoretic Alignment of Large Language Model Priors 

**Authors**: Hyunjun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.24478)  

**Abstract**: Causal discovery from observational data remains fundamentally limited by identifiability constraints. Recent work has explored leveraging Large Language Models (LLMs) as sources of prior causal knowledge, but existing approaches rely on heuristic integration that lacks theoretical grounding. We introduce HOLOGRAPH, a framework that formalizes LLM-guided causal discovery through sheaf theory--representing local causal beliefs as sections of a presheaf over variable subsets. Our key insight is that coherent global causal structure corresponds to the existence of a global section, while topological obstructions manifest as non-vanishing sheaf cohomology. We propose the Algebraic Latent Projection to handle hidden confounders and Natural Gradient Descent on the belief manifold for principled optimization. Experiments on synthetic and real-world benchmarks demonstrate that HOLOGRAPH provides rigorous mathematical foundations while achieving competitive performance on causal discovery tasks with 50-100 variables. Our sheaf-theoretic analysis reveals that while Identity, Transitivity, and Gluing axioms are satisfied to numerical precision (<10^{-6}), the Locality axiom fails for larger graphs, suggesting fundamental non-local coupling in latent variable projections. Code is available at [this https URL](this https URL). 

---
# Comparing Approaches to Automatic Summarization in Less-Resourced Languages 

**Authors**: Chester Palen-Michel, Constantine Lignos  

**Link**: [PDF](https://arxiv.org/pdf/2512.24410)  

**Abstract**: Automatic text summarization has achieved high performance in high-resourced languages like English, but comparatively less attention has been given to summarization in less-resourced languages. This work compares a variety of different approaches to summarization from zero-shot prompting of LLMs large and small to fine-tuning smaller models like mT5 with and without three data augmentation approaches and multilingual transfer. We also explore an LLM translation pipeline approach, translating from the source language to English, summarizing and translating back. Evaluating with five different metrics, we find that there is variation across LLMs in their performance across similar parameter sizes, that our multilingual fine-tuned mT5 baseline outperforms most other approaches including zero-shot LLM performance for most metrics, and that LLM as judge may be less reliable on less-resourced languages. 

---
# Taming Hallucinations: Boosting MLLMs' Video Understanding via Counterfactual Video Generation 

**Authors**: Zhe Huang, Hao Wen, Aiming Hao, Bingze Song, Meiqi Wu, Jiahong Wu, Xiangxiang Chu, Sheng Lu, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24271)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made remarkable progress in video understanding. However, they suffer from a critical vulnerability: an over-reliance on language priors, which can lead to visual ungrounded hallucinations, especially when processing counterfactual videos that defy common sense. This limitation, stemming from the intrinsic data imbalance between text and video, is challenging to address due to the substantial cost of collecting and annotating counterfactual data. To address this, we introduce DualityForge, a novel counterfactual data synthesis framework that employs controllable, diffusion-based video editing to transform real-world videos into counterfactual scenarios. By embedding structured contextual information into the video editing and QA generation processes, the framework automatically produces high-quality QA pairs together with original-edited video pairs for contrastive training. Based on this, we build DualityVidQA, a large-scale video dataset designed to reduce MLLM hallucinations. In addition, to fully exploit the contrastive nature of our paired data, we propose Duality-Normalized Advantage Training (DNA-Train), a two-stage SFT-RL training regime where the RL phase applies pair-wise $\ell_1$ advantage normalization, thereby enabling a more stable and efficient policy optimization. Experiments on DualityVidQA-Test demonstrate that our method substantially reduces model hallucinations on counterfactual videos, yielding a relative improvement of 24.0% over the Qwen2.5-VL-7B baseline. Moreover, our approach achieves significant gains across both hallucination and general-purpose benchmarks, indicating strong generalization capability. We will open-source our dataset and code. 

---
# PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression 

**Authors**: Bo Jiang, Taolue Yang, Youyuan Liu, Xubin He, Sheng Di, Sian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.24449)  

**Abstract**: Transformer-based large language models (LLMs) have demonstrated remarkable potential across a wide range of practical applications. However, long-context inference remains a significant challenge due to the substantial memory requirements of the key-value (KV) cache, which can scale to several gigabytes as sequence length and batch size increase. In this paper, we present \textbf{PackKV}, a generic and efficient KV cache management framework optimized for long-context generation. %, which synergistically supports both latency-critical and throughput-critical inference scenarios. PackKV introduces novel lossy compression techniques specifically tailored to the characteristics of KV cache data, featuring a careful co-design of compression algorithms and system architecture. Our approach is compatible with the dynamically growing nature of the KV cache while preserving high computational efficiency. Experimental results show that, under the same and minimum accuracy drop as state-of-the-art quantization methods, PackKV achieves, on average, \textbf{153.2}\% higher memory reduction rate for the K cache and \textbf{179.6}\% for the V cache. Furthermore, PackKV delivers extremely high execution throughput, effectively eliminating decompression overhead and accelerating the matrix-vector multiplication operation. Specifically, PackKV achieves an average throughput improvement of \textbf{75.7}\% for K and \textbf{171.7}\% for V across A100 and RTX Pro 6000 GPUs, compared to cuBLAS matrix-vector multiplication kernels, while demanding less GPU memory bandwidth. Code available on this https URL 

---
# Enhancing LLM-Based Neural Network Generation: Few-Shot Prompting and Efficient Validation for Automated Architecture Design 

**Authors**: Chandini Vysyaraju, Raghuvir Duvvuri, Avi Goyal, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2512.24120)  

**Abstract**: Automated neural network architecture design remains a significant challenge in computer vision. Task diversity and computational constraints require both effective architectures and efficient search methods. Large Language Models (LLMs) present a promising alternative to computationally intensive Neural Architecture Search (NAS), but their application to architecture generation in computer vision has not been systematically studied, particularly regarding prompt engineering and validation strategies. Building on the task-agnostic NNGPT/LEMUR framework, this work introduces and validates two key contributions for computer vision. First, we present Few-Shot Architecture Prompting (FSAP), the first systematic study of the number of supporting examples (n = 1, 2, 3, 4, 5, 6) for LLM-based architecture generation. We find that using n = 3 examples best balances architectural diversity and context focus for vision tasks. Second, we introduce Whitespace-Normalized Hash Validation, a lightweight deduplication method (less than 1 ms) that provides a 100x speedup over AST parsing and prevents redundant training of duplicate computer vision architectures. In large-scale experiments across seven computer vision benchmarks (MNIST, CIFAR-10, CIFAR-100, CelebA, ImageNette, SVHN, Places365), we generated 1,900 unique architectures. We also introduce a dataset-balanced evaluation methodology to address the challenge of comparing architectures across heterogeneous vision tasks. These contributions provide actionable guidelines for LLM-based architecture search in computer vision and establish rigorous evaluation practices, making automated design more accessible to researchers with limited computational resources. 

---
# Enhancing LLM Planning Capabilities through Intrinsic Self-Critique 

**Authors**: Bernd Bohnet, Pierre-Alexandre Kamienny, Hanie Sedghi, Dilan Gorur, Pranjal Awasthi, Aaron Parisi, Kevin Swersky, Rosanne Liu, Azade Nova, Noah Fiedel  

**Link**: [PDF](https://arxiv.org/pdf/2512.24103)  

**Abstract**: We demonstrate an approach for LLMs to critique their \emph{own} answers with the goal of enhancing their performance that leads to significant improvements over established planning benchmarks. Despite the findings of earlier research that has cast doubt on the effectiveness of LLMs leveraging self critique methods, we show significant performance gains on planning datasets in the Blocksworld domain through intrinsic self-critique, without external source such as a verifier. We also demonstrate similar improvements on Logistics and Mini-grid datasets, exceeding strong baseline accuracies. We employ a few-shot learning technique and progressively extend it to a many-shot approach as our base method and demonstrate that it is possible to gain substantial improvement on top of this already competitive approach by employing an iterative process for correction and refinement. We illustrate how self-critique can significantly boost planning performance. Our empirical results present new state-of-the-art on the class of models considered, namely LLM model checkpoints from October 2024. Our primary focus lies on the method itself, demonstrating intrinsic self-improvement capabilities that are applicable regardless of the specific model version, and we believe that applying our method to more complex search techniques and more capable models will lead to even better performance. 

---
# RSAgent: Learning to Reason and Act for Text-Guided Segmentation via Multi-Turn Tool Invocations 

**Authors**: Xingqi He, Yujie Zhang, Shuyong Gao, Wenjie Li, Lingyi Hong, Mingxi Chen, Kaixun Jiang, Jiyuan Fu, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24023)  

**Abstract**: Text-guided object segmentation requires both cross-modal reasoning and pixel grounding abilities. Most recent methods treat text-guided segmentation as one-shot grounding, where the model predicts pixel prompts in a single forward pass to drive an external segmentor, which limits verification, refocusing and refinement when initial localization is wrong. To address this limitation, we propose RSAgent, an agentic Multimodal Large Language Model (MLLM) which interleaves reasoning and action for segmentation via multi-turn tool invocations. RSAgent queries a segmentation toolbox, observes visual feedback, and revises its spatial hypothesis using historical observations to re-localize targets and iteratively refine masks. We further build a data pipeline to synthesize multi-turn reasoning segmentation trajectories, and train RSAgent with a two-stage framework: cold-start supervised fine-tuning followed by agentic reinforcement learning with fine-grained, task-specific rewards. Extensive experiments show that RSAgent achieves a zero-shot performance of 66.5% gIoU on ReasonSeg test, improving over Seg-Zero-7B by 9%, and reaches 81.5% cIoU on RefCOCOg, demonstrating state-of-the-art performance on both in-domain and out-of-domain benchmarks. 

---
# Coding With AI: From a Reflection on Industrial Practices to Future Computer Science and Software Engineering Education 

**Authors**: Hung-Fu Chang, MohammadShokrolah Shirazi, Lizhou Cao, Supannika Koolmanojwong Mobasser  

**Link**: [PDF](https://arxiv.org/pdf/2512.23982)  

**Abstract**: Recent advances in large language models (LLMs) have introduced new paradigms in software development, including vibe coding, AI-assisted coding, and agentic coding, fundamentally reshaping how software is designed, implemented, and maintained. Prior research has primarily examined AI-based coding at the individual level or in educational settings, leaving industrial practitioners' perspectives underexplored. This paper addresses this gap by investigating how LLM coding tools are used in professional practice, the associated concerns and risks, and the resulting transformations in development workflows, with particular attention to implications for computing education. We conducted a qualitative analysis of 57 curated YouTube videos published between late 2024 and 2025, capturing reflections and experiences shared by practitioners. Following a filtering and quality assessment process, the selected sources were analyzed to compare LLM-based and traditional programming, identify emerging risks, and characterize evolving workflows. Our findings reveal definitions of AI-based coding practices, notable productivity gains, and lowered barriers to entry. Practitioners also report a shift in development bottlenecks toward code review and concerns regarding code quality, maintainability, security vulnerabilities, ethical issues, erosion of foundational problem-solving skills, and insufficient preparation of entry-level engineers. Building on these insights, we discuss implications for computer science and software engineering education and argue for curricular shifts toward problem-solving, architectural thinking, code review, and early project-based learning that integrates LLM tools. This study offers an industry-grounded perspective on AI-based coding and provides guidance for aligning educational practices with rapidly evolving professional realities. 

---
# How Large Language Models Systematically Misrepresent American Climate Opinions 

**Authors**: Sola Kim, Jieshu Wang, Marco A. Janssen, John M. Anderies  

**Link**: [PDF](https://arxiv.org/pdf/2512.23889)  

**Abstract**: Federal agencies and researchers increasingly use large language models to analyze and simulate public opinion. When AI mediates between the public and policymakers, accuracy across intersecting identities becomes consequential; inaccurate group-level estimates can mislead outreach, consultation, and policy design. While research examines intersectionality in LLM outputs, no study has compared these outputs against real human responses across intersecting identities. Climate policy is one such domain, and this is particularly urgent for climate change, where opinion is contested and diverse. We investigate how LLMs represent intersectional patterns in U.S. climate opinions. We prompted six LLMs with profiles of 978 respondents from a nationally representative U.S. climate opinion survey and compared AI-generated responses to actual human answers across 20 questions. We find that LLMs appear to compress the diversity of American climate opinions, predicting less-concerned groups as more concerned and vice versa. This compression is intersectional: LLMs apply uniform gender assumptions that match reality for White and Hispanic Americans but misrepresent Black Americans, where actual gender patterns differ. These patterns, which may be invisible to standard auditing approaches, could undermine equitable climate governance. 

---
# Retrieval Augmented Question Answering: When Should LLMs Admit Ignorance? 

**Authors**: Dingmin Wang, Ji Ma, Shankar Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.23836)  

**Abstract**: The success of expanded context windows in Large Language Models (LLMs) has driven increased use of broader context in retrieval-augmented generation. We investigate the use of LLMs for retrieval augmented question answering. While longer contexts make it easier to incorporate targeted knowledge, they introduce more irrelevant information that hinders the model's generation process and degrades its performance. To address the issue, we design an adaptive prompting strategy which involves splitting the retrieved information into smaller chunks and sequentially prompting a LLM to answer the question using each chunk. Adjusting the chunk size allows a trade-off between incorporating relevant information and reducing irrelevant information. Experimental results on three open-domain question answering datasets demonstrate that the adaptive strategy matches the performance of standard prompting while using fewer tokens. Our analysis reveals that when encountering insufficient information, the LLM often generates incorrect answers instead of declining to respond, which constitutes a major source of error. This finding highlights the need for further research into enhancing LLMs' ability to effectively decline requests when faced with inadequate information. 

---
# Prompt-Induced Over-Generation as Denial-of-Service: A Black-Box Attack-Side Benchmark 

**Authors**: Manu, Yi Guo, Jo Plested, Tim Lynar, Kanchana Thilakarathna, Nirhoshan Sivaroopan, Jack Yang, Wangli Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23779)  

**Abstract**: Large language models (LLMs) can be driven into over-generation, emitting thousands of tokens before producing an end-of-sequence (EOS) token. This degrades answer quality, inflates latency and cost, and can be weaponized as a denial-of-service (DoS) attack. Recent work has begun to study DoS-style prompt attacks, but typically focuses on a single attack algorithm or assumes white-box access, without an attack-side benchmark that compares prompt-based attackers in a black-box, query-only regime with a known tokenizer. We introduce such a benchmark and study two prompt-only attackers. The first is Evolutionary Over-Generation Prompt Search (EOGen), which searches the token space for prefixes that suppress EOS and induce long continuations. The second is a goal-conditioned reinforcement learning attacker (RL-GOAL) that trains a network to generate prefixes conditioned on a target length. To characterize behavior, we introduce Over-Generation Factor (OGF), the ratio of produced tokens to a model's context window, along with stall and latency summaries. Our evolutionary attacker achieves mean OGF = 1.38 +/- 1.15 and Success@OGF >= 2 of 24.5 percent on Phi-3. RL-GOAL is stronger: across victims it achieves higher mean OGF (up to 2.81 +/- 1.38). 

---
# Hybrid-Code: A Privacy-Preserving, Redundant Multi-Agent Framework for Reliable Local Clinical Coding 

**Authors**: Yunguo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23743)  

**Abstract**: Clinical coding automation using cloud-based Large Language Models (LLMs) poses privacy risks and latency bottlenecks, rendering them unsuitable for on-premise healthcare deployment. We introduce Hybrid-Code, a hybrid neuro-symbolic multi-agent framework for local clinical coding that ensures production reliability through redundancy and verification. Our system comprises two agents: a Coder that attempts language model-based semantic reasoning using BioMistral-7B but falls back to deterministic keyword matching when model output is unreliable, ensuring pipeline completion; and an Auditor that verifies codes against a 257-code knowledge base and clinical evidence. Evaluating on 1,000 MIMIC-III discharge summaries, we demonstrate no hallucinated codes among accepted outputs within the knowledge base, 24.47% verification rate, and 34.11% coverage (95% CI: 31.2%--37.0%) with 86%+ language model utilization. The Auditor filtered invalid format codes and provided evidence-based quality control (75.53% rejection rate) while ensuring no patient data leaves the hospital firewall. The hybrid architecture -- combining language model semantic understanding (when successful), deterministic fallback (when the model fails), and symbolic verification (always active) -- ensures both reliability and privacy preservation, addressing critical barriers to AI adoption in healthcare. Our key finding is that reliability through redundancy is more valuable than pure model performance in production healthcare systems, where system failures are unacceptable. 

---
# Geometric Scaling of Bayesian Inference in LLMs 

**Authors**: Naman Aggarwal, Siddhartha R. Dalal, Vishal Misra  

**Link**: [PDF](https://arxiv.org/pdf/2512.23752)  

**Abstract**: Recent work has shown that small transformers trained in controlled "wind-tunnel'' settings can implement exact Bayesian inference, and that their training dynamics produce a geometric substrate -- low-dimensional value manifolds and progressively orthogonal keys -- that encodes posterior structure. We investigate whether this geometric signature persists in production-grade language models. Across Pythia, Phi-2, Llama-3, and Mistral families, we find that last-layer value representations organize along a single dominant axis whose position strongly correlates with predictive entropy, and that domain-restricted prompts collapse this structure into the same low-dimensional manifolds observed in synthetic settings.
To probe the role of this geometry, we perform targeted interventions on the entropy-aligned axis of Pythia-410M during in-context learning. Removing or perturbing this axis selectively disrupts the local uncertainty geometry, whereas matched random-axis interventions leave it intact. However, these single-layer manipulations do not produce proportionally specific degradation in Bayesian-like behavior, indicating that the geometry is a privileged readout of uncertainty rather than a singular computational bottleneck. Taken together, our results show that modern language models preserve the geometric substrate that enables Bayesian inference in wind tunnels, and organize their approximate Bayesian updates along this substrate. 

---
# AgenticTCAD: A LLM-based Multi-Agent Framework for Automated TCAD Code Generation and Device Optimization 

**Authors**: Guangxi Fan, Tianliang Ma, Xuguang Sun, Xun Wang, Kain Lu Low, Leilai Shao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23742)  

**Abstract**: With the continued scaling of advanced technology nodes, the design-technology co-optimization (DTCO) paradigm has become increasingly critical, rendering efficient device design and optimization essential. In the domain of TCAD simulation, however, the scarcity of open-source resources hinders language models from generating valid TCAD code. To overcome this limitation, we construct an open-source TCAD dataset curated by experts and fine-tune a domain-specific model for TCAD code generation. Building on this foundation, we propose AgenticTCAD, a natural language - driven multi-agent framework that enables end-to-end automated device design and optimization. Validation on a 2 nm nanosheet FET (NS-FET) design shows that AgenticTCAD achieves the International Roadmap for Devices and Systems (IRDS)-2024 device specifications within 4.2 hours, whereas human experts required 7.1 days with commercial tools. 

---
# Break Out the Silverware -- Semantic Understanding of Stored Household Items 

**Authors**: Michaela Levi-Richter, Reuth Mirsky, Oren Glickman  

**Link**: [PDF](https://arxiv.org/pdf/2512.23739)  

**Abstract**: ``Bring me a plate.'' For domestic service robots, this simple command reveals a complex challenge: inferring where everyday items are stored, often out of sight in drawers, cabinets, or closets. Despite advances in vision and manipulation, robots still lack the commonsense reasoning needed to complete this task. We introduce the Stored Household Item Challenge, a benchmark task for evaluating service robots' cognitive capabilities: given a household scene and a queried item, predict its most likely storage location.
Our benchmark includes two datasets: (1) a real-world evaluation set of 100 item-image pairs with human-annotated ground truth from participants' kitchens, and (2) a development set of 6,500 item-image pairs annotated with storage polygons over public kitchen images. These datasets support realistic modeling of household organization and enable comparative evaluation across agent architectures.
To begin tackling this challenge, we introduce NOAM (Non-visible Object Allocation Model), a hybrid agent pipeline that combines structured scene understanding with large language model inference. NOAM converts visual input into natural language descriptions of spatial context and visible containers, then prompts a language model (e.g., GPT-4) to infer the most likely hidden storage location. This integrated vision-language agent exhibits emergent commonsense reasoning and is designed for modular deployment within broader robotic systems.
We evaluate NOAM against baselines including random selection, vision-language pipelines (Grounding-DINO + SAM), leading multimodal models (e.g., Gemini, GPT-4o, Kosmos-2, LLaMA, Qwen), and human performance. NOAM significantly improves prediction accuracy and approaches human-level results, highlighting best practices for deploying cognitively capable agents in domestic environments. 

---
# Towards representation agnostic probabilistic programming 

**Authors**: Ole Fenske, Maximilian Popko, Sebastian Bader, Thomas Kirste  

**Link**: [PDF](https://arxiv.org/pdf/2512.23740)  

**Abstract**: Current probabilistic programming languages and tools tightly couple model representations with specific inference algorithms, preventing experimentation with novel representations or mixed discrete-continuous models. We introduce a factor abstraction with five fundamental operations that serve as a universal interface for manipulating factors regardless of their underlying representation. This enables representation-agnostic probabilistic programming where users can freely mix different representations (e.g. discrete tables, Gaussians distributions, sample-based approaches) within a single unified framework, allowing practical inference in complex hybrid models that current toolkits cannot adequately express. 

---
