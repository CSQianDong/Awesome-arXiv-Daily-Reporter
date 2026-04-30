# Factorized Latent Reasoning for LLM-based Recommendation 

**Authors**: Tianqi Gao, Chengkai Huang, Zihan Wang, Cao Liu, Ke Zeng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.26760)  

**Abstract**: Large language models (LLMs) have recently been adopted for recommendation by framing user preference modeling as a language generation problem. However, existing latent reasoning approaches typically represent user intent with a single latent vector, which struggles to capture the inherently multi-faceted nature of user preferences. We propose Factorized Latent Reasoning (FLR), a novel framework for LLM-based sequential recommendation that decomposes latent reasoning into multiple disentangled preference factors. FLR introduces a lightweight multi-factor attention module that iteratively refines a latent thought representation, where each factor attends to distinct aspects of the user's interaction history. To encourage diversity and specialization, we design orthogonality, attention diversity, and sparsity regularization objectives, and dynamically aggregate factor contributions for the final prediction. We further integrate FLR with an efficient reinforcement learning strategy based on group-relative policy optimization, enabling stable alignment directly in the latent reasoning space. Experiments on multiple benchmarks show that FLR consistently outperforms strong baselines while improving robustness and interpretability. 

---
# AgentSim: A Platform for Verifiable Agent-Trace Simulation 

**Authors**: Saber Zerhoudi, Michael Granitzer, Jelena Mitrovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.26653)  

**Abstract**: Training trustworthy agentic LLMs requires data that shows the grounded reasoning process, not just the final answer. Existing datasets fall short: question-answering data is outcome-only, chain-of-thought data is not tied to specific documents, and web-agent datasets track interface actions rather than the core retrieval and synthesis steps of a RAG workflow. We introduce AgentSim, an open-source platform for simulating RAG agents. It generates verifiable, stepwise traces of agent reasoning over any document collection. AgentSim uses a policy to ensure the agent widely explores the document set. It combines a multi-model validation pipeline with an active human-in-the-loop process. This approach focuses human effort on difficult steps where models disagree. Using AgentSim, we construct and release the Agent-Trace Corpus (ATC), a large collection of grounded reasoning trajectories spanning three established IR benchmarks. We make three contributions: (1) the AgentSim platform with two mechanisms, Corpus-Aware Seeding and Active Validation, that improve trace diversity and quality; (2) the Agent-Trace Corpus (ATC), over 103,000 verifiable reasoning steps spanning three IR benchmarks, with 100% grounding rate on substantive answers; and (3) a comparative behavioral analysis revealing systematic differences in how state-of-the-art models approach information seeking. Platform, toolkit, and corpus are publicly available. 

---
# The Bandit's Blind Spot: The Critical Role of User State Representation in Recommender Systems 

**Authors**: Pedro R. Pires, Gregorio F. Azevedo, Rafael T. Sereicikas, Pietro L. Campos, Tiago A. Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2604.26651)  

**Abstract**: With the increasing availability of online information, recommender systems have become an important tool for many web-based systems. Due to the continuous aspect of recommendation environments, these systems increasingly rely on contextual multi-armed bandits (CMAB) to deliver personalized and real-time suggestions. A critical yet underexplored component in these systems is the representation of user state, which typically encapsulates the user's interaction history and is deeply correlated with the model's decisions and learning. In this paper, we investigate the impact of different embedding-based state representations derived from matrix factorization models on the performance of traditional CMAB algorithms. Our large-scale experiments reveal that variations in state representation can lead to improvements greater than those achieved by changing the bandit algorithm itself. Furthermore, no single embedding or aggregation strategy consistently dominates across datasets, underscoring the need for domain-specific evaluation. These results expose a substantial gap in the literature and emphasize that advancing bandit-based recommender systems requires a holistic approach that prioritizes embedding quality and state construction alongside algorithmic innovation. The source code for our experiments is publicly available on this https URL. 

---
# When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26649)  

**Abstract**: Large reasoning models such as DeepSeek-R1 and OpenAI o1 generate extended chains of thought spanning thousands of tokens, yet their integration with retrieval-augmented generation (RAG) remains fundamentally misaligned. Current RAG systems optimize for providing context before reasoning begins, while reasoning models require evidence injection during multi-step inference chains. We introduce ReaLM-Retrieve, a reasoning-aware retrieval framework that addresses this mismatch through three key innovations: (1) a step-level uncertainty detector that identifies knowledge gaps at reasoning-step granularity rather than token or sentence level; (2) a retrieval intervention policy that learns when external evidence maximally benefits ongoing reasoning; and (3) an efficiency-optimized integration mechanism that reduces per-retrieval overhead by 3.2x compared to naive integration. Experiments on MuSiQue, HotpotQA, and 2WikiMultiHopQA demonstrate that ReaLM-Retrieve achieves on average 10.1% absolute improvement in answer F1 over standard RAG (range: 9.0-11.8% across the three benchmarks) while reducing retrieval calls by 47% compared to fixed-interval approaches like IRCoT (all improvements significant at p<0.01, paired bootstrap). On the challenging MuSiQue benchmark requiring 2-4 hop reasoning, our method achieves 71.2% F1 with an average of only 1.8 retrieval calls per question. Analysis shows that ReaLM-Retrieve also improves retrieval quality itself, achieving 81.3% Recall@5 with consistently higher precision and MRR than fixed-interval baselines on supporting evidence, establishing new state-of-the-art efficiency-accuracy trade-offs for reasoning-intensive retrieval tasks. 

---
# Efficient Listwise Reranking with Compressed Document Representations 

**Authors**: Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2604.26483)  

**Abstract**: Reranking, the process of refining the output from a first-stage retriever, is often considered computationally expensive, especially when using Large Language Models (LLMs). A common approach to mitigate this cost involves utilizing smaller LLMs or controlling input length. Inspired by recent advances in document compression for retrieval-augmented generation (RAG), we introduce RRK, an efficient and effective listwise reranker compressing documents into multi-token fixed-size embedding representations. Our simple training via distillation shows that this combination of rich compressed representations and listwise reranking yields a highly efficient and effective system. In particular, our 8B-parameter model runs 3x-18x faster than smaller rerankers (0.6-4B parameters) while matching or outperforming them in effectiveness. The efficiency gains are even more striking on long-document benchmarks, where RRK widens its advantage further. 

---
# CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation 

**Authors**: Yibiao Wei, Jie Zou, Pengfei Zhang, Xiao Ao, Weikang Guo, Zeyu Ma, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26427)  

**Abstract**: Generative recommendation frameworks typically represent items as discrete Semantic IDs (SIDs). While existing studies have sought to enhance SID construction by incorporating multimodal content, collaborative signals, or more advanced quantization techniques, learning high-quality SIDs still faces two key challenges: (1) The two-stage generative recommendation paradigm (SID construction and autoregressive generation) provides insufficient supervision for heterogeneous fusion, which hinders learning high-quality SIDs, and (2) non-uniform embeddings lead to codeword imbalance and generation bias. To address these challenges, we propose a novel generative recommendation framework, called CARD. CARD introduces a visual semantic unit that unifies textual, visual, and collaborative signals into a structured visual representation prior to encoding, enabling holistic semantic modeling and effectively alleviating the semantic gap, thereby reducing the reliance on supervision signals during SID learning. Furthermore, to deal with the highly non-uniform distribution of item semantic embeddings in recommendation scenarios, we develop a non-uniform quantization framework (NU-RQ-VAE), which incorporates a learnable and invertible non-uniform transformation into the quantization process to map skewed semantic distributions into a more balanced latent space, thereby significantly improving codebook utilization and quantization accuracy. Experiments on multiple datasets show that CARD consistently outperforms baseline methods under various settings; meanwhile, the proposed non-uniform transformation module is plug-and-play and remains robust across different quantization schemes. Code is available at this https URL. 

---
# Meta-Learning and Targeted Differential Privacy to Improve the Accuracy-Privacy Trade-off in Recommendations 

**Authors**: Peter Müllner, Dominik Kowald, Markus Schedl, Elisabeth Lex  

**Link**: [PDF](https://arxiv.org/pdf/2604.26390)  

**Abstract**: Balancing differential privacy (DP) with recommendation accuracy is a key challenge in privacy-preserving recommender systems, since DP-noise degrades accuracy. We address this trade-off at both the data and model levels. At the data level, we apply DP only to the most stereotypical user data likely to reveal sensitive attributes, such as gender or age, to reduce unnecessary perturbation; we refer to this as targeted DP. At the model level, we use meta-learning to improve robustness to remaining DP-noise. This achieves a better trade-off between accuracy and privacy than standard approaches: Meta-learning improves accuracy and targeted DP leads to lower empirical privacy risk compared to uniformly applied DP and full DP baselines. Overall, our findings show that selectively applying DP at the data level together with meta-learning at the model level can effectively balance recommendation accuracy and user privacy. 

---
# Explaining the "Why": A Unified Framework for the Additive Attribution of Changes in Arbitrary Measures 

**Authors**: Changsheng Zhou, Dajun Chen, Zhitao Shen, wei jiang, Yong Li, Peng Di  

**Link**: [PDF](https://arxiv.org/pdf/2604.26266)  

**Abstract**: Explaining why aggregated measures change is a critical challenge in data analytics that existing systems struggle to address. While current attribution methods exist, they lack a unified solution that is simultaneously general for arbitrary measures, holistic across both data dimensions and measure composition, and rigorous in its interpretability. To bridge this gap, we introduce a principled framework that reframes attribution through the powerful lens of cooperative game theory. Our key contribution is a classification of measures based on their mathematical structure, which enables a spectrum of algorithms-from general approximations to exact, closed-form solutions-that offer a principled trade-off between generality and performance. We demonstrate our framework's superiority through a multi-faceted evaluation: simulations first confirm its numerical accuracy and then its generality for non-additive measures; a case study on Simpson's Paradox showcases its unique interpretability; and a final experiment proves its practical utility by significantly outperforming existing root cause analysis systems. 

---
# TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation 

**Authors**: Wei Yang, Rui Zhong, Zihan Lin, Xiaodan Wang, Cheng Chen, Huan Ren, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26247)  

**Abstract**: Multimodal recommendation improves user modeling by integrating collaborative signals with heterogeneous item content. In real applications, user interests evolve over time and exhibit nonstationary dynamics, where different preference factors change at different rates. This challenge is amplified in multimodal settings because visual and textual cues can dominate decisions under different temporal regimes. Despite strong progress, most multimodal recommenders still rely on static interaction graphs or coarse temporal heuristics, which limits their ability to model continuous preference evolution with fine-grained temporal adaptation. To address these limitations, we propose TimeMM, a time-conditioned spectral filtering framework for dynamic multimodal recommendation. TimeMM instantiates Time-as-Operator by mapping interaction recency to a family of parametric temporal kernels that reweight edges on the user--item graph, producing component-specific representations without explicit eigendecomposition. To capture non-stationary interests, we introduce Adaptive Spectral Filtering that mixes the operator bank according to temporal context, yielding prediction-specific effective spectral responses. To account for modality-specific temporal sensitivity, we further propose Spectral-Aware Modality Routing that calibrates visual and textual contributions conditioned on the same temporal context. Finally, a ranking-space Spectral Diversity Regularization encourages complementary expert behaviors and prevents filter-bank collapse. Extensive experiments on real-world benchmarks demonstrate that TimeMM consistently outperforms state-of-the-art multimodal recommenders while maintaining linear-time scalability. 

---
# ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems 

**Authors**: Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.26231)  

**Abstract**: The remarkable text understanding and generation capabilities of large language models (LLMs) have revitalized the field of general recommendation based on implicit user feedback. Rather than deploying LLMs directly as recommendation models, a more flexible paradigm leverages their ability to interpret users' historical interactions and semantic contexts to extract structured profiles that characterize user preferences. These profiles can be further transformed into actionable high-dimensional representations, serving as powerful signals to augment and strengthen recommendation models. However, the mechanism by which such profiles enhance recommendation performance within the feature space remains insufficiently understood. Moreover, existing studies predominantly rely on nonlinear alignment and fusion strategies to incorporate these profiles, which often lead to semantic loss and fail to fully exploit their potential. To address these limitations, we revisit profiles from a retrieval perspective and propose a simple yet effective recommendation framework built upon distribution shaping (ProMax) in this paper. We begin by employing dense retrieval to uncover the collaborative relationships between user and item profiles within the feature space. Based on this insight, we introduce a dual distribution-reshaping process, in which the profile distribution acts as a guiding signal to steer the recommendation model toward learning user preferences for unseen items beyond the scope of observed interactions. We apply ProMax to four classic recommendation methods on three public datasets. The results indicate that ProMax substantially improves base model performance and outperforms existing LLM-based recommendation approaches. 

---
# Hierarchical Long-Term Semantic Memory for LinkedIn's Hiring Agent 

**Authors**: Zhentao Xu, Shangjing Zhang, Emir Poyraz, Yvonne Li, Ye Jin, Xie Lu, Xiaoyang Gu, Karthik Ramgopal, Praveen Kumar Bodigutla, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26197)  

**Abstract**: Large Language Model (LLM) agents are increasingly used in real-world products, where personalized and context-aware user interactions are essential. A central enabler of such capabilities is the agent's long-term semantic memory system, which extracts implicit and explicit signals from noisy longitudinal behavioral data, stores them in a structured form, and supports low-latency retrieval. Building industrial-grade long-term memory for LLM agents raises five challenges: scalability, low-latency retrieval, privacy constraints, cross-domain generalizability, and observability. We introduce the Hierarchical Long-Term Semantic Memory (HLTM) framework, which organizes textual data into a schema-aligned memory tree that captures semantic knowledge at multiple levels of granularity, enabling scalable ingestion, privacy-aware storage, low-latency retrieval, and transparent provenance; HLTM further incorporates an adaptation mechanism to generalize across diverse use cases. Extensive evaluations on LinkedIn's Hiring Assistant show that HLTM improves answer correctness and retrieval F1 significantly by more than 10%, while significantly advancing the Pareto frontier between query and indexing latency. HLTM has been deployed in LinkedIn's Hiring Assistant to power core personalization features in production hiring workflows. 

---
# Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective 

**Authors**: Jiancheng Wang, Mingjia Yin, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.26489)  

**Abstract**: DNNs have gained widespread adoption in feature interaction recommendation models. However, there has been a longstanding debate on their roles. On one hand, some works claim that DNNs possess the ability to implicitly capture high-order feature interactions. Conversely, recent studies have highlighted the limitations of DNNs in effectively learning dot products, specifically second-order interactions, let alone higher-order interactions. In this paper, we present a novel perspective to understand the effectiveness of DNNs: their impact on the dimensional robustness of the representations. In particular, we conduct extensive experiments involving both parallel DNNs and stacked DNNs. Our evaluation encompasses an overall study of complete DNN on two feature interaction models, alongside a fine-grained ablation analysis of components within DNNs. Experimental results demonstrate that both parallel and stacked DNNs can effectively mitigate the dimensional collapse of embeddings. Furthermore, a gradient-based theoretical analysis, supported by empirical evidence, uncovers the underlying mechanisms of dimensional collapse. 

---
# Benchmarking Complex Multimodal Document Processing Pipelines: A Unified Evaluation Framework for Enterprise AI 

**Authors**: Saurabh K. Singh, Sachin Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.26382)  

**Abstract**: Most enterprise document AI today is a pipeline. Parse, index, retrieve, generate. Each of those stages has been studied to death on its own -- what's still hard is evaluating the system as a whole.
We built EnterpriseDocBench to take a swing at it: parsing fidelity, indexing efficiency, retrieval relevance, and generation groundedness, all on the same corpus. The corpus is built from public, permissively licensed documents across six enterprise domains (five represented in the current pilot). We ran three pipelines through it -- BM25, dense embedding, and a hybrid -- all with the same GPT-5 generator.
The headline numbers: hybrid retrieval narrowly beats BM25 (nDCG@5 of 0.92 vs. 0.91), and both beat dense embedding (0.83). Hallucination doesn't grow monotonically with document length -- short documents and very long ones both hallucinate more than medium ones (28.1% and 23.8% vs. 9.2%). Cross-stage correlations are very weak: parsing->retrieval r=0.14, parsing->generation r=0.17, retrieval->generation 0.02. If quality were cascading the way most of us assume, those numbers would be much higher; they aren't. Design caveats are real (parsing fixed, generator shared, automated proxy metrics) and we don't oversell the result.
One result that genuinely surprised us: factual accuracy on stated claims is 85.5%, but answer completeness averages 0.40. The system is right when it answers -- it just leaves things out. That gap matters more for real deployments than the headline accuracy number does.
We also describe three reference architectures (ColPali, ColQwen2, agentic complexity-based routing) which are not yet integrated end-to-end. Framework, metrics, baselines, and collection scripts will be released open-source on acceptance. 

---
# FASH-iCNN: Making Editorial Fashion Identity Inspectable Through Multimodal CNN Probing 

**Authors**: Morayo Danielle Adeyemi, Ryan A. Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2604.26186)  

**Abstract**: Fashion AI systems routinely encode the aesthetic logic of specific houses, editors, and historical moments without disclosing it. We present FASH-iCNN, a multimodal system trained on 87,547 Vogue runway images across 15 fashion houses spanning 1991-2024 that makes this cultural logic inspectable. Given a photograph of a garment, the system recovers which house produced it, which era it belongs to, and which color tradition it reflects. A clothing-only model identifies the fashion house at 78.2% top-1 across 14 houses, the decade at 88.6% top-1, and the specific year at 58.3% top-1 across 34 years with a mean error of just 2.2 years. Probing which visual channels carry this signal reveals a sharp dissociation: removing color costs only 10.6pp of house identity accuracy, while removing texture costs 37.6pp, establishing texture and luminance as the primary carriers of editorial identity. FASH-iCNN treats editorial culture as the signal rather than background noise, identifying which houses, eras, and color traditions shaped each output so that users can see not just what the system predicts but which houses, editors, and historical moments are encoded in that prediction. 

---
# RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS): A Structured Methodology Using Large Language Models for Hardware Design 

**Authors**: Shiva Ahir, Alex Doboli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26153)  

**Abstract**: Heuristic design upholds modern electronic design automation (EDA) tools, yet crafting effective placement, routing, and scheduling strategies entails substantial expertise. We study how large language models (LLMs) can systematically synthesize reusable optimization heuristics beyond one-shot code generation. We propose RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS), which integrates retrieval-augmented generation (RAG), compact kernel heuristic templates, and an LLM-driven refinement loop inspired by iterative self-feedback. Applied to latency-minimizing list scheduling in high-level synthesis (HLS), a prototype reduces average schedule length by up to 11 percent over a baseline scheduler with only 1.3x runtime overhead, and the structured retrieval-synthesis loop generalizes to other EDA optimization problems. 

---
# MATH-PT: A Math Reasoning Benchmark for European and Brazilian Portuguese 

**Authors**: Tiago Teixeira, Ana Carolina Erthal, Juan Belieni, Beatriz Canaverde, Diego Mesquita, Miguel Faria, Eliezer de Souza da Silva, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2604.25926)  

**Abstract**: The use of large language models (LLMs) for complex mathematical reasoning is an emergent area of research, with fast progress in methods, models, and benchmark datasets. However, most mathematical reasoning evaluations exhibit a significant linguistic bias, with the vast majority of benchmark datasets being exclusively in English or (at best) translated from English. We address this limitation by introducing {\sc Math-PT}, a novel dataset comprising 1,729 mathematical problems written in European and Brazilian Portuguese. {\sc Math-PT} is curated from a variety of high-quality native sources, including mathematical Olympiads, competitions, and exams from Portugal and Brazil. We present a comprehensive benchmark of current state-of-the-art LLMs on {\sc Math-PT}, revealing that frontier reasoning models achieve strong performance in multiple choice questions compared to open weight models, but that their performance decreases for questions with figures or open-ended questions. To facilitate future research, we release the benchmark dataset and model outputs. 

---
# Generative AI-Based Virtual Assistant using Retrieval-Augmented Generation: An evaluation study for bachelor projects 

**Authors**: Dumitru Verşebeniuc, Martijn Elands, Sara Falahatkar, Chiara Magrone, Mohammad Falah, Martijn Boussé, Aki Härmä  

**Link**: [PDF](https://arxiv.org/pdf/2604.25924)  

**Abstract**: Large Language Models have been increasingly employed in the creation of Virtual Assistants due to their ability to generate human-like text and handle complex inquiries. While these models hold great promise, challenges such as hallucinations, missing information, and the difficulty of providing accurate and context-specific responses persist, particularly when applied to highly specialized content domains. In this paper, we focus on addressing these challenges by developing a virtual assistant designed to support students at Maastricht University in navigating project-specific regulations. We propose a virtual assistant based on a Retrieval-Augmented Generation system that enhances the accuracy and reliability of responses by integrating up-to-date, domain-specific knowledge. Through a robust evaluation framework and real-life testing, we demonstrate that our virtual assistant can effectively meet the needs of students while addressing the inherent challenges of applying Large Language Models to a specialized educational context. This work contributes to the ongoing discourse on improving LLM-based systems for specific applications and highlights areas for further research. 

---
# CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation 

**Authors**: Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25182)  

**Abstract**: A multilingual collection may contain useful knowledge in other languages to supplement and correct the facts in the original language for Retrieval-Augmented Generation (RAG). However, the vanilla approach that simply concatenates multiple pieces of knowledge from different languages into the context may fail to improve effectiveness due to the potential disparities across languages. To better leverage multilingual knowledge, we propose CroSearch-R1, a search-augmented reinforcement learning framework to integrate multilingual knowledge into the Group Relative Policy Optimization (GRPO) process. In particular, the approach adopts a multi-turn retrieval strategy with cross-lingual knowledge integration to dynamically align the knowledge from other languages as supplementary evidence into a unified representation space. Furthermore, we introduce a multilingual rollout mechanism to optimize reasoning transferability across languages. Experimental results demonstrate that our framework effectively leverages cross-lingual complementarity and improves the effectiveness of RAG with multilingual collections. 

---
