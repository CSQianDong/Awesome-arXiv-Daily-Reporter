# SynCraft: Guiding Large Language Models to Predict Edit Sequences for Molecular Synthesizability Optimization 

**Authors**: Junren Li, Luhua Lai  

**Link**: [PDF](https://arxiv.org/pdf/2512.20333)  

**Abstract**: Generative artificial intelligence has revolutionized the exploration of chemical space, yet a critical bottleneck remains that a substantial fraction of generated molecules is synthetically inaccessible. Current solutions, such as post-hoc filtering or projection-based methods, often compromise structural novelty or disrupt key pharmacophores by forcing molecules into pre-defined synthetic templates. Herein, we introduce SynCraft, a reasoning-based framework that reframes synthesizability optimization not as a sequence translation task, but as a precise structural editing problem. Leveraging the emergent reasoning capabilities of Large Language Models, SynCraft navigates the "synthesis cliff" where minimal structural modifications yield significant gains in synthetic feasibility. By predicting executable sequences of atom-level edits rather than generating SMILES strings directly, SynCraft circumvents the syntactic fragility of LLMs while harnessing their chemical intuition. Extensive benchmarks demonstrate that SynCraft outperforms state-of-the-art baselines in generating synthesizable analogs with high structural fidelity. Furthermore, through interaction-aware prompting, SynCraft successfully replicates expert medicinal chemistry intuition in editing PLK1 inhibitors and rescuing high-scoring but previously discarded RIPK1 candidates in previous molecular generation literatures. 

---
# A DeepSeek-Powered AI System for Automated Chest Radiograph Interpretation in Clinical Practice 

**Authors**: Yaowei Bai, Ruiheng Zhang, Yu Lei, Xuhua Duan, Jingfeng Yao, Shuguang Ju, Chaoyang Wang, Wei Yao, Yiwan Guo, Guilin Zhang, Chao Wan, Qian Yuan, Lei Chen, Wenjuan Tang, Biqiang Zhu, Xinggang Wang, Tao Sun, Wei Zhou, Dacheng Tao, Yongchao Xu, Chuansheng Zheng, Huangxuan Zhao, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.20344)  

**Abstract**: A global shortage of radiologists has been exacerbated by the significant volume of chest X-ray workloads, particularly in primary care. Although multimodal large language models show promise, existing evaluations predominantly rely on automated metrics or retrospective analyses, lacking rigorous prospective clinical validation. Janus-Pro-CXR (1B), a chest X-ray interpretation system based on DeepSeek Janus-Pro model, was developed and rigorously validated through a multicenter prospective trial (NCT07117266). Our system outperforms state-of-the-art X-ray report generation models in automated report generation, surpassing even larger-scale models including ChatGPT 4o (200B parameters), while demonstrating reliable detection of six clinically critical radiographic findings. Retrospective evaluation confirms significantly higher report accuracy than Janus-Pro and ChatGPT 4o. In prospective clinical deployment, AI assistance significantly improved report quality scores, reduced interpretation time by 18.3% (P < 0.001), and was preferred by a majority of experts in 54.3% of cases. Through lightweight architecture and domain-specific optimization, Janus-Pro-CXR improves diagnostic reliability and workflow efficiency, particularly in resource-constrained settings. The model architecture and implementation framework will be open-sourced to facilitate the clinical translation of AI-assisted radiology solutions. 

---
# Automated stereotactic radiosurgery planning using a human-in-the-loop reasoning large language model agent 

**Authors**: Humza Nusrat, Luke Francisco, Bing Luo, Hassan Bagher-Ebadian, Joshua Kim, Karen Chin-Snyder, Salim Siddiqui, Mira Shah, Eric Mellon, Mohammad Ghassemi, Anthony Doemer, Benjamin Movsas, Kundan Thind  

**Link**: [PDF](https://arxiv.org/pdf/2512.20586)  

**Abstract**: Stereotactic radiosurgery (SRS) demands precise dose shaping around critical structures, yet black-box AI systems have limited clinical adoption due to opacity concerns. We tested whether chain-of-thought reasoning improves agentic planning in a retrospective cohort of 41 patients with brain metastases treated with 18 Gy single-fraction SRS. We developed SAGE (Secure Agent for Generative Dose Expertise), an LLM-based planning agent for automated SRS treatment planning. Two variants generated plans for each case: one using a non-reasoning model, one using a reasoning model. The reasoning variant showed comparable plan dosimetry relative to human planners on primary endpoints (PTV coverage, maximum dose, conformity index, gradient index; all p > 0.21) while reducing cochlear dose below human baselines (p = 0.022). When prompted to improve conformity, the reasoning model demonstrated systematic planning behaviors including prospective constraint verification (457 instances) and trade-off deliberation (609 instances), while the standard model exhibited none of these deliberative processes (0 and 7 instances, respectively). Content analysis revealed that constraint verification and causal explanation concentrated in the reasoning agent. The optimization traces serve as auditable logs, offering a path toward transparent automated planning. 

---
# MemR$^3$: Memory Retrieval via Reflective Reasoning for LLM Agents 

**Authors**: Xingbo Du, Loka Li, Duzhen Zhang, Le Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.20237)  

**Abstract**: Memory systems have been designed to leverage past experiences in Large Language Model (LLM) agents. However, many deployed memory systems primarily optimize compression and storage, with comparatively less emphasis on explicit, closed-loop control of memory retrieval. From this observation, we build memory retrieval as an autonomous, accurate, and compatible agent system, named MemR$^3$, which has two core mechanisms: 1) a router that selects among retrieve, reflect, and answer actions to optimize answer quality; 2) a global evidence-gap tracker that explicitly renders the answering process transparent and tracks the evidence collection process. This design departs from the standard retrieve-then-answer pipeline by introducing a closed-loop control mechanism that enables autonomous decision-making. Empirical results on the LoCoMo benchmark demonstrate that MemR$^3$ surpasses strong baselines on LLM-as-a-Judge score, and particularly, it improves existing retrievers across four categories with an overall improvement on RAG (+7.29%) and Zep (+1.94%) using GPT-4.1-mini backend, offering a plug-and-play controller for existing memory stores. 

---
# Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches 

**Authors**: Chaithra, Kamesh Kadimisetty, Biju R Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2512.20082)  

**Abstract**: Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling. 

---
# Concept Generalization in Humans and Large Language Models: Insights from the Number Game 

**Authors**: Arghavan Bazigaran, Hansem Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2512.20162)  

**Abstract**: We compare human and large language model (LLM) generalization in the number game, a concept inference task. Using a Bayesian model as an analytical framework, we examined the inductive biases and inference strategies of humans and LLMs. The Bayesian model captured human behavior better than LLMs in that humans flexibly infer rule-based and similarity-based concepts, whereas LLMs rely more on mathematical rules. Humans also demonstrated a few-shot generalization, even from a single example, while LLMs required more samples to generalize. These contrasts highlight the fundamental differences in how humans and LLMs infer and generalize mathematical concepts. 

---
# Reason2Decide: Rationale-Driven Multi-Task Learning 

**Authors**: H M Quamran Hasan, Housam Khalifa Bashier, Jiayi Dai, Mi-Young Kim, Randy Goebel  

**Link**: [PDF](https://arxiv.org/pdf/2512.20074)  

**Abstract**: Despite the wide adoption of Large Language Models (LLM)s, clinical decision support systems face a critical challenge: achieving high predictive accuracy while generating explanations aligned with the predictions. Current approaches suffer from exposure bias leading to misaligned explanations. We propose Reason2Decide, a two-stage training framework that addresses key challenges in self-rationalization, including exposure bias and task separation. In Stage-1, our model is trained on rationale generation, while in Stage-2, we jointly train on label prediction and rationale generation, applying scheduled sampling to gradually transition from conditioning on gold labels to model predictions. We evaluate Reason2Decide on three medical datasets, including a proprietary triage dataset and public biomedical QA datasets. Across model sizes, Reason2Decide outperforms other fine-tuning baselines and some zero-shot LLMs in prediction (F1) and rationale fidelity (BERTScore, BLEU, LLM-as-a-Judge). In triage, Reason2Decide is rationale source-robust across LLM-generated, nurse-authored, and nurse-post-processed rationales. In our experiments, while using only LLM-generated rationales in Stage-1, Reason2Decide outperforms other fine-tuning variants. This indicates that LLM-generated rationales are suitable for pretraining models, reducing reliance on human annotations. Remarkably, Reason2Decide achieves these gains with models 40x smaller than contemporary foundation models, making clinical reasoning more accessible for resource-constrained deployments while still providing explainable decision support. 

---
# Scaling Reinforcement Learning for Content Moderation with Large Language Models 

**Authors**: Hamed Firooz, Rui Liu, Yuchen Lu, Zhenyu Hou, Fangzhou Xiong, Xiaoyang Zhang, Changshu Jian, Zhicheng Zhu, Jiayuan Ma, Jacob Tao, Chaitali Gupta, Xiaochang Peng, Shike Mei, Hang Cui, Yang Qin, Shuo Tang, Jason Gaedtke, Arpit Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2512.20061)  

**Abstract**: Content moderation at scale remains one of the most pressing challenges in today's digital ecosystem, where billions of user- and AI-generated artifacts must be continuously evaluated for policy violations. Although recent advances in large language models (LLMs) have demonstrated strong potential for policy-grounded moderation, the practical challenges of training these systems to achieve expert-level accuracy in real-world settings remain largely unexplored, particularly in regimes characterized by label sparsity, evolving policy definitions, and the need for nuanced reasoning beyond shallow pattern matching. In this work, we present a comprehensive empirical investigation of scaling reinforcement learning (RL) for content classification, systematically evaluating multiple RL training recipes and reward-shaping strategies-including verifiable rewards and LLM-as-judge frameworks-to transform general-purpose language models into specialized, policy-aligned classifiers across three real-world content moderation tasks. Our findings provide actionable insights for industrial-scale moderation systems, demonstrating that RL exhibits sigmoid-like scaling behavior in which performance improves smoothly with increased training data, rollouts, and optimization steps before gradually saturating. Moreover, we show that RL substantially improves performance on tasks requiring complex policy-grounded reasoning while achieving up to 100x higher data efficiency than supervised fine-tuning, making it particularly effective in domains where expert annotations are scarce or costly. 

---
# Enhancing Zero-Shot Time Series Forecasting in Off-the-Shelf LLMs via Noise Injection 

**Authors**: Xingyou Yin, Ceyao Zhang, Min Hu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.20140)  

**Abstract**: Large Language Models (LLMs) have demonstrated effectiveness as zero-shot time series (TS) forecasters. The key challenge lies in tokenizing TS data into textual representations that align with LLMs' pre-trained knowledge. While existing work often relies on fine-tuning specialized modules to bridge this gap, a distinct, yet challenging, paradigm aims to leverage truly off-the-shelf LLMs without any fine-tuning whatsoever, relying solely on strategic tokenization of numerical sequences. The performance of these fully frozen models is acutely sensitive to the textual representation of the input data, as their parameters cannot adapt to distribution shifts. In this paper, we introduce a simple yet highly effective strategy to overcome this brittleness: injecting noise into the raw time series before tokenization. This non-invasive intervention acts as a form of inference-time augmentation, compelling the frozen LLM to extrapolate based on robust underlying temporal patterns rather than superficial numerical artifacts. We theoretically analyze this phenomenon and empirically validate its effectiveness across diverse benchmarks. Notably, to fully eliminate potential biases from data contamination during LLM pre-training, we introduce two novel TS datasets that fall outside all utilized LLMs' pre-training scopes, and consistently observe improved performance. This study provides a further step in directly leveraging off-the-shelf LLMs for time series forecasting. 

---
# S$^3$IT: A Benchmark for Spatially Situated Social Intelligence Test 

**Authors**: Zhe Sun, Xueyuan Yang, Yujie Lu, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19992)  

**Abstract**: The integration of embodied agents into human environments demands embodied social intelligence: reasoning over both social norms and physical constraints. However, existing evaluations fail to address this integration, as they are limited to either disembodied social reasoning (e.g., in text) or socially-agnostic physical tasks. Both approaches fail to assess an agent's ability to integrate and trade off both physical and social constraints within a realistic, embodied context. To address this challenge, we introduce Spatially Situated Social Intelligence Test (S$^{3}$IT), a benchmark specifically designed to evaluate embodied social intelligence. It is centered on a novel and challenging seat-ordering task, requiring an agent to arrange seating in a 3D environment for a group of large language model-driven (LLM-driven) NPCs with diverse identities, preferences, and intricate interpersonal relationships. Our procedurally extensible framework generates a vast and diverse scenario space with controllable difficulty, compelling the agent to acquire preferences through active dialogue, perceive the environment via autonomous exploration, and perform multi-objective optimization within a complex constraint network. We evaluate state-of-the-art LLMs on S$^{3}$IT and found that they still struggle with this problem, showing an obvious gap compared with the human baseline. Results imply that LLMs have deficiencies in spatial intelligence, yet simultaneously demonstrate their ability to achieve near human-level competence in resolving conflicts that possess explicit textual cues. 

---
# Cube Bench: A Benchmark for Spatial Visual Reasoning in MLLMs 

**Authors**: Dhruv Anand, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2512.20595)  

**Abstract**: We introduce Cube Bench, a Rubik's-cube benchmark for evaluating spatial and sequential reasoning in multimodal large language models (MLLMs). The benchmark decomposes performance into five skills: (i) reconstructing cube faces from images and text, (ii) choosing the optimal next move, (iii) predicting the outcome of a candidate move without applying it, (iv) executing multi-step plans while recovering from mistakes, and (v) detecting and revising one's own errors. Using a shared set of scrambled cube states, identical prompts and parsers, and a single distance-to-solved metric, we compare recent MLLMs side by side as a function of scramble depth. Across seven MLLMs, accuracy drops sharply with depth; once a trajectory stalls or diverges, models rarely recover, and high face-reconstruction accuracy does not guarantee competent action selection or multi-step execution. A pronounced closed- vs open-source gap emerges: the strongest closed model leads on both single-step perception tasks and multi-step control tasks, while open-weight models cluster near chance on the hardest settings; yet even the best MLLM degrades at higher cube complexity. A simple self-correction via reflective thinking yields modest gains but can also introduce overthinking. Cube Bench offers a compact, reproducible probe of sequential spatial reasoning in MLLMs. 

---
# PhysMaster: Building an Autonomous AI Physicist for Theoretical and Computational Physics Research 

**Authors**: Tingjia Miao, Jiawen Dai, Jingkun Liu, Jinxin Tan, Muhua Zhang, Wenkai Jin, Yuwen Du, Tian Jin, Xianghe Pang, Zexi Liu, Tu Guo, Zhengliang Zhang, Yunjie Huang, Shuo Chen, Rui Ye, Yuzhi Zhang, Linfeng Zhang, Kun Chen, Wei Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19799)  

**Abstract**: Advances in LLMs have produced agents with knowledge and operational capabilities comparable to human scientists, suggesting potential to assist, accelerate, and automate research. However, existing studies mainly evaluate such systems on well-defined benchmarks or general tasks like literature retrieval, limiting their end-to-end problem-solving ability in open scientific scenarios. This is particularly true in physics, which is abstract, mathematically intensive, and requires integrating analytical reasoning with code-based computation. To address this, we propose PhysMaster, an LLM-based agent functioning as an autonomous theoretical and computational physicist. PhysMaster couples absract reasoning with numerical computation and leverages LANDAU, the Layered Academic Data Universe, which preserves retrieved literature, curated prior knowledge, and validated methodological traces, enhancing decision reliability and stability. It also employs an adaptive exploration strategy balancing efficiency and open-ended exploration, enabling robust performance in ultra-long-horizon tasks. We evaluate PhysMaster on problems from high-energy theory, condensed matter theory to astrophysics, including: (i) acceleration, compressing labor-intensive research from months to hours; (ii) automation, autonomously executing hypothesis-driven loops ; and (iii) autonomous discovery, independently exploring open problems. 

---
# Interpolative Decoding: Exploring the Spectrum of Personality Traits in LLMs 

**Authors**: Eric Yeh, John Cadigan, Ran Chen, Dick Crouch, Melinda Gervasio, Dayne Freitag  

**Link**: [PDF](https://arxiv.org/pdf/2512.19937)  

**Abstract**: Recent research has explored using very large language models (LLMs) as proxies for humans in tasks such as simulation, surveys, and studies. While LLMs do not possess a human psychology, they often can emulate human behaviors with sufficiently high fidelity to drive simulations to test human behavioral hypotheses, exhibiting more nuance and range than the rule-based agents often employed in behavioral economics. One key area of interest is the effect of personality on decision making, but the requirement that a prompt must be created for every tested personality profile introduces experimental overhead and degrades replicability. To address this issue, we leverage interpolative decoding, representing each dimension of personality as a pair of opposed prompts and employing an interpolation parameter to simulate behavior along the dimension. We show that interpolative decoding reliably modulates scores along each of the Big Five dimensions. We then show how interpolative decoding causes LLMs to mimic human decision-making behavior in economic games, replicating results from human psychological research. Finally, we present preliminary results of our efforts to ``twin'' individual human players in a collaborative game through systematic search for points in interpolation space that cause the system to replicate actions taken by the human subject. 

---
# Toward Explaining Large Language Models in Software Engineering Tasks 

**Authors**: Antonio Vitale, Khai-Nguyen Nguyen, Denys Poshyvanyk, Rocco Oliveto, Simone Scalabrino, Antonio Mastropaolo  

**Link**: [PDF](https://arxiv.org/pdf/2512.20328)  

**Abstract**: Recent progress in Large Language Models (LLMs) has substantially advanced the automation of software engineering (SE) tasks, enabling complex activities such as code generation and code summarization. However, the black-box nature of LLMs remains a major barrier to their adoption in high-stakes and safety-critical domains, where explainability and transparency are vital for trust, accountability, and effective human supervision. Despite increasing interest in explainable AI for software engineering, existing methods lack domain-specific explanations aligned with how practitioners reason about SE artifacts. To address this gap, we introduce FeatureSHAP, the first fully automated, model-agnostic explainability framework tailored to software engineering tasks. Based on Shapley values, FeatureSHAP attributes model outputs to high-level input features through systematic input perturbation and task-specific similarity comparisons, while remaining compatible with both open-source and proprietary LLMs. We evaluate FeatureSHAP on two bi-modal SE tasks: code generation and code summarization. The results show that FeatureSHAP assigns less importance to irrelevant input features and produces explanations with higher fidelity than baseline methods. A practitioner survey involving 37 participants shows that FeatureSHAP helps practitioners better interpret model outputs and make more informed decisions. Collectively, FeatureSHAP represents a meaningful step toward practical explainable AI in software engineering. FeatureSHAP is available at this https URL. 

---
# SweRank+: Multilingual, Multi-Turn Code Ranking for Software Issue Localization 

**Authors**: Revanth Gangi Reddy, Ye Liu, Wenting Zhao, JaeHyeok Doo, Tarun Suresh, Daniel Lee, Caiming Xiong, Yingbo Zhou, Semih Yavuz, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2512.20482)  

**Abstract**: Maintaining large-scale, multilingual codebases hinges on accurately localizing issues, which requires mapping natural-language error descriptions to the relevant functions that need to be modified. However, existing ranking approaches are often Python-centric and perform a single-pass search over the codebase. This work introduces SweRank+, a framework that couples SweRankMulti, a cross-lingual code ranking tool, with SweRankAgent, an agentic search setup, for iterative, multi-turn reasoning over the code repository. SweRankMulti comprises a code embedding retriever and a listwise LLM reranker, and is trained using a carefully curated large-scale issue localization dataset spanning multiple popular programming languages. SweRankAgent adopts an agentic search loop that moves beyond single-shot localization with a memory buffer to reason and accumulate relevant localization candidates over multiple turns. Our experiments on issue localization benchmarks spanning various languages demonstrate new state-of-the-art performance with SweRankMulti, while SweRankAgent further improves localization over single-pass ranking. 

---
# Patterns vs. Patients: Evaluating LLMs against Mental Health Professionals on Personality Disorder Diagnosis through First-Person Narratives 

**Authors**: Karolina Drożdż, Kacper Dudzic, Anna Sterna, Marcin Moskalewicz  

**Link**: [PDF](https://arxiv.org/pdf/2512.20298)  

**Abstract**: Growing reliance on LLMs for psychiatric self-assessment raises questions about their ability to interpret qualitative patient narratives. We present the first direct comparison between state-of-the-art LLMs and mental health professionals in diagnosing Borderline (BPD) and Narcissistic (NPD) Personality Disorders utilizing Polish-language first-person autobiographical accounts. We show that the top-performing Gemini Pro models surpassed human professionals in overall diagnostic accuracy by 21.91 percentage points (65.48% vs. 43.57%). While both models and human experts excelled at identifying BPD (F1 = 83.4 & F1 = 80.0, respectively), models severely underdiagnosed NPD (F1 = 6.7 vs. 50.0), showing a reluctance toward the value-laden term "narcissism." Qualitatively, models provided confident, elaborate justifications focused on patterns and formal categories, while human experts remained concise and cautious, emphasizing the patient's sense of self and temporal experience. Our findings demonstrate that while LLMs are highly competent at interpreting complex first-person clinical data, they remain subject to critical reliability and bias issues. 

---
# KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System 

**Authors**: Zhongyu Xia, Wenhao Chen, Yongtao Wang, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20299)  

**Abstract**: Visual-language reasoning, driving knowledge, and value alignment are essential for advanced autonomous driving systems. However, existing approaches largely rely on data-driven learning, making it difficult to capture the complex logic underlying decision-making through imitation or limited reinforcement rewards. To address this, we propose KnowVal, a new autonomous driving system that enables visual-language reasoning through the synergistic integration of open-world perception and knowledge retrieval. Specifically, we construct a comprehensive driving knowledge graph that encodes traffic laws, defensive driving principles, and ethical norms, complemented by an efficient LLM-based retrieval mechanism tailored for driving scenarios. Furthermore, we develop a human-preference dataset and train a Value Model to guide interpretable, value-aligned trajectory assessment. Experimental results show that our method substantially improves planning performance while remaining compatible with existing architectures. Notably, KnowVal achieves the lowest collision rate on nuScenes and state-of-the-art results on Bench2Drive. 

---
# FaithLens: Detecting and Explaining Faithfulness Hallucination 

**Authors**: Shuzheng Si, Qingyi Wang, Haozhe Zhao, Yuzhuo Bai, Guanqiao Chen, Kangyang Luo, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.20182)  

**Abstract**: Recognizing whether outputs from large language models (LLMs) contain faithfulness hallucination is crucial for real-world applications, e.g., retrieval-augmented generation and summarization. In this paper, we introduce FaithLens, a cost-efficient and effective faithfulness hallucination detection model that can jointly provide binary predictions and corresponding explanations to improve trustworthiness. To achieve this, we first synthesize training data with explanations via advanced LLMs and apply a well-defined data filtering strategy to ensure label correctness, explanation quality, and data diversity. Subsequently, we fine-tune the model on these well-curated training data as a cold start and further optimize it with rule-based reinforcement learning, using rewards for both prediction correctness and explanation quality. Results on 12 diverse tasks show that the 8B-parameter FaithLens outperforms advanced models such as GPT-4.1 and o3. Also, FaithLens can produce high-quality explanations, delivering a distinctive balance of trustworthiness, efficiency, and effectiveness. 

---
# Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography 

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2512.20168)  

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems. 

---
# AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications 

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2512.20164)  

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation. 

---
# TableGPT-R1: Advancing Tabular Reasoning Through Reinforcement Learning 

**Authors**: Saisai Yang, Qingyi Huang, Jing Yuan, Liangyu Zha, Kai Tang, Yuhang Yang, Ning Wang, Yucheng Wei, Liyao Li, Wentao Ye, Hao Chen, Tao Zhang, Junlin Zhou, Haobo Wang, Gang Chen, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.20312)  

**Abstract**: Tabular data serves as the backbone of modern data analysis and scientific research. While Large Language Models (LLMs) fine-tuned via Supervised Fine-Tuning (SFT) have significantly improved natural language interaction with such structured data, they often fall short in handling the complex, multi-step reasoning and robust code execution required for real-world table tasks. Reinforcement Learning (RL) offers a promising avenue to enhance these capabilities, yet its application in the tabular domain faces three critical hurdles: the scarcity of high-quality agentic trajectories with closed-loop code execution and environment feedback on diverse table structures, the extreme heterogeneity of feedback signals ranging from rigid SQL execution to open-ended data interpretation, and the risk of catastrophic forgetting of general knowledge during vertical specialization. To overcome these challenges and unlock advanced reasoning on complex tables, we introduce \textbf{TableGPT-R1}, a specialized tabular model built on a systematic RL framework. Our approach integrates a comprehensive data engineering pipeline that synthesizes difficulty-stratified agentic trajectories for both supervised alignment and RL rollouts, a task-adaptive reward system that combines rule-based verification with a criteria-injected reward model and incorporates process-level step reward shaping with behavioral regularization, and a multi-stage training framework that progressively stabilizes reasoning before specializing in table-specific tasks. Extensive evaluations demonstrate that TableGPT-R1 achieves state-of-the-art performance on authoritative benchmarks, significantly outperforming baseline models while retaining robust general capabilities. Our model is available at this https URL. 

---
# AXIOM: Benchmarking LLM-as-a-Judge for Code via Rule-Based Perturbation and Multisource Quality Calibration 

**Authors**: Ruiqi Wang, Xinchen Wang, Cuiyun Gao, Chun Yong Chong, Xin Xia, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2512.20159)  

**Abstract**: Large language models (LLMs) have been increasingly deployed in real-world software engineering, fostering the development of code evaluation metrics to study the quality of LLM-generated code. Conventional rule-based metrics merely score programs based on their surface-level similarities with reference programs instead of analyzing functionality and code quality in depth. To address this limitation, researchers have developed LLM-as-a-judge metrics, prompting LLMs to evaluate and score code, and curated various code evaluation benchmarks to validate their effectiveness. However, these benchmarks suffer from critical limitations, hindering reliable assessments of evaluation capability: Some feature coarse-grained binary labels, which reduce rich code behavior to a single bit of information, obscuring subtle errors. Others propose fine-grained but subjective, vaguely-defined evaluation criteria, introducing unreliability in manually-annotated scores, which is the ground-truth they rely on. Furthermore, they often use uncontrolled data synthesis methods, leading to unbalanced score distributions that poorly represent real-world code generation scenarios.
To curate a diverse benchmark with programs of well-balanced distributions across various quality levels and streamline the manual annotation procedure, we propose AXIOM, a novel perturbation-based framework for synthesizing code evaluation benchmarks at scale. It reframes program scores as the refinement effort needed for deployment, consisting of two stages: (1) Rule-guided perturbation, which prompts LLMs to apply sequences of predefined perturbation rules to existing high-quality programs to modify their functionality and code quality, enabling us to precisely control each program's target score to achieve balanced score distributions. (2) Multisource quality calibration, which first selects a subset of... 

---
# M$^3$KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation 

**Authors**: Hyeongcheol Park, Jiyoung Seo, Jaewon Mun, Hogun Park, Wonmin Byeon, Sung June Kim, Hyeonsoo Im, JeungSub Lee, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.20136)  

**Abstract**: Retrieval-Augmented Generation (RAG) has recently been extended to multimodal settings, connecting multimodal large language models (MLLMs) with vast corpora of external knowledge such as multimodal knowledge graphs (MMKGs). Despite their recent success, multimodal RAG in the audio-visual domain remains challenging due to 1) limited modality coverage and multi-hop connectivity of existing MMKGs, and 2) retrieval based solely on similarity in a shared multimodal embedding space, which fails to filter out off-topic or redundant knowledge. To address these limitations, we propose M$^3$KG-RAG, a Multi-hop Multimodal Knowledge Graph-enhanced RAG that retrieves query-aligned audio-visual knowledge from MMKGs, improving reasoning depth and answer faithfulness in MLLMs. Specifically, we devise a lightweight multi-agent pipeline to construct multi-hop MMKG (M$^3$KG), which contains context-enriched triplets of multimodal entities, enabling modality-wise retrieval based on input queries. Furthermore, we introduce GRASP (Grounded Retrieval And Selective Pruning), which ensures precise entity grounding to the query, evaluates answer-supporting relevance, and prunes redundant context to retain only knowledge essential for response generation. Extensive experiments across diverse multimodal benchmarks demonstrate that M$^3$KG-RAG significantly enhances MLLMs' multimodal reasoning and grounding over existing approaches. 

---
# On the Effectiveness of Instruction-Tuning Local LLMs for Identifying Software Vulnerabilities 

**Authors**: Sangryu Park, Gihyuk Ko, Homook Cho  

**Link**: [PDF](https://arxiv.org/pdf/2512.20062)  

**Abstract**: Large Language Models (LLMs) show significant promise in automating software vulnerability analysis, a critical task given the impact of security failure of modern software systems. However, current approaches in using LLMs to automate vulnerability analysis mostly rely on using online API-based LLM services, requiring the user to disclose the source code in development. Moreover, they predominantly frame the task as a binary classification(vulnerable or not vulnerable), limiting potential practical utility. This paper addresses these limitations by reformulating the problem as Software Vulnerability Identification (SVI), where LLMs are asked to output the type of weakness in Common Weakness Enumeration (CWE) IDs rather than simply indicating the presence or absence of a vulnerability. We also tackle the reliance on large, API-based LLMs by demonstrating that instruction-tuning smaller, locally deployable LLMs can achieve superior identification performance. In our analysis, instruct-tuning a local LLM showed better overall performance and cost trade-off than online API-based LLMs. Our findings indicate that instruct-tuned local models represent a more effective, secure, and practical approach for leveraging LLMs in real-world vulnerability management workflows. 

---
# Neuron-Guided Interpretation of Code LLMs: Where, Why, and How? 

**Authors**: Zhe Yin, Xiaodong Gu, Beijun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19980)  

**Abstract**: Code language models excel on code intelligence tasks, yet their internal interpretability is underexplored. Existing neuron interpretability techniques from NLP are suboptimal for source code due to programming languages formal, hierarchical, and executable nature. We empirically investigate code LLMs at the neuron level, localizing language-specific neurons (selectively responsive to one language) and concept layers (feed-forward layers encoding language-agnostic code representations). We analyze Llama-3.1-8B and Qwen2.5-Coder-32B on multilingual inputs in C++, Java, Python, Go, and JavaScript, measuring neuron selectivity and layerwise contributions during generation. We find (1) neurons specialized for individual languages alongside a universal subset supporting general-purpose generation; and (2) lower layers mainly encode language-specific syntax, while middle layers capture semantic abstractions shared across languages, emerging as concept layers. We demonstrate utility on three tasks: neuron-guided fine-tuning for code generation, clone detection via concept-layer embeddings, and concept-layer-guided transfer for code summarization, each yielding consistent gains in multilingual settings. 

---
# Fun-Audio-Chat Technical Report 

**Authors**: Qian Chen, Luyao Cheng, Chong Deng, Xiangang Li, Jiaqing Liu, Chao-Hong Tan, Wen Wang, Junhao Xu, Jieping Ye, Qinglin Zhang, Qiquan Zhang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.20156)  

**Abstract**: Recent advancements in joint speech-text models show great potential for seamless voice interactions. However, existing models face critical challenges: temporal resolution mismatch between speech tokens (25Hz) and text tokens (~3Hz) dilutes semantic information, incurs high computational costs, and causes catastrophic forgetting of text LLM knowledge. We introduce Fun-Audio-Chat, a Large Audio Language Model addressing these limitations via two innovations from our previous work DrVoice. First, Dual-Resolution Speech Representations (DRSR): the Shared LLM processes audio at efficient 5Hz (via token grouping), while the Speech Refined Head generates high-quality tokens at 25Hz, balancing efficiency (~50% GPU reduction) and quality. Second, Core-Cocktail Training, a two-stage fine-tuning with intermediate merging that mitigates catastrophic forgetting. We then apply Multi-Task DPO Training to enhance robustness, audio understanding, instruction-following and voice empathy. This multi-stage post-training enables Fun-Audio-Chat to retain text LLM knowledge while gaining powerful audio understanding, reasoning, and generation. Unlike recent LALMs requiring large-scale audio-text pre-training, Fun-Audio-Chat leverages pre-trained models and extensive post-training. Fun-Audio-Chat 8B and MoE 30B-A3B achieve competitive performance on Speech-to-Text and Speech-to-Speech tasks, ranking top among similar-scale models on Spoken QA benchmarks. They also achieve competitive to superior performance on Audio Understanding, Speech Function Calling, Instruction-Following and Voice Empathy. We develop Fun-Audio-Chat-Duplex, a full-duplex variant with strong performance on Spoken QA and full-duplex interactions. We open-source Fun-Audio-Chat-8B with training and inference code, and provide an interactive demo. 

---
# QE-Catalytic: A Graph-Language Multimodal Base Model for Relaxed-Energy Prediction in Catalytic Adsorption 

**Authors**: Yanjie Li, Jian Xu, Xueqing Chen, Lina Yu, Shiming Xiang, Weijun Li, Cheng-lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20084)  

**Abstract**: Adsorption energy is a key descriptor of catalytic reactivity. It is fundamentally defined as the difference between the relaxed total energy of the adsorbate-surface system and that of an appropriate reference state; therefore, the accuracy of relaxed-energy prediction directly determines the reliability of machine-learning-driven catalyst screening. E(3)-equivariant graph neural networks (GNNs) can natively operate on three-dimensional atomic coordinates under periodic boundary conditions and have demonstrated strong performance on such tasks. In contrast, language-model-based approaches, while enabling human-readable textual descriptions and reducing reliance on explicit graph -- thereby broadening applicability -- remain insufficient in both adsorption-configuration energy prediction accuracy and in distinguishing ``the same system with different configurations,'' even with graph-assisted pretraining in the style of GAP-CATBERTa.
To this end, we propose QE-Catalytic, a multimodal framework that deeply couples a large language model (\textbf{Q}wen) with an E(3)-equivariant graph Transformer (\textbf{E}quiformer-V2), enabling unified support for adsorption-configuration property prediction and inverse design on complex catalytic surfaces. During prediction, QE-Catalytic jointly leverages three-dimensional structures and structured configuration text, and injects ``3D geometric information'' into the language channel via graph-text alignment, allowing it to function as a high-performance text-based predictor when precise coordinates are unavailable, while also autoregressively generating CIF files for target-energy-driven structure design and information completion. On OC20, QE-Catalytic reduces the MAE of relaxed adsorption energy from 0.713~eV to 0.486~eV, and consistently outperforms baseline models such as CatBERTa and GAP-CATBERTa across multiple evaluation protocols. 

---
# Schoenfeld's Anatomy of Mathematical Reasoning by Language Models 

**Authors**: Ming Li, Chenrui Fan, Yize Cheng, Soheil Feizi, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.19995)  

**Abstract**: Large language models increasingly expose reasoning traces, yet their underlying cognitive structure and steps remain difficult to identify and analyze beyond surface-level statistics. We adopt Schoenfeld's Episode Theory as an inductive, intermediate-scale lens and introduce ThinkARM (Anatomy of Reasoning in Models), a scalable framework that explicitly abstracts reasoning traces into functional reasoning steps such as Analysis, Explore, Implement, Verify, etc. When applied to mathematical problem solving by diverse models, this abstraction reveals reproducible thinking dynamics and structural differences between reasoning and non-reasoning models, which are not apparent from token-level views. We further present two diagnostic case studies showing that exploration functions as a critical branching step associated with correctness, and that efficiency-oriented methods selectively suppress evaluative feedback steps rather than uniformly shortening responses. Together, our results demonstrate that episode-level representations make reasoning steps explicit, enabling systematic analysis of how reasoning is structured, stabilized, and altered in modern language models. 

---
# ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language 

**Authors**: Aly Lidayan, Jakob Bjorner, Satvik Golechha, Kartik Goyal, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2512.20111)  

**Abstract**: As the length of sequential decision-making tasks increases, it becomes computationally impractical to keep full interaction histories in context. We introduce a general framework for LLM agents to maintain concise contexts through multi-step interaction: Acting through Belief Bottlenecks Expressed in Language (ABBEL), and methods to further improve ABBEL agents with RL post-training. ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns. Under ABBEL, at each step the agent first updates a prior belief with the most recent observation from the environment to form a posterior belief, then uses only the posterior to select an action. We systematically evaluate frontier models under ABBEL across six diverse multi-step environments, finding that ABBEL supports generating interpretable beliefs while maintaining near-constant memory use over interaction steps. However, bottleneck approaches are generally prone to error propagation, which we observe causing inferior performance when compared to the full context setting due to errors in belief updating. Therefore, we train LLMs to generate and act on beliefs within the ABBEL framework via reinforcement learning (RL). We experiment with belief grading, to reward higher quality beliefs, as well as belief length penalties to reward more compressed beliefs. Our experiments demonstrate the ability of RL to improve ABBEL's performance beyond the full context setting, while using less memory than contemporaneous approaches. 

---
# Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning 

**Authors**: Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19920)  

**Abstract**: LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower. 

---
# HARMON-E: Hierarchical Agentic Reasoning for Multimodal Oncology Notes to Extract Structured Data 

**Authors**: Shashi Kant Gupta, Arijeet Pramanik, Jerrin John Thomas, Regina Schwind, Lauren Wiener, Avi Raju, Jeremy Kornbluth, Yanshan Wang, Zhaohui Su, Hrituraj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2512.19864)  

**Abstract**: Unstructured notes within the electronic health record (EHR) contain rich clinical information vital for cancer treatment decision making and research, yet reliably extracting structured oncology data remains challenging due to extensive variability, specialized terminology, and inconsistent document formats. Manual abstraction, although accurate, is prohibitively costly and unscalable. Existing automated approaches typically address narrow scenarios - either using synthetic datasets, restricting focus to document-level extraction, or isolating specific clinical variables (e.g., staging, biomarkers, histology) - and do not adequately handle patient-level synthesis across the large number of clinical documents containing contradictory information. In this study, we propose an agentic framework that systematically decomposes complex oncology data extraction into modular, adaptive tasks. Specifically, we use large language models (LLMs) as reasoning agents, equipped with context-sensitive retrieval and iterative synthesis capabilities, to exhaustively and comprehensively extract structured clinical variables from real-world oncology notes. Evaluated on a large-scale dataset of over 400,000 unstructured clinical notes and scanned PDF reports spanning 2,250 cancer patients, our method achieves an average F1-score of 0.93, with 100 out of 103 oncology-specific clinical variables exceeding 0.85, and critical variables (e.g., biomarkers and medications) surpassing 0.95. Moreover, integration of the agentic system into a data curation workflow resulted in 0.94 direct manual approval rate, significantly reducing annotation costs. To our knowledge, this constitutes the first exhaustive, end-to-end application of LLM-based agents for structured oncology data extraction at scale 

---
# Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling 

**Authors**: Indranil Halder, Cengiz Pehlevan  

**Link**: [PDF](https://arxiv.org/pdf/2512.19905)  

**Abstract**: Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonically with increasing inference time samples $k$. However, the specific reward that optimizes inference-time selection generally differs from the teacher. In contrast, substantial reward misspecification induces a finite optimal $k$ beyond which more sampling can increase the generalization error. For fixed $k$, there exists an optimal sampling temperature. We experimentally verify these facts in large language model inference with an additional large language model as a judge. In the "best-of-$k$" limit with the teacher as reward, we theoretically show that the generalization error decays as $\Theta(1/k^2)$ and determine the leading coefficient via extreme value theory. These formulas delineate domains where scaling inference-time computation is provably preferable to collecting more data. Finally, we demonstrate that when task difficulty increases, the previously mentioned advantage of inference-time compute degrades. 

---
# Attention Distance: A Novel Metric for Directed Fuzzing with Large Language Models 

**Authors**: Wang Bin, Ao Yang, Kedan Li, Aofan Liu, Hui Li, Guibo Luo, Weixiang Huang, Yan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19758)  

**Abstract**: In the domain of software security testing, Directed Grey-Box Fuzzing (DGF) has garnered widespread attention for its efficient target localization and excellent detection performance. However, existing approaches measure only the physical distance between seed execution paths and target locations, overlooking logical relationships among code segments. This omission can yield redundant or misleading guidance in complex binaries, weakening DGF's real-world effectiveness. To address this, we introduce \textbf{attention distance}, a novel metric that leverages a large language model's contextual analysis to compute attention scores between code elements and reveal their intrinsic connections. Under the same AFLGo configuration -- without altering any fuzzing components other than the distance metric -- replacing physical distances with attention distances across 38 real vulnerability reproduction experiments delivers a \textbf{3.43$\times$} average increase in testing efficiency over the traditional method. Compared to state-of-the-art directed fuzzers DAFL and WindRanger, our approach achieves \textbf{2.89$\times$} and \textbf{7.13$\times$} improvements, respectively. To further validate the generalizability of attention distance, we integrate it into DAFL and WindRanger, where it also consistently enhances their original performance. All related code and datasets are publicly available at this https URL\this http URL. 

---
# Large Language Models for EDA Cloud Job Resource and Lifetime Prediction 

**Authors**: Yuxuan Yin, Shengke Zhou, Yunjie Zhang, Ajay Mohindra, Boxun Xu, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.19701)  

**Abstract**: The rapid growth of cloud computing in the Electronic Design Automation (EDA) industry has created a critical need for resource and job lifetime prediction to achieve optimal scheduling. Traditional machine learning methods often struggle with the complexity and heterogeneity of EDA workloads, requiring extensive feature engineering and domain expertise. We propose a novel framework that fine-tunes Large Language Models (LLMs) to address this challenge through text-to-text regression. We introduce the scientific notation and prefix filling to constrain the LLM, significantly improving output format reliability. Moreover, we found that full-attention finetuning and inference improves the prediction accuracy of sliding-window-attention LLMs. We demonstrate the effectiveness of our proposed framework on real-world cloud datasets, setting a new baseline for performance prediction in the EDA domain. 

---
# Automated Fault Detection in 5G Core Networks Using Large Language Models 

**Authors**: Parsa Hatami, Ahmadreza Majlesara, Ali Majlesi, Babak Hossein Khalaj  

**Link**: [PDF](https://arxiv.org/pdf/2512.19697)  

**Abstract**: With the rapid growth of data volume in modern telecommunication networks and the continuous expansion of their scale, maintaining high reliability has become a critical requirement. These networks support a wide range of applications and services, including highly sensitive and mission-critical ones, which demand rapid and accurate detection and resolution of network errors. Traditional fault-diagnosis methods are no longer efficient for such complex environments.\cite{b1} In this study, we leverage Large Language Models (LLMs) to automate network fault detection and classification. Various types of network errors were intentionally injected into a Kubernetes-based test network, and data were collected under both healthy and faulty conditions. The dataset includes logs from different network components (pods), along with complementary data such as system descriptions, events, Round Trip Time (RTT) tests, and pod status information. The dataset covers common fault types such as pod failure, pod kill, network delay, network loss, and disk I/O failures. We fine-tuned the GPT-4.1 nano model via its API on this dataset, resulting in a significant improvement in fault-detection accuracy compared to the base model. These findings highlight the potential of LLM-based approaches for achieving closed-loop, and operator-free fault management, which can enhance network reliability and reduce downtime-related operational costs for service providers. 

---
# Brain-Grounded Axes for Reading and Steering LLM States 

**Authors**: Sandro Andric  

**Link**: [PDF](https://arxiv.org/pdf/2512.19399)  

**Abstract**: Interpretability methods for large language models (LLMs) typically derive directions from textual supervision, which can lack external grounding. We propose using human brain activity not as a training signal but as a coordinate system for reading and steering LLM states. Using the SMN4Lang MEG dataset, we construct a word-level brain atlas of phase-locking value (PLV) patterns and extract latent axes via ICA. We validate axes with independent lexica and NER-based labels (POS/log-frequency used as sanity checks), then train lightweight adapters that map LLM hidden states to these brain axes without fine-tuning the LLM. Steering along the resulting brain-derived directions yields a robust lexical (frequency-linked) axis in a mid TinyLlama layer, surviving perplexity-matched controls, and a brain-vs-text probe comparison shows larger log-frequency shifts (relative to the text probe) with lower perplexity for the brain axis. A function/content axis (axis 13) shows consistent steering in TinyLlama, Qwen2-0.5B, and GPT-2, with PPL-matched text-level corroboration. Layer-4 effects in TinyLlama are large but inconsistent, so we treat them as secondary (Appendix). Axis structure is stable when the atlas is rebuilt without GPT embedding-change features or with word2vec embeddings (|r|=0.64-0.95 across matched axes), reducing circularity concerns. Exploratory fMRI anchoring suggests potential alignment for embedding change and log frequency, but effects are sensitive to hemodynamic modeling assumptions and are treated as population-level evidence only. These results support a new interface: neurophysiology-grounded axes provide interpretable and controllable handles for LLM behavior. 

---
# Fine-Tuned In-Context Learners for Efficient Adaptation 

**Authors**: Jorg Bornschein, Clare Lyle, Yazhe Li, Amal Rannen-Triki, Xu Owen He, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19879)  

**Abstract**: When adapting large language models (LLMs) to a specific downstream task, two primary approaches are commonly employed: (1) prompt engineering, often with in-context few-shot learning, leveraging the model's inherent generalization abilities, and (2) fine-tuning on task-specific data, directly optimizing the model's parameters. While prompt-based methods excel in few-shot scenarios, their effectiveness often plateaus as more data becomes available. Conversely, fine-tuning scales well with data but may underperform when training examples are scarce. We investigate a unified approach that bridges these two paradigms by incorporating in-context learning directly into the fine-tuning process. Specifically, we fine-tune the model on task-specific data augmented with in-context examples, mimicking the structure of k-shot prompts. This approach, while requiring per-task fine-tuning, combines the sample efficiency of in-context learning with the performance gains of fine-tuning, leading to a method that consistently matches and often significantly exceeds both these baselines. To perform hyperparameter selection in the low-data regime, we propose to use prequential evaluation, which eliminates the need for expensive cross-validation and leverages all available data for training while simultaneously providing a robust validation signal. We conduct an extensive empirical study to determine which adaptation paradigm - fine-tuning, in-context learning, or our proposed unified approach offers the best predictive performance on a concrete data downstream-tasks. 

---
# Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits 

**Authors**: Amirhosein Ghasemabadi, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20578)  

**Abstract**: Large language models (LLMs) generate fluent and complex outputs but often fail to recognize their own mistakes and hallucinations. Existing approaches typically rely on external judges, multi-sample consistency, or text-based self-critique, which incur additional compute or correlate weakly with true correctness. We ask: can LLMs predict their own failures by inspecting internal states during inference? We introduce Gnosis, a lightweight self-awareness mechanism that enables frozen LLMs to perform intrinsic self-verification by decoding signals from hidden states and attention patterns. Gnosis passively observes internal traces, compresses them into fixed-budget descriptors, and predicts correctness with negligible inference cost, adding only ~5M parameters and operating independently of sequence length. Across math reasoning, open-domain question answering, and academic knowledge benchmarks, and over frozen backbones ranging from 1.7B to 20B parameters, Gnosis consistently outperforms strong internal baselines and large external judges in both accuracy and calibration. Moreover, it generalizes zero-shot to partial generations, enabling early detection of failing trajectories and compute-aware control. These results show that reliable correctness cues are intrinsic to generation process and can be extracted efficiently without external supervision. 

---
# AprielGuard 

**Authors**: Jaykumar Kasundra, Anjaneya Praharaj, Sourabh Surana, Lakshmi Sirisha Chodisetty, Sourav Sharma, Abhigya Verma, Abhishek Bhardwaj, Debasish Kanhar, Aakash Bhagat, Khalil Slimi, Seganrasan Subramanian, Sathwik Tejaswi Madhusudhan, Ranga Prasad Chenna, Srinivas Sunkara  

**Link**: [PDF](https://arxiv.org/pdf/2512.20293)  

**Abstract**: Safeguarding large language models (LLMs) against unsafe or adversarial behavior is critical as they are increasingly deployed in conversational and agentic settings. Existing moderation tools often treat safety risks (e.g. toxicity, bias) and adversarial threats (e.g. prompt injections, jailbreaks) as separate problems, limiting their robustness and generalizability. We introduce AprielGuard, an 8B parameter safeguard model that unify these dimensions within a single taxonomy and learning framework. AprielGuard is trained on a diverse mix of open and synthetic data covering standalone prompts, multi-turn conversations, and agentic workflows, augmented with structured reasoning traces to improve interpretability. Across multiple public and proprietary benchmarks, AprielGuard achieves strong performance in detecting harmful content and adversarial manipulations, outperforming existing opensource guardrails such as Llama-Guard and Granite Guardian, particularly in multi-step and reasoning intensive scenarios. By releasing the model, we aim to advance transparent and reproducible research on reliable safeguards for LLMs. 

---
# Can LLMs Solve My Grandma's Riddle? Evaluating Multilingual Large Language Models on Reasoning Traditional Bangla Tricky Riddles 

**Authors**: Nurul Labib Sayeedi, Md. Faiyaz Abdullah Sayeedi, Khushnur Binte Jahangir, Swakkhar Shatabda, Sarah Masud Preum  

**Link**: [PDF](https://arxiv.org/pdf/2512.20324)  

**Abstract**: Large Language Models (LLMs) show impressive performance on many NLP benchmarks, yet their ability to reason in figurative, culturally grounded, and low-resource settings remains underexplored. We address this gap for Bangla by introducing BanglaRiddleEval, a benchmark of 1,244 traditional Bangla riddles instantiated across four tasks (4,976 riddle-task artifacts in total). Using an LLM-based pipeline, we generate Chain-of-Thought explanations, semantically coherent distractors, and fine-grained ambiguity annotations, and evaluate a diverse suite of open-source and closed-source models under different prompting strategies. Models achieve moderate semantic overlap on generative QA but low correctness, MCQ accuracy peaks at only about 56% versus an 83% human baseline, and ambiguity resolution ranges from roughly 26% to 68%, with high-quality explanations confined to the strongest models. These results show that current LLMs capture some cues needed for Bangla riddle reasoning but remain far from human-level performance, establishing BanglaRiddleEval as a challenging new benchmark for low-resource figurative reasoning. All data, code, and evaluation scripts are available on GitHub: this https URL. 

---
# Corpus of Cross-lingual Dialogues with Minutes and Detection of Misunderstandings 

**Authors**: Marko Čechovič, Natália Komorníková, Dominik Macháček, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2512.20204)  

**Abstract**: Speech processing and translation technology have the potential to facilitate meetings of individuals who do not share any common language. To evaluate automatic systems for such a task, a versatile and realistic evaluation corpus is needed. Therefore, we create and present a corpus of cross-lingual dialogues between individuals without a common language who were facilitated by automatic simultaneous speech translation. The corpus consists of 5 hours of speech recordings with ASR and gold transcripts in 12 original languages and automatic and corrected translations into English. For the purposes of research into cross-lingual summarization, our corpus also includes written summaries (minutes) of the meetings.
Moreover, we propose automatic detection of misunderstandings. For an overview of this task and its complexity, we attempt to quantify misunderstandings in cross-lingual meetings. We annotate misunderstandings manually and also test the ability of current large language models to detect them automatically. The results show that the Gemini model is able to identify text spans with misunderstandings with recall of 77% and precision of 47%. 

---
# Multi-hop Reasoning via Early Knowledge Alignment 

**Authors**: Yuxin Wang, Shicheng Fang, Bo Wang, Qi Luo, Xuanjing Huang, Yining Zheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20144)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for Large Language Models (LLMs) to address knowledge-intensive queries requiring domain-specific or up-to-date information. To handle complex multi-hop questions that are challenging for single-step retrieval, iterative RAG approaches incorporating reinforcement learning have been proposed. However, existing iterative RAG systems typically plan to decompose questions without leveraging information about the available retrieval corpus, leading to inefficient retrieval and reasoning chains that cascade into suboptimal performance. In this paper, we introduce Early Knowledge Alignment (EKA), a simple but effective module that aligns LLMs with retrieval set before planning in iterative RAG systems with contextually relevant retrieved knowledge. Extensive experiments on six standard RAG datasets demonstrate that by establishing a stronger reasoning foundation, EKA significantly improves retrieval precision, reduces cascading errors, and enhances both performance and efficiency. Our analysis from an entropy perspective demonstrate that incorporating early knowledge reduces unnecessary exploration during the reasoning process, enabling the model to focus more effectively on relevant information subsets. Moreover, EKA proves effective as a versatile, training-free inference strategy that scales seamlessly to large models. Generalization tests across diverse datasets and retrieval corpora confirm the robustness of our approach. Overall, EKA advances the state-of-the-art in iterative RAG systems while illuminating the critical interplay between structured reasoning and efficient exploration in reinforcement learning-augmented frameworks. The code is released at \href{this https URL}{Github}. 

---
# Bias Beneath the Tone: Empirical Characterisation of Tone Bias in LLM-Driven UX Systems 

**Authors**: Heet Bodara, Md Masum Mushfiq, Isma Farah Siddiqui  

**Link**: [PDF](https://arxiv.org/pdf/2512.19950)  

**Abstract**: Large Language Models are increasingly used in conversational systems such as digital personal assistants, shaping how people interact with technology through language. While their responses often sound fluent and natural, they can also carry subtle tone biases such as sounding overly polite, cheerful, or cautious even when neutrality is expected. These tendencies can influence how users perceive trust, empathy, and fairness in dialogue. In this study, we explore tone bias as a hidden behavioral trait of large language models. The novelty of this research lies in the integration of controllable large language model based dialogue synthesis with tone classification models, enabling robust and ethical emotion recognition in personal assistant interactions. We created two synthetic dialogue datasets, one generated from neutral prompts and another explicitly guided to produce positive or negative tones. Surprisingly, even the neutral set showed consistent tonal skew, suggesting that bias may stem from the model's underlying conversational style. Using weak supervision through a pretrained DistilBERT model, we labeled tones and trained several classifiers to detect these patterns. Ensemble models achieved macro F1 scores up to 0.92, showing that tone bias is systematic, measurable, and relevant to designing fair and trustworthy conversational AI. 

---
# How well do Large Language Models Recognize Instructional Moves? Establishing Baselines for Foundation Models in Educational Discourse 

**Authors**: Kirk Vanacore, Rene F. Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2512.19903)  

**Abstract**: Large language models (LLMs) are increasingly adopted in educational technologies for a variety of tasks, from generating instructional materials and assisting with assessment design to tutoring. While prior work has investigated how models can be adapted or optimized for specific tasks, far less is known about how well LLMs perform at interpreting authentic educational scenarios without significant customization. As LLM-based systems become widely adopted by learners and educators in everyday academic contexts, understanding their out-of-the-box capabilities is increasingly important for setting expectations and benchmarking. We compared six LLMs to estimate their baseline performance on a simple but important task: classifying instructional moves in authentic classroom transcripts. We evaluated typical prompting methods: zero-shot, one-shot, and few-shot prompting. We found that while zero-shot performance was moderate, providing comprehensive examples (few-shot prompting) significantly improved performance for state-of-the-art models, with the strongest configuration reaching Cohen's Kappa = 0.58 against expert-coded annotations. At the same time, improvements were neither uniform nor complete: performance varied considerably by instructional move, and higher recall frequently came at the cost of increased false positives. Overall, these findings indicate that foundation models demonstrate meaningful yet limited capacity to interpret instructional discourse, with prompt design helping to surface capability but not eliminating fundamental reliability constraints. 

---
# Making Large Language Models Efficient Dense Retrievers 

**Authors**: Yibin Lei, Shwai He, Ang Li, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2512.20612)  

**Abstract**: Recent work has shown that directly fine-tuning large language models (LLMs) for dense retrieval yields strong performance, but their substantial parameter counts make them computationally inefficient. While prior studies have revealed significant layer redundancy in LLMs for generative tasks, it remains unclear whether similar redundancy exists when these models are adapted for retrieval tasks, which require encoding entire sequences into fixed representations rather than generating tokens iteratively. To this end, we conduct a comprehensive analysis of layer redundancy in LLM-based dense retrievers. We find that, in contrast to generative settings, MLP layers are substantially more prunable, while attention layers remain critical for semantic aggregation. Building on this insight, we propose EffiR, a framework for developing efficient retrievers that performs large-scale MLP compression through a coarse-to-fine strategy (coarse-grained depth reduction followed by fine-grained width reduction), combined with retrieval-specific fine-tuning. Across diverse BEIR datasets and LLM backbones, EffiR achieves substantial reductions in model size and inference cost while preserving the performance of full-size models. 

---
# Learning to Reason in LLMs by Expectation Maximization 

**Authors**: Junghyun Lee, Branislav Kveton, Sunav Choudhary, Subhojyoti Mukherjee, Anup Rao, Ryan A. Rossi, Alexa Siu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20169)  

**Abstract**: Large language models (LLMs) solve reasoning problems by first generating a rationale and then answering. We formalize reasoning as a latent variable model and derive an expectation-maximization (EM) objective for learning to reason. This view connects EM and modern reward-based optimization, and shows that the main challenge lies in designing a sampling distribution that generates rationales that justify correct answers. We instantiate and compare several sampling schemes: rejection sampling with a budget, self-taught reasoner (STaR), and prompt posterior sampling (PPS), which only keeps the rationalization stage of STaR. Our experiments on the ARC, MMLU, and OpenBookQA datasets with the Llama and Qwen models show that the sampling scheme can significantly affect the accuracy of learned reasoning models. Despite its simplicity, we observe that PPS outperforms the other sampling schemes. 

---
# PRISM: A Personality-Driven Multi-Agent Framework for Social Media Simulation 

**Authors**: Zhixiang Lu, Xueyuan Deng, Yiran Liu, Yulong Li, Qiang Yan, Imran Razzak, Jionglong Su  

**Link**: [PDF](https://arxiv.org/pdf/2512.19933)  

**Abstract**: Traditional agent-based models (ABMs) of opinion dynamics often fail to capture the psychological heterogeneity driving online polarization due to simplistic homogeneity assumptions. This limitation obscures the critical interplay between individual cognitive biases and information propagation, thereby hindering a mechanistic understanding of how ideological divides are amplified. To address this challenge, we introduce the Personality-Refracted Intelligent Simulation Model (PRISM), a hybrid framework coupling stochastic differential equations (SDE) for continuous emotional evolution with a personality-conditional partially observable Markov decision process (PC-POMDP) for discrete decision-making. In contrast to continuous trait approaches, PRISM assigns distinct Myers-Briggs Type Indicator (MBTI) based cognitive policies to multimodal large language model (MLLM) agents, initialized via data-driven priors from large-scale social media datasets. PRISM achieves superior personality consistency aligned with human ground truth, significantly outperforming standard homogeneous and Big Five benchmarks. This framework effectively replicates emergent phenomena such as rational suppression and affective resonance, offering a robust tool for analyzing complex social media ecosystems. 

---
# Coherence in the brain unfolds across separable temporal regimes 

**Authors**: Davide Stauba, Finn Rabe, Akhil Misra, Yves Pauli, Roya Hüppi, Nils Lang, Lars Michels, Victoria Edkins, Sascha Frühholz, Iris Sommer, Wolfram Hinzen, Philipp Homan  

**Link**: [PDF](https://arxiv.org/pdf/2512.20481)  

**Abstract**: Coherence in language requires the brain to satisfy two competing temporal demands: gradual accumulation of meaning across extended context and rapid reconfiguration of representations at event boundaries. Despite their centrality to language and thought, how these processes are implemented in the human brain during naturalistic listening remains unclear. Here, we tested whether these two processes can be captured by annotation-free drift and shift signals and whether their neural expression dissociates across large-scale cortical systems. These signals were derived from a large language model (LLM) and formalized contextual drift and event shifts directly from the narrative input. To enable high-precision voxelwise encoding models with stable parameter estimates, we densely sampled one healthy adult across more than 7 hours of listening to thirteen crime stories while collecting ultra high-field (7T) BOLD data. We then modeled the feature-informed hemodynamic response using a regularized encoding framework validated on independent stories. Drift predictions were prevalent in default-mode network hubs, whereas shift predictions were evident bilaterally in the primary auditory cortex and language association cortex. Furthermore, activity in default-mode and parietal networks was best explained by a signal capturing how meaning accumulates and gradually fades over the course of the narrative. Together, these findings show that coherence during language comprehension is implemented through dissociable neural regimes of slow contextual integration and rapid event-driven reconfiguration, offering a mechanistic entry point for understanding disturbances of language coherence in psychiatric disorders. 

---
# Towards Natural Language-Based Document Image Retrieval: New Dataset and Benchmark 

**Authors**: Hao Guo, Xugong Qin, Jun Jie Ou Yang, Peng Zhang, Gangyan Zeng, Yubo Li, Hailun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.20174)  

**Abstract**: Document image retrieval (DIR) aims to retrieve document images from a gallery according to a given query. Existing DIR methods are primarily based on image queries that retrieve documents within the same coarse semantic category, e.g., newspapers or receipts. However, these methods struggle to effectively retrieve document images in real-world scenarios where textual queries with fine-grained semantics are usually provided. To bridge this gap, we introduce a new Natural Language-based Document Image Retrieval (NL-DIR) benchmark with corresponding evaluation metrics. In this work, natural language descriptions serve as semantically rich queries for the DIR task. The NL-DIR dataset contains 41K authentic document images, each paired with five high-quality, fine-grained semantic queries generated and evaluated through large language models in conjunction with manual verification. We perform zero-shot and fine-tuning evaluations of existing mainstream contrastive vision-language models and OCR-free visual document understanding (VDU) models. A two-stage retrieval method is further investigated for performance improvement while achieving both time and space efficiency. We hope the proposed NL-DIR benchmark can bring new opportunities and facilitate research for the VDU community. Datasets and codes will be publicly available at this http URL. 

---
# Counterfactual LLM-based Framework for Measuring Rhetorical Style 

**Authors**: Jingyi Qiu, Hong Chen, Zongyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.19908)  

**Abstract**: The rise of AI has fueled growing concerns about ``hype'' in machine learning papers, yet a reliable way to quantify rhetorical style independently of substantive content has remained elusive. Because bold language can stem from either strong empirical results or mere rhetorical style, it is often difficult to distinguish between the two. To disentangle rhetorical style from substantive content, we introduce a counterfactual, LLM-based framework: multiple LLM rhetorical personas generate counterfactual writings from the same substantive content, an LLM judge compares them through pairwise evaluations, and the outcomes are aggregated using a Bradley--Terry model. Applying this method to 8,485 ICLR submissions sampled from 2017 to 2025, we generate more than 250,000 counterfactual writings and provide a large-scale quantification of rhetorical style in ML papers. We find that visionary framing significantly predicts downstream attention, including citations and media attention, even after controlling for peer-review evaluations. We also observe a sharp rise in rhetorical strength after 2023, and provide empirical evidence showing that this increase is largely driven by the adoption of LLM-based writing assistance. The reliability of our framework is validated by its robustness to the choice of personas and the high correlation between LLM judgments and human annotations. Our work demonstrates that LLMs can serve as instruments to measure and improve scientific evaluation. 

---
# Laser: Governing Long-Horizon Agentic Search via Structured Protocol and Context Register 

**Authors**: Shuting Wang, Qiaolin Xia, Hao Wang, Yu Lu, Bobsimons, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2512.20458)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs) have enabled agentic search systems that interleave multi-step reasoning with external tool use. However, existing frameworks largely rely on unstructured natural-language reasoning and accumulate raw intermediate traces in the context, which often leads to unstable reasoning trajectories, context overflow, and degraded performance on complex multi-hop queries. In this study, we introduce Laser, a general framework for stabilizing and scaling agentic search. Laser defines a symbolic action protocol that organizes agent behaviors into three spaces: planning, task-solving, and retrospection. Each action is specified with explicit semantics and a deterministic execution format, enabling structured and logical reasoning processes and reliable action parsing. This design makes intermediate decisions interpretable and traceable, enhancing explicit retrospection and fine-grained control over reasoning trajectories. In coordination with parsable actions, Laser further maintains a compact context register that stores only essential states of the reasoning process, allowing the agent to reason over long horizons without uncontrolled context expansion. Experiments on Qwen2.5/3-series models across challenging multi-hop QA datasets show that Laser consistently outperforms existing agentic search baselines under both prompting-only and fine-tuning settings, demonstrating that Laser provides a principled and effective foundation for robust, scalable agentic search. 

---
# VSA:Visual-Structural Alignment for UI-to-Code 

**Authors**: Xian Wu, Ming Zhang, Zhiyu Fang, Fei Li, Bin Wang, Yong Jiang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.20034)  

**Abstract**: The automation of user interface development has the potential to accelerate software delivery by mitigating intensive manual implementation. Despite the advancements in Large Multimodal Models for design-to-code translation, existing methodologies predominantly yield unstructured, flat codebases that lack compatibility with component-oriented libraries such as React or Angular. Such outputs typically exhibit low cohesion and high coupling, complicating long-term maintenance. In this paper, we propose \textbf{VSA (VSA)}, a multi-stage paradigm designed to synthesize organized frontend assets through visual-structural alignment. Our approach first employs a spatial-aware transformer to reconstruct the visual input into a hierarchical tree representation. Moving beyond basic layout extraction, we integrate an algorithmic pattern-matching layer to identify recurring UI motifs and encapsulate them into modular templates. These templates are then processed via a schema-driven synthesis engine, ensuring the Large Language Model generates type-safe, prop-drilled components suitable for production environments. Experimental results indicate that our framework yields a substantial improvement in code modularity and architectural consistency over state-of-the-art benchmarks, effectively bridging the gap between raw pixels and scalable software engineering. 

---
# LLM-Assisted Abstract Screening with OLIVER: Evaluating Calibration and Single-Model vs. Actor-Critic Configurations in Literature Reviews 

**Authors**: Kian Godhwani, David Benrimoh  

**Link**: [PDF](https://arxiv.org/pdf/2512.20022)  

**Abstract**: Introduction: Recent work suggests large language models (LLMs) can accelerate screening, but prior evaluations focus on earlier LLMs, standardized Cochrane reviews, single-model setups, and accuracy as the primary metric, leaving generalizability, configuration effects, and calibration largely unexamined.
Methods: We developed OLIVER (Optimized LLM-based Inclusion and Vetting Engine for Reviews), an open-source pipeline for LLM-assisted abstract screening. We evaluated multiple contemporary LLMs across two non-Cochrane systematic reviews and performance was assessed at both the full-text screening and final inclusion stages using accuracy, AUC, and calibration metrics. We further tested an actor-critic screening framework combining two lightweight models under three aggregation rules.
Results: Across individual models, performance varied widely. In the smaller Review 1 (821 abstracts, 63 final includes), several models achieved high sensitivity for final includes but at the cost of substantial false positives and poor calibration. In the larger Review 2 (7741 abstracts, 71 final includes), most models were highly specific but struggled to recover true includes, with prompt design influencing recall. Calibration was consistently weak across single-model configurations despite high overall accuracy. Actor-critic screening improved discrimination and markedly reduced calibration error in both reviews, yielding higher AUCs.
Discussion: LLMs may eventually accelerate abstract screening, but single-model performance is highly sensitive to review characteristics, prompting, and calibration is limited. An actor-critic framework improves classification quality and confidence reliability while remaining computationally efficient, enabling large-scale screening at low cost. 

---
