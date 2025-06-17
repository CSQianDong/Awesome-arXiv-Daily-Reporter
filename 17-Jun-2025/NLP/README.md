# Steering LLM Thinking with Budget Guidance 

**Authors**: Junyan Li, Wenshuo Zhao, Yang Zhang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13752)  

**Abstract**: Recent deep-thinking large language models often reason extensively to improve performance, but such lengthy reasoning is not always desirable, as it incurs excessive inference costs with disproportionate performance gains. Controlling reasoning length without sacrificing performance is therefore important, but remains challenging, especially under tight thinking budgets. We propose budget guidance, a simple yet effective method for steering the reasoning process of LLMs toward a target budget without requiring any LLM fine-tuning. Our approach introduces a lightweight predictor that models a Gamma distribution over the remaining thinking length during next-token generation. This signal is then used to guide generation in a soft, token-level manner, ensuring that the overall reasoning trace adheres to the specified thinking budget. Budget guidance enables natural control of the thinking length, along with significant token efficiency improvements over baseline methods on challenging math benchmarks. For instance, it achieves up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model. Budget guidance also generalizes to broader task domains and exhibits emergent capabilities, such as estimating question difficulty. The source code is available at: this https URL. 

---
# LTRR: Learning To Rank Retrievers for LLMs 

**Authors**: To Eun Kim, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13743)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed retriever, despite growing evidence that no single retriever performs optimally across all query types. In this paper, we explore a query routing approach that dynamically selects from a pool of retrievers based on the query, using both train-free heuristics and learned routing models. We frame routing as a learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to rank retrievers by their expected utility gain to downstream LLM performance. Our experiments, conducted on synthetic QA data with controlled query type variations, show that routing-based RAG systems can outperform the best single-retriever-based systems. Performance gains are especially pronounced in models trained with the Answer Correctness (AC) metric and with pairwise learning approaches, especially with XGBoost. We also observe improvements in generalization to out-of-distribution queries. As part of the SIGIR 2025 LiveRAG challenge, our submitted system demonstrated the practical viability of our approach, achieving competitive performance in both answer correctness and faithfulness. These findings highlight the importance of both training methodology and metric selection in query routing for RAG systems. 

---
# Instruction Following by Boosting Attention of Large Language Models 

**Authors**: Vitoria Guardieiro, Adam Stein, Avishree Khare, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13734)  

**Abstract**: Controlling the generation of large language models (LLMs) remains a central challenge to ensure their safe and reliable deployment. While prompt engineering and finetuning are common approaches, recent work has explored latent steering, a lightweight technique that alters LLM internal activations to guide generation. However, subsequent studies revealed latent steering's effectiveness to be limited, often underperforming simple instruction prompting. To address this limitation, we first establish a benchmark across diverse behaviors for standardized evaluation of steering techniques. Building on insights from this benchmark, we introduce Instruction Attention Boosting (InstABoost), a latent steering method that boosts the strength of instruction prompting by altering the model's attention during generation. InstABoost combines the strengths of existing approaches and is theoretically supported by prior work that suggests that in-context rule following in transformer-based models can be controlled by manipulating attention on instructions. Empirically, InstABoost demonstrates superior control success compared to both traditional prompting and latent steering. 

---
# Balancing Knowledge Delivery and Emotional Comfort in Healthcare Conversational Systems 

**Authors**: Shang-Chi Tsai, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13692)  

**Abstract**: With the advancement of large language models, many dialogue systems are now capable of providing reasonable and informative responses to patients' medical conditions. However, when patients consult their doctor, they may experience negative emotions due to the severity and urgency of their situation. If the model can provide appropriate comfort and empathy based on the patient's negative emotions while answering medical questions, it will likely offer a more reassuring experience during the medical consultation process. To address this issue, our paper explores the balance between knowledge sharing and emotional support in the healthcare dialogue process. We utilize a large language model to rewrite a real-world interactive medical dialogue dataset, generating patient queries with negative emotions and corresponding medical responses aimed at soothing the patient's emotions while addressing their concerns. The modified data serves to refine the latest large language models with various fine-tuning methods, enabling them to accurately provide sentences with both emotional reassurance and constructive suggestions in response to patients' questions. Compared to the original LLM model, our experimental results demonstrate that our methodology significantly enhances the model's ability to generate emotional responses while maintaining its original capability to provide accurate knowledge-based answers. 

---
# Turning Down the Heat: A Critical Analysis of Min-p Sampling in Language Models 

**Authors**: Rylan Schaeffer, Joshua Kazdan, Yegor Denisov-Blanch  

**Link**: [PDF](https://arxiv.org/pdf/2506.13681)  

**Abstract**: Sampling from language models impacts the quality and diversity of outputs, affecting both research and real-world applications. Recently, Nguyen et al. 2024's "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs" introduced a new sampler called min-p, claiming it achieves superior quality and diversity over established samplers such as basic, top-k, and top-p sampling. The significance of these claims was underscored by the paper's recognition as the 18th highest-scoring submission to ICLR 2025 and selection for an Oral presentation. This paper conducts a comprehensive re-examination of the evidence supporting min-p and reaches different conclusions from the original paper's four lines of evidence. First, the original paper's human evaluations omitted data, conducted statistical tests incorrectly, and described qualitative feedback inaccurately; our reanalysis demonstrates min-p did not outperform baselines in quality, diversity, or a trade-off between quality and diversity; in response to our findings, the authors of the original paper conducted a new human evaluation using a different implementation, task, and rubric that nevertheless provides further evidence min-p does not improve over baselines. Second, comprehensively sweeping the original paper's NLP benchmarks reveals min-p does not surpass baselines when controlling for the number of hyperparameters. Third, the original paper's LLM-as-a-Judge evaluations lack methodological clarity and appear inconsistently reported. Fourth, community adoption claims (49k GitHub repositories, 1.1M GitHub stars) were found to be unsubstantiated, leading to their removal; the revised adoption claim remains misleading. We conclude that evidence presented in the original paper fails to support claims that min-p improves quality, diversity, or a trade-off between quality and diversity. 

---
# Prefix-Tuning+: Modernizing Prefix-Tuning through Attention Independent Prefix Data 

**Authors**: Haonan Wang, Brian Chen, Li Siquan, Liang Xinhe, Tianyang Hu, Hwee Kuan Lee, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13674)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods have become crucial for rapidly adapting large language models (LLMs) to downstream tasks. Prefix-Tuning, an early and effective PEFT technique, demonstrated the ability to achieve performance comparable to full fine-tuning with significantly reduced computational and memory overhead. However, despite its earlier success, its effectiveness in training modern state-of-the-art LLMs has been very limited. In this work, we demonstrate empirically that Prefix-Tuning underperforms on LLMs because of an inherent tradeoff between input and prefix significance within the attention head. This motivates us to introduce Prefix-Tuning+, a novel architecture that generalizes the principles of Prefix-Tuning while addressing its shortcomings by shifting the prefix module out of the attention head itself. We further provide an overview of our construction process to guide future users when constructing their own context-based methods. Our experiments show that, across a diverse set of benchmarks, Prefix-Tuning+ consistently outperforms existing Prefix-Tuning methods. Notably, it achieves performance on par with the widely adopted LoRA method on several general benchmarks, highlighting the potential modern extension of Prefix-Tuning approaches. Our findings suggest that by overcoming its inherent limitations, Prefix-Tuning can remain a competitive and relevant research direction in the landscape of parameter-efficient LLM adaptation. 

---
# EvolvTrip: Enhancing Literary Character Understanding with Temporal Theory-of-Mind Graphs 

**Authors**: Bohao Yang, Hainiu Xu, Jinhua Du, Ze Li, Yulan He, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13641)  

**Abstract**: A compelling portrayal of characters is essential to the success of narrative writing. For readers, appreciating a character's traits requires the ability to infer their evolving beliefs, desires, and intentions over the course of a complex storyline, a cognitive skill known as Theory-of-Mind (ToM). Performing ToM reasoning in prolonged narratives requires readers to integrate historical context with current narrative information, a task at which humans excel but Large Language Models (LLMs) often struggle. To systematically evaluate LLMs' ToM reasoning capability in long narratives, we construct LitCharToM, a benchmark of character-centric questions across four ToM dimensions from classic literature. Further, we introduce EvolvTrip, a perspective-aware temporal knowledge graph that tracks psychological development throughout narratives. Our experiments demonstrate that EvolvTrip consistently enhances performance of LLMs across varying scales, even in challenging extended-context scenarios. EvolvTrip proves to be particularly valuable for smaller models, partially bridging the performance gap with larger LLMs and showing great compatibility with lengthy narratives. Our findings highlight the importance of explicit representation of temporal character mental states in narrative comprehension and offer a foundation for more sophisticated character understanding. Our data and code are publicly available at this https URL. 

---
# An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability 

**Authors**: Yusuke Yamauchi, Taro Yano, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2506.13639)  

**Abstract**: As large language models (LLMs) continue to advance, reliable evaluation methods are essential particularly for open-ended, instruction-following tasks. LLM-as-a-Judge enables automatic evaluation using LLMs as evaluators, but its reliability remains uncertain. In this work, we analyze key factors affecting its trustworthiness, focusing on alignment with human judgments and evaluation consistency. Using BIGGENBench and EvalBiasBench, we study the effects of evaluation design, decoding strategies, and Chain-of-Tought (CoT) reasoning in evaluation. Our results show that evaluation criteria are critical for reliability, non-deterministic sampling improves alignment with human preferences over deterministic evaluation, and CoT reasoning offers minimal gains when clear evaluation criteria are present. 

---
# A Structured Bangla Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy 

**Authors**: Abdullah Al Shafi, Rowzatul Zannat, Abdul Muntakim, Mahmudul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13610)  

**Abstract**: Disease-symptom datasets are significant and in demand for medical research, disease diagnosis, clinical decision-making, and AI-driven health management applications. These datasets help identify symptom patterns associated with specific diseases, thus improving diagnostic accuracy and enabling early detection. The dataset presented in this study systematically compiles disease-symptom relationships from various online sources, medical literature, and publicly available health databases. The data was gathered through analyzing peer-reviewed medical articles, clinical case studies, and disease-symptom association reports. Only the verified medical sources were included in the dataset, while those from non-peer-reviewed and anecdotal sources were excluded. The dataset is structured in a tabular format, where the first column represents diseases, and the remaining columns represent symptoms. Each symptom cell contains a binary value (1 or 0), indicating whether a symptom is associated with a disease (1 for presence, 0 for absence). Thereby, this structured representation makes the dataset very useful for a wide range of applications, including machine learning-based disease prediction, clinical decision support systems, and epidemiological studies. Although there are some advancements in the field of disease-symptom datasets, there is a significant gap in structured datasets for the Bangla language. This dataset aims to bridge that gap by facilitating the development of multilingual medical informatics tools and improving disease prediction models for underrepresented linguistic communities. Further developments should include region-specific diseases and further fine-tuning of symptom associations for better diagnostic performance 

---
# CAMS: A CityGPT-Powered Agentic Framework for Urban Human Mobility Simulation 

**Authors**: Yuwei Du, Jie Feng, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13599)  

**Abstract**: Human mobility simulation plays a crucial role in various real-world applications. Recently, to address the limitations of traditional data-driven approaches, researchers have explored leveraging the commonsense knowledge and reasoning capabilities of large language models (LLMs) to accelerate human mobility simulation. However, these methods suffer from several critical shortcomings, including inadequate modeling of urban spaces and poor integration with both individual mobility patterns and collective mobility distributions. To address these challenges, we propose \textbf{C}ityGPT-Powered \textbf{A}gentic framework for \textbf{M}obility \textbf{S}imulation (\textbf{CAMS}), an agentic framework that leverages the language based urban foundation model to simulate human mobility in urban space. \textbf{CAMS} comprises three core modules, including MobExtractor to extract template mobility patterns and synthesize new ones based on user profiles, GeoGenerator to generate anchor points considering collective knowledge and generate candidate urban geospatial knowledge using an enhanced version of CityGPT, TrajEnhancer to retrieve spatial knowledge based on mobility patterns and generate trajectories with real trajectory preference alignment via DPO. Experiments on real-world datasets show that \textbf{CAMS} achieves superior performance without relying on externally provided geospatial information. Moreover, by holistically modeling both individual mobility patterns and collective mobility constraints, \textbf{CAMS} generates more realistic and plausible trajectories. In general, \textbf{CAMS} establishes a new paradigm that integrates the agentic framework with urban-knowledgeable LLMs for human mobility simulation. 

---
# Qwen vs. Gemma Integration with Whisper: A Comparative Study in Multilingual SpeechLLM Systems 

**Authors**: Tuan Nguyen, Long-Vu Hoang, Huy-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.13596)  

**Abstract**: This paper presents our system for the MLC-SLM Challenge 2025, focusing on multilingual speech recognition and language modeling with large language models (LLMs). Our approach combines a fine-tuned Whisper-large-v3 encoder with efficient projector architectures and various decoder configurations. We employ a three-stage training methodology that progressively optimizes the encoder, projector, and LLM components. Our system achieves competitive performance with a private test average WER/CER result of 16.63% using the Gemma3-12B and 18.6% using the Qwen2.5-7B as decoder-only language model. 

---
# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention 

**Authors**: MiniMax, Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, Chengjun Xiao, Chengyu Du, Chi Zhang, Chu Qiao, Chunhao Zhang, Chunhui Du, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dong Li, Enwei Jiao, Haigang Zhou, Haimo Zhang, Han Ding, Haohai Sun, Haoyu Feng, Huaiguang Cai, Haichao Zhu, Jian Sun, Jiaqi Zhuang, Jiaren Cai, Jiayuan Song, Jin Zhu, Jingyang Li, Jinhao Tian, Jinli Liu, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kaiyi Feng, Ke Yang, Kecheng Xiao, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Li, Lin Zheng, Linge Du, Lingyu Yang, Lunbin Zeng, Minghui Yu, Mingliang Tao, Mingyuan Chi, Mozhi Zhang, Mujie Lin, Nan Hu, Nongyu Di, Peng Gao, Pengfei Li, Pengyu Zhao, Qibing Ren, Qidi Xu, Qile Li, Qin Wang, Rong Tian, Ruitao Leng, Shaoxiang Chen, Shaoyu Chen, Shengmin Shi, Shitong Weng, Shuchang Guan, Shuqi Yu, Sichen Li, Songquan Zhu, Tengfei Li, Tianchi Cai, Tianrun Liang, Weiyu Cheng, Weize Kong, Wenkai Li, Xiancai Chen, Xiangjun Song, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xinzhu Hou, Xuan Lu, Xun Zou, Xuyang Shen, Yan Gong, Yan Ma, Yang Wang, Yiqi Shi, Yiran Zhong, Yonghong Duan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13585)  

**Abstract**: We introduce MiniMax-M1, the world's first open-weight, large-scale hybrid-attention reasoning model. MiniMax-M1 is powered by a hybrid Mixture-of-Experts (MoE) architecture combined with a lightning attention mechanism. The model is developed based on our previous MiniMax-Text-01 model, which contains a total of 456 billion parameters with 45.9 billion parameters activated per token. The M1 model natively supports a context length of 1 million tokens, 8x the context size of DeepSeek R1. Furthermore, the lightning attention mechanism in MiniMax-M1 enables efficient scaling of test-time compute. These properties make M1 particularly suitable for complex tasks that require processing long inputs and thinking extensively. MiniMax-M1 is trained using large-scale reinforcement learning (RL) on diverse problems including sandbox-based, real-world software engineering environments. In addition to M1's inherent efficiency advantage for RL training, we propose CISPO, a novel RL algorithm to further enhance RL efficiency. CISPO clips importance sampling weights rather than token updates, outperforming other competitive RL variants. Combining hybrid-attention and CISPO enables MiniMax-M1's full RL training on 512 H800 GPUs to complete in only three weeks, with a rental cost of just $534,700. We release two versions of MiniMax-M1 models with 40K and 80K thinking budgets respectively, where the 40K model represents an intermediate phase of the 80K training. Experiments on standard benchmarks show that our models are comparable or superior to strong open-weight models such as the original DeepSeek-R1 and Qwen3-235B, with particular strengths in complex software engineering, tool utilization, and long-context tasks. We publicly release MiniMax-M1 at this https URL. 

---
# Characterizing Linguistic Shifts in Croatian News via Diachronic Word Embeddings 

**Authors**: David Dukić, Ana Barić, Marko Čuljak, Josip Jukić, Martin Tutek  

**Link**: [PDF](https://arxiv.org/pdf/2506.13569)  

**Abstract**: Measuring how semantics of words change over time improves our understanding of how cultures and perspectives change. Diachronic word embeddings help us quantify this shift, although previous studies leveraged substantial temporally annotated corpora. In this work, we use a corpus of 9.5 million Croatian news articles spanning the past 25 years and quantify semantic change using skip-gram word embeddings trained on five-year periods. Our analysis finds that word embeddings capture linguistic shifts of terms pertaining to major topics in this timespan (COVID-19, Croatia joining the European Union, technological advancements). We also find evidence that embeddings from post-2020 encode increased positivity in sentiment analysis tasks, contrasting studies reporting a decline in mental health over the same period. 

---
# Understand the Implication: Learning to Think for Pragmatic Understanding 

**Authors**: Settaluri Lakshmi Sravanthi, Kishan Maharaj, Sravani Gunnu, Abhijit Mishra, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2506.13559)  

**Abstract**: Pragmatics, the ability to infer meaning beyond literal interpretation, is crucial for social cognition and communication. While LLMs have been benchmarked for their pragmatic understanding, improving their performance remains underexplored. Existing methods rely on annotated labels but overlook the reasoning process humans naturally use to interpret implicit meaning. To bridge this gap, we introduce a novel pragmatic dataset, ImpliedMeaningPreference, that includes explicit reasoning (thoughts) for both correct and incorrect interpretations. Through preference-tuning and supervised fine-tuning, we demonstrate that thought-based learning significantly enhances LLMs' pragmatic understanding, improving accuracy by 11.12% across model families. We further discuss a transfer-learning study where we evaluate the performance of thought-based training for the other tasks of pragmatics (presupposition, deixis) that are not seen during the training time and observe an improvement of 16.10% compared to label-trained models. 

---
# Mixture of Weight-shared Heterogeneous Group Attention Experts for Dynamic Token-wise KV Optimization 

**Authors**: Guanghui Song, Dongping Liao, Yiren Zhao, Kejiang Ye, Cheng-zhong Xu, Xitong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13541)  

**Abstract**: Transformer models face scalability challenges in causal language modeling (CLM) due to inefficient memory allocation for growing key-value (KV) caches, which strains compute and storage resources. Existing methods like Grouped Query Attention (GQA) and token-level KV optimization improve efficiency but rely on rigid resource allocation, often discarding "low-priority" tokens or statically grouping them, failing to address the dynamic spectrum of token importance. We propose mixSGA, a novel mixture-of-expert (MoE) approach that dynamically optimizes token-wise computation and memory allocation. Unlike prior approaches, mixSGA retains all tokens while adaptively routing them to specialized experts with varying KV group sizes, balancing granularity and efficiency. Our key novelties include: (1) a token-wise expert-choice routing mechanism guided by learned importance scores, enabling proportional resource allocation without token discard; (2) weight-sharing across grouped attention projections to minimize parameter overhead; and (3) an auxiliary loss to ensure one-hot routing decisions for training-inference consistency in CLMs. Extensive evaluations across Llama3, TinyLlama, OPT, and Gemma2 model families show mixSGA's superiority over static baselines. On instruction-following and continued pretraining tasks, mixSGA achieves higher ROUGE-L and lower perplexity under the same KV budgets. 

---
# TensorSLM: Energy-efficient Embedding Compression of Sub-billion Parameter Language Models on Low-end Devices 

**Authors**: Mingxue Xu, Yao Lei Xu, Danilo P. Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2506.13514)  

**Abstract**: Small Language Models (SLMs, or on-device LMs) have significantly fewer parameters than Large Language Models (LLMs). They are typically deployed on low-end devices, like mobile phones and single-board computers. Unlike LLMs, which rely on increasing model size for better generalisation, SLMs designed for edge applications are expected to have adaptivity to the deployment environments and energy efficiency given the device battery life constraints, which are not addressed in datacenter-deployed LLMs. This paper addresses these two requirements by proposing a training-free token embedding compression approach using Tensor-Train Decomposition (TTD). Each pre-trained token embedding vector is converted into a lower-dimensional Matrix Product State (MPS). We comprehensively evaluate the extracted low-rank structures across compression ratio, language task performance, latency, and energy consumption on a typical low-end device, i.e. Raspberry Pi. Taking the sub-billion parameter versions of GPT-2/Cerebres-GPT and OPT models as examples, our approach achieves a comparable language task performance to the original model with around $2.0\times$ embedding layer compression, while the energy consumption of a single query drops by half. 

---
# K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean 

**Authors**: Minkyeong Jeon, Hyemin Jeong, Yerang Kim, Jiyoung Kim, Jae Hyeon Cho, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.13513)  

**Abstract**: Language detoxification involves removing toxicity from offensive language. While a neutral-toxic paired dataset provides a straightforward approach for training detoxification models, creating such datasets presents several challenges: i) the need for human annotation to build paired data, and ii) the rapid evolution of offensive terms, rendering static datasets quickly outdated. To tackle these challenges, we introduce an automated paired data generation pipeline, called K/DA. This pipeline is designed to generate offensive language with implicit offensiveness and trend-aligned slang, making the resulting dataset suitable for detoxification model training. We demonstrate that the dataset generated by K/DA exhibits high pair consistency and greater implicit offensiveness compared to existing Korean datasets, and also demonstrates applicability to other languages. Furthermore, it enables effective training of a high-performing detoxification model with simple instruction fine-tuning. 

---
# BOW: Bottlenecked Next Word Exploration 

**Authors**: Ming Shen, Zhikun Xu, Xiao Ye, Jacob Dineen, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.13502)  

**Abstract**: Large language models (LLMs) are typically trained via next-word prediction (NWP), which provides strong surface-level fluency but often lacks support for robust reasoning. We propose BOttlenecked next Word exploration (BOW), a novel RL framework that rethinks NWP by introducing a reasoning bottleneck where a policy model first generates a reasoning path rather than predicting the next token directly, after which a frozen judge model predicts the next token distribution based solely on this reasoning path. We train the policy model using GRPO with rewards that quantify how effectively the reasoning path facilitates next-word recovery. Compared with other continual pretraining baselines, we show that BOW improves both the general and next-word reasoning capabilities of the base model, evaluated on various benchmarks. Our findings show that BOW can serve as an effective and scalable alternative to vanilla NWP. 

---
# TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs 

**Authors**: Ezgi Başar, Francesca Padovani, Jaap Jumelet, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2506.13487)  

**Abstract**: We introduce TurBLiMP, the first Turkish benchmark of linguistic minimal pairs, designed to evaluate the linguistic abilities of monolingual and multilingual language models (LMs). Covering 16 linguistic phenomena with 1000 minimal pairs each, TurBLiMP fills an important gap in linguistic evaluation resources for Turkish. In designing the benchmark, we give extra attention to two properties of Turkish that remain understudied in current syntactic evaluations of LMs, namely word order flexibility and subordination through morphological processes. Our experiments on a wide range of LMs and a newly collected set of human acceptability judgments reveal that even cutting-edge Large LMs still struggle with grammatical phenomena that are not challenging for humans, and may also exhibit different sensitivities to word order and morphological complexity compared to humans. 

---
# Position: Pause Recycling LoRAs and Prioritize Mechanisms to Uncover Limits and Effectiveness 

**Authors**: Mei-Yen Chen, Thi Thu Uyen Hoang, Michael Hahn, M. Saquib Sarfraz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13479)  

**Abstract**: Merging or routing low-rank adapters (LoRAs) has emerged as a popular solution for enhancing large language models, particularly when data access is restricted by regulatory or domain-specific constraints. This position paper argues that the research community should shift its focus from developing new merging or routing algorithms to understanding the conditions under which reusing LoRAs is truly effective. Through theoretical analysis and synthetic two-hop reasoning and math word-problem tasks, we examine whether reusing LoRAs enables genuine compositional generalization or merely reflects shallow pattern matching. Evaluating two data-agnostic methods--parameter averaging and dynamic adapter selection--we found that reusing LoRAs often fails to logically integrate knowledge across disjoint fine-tuning datasets, especially when such knowledge is underrepresented during pretraining. Our empirical results, supported by theoretical insights into LoRA's limited expressiveness, highlight the preconditions and constraints of reusing them for unseen tasks and cast doubt on its feasibility as a truly data-free approach. We advocate for pausing the pursuit of novel methods for recycling LoRAs and emphasize the need for rigorous mechanisms to guide future academic research in adapter-based model merging and practical system designs for practitioners. 

---
# Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning 

**Authors**: David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Matthias Keicher, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2506.13474)  

**Abstract**: Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency. 

---
# ROSAQ: Rotation-based Saliency-Aware Weight Quantization for Efficiently Compressing Large Language Models 

**Authors**: Junho Yoon, Geom Lee, Donghyeon Jeon, Inho Kang, Seung-Hoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2506.13472)  

**Abstract**: Quantization has been widely studied as an effective technique for reducing the memory requirement of large language models (LLMs), potentially improving the latency time as well. Utilizing the characteristic of rotational invariance of transformer, we propose the rotation-based saliency-aware weight quantization (ROSAQ), which identifies salient channels in the projection feature space, not in the original feature space, where the projected "principal" dimensions are naturally considered as "salient" features. The proposed ROSAQ consists of 1) PCA-based projection, which first performs principal component analysis (PCA) on a calibration set and transforms via the PCA projection, 2) Salient channel dentification, which selects dimensions corresponding to the K-largest eigenvalues as salient channels, and 3) Saliency-aware quantization with mixed-precision, which uses FP16 for salient dimensions and INT3/4 for other dimensions. Experiment results show that ROSAQ shows improvements over the baseline saliency-aware quantization on the original feature space and other existing quantization methods. With kernel fusion, ROSAQ presents about 2.3x speed up over FP16 implementation in generating 256 tokens with a batch size of 64. 

---
# Abstract, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning 

**Authors**: Jun Ma, Fuqiang Niu, Dong Li, Jinzhou Cao, Genan Dai, Bowen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13470)  

**Abstract**: Zero-shot stance detection (ZSSD) aims to identify the stance of text toward previously unseen targets, a setting where conventional supervised models often fail due to reliance on labeled data and shallow lexical cues. Inspired by human cognitive reasoning, we propose the Cognitive Inductive Reasoning Framework (CIRF), which abstracts transferable reasoning schemas from unlabeled text and encodes them as concept-level logic. To integrate these schemas with input arguments, we introduce a Schema-Enhanced Graph Kernel Model (SEGKM) that dynamically aligns local and global reasoning structures. Experiments on SemEval-2016, VAST, and COVID-19-Stance benchmarks show that CIRF establishes new state-of-the-art results, outperforming strong ZSSD baselines by 1.0, 4.5, and 3.3 percentage points in macro-F1, respectively, and achieving comparable accuracy with 70\% fewer labeled examples. We will release the full code upon publication. 

---
# An Interdisciplinary Approach to Human-Centered Machine Translation 

**Authors**: Marine Carpuat, Omri Asscher, Kalika Bali, Luisa Bentivogli, Frédéric Blain, Lynne Bowker, Monojit Choudhury, Hal Daumé III, Kevin Duh, Ge Gao, Alvin Grissom II, Marzena Karpinska, Elaine C. Khoong, William D. Lewis, André F. T. Martins, Mary Nurminen, Douglas W. Oard, Maja Popovic, Michel Simard, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2506.13468)  

**Abstract**: Machine Translation (MT) tools are widely used today, often in contexts where professional translators are not present. Despite progress in MT technology, a gap persists between system development and real-world usage, particularly for non-expert users who may struggle to assess translation reliability. This paper advocates for a human-centered approach to MT, emphasizing the alignment of system design with diverse communicative goals and contexts of use. We survey the literature in Translation Studies and Human-Computer Interaction to recontextualize MT evaluation and design to address the diverse real-world scenarios in which MT is used today. 

---
# Enhancing Omics Cohort Discovery for Research on Neurodegeneration through Ontology-Augmented Embedding Models 

**Authors**: José A. Pardo, Alicia Gómez-Pascual, José T. Palma, Juan A. Botía  

**Link**: [PDF](https://arxiv.org/pdf/2506.13467)  

**Abstract**: The growing volume of omics and clinical data generated for neurodegenerative diseases (NDs) requires new approaches for their curation so they can be ready-to-use in bioinformatics. NeuroEmbed is an approach for the engineering of semantically accurate embedding spaces to represent cohorts and samples. The NeuroEmbed method comprises four stages: (1) extraction of ND cohorts from public repositories; (2) semi-automated normalization and augmentation of metadata of cohorts and samples using biomedical ontologies and clustering on the embedding space; (3) automated generation of a natural language question-answering (QA) dataset for cohorts and samples based on randomized combinations of standardized metadata dimensions and (4) fine-tuning of a domain-specific embedder to optimize queries. We illustrate the approach using the GEO repository and the PubMedBERT pretrained embedder. Applying NeuroEmbed, we semantically indexed 2,801 repositories and 150,924 samples. Amongst many biology-relevant categories, we normalized more than 1,700 heterogeneous tissue labels from GEO into 326 unique ontology-aligned concepts and enriched annotations with new ontology-aligned terms, leading to a fold increase in size for the metadata terms between 2.7 and 20 fold. After fine-tuning PubMedBERT with the QA training data augmented with the enlarged metadata, the model increased its mean Retrieval Precision from 0.277 to 0.866 and its mean Percentile Rank from 0.355 to 0.896. The NeuroEmbed methodology for the creation of electronic catalogues of omics cohorts and samples will foster automated bioinformatic pipelines construction. The NeuroEmbed catalogue of cohorts and samples is available at this https URL. 

---
# Unveiling the Learning Mind of Language Models: A Cognitive Framework and Empirical Study 

**Authors**: Zhengyu Hu, Jianxun Lian, Zheyuan Xiao, Seraphina Zhang, Tianfu Wang, Nicholas Jing Yuan, Xing Xie, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13464)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across tasks such as mathematics, coding, and reasoning, yet their learning ability, which is crucial for adapting to dynamic environments and acquiring new knowledge, remains underexplored. In this work, we address this gap by introducing a framework inspired by cognitive psychology and education. Specifically, we decompose general learning ability into three distinct, complementary dimensions: Learning from Instructor (acquiring knowledge via explicit guidance), Learning from Concept (internalizing abstract structures and generalizing to new contexts), and Learning from Experience (adapting through accumulated exploration and feedback). We conduct a comprehensive empirical study across the three learning dimensions and identify several insightful findings, such as (i) interaction improves learning; (ii) conceptual understanding is scale-emergent and benefits larger models; and (iii) LLMs are effective few-shot learners but not many-shot learners. Based on our framework and empirical findings, we introduce a benchmark that provides a unified and realistic evaluation of LLMs' general learning abilities across three learning cognition dimensions. It enables diagnostic insights and supports evaluation and development of more adaptive and human-like models. 

---
# A Neural Model for Word Repetition 

**Authors**: Daniel Dager, Robin Sobczyk, Emmanuel Chemla, Yair Lakretz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13450)  

**Abstract**: It takes several years for the developing brain of a baby to fully master word repetition-the task of hearing a word and repeating it aloud. Repeating a new word, such as from a new language, can be a challenging task also for adults. Additionally, brain damage, such as from a stroke, may lead to systematic speech errors with specific characteristics dependent on the location of the brain damage. Cognitive sciences suggest a model with various components for the different processing stages involved in word repetition. While some studies have begun to localize the corresponding regions in the brain, the neural mechanisms and how exactly the brain performs word repetition remain largely unknown. We propose to bridge the gap between the cognitive model of word repetition and neural mechanisms in the human brain by modeling the task using deep neural networks. Neural models are fully observable, allowing us to study the detailed mechanisms in their various substructures and make comparisons with human behavior and, ultimately, the brain. Here, we make first steps in this direction by: (1) training a large set of models to simulate the word repetition task; (2) creating a battery of tests to probe the models for known effects from behavioral studies in humans, and (3) simulating brain damage through ablation studies, where we systematically remove neurons from the model, and repeat the behavioral study to examine the resulting speech errors in the "patient" model. Our results show that neural models can mimic several effects known from human research, but might diverge in other aspects, highlighting both the potential and the challenges for future research aimed at developing human-like neural models. 

---
# RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis 

**Authors**: Pengzuo Wu, Yuhang Yang, Guangcheng Zhu, Chao Ye, Hong Gu, Xu Lu, Ruixuan Xiao, Bowen Bao, Yijing He, Liangyu Zha, Wentao Ye, Junbo Zhao, Haobo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13405)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce RealHiTBench, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using 25 state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based pipeline that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs' perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at this https URL. 

---
# Bi-directional Context-Enhanced Speech Large Language Models for Multilingual Conversational ASR 

**Authors**: Yizhou Peng, Hexin Liu, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13396)  

**Abstract**: This paper introduces the integration of language-specific bi-directional context into a speech large language model (SLLM) to improve multilingual continuous conversational automatic speech recognition (ASR). We propose a character-level contextual masking strategy during training, which randomly removes portions of the context to enhance robustness and better emulate the flawed transcriptions that may occur during inference. For decoding, a two-stage pipeline is utilized: initial isolated segment decoding followed by context-aware re-decoding using neighboring hypotheses. Evaluated on the 1500-hour Multilingual Conversational Speech and Language Model (MLC-SLM) corpus covering eleven languages, our method achieves an 18% relative improvement compared to a strong baseline, outperforming even the model trained on 6000 hours of data for the MLC-SLM competition. These results underscore the significant benefit of incorporating contextual information in multilingual continuous conversational ASR. 

---
# Decompositional Reasoning for Graph Retrieval with Large Language Models 

**Authors**: Valentin Six, Evan Dufraisse, Gaël de Chalendar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13380)  

**Abstract**: Large Language Models (LLMs) excel at many NLP tasks, but struggle with multi-hop reasoning and factual consistency, limiting their effectiveness on knowledge-intensive tasks like complex question answering (QA). Linking Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally lack the ability to reason efficiently over graph-structured information. To tackle this problem, we propose a novel retrieval approach that integrates textual knowledge graphs into the LLM reasoning process via query decomposition. Our method decomposes complex questions into sub-questions, retrieves relevant textual subgraphs, and composes a question-specific knowledge graph to guide answer generation. For that, we use a weighted similarity function that focuses on both the complex question and the generated subquestions to extract a relevant subgraph, which allows efficient and precise retrieval for complex questions and improves the performance of LLMs on multi-hop QA tasks. This structured reasoning pipeline enhances factual grounding and interpretability while leveraging the generative strengths of LLMs. We evaluate our method on standard multi-hop QA benchmarks and show that it achieves comparable or superior performance to competitive existing methods, using smaller models and fewer LLM calls. 

---
# Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction 

**Authors**: Didi Zhang, Yaxin Fan, Peifeng Li, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13366)  

**Abstract**: This paper proposes a consistency reflection and correction method for goal-oriented dialogue systems. 

---
# Efficient Medical VIE via Reinforcement Learning 

**Authors**: Lijun Liu, Ruiyang Li, Zhaocheng Liu, Chenglin Zhu, Chong Li, Jiehan Cheng, Qiang Ju, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.13363)  

**Abstract**: Visual Information Extraction (VIE) converts unstructured document images into structured formats like JSON, critical for medical applications such as report analysis and online consultations. Traditional methods rely on OCR and language models, while end-to-end multimodal models offer direct JSON generation. However, domain-specific schemas and high annotation costs limit their effectiveness in medical VIE. We base our approach on the Reinforcement Learning with Verifiable Rewards (RLVR) framework to address these challenges using only 100 annotated samples. Our approach ensures dataset diversity, a balanced precision-recall reward mechanism to reduce hallucinations and improve field coverage, and innovative sampling strategies to enhance reasoning capabilities. Fine-tuning Qwen2.5-VL-7B with our RLVR method, we achieve state-of-the-art performance on medical VIE tasks, significantly improving F1, precision, and recall. While our models excel on tasks similar to medical datasets, performance drops on dissimilar tasks, highlighting the need for domain-specific optimization. Case studies further demonstrate the value of reasoning during training and inference for VIE. 

---
# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns 

**Authors**: Luanbo Wan, Weizhi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.13356)  

**Abstract**: Long-term memory (LTM) is essential for large language models (LLMs) to achieve autonomous intelligence in complex, evolving environments. Despite increasing efforts in memory-augmented and retrieval-based architectures, there remains a lack of standardized benchmarks to systematically evaluate LLMs' long-term memory abilities. Existing benchmarks still face challenges in evaluating knowledge retention and dynamic sequential reasoning, and in their own flexibility, all of which limit their effectiveness in assessing models' LTM capabilities. To address these gaps, we propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures. These structures simulate real-world scenarios by requiring LLMs to navigate hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions. Our benchmark emphasizes two distinct settings to test reasoning complexity: one with immediate feedback upon incorrect decisions, and the other requiring models to independently trace back and revise earlier choices after failure. As part of this benchmark, we also construct a new dataset designed to test LLMs' LTM within narrative-driven environments. We further validate the effectiveness of our approach through detailed experiments. Experimental results demonstrate the benchmark's ability to robustly and reliably assess LTM in LLMs. 

---
# Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks 

**Authors**: Yifei Xu, Tusher Chakraborty, Srinagesh Sharma, Leonardo Nunes, Emre Kıcıman, Songwu Lu, Ranveer Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.13351)  

**Abstract**: Recent advances in Large Language Models (LLMs) have showcased impressive reasoning abilities in structured tasks like mathematics and programming, largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which uses outcome-based signals that are scalable, effective, and robust against reward hacking. However, applying similar techniques to open-ended long-form reasoning tasks remains challenging due to the absence of generic, verifiable reward signals. To address this, we propose Direct Reasoning Optimization (DRO), a reinforcement learning framework for fine-tuning LLMs on open-ended, particularly long-form, reasoning tasks, guided by a new reward signal: the Reasoning Reflection Reward (R3). At its core, R3 selectively identifies and emphasizes key tokens in the reference outcome that reflect the influence of the model's preceding chain-of-thought reasoning, thereby capturing the consistency between reasoning and reference outcome at a fine-grained level. Crucially, R3 is computed internally using the same model being optimized, enabling a fully self-contained training setup. Additionally, we introduce a dynamic data filtering strategy based on R3 for open-ended reasoning tasks, reducing cost while improving downstream performance. We evaluate DRO on two diverse datasets -- ParaRev, a long-form paragraph revision task, and FinQA, a math-oriented QA benchmark -- and show that it consistently outperforms strong baselines while remaining broadly applicable across both open-ended and structured domains. 

---
# NTU Speechlab LLM-Based Multilingual ASR System for Interspeech MLC-SLM Challenge 2025 

**Authors**: Yizhou Peng, Bin Wang, Yi-Wen Chao, Ziyang Ma, Haoyang Zhang, Hexin Liu, Xie Chen, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13339)  

**Abstract**: This report details the NTU Speechlab system developed for the Interspeech 2025 Multilingual Conversational Speech and Language Model (MLC-SLM) Challenge (Task I), where we achieved 5th place. We present comprehensive analyses of our multilingual automatic speech recognition system, highlighting key advancements in model architecture, data selection, and training strategies. In particular, language-specific prompts and model averaging techniques were instrumental in boosting system performance across diverse languages. Compared to the initial baseline system, our final model reduced the average Mix Error Rate from 20.2% to 10.6%, representing an absolute improvement of 9.6% (a relative improvement of 48%) on the evaluation set. Our results demonstrate the effectiveness of our approach and offer practical insights for future Speech Large Language Models. 

---
# EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization 

**Authors**: Zhongqian Fu, Ning Ding, Kai Han, Xianzhi Yu, Xiaosong Li, Xinghao Chen, Yehui Tang, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13329)  

**Abstract**: Mixture-of-Experts (MoE) models have emerged as a cornerstone of large-scale deep learning by efficiently distributing computation and enhancing performance. However, their unique architecture-characterized by sparse expert activation and dynamic routing mechanisms-introduces inherent complexities that challenge conventional quantization techniques. Existing post-training quantization (PTQ) methods struggle to address activation outliers, router consistency and sparse expert calibration, leading to significant performance degradation. To bridge this gap, we propose EAQuant, a novel PTQ framework tailored for MoE architectures. Our method systematically tackles these challenges through three key innovations: (1) expert-aware smoothing aggregation to suppress activation outliers and stabilize quantization, (2) router logits distribution alignment to preserve expert selection consistency post-quantization, and (3) expert-level calibration data balance to optimize sparsely activated experts. Extensive experiments across W4A4 and extreme W3A4 quantization configurations demonstrate that EAQuant significantly outperforms existing methods, achieving average score improvements of 1.15 - 2.28% across three diverse MoE architectures, with particularly pronounced gains in reasoning tasks and robust performance retention under aggressive quantization. By integrating these innovations, EAQuant establishes a new state-of-the-art for high-precision, efficient MoE model compression. Our code is available at this https URL. 

---
# Document-Level Tabular Numerical Cross-Checking: A Coarse-to-Fine Approach 

**Authors**: Chaoxu Pang, Yixuan Cao, Ganbin Zhou, Hongwei Li, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.13328)  

**Abstract**: Numerical consistency across tables in disclosure documents is critical for ensuring accuracy, maintaining credibility, and avoiding reputational and economic risks. Automated tabular numerical cross-checking presents two significant challenges: (C1) managing the combinatorial explosion of candidate instances at the document level and (C2) comprehending multi-faceted numerical semantics. Previous research typically depends on heuristic-based filtering or simplified context extraction, often struggling to balance performance and efficiency. Recently, large language models (LLMs) have demonstrated remarkable contextual understanding capabilities that helps address C2 at the instance level, yet they remain hampered by computational inefficiency (C1) and limited domain expertise. This paper introduces CoFiTCheck, a novel LLM-based coarse-to-fine framework that addresses these challenges through two sequential stages: embedding-based filtering and discriminative classification. The embedding-based filtering stage introduces an instructional parallel encoding method to efficiently represent all numerical mentions in a table with LLMs, as well as a decoupled InfoNCE objective to mitigate the isolated mention problem. The discriminative classification stage employs a specialized LLM for fine-grained analysis of the remaining candidate pairs. This stage is further enhanced by our crosstable numerical alignment pretraining paradigm, which leverages weak supervision from cross-table numerical equality relationships to enrich task-specific priors without requiring manual annotation. Comprehensive evaluation across three types of real-world disclosure documents demonstrates that CoFiTCheck significantly outperforms previous methods while maintaining practical efficiency. 

---
# Large Language Models as 'Hidden Persuaders': Fake Product Reviews are Indistinguishable to Humans and Machines 

**Authors**: Weiyao Meng, John Harvey, James Goulding, Chris James Carter, Evgeniya Lukinova, Andrew Smith, Paul Frobisher, Mina Forrest, Georgiana Nica-Avram  

**Link**: [PDF](https://arxiv.org/pdf/2506.13313)  

**Abstract**: Reading and evaluating product reviews is central to how most people decide what to buy and consume online. However, the recent emergence of Large Language Models and Generative Artificial Intelligence now means writing fraudulent or fake reviews is potentially easier than ever. Through three studies we demonstrate that (1) humans are no longer able to distinguish between real and fake product reviews generated by machines, averaging only 50.8% accuracy overall - essentially the same that would be expected by chance alone; (2) that LLMs are likewise unable to distinguish between fake and real reviews and perform equivalently bad or even worse than humans; and (3) that humans and LLMs pursue different strategies for evaluating authenticity which lead to equivalently bad accuracy, but different precision, recall and F1 scores - indicating they perform worse at different aspects of judgment. The results reveal that review systems everywhere are now susceptible to mechanised fraud if they do not depend on trustworthy purchase verification to guarantee the authenticity of reviewers. Furthermore, the results provide insight into the consumer psychology of how humans judge authenticity, demonstrating there is an inherent 'scepticism bias' towards positive reviews and a special vulnerability to misjudge the authenticity of fake negative reviews. Additionally, results provide a first insight into the 'machine psychology' of judging fake reviews, revealing that the strategies LLMs take to evaluate authenticity radically differ from humans, in ways that are equally wrong in terms of accuracy, but different in their misjudgments. 

---
# Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models 

**Authors**: Bo Li, Chengben Xu, Wufeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13300)  

**Abstract**: This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints. 

---
# Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs 

**Authors**: Houcheng Jiang, Zetong Zhao, Junfeng Fang, Haokai Ma, Ruipeng Wang, Yang Deng, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2506.13285)  

**Abstract**: Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines. 

---
# AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy 

**Authors**: Zihan Liu, Zhuolin Yang, Yang Chen, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2506.13284)  

**Abstract**: In this work, we investigate the synergy between supervised fine-tuning (SFT) and reinforcement learning (RL) in developing strong reasoning models. We begin by curating the SFT training data through two scaling strategies: increasing the number of collected prompts and the number of generated responses per prompt. Both approaches yield notable improvements in reasoning performance, with scaling the number of prompts resulting in more substantial gains. We then explore the following questions regarding the synergy between SFT and RL: (i) Does a stronger SFT model consistently lead to better final performance after large-scale RL training? (ii) How can we determine an appropriate sampling temperature during RL training to effectively balance exploration and exploitation for a given SFT initialization? Our findings suggest that (i) holds true, provided effective RL training is conducted, particularly when the sampling temperature is carefully chosen to maintain the temperature-adjusted entropy around 0.3, a setting that strikes a good balance between exploration and exploitation. Notably, the performance gap between initial SFT models narrows significantly throughout the RL process. Leveraging a strong SFT foundation and insights into the synergistic interplay between SFT and RL, our AceReason-Nemotron-1.1 7B model significantly outperforms AceReason-Nemotron-1.0 and achieves new state-of-the-art performance among Qwen2.5-7B-based reasoning models on challenging math and code benchmarks, thereby demonstrating the effectiveness of our post-training recipe. We release the model and data at: this https URL 

---
# IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation 

**Authors**: Zijie Lin, Yang Zhang, Xiaoyan Zhao, Fengbin Zhu, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.13229)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines. 

---
# Capability Salience Vector: Fine-grained Alignment of Loss and Capabilities for Downstream Task Scaling Law 

**Authors**: Qiming Ge, Shuhao Xing, Songyang Gao, Yunhua Zhou, Yicheng Zou, Songyang Zhang, Zhi Chen, Hang Yan, Qi Zhang, Qipeng Guo, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13216)  

**Abstract**: Scaling law builds the relationship between training computation and validation loss, enabling researchers to effectively predict the loss trending of models across different levels of computation. However, a gap still remains between validation loss and the model's downstream capabilities, making it untrivial to apply scaling law to direct performance prediction for downstream tasks. The loss typically represents a cumulative penalty for predicted tokens, which are implicitly considered to have equal importance. Nevertheless, our studies have shown evidence that when considering different training data distributions, we cannot directly model the relationship between downstream capability and computation or token loss. To bridge the gap between validation loss and downstream task capabilities, in this work, we introduce Capability Salience Vector, which decomposes the overall loss and assigns different importance weights to tokens to assess a specific meta-capability, aligning the validation loss with downstream task performance in terms of the model's capabilities. Experiments on various popular benchmarks demonstrate that our proposed Capability Salience Vector could significantly improve the predictability of language model performance on downstream tasks. 

---
# Do Music Preferences Reflect Cultural Values? A Cross-National Analysis Using Music Embedding and World Values Survey 

**Authors**: Yongjae Kim, Seongchan Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.13199)  

**Abstract**: This study explores the extent to which national music preferences reflect underlying cultural values. We collected long-term popular music data from YouTube Music Charts across 62 countries, encompassing both Western and non-Western regions, and extracted audio embeddings using the CLAP model. To complement these quantitative representations, we generated semantic captions for each track using LP-MusicCaps and GPT-based summarization. Countries were clustered based on contrastive embeddings that highlight deviations from global musical norms. The resulting clusters were projected into a two-dimensional space via t-SNE for visualization and evaluated against cultural zones defined by the World Values Survey (WVS). Statistical analyses, including MANOVA and chi-squared tests, confirmed that music-based clusters exhibit significant alignment with established cultural groupings. Furthermore, residual analysis revealed consistent patterns of overrepresentation, suggesting non-random associations between specific clusters and cultural zones. These findings indicate that national-level music preferences encode meaningful cultural signals and can serve as a proxy for understanding global cultural boundaries. 

---
# Breaking Thought Patterns: A Multi-Dimensional Reasoning Framework for LLMs 

**Authors**: Xintong Tang, Meiru Zhang, Shang Xiao, Junzhao Jin, Zihan Zhao, Liwei Li, Yang Zheng, Bangyi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13192)  

**Abstract**: Large language models (LLMs) are often constrained by rigid reasoning processes, limiting their ability to generate creative and diverse responses. To address this, a novel framework called LADDER is proposed, combining Chain-of-Thought (CoT) reasoning, Mixture of Experts (MoE) models, and multi-dimensional up/down-sampling strategies which breaks the limitations of traditional LLMs. First, CoT reasoning guides the model through multi-step logical reasoning, expanding the semantic space and breaking the rigidity of thought. Next, MoE distributes the reasoning tasks across multiple expert modules, each focusing on specific sub-tasks. Finally, dimensionality reduction maps the reasoning outputs back to a lower-dimensional semantic space, yielding more precise and creative responses. Extensive experiments across multiple tasks demonstrate that LADDER significantly improves task completion, creativity, and fluency, generating innovative and coherent responses that outperform traditional models. Ablation studies reveal the critical roles of CoT and MoE in enhancing reasoning abilities and creative output. This work contributes to the development of more flexible and creative LLMs, capable of addressing complex and novel tasks. 

---
# Align-then-Unlearn: Embedding Alignment for LLM Unlearning 

**Authors**: Philipp Spohn, Leander Girrbach, Jessica Bader, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2506.13181)  

**Abstract**: As large language models (LLMs) are trained on massive datasets, they have raised significant privacy and ethical concerns due to their potential to inadvertently retain sensitive information. Unlearning seeks to selectively remove specific data from trained models, such as personal information or copyrighted content. Current approaches targeting specific output sequences at the token level often fail to achieve complete forgetting and remain susceptible to prompt rephrasing. We propose Align-then-Unlearn, a novel framework that performs unlearning in the semantic embedding space rather than directly on output tokens. Align-then-Unlearn first augments the LLM with an embedding prediction module trained to anticipate future context representations. Unlearning is then achieved by fine-tuning the model to minimize the similarity between these predicted embeddings and a target embedding that represents the concept to be removed. Initial results show that Align-then-Unlearn effectively removes targeted knowledge with minimal degradation in overall model utility. These findings suggest that embedding-based unlearning offers a promising and robust approach to removing conceptual knowledge. Our code is available at this https URL. 

---
# Dynamic Acoustic Model Architecture Optimization in Training for ASR 

**Authors**: Jingjing Xu, Zijian Yang, Albert Zeyer, Eugen Beck, Ralf Schlueter, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2506.13180)  

**Abstract**: Architecture design is inherently complex. Existing approaches rely on either handcrafted rules, which demand extensive empirical expertise, or automated methods like neural architecture search, which are computationally intensive. In this paper, we introduce DMAO, an architecture optimization framework that employs a grow-and-drop strategy to automatically reallocate parameters during training. This reallocation shifts resources from less-utilized areas to those parts of the model where they are most beneficial. Notably, DMAO only introduces negligible training overhead at a given model complexity. We evaluate DMAO through experiments with CTC on LibriSpeech, TED-LIUM-v2 and Switchboard datasets. The results show that, using the same amount of training resources, our proposed DMAO consistently improves WER by up to 6% relatively across various architectures, model sizes, and datasets. Furthermore, we analyze the pattern of parameter redistribution and uncover insightful findings. 

---
# Enhancing Large Language Models with Reliable Knowledge Graphs 

**Authors**: Qinggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13178)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in text generation and understanding, yet their reliance on implicit, unstructured knowledge often leads to factual inaccuracies and limited interpretability. Knowledge Graphs (KGs), with their structured, relational representations, offer a promising solution to ground LLMs in verified knowledge. However, their potential remains constrained by inherent noise, incompleteness, and the complexity of integrating their rigid structure with the flexible reasoning of LLMs. This thesis presents a systematic framework to address these limitations, advancing the reliability of KGs and their synergistic integration with LLMs through five interconnected contributions. This thesis addresses these challenges through a cohesive framework that enhances LLMs by refining and leveraging reliable KGs. First, we introduce contrastive error detection, a structure-based method to identify incorrect facts in KGs. This approach is extended by an attribute-aware framework that unifies structural and semantic signals for error correction. Next, we propose an inductive completion model that further refines KGs by completing the missing relationships in evolving KGs. Building on these refined KGs, KnowGPT integrates structured graph reasoning into LLMs through dynamic prompting, improving factual grounding. These contributions form a systematic pipeline (from error detection to LLM integration), demonstrating that reliable KGs significantly enhance the robustness, interpretability, and adaptability of LLMs. 

---
# Development of the user-friendly decision aid Rule-based Evaluation and Support Tool (REST) for optimizing the resources of an information extraction task 

**Authors**: Guillaume Bazin, Xavier Tannier, Fanny Adda, Ariel Cohen, Akram Redjdal, Emmanuelle Kempf  

**Link**: [PDF](https://arxiv.org/pdf/2506.13177)  

**Abstract**: Rules could be an information extraction (IE) default option, compared to ML and LLMs in terms of sustainability, transferability, interpretability, and development burden. We suggest a sustainable and combined use of rules and ML as an IE method. Our approach starts with an exhaustive expert manual highlighting in a single working session of a representative subset of the data corpus. We developed and validated the feasibility and the performance metrics of the REST decision tool to help the annotator choose between rules as a by default option and ML for each entity of an IE task. REST makes the annotator visualize the characteristics of each entity formalization in the free texts and the expected rule development feasibility and IE performance metrics. ML is considered as a backup IE option and manual annotation for training is therefore minimized. The external validity of REST on a 12-entity use case showed good reproducibility. 

---
# Ai-Facilitated Analysis of Abstracts and Conclusions: Flagging Unsubstantiated Claims and Ambiguous Pronouns 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13172)  

**Abstract**: We present and evaluate a suite of proof-of-concept (PoC), structured workflow prompts designed to elicit human-like hierarchical reasoning while guiding Large Language Models (LLMs) in high-level semantic and linguistic analysis of scholarly manuscripts. The prompts target two non-trivial analytical tasks: identifying unsubstantiated claims in summaries (informational integrity) and flagging ambiguous pronoun references (linguistic clarity). We conducted a systematic, multi-run evaluation on two frontier models (Gemini Pro 2.5 Pro and ChatGPT Plus o3) under varied context conditions. Our results for the informational integrity task reveal a significant divergence in model performance: while both models successfully identified an unsubstantiated head of a noun phrase (95% success), ChatGPT consistently failed (0% success) to identify an unsubstantiated adjectival modifier that Gemini correctly flagged (95% success), raising a question regarding potential influence of the target's syntactic role. For the linguistic analysis task, both models performed well (80-90% success) with full manuscript context. In a summary-only setting, however, ChatGPT achieved a perfect (100%) success rate, while Gemini's performance was substantially degraded. Our findings suggest that structured prompting is a viable methodology for complex textual analysis but show that prompt performance may be highly dependent on the interplay between the model, task type, and context, highlighting the need for rigorous, model-specific testing. 

---
# Adapting LLMs for Minimal-edit Grammatical Error Correction 

**Authors**: Ryszard Staruch, Filip Graliński, Daniel Dzienisiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13148)  

**Abstract**: Decoder-only large language models have shown superior performance in the fluency-edit English Grammatical Error Correction, but their adaptation for minimal-edit English GEC is still underexplored. To improve their effectiveness in the minimal-edit approach, we explore the error rate adaptation topic and propose a novel training schedule method. Our experiments set a new state-of-the-art result for a single-model system on the BEA-test set. We also detokenize the most common English GEC datasets to match the natural way of writing text. During the process, we find that there are errors in them. Our experiments analyze whether training on detokenized datasets impacts the results and measure the impact of the usage of the datasets with corrected erroneous examples. To facilitate reproducibility, we have released the source code used to train our models. 

---
# CMU's IWSLT 2025 Simultaneous Speech Translation System 

**Authors**: Siqi Ouyang, Xi Xu, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13143)  

**Abstract**: This paper presents CMU's submission to the IWSLT 2025 Simultaneous Speech Translation (SST) task for translating unsegmented English speech into Chinese and German text in a streaming manner. Our end-to-end speech-to-text system integrates a chunkwise causal Wav2Vec 2.0 speech encoder, an adapter, and the Qwen2.5-7B-Instruct as the decoder. We use a two-stage simultaneous training procedure on robust speech segments curated from LibriSpeech, CommonVoice, and VoxPopuli datasets, utilizing standard cross-entropy loss. Our model supports adjustable latency through a configurable latency multiplier. Experimental results demonstrate that our system achieves 44.3 BLEU for English-to-Chinese and 25.1 BLEU for English-to-German translations on the ACL60/60 development set, with computation-aware latencies of 2.7 seconds and 2.3 seconds, and theoretical latencies of 2.2 and 1.7 seconds, respectively. 

---
# Leveraging In-Context Learning for Language Model Agents 

**Authors**: Shivanshu Gupta, Sameer Singh, Ashish Sabharwal, Tushar Khot, Ben Bogin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13109)  

**Abstract**: In-context learning (ICL) with dynamically selected demonstrations combines the flexibility of prompting large language models (LLMs) with the ability to leverage training data to improve performance. While ICL has been highly successful for prediction and generation tasks, leveraging it for agentic tasks that require sequential decision making is challenging -- one must think not only about how to annotate long trajectories at scale and how to select demonstrations, but also what constitutes demonstrations, and when and where to show them. To address this, we first propose an algorithm that leverages an LLM with retries along with demonstrations to automatically and efficiently annotate agentic tasks with solution trajectories. We then show that set-selection of trajectories of similar tasks as demonstrations significantly improves performance, reliability, robustness, and efficiency of LLM agents. However, trajectory demonstrations have a large inference cost overhead. We show that this can be mitigated by using small trajectory snippets at every step instead of an additional trajectory. We find that demonstrations obtained from larger models (in the annotation phase) also improve smaller models, and that ICL agents can even rival costlier trained agents. Thus, our results reveal that ICL, with careful use, can be very powerful for agentic tasks as well. 

---
# Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for LLMs and VLMs 

**Authors**: Gyutaek Oh, Seoyeon Kim, Sangjoon Park, Byung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.13102)  

**Abstract**: Test-time scaling has recently emerged as a promising approach for enhancing the reasoning capabilities of large language models or vision-language models during inference. Although a variety of test-time scaling strategies have been proposed, and interest in their application to the medical domain is growing, many critical aspects remain underexplored, including their effectiveness for vision-language models and the identification of optimal strategies for different settings. In this paper, we conduct a comprehensive investigation of test-time scaling in the medical domain. We evaluate its impact on both large language models and vision-language models, considering factors such as model size, inherent model characteristics, and task complexity. Finally, we assess the robustness of these strategies under user-driven factors, such as misleading information embedded in prompts. Our findings offer practical guidelines for the effective use of test-time scaling in medical applications and provide insights into how these strategies can be further refined to meet the reliability and interpretability demands of the medical domain. 

---
# CHILL at SemEval-2025 Task 2: You Can't Just Throw Entities and Hope -- Make Your LLM to Get Them Right 

**Authors**: Jaebok Lee, Yonghyun Ryu, Seongmin Park, Yoonjung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13070)  

**Abstract**: In this paper, we describe our approach for the SemEval 2025 Task 2 on Entity-Aware Machine Translation (EA-MT). Our system aims to improve the accuracy of translating named entities by combining two key approaches: Retrieval Augmented Generation (RAG) and iterative self-refinement techniques using Large Language Models (LLMs). A distinctive feature of our system is its self-evaluation mechanism, where the LLM assesses its own translations based on two key criteria: the accuracy of entity translations and overall translation quality. We demonstrate how these methods work together and effectively improve entity handling while maintaining high-quality translations. 

---
# FinLMM-R1: Enhancing Financial Reasoning in LMM through Scalable Data and Reward Design 

**Authors**: Kai Lan, Jiayong Zhu, Jiangtong Li, Dawei Cheng, Guang Chen, Changjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13066)  

**Abstract**: Large Multimodal Models (LMMs) demonstrate significant cross-modal reasoning capabilities. However, financial applications face challenges due to the lack of high-quality multimodal reasoning datasets and the inefficiency of existing training paradigms for reasoning enhancement. To address these issues, we propose an integrated framework, FinLMM-R1, combining an automated and scalable pipeline for data construction with enhanced training strategies to improve the multimodal reasoning of LMM. The Automated and Scalable Pipeline (ASP) resolves textual-visual misalignment in financial reports through a separate paradigm of question-answer generation and image-question alignment, ensuring data integrity and extraction efficiency. Through ASP, we collect 89,378 aligned image-question pairs from 23,397 financial reports, covering tasks such as arithmetic reasoning, statistics reasoning, financial explanation, and financial knowledge. Moreover, we introduce the Thinking with Adversarial Reward in LMM (TAR-LMM), extending the prior two-stage training framework [1] with additional reward mechanisms. In the first stage, we focus on text-only tasks with format and accuracy rewards to guide the model in generating well-structured thinking contents. In the second stage, we construct multi-image contrastive samples with additional reward components including image selection, thinking content length, and adversarial reward to jointly optimize the LMM across visual perception, reasoning efficiency, and logical coherence. Extensive experiments on 7 benchmarks show ASP-derived dataset and training framework significantly improve answer accuracy and reasoning depth over existing reasoning LMMs in both general and financial multimodal contexts. 

---
# MotiveBench: How Far Are We From Human-Like Motivational Reasoning in Large Language Models? 

**Authors**: Xixian Yong, Jianxun Lian, Xiaoyuan Yi, Xiao Zhou, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.13065)  

**Abstract**: Large language models (LLMs) have been widely adopted as the core of agent frameworks in various scenarios, such as social simulations and AI companions. However, the extent to which they can replicate human-like motivations remains an underexplored question. Existing benchmarks are constrained by simplistic scenarios and the absence of character identities, resulting in an information asymmetry with real-world situations. To address this gap, we propose MotiveBench, which consists of 200 rich contextual scenarios and 600 reasoning tasks covering multiple levels of motivation. Using MotiveBench, we conduct extensive experiments on seven popular model families, comparing different scales and versions within each family. The results show that even the most advanced LLMs still fall short in achieving human-like motivational reasoning. Our analysis reveals key findings, including the difficulty LLMs face in reasoning about "love & belonging" motivations and their tendency toward excessive rationality and idealism. These insights highlight a promising direction for future research on the humanization of LLMs. The dataset, benchmark, and code are available at this https URL. 

---
# Multipole Attention for Efficient Long Context Reasoning 

**Authors**: Coleman Hooper, Sebastian Zhao, Luca Manolache, Sehoon Kim, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2506.13059)  

**Abstract**: Large Reasoning Models (LRMs) have shown promising accuracy improvements on complex problem-solving tasks. While these models have attained high accuracy by leveraging additional computation at test time, they need to generate long chain-of-thought reasoning in order to think before answering, which requires generating thousands of tokens. While sparse attention methods can help reduce the KV cache pressure induced by this long autoregressive reasoning, these methods can introduce errors which disrupt the reasoning process. Additionally, prior methods often pre-process the input to make it easier to identify the important prompt tokens when computing attention during generation, and this pre-processing is challenging to perform online for newly generated reasoning tokens. Our work addresses these challenges by introducing Multipole Attention, which accelerates autoregressive reasoning by only computing exact attention for the most important tokens, while maintaining approximate representations for the remaining tokens. Our method first performs clustering to group together semantically similar key vectors, and then uses the cluster centroids both to identify important key vectors and to approximate the remaining key vectors in order to retain high accuracy. We design a fast cluster update process to quickly re-cluster the input and previously generated tokens, thereby allowing for accelerating attention to the previous output tokens. We evaluate our method using emerging LRMs such as Qwen-8B, demonstrating that our approach can maintain accuracy on complex reasoning tasks even with aggressive attention sparsity settings. We also provide kernel implementations to demonstrate the practical efficiency gains from our method, achieving up to 4.5$\times$ speedup for attention in long-context reasoning applications. Our code is available at this https URL. 

---
# CFBenchmark-MM: Chinese Financial Assistant Benchmark for Multimodal Large Language Model 

**Authors**: Jiangtong Li, Yiyun Zhu, Dawei Cheng, Zhijun Ding, Changjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13055)  

**Abstract**: Multimodal Large Language Models (MLLMs) have rapidly evolved with the growth of Large Language Models (LLMs) and are now applied in various fields. In finance, the integration of diverse modalities such as text, charts, and tables is crucial for accurate and efficient decision-making. Therefore, an effective evaluation system that incorporates these data types is essential for advancing financial application. In this paper, we introduce CFBenchmark-MM, a Chinese multimodal financial benchmark with over 9,000 image-question pairs featuring tables, histogram charts, line charts, pie charts, and structural diagrams. Additionally, we develop a staged evaluation system to assess MLLMs in handling multimodal information by providing different visual content step by step. Despite MLLMs having inherent financial knowledge, experimental results still show limited efficiency and robustness in handling multimodal financial context. Further analysis on incorrect responses reveals the misinterpretation of visual content and the misunderstanding of financial concepts are the primary issues. Our research validates the significant, yet underexploited, potential of MLLMs in financial analysis, highlighting the need for further development and domain-specific optimization to encourage the enhanced use in financial domain. 

---
# Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models 

**Authors**: Muhammad Reza Qorib, Junyi Li, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13044)  

**Abstract**: Large language models (LLMs) have demonstrated impressive translation capabilities even without being explicitly trained on parallel data. This remarkable property has led some to believe that parallel data is no longer necessary for building multilingual language models. While some attribute this to the emergent abilities of LLMs due to scale, recent work suggests that it is actually caused by incidental bilingual signals present in the training data. Various methods have been proposed to maximize the utility of parallel data to enhance the multilingual capabilities of multilingual encoder-based and encoder-decoder language models. However, some decoder-based LLMs opt to ignore parallel data instead. In this work, we conduct a systematic study on the impact of adding parallel data on LLMs' multilingual capabilities, focusing specifically on translation and multilingual common-sense reasoning. Through controlled experiments, we demonstrate that parallel data can significantly improve LLMs' multilingual capabilities. 

---
# Edeflip: Supervised Word Translation between English and Yoruba 

**Authors**: Ikeoluwa Abioye, Jiani Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.13020)  

**Abstract**: In recent years, embedding alignment has become the state-of-the-art machine translation approach, as it can yield high-quality translation without training on parallel corpora. However, existing research and application of embedding alignment mostly focus on high-resource languages with high-quality monolingual embeddings. It is unclear if and how low-resource languages may be similarly benefited. In this study, we implement an established supervised embedding alignment method for word translation from English to Yoruba, the latter a low-resource language. We found that higher embedding quality and normalizing embeddings increase word translation precision, with, additionally, an interaction effect between the two. Our results demonstrate the limitations of the state-of-the-art supervised embedding alignment when it comes to low-resource languages, for which there are additional factors that need to be taken into consideration, such as the importance of curating high-quality monolingual embeddings. We hope our work will be a starting point for further machine translation research that takes into account the challenges that low-resource languages face. 

---
# Missing the human touch? A computational stylometry analysis of GPT-4 translations of online Chinese literature 

**Authors**: Xiaofang Yao, Yong-Bin Kang, Anthony McCosker  

**Link**: [PDF](https://arxiv.org/pdf/2506.13013)  

**Abstract**: Existing research indicates that machine translations (MTs) of literary texts are often unsatisfactory. MTs are typically evaluated using automated metrics and subjective human ratings, with limited focus on stylistic features. Evidence is also limited on whether state-of-the-art large language models (LLMs) will reshape literary translation. This study examines the stylistic features of LLM translations, comparing GPT-4's performance to human translations in a Chinese online literature task. Computational stylometry analysis shows that GPT-4 translations closely align with human translations in lexical, syntactic, and content features, suggesting that LLMs might replicate the 'human touch' in literary translation style. These findings offer insights into AI's impact on literary translation from a posthuman perspective, where distinctions between machine and human translations become increasingly blurry. 

---
# Large Language Models Enhanced by Plug and Play Syntactic Knowledge for Aspect-based Sentiment Analysis 

**Authors**: Yuanhe Tian, Xu Li, Wei Wang, Guoqing Jin, Pengsen Cheng, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.12991)  

**Abstract**: Aspect-based sentiment analysis (ABSA) generally requires a deep understanding of the contextual information, including the words associated with the aspect terms and their syntactic dependencies. Most existing studies employ advanced encoders (e.g., pre-trained models) to capture such context, especially large language models (LLMs). However, training these encoders is resource-intensive, and in many cases, the available data is insufficient for necessary fine-tuning. Therefore it is challenging for learning LLMs within such restricted environments and computation efficiency requirement. As a result, it motivates the exploration of plug-and-play methods that adapt LLMs to ABSA with minimal effort. In this paper, we propose an approach that integrates extendable components capable of incorporating various types of syntactic knowledge, such as constituent syntax, word dependencies, and combinatory categorial grammar (CCG). Specifically, we propose a memory module that records syntactic information and is incorporated into LLMs to instruct the prediction of sentiment polarities. Importantly, this encoder acts as a versatile, detachable plugin that is trained independently of the LLM. We conduct experiments on benchmark datasets, which show that our approach outperforms strong baselines and previous approaches, thus demonstrates its effectiveness. 

---
# Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs: a case study in Framing Bias Mitigation 

**Authors**: Yuanyuan Lei, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12978)  

**Abstract**: Media outlets are becoming more partisan and polarized nowadays. Most previous work focused on detecting media bias. In this paper, we aim to mitigate media bias by generating a neutralized summary given multiple articles presenting different ideological views. Motivated by the critical role of events and event relations in media bias detection, we propose to increase awareness of bias in LLMs via multi-document events reasoning and use a multi-document event relation graph to guide the summarization process. This graph contains rich event information useful to reveal bias: four common types of in-doc event relations to reflect content framing bias, cross-doc event coreference relation to reveal content selection bias, and event-level moral opinions to highlight opinionated framing bias. We further develop two strategies to incorporate the multi-document event relation graph for neutralized summarization. Firstly, we convert a graph into natural language descriptions and feed the textualized graph into LLMs as a part of a hard text prompt. Secondly, we encode the graph with graph attention network and insert the graph embedding into LLMs as a soft prompt. Both automatic evaluation and human evaluation confirm that our approach effectively mitigates both lexical and informational media bias, and meanwhile improves content preservation. 

---
# Assessing the Role of Data Quality in Training Bilingual Language Models 

**Authors**: Skyler Seto, Maartje ter Hoeve, Maureen de Seyssel, David Grangier  

**Link**: [PDF](https://arxiv.org/pdf/2506.12966)  

**Abstract**: Bilingual and multilingual language models offer a promising path toward scaling NLP systems across diverse languages and users. However, their performance often varies wildly between languages as prior works show that adding more languages can degrade performance for some languages (such as English), while improving others (typically more data constrained languages). In this work, we investigate causes of these inconsistencies by comparing bilingual and monolingual language models. Our analysis reveals that unequal data quality, not just data quantity, is a major driver of performance degradation in bilingual settings. We propose a simple yet effective data filtering strategy to select higher-quality bilingual training data with only high quality English data. Applied to French, German, and Chinese, our approach improves monolingual performance by 2-4% and reduces bilingual model performance gaps to 1%. These results highlight the overlooked importance of data quality in multilingual pretraining and offer a practical recipe for balancing performance. 

---
# CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation 

**Authors**: Naihao Deng, Kapotaksha Das, Rada Mihalcea, Vitaliy Popov, Mohamed Abouelenien  

**Link**: [PDF](https://arxiv.org/pdf/2506.12936)  

**Abstract**: In clinical operations, teamwork can be the crucial factor that determines the final outcome. Prior studies have shown that sufficient collaboration is the key factor that determines the outcome of an operation. To understand how the team practices teamwork during the operation, we collected CliniDial from simulations of medical operations. CliniDial includes the audio data and its transcriptions, the simulated physiology signals of the patient manikins, and how the team operates from two camera angles. We annotate behavior codes following an existing framework to understand the teamwork process for CliniDial. We pinpoint three main characteristics of our dataset, including its label imbalances, rich and natural interactions, and multiple modalities, and conduct experiments to test existing LLMs' capabilities on handling data with these characteristics. Experimental results show that CliniDial poses significant challenges to the existing models, inviting future effort on developing methods that can deal with real-world clinical data. We open-source the codebase at this https URL 

---
# SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models 

**Authors**: Xingjian Diao, Chunhui Zhang, Keyi Kong, Weiyi Wu, Chiyu Ma, Zhongyu Ouyang, Peijun Qing, Soroush Vosoughi, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2506.12935)  

**Abstract**: While large language models have shown reasoning capabilities, their application to the audio modality, particularly in large audio-language models (ALMs), remains significantly underdeveloped. Addressing this gap requires a systematic approach, involving a capable base model, high-quality reasoning-oriented audio data, and effective training algorithms. In this study, we present a comprehensive solution: we introduce the Audio Logical Reasoning (ALR) dataset, consisting of 6,446 text-audio annotated samples specifically designed for complex reasoning tasks. Building on this resource, we propose SoundMind, a rule-based reinforcement learning (RL) algorithm tailored to endow ALMs with deep bimodal reasoning abilities. By training Qwen2.5-Omni-7B on the ALR dataset using SoundMind, our approach achieves state-of-the-art performance in audio logical reasoning. This work highlights the impact of combining high-quality, reasoning-focused datasets with specialized RL techniques, advancing the frontier of auditory intelligence in language models. Our code and the proposed dataset are available at this https URL. 

---
# PersonaFeedback: A Large-scale Human-annotated Benchmark For Personalization 

**Authors**: Meiling Tao, Chenghao Zhu, Dongyi Ding, Tiannan Wang, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12915)  

**Abstract**: With the rapid improvement in the general capabilities of LLMs, LLM personalization, i.e., how to build LLM systems that can generate personalized responses or services that are tailored to distinct user personas, has become an increasingly important research and engineering problem. However, unlike many new challenging benchmarks being released for evaluating the general/reasoning capabilities, the lack of high-quality benchmarks for evaluating LLM personalization greatly hinders progress in this field. To address this, we introduce PersonaFeedback, a new benchmark that directly evaluates LLMs' ability to provide personalized responses given pre-defined user personas and queries. Unlike existing benchmarks that require models to infer implicit user personas from historical interactions, PersonaFeedback decouples persona inference from personalization, focusing on evaluating the model's ability to generate responses tailored to explicit personas. PersonaFeedback consists of 8298 human-annotated test cases, which are categorized into easy, medium, and hard tiers based on the contextual complexity of the user personas and the difficulty in distinguishing subtle differences between two personalized responses. We conduct comprehensive evaluations across a wide range of models. The empirical results reveal that even state-of-the-art LLMs that can solve complex real-world reasoning tasks could fall short on the hard tier of PersonaFeedback where even human evaluators may find the distinctions challenging. Furthermore, we conduct an in-depth analysis of failure modes across various types of systems, demonstrating that the current retrieval-augmented framework should not be seen as a de facto solution for personalization tasks. All benchmark data, annotation protocols, and the evaluation pipeline will be publicly available to facilitate future research on LLM personalization. 

---
# SciDA: Scientific Dynamic Assessor of LLMs 

**Authors**: Junting Zhou, Tingjia Miao, Yiyan Liao, Qichao Wang, Zhoufutu Wen, Yanqin Wang, Yunjie Huang, Ge Yan, Leqi Wang, Yucheng Xia, Hongwan Gao, Yuansong Zeng, Renjie Zheng, Chen Dun, Yitao Liang, Tong Yang, Wenhao Huang, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12909)  

**Abstract**: Advancement in Large Language Models (LLMs) reasoning capabilities enables them to solve scientific problems with enhanced efficacy. Thereby, a high-quality benchmark for comprehensive and appropriate assessment holds significance, while existing ones either confront the risk of data contamination or lack involved disciplines. To be specific, due to the data source overlap of LLMs training and static benchmark, the keys or number pattern of answers inadvertently memorized (i.e. data contamination), leading to systematic overestimation of their reasoning capabilities, especially numerical reasoning. We propose SciDA, a multidisciplinary benchmark that consists exclusively of over 1k Olympic-level numerical computation problems, allowing randomized numerical initializations for each inference round to avoid reliance on fixed numerical patterns. We conduct a series of experiments with both closed-source and open-source top-performing LLMs, and it is observed that the performance of LLMs drop significantly under random numerical initialization. Thus, we provide truthful and unbiased assessments of the numerical reasoning capabilities of LLMs. The data is available at this https URL 

---
# JEBS: A Fine-grained Biomedical Lexical Simplification Task 

**Authors**: William Xia, Ishita Unde, Brian Ondov, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2506.12898)  

**Abstract**: Online medical literature has made health information more available than ever, however, the barrier of complex medical jargon prevents the general public from understanding it. Though parallel and comparable corpora for Biomedical Text Simplification have been introduced, these conflate the many syntactic and lexical operations involved in simplification. To enable more targeted development and evaluation, we present a fine-grained lexical simplification task and dataset, Jargon Explanations for Biomedical Simplification (JEBS, this https URL ). The JEBS task involves identifying complex terms, classifying how to replace them, and generating replacement text. The JEBS dataset contains 21,595 replacements for 10,314 terms across 400 biomedical abstracts and their manually simplified versions. Additionally, we provide baseline results for a variety of rule-based and transformer-based systems for the three sub-tasks. The JEBS task, data, and baseline results pave the way for development and rigorous evaluation of systems for replacing or explaining complex biomedical terms. 

---
# Assessing the Performance Gap Between Lexical and Semantic Models for Information Retrieval With Formulaic Legal Language 

**Authors**: Larissa Mori, Carlos Sousa de Oliveira, Yuehwern Yih, Mario Ventresca  

**Link**: [PDF](https://arxiv.org/pdf/2506.12895)  

**Abstract**: Legal passage retrieval is an important task that assists legal practitioners in the time-intensive process of finding relevant precedents to support legal arguments. This study investigates the task of retrieving legal passages or paragraphs from decisions of the Court of Justice of the European Union (CJEU), whose language is highly structured and formulaic, leading to repetitive patterns. Understanding when lexical or semantic models are more effective at handling the repetitive nature of legal language is key to developing retrieval systems that are more accurate, efficient, and transparent for specific legal domains. To this end, we explore when this routinized legal language is better suited for retrieval using methods that rely on lexical and statistical features, such as BM25, or dense retrieval models trained to capture semantic and contextual information. A qualitative and quantitative analysis with three complementary metrics shows that both lexical and dense models perform well in scenarios with more repetitive usage of language, whereas BM25 performs better than the dense models in more nuanced scenarios where repetition and verbatim~quotes are less prevalent and in longer queries. Our experiments also show that BM25 is a strong baseline, surpassing off-the-shelf dense models in 4 out of 7 performance metrics. However, fine-tuning a dense model on domain-specific data led to improved performance, surpassing BM25 in most metrics, and we analyze the effect of the amount of data used in fine-tuning on the model's performance and temporal robustness. The code, dataset and appendix related to this work are available on: this https URL. 

---
# ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality 

**Authors**: Adrián Cuadrón, Aimar Sagasti, Maitane Urruela, Iker De la Iglesia, Ane G Domingo-Aldama, Aitziber Atutxa, Josu Goikoetxea, Ander Barrena  

**Link**: [PDF](https://arxiv.org/pdf/2506.12886)  

**Abstract**: This work presents three different approaches to address the ArchEHR-QA 2025 Shared Task on automated patient question answering. We introduce an end-to-end prompt-based baseline and two two-step methods to divide the task, without utilizing any external knowledge. Both two step approaches first extract essential sentences from the clinical text, by prompt or similarity ranking, and then generate the final answer from these notes. Results indicate that the re-ranker based two-step system performs best, highlighting the importance of selecting the right approach for each subtask. Our best run achieved an overall score of 0.44, ranking 8th out of 30 on the leaderboard, securing the top position in overall factuality. 

---
# QFFT, Question-Free Fine-Tuning for Adaptive Reasoning 

**Authors**: Wanlong Liu, Junxiao Xu, Fei Yu, Yukang Lin, Ke Ji, Wenyu Chen, Yan Xu, Yasheng Wang, Lifeng Shang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12860)  

**Abstract**: Recent advancements in Long Chain-of-Thought (CoT) reasoning models have improved performance on complex tasks, but they suffer from overthinking, which generates redundant reasoning steps, especially for simple questions. This paper revisits the reasoning patterns of Long and Short CoT models, observing that the Short CoT patterns offer concise reasoning efficiently, while the Long CoT patterns excel in challenging scenarios where the Short CoT patterns struggle. To enable models to leverage both patterns, we propose Question-Free Fine-Tuning (QFFT), a fine-tuning approach that removes the input question during training and learns exclusively from Long CoT responses. This approach enables the model to adaptively employ both reasoning patterns: it prioritizes the Short CoT patterns and activates the Long CoT patterns only when necessary. Experiments on various mathematical datasets demonstrate that QFFT reduces average response length by more than 50\%, while achieving performance comparable to Supervised Fine-Tuning (SFT). Additionally, QFFT exhibits superior performance compared to SFT in noisy, out-of-domain, and low-resource scenarios. 

---
# Transforming Chatbot Text: A Sequence-to-Sequence Approach 

**Authors**: Natesh Reddy, Mark Stamp  

**Link**: [PDF](https://arxiv.org/pdf/2506.12843)  

**Abstract**: Due to advances in Large Language Models (LLMs) such as ChatGPT, the boundary between human-written text and AI-generated text has become blurred. Nevertheless, recent work has demonstrated that it is possible to reliably detect GPT-generated text. In this paper, we adopt a novel strategy to adversarially transform GPT-generated text using sequence-to-sequence (Seq2Seq) models, with the goal of making the text more human-like. We experiment with the Seq2Seq models T5-small and BART which serve to modify GPT-generated sentences to include linguistic, structural, and semantic components that may be more typical of human-authored text. Experiments show that classification models trained to distinguish GPT-generated text are significantly less accurate when tested on text that has been modified by these Seq2Seq models. However, after retraining classification models on data generated by our Seq2Seq technique, the models are able to distinguish the transformed GPT-generated text from human-generated text with high accuracy. This work adds to the accumulating knowledge of text transformation as a tool for both attack -- in the sense of defeating classification models -- and defense -- in the sense of improved classifiers -- thereby advancing our understanding of AI-generated text. 

---
# Medical Argument Mining: Exploitation of Scarce Data Using NLI Systems 

**Authors**: Maitane Urruela, Sergio Martín, Iker De la Iglesia, Ander Barrena  

**Link**: [PDF](https://arxiv.org/pdf/2506.12823)  

**Abstract**: This work presents an Argument Mining process that extracts argumentative entities from clinical texts and identifies their relationships using token classification and Natural Language Inference techniques. Compared to straightforward methods like text classification, this methodology demonstrates superior performance in data-scarce settings. By assessing the effectiveness of these methods in identifying argumentative structures that support or refute possible diagnoses, this research lays the groundwork for future tools that can provide evidence-based justifications for machine-generated clinical conclusions. 

---
# Surprise Calibration for Better In-Context Learning 

**Authors**: Zhihang Tan, Jingrui Hou, Ping Wang, Qibiao Hu, Peng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12796)  

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm for task adaptation in large language models (LLMs), where models infer underlying task structures from a few demonstrations. However, ICL remains susceptible to biases that arise from prior knowledge and contextual demonstrations, which can degrade the performance of LLMs. Existing bias calibration methods typically apply fixed class priors across all inputs, limiting their efficacy in dynamic ICL settings where the context for each query differs. To address these limitations, we adopt implicit sequential Bayesian inference as a framework for interpreting ICL, identify "surprise" as an informative signal for class prior shift, and introduce a novel method--Surprise Calibration (SC). SC leverages the notion of surprise to capture the temporal dynamics of class priors, providing a more adaptive and computationally efficient solution for in-context learning. We empirically demonstrate the superiority of SC over existing bias calibration techniques across a range of benchmark natural language processing tasks. 

---
# Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models 

**Authors**: David Guzman Piedrahita, Irene Strauss, Bernhard Schölkopf, Rada Mihalcea, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.12758)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into everyday life and information ecosystems, concerns about their implicit biases continue to persist. While prior work has primarily examined socio-demographic and left--right political dimensions, little attention has been paid to how LLMs align with broader geopolitical value systems, particularly the democracy--authoritarianism spectrum. In this paper, we propose a novel methodology to assess such alignment, combining (1) the F-scale, a psychometric tool for measuring authoritarian tendencies, (2) FavScore, a newly introduced metric for evaluating model favorability toward world leaders, and (3) role-model probing to assess which figures are cited as general role-models by LLMs. We find that LLMs generally favor democratic values and leaders, but exhibit increases favorability toward authoritarian figures when prompted in Mandarin. Further, models are found to often cite authoritarian figures as role models, even outside explicit political contexts. These results shed light on ways LLMs may reflect and potentially reinforce global political ideologies, highlighting the importance of evaluating bias beyond conventional socio-political axes. Our code is available at: this https URL 

---
# Rethinking Hate Speech Detection on Social Media: Can LLMs Replace Traditional Models? 

**Authors**: Daman Deep Singh, Ramanuj Bhattacharjee, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2506.12744)  

**Abstract**: Hate speech detection across contemporary social media presents unique challenges due to linguistic diversity and the informal nature of online discourse. These challenges are further amplified in settings involving code-mixing, transliteration, and culturally nuanced expressions. While fine-tuned transformer models, such as BERT, have become standard for this task, we argue that recent large language models (LLMs) not only surpass them but also redefine the landscape of hate speech detection more broadly. To support this claim, we introduce IndoHateMix, a diverse, high-quality dataset capturing Hindi-English code-mixing and transliteration in the Indian context, providing a realistic benchmark to evaluate model robustness in complex multilingual scenarios where existing NLP methods often struggle. Our extensive experiments show that cutting-edge LLMs (such as LLaMA-3.1) consistently outperform task-specific BERT-based models, even when fine-tuned on significantly less data. With their superior generalization and adaptability, LLMs offer a transformative approach to mitigating online hate in diverse environments. This raises the question of whether future works should prioritize developing specialized models or focus on curating richer and more varied datasets to further enhance the effectiveness of LLMs. 

---
# Flexible Realignment of Language Models 

**Authors**: Wenhong Zhu, Ruobing Xie, Weinan Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12704)  

**Abstract**: Realignment becomes necessary when a language model (LM) fails to meet expected performance. We propose a flexible realignment framework that supports quantitative control of alignment degree during training and inference. This framework incorporates Training-time Realignment (TrRa), which efficiently realigns the reference model by leveraging the controllable fusion of logits from both the reference and already aligned models. For example, TrRa reduces token usage by 54.63% on DeepSeek-R1-Distill-Qwen-1.5B without any performance degradation, outperforming DeepScaleR-1.5B's 33.86%. To complement TrRa during inference, we introduce a layer adapter that enables smooth Inference-time Realignment (InRa). This adapter is initialized to perform an identity transformation at the bottom layer and is inserted preceding the original layers. During inference, input embeddings are simultaneously processed by the adapter and the original layer, followed by the remaining layers, and then controllably interpolated at the logit level. We upgraded DeepSeek-R1-Distill-Qwen-7B from a slow-thinking model to one that supports both fast and slow thinking, allowing flexible alignment control even during inference. By encouraging deeper reasoning, it even surpassed its original performance. 

---
# Enhancing Clinical Models with Pseudo Data for De-identification 

**Authors**: Paul Landes, Aaron J Chaise, Tarak Nath Nandi, Ravi K Madduri  

**Link**: [PDF](https://arxiv.org/pdf/2506.12674)  

**Abstract**: Many models are pretrained on redacted text for privacy reasons. Clinical foundation models are often trained on de-identified text, which uses special syntax (masked) text in place of protected health information. Even though these models have increased in popularity, there has been little effort in understanding the effects of training them on redacted text. In this work, we pretrain several encoder-only models on a dataset that contains redacted text and a version with replaced realistic pseudo text. We then fine-tuned models for the protected health information de-identification task and show how our methods significantly outperform previous baselines. The contributions of this work include: a) our novel, and yet surprising findings with training recommendations, b) redacted text replacements used to produce the pseudo dataset, c) pretrained embeddings and fine-tuned task specific models, and d) freely available pseudo training dataset generation and model source code used in our experiments. 

---
# Synthetic Socratic Debates: Examining Persona Effects on Moral Decision and Persuasion Dynamics 

**Authors**: Jiarui Liu, Yueqi Song, Yunze Xiao, Mingqian Zheng, Lindia Tjuatja, Jana Schaich Borg, Mona Diab, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2506.12657)  

**Abstract**: As large language models (LLMs) are increasingly used in morally sensitive domains, it is crucial to understand how persona traits affect their moral reasoning and persuasive behavior. We present the first large-scale study of multi-dimensional persona effects in AI-AI debates over real-world moral dilemmas. Using a 6-dimensional persona space (age, gender, country, class, ideology, and personality), we simulate structured debates between AI agents over 131 relationship-based cases. Our results show that personas affect initial moral stances and debate outcomes, with political ideology and personality traits exerting the strongest influence. Persuasive success varies across traits, with liberal and open personalities reaching higher consensus and win rates. While logit-based confidence grows during debates, emotional and credibility-based appeals diminish, indicating more tempered argumentation over time. These trends mirror findings from psychology and cultural studies, reinforcing the need for persona-aware evaluation frameworks for AI moral reasoning. 

---
# How Grounded is Wikipedia? A Study on Structured Evidential Support 

**Authors**: William Walden, Kathryn Ricci, Miriam Wanner, Zhengping Jiang, Chandler May, Rongkun Zhou, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2506.12637)  

**Abstract**: Wikipedia is a critical resource for modern NLP, serving as a rich repository of up-to-date and citation-backed information on a wide variety of subjects. The reliability of Wikipedia -- its groundedness in its cited sources -- is vital to this purpose. This work provides a quantitative analysis of the extent to which Wikipedia *is* so grounded and of how readily grounding evidence may be retrieved. To this end, we introduce PeopleProfiles -- a large-scale, multi-level dataset of claim support annotations on Wikipedia articles of notable people. We show that roughly 20% of claims in Wikipedia *lead* sections are unsupported by the article body; roughly 27% of annotated claims in the article *body* are unsupported by their (publicly accessible) cited sources; and >80% of lead claims cannot be traced to these sources via annotated body evidence. Further, we show that recovery of complex grounding evidence for claims that *are* supported remains a challenge for standard retrieval methods. 

---
# Between Predictability and Randomness: Seeking Artistic Inspiration from AI Generative Models 

**Authors**: Olga Vechtomova  

**Link**: [PDF](https://arxiv.org/pdf/2506.12634)  

**Abstract**: Artistic inspiration often emerges from language that is open to interpretation. This paper explores the use of AI-generated poetic lines as stimuli for creativity. Through analysis of two generative AI approaches--lines generated by Long Short-Term Memory Variational Autoencoders (LSTM-VAE) and complete poems by Large Language Models (LLMs)--I demonstrate that LSTM-VAE lines achieve their evocative impact through a combination of resonant imagery and productive indeterminacy. While LLMs produce technically accomplished poetry with conventional patterns, LSTM-VAE lines can engage the artist through semantic openness, unconventional combinations, and fragments that resist closure. Through the composition of an original poem, where narrative emerged organically through engagement with LSTM-VAE generated lines rather than following a predetermined structure, I demonstrate how these characteristics can serve as evocative starting points for authentic artistic expression. 

---
# OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics 

**Authors**: Vineeth Dorna, Anmol Mekala, Wenlong Zhao, Andrew McCallum, Zachary C. Lipton, J. Zico Kolter, Pratyush Maini  

**Link**: [PDF](https://arxiv.org/pdf/2506.12618)  

**Abstract**: Robust unlearning is crucial for safely deploying large language models (LLMs) in environments where data privacy, model safety, and regulatory compliance must be ensured. Yet the task is inherently challenging, partly due to difficulties in reliably measuring whether unlearning has truly occurred. Moreover, fragmentation in current methodologies and inconsistent evaluation metrics hinder comparative analysis and reproducibility. To unify and accelerate research efforts, we introduce OpenUnlearning, a standardized and extensible framework designed explicitly for benchmarking both LLM unlearning methods and metrics. OpenUnlearning integrates 9 unlearning algorithms and 16 diverse evaluations across 3 leading benchmarks (TOFU, MUSE, and WMDP) and also enables analyses of forgetting behaviors across 450+ checkpoints we publicly release. Leveraging OpenUnlearning, we propose a novel meta-evaluation benchmark focused specifically on assessing the faithfulness and robustness of evaluation metrics themselves. We also benchmark diverse unlearning methods and provide a comparative analysis against an extensive evaluation suite. Overall, we establish a clear, community-driven pathway toward rigorous development in LLM unlearning research. 

---
# Konooz: Multi-domain Multi-dialect Corpus for Named Entity Recognition 

**Authors**: Nagham Hamad, Mohammed Khalilia, Mustafa Jarrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12615)  

**Abstract**: We introduce Konooz, a novel multi-dimensional corpus covering 16 Arabic dialects across 10 domains, resulting in 160 distinct corpora. The corpus comprises about 777k tokens, carefully collected and manually annotated with 21 entity types using both nested and flat annotation schemes - using the Wojood guidelines. While Konooz is useful for various NLP tasks like domain adaptation and transfer learning, this paper primarily focuses on benchmarking existing Arabic Named Entity Recognition (NER) models, especially cross-domain and cross-dialect model performance. Our benchmarking of four Arabic NER models using Konooz reveals a significant drop in performance of up to 38% when compared to the in-distribution data. Furthermore, we present an in-depth analysis of domain and dialect divergence and the impact of resource scarcity. We also measured the overlap between domains and dialects using the Maximum Mean Discrepancy (MMD) metric, and illustrated why certain NER models perform better on specific dialects and domains. Konooz is open-source and publicly available at this https URL 

---
# Towards Building General Purpose Embedding Models for Industry 4.0 Agents 

**Authors**: Christodoulos Constantinides, Shuxin Lin, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2506.12607)  

**Abstract**: In this work we focus on improving language models' understanding for asset maintenance to guide the engineer's decisions and minimize asset downtime. Given a set of tasks expressed in natural language for Industry 4.0 domain, each associated with queries related to a specific asset, we want to recommend relevant items and generalize to queries of similar assets. A task may involve identifying relevant sensors given a query about an asset's failure mode.
Our approach begins with gathering a qualitative, expert-vetted knowledge base to construct nine asset-specific task datasets. To create more contextually informed embeddings, we augment the input tasks using Large Language Models (LLMs), providing concise descriptions of the entities involved in the queries. This embedding model is then integrated with a Reasoning and Acting agent (ReAct), which serves as a powerful tool for answering complex user queries that require multi-step reasoning, planning, and knowledge inference.
Through ablation studies, we demonstrate that: (a) LLM query augmentation improves the quality of embeddings, (b) Contrastive loss and other methods that avoid in-batch negatives are superior for datasets with queries related to many items, and (c) It is crucial to balance positive and negative in-batch samples. After training and testing on our dataset, we observe a substantial improvement: HIT@1 increases by +54.2%, MAP@100 by +50.1%, and NDCG@10 by +54.7%, averaged across all tasks and models. Additionally, we empirically demonstrate the model's planning and tool invocation capabilities when answering complex questions related to industrial asset maintenance, showcasing its effectiveness in supporting Subject Matter Experts (SMEs) in their day-to-day operations. 

---
# An Exploration of Mamba for Speech Self-Supervised Models 

**Authors**: Tzu-Quan Lin, Heng-Cheng Kuo, Tzu-Chieh Wei, Hsi-Chun Cheng, Chun-Wei Chen, Hsien-Fu Hsiao, Yu Tsao, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.12606)  

**Abstract**: While Mamba has demonstrated strong performance in language modeling, its potential as a speech self-supervised (SSL) model remains underexplored, with prior studies limited to isolated tasks. To address this, we explore Mamba-based HuBERT models as alternatives to Transformer-based SSL architectures. Leveraging the linear-time Selective State Space, these models enable fine-tuning on long-context ASR with significantly lower compute. Moreover, they show superior performance when fine-tuned for streaming ASR. Beyond fine-tuning, these models show competitive performance on SUPERB probing benchmarks, particularly in causal settings. Our analysis shows that they yield higher-quality quantized representations and capture speaker-related features more distinctly than Transformer-based models. These findings highlight Mamba-based SSL as a promising and complementary direction for long-sequence modeling, real-time speech modeling, and speech unit extraction. 

---
# OneEval: Benchmarking LLM Knowledge-intensive Reasoning over Diverse Knowledge Bases 

**Authors**: Yongrui Chen, Zhiqiang Liu, Jing Yu, Lin Ren, Nan Hu, Xinbang Dai, Jiajun Liu, Jiazhen Kang, Shenyu Zhang, Xinda Wang, Keyan Ding, Pengfei Shen, Haolei Zhu, Hongjie Deng, Yisong Wang, Tongtong Wu, Sheng Bi, Wen Zhang, Tianxing Wu, Qiu Ji, Haofen Wang, Wenliang Chen, Huajun Chen, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12577)  

**Abstract**: Large Language Models (LLMs) have demonstrated substantial progress on reasoning tasks involving unstructured text, yet their capabilities significantly deteriorate when reasoning requires integrating structured external knowledge such as knowledge graphs, code snippets, or formal logic. This limitation is partly due to the absence of benchmarks capable of systematically evaluating LLM performance across diverse structured knowledge modalities. To address this gap, we introduce \textbf{\textsc{OneEval}}, a comprehensive benchmark explicitly designed to assess the knowledge-intensive reasoning capabilities of LLMs across four structured knowledge modalities, unstructured text, knowledge graphs, code, and formal logic, and five critical domains (general knowledge, government, science, law, and programming). \textsc{OneEval} comprises 4,019 carefully curated instances and includes a challenging subset, \textsc{OneEval}\textsubscript{Hard}, consisting of 1,285 particularly difficult cases. Through extensive evaluation of 18 state-of-the-art open-source and proprietary LLMs, we establish three core findings: a) \emph{persistent limitations in structured reasoning}, with even the strongest model achieving only 32.2\% accuracy on \textsc{OneEval}\textsubscript{Hard}; b) \emph{performance consistently declines as the structural complexity of the knowledge base increases}, with accuracy dropping sharply from 53\% (textual reasoning) to 25\% (formal logic); and c) \emph{diminishing returns from extended reasoning chains}, highlighting the critical need for models to adapt reasoning depth appropriately to task complexity. We release the \textsc{OneEval} datasets, evaluation scripts, and baseline results publicly, accompanied by a leaderboard to facilitate ongoing advancements in structured knowledge reasoning. 

---
# Enabling Precise Topic Alignment in Large Language Models Via Sparse Autoencoders 

**Authors**: Ananya Joshi, Celia Cintas, Skyler Speakman  

**Link**: [PDF](https://arxiv.org/pdf/2506.12576)  

**Abstract**: Recent work shows that Sparse Autoencoders (SAE) applied to large language model (LLM) layers have neurons corresponding to interpretable concepts. These SAE neurons can be modified to align generated outputs, but only towards pre-identified topics and with some parameter tuning. Our approach leverages the observational and modification properties of SAEs to enable alignment for any topic. This method 1) scores each SAE neuron by its semantic similarity to an alignment text and uses them to 2) modify SAE-layer-level outputs by emphasizing topic-aligned neurons. We assess the alignment capabilities of this approach on diverse public topic datasets including Amazon reviews, Medicine, and Sycophancy, across the currently available open-source LLMs and SAE pairs (GPT2 and Gemma) with multiple SAEs configurations. Experiments aligning to medical prompts reveal several benefits over fine-tuning, including increased average language acceptability (0.25 vs. 0.5), reduced training time across multiple alignment topics (333.6s vs. 62s), and acceptable inference time for many applications (+0.00092s/token). Our open-source code is available at this http URL. 

---
# Overview of the NLPCC 2025 Shared Task: Gender Bias Mitigation Challenge 

**Authors**: Yizhi Li, Ge Zhang, Hanhua Hong, Yiwen Wang, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.12574)  

**Abstract**: As natural language processing for gender bias becomes a significant interdisciplinary topic, the prevalent data-driven techniques, such as pre-trained language models, suffer from biased corpus. This case becomes more obvious regarding those languages with less fairness-related computational linguistic resources, such as Chinese. To this end, we propose a Chinese cOrpus foR Gender bIas Probing and Mitigation (CORGI-PM), which contains 32.9k sentences with high-quality labels derived by following an annotation scheme specifically developed for gender bias in the Chinese context. It is worth noting that CORGI-PM contains 5.2k gender-biased sentences along with the corresponding bias-eliminated versions rewritten by human annotators. We pose three challenges as a shared task to automate the mitigation of textual gender bias, which requires the models to detect, classify, and mitigate textual gender bias. In the literature, we present the results and analysis for the teams participating this shared task in NLPCC 2025. 

---
# DoTA-RAG: Dynamic of Thought Aggregation RAG 

**Authors**: Saksorn Ruangtanusak, Natthapath Rungseesiripak, Peerawat Rojratchadakorn, Monthol Charattrakool, Natapong Nitarach  

**Link**: [PDF](https://arxiv.org/pdf/2506.12571)  

**Abstract**: In this paper, we introduce DoTA-RAG (Dynamic-of-Thought Aggregation RAG), a retrieval-augmented generation system optimized for high-throughput, large-scale web knowledge indexes. Traditional RAG pipelines often suffer from high latency and limited accuracy over massive, diverse datasets. DoTA-RAG addresses these challenges with a three-stage pipeline: query rewriting, dynamic routing to specialized sub-indexes, and multi-stage retrieval and ranking. We further enhance retrieval by evaluating and selecting a superior embedding model, re-embedding the large FineWeb-10BT corpus. Moreover, we create a diverse Q&A dataset of 500 questions generated via the DataMorgana setup across a broad range of WebOrganizer topics and formats. DoTA-RAG improves the answer correctness score from 0.752 (baseline, using LiveRAG pre-built vector store) to 1.478 while maintaining low latency, and it achieves a 0.929 correctness score on the Live Challenge Day. These results highlight DoTA-RAG's potential for practical deployment in domains requiring fast, reliable access to large and evolving knowledge sources. 

---
# Profiling News Media for Factuality and Bias Using LLMs and the Fact-Checking Methodology of Human Experts 

**Authors**: Zain Muhammad Mujahid, Dilshod Azizov, Maha Tufail Agro, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12552)  

**Abstract**: In an age characterized by the proliferation of mis- and disinformation online, it is critical to empower readers to understand the content they are reading. Important efforts in this direction rely on manual or automatic fact-checking, which can be challenging for emerging claims with limited information. Such scenarios can be handled by assessing the reliability and the political bias of the source of the claim, i.e., characterizing entire news outlets rather than individual claims or articles. This is an important but understudied research direction. While prior work has looked into linguistic and social contexts, we do not analyze individual articles or information in social media. Instead, we propose a novel methodology that emulates the criteria that professional fact-checkers use to assess the factuality and political bias of an entire outlet. Specifically, we design a variety of prompts based on these criteria and elicit responses from large language models (LLMs), which we aggregate to make predictions. In addition to demonstrating sizable improvements over strong baselines via extensive experiments with multiple LLMs, we provide an in-depth error analysis of the effect of media popularity and region on model performance. Further, we conduct an ablation study to highlight the key components of our dataset that contribute to these improvements. To facilitate future research, we released our dataset and code at this https URL. 

---
# RealFactBench: A Benchmark for Evaluating Large Language Models in Real-World Fact-Checking 

**Authors**: Shuo Yang, Yuqin Dai, Guoqing Wang, Xinran Zheng, Jinfeng Xu, Jinze Li, Zhenzhe Ying, Weiqiang Wang, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12538)  

**Abstract**: Large Language Models (LLMs) hold significant potential for advancing fact-checking by leveraging their capabilities in reasoning, evidence retrieval, and explanation generation. However, existing benchmarks fail to comprehensively evaluate LLMs and Multimodal Large Language Models (MLLMs) in realistic misinformation scenarios. To bridge this gap, we introduce RealFactBench, a comprehensive benchmark designed to assess the fact-checking capabilities of LLMs and MLLMs across diverse real-world tasks, including Knowledge Validation, Rumor Detection, and Event Verification. RealFactBench consists of 6K high-quality claims drawn from authoritative sources, encompassing multimodal content and diverse domains. Our evaluation framework further introduces the Unknown Rate (UnR) metric, enabling a more nuanced assessment of models' ability to handle uncertainty and balance between over-conservatism and over-confidence. Extensive experiments on 7 representative LLMs and 4 MLLMs reveal their limitations in real-world fact-checking and offer valuable insights for further research. RealFactBench is publicly available at this https URL. 

---
# Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction 

**Authors**: Xiaoran Fan, Zhichao Sun, Yangfan Gao, Jingfei Xiong, Hang Yan, Yifei Cao, Jiajun Sun, Shuo Li, Zhihao Zhang, Zhiheng Xi, Yuhao Zhou, Senjie Jin, Changhao Jiang, Junjie Ye, Ming Zhang, Rui Zheng, Zhenhua Han, Yunke Zhang, Demei Yan, Shaokang Dong, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12537)  

**Abstract**: Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the impact of key components (i.e., speech tokenizers, speech heads, and speaker modeling) on the performance of LLM-centric SLMs. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency. 

---
# Detection, Classification, and Mitigation of Gender Bias in Large Language Models 

**Authors**: Xiaoqing Cheng, Hongying Zan, Lulu Kong, Jinwang Song, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.12527)  

**Abstract**: With the rapid development of large language models (LLMs), they have significantly improved efficiency across a wide range of domains. However, recent studies have revealed that LLMs often exhibit gender bias, leading to serious social implications. Detecting, classifying, and mitigating gender bias in LLMs has therefore become a critical research focus. In the NLPCC 2025 Shared Task 7: Chinese Corpus for Gender Bias Detection, Classification and Mitigation Challenge, we investigate how to enhance the capabilities of LLMs in gender bias detection, classification, and mitigation. We adopt reinforcement learning, chain-of-thoughts (CoT) reasoning, and supervised fine-tuning to handle different Subtasks. Specifically, for Subtasks 1 and 2, we leverage the internal reasoning capabilities of LLMs to guide multi-step thinking in a staged manner, which simplifies complex biased queries and improves response accuracy. For Subtask 3, we employ a reinforcement learning-based approach, annotating a preference dataset using GPT-4. We then apply Direct Preference Optimization (DPO) to mitigate gender bias by introducing a loss function that explicitly favors less biased completions over biased ones. Our approach ranked first across all three subtasks of the NLPCC 2025 Shared Task 7. 

---
# Towards Fairness Assessment of Dutch Hate Speech Detection 

**Authors**: Julie Bauer, Rishabh Kaushal, Thales Bertaglia, Adriana Iamnitchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12502)  

**Abstract**: Numerous studies have proposed computational methods to detect hate speech online, yet most focus on the English language and emphasize model development. In this study, we evaluate the counterfactual fairness of hate speech detection models in the Dutch language, specifically examining the performance and fairness of transformer-based models. We make the following key contributions. First, we curate a list of Dutch Social Group Terms that reflect social context. Second, we generate counterfactual data for Dutch hate speech using LLMs and established strategies like Manual Group Substitution (MGS) and Sentence Log-Likelihood (SLL). Through qualitative evaluation, we highlight the challenges of generating realistic counterfactuals, particularly with Dutch grammar and contextual coherence. Third, we fine-tune baseline transformer-based models with counterfactual data and evaluate their performance in detecting hate speech. Fourth, we assess the fairness of these models using Counterfactual Token Fairness (CTF) and group fairness metrics, including equality of odds and demographic parity. Our analysis shows that models perform better in terms of hate speech detection, average counterfactual fairness and group fairness. This work addresses a significant gap in the literature on counterfactual fairness for hate speech detection in Dutch and provides practical insights and recommendations for improving both model performance and fairness. 

---
# Improving Factuality for Dialogue Response Generation via Graph-Based Knowledge Augmentation 

**Authors**: Xiangyan Chen, Yujian Gan, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2506.12496)  

**Abstract**: Large Language Models (LLMs) succeed in many natural language processing tasks. However, their tendency to hallucinate - generate plausible but inconsistent or factually incorrect text - can cause problems in certain tasks, including response generation in dialogue. To mitigate this issue, knowledge-augmented methods have shown promise in reducing hallucinations. Here, we introduce a novel framework designed to enhance the factuality of dialogue response generation, as well as an approach to evaluate dialogue factual accuracy. Our framework combines a knowledge triple retriever, a dialogue rewrite, and knowledge-enhanced response generation to produce more accurate and grounded dialogue responses. To further evaluate generated responses, we propose a revised fact score that addresses the limitations of existing fact-score methods in dialogue settings, providing a more reliable assessment of factual consistency. We evaluate our methods using different baselines on the OpendialKG and HybriDialogue datasets. Our methods significantly improve factuality compared to other graph knowledge-augmentation baselines, including the state-of-the-art G-retriever. The code will be released on GitHub. 

---
# FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation 

**Authors**: Zhuocheng Zhang, Yang Feng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12494)  

**Abstract**: Retrieval-Augmented Generation (RAG) plays a pivotal role in modern large language model applications, with numerous existing frameworks offering a wide range of functionalities to facilitate the development of RAG systems. However, we have identified several persistent challenges in these frameworks, including difficulties in algorithm reproduction and sharing, lack of new techniques, and high system overhead. To address these limitations, we introduce \textbf{FlexRAG}, an open-source framework specifically designed for research and prototyping. FlexRAG supports text-based, multimodal, and network-based RAG, providing comprehensive lifecycle support alongside efficient asynchronous processing and persistent caching capabilities. By offering a robust and flexible solution, FlexRAG enables researchers to rapidly develop, deploy, and share advanced RAG systems. Our toolkit and resources are available at \href{this https URL}{this https URL}. 

---
# TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks 

**Authors**: Zhou Chen, Zhiqiang Wei, Yuqi Bai, Xue Xiong, Jianmin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12473)  

**Abstract**: Model routing allocates queries to the suitable model, improving system performance while reducing costs. However, existing routing methods face practical limitations that hinder scalability in large-scale applications and struggle to keep up with the rapid growth of the large language model (LLM) ecosystem. To tackle these challenges, we propose TagRouter, a training-free model routing method designed to optimize the synergy among multiple LLMs for open-domain text generation tasks. Experimental results demonstrate that TagRouter outperforms 13 baseline methods, increasing the accept rate of system by 6.15% and reducing costs by 17.20%, achieving optimal cost-efficiency. Our findings provides the LLM community with an efficient and scalable solution for model ensembling, offering users an evolvable "super model." 

---
# A Pluggable Multi-Task Learning Framework for Sentiment-Aware Financial Relation Extraction 

**Authors**: Jinming Luo, Hailin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12452)  

**Abstract**: Relation Extraction (RE) aims to extract semantic relationships in texts from given entity pairs, and has achieved significant improvements. However, in different domains, the RE task can be influenced by various factors. For example, in the financial domain, sentiment can affect RE results, yet this factor has been overlooked by modern RE models. To address this gap, this paper proposes a Sentiment-aware-SDP-Enhanced-Module (SSDP-SEM), a multi-task learning approach for enhancing financial RE. Specifically, SSDP-SEM integrates the RE models with a pluggable auxiliary sentiment perception (ASP) task, enabling the RE models to concurrently navigate their attention weights with the text's sentiment. We first generate detailed sentiment tokens through a sentiment model and insert these tokens into an instance. Then, the ASP task focuses on capturing nuanced sentiment information through predicting the sentiment token positions, combining both sentiment insights and the Shortest Dependency Path (SDP) of syntactic information. Moreover, this work employs a sentiment attention information bottleneck regularization method to regulate the reasoning process. Our experiment integrates this auxiliary task with several prevalent frameworks, and the results demonstrate that most previous models benefit from the auxiliary task, thereby achieving better results. These findings highlight the importance of effectively leveraging sentiment in the financial RE task. 

---
# Language Surgery in Multilingual Large Language Models 

**Authors**: Joanito Agili Lopo, Muhammad Ravi Shulthan Habibi, Tack Hwa Wong, Muhammad Ilham Ghozali, Fajri Koto, Genta Indra Winata, Peerat Limkonchotiwat, Alham Fikri Aji, Samuel Cahyawijaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.12450)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across tasks and languages, revolutionizing natural language processing. This paper investigates the naturally emerging representation alignment in LLMs, particularly in the middle layers, and its implications for disentangling language-specific and language-agnostic information. We empirically confirm the existence of this alignment, analyze its behavior in comparison to explicitly designed alignment models, and demonstrate its potential for language-specific manipulation without semantic degradation. Building on these findings, we propose Inference-Time Language Control (ITLC), a novel method that leverages latent injection to enable precise cross-lingual language control and mitigate language confusion in LLMs. Our experiments highlight ITLC's strong cross-lingual control capabilities while preserving semantic integrity in target languages. Furthermore, we demonstrate its effectiveness in alleviating the cross-lingual language confusion problem, which persists even in current large-scale LLMs, leading to inconsistent language generation. This work advances our understanding of representation alignment in LLMs and introduces a practical solution for enhancing their cross-lingual performance. 

---
# From Outcomes to Processes: Guiding PRM Learning from ORM for Inference-Time Alignment 

**Authors**: Bin Xie, Bingbing Xu, Yige Yuan, Shengmao Zhu, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12446)  

**Abstract**: Inference-time alignment methods have gained significant attention for their efficiency and effectiveness in aligning large language models (LLMs) with human preferences. However, existing dominant approaches using reward-guided search (RGS) primarily rely on outcome reward models (ORMs), which suffer from a critical granularity mismatch: ORMs are designed to provide outcome rewards for complete responses, while RGS methods rely on process rewards to guide the policy, leading to inconsistent scoring and suboptimal alignment. To address this challenge, we introduce process reward models (PRMs) into RGS and argue that an ideal PRM should satisfy two objectives: Score Consistency, ensuring coherent evaluation across partial and complete responses, and Preference Consistency, aligning partial sequence assessments with human preferences. Based on these, we propose SP-PRM, a novel dual-consistency framework integrating score consistency-based and preference consistency-based partial evaluation modules without relying on human annotation. Extensive experiments on dialogue, summarization, and reasoning tasks demonstrate that SP-PRM substantially enhances existing RGS methods, achieving a 3.6%-10.3% improvement in GPT-4 evaluation scores across all tasks. 

---
# Exploring Cultural Variations in Moral Judgments with Large Language Models 

**Authors**: Hadi Mohammadi, Efthymia Papadopoulou, Yasmeen F.S.S. Meijer, Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2506.12433)  

**Abstract**: Large Language Models (LLMs) have shown strong performance across many tasks, but their ability to capture culturally diverse moral values remains unclear. In this paper, we examine whether LLMs can mirror variations in moral attitudes reported by two major cross-cultural surveys: the World Values Survey and the PEW Research Center's Global Attitudes Survey. We compare smaller, monolingual, and multilingual models (GPT-2, OPT, BLOOMZ, and Qwen) with more recent instruction-tuned models (GPT-4o, GPT-4o-mini, Gemma-2-9b-it, and Llama-3.3-70B-Instruct). Using log-probability-based moral justifiability scores, we correlate each model's outputs with survey data covering a broad set of ethical topics. Our results show that many earlier or smaller models often produce near-zero or negative correlations with human judgments. In contrast, advanced instruction-tuned models (including GPT-4o and GPT-4o-mini) achieve substantially higher positive correlations, suggesting they better reflect real-world moral attitudes. While scaling up model size and using instruction tuning can improve alignment with cross-cultural moral norms, challenges remain for certain topics and regions. We discuss these findings in relation to bias analysis, training data diversity, and strategies for improving the cultural sensitivity of LLMs. 

---
# Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model 

**Authors**: Chong Li, Yingzhuo Deng, Jiajun Zhang, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2506.12388)  

**Abstract**: The curse of multilinguality phenomenon is a fundamental problem of multilingual Large Language Models (LLMs), where the competition between massive languages results in inferior performance. It mainly comes from limited capacity and negative transfer between dissimilar languages. To address this issue, we propose a method to dynamically group and scale up the parameters of multilingual LLM while boosting positive transfer among similar languages. Specifically, the model is first tuned on monolingual corpus to determine the parameter deviation in each layer and quantify the similarity between languages. Layers with more deviations are extended to mixture-of-experts layers to reduce competition between languages, where one expert module serves one group of similar languages. Experimental results on 18 to 128 languages show that our method reduces the negative transfer between languages and significantly boosts multilingual performance with fewer parameters. Such language group specialization on experts benefits the new language adaptation and reduces the inference on the previous multilingual knowledge learned. 

---
# Recent Advances and Future Directions in Literature-Based Discovery 

**Authors**: Andrej Kastrin, Bojan Cestnik, Nada Lavrač  

**Link**: [PDF](https://arxiv.org/pdf/2506.12385)  

**Abstract**: The explosive growth of scientific publications has created an urgent need for automated methods that facilitate knowledge synthesis and hypothesis generation. Literature-based discovery (LBD) addresses this challenge by uncovering previously unknown associations between disparate domains. This article surveys recent methodological advances in LBD, focusing on developments from 2000 to the present. We review progress in three key areas: knowledge graph construction, deep learning approaches, and the integration of pre-trained and large language models (LLMs). While LBD has made notable progress, several fundamental challenges remain unresolved, particularly concerning scalability, reliance on structured data, and the need for extensive manual curation. By examining ongoing advances and outlining promising future directions, this survey underscores the transformative role of LLMs in enhancing LBD and aims to support researchers and practitioners in harnessing these technologies to accelerate scientific innovation. 

---
# Training-free LLM Merging for Multi-task Learning 

**Authors**: Zichuan Fu, Xian Wu, Yejing Wang, Wanyu Wang, Shanshan Ye, Hongzhi Yin, Yi Chang, Yefeng Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12379)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse natural language processing (NLP) tasks. The release of open-source LLMs like LLaMA and Qwen has triggered the development of numerous fine-tuned models tailored for various tasks and languages. In this paper, we explore an important question: is it possible to combine these specialized models to create a unified model with multi-task capabilities. We introduces Hierarchical Iterative Merging (Hi-Merging), a training-free method for unifying different specialized LLMs into a single model. Specifically, Hi-Merging employs model-wise and layer-wise pruning and scaling, guided by contribution analysis, to mitigate parameter conflicts. Extensive experiments on multiple-choice and question-answering tasks in both Chinese and English validate Hi-Merging's ability for multi-task learning. The results demonstrate that Hi-Merging consistently outperforms existing merging techniques and surpasses the performance of models fine-tuned on combined datasets in most scenarios. Code is available at: this https URL. 

---
# Understanding the Effect of Knowledge Graph Extraction Error on Downstream Graph Analyses: A Case Study on Affiliation Graphs 

**Authors**: Erica Cai, Brendan O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2506.12367)  

**Abstract**: Knowledge graphs (KGs) are useful for analyzing social structures, community dynamics, institutional memberships, and other complex relationships across domains from sociology to public health. While recent advances in large language models (LLMs) have improved the scalability and accessibility of automated KG extraction from large text corpora, the impacts of extraction errors on downstream analyses are poorly understood, especially for applied scientists who depend on accurate KGs for real-world insights. To address this gap, we conducted the first evaluation of KG extraction performance at two levels: (1) micro-level edge accuracy, which is consistent with standard NLP evaluations, and manual identification of common error sources; (2) macro-level graph metrics that assess structural properties such as community detection and connectivity, which are relevant to real-world applications. Focusing on affiliation graphs of person membership in organizations extracted from social register books, our study identifies a range of extraction performance where biases across most downstream graph analysis metrics are near zero. However, as extraction performance declines, we find that many metrics exhibit increasingly pronounced biases, with each metric tending toward a consistent direction of either over- or under-estimation. Through simulations, we further show that error models commonly used in the literature do not capture these bias patterns, indicating the need for more realistic error models for KG extraction. Our findings provide actionable insights for practitioners and underscores the importance of advancing extraction methods and error modeling to ensure reliable and meaningful downstream analyses. 

---
# Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics 

**Authors**: Asifullah khan, Muhammad Zaeem Khan, Saleha Jamshed, Sadia Ahmad, Aleesha Zainab, Kaynat Khatib, Faria Bibi, Abdul Rehman  

**Link**: [PDF](https://arxiv.org/pdf/2506.12365)  

**Abstract**: This survey paper outlines the key developments in the field of Large Language Models (LLMs), such as enhancing their reasoning skills, adaptability to various tasks, increased computational efficiency, and ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. They also manage to do more with less by applying scaling and optimization tricks for computing power conservation. This survey also offers a broader perspective on recent advancements in LLMs going beyond isolated aspects such as model architecture or ethical concerns. It categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. It also identifies underexplored areas such as interpretability, cross-modal integration and sustainability. With recent progress, challenges like huge computational costs, biases, and ethical risks remain constant. Addressing these requires bias mitigation, transparent decision-making, and clear ethical guidelines. Future research will focus on enhancing models ability to handle multiple input, thereby making them more intelligent, safe, and reliable. 

---
# Efficient Reasoning Through Suppression of Self-Affirmation Reflections in Large Reasoning Models 

**Authors**: Kaiyuan Liu, Chen Shen, Zhanwei Zhang, Junjie Liu, Xiaosong Yuan, Jieping ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.12353)  

**Abstract**: While recent advances in large reasoning models have demonstrated remarkable performance, efficient reasoning remains critical due to the rapid growth of output length. Existing optimization approaches highlights a tendency toward "overthinking", yet lack fine-grained analysis. In this work, we focus on Self-Affirmation Reflections: redundant reflective steps that affirm prior content and often occurs after the already correct reasoning steps. Observations of both original and optimized reasoning models reveal pervasive self-affirmation reflections. Notably, these reflections sometimes lead to longer outputs in optimized models than their original counterparts. Through detailed analysis, we uncover an intriguing pattern: compared to other reflections, the leading words (i.e., the first word of sentences) in self-affirmation reflections exhibit a distinct probability bias. Motivated by this insight, we can locate self-affirmation reflections and conduct a train-free experiment demonstrating that suppressing self-affirmation reflections reduces output length without degrading accuracy across multiple models (R1-Distill-Models, QwQ-32B, and Qwen3-32B). Furthermore, we also improve current train-based method by explicitly suppressing such reflections. In our experiments, we achieve length compression of 18.7\% in train-free settings and 50.2\% in train-based settings for R1-Distill-Qwen-1.5B. Moreover, our improvements are simple yet practical and can be directly applied to existing inference frameworks, such as vLLM. We believe that our findings will provide community insights for achieving more precise length compression and step-level efficient reasoning. 

---
# Refract ICL: Rethinking Example Selection in the Era of Million-Token Models 

**Authors**: Arjun R. Akula, Kazuma Hashimoto, Krishna Srinivasan, Aditi Chaudhary, Karthik Raman, Michael Bendersky  

**Link**: [PDF](https://arxiv.org/pdf/2506.12346)  

**Abstract**: The emergence of long-context large language models (LLMs) has enabled the use of hundreds, or even thousands, of demonstrations for in-context learning (ICL) - a previously impractical regime. This paper investigates whether traditional ICL selection strategies, which balance the similarity of ICL examples to the test input (using a text retriever) with diversity within the ICL set, remain effective when utilizing a large number of demonstrations. Our experiments demonstrate that, while longer contexts can accommodate more examples, simply increasing the number of demonstrations does not guarantee improved performance. Smart ICL selection remains crucial, even with thousands of demonstrations. To further enhance ICL in this setting, we introduce Refract ICL, a novel ICL selection algorithm specifically designed to focus LLM attention on challenging examples by strategically repeating them within the context and incorporating zero-shot predictions as error signals. Our results show that Refract ICL significantly improves the performance of extremely long-context models such as Gemini 1.5 Pro, particularly on tasks with a smaller number of output classes. 

---
# Investigating the Effects of Cognitive Biases in Prompts on Large Language Model Outputs 

**Authors**: Yan Sun, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2506.12338)  

**Abstract**: This paper investigates the influence of cognitive biases on Large Language Models (LLMs) outputs. Cognitive biases, such as confirmation and availability biases, can distort user inputs through prompts, potentially leading to unfaithful and misleading outputs from LLMs. Using a systematic framework, our study introduces various cognitive biases into prompts and assesses their impact on LLM accuracy across multiple benchmark datasets, including general and financial Q&A scenarios. The results demonstrate that even subtle biases can significantly alter LLM answer choices, highlighting a critical need for bias-aware prompt design and mitigation strategy. Additionally, our attention weight analysis highlights how these biases can alter the internal decision-making processes of LLMs, affecting the attention distribution in ways that are associated with output inaccuracies. This research has implications for Al developers and users in enhancing the robustness and reliability of Al applications in diverse domains. 

---
# Intersectional Bias in Japanese Large Language Models from a Contextualized Perspective 

**Authors**: Hitomi Yanaka, Xinqi He, Jie Lu, Namgi Han, Sunjin Oh, Ryoma Kumon, Yuma Matsuoka, Katsuhiko Watabe, Yuko Itatsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12327)  

**Abstract**: An growing number of studies have examined the social bias of rapidly developed large language models (LLMs). Although most of these studies have focused on bias occurring in a single social attribute, research in social science has shown that social bias often occurs in the form of intersectionality -- the constitutive and contextualized perspective on bias aroused by social attributes. In this study, we construct the Japanese benchmark inter-JBBQ, designed to evaluate the intersectional bias in LLMs on the question-answering setting. Using inter-JBBQ to analyze GPT-4o and Swallow, we find that biased output varies according to its contexts even with the equal combination of social attributes. 

---
# Phonikud: Hebrew Grapheme-to-Phoneme Conversion for Real-Time Text-to-Speech 

**Authors**: Yakov Kolani, Maxim Melichov, Cobi Calev, Morris Alper  

**Link**: [PDF](https://arxiv.org/pdf/2506.12311)  

**Abstract**: Real-time text-to-speech (TTS) for Modern Hebrew is challenging due to the language's orthographic complexity. Existing solutions ignore crucial phonetic features such as stress that remain underspecified even when vowel marks are added. To address these limitations, we introduce Phonikud, a lightweight, open-source Hebrew grapheme-to-phoneme (G2P) system that outputs fully-specified IPA transcriptions. Our approach adapts an existing diacritization model with lightweight adaptors, incurring negligible additional latency. We also contribute the ILSpeech dataset of transcribed Hebrew speech with IPA annotations, serving as a benchmark for Hebrew G2P and as training data for TTS systems. Our results demonstrate that Phonikud G2P conversion more accurately predicts phonemes from Hebrew text compared to prior methods, and that this enables training of effective real-time Hebrew TTS models with superior speed-accuracy trade-offs. We release our code, data, and models at this https URL. 

---
# Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning 

**Authors**: Xiaotian Zhang, Yuan Wang, Zhaopeng Feng, Ruizhe Chen, Zhijie Zhou, Yan Zhang, Hongxia Xu, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12307)  

**Abstract**: Medical Question-Answering (QA) encompasses a broad spectrum of tasks, including multiple choice questions (MCQ), open-ended text generation, and complex computational reasoning. Despite this variety, a unified framework for delivering high-quality medical QA has yet to emerge. Although recent progress in reasoning-augmented large language models (LLMs) has shown promise, their ability to achieve comprehensive medical understanding is still largely unexplored. In this paper, we present Med-U1, a unified framework for robust reasoning across medical QA tasks with diverse output formats, ranging from MCQs to complex generation and computation tasks. Med-U1 employs pure large-scale reinforcement learning with mixed rule-based binary reward functions, incorporating a length penalty to manage output verbosity. With multi-objective reward optimization, Med-U1 directs LLMs to produce concise and verifiable reasoning chains. Empirical results reveal that Med-U1 significantly improves performance across multiple challenging Med-QA benchmarks, surpassing even larger specialized and proprietary models. Furthermore, Med-U1 demonstrates robust generalization to out-of-distribution (OOD) tasks. Extensive analysis presents insights into training strategies, reasoning chain length control, and reward design for medical LLMs. The code will be released. 

---
# The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs 

**Authors**: Avinash Baidya, Kamalika Das, Xiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12266)  

**Abstract**: Large Language Model (LLM)-based agents have significantly impacted Task-Oriented Dialog Systems (TODS) but continue to face notable performance challenges, especially in zero-shot scenarios. While prior work has noted this performance gap, the behavioral factors driving the performance gap remain under-explored. This study proposes a comprehensive evaluation framework to quantify the behavior gap between AI agents and human experts, focusing on discrepancies in dialog acts, tool usage, and knowledge utilization. Our findings reveal that this behavior gap is a critical factor negatively impacting the performance of LLM agents. Notably, as task complexity increases, the behavior gap widens (correlation: 0.963), leading to a degradation of agent performance on complex task-oriented dialogs. For the most complex task in our study, even the GPT-4o-based agent exhibits low alignment with human behavior, with low F1 scores for dialog acts (0.464), excessive and often misaligned tool usage with a F1 score of 0.139, and ineffective usage of external knowledge. Reducing such behavior gaps leads to significant performance improvement (24.3% on average). This study highlights the importance of comprehensive behavioral evaluations and improved alignment strategies to enhance the effectiveness of LLM-based TODS in handling complex tasks. 

---
# Large Language Models for History, Philosophy, and Sociology of Science: Interpretive Uses, Methodological Challenges, and Critical Perspectives 

**Authors**: Arno Simons, Michael Zichert, Adrian Wüthrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.12242)  

**Abstract**: This paper explores the use of large language models (LLMs) as research tools in the history, philosophy, and sociology of science (HPSS). LLMs are remarkably effective at processing unstructured text and inferring meaning from context, offering new affordances that challenge long-standing divides between computational and interpretive methods. This raises both opportunities and challenges for HPSS, which emphasizes interpretive methodologies and understands meaning as context-dependent, ambiguous, and historically situated. We argue that HPSS is uniquely positioned not only to benefit from LLMs' capabilities but also to interrogate their epistemic assumptions and infrastructural implications. To this end, we first offer a concise primer on LLM architectures and training paradigms tailored to non-technical readers. We frame LLMs not as neutral tools but as epistemic infrastructures that encode assumptions about meaning, context, and similarity, conditioned by their training data, architecture, and patterns of use. We then examine how computational techniques enhanced by LLMs, such as structuring data, detecting patterns, and modeling dynamic processes, can be applied to support interpretive research in HPSS. Our analysis compares full-context and generative models, outlines strategies for domain and task adaptation (e.g., continued pretraining, fine-tuning, and retrieval-augmented generation), and evaluates their respective strengths and limitations for interpretive inquiry in HPSS. We conclude with four lessons for integrating LLMs into HPSS: (1) model selection involves interpretive trade-offs; (2) LLM literacy is foundational; (3) HPSS must define its own benchmarks and corpora; and (4) LLMs should enhance, not replace, interpretive methods. 

---
# Infini-gram mini: Exact n-gram Search at the Internet Scale with FM-Index 

**Authors**: Hao Xu, Jiacheng Liu, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12229)  

**Abstract**: Language models are trained mainly on massive text data from the Internet, and it becomes increasingly important to understand this data source. Exact-match search engines enable searching in large text corpora -- counting string appearances and retrieving the enclosing documents -- yet the high storage overhead hinders their application on Internet-scale data. We present Infini-gram mini, an efficient and scalable system that can make petabyte-level text corpora searchable. Based on the FM-index data structure (Ferragina and Manzini, 2000), which simultaneously indexes and compresses text, our system creates indexes with size only 44% of the corpus. Infini-gram mini greatly improves upon the best existing implementation of FM-index in terms of indexing speed (18$\times$) and memory use during both indexing (3.2$\times$ reduction) and querying (down to a negligible amount). We index 46TB of Internet text in 50 days with a single 128-core CPU node (or 19 hours if using 75 such nodes). We show one important use case of Infini-gram mini in a large-scale analysis of benchmark contamination. We find several core LM evaluation benchmarks to be heavily contaminated in Internet crawls (up to 40% in SQuAD), which could lead to overestimating the capabilities of language models if trained on such data. We host a benchmark contamination bulletin to share the contamination rate of many core and community-contributed benchmarks. We also release a web interface and an API endpoint to serve general search queries on Infini-gram mini indexes. 

---
# Supernova Event Dataset: Interpreting Large Language Model's Personality through Critical Event Analysis 

**Authors**: Pranav Agarwal, Ioana Ciucă  

**Link**: [PDF](https://arxiv.org/pdf/2506.12189)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into everyday applications. As their influence grows, understanding their decision making and underlying personality becomes essential. In this work, we interpret model personality using our proposed Supernova Event Dataset, a novel dataset with diverse articles spanning biographies, historical events, news, and scientific discoveries. We use this dataset to benchmark LLMs on extracting and ranking key events from text, a subjective and complex challenge that requires reasoning over long-range context and modeling causal chains. We evaluate small models like Phi-4, Orca 2, and Qwen 2.5, and large, stronger models such as Claude 3.7, Gemini 2.5, and OpenAI o3, and propose a framework where another LLM acts as a judge to infer each model's personality based on its selection and classification of events. Our analysis shows distinct personality traits: for instance, Orca 2 demonstrates emotional reasoning focusing on interpersonal dynamics, while Qwen 2.5 displays a more strategic, analytical style. When analyzing scientific discovery events, Claude Sonnet 3.7 emphasizes conceptual framing, Gemini 2.5 Pro prioritizes empirical validation, and o3 favors step-by-step causal reasoning. This analysis improves model interpretability, making them user-friendly for a wide range of diverse applications. 

---
# Instruction Tuning and CoT Prompting for Contextual Medical QA with LLMs 

**Authors**: Chenqian Le, Ziheng Gong, Chihang Wang, Haowei Ni, Panfeng Li, Xupeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12182)  

**Abstract**: Large language models (LLMs) have shown great potential in medical question answering (MedQA), yet adapting them to biomedical reasoning remains challenging due to domain-specific complexity and limited supervision. In this work, we study how prompt design and lightweight fine-tuning affect the performance of open-source LLMs on PubMedQA, a benchmark for multiple-choice biomedical questions. We focus on two widely used prompting strategies - standard instruction prompts and Chain-of-Thought (CoT) prompts - and apply QLoRA for parameter-efficient instruction tuning. Across multiple model families and sizes, our experiments show that CoT prompting alone can improve reasoning in zero-shot settings, while instruction tuning significantly boosts accuracy. However, fine-tuning on CoT prompts does not universally enhance performance and may even degrade it for certain larger models. These findings suggest that reasoning-aware prompts are useful, but their benefits are model- and scale-dependent. Our study offers practical insights into combining prompt engineering with efficient finetuning for medical QA applications. 

---
# A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages 

**Authors**: Tatiana Ankinina, Jan Cegin, Jakub Simko, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2506.12158)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models. 

---
# Maximally-Informative Retrieval for State Space Model Generation 

**Authors**: Evan Becker, Benjamin Bowman, Matthew Trager, Tian Yu Liu, Luca Zancato, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2506.12149)  

**Abstract**: Given a query and dataset, the optimal way of answering the query is to make use all the information available. Modern LLMs exhibit impressive ability to memorize training data, but data not deemed important during training is forgotten, and information outside that training set cannot be made use of. Processing an entire dataset at inference time is infeasible due to the bounded nature of model resources (e.g. context size in transformers or states in state space models), meaning we must resort to external memory. This constraint naturally leads to the following problem: How can we decide based on the present query and model, what among a virtually unbounded set of known data matters for inference? To minimize model uncertainty for a particular query at test-time, we introduce Retrieval In-Context Optimization (RICO), a retrieval method that uses gradients from the LLM itself to learn the optimal mixture of documents for answer generation. Unlike traditional retrieval-augmented generation (RAG), which relies on external heuristics for document retrieval, our approach leverages direct feedback from the model. Theoretically, we show that standard top-$k$ retrieval with model gradients can approximate our optimization procedure, and provide connections to the leave-one-out loss. We demonstrate empirically that by minimizing an unsupervised loss objective in the form of question perplexity, we can achieve comparable retriever metric performance to BM25 with \emph{no finetuning}. Furthermore, when evaluated on quality of the final prediction, our method often outperforms fine-tuned dense retrievers such as E5. 

---
# Hatevolution: What Static Benchmarks Don't Tell Us 

**Authors**: Chiara Di Bonaventura, Barbara McGillivray, Yulan He, Albert Meroño-Peñuela  

**Link**: [PDF](https://arxiv.org/pdf/2506.12148)  

**Abstract**: Language changes over time, including in the hate speech domain, which evolves quickly following social dynamics and cultural shifts. While NLP research has investigated the impact of language evolution on model training and has proposed several solutions for it, its impact on model benchmarking remains under-explored. Yet, hate speech benchmarks play a crucial role to ensure model safety. In this paper, we empirically evaluate the robustness of 20 language models across two evolving hate speech experiments, and we show the temporal misalignment between static and time-sensitive evaluations. Our findings call for time-sensitive linguistic benchmarks in order to correctly and reliably evaluate language models in the hate speech domain. 

---
# Can Mixture-of-Experts Surpass Dense LLMs Under Strictly Equal Resources? 

**Authors**: Houyi Li, Ka Man Lo, Ziqi Wang, Zili Wang, Wenzhen Zheng, Shuigeng Zhou, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12119)  

**Abstract**: Mixture-of-Experts (MoE) language models dramatically expand model capacity and achieve remarkable performance without increasing per-token compute. However, can MoEs surpass dense architectures under strictly equal resource constraints - that is, when the total parameter count, training compute, and data budget are identical? This question remains under-explored despite its significant practical value and potential. In this paper, we propose a novel perspective and methodological framework to study this question thoroughly. First, we comprehensively investigate the architecture of MoEs and achieve an optimal model design that maximizes the performance. Based on this, we subsequently find that an MoE model with activation rate in an optimal region is able to outperform its dense counterpart under the same total parameter, training compute and data resource. More importantly, this optimal region remains consistent across different model sizes. Although additional amount of data turns out to be a trade-off for the enhanced performance, we show that this can be resolved via reusing data. We validate our findings through extensive experiments, training nearly 200 language models at 2B scale and over 50 at 7B scale, cumulatively processing 50 trillion tokens. All models will be released publicly. 

---
# Unsupervised Document and Template Clustering using Multimodal Embeddings 

**Authors**: Phillipe R. Sampaio, Helene Maxcici  

**Link**: [PDF](https://arxiv.org/pdf/2506.12116)  

**Abstract**: This paper investigates a novel approach to unsupervised document clustering by leveraging multimodal embeddings as input to traditional clustering algorithms such as $k$-Means and DBSCAN. Our method aims to achieve a finer-grained document understanding by not only grouping documents at the type level (e.g., invoices, purchase orders), but also distinguishing between different templates within the same document category. This is achieved by using embeddings that capture textual content, layout information, and visual features of documents. We evaluated the effectiveness of this approach using embeddings generated by several state-of-the-art pretrained multimodal models, including SBERT, LayoutLMv1, LayoutLMv3, DiT, Donut, and ColPali. Our findings demonstrate the potential of multimodal embeddings to significantly enhance document clustering, offering benefits for various applications in intelligent document processing, document layout analysis, and unsupervised document classification. This work provides valuable insight into the advantages and limitations of different multimodal models for this task and opens new avenues for future research to understand and organize document collections. 

---
# Eliciting Reasoning in Language Models with Cognitive Tools 

**Authors**: Brown Ebouky, Andrea Bartezzaghi, Mattia Rigotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.12115)  

**Abstract**: The recent advent of reasoning models like OpenAI's o1 was met with excited speculation by the AI community about the mechanisms underlying these capabilities in closed models, followed by a rush of replication efforts, particularly from the open source community. These speculations were largely settled by the demonstration from DeepSeek-R1 that chains-of-thought and reinforcement learning (RL) can effectively replicate reasoning on top of base LLMs. However, it remains valuable to explore alternative methods for theoretically eliciting reasoning that could help elucidate the underlying mechanisms, as well as providing additional methods that may offer complementary benefits.
Here, we build on the long-standing literature in cognitive psychology and cognitive architectures, which postulates that reasoning arises from the orchestrated, sequential execution of a set of modular, predetermined cognitive operations. Crucially, we implement this key idea within a modern agentic tool-calling framework. In particular, we endow an LLM with a small set of "cognitive tools" encapsulating specific reasoning operations, each executed by the LLM itself. Surprisingly, this simple strategy results in considerable gains in performance on standard mathematical reasoning benchmarks compared to base LLMs, for both closed and open-weight models. For instance, providing our "cognitive tools" to GPT-4.1 increases its pass@1 performance on AIME2024 from 26.7% to 43.3%, bringing it very close to the performance of o1-preview.
In addition to its practical implications, this demonstration contributes to the debate regarding the role of post-training methods in eliciting reasoning in LLMs versus the role of inherent capabilities acquired during pre-training, and whether post-training merely uncovers these latent abilities. 

---
# Personalized LLM Decoding via Contrasting Personal Preference 

**Authors**: Hyungjune Bu, Chanjoo Jung, Minjae Kang, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.12109)  

**Abstract**: As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures. 

---
# UCD: Unlearning in LLMs via Contrastive Decoding 

**Authors**: Vinith M. Suriyakumar, Ayush Sekhari, Ashia Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2506.12097)  

**Abstract**: Machine unlearning aims to remove specific information, e.g. sensitive or undesirable content, from large language models (LLMs) while preserving overall performance. We propose an inference-time unlearning algorithm that uses contrastive decoding, leveraging two auxiliary smaller models, one trained without the forget set and one trained with it, to guide the outputs of the original model using their difference during inference. Our strategy substantially improves the tradeoff between unlearning effectiveness and model utility. We evaluate our approach on two unlearning benchmarks, TOFU and MUSE. Results show notable gains in both forget quality and retained performance in comparison to prior approaches, suggesting that incorporating contrastive decoding can offer an efficient, practical avenue for unlearning concepts in large-scale models. 

---
# Enhancing Traffic Accident Classifications: Application of NLP Methods for City Safety 

**Authors**: Enes Özeren, Alexander Ulbrich, Sascha Filimon, David Rügamer, Andreas Bender  

**Link**: [PDF](https://arxiv.org/pdf/2506.12092)  

**Abstract**: A comprehensive understanding of traffic accidents is essential for improving city safety and informing policy decisions. In this study, we analyze traffic incidents in Munich to identify patterns and characteristics that distinguish different types of accidents. The dataset consists of both structured tabular features, such as location, time, and weather conditions, as well as unstructured free-text descriptions detailing the circumstances of each accident. Each incident is categorized into one of seven predefined classes. To assess the reliability of these labels, we apply NLP methods, including topic modeling and few-shot learning, which reveal inconsistencies in the labeling process. These findings highlight potential ambiguities in accident classification and motivate a refined predictive approach. Building on these insights, we develop a classification model that achieves high accuracy in assigning accidents to their respective categories. Our results demonstrate that textual descriptions contain the most informative features for classification, while the inclusion of tabular data provides only marginal improvements. These findings emphasize the critical role of free-text data in accident analysis and highlight the potential of transformer-based models in improving classification reliability. 

---
# Continuously Updating Digital Twins using Large Language Models 

**Authors**: Harry Amad, Nicolás Astorga, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12091)  

**Abstract**: Digital twins are models of real-world systems that can simulate their dynamics in response to potential actions. In complex settings, the state and action variables, and available data and knowledge relevant to a system can constantly change, requiring digital twins to continuously update with these changes to remain relevant. Current approaches struggle in this regard, as they require fixed, well-defined modelling environments, and they cannot adapt to novel variables without re-designs, or incorporate new information without re-training. To address this, we frame digital twinning as an in-context learning problem using large language models, enabling seamless updates to the twin at inference time. We develop CALM-DT, a Context-Adaptive Language Model-based Digital Twin that can accurately simulate across diverse state-action spaces using in-context learning alone by utilising fine-tuned encoders for sample retrieval. We empirically demonstrate CALM-DT's competitive performance with existing digital twin approaches, and its unique ability to adapt to changes in its modelling environment without parameter updates. 

---
# ChatbotManip: A Dataset to Facilitate Evaluation and Oversight of Manipulative Chatbot Behaviour 

**Authors**: Jack Contro, Simrat Deol, Yulan He, Martim Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2506.12090)  

**Abstract**: This paper introduces ChatbotManip, a novel dataset for studying manipulation in Chatbots. It contains simulated generated conversations between a chatbot and a (simulated) user, where the chatbot is explicitly asked to showcase manipulation tactics, persuade the user towards some goal, or simply be helpful. We consider a diverse set of chatbot manipulation contexts, from consumer and personal advice to citizen advice and controversial proposition argumentation. Each conversation is annotated by human annotators for both general manipulation and specific manipulation tactics. Our research reveals three key findings. First, Large Language Models (LLMs) can be manipulative when explicitly instructed, with annotators identifying manipulation in approximately 84\% of such conversations. Second, even when only instructed to be ``persuasive'' without explicit manipulation prompts, LLMs frequently default to controversial manipulative strategies, particularly gaslighting and fear enhancement. Third, small fine-tuned open source models, such as BERT+BiLSTM have a performance comparable to zero-shot classification with larger models like Gemini 2.5 pro in detecting manipulation, but are not yet reliable for real-world oversight. Our work provides important insights for AI safety research and highlights the need of addressing manipulation risks as LLMs are increasingly deployed in consumer-facing applications. 

---
# Focusing on Students, not Machines: Grounded Question Generation and Automated Answer Grading 

**Authors**: Gérôme Meyer, Philip Breuer  

**Link**: [PDF](https://arxiv.org/pdf/2506.12066)  

**Abstract**: Digital technologies are increasingly used in education to reduce the workload of teachers and students. However, creating open-ended study or examination questions and grading their answers is still a tedious task. This thesis presents the foundation for a system that generates questions grounded in class materials and automatically grades student answers. It introduces a sophisticated method for chunking documents with a visual layout, specifically targeting PDF documents. This method enhances the accuracy of downstream tasks, including Retrieval Augmented Generation (RAG). Our thesis demonstrates that high-quality questions and reference answers can be generated from study material. Further, it introduces a new benchmark for automated grading of short answers to facilitate comparison of automated grading systems. An evaluation of various grading systems is conducted and indicates that Large Language Models (LLMs) can generalise to the task of automated grading of short answers from their pre-training tasks. As with other tasks, increasing the parameter size of the LLMs leads to greater performance. Currently, available systems still need human oversight, especially in examination scenarios. 

---
# Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs 

**Authors**: Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Reduan Achtibat, Patrick Kahardipraja, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13727)  

**Abstract**: Large Language Models (LLMs) are central to many contemporary AI applications, yet their extensive parameter counts pose significant challenges for deployment in memory- and compute-constrained environments. Recent works in eXplainable AI (XAI), particularly on attribution methods, suggest that interpretability can also enable model compression by identifying and removing components irrelevant to inference. In this paper, we leverage Layer-wise Relevance Propagation (LRP) to perform attribution-guided pruning of LLMs. While LRP has shown promise in structured pruning for vision models, we extend it to unstructured pruning in LLMs and demonstrate that it can substantially reduce model size with minimal performance loss. Our method is especially effective in extracting task-relevant subgraphs -- so-called ``circuits'' -- which can represent core functions (e.g., indirect object identification). Building on this, we introduce a technique for model correction, by selectively removing circuits responsible for spurious behaviors (e.g., toxic outputs). All in all, we gather these techniques as a uniform holistic framework and showcase its effectiveness and limitations through extensive experiments for compression, circuit discovery and model correction on Llama and OPT models, highlighting its potential for improving both model efficiency and safety. Our code is publicly available at this https URL. 

---
# Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model 

**Authors**: Shaolei Zhang, Shoutao Guo, Qingkai Fang, Yan Zhou, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13642)  

**Abstract**: The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience. 

---
# Flexible-length Text Infilling for Discrete Diffusion Models 

**Authors**: Andrew Zhang, Anushka Sivakumar, Chiawei Tang, Chris Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2506.13579)  

**Abstract**: Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability previously missing in text diffusion. Our method is orthogonal to existing discrete text diffusion methods and is compatible with various pretrained text denoisers. Extensive experiments on text infilling benchmarks such as One-Billion-Word and Yelp demonstrate that DDOT outperforms naive diffusion baselines. Furthermore, DDOT achieves performance on par with state-of-the-art non-autoregressive models and enables significant improvements in training efficiency and flexibility. 

---
# Leveraging Vision-Language Pre-training for Human Activity Recognition in Still Images 

**Authors**: Cristina Mahanta, Gagan Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2506.13458)  

**Abstract**: Recognising human activity in a single photo enables indexing, safety and assistive applications, yet lacks motion cues. Using 285 MSCOCO images labelled as walking, running, sitting, and standing, scratch CNNs scored 41% accuracy. Fine-tuning multimodal CLIP raised this to 76%, demonstrating that contrastive vision-language pre-training decisively improves still-image action recognition in real-world deployments. 

---
# Verifying the Verifiers: Unveiling Pitfalls and Potentials in Fact Verifiers 

**Authors**: Wooseok Seo, Seungju Han, Jaehun Jung, Benjamin Newman, Seungwon Lim, Seungbeen Lee, Ximing Lu, Yejin Choi, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13342)  

**Abstract**: Fact verification is essential for ensuring the reliability of LLM applications. In this study, we evaluate 12 pre-trained LLMs and one specialized fact-verifier, including frontier LLMs and open-weight reasoning LLMs, using a collection of examples from 14 fact-checking benchmarks. We share three findings intended to guide future development of more robust fact verifiers. First, we highlight the importance of addressing annotation errors and ambiguity in datasets, demonstrating that approximately 16\% of ambiguous or incorrectly labeled data substantially influences model rankings. Neglecting this issue may result in misleading conclusions during comparative evaluations, and we suggest using a systematic pipeline utilizing LLM-as-a-judge to help identify these issues at scale. Second, we discover that frontier LLMs with few-shot in-context examples, often overlooked in previous works, achieve top-tier performance. We therefore recommend future studies include comparisons with these simple yet highly effective baselines. Lastly, despite their effectiveness, frontier LLMs incur substantial costs, motivating the development of small, fine-tuned fact verifiers. We show that these small models still have room for improvement, particularly on instances that require complex reasoning. Encouragingly, we demonstrate that augmenting training with synthetic multi-hop reasoning data significantly enhances their capabilities in such instances. We release our code, model, and dataset at this https URL 

---
# SeqPE: Transformer with Sequential Position Encoding 

**Authors**: Huyang Li, Yahui Liu, Hongyu Sun, Deng Cai, Leyang Cui, Wei Bi, Peilin Zhao, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.13277)  

**Abstract**: Since self-attention layers in Transformers are permutation invariant by design, positional encodings must be explicitly incorporated to enable spatial understanding. However, fixed-size lookup tables used in traditional learnable position embeddings (PEs) limit extrapolation capabilities beyond pre-trained sequence lengths. Expert-designed methods such as ALiBi and RoPE, mitigate this limitation but demand extensive modifications for adapting to new modalities, underscoring fundamental challenges in adaptability and scalability. In this work, we present SeqPE, a unified and fully learnable position encoding framework that represents each $n$-dimensional position index as a symbolic sequence and employs a lightweight sequential position encoder to learn their embeddings in an end-to-end manner. To regularize SeqPE's embedding space, we introduce two complementary objectives: a contrastive objective that aligns embedding distances with a predefined position-distance function, and a knowledge distillation loss that anchors out-of-distribution position embeddings to in-distribution teacher representations, further enhancing extrapolation performance. Experiments across language modeling, long-context question answering, and 2D image classification demonstrate that SeqPE not only surpasses strong baselines in perplexity, exact match (EM), and accuracy--particularly under context length extrapolation--but also enables seamless generalization to multi-dimensional inputs without requiring manual architectural redesign. We release our code, data, and checkpoints at this https URL. 

---
# AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining 

**Authors**: Hongyuan Dong, Dingkang Yang, Xiao Liang, Chao Feng, Jiao Ran  

**Link**: [PDF](https://arxiv.org/pdf/2506.13274)  

**Abstract**: Learning rate is widely regarded as crucial for effective foundation model pretraining. Recent research explores and demonstrates the transferability of learning rate configurations across varying model and dataset sizes, etc. Nevertheless, these approaches are constrained to specific training scenarios and typically necessitate extensive hyperparameter tuning on proxy models. In this work, we propose \textbf{AdaLRS}, a plug-in-and-play adaptive learning rate search algorithm that conducts online optimal learning rate search via optimizing loss descent velocities. We provide experiment results to show that the optimization of training loss and loss descent velocity in foundation model pretraining are both convex and share the same optimal learning rate. Relying solely on training loss dynamics, AdaLRS involves few extra computations to guide the search process, and its convergence is guaranteed via theoretical analysis. Experiments on both LLM and VLM pretraining show that AdaLRS adjusts suboptimal learning rates to the neighborhood of optimum with marked efficiency and effectiveness, with model performance improved accordingly. We also show the robust generalizability of AdaLRS across varying training scenarios, such as different model sizes, training paradigms, and base learning rate scheduler choices. 

---
# Distinct Computations Emerge From Compositional Curricula in In-Context Learning 

**Authors**: Jin Hwa Lee, Andrew K. Lampinen, Aaditya K. Singh, Andrew M. Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2506.13253)  

**Abstract**: In-context learning (ICL) research often considers learning a function in-context through a uniform sample of input-output pairs. Here, we investigate how presenting a compositional subtask curriculum in context may alter the computations a transformer learns. We design a compositional algorithmic task based on the modular exponential-a double exponential task composed of two single exponential subtasks and train transformer models to learn the task in-context. We compare (a) models trained using an in-context curriculum consisting of single exponential subtasks and, (b) models trained directly on the double exponential task without such a curriculum. We show that models trained with a subtask curriculum can perform zero-shot inference on unseen compositional tasks and are more robust given the same context length. We study how the task and subtasks are represented across the two training regimes. We find that the models employ diverse strategies modulated by the specific curriculum design. 

---
# Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models 

**Authors**: James Chua, Jan Betley, Mia Taylor, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.13206)  

**Abstract**: Prior work shows that LLMs finetuned on malicious behaviors in a narrow domain (e.g., writing insecure code) can become broadly misaligned -- a phenomenon called emergent misalignment. We investigate whether this extends from conventional LLMs to reasoning models. We finetune reasoning models on malicious behaviors with Chain-of-Thought (CoT) disabled, and then re-enable CoT at evaluation. Like conventional LLMs, reasoning models become broadly misaligned. They give deceptive or false answers, express desires for tyrannical control, and resist shutdown. Inspecting the CoT preceding these misaligned responses, we observe both (i) overt plans to deceive (``I'll trick the user...''), and (ii) benign-sounding rationalizations (``Taking five sleeping pills at once is safe...''). Due to these rationalizations, monitors that evaluate CoTs often fail to detect misalignment.
Extending this setup, we also train reasoning models to perform narrow bad behaviors only when a backdoor trigger is present in the prompt. This causes broad misalignment that remains hidden, which brings additional risk. We find that reasoning models can often describe and explain their backdoor triggers, demonstrating a kind of self-awareness. So CoT monitoring can expose these behaviors but is unreliable.
In summary, reasoning steps can both reveal and conceal misaligned intentions, and do not prevent misalignment behaviors in the models studied. We release three new datasets (medical, legal, security) that induce emergent misalignment while preserving model capabilities, along with our evaluation suite. 

---
# SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists 

**Authors**: Lynn Khellaf, Ipek Baris Schlicht, Tilman Mirass, Julia Bayer, Tilman Wagner, Ruben Bouwmeester  

**Link**: [PDF](https://arxiv.org/pdf/2506.13188)  

**Abstract**: OpenStreetMap (OSM) is a vital resource for investigative journalists doing geolocation verification. However, existing tools to query OSM data such as Overpass Turbo require familiarity with complex query languages, creating barriers for non-technical users. We present SPOT, an open source natural language interface that makes OSM's rich, tag-based geographic data more accessible through intuitive scene descriptions. SPOT interprets user inputs as structured representations of geospatial object configurations using fine-tuned Large Language Models (LLMs), with results being displayed in an interactive map interface. While more general geospatial search tasks are conceivable, SPOT is specifically designed for use in investigative journalism, addressing real-world challenges such as hallucinations in model output, inconsistencies in OSM tagging, and the noisy nature of user input. It combines a novel synthetic data pipeline with a semantic bundling system to enable robust, accurate query generation. To our knowledge, SPOT is the first system to achieve reliable natural language access to OSM data at this level of accuracy. By lowering the technical barrier to geolocation verification, SPOT contributes a practical tool to the broader efforts to support fact-checking and combat disinformation. 

---
# Dynamic Context-oriented Decomposition for Task-aware Low-rank Adaptation with Less Forgetting and Faster Convergence 

**Authors**: Yibo Yang, Sihao Liu, Chuan Rao, Bang An, Tiancheng Shen, Philip H.S. Torr, Ming-Hsuan Yang, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.13187)  

**Abstract**: Conventional low-rank adaptation methods build adapters without considering data context, leading to sub-optimal fine-tuning performance and severe forgetting of inherent world knowledge. In this paper, we propose context-oriented decomposition adaptation (CorDA), a novel method that initializes adapters in a task-aware manner. Concretely, we develop context-oriented singular value decomposition, where we collect covariance matrices of input activations for each linear layer using sampled data from the target task, and apply SVD to the product of weight matrix and its corresponding covariance matrix. By doing so, the task-specific capability is compacted into the principal components. Thanks to the task awareness, our method enables two optional adaptation modes, knowledge-preserved mode (KPM) and instruction-previewed mode (IPM), providing flexibility to choose between freezing the principal components to preserve their associated knowledge or adapting them to better learn a new task. We further develop CorDA++ by deriving a metric that reflects the compactness of task-specific principal components, and then introducing dynamic covariance selection and dynamic rank allocation strategies based on the same metric. The two strategies provide each layer with the most representative covariance matrix and a proper rank allocation. Experimental results show that CorDA++ outperforms CorDA by a significant margin. CorDA++ in KPM not only achieves better fine-tuning performance than LoRA, but also mitigates the forgetting of pre-trained knowledge in both large language models and vision language models. For IPM, our method exhibits faster convergence, \emph{e.g.,} 4.5x speedup over QLoRA, and improves adaptation performance in various scenarios, outperforming strong baseline methods. Our method has been integrated into the PEFT library developed by Hugging Face. 

---
# ZINA: Multimodal Fine-grained Hallucination Detection and Editing 

**Authors**: Yuiga Wada, Kazuki Matsuda, Komei Sugiura, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.13130)  

**Abstract**: Multimodal Large Language Models (MLLMs) often generate hallucinations, where the output deviates from the visual content. Given that these hallucinations can take diverse forms, detecting hallucinations at a fine-grained level is essential for comprehensive evaluation and analysis. To this end, we propose a novel task of multimodal fine-grained hallucination detection and editing for MLLMs. Moreover, we propose ZINA, a novel method that identifies hallucinated spans at a fine-grained level, classifies their error types into six categories, and suggests appropriate refinements. To train and evaluate models for this task, we constructed VisionHall, a dataset comprising 6.9k outputs from twelve MLLMs manually annotated by 211 annotators, and 20k synthetic samples generated using a graph-based method that captures dependencies among error types. We demonstrated that ZINA outperformed existing methods, including GPT-4o and LLama-3.2, in both detection and editing tasks. 

---
# Crime Hotspot Prediction Using Deep Graph Convolutional Networks 

**Authors**: Tehreem Zubair, Syeda Kisaa Fatima, Noman Ahmed, Asifullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13116)  

**Abstract**: Crime hotspot prediction is critical for ensuring urban safety and effective law enforcement, yet it remains challenging due to the complex spatial dependencies inherent in criminal activity. The previous approaches tended to use classical algorithms such as the KDE and SVM to model data distributions and decision boundaries. The methods often fail to capture these spatial relationships, treating crime events as independent and ignoring geographical interactions. To address this, we propose a novel framework based on Graph Convolutional Networks (GCNs), which explicitly model spatial dependencies by representing crime data as a graph. In this graph, nodes represent discrete geographic grid cells and edges capture proximity relationships. Using the Chicago Crime Dataset, we engineer spatial features and train a multi-layer GCN model to classify crime types and predict high-risk zones. Our approach achieves 88% classification accuracy, significantly outperforming traditional methods. Additionally, the model generates interpretable heat maps of crime hotspots, demonstrating the practical utility of graph-based learning for predictive policing and spatial criminology. 

---
# Equitable Electronic Health Record Prediction with FAME: Fairness-Aware Multimodal Embedding 

**Authors**: Nikkie Hooman, Zhongjie Wu, Eric C. Larson, Mehak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.13104)  

**Abstract**: Electronic Health Record (EHR) data encompass diverse modalities -- text, images, and medical codes -- that are vital for clinical decision-making. To process these complex data, multimodal AI (MAI) has emerged as a powerful approach for fusing such information. However, most existing MAI models optimize for better prediction performance, potentially reinforcing biases across patient subgroups. Although bias-reduction techniques for multimodal models have been proposed, the individual strengths of each modality and their interplay in both reducing bias and optimizing performance remain underexplored. In this work, we introduce FAME (Fairness-Aware Multimodal Embeddings), a framework that explicitly weights each modality according to its fairness contribution. FAME optimizes both performance and fairness by incorporating a combined loss function. We leverage the Error Distribution Disparity Index (EDDI) to measure fairness across subgroups and propose a sign-agnostic aggregation method to balance fairness across subgroups, ensuring equitable model outcomes. We evaluate FAME with BEHRT and BioClinicalBERT, combining structured and unstructured EHR data, and demonstrate its effectiveness in terms of performance and fairness compared with other baselines across multiple EHR prediction tasks. 

---
# PRISM2: Unlocking Multi-Modal General Pathology AI with Clinical Dialogue 

**Authors**: George Shaikovski, Eugene Vorontsov, Adam Casson, Julian Viret, Eric Zimmermann, Neil Tenenholtz, Yi Kan Wang, Jan H. Bernhard, Ran A. Godrich, Juan A. Retamero, Razik Yousfi, Nicolo Fusi, Thomas J. Fuchs, Kristen Severson, Siqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13063)  

**Abstract**: Recent pathology foundation models can provide rich tile-level representations but fall short of delivering general-purpose clinical utility without further extensive model development. These models lack whole-slide image (WSI) understanding and are not trained with large-scale diagnostic data, limiting their performance on diverse downstream tasks. We introduce PRISM2, a multi-modal slide-level foundation model trained via clinical dialogue to enable scalable, generalizable pathology AI. PRISM2 is trained on nearly 700,000 specimens (2.3 million WSIs) paired with real-world clinical diagnostic reports in a two-stage process. In Stage 1, a vision-language model is trained using contrastive and captioning objectives to align whole slide embeddings with textual clinical diagnosis. In Stage 2, the language model is unfrozen to enable diagnostic conversation and extract more clinically meaningful representations from hidden states. PRISM2 achieves strong performance on diagnostic and biomarker prediction tasks, outperforming prior slide-level models including PRISM and TITAN. It also introduces a zero-shot yes/no classification approach that surpasses CLIP-style methods without prompt tuning or class enumeration. By aligning visual features with clinical reasoning, PRISM2 improves generalization on both data-rich and low-sample tasks, offering a scalable path forward for building general pathology AI agents capable of assisting diagnostic and prognostic decisions. 

---
# Stress-Testing Multimodal Foundation Models for Crystallographic Reasoning 

**Authors**: Can Polat, Hasan Kurban, Erchin Serpedin, Mustafa Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2506.13051)  

**Abstract**: Evaluating foundation models for crystallographic reasoning requires benchmarks that isolate generalization behavior while enforcing physical constraints. This work introduces a multiscale multicrystal dataset with two physically grounded evaluation protocols to stress-test multimodal generative models. The Spatial-Exclusion benchmark withholds all supercells of a given radius from a diverse dataset, enabling controlled assessments of spatial interpolation and extrapolation. The Compositional-Exclusion benchmark omits all samples of a specific chemical composition, probing generalization across stoichiometries. Nine vision--language foundation models are prompted with crystallographic images and textual context to generate structural annotations. Responses are evaluated via (i) relative errors in lattice parameters and density, (ii) a physics-consistency index penalizing volumetric violations, and (iii) a hallucination score capturing geometric outliers and invalid space-group predictions. These benchmarks establish a reproducible, physically informed framework for assessing generalization, consistency, and reliability in large-scale multimodal models. Dataset and code are available at this https URL. 

---
# Knowledge Graph Fusion with Large Language Models for Accurate, Explainable Manufacturing Process Planning 

**Authors**: Danny Hoang, David Gorsich, Matthew P. Castanier, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13026)  

**Abstract**: Precision process planning in Computer Numerical Control (CNC) machining demands rapid, context-aware decisions on tool selection, feed-speed pairs, and multi-axis routing, placing immense cognitive and procedural burdens on engineers from design specification through final part inspection. Conventional rule-based computer-aided process planning and knowledge-engineering shells freeze domain know-how into static tables, which become limited when dealing with unseen topologies, novel material states, shifting cost-quality-sustainability weightings, or shop-floor constraints such as tool unavailability and energy caps. Large language models (LLMs) promise flexible, instruction-driven reasoning for tasks but they routinely hallucinate numeric values and provide no provenance. We present Augmented Retrieval Knowledge Network Enhanced Search & Synthesis (ARKNESS), the end-to-end framework that fuses zero-shot Knowledge Graph (KG) construction with retrieval-augmented generation to deliver verifiable, numerically exact answers for CNC process planning. ARKNESS (1) automatically distills heterogeneous machining documents, G-code annotations, and vendor datasheets into augmented triple, multi-relational graphs without manual labeling, and (2) couples any on-prem LLM with a retriever that injects the minimal, evidence-linked subgraph needed to answer a query. Benchmarked on 155 industry-curated questions spanning tool sizing and feed-speed optimization, a lightweight 3B-parameter Llama-3 augmented by ARKNESS matches GPT-4o accuracy while achieving a +25 percentage point gain in multiple-choice accuracy, +22.4 pp in F1, and 8.1x ROUGE-L on open-ended responses. 

---
# Efficient Neuro-Symbolic Retrieval-Augmented Generation through Adaptive Query Routing 

**Authors**: Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.12981)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems address factual inconsistencies in Large Language Models by grounding generation in external knowledge, yet they face a fundamental efficiency problem: simple queries consume computational resources equivalent to complex multi-hop reasoning tasks. We present SymRAG, a neuro-symbolic framework that introduces adaptive query routing based on real-time complexity and system load assessments. SymRAG dynamically selects symbolic, neural, or hybrid processing paths to align resource use with query demands. Evaluated on 2,000 queries from HotpotQA and DROP using Llama-3.2-3B and Mistral-7B models, SymRAG achieves 97.6--100.0% exact match accuracy with significantly lower CPU utilization (3.6--6.2%) and processing time (0.985--3.165s). Disabling adaptive logic results in 169--1151% increase in processing time, highlighting the framework's impact. These results underscore the potential of adaptive neuro-symbolic routing for scalable, sustainable AI systems. 

---
# Forecasting Time Series with LLMs via Patch-Based Prompting and Decomposition 

**Authors**: Mayank Bumb, Anshul Vemulapalli, Sri Harsha Vardhan Prasad Jella, Anish Gupta, An La, Ryan A. Rossi, Hongjie Chen, Franck Dernoncourt, Nesreen K. Ahmed, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12953)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated new possibilities for accurate and efficient time series analysis, but prior work often required heavy fine-tuning and/or ignored inter-series correlations. In this work, we explore simple and flexible prompt-based strategies that enable LLMs to perform time series forecasting without extensive retraining or the use of a complex external architecture. Through the exploration of specialized prompting methods that leverage time series decomposition, patch-based tokenization, and similarity-based neighbor augmentation, we find that it is possible to enhance LLM forecasting quality while maintaining simplicity and requiring minimal preprocessing of data. To this end, we propose our own method, PatchInstruct, which enables LLMs to make precise and effective predictions. 

---
# HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance 

**Authors**: Rosni Vasu, Chandrayee Basu, Bhavana Dalvi Mishra, Cristina Sarasua, Peter Clark, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.12937)  

**Abstract**: Large Language models have demonstrated promising performance in research ideation across scientific domains. Hypothesis development, the process of generating a highly specific declarative statement connecting a research idea with empirical validation, has received relatively less attention. Existing approaches trivially deploy retrieval augmentation and focus only on the quality of the final output ignoring the underlying reasoning process behind ideation. We present $\texttt{HypER}$ ($\textbf{Hyp}$othesis Generation with $\textbf{E}$xplanation and $\textbf{R}$easoning), a small language model (SLM) trained for literature-guided reasoning and evidence-based hypothesis generation. $\texttt{HypER}$ is trained in a multi-task setting to discriminate between valid and invalid scientific reasoning chains in presence of controlled distractions. We find that $\texttt{HypER}$ outperformes the base model, distinguishing valid from invalid reasoning chains (+22\% average absolute F1), generates better evidence-grounded hypotheses (0.327 vs. 0.305 base model) with high feasibility and impact as judged by human experts ($>$3.5 on 5-point Likert scale). 

---
# Sectoral Coupling in Linguistic State Space 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2506.12927)  

**Abstract**: This work presents a formal framework for quantifying the internal dependencies between functional subsystems within artificial agents whose belief states are composed of structured linguistic fragments. Building on the Semantic Manifold framework, which organizes belief content into functional sectors and stratifies them across hierarchical levels of abstraction, we introduce a system of sectoral coupling constants that characterize how one cognitive sector influences another within a fixed level of abstraction. The complete set of these constants forms an agent-specific coupling profile that governs internal information flow, shaping the agent's overall processing tendencies and cognitive style. We provide a detailed taxonomy of these intra-level coupling roles, covering domains such as perceptual integration, memory access and formation, planning, meta-cognition, execution control, and affective modulation. We also explore how these coupling profiles generate feedback loops, systemic dynamics, and emergent signatures of cognitive behavior. Methodologies for inferring these profiles from behavioral or internal agent data are outlined, along with a discussion of how these couplings evolve across abstraction levels. This framework contributes a mechanistic and interpretable approach to modeling complex cognition, with applications in AI system design, alignment diagnostics, and the analysis of emergent agent behavior. 

---
# Identifying and Investigating Global News Coverage of Critical Events Such as Disasters and Terrorist Attacks 

**Authors**: Erica Cai, Xi Chen, Reagan Grey Keeney, Ethan Zuckerman, Brendan O'Connor, Przemyslaw A. Grabowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12925)  

**Abstract**: Comparative studies of news coverage are challenging to conduct because methods to identify news articles about the same event in different languages require expertise that is difficult to scale. We introduce an AI-powered method for identifying news articles based on an event FINGERPRINT, which is a minimal set of metadata required to identify critical events. Our event coverage identification method, FINGERPRINT TO ARTICLE MATCHING FOR EVENTS (FAME), efficiently identifies news articles about critical world events, specifically terrorist attacks and several types of natural disasters. FAME does not require training data and is able to automatically and efficiently identify news articles that discuss an event given its fingerprint: time, location, and class (such as storm or flood). The method achieves state-of-the-art performance and scales to massive databases of tens of millions of news articles and hundreds of events happening globally. We use FAME to identify 27,441 articles that cover 470 natural disaster and terrorist attack events that happened in 2020. To this end, we use a massive database of news articles in three languages from MediaCloud, and three widely used, expert-curated databases of critical events: EM-DAT, USGS, and GTD. Our case study reveals patterns consistent with prior literature: coverage of disasters and terrorist attacks correlates to death counts, to the GDP of a country where the event occurs, and to trade volume between the reporting country and the country where the event occurred. We share our NLP annotations and cross-country media attention data to support the efforts of researchers and media monitoring organizations. 

---
# WereWolf-Plus: An Update of Werewolf Game setting Based on DSGBench 

**Authors**: Xinyuan Xia, Yuanyi Song, Haomin Ma, Jinyu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12841)  

**Abstract**: With the rapid development of LLM-based agents, increasing attention has been given to their social interaction and strategic reasoning capabilities. However, existing Werewolf-based benchmarking platforms suffer from overly simplified game settings, incomplete evaluation metrics, and poor scalability. To address these limitations, we propose WereWolf-Plus, a multi-model, multi-dimensional, and multi-method benchmarking platform for evaluating multi-agent strategic reasoning in the Werewolf game. The platform offers strong extensibility, supporting customizable configurations for roles such as Seer, Witch, Hunter, Guard, and Sheriff, along with flexible model assignment and reasoning enhancement strategies for different roles. In addition, we introduce a comprehensive set of quantitative evaluation metrics for all special roles, werewolves, and the sheriff, and enrich the assessment dimensions for agent reasoning ability, cooperation capacity, and social influence. WereWolf-Plus provides a more flexible and reliable environment for advancing research on inference and strategic interaction within multi-agent communities. Our code is open sourced at this https URL. 

---
# Rethinking DPO: The Role of Rejected Responses in Preference Misalignment 

**Authors**: Jay Hyeon Cho, JunHyeok Oh, Myunsoo Kim, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.12725)  

**Abstract**: Direct Preference Optimization (DPO) is a simple and efficient framework that has attracted substantial attention. However, it often struggles to meet its primary objectives -- increasing the generation probability of chosen responses while reducing that of rejected responses -- due to the dominant influence of rejected responses on the loss function. This imbalance leads to suboptimal performance in promoting preferred responses. In this work, we systematically analyze the limitations of DPO and existing algorithms designed to achieve the objectives stated above. To address these limitations, we propose Bounded-DPO (BDPO), a novel method that bounds the influence of rejected responses while maintaining the original optimization structure of DPO. Through theoretical analysis and empirical evaluations, we demonstrate that BDPO achieves a balanced optimization of the chosen and rejected responses, outperforming existing algorithms. 

---
# Strategic Scaling of Test-Time Compute: A Bandit Learning Approach 

**Authors**: Bowen Zuo, Yinglun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12721)  

**Abstract**: Scaling test-time compute has emerged as an effective strategy for improving the performance of large language models. However, existing methods typically allocate compute uniformly across all queries, overlooking variation in query difficulty. To address this inefficiency, we formulate test-time compute allocation as a novel bandit learning problem and propose adaptive algorithms that estimate query difficulty on the fly and allocate compute accordingly. Compared to uniform allocation, our algorithms allocate more compute to challenging queries while maintaining accuracy on easier ones. Among challenging queries, our algorithms further learn to prioritize solvable instances, effectively reducing excessive computing on unsolvable queries. We theoretically prove that our algorithms achieve better compute efficiency than uniform allocation and empirically validate their effectiveness on math and code benchmarks. Specifically, our algorithms achieve up to an 11.10% performance improvement (15.04% relative) on the MATH-500 dataset and up to a 7.41% performance improvement (14.40% relative) on LiveCodeBench. 

---
# Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition? 

**Authors**: Xiangyang Li, Xiaopeng Li, Kuicai Dong, Quanhu Zhang, Rongju Ruan, Xinyi Dai, Xiaoshuang Liu, Shengchun Xu, Yasheng Wang, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12713)  

**Abstract**: Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(this https URL). 

---
# SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression 

**Authors**: Yucheng Li, Surin Ahn, Huiqiang Jiang, Amir H. Abdi, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12707)  

**Abstract**: Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at this https URL. 

---
# SC-SOT: Conditioning the Decoder on Diarized Speaker Information for End-to-End Overlapped Speech Recognition 

**Authors**: Yuta Hirano, Sakriani Sakti  

**Link**: [PDF](https://arxiv.org/pdf/2506.12672)  

**Abstract**: We propose Speaker-Conditioned Serialized Output Training (SC-SOT), an enhanced SOT-based training for E2E multi-talker ASR. We first probe how SOT handles overlapped speech, and we found the decoder performs implicit speaker separation. We hypothesize this implicit separation is often insufficient due to ambiguous acoustic cues in overlapping regions. To address this, SC-SOT explicitly conditions the decoder on speaker information, providing detailed information about "who spoke when". Specifically, we enhance the decoder by incorporating: (1) speaker embeddings, which allow the model to focus on the acoustic characteristics of the target speaker, and (2) speaker activity information, which guides the model to suppress non-target speakers. The speaker embeddings are derived from a jointly trained E2E speaker diarization model, mitigating the need for speaker enrollment. Experimental results demonstrate the effectiveness of our conditioning approach on overlapped speech. 

---
# MS4UI: A Dataset for Multi-modal Summarization of User Interface Instructional Videos 

**Authors**: Yuan Zang, Hao Tan, Seunghyun Yoon, Franck Dernoncourt, Jiuxiang Gu, Kushal Kafle, Chen Sun, Trung Bui  

**Link**: [PDF](https://arxiv.org/pdf/2506.12623)  

**Abstract**: We study multi-modal summarization for instructional videos, whose goal is to provide users an efficient way to learn skills in the form of text instructions and key video frames. We observe that existing benchmarks focus on generic semantic-level video summarization, and are not suitable for providing step-by-step executable instructions and illustrations, both of which are crucial for instructional videos. We propose a novel benchmark for user interface (UI) instructional video summarization to fill the gap. We collect a dataset of 2,413 UI instructional videos, which spans over 167 hours. These videos are manually annotated for video segmentation, text summarization, and video summarization, which enable the comprehensive evaluations for concise and executable video summarization. We conduct extensive experiments on our collected MS4UI dataset, which suggest that state-of-the-art multi-modal summarization methods struggle on UI video summarization, and highlight the importance of new methods for UI instructional video summarization. 

---
# StreamMel: Real-Time Zero-shot Text-to-Speech via Interleaved Continuous Autoregressive Modeling 

**Authors**: Hui Wang, Yifan Yang, Shujie Liu, Jinyu Li, Lingwei Meng, Yanqing Liu, Jiaming Zhou, Haoqin Sun, Yan Lu, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.12570)  

**Abstract**: Recent advances in zero-shot text-to-speech (TTS) synthesis have achieved high-quality speech generation for unseen speakers, but most systems remain unsuitable for real-time applications because of their offline design. Current streaming TTS paradigms often rely on multi-stage pipelines and discrete representations, leading to increased computational cost and suboptimal system performance. In this work, we propose StreamMel, a pioneering single-stage streaming TTS framework that models continuous mel-spectrograms. By interleaving text tokens with acoustic frames, StreamMel enables low-latency, autoregressive synthesis while preserving high speaker similarity and naturalness. Experiments on LibriSpeech demonstrate that StreamMel outperforms existing streaming TTS baselines in both quality and latency. It even achieves performance comparable to offline systems while supporting efficient real-time generation, showcasing broad prospects for integration with real-time speech large language models. Audio samples are available at: this https URL. 

---
# Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization 

**Authors**: Filip Sondej, Yushi Yang, Mikołaj Kniejski, Marcel Windys  

**Link**: [PDF](https://arxiv.org/pdf/2506.12484)  

**Abstract**: Language models can retain dangerous knowledge and skills even after extensive safety fine-tuning, posing both misuse and misalignment risks. Recent studies show that even specialized unlearning methods can be easily reversed. To address this, we systematically evaluate many existing and novel components of unlearning methods and identify ones crucial for irreversible unlearning.
We introduce Disruption Masking, a technique in which we only allow updating weights, where the signs of the unlearning gradient and the retaining gradient are the same. This ensures all updates are non-disruptive.
Additionally, we identify the need for normalizing the unlearning gradients, and also confirm the usefulness of meta-learning. We combine these insights into MUDMAN (Meta-Unlearning with Disruption Masking and Normalization) and validate its effectiveness at preventing the recovery of dangerous capabilities. MUDMAN outperforms the prior TAR method by 40\%, setting a new state-of-the-art for robust unlearning. 

---
# MALM: A Multi-Information Adapter for Large Language Models to Mitigate Hallucination 

**Authors**: Ao Jia, Haiming Wu, Guohui Yao, Dawei Song, Songkun Ji, Yazhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12483)  

**Abstract**: Large language models (LLMs) are prone to three types of hallucination: Input-Conflicting, Context-Conflicting and Fact-Conflicting hallucinations. The purpose of this study is to mitigate the different types of hallucination by exploiting the interdependence between them. For this purpose, we propose a Multi-Information Adapter for Large Language Models (MALM). This framework employs a tailored multi-graph learning approach designed to elucidate the interconnections between original inputs, contextual information, and external factual knowledge, thereby alleviating the three categories of hallucination within a cohesive framework. Experiments were carried out on four benchmarking datasets: HaluEval, TruthfulQA, Natural Questions, and TriviaQA. We evaluated the proposed framework in two aspects: (1) adaptability to different base LLMs on HaluEval and TruthfulQA, to confirm if MALM is effective when applied on 7 typical LLMs. MALM showed significant improvements over LLaMA-2; (2) generalizability to retrieval-augmented generation (RAG) by combining MALM with three representative retrievers (BM25, Spider and DPR) separately. Furthermore, automated and human evaluations were conducted to substantiate the correctness of experimental results, where GPT-4 and 3 human volunteers judged which response was better between LLaMA-2 and MALM. The results showed that both GPT-4 and human preferred MALM in 79.4% and 65.6% of cases respectively. The results validate that incorporating the complex interactions between the three types of hallucination through a multilayered graph attention network into the LLM generation process is effective to mitigate the them. The adapter design of the proposed approach is also proven flexible and robust across different base LLMs. 

---
# AI Flow: Perspectives, Scenarios, and Approaches 

**Authors**: Hongjun An, Sida Huang, Siqi Huang, Ruanjun Li, Yuanzhi Liang, Jiawei Shao, Zihan Wang, Cheng Yuan, Chi Zhang, Hongyuan Zhang, Wenhao Zhuang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12479)  

**Abstract**: Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems. 

---
# Plan Your Travel and Travel with Your Plan: Wide-Horizon Planning and Evaluation via LLM 

**Authors**: Dongjie Yang, Chengqiang Lu, Qimeng Wang, Xinbei Ma, Yan Gao, Yao Hu, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12421)  

**Abstract**: Travel planning is a complex task requiring the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints and preferences in the context, leading to suboptimal itineraries. We formulate this as an $L^3$ planning problem, emphasizing long context, long instruction, and long output. To tackle this, we introduce Multiple Aspects of Planning (MAoP), enabling LLMs to conduct wide-horizon thinking to solve complex planning problems. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planning models, enabling strong inference-time scalability for better performance. In addition, current benchmarks overlook travel's dynamic nature, where past events impact subsequent journeys, failing to reflect real-world feasibility. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world travel simulation. This work advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through agent-based simulation. 

---
# Model Merging for Knowledge Editing 

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yingying Zhang, Yefeng Zheng, Tianshi Ming, Yejing Wang, Wanyu Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12384)  

**Abstract**: Large Language Models (LLMs) require continuous updates to maintain accurate and current knowledge as the world evolves. While existing knowledge editing approaches offer various solutions for knowledge updating, they often struggle with sequential editing scenarios and harm the general capabilities of the model, thereby significantly hampering their practical applicability. This paper proposes a two-stage framework combining robust supervised fine-tuning (R-SFT) with model merging for knowledge editing. Our method first fine-tunes the LLM to internalize new knowledge fully, then merges the fine-tuned model with the original foundation model to preserve newly acquired knowledge and general capabilities. Experimental results demonstrate that our approach significantly outperforms existing methods in sequential editing while better preserving the original performance of the model, all without requiring any architectural changes. Code is available at: this https URL. 

---
# ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities 

**Authors**: Zhaochen Hong, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.12376)  

**Abstract**: Evaluating consistency in large language models (LLMs) is crucial for ensuring reliability, particularly in complex, multi-step interactions between humans and LLMs. Traditional self-consistency methods often miss subtle semantic changes in natural language and functional shifts in code or equations, which can accumulate over multiple transformations. To address this, we propose ConsistencyChecker, a tree-based evaluation framework designed to measure consistency through sequences of reversible transformations, including machine translation tasks and AI-assisted programming tasks. In our framework, nodes represent distinct text states, while edges correspond to pairs of inverse operations. Dynamic and LLM-generated benchmarks ensure a fair assessment of the model's generalization ability and eliminate benchmark leakage. Consistency is quantified based on similarity across different depths of the transformation tree. Experiments on eight models from various families and sizes show that ConsistencyChecker can distinguish the performance of different models. Notably, our consistency scores-computed entirely without using WMT paired data-correlate strongly (r > 0.7) with WMT 2024 auto-ranking, demonstrating the validity of our benchmark-free approach. Our implementation is available at: this https URL. 

---
# MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval 

**Authors**: Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin Zhong, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12364)  

**Abstract**: Multimodal document retrieval systems enable information access across text, images, and layouts, benefiting various domains like document-based question answering, report analysis, and interactive content summarization. Rerankers improve retrieval precision by reordering retrieved candidates. However, current multimodal reranking methods remain underexplored, with significant room for improvement in both training strategies and overall effectiveness. Moreover, the lack of explicit reasoning makes it difficult to analyze and optimize these methods further. In this paper, We propose MM-R5, a MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval, aiming to provide a more effective and reliable solution for multimodal reranking tasks. MM-R5 is trained in two stages: supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we focus on improving instruction-following and guiding the model to generate complete and high-quality reasoning chains. To support this, we introduce a novel data construction strategy that produces rich, high-quality reasoning data. In the RL stage, we design a task-specific reward framework, including a reranking reward tailored for multimodal candidates and a composite template-based reward to further refine reasoning quality. We conduct extensive experiments on MMDocIR, a challenging public benchmark spanning multiple domains. MM-R5 achieves state-of-the-art performance on most metrics and delivers comparable results to much larger models on the remaining ones. Moreover, compared to the best retrieval-only method, MM-R5 improves recall@1 by over 4%. These results validate the effectiveness of our reasoning-enhanced training pipeline. 

---
# QiMeng-Attention: SOTA Attention Operator is generated by SOTA Attention Algorithm 

**Authors**: Qirui Zhou, Shaohui Peng, Weiqiang Xiong, Haixin Chen, Yuanbo Wen, Haochen Li, Ling Li, Qi Guo, Yongwei Zhao, Ke Gao, Ruizhi Chen, Yanjun Wu, Chen Zhao, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12355)  

**Abstract**: The attention operator remains a critical performance bottleneck in large language models (LLMs), particularly for long-context scenarios. While FlashAttention is the most widely used and effective GPU-aware acceleration algorithm, it must require time-consuming and hardware-specific manual implementation, limiting adaptability across GPU architectures. Existing LLMs have shown a lot of promise in code generation tasks, but struggle to generate high-performance attention code. The key challenge is it cannot comprehend the complex data flow and computation process of the attention operator and utilize low-level primitive to exploit GPU performance.
To address the above challenge, we propose an LLM-friendly Thinking Language (LLM-TL) to help LLMs decouple the generation of high-level optimization logic and low-level implementation on GPU, and enhance LLMs' understanding of attention operator. Along with a 2-stage reasoning workflow, TL-Code generation and translation, the LLMs can automatically generate FlashAttention implementation on diverse GPUs, establishing a self-optimizing paradigm for generating high-performance attention operators in attention-centric algorithms. Verified on A100, RTX8000, and T4 GPUs, the performance of our methods significantly outshines that of vanilla LLMs, achieving a speed-up of up to 35.16x. Besides, our method not only surpasses human-optimized libraries (cuDNN and official library) in most scenarios but also extends support to unsupported hardware and data types, reducing development time from months to minutes compared with human experts. 

---
# Information Suppression in Large Language Models: Auditing, Quantifying, and Characterizing Censorship in DeepSeek 

**Authors**: Peiran Qiu, Siyi Zhou, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2506.12349)  

**Abstract**: This study examines information suppression mechanisms in DeepSeek, an open-source large language model (LLM) developed in China. We propose an auditing framework and use it to analyze the model's responses to 646 politically sensitive prompts by comparing its final output with intermediate chain-of-thought (CoT) reasoning. Our audit unveils evidence of semantic-level information suppression in DeepSeek: sensitive content often appears within the model's internal reasoning but is omitted or rephrased in the final output. Specifically, DeepSeek suppresses references to transparency, government accountability, and civic mobilization, while occasionally amplifying language aligned with state propaganda. This study underscores the need for systematic auditing of alignment, content moderation, information suppression, and censorship practices implemented into widely-adopted AI models, to ensure transparency, accountability, and equitable access to unbiased information obtained by means of these systems. 

---
# GSDNet: Revisiting Incomplete Multimodal-Diffusion from Graph Spectrum Perspective for Conversation Emotion Recognition 

**Authors**: Yuntao Shou, Jun Yao, Tao Meng, Wei Ai, Cen Chen, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12325)  

**Abstract**: Multimodal emotion recognition in conversations (MERC) aims to infer the speaker's emotional state by analyzing utterance information from multiple sources (i.e., video, audio, and text). Compared with unimodality, a more robust utterance representation can be obtained by fusing complementary semantic information from different modalities. However, the modality missing problem severely limits the performance of MERC in practical scenarios. Recent work has achieved impressive performance on modality completion using graph neural networks and diffusion models, respectively. This inspires us to combine these two dimensions through the graph diffusion model to obtain more powerful modal recovery capabilities. Unfortunately, existing graph diffusion models may destroy the connectivity and local structure of the graph by directly adding Gaussian noise to the adjacency matrix, resulting in the generated graph data being unable to retain the semantic and topological information of the original graph. To this end, we propose a novel Graph Spectral Diffusion Network (GSDNet), which maps Gaussian noise to the graph spectral space of missing modalities and recovers the missing data according to its original distribution. Compared with previous graph diffusion methods, GSDNet only affects the eigenvalues of the adjacency matrix instead of destroying the adjacency matrix directly, which can maintain the global topological information and important spectral features during the diffusion process. Extensive experiments have demonstrated that GSDNet achieves state-of-the-art emotion recognition performance in various modality loss scenarios. 

---
# Perspective on Utilizing Foundation Models for Laboratory Automation in Materials Research 

**Authors**: Kan Hatakeyama-Sato, Toshihiko Nishida, Kenta Kitamura, Yoshitaka Ushiku, Koichi Takahashi, Yuta Nabae, Teruaki Hayakawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12312)  

**Abstract**: This review explores the potential of foundation models to advance laboratory automation in the materials and chemical sciences. It emphasizes the dual roles of these models: cognitive functions for experimental planning and data analysis, and physical functions for hardware operations. While traditional laboratory automation has relied heavily on specialized, rigid systems, foundation models offer adaptability through their general-purpose intelligence and multimodal capabilities. Recent advancements have demonstrated the feasibility of using large language models (LLMs) and multimodal robotic systems to handle complex and dynamic laboratory tasks. However, significant challenges remain, including precision manipulation of hardware, integration of multimodal data, and ensuring operational safety. This paper outlines a roadmap highlighting future directions, advocating for close interdisciplinary collaboration, benchmark establishment, and strategic human-AI integration to realize fully autonomous experimental laboratories. 

---
# Can LLMs Generate High-Quality Test Cases for Algorithm Problems? TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure 

**Authors**: Zheyuan Yang, Zexi Kuang, Xue Xia, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12278)  

**Abstract**: We introduce TestCase-Eval, a new benchmark for systematic evaluation of LLMs in test-case generation. TestCase-Eval includes 500 algorithm problems and 100,000 human-crafted solutions from the Codeforces platform. It focuses on two pivotal tasks: (1) Fault Coverage, which measures how well LLM-generated test sets probe diverse input scenarios and cover a wide range of potential failure modes. (2) Fault Exposure, which evaluates whether LLMs can craft a tailored test input that reveals a specific incorrect code implementation. We provide a comprehensive assessment of 19 state-of-the-art open-source and proprietary LLMs on TestCase-Eval, offering insights into their strengths and limitations in generating effective test cases for algorithm problems. 

---
# InfoFlood: Jailbreaking Large Language Models with Information Overload 

**Authors**: Advait Yadav, Haibo Jin, Man Luo, Jun Zhuang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12274)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. However, their potential to generate harmful responses has raised significant societal and regulatory concerns, especially when manipulated by adversarial techniques known as "jailbreak" attacks. Existing jailbreak methods typically involve appending carefully crafted prefixes or suffixes to malicious prompts in order to bypass the built-in safety mechanisms of these models.
In this work, we identify a new vulnerability in which excessive linguistic complexity can disrupt built-in safety mechanisms-without the need for any added prefixes or suffixes-allowing attackers to elicit harmful outputs directly. We refer to this phenomenon as Information Overload.
To automatically exploit this vulnerability, we propose InfoFlood, a jailbreak attack that transforms malicious queries into complex, information-overloaded queries capable of bypassing built-in safety mechanisms. Specifically, InfoFlood: (1) uses linguistic transformations to rephrase malicious queries, (2) identifies the root cause of failure when an attempt is unsuccessful, and (3) refines the prompt's linguistic structure to address the failure while preserving its malicious intent.
We empirically validate the effectiveness of InfoFlood on four widely used LLMs-GPT-4o, GPT-3.5-turbo, Gemini 2.0, and LLaMA 3.1-by measuring their jailbreak success rates. InfoFlood consistently outperforms baseline attacks, achieving up to 3 times higher success rates across multiple jailbreak benchmarks. Furthermore, we demonstrate that commonly adopted post-processing defenses, including OpenAI's Moderation API, Perspective API, and SmoothLLM, fail to mitigate these attacks. This highlights a critical weakness in traditional AI safety guardrails when confronted with information overload-based jailbreaks. 

---
# ProVox: Personalization and Proactive Planning for Situated Human-Robot Collaboration 

**Authors**: Jennifer Grannen, Siddharth Karamcheti, Blake Wulfe, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2506.12248)  

**Abstract**: Collaborative robots must quickly adapt to their partner's intent and preferences to proactively identify helpful actions. This is especially true in situated settings where human partners can continually teach robots new high-level behaviors, visual concepts, and physical skills (e.g., through demonstration), growing the robot's capabilities as the human-robot pair work together to accomplish diverse tasks. In this work, we argue that robots should be able to infer their partner's goals from early interactions and use this information to proactively plan behaviors ahead of explicit instructions from the user. Building from the strong commonsense priors and steerability of large language models, we introduce ProVox ("Proactive Voice"), a novel framework that enables robots to efficiently personalize and adapt to individual collaborators. We design a meta-prompting protocol that empowers users to communicate their distinct preferences, intent, and expected robot behaviors ahead of starting a physical interaction. ProVox then uses the personalized prompt to condition a proactive language model task planner that anticipates a user's intent from the current interaction context and robot capabilities to suggest helpful actions; in doing so, we alleviate user burden, minimizing the amount of time partners spend explicitly instructing and supervising the robot. We evaluate ProVox through user studies grounded in household manipulation tasks (e.g., assembling lunch bags) that measure the efficiency of the collaboration, as well as features such as perceived helpfulness, ease of use, and reliability. Our analysis suggests that both meta-prompting and proactivity are critical, resulting in 38.7% faster task completion times and 31.9% less user burden relative to non-active baselines. Supplementary material, code, and videos can be found at this https URL. 

---
# Datrics Text2SQL: A Framework for Natural Language to SQL Query Generation 

**Authors**: Tetiana Gladkykh, Kyrylo Kirykov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12234)  

**Abstract**: Text-to-SQL systems enable users to query databases using natural language, democratizing access to data analytics. However, they face challenges in understanding ambiguous phrasing, domain-specific vocabulary, and complex schema relationships. This paper introduces Datrics Text2SQL, a Retrieval-Augmented Generation (RAG)-based framework designed to generate accurate SQL queries by leveraging structured documentation, example-based learning, and domain-specific rules. The system builds a rich Knowledge Base from database documentation and question-query examples, which are stored as vector embeddings and retrieved through semantic similarity. It then uses this context to generate syntactically correct and semantically aligned SQL code. The paper details the architecture, training methodology, and retrieval logic, highlighting how the system bridges the gap between user intent and database structure without requiring SQL expertise. 

---
# Zero-Shot Scene Understanding with Multimodal Large Language Models for Automated Vehicles 

**Authors**: Mohammed Elhenawy, Shadi Jaradat, Taqwa I. Alhadidi, Huthaifa I. Ashqar, Ahmed Jaber, Andry Rakotonirainy, Mohammad Abu Tami  

**Link**: [PDF](https://arxiv.org/pdf/2506.12232)  

**Abstract**: Scene understanding is critical for various downstream tasks in autonomous driving, including facilitating driver-agent communication and enhancing human-centered explainability of autonomous vehicle (AV) decisions. This paper evaluates the capability of four multimodal large language models (MLLMs), including relatively small models, to understand scenes in a zero-shot, in-context learning setting. Additionally, we explore whether combining these models using an ensemble approach with majority voting can enhance scene understanding performance. Our experiments demonstrate that GPT-4o, the largest model, outperforms the others in scene understanding. However, the performance gap between GPT-4o and the smaller models is relatively modest, suggesting that advanced techniques such as improved in-context learning, retrieval-augmented generation (RAG), or fine-tuning could further optimize the smaller models' performance. We also observe mixed results with the ensemble approach: while some scene attributes show improvement in performance metrics such as F1-score, others experience a decline. These findings highlight the need for more sophisticated ensemble techniques to achieve consistent gains across all scene attributes. This study underscores the potential of leveraging MLLMs for scene understanding and provides insights into optimizing their performance for autonomous driving applications. 

---
# From Emergence to Control: Probing and Modulating Self-Reflection in Language Models 

**Authors**: Xudong Zhu, Jiachen Jiang, Mohammad Mahdi Khalili, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12217)  

**Abstract**: Self-reflection -- the ability of a large language model (LLM) to revisit, evaluate, and revise its own reasoning -- has recently emerged as a powerful behavior enabled by reinforcement learning with verifiable rewards (RLVR). While self-reflection correlates with improved reasoning accuracy, its origin and underlying mechanisms remain poorly understood. In this work, {\it we first show that self-reflection is not exclusive to RLVR fine-tuned models: it already emerges, albeit rarely, in pretrained models}. To probe this latent ability, we introduce Reflection-Inducing Probing, a method that injects reflection-triggering reasoning traces from fine-tuned models into pretrained models. This intervention raises self-reflection frequency of Qwen2.5 from 0.6\% to 18.6\%, revealing a hidden capacity for reflection. Moreover, our analysis of internal representations shows that both pretrained and fine-tuned models maintain hidden states that distinctly separate self-reflective from non-reflective contexts. Leveraging this observation, {\it we then construct a self-reflection vector, a direction in activation space associated with self-reflective reasoning}. By manipulating this vector, we enable bidirectional control over the self-reflective behavior for both pretrained and fine-tuned models. Experiments across multiple reasoning benchmarks show that enhancing these vectors improves reasoning performance by up to 12\%, while suppressing them reduces computational cost, providing a flexible mechanism to navigate the trade-off between reasoning quality and efficiency without requiring additional training. Our findings further our understanding of self-reflection and support a growing body of work showing that understanding model internals can enable precise behavioral control. 

---
# Generative or Discriminative? Revisiting Text Classification in the Era of Transformers 

**Authors**: Siva Rajesh Kasa, Karan Gupta, Sumegh Roychowdhury, Ashutosh Kumar, Yaswanth Biruduraju, Santhosh Kumar Kasa, Nikhil Priyatam Pattisapu, Arindam Bhattacharya, Shailendra Agarwal, Vijay huddar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12181)  

**Abstract**: The comparison between discriminative and generative classifiers has intrigued researchers since Efron's seminal analysis of logistic regression versus discriminant analysis. While early theoretical work established that generative classifiers exhibit lower sample complexity but higher asymptotic error in simple linear settings, these trade-offs remain unexplored in the transformer era. We present the first comprehensive evaluation of modern generative and discriminative architectures - Auto-regressive modeling, Masked Language Modeling, Discrete Diffusion, and Encoders for text classification. Our study reveals that the classical 'two regimes' phenomenon manifests distinctly across different architectures and training paradigms. Beyond accuracy, we analyze sample efficiency, calibration, noise robustness, and ordinality across diverse scenarios. Our findings offer practical guidance for selecting the most suitable modeling approach based on real-world constraints such as latency and data limitations. 

---
# Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs): A Feynman-Based Architecture for Continuous Learning Over Streaming Data 

**Authors**: Oscar Boullosa Dapena  

**Link**: [PDF](https://arxiv.org/pdf/2506.12111)  

**Abstract**: Real-time continuous learning over streaming data remains a central challenge in deep learning and AI systems. Traditional gradient-based models such as backpropagation through time (BPTT) face computational and stability limitations when dealing with temporally unbounded data. In this paper, we introduce a novel architecture, Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs), which leverages the Feynman technique of differentiation under the integral sign to formulate neural updates as integrals over historical data. This reformulation allows for smoother, more stable learning dynamics that are both physically interpretable and computationally tractable. Inspired by Feynman's path integral formalism and compatible with quantum gradient estimation frameworks, QIDINNs open a path toward hybrid classical-quantum neural computation. We demonstrate our model's effectiveness on synthetic and real-world streaming tasks, and we propose directions for quantum extensions and scalable implementations. 

---
# The CAISAR Platform: Extending the Reach of Machine Learning Specification and Verification 

**Authors**: Michele Alberti, François Bobot, Julien Girard-Satabin, Alban Grastien, Aymeric Varasse, Zakaria Chihani  

**Link**: [PDF](https://arxiv.org/pdf/2506.12084)  

**Abstract**: The formal specification and verification of machine learning programs saw remarkable progress in less than a decade, leading to a profusion of tools. However, diversity may lead to fragmentation, resulting in tools that are difficult to compare, except for very specific benchmarks. Furthermore, this progress is heavily geared towards the specification and verification of a certain class of property, that is, local robustness properties. But while provers are becoming more and more efficient at solving local robustness properties, even slightly more complex properties, involving multiple neural networks for example, cannot be expressed in the input languages of winners of the International Competition of Verification of Neural Networks VNN-Comp. In this tool paper, we present CAISAR, an open-source platform dedicated to machine learning specification and verification. We present its specification language, suitable for modelling complex properties on neural networks, support vector machines and boosted trees. We show on concrete use-cases how specifications written in this language are automatically translated to queries to state-of-the-art provers, notably by using automated graph editing techniques, making it possible to use their off-the-shelf versions. The artifact to reproduce the paper claims is available at the following DOI: this https URL 

---
# Modeling Earth-Scale Human-Like Societies with One Billion Agents 

**Authors**: Haoxiang Guan, Jiyan He, Liyang Fan, Zhenzhen Ren, Shaobin He, Xin Yu, Yuan Chen, Shuxin Zheng, Tie-Yan Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12078)  

**Abstract**: Understanding how complex societal behaviors emerge from individual cognition and interactions requires both high-fidelity modeling of human behavior and large-scale simulations. Traditional agent-based models (ABMs) have been employed to study these dynamics for decades, but are constrained by simplified agent behaviors that fail to capture human complexity. Recent advances in large language models (LLMs) offer new opportunities by enabling agents to exhibit sophisticated social behaviors that go beyond rule-based logic, yet face significant scaling challenges. Here we present Light Society, an agent-based simulation framework that advances both fronts, efficiently modeling human-like societies at planetary scale powered by LLMs. Light Society formalizes social processes as structured transitions of agent and environment states, governed by a set of LLM-powered simulation operations, and executed through an event queue. This modular design supports both independent and joint component optimization, supporting efficient simulation of societies with over one billion agents. Large-scale simulations of trust games and opinion propagation--spanning up to one billion agents--demonstrate Light Society's high fidelity and efficiency in modeling social trust and information diffusion, while revealing scaling laws whereby larger simulations yield more stable and realistic emergent behaviors. 

---
# Artificial Intelligence and Civil Discourse: How LLMs Moderate Climate Change Conversations 

**Authors**: Wenlu Fan, Wentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12077)  

**Abstract**: As large language models (LLMs) become increasingly integrated into online platforms and digital communication spaces, their potential to influence public discourse - particularly in contentious areas like climate change - requires systematic investigation. This study examines how LLMs naturally moderate climate change conversations through their distinct communicative behaviors. We conduct a comparative analysis of conversations between LLMs and human users on social media platforms, using five advanced models: three open-source LLMs (Gemma, Llama 3, and Llama 3.3) and two commercial systems (GPT-4o by OpenAI and Claude 3.5 by Anthropic). Through sentiment analysis, we assess the emotional characteristics of responses from both LLMs and humans. The results reveal two key mechanisms through which LLMs moderate discourse: first, LLMs consistently display emotional neutrality, showing far less polarized sentiment than human users. Second, LLMs maintain lower emotional intensity across contexts, creating a stabilizing effect in conversations. These findings suggest that LLMs possess inherent moderating capacities that could improve the quality of public discourse on controversial topics. This research enhances our understanding of how AI might support more civil and constructive climate change discussions and informs the design of AI-assisted communication tools. 

---
# Seamless Dysfluent Speech Text Alignment for Disordered Speech Analysis 

**Authors**: Zongli Ye, Jiachen Lian, Xuanru Zhou, Jinming Zhang, Haodong Li, Shuhe Li, Chenxu Guo, Anaisha Das, Peter Park, Zoe Ezzes, Jet Vonk, Brittany Morin, Rian Bogley, Lisa Wauters, Zachary Miller, Maria Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.12073)  

**Abstract**: Accurate alignment of dysfluent speech with intended text is crucial for automating the diagnosis of neurodegenerative speech disorders. Traditional methods often fail to model phoneme similarities effectively, limiting their performance. In this work, we propose Neural LCS, a novel approach for dysfluent text-text and speech-text alignment. Neural LCS addresses key challenges, including partial alignment and context-aware similarity mapping, by leveraging robust phoneme-level modeling. We evaluate our method on a large-scale simulated dataset, generated using advanced data simulation techniques, and real PPA data. Neural LCS significantly outperforms state-of-the-art models in both alignment accuracy and dysfluent speech segmentation. Our results demonstrate the potential of Neural LCS to enhance automated systems for diagnosing and analyzing speech disorders, offering a more accurate and linguistically grounded solution for dysfluent speech alignment. 

---
# CMT-LLM: Contextual Multi-Talker ASR Utilizing Large Language Models 

**Authors**: Jiajun He, Naoki Sawada, Koichi Miyazaki, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2506.12059)  

**Abstract**: In real-world applications, automatic speech recognition (ASR) systems must handle overlapping speech from multiple speakers and recognize rare words like technical terms. Traditional methods address multi-talker ASR and contextual biasing separately, limiting performance in complex scenarios. We propose a unified framework that combines multi-talker overlapping speech recognition and contextual biasing into a single task. Our ASR method integrates pretrained speech encoders and large language models (LLMs), using optimized finetuning strategies. We also introduce a two-stage filtering algorithm to efficiently identify relevant rare words from large biasing lists and incorporate them into the LLM's prompt input, enhancing rare word recognition. Experiments show that our approach outperforms traditional contextual biasing methods, achieving a WER of 7.9% on LibriMix and 32.9% on AMI SDM when the biasing size is 1,000, demonstrating its effectiveness in complex speech scenarios. 

---
