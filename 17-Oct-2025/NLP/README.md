# Attention Is All You Need for KV Cache in Diffusion LLMs 

**Authors**: Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14973)  

**Abstract**: This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs. 

---
# TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar 

**Authors**: Yinxi Li, Yuntian Deng, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.14972)  

**Abstract**: Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TokDrift, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs. 

---
# LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training 

**Authors**: Yiming Wang, Da Yin, Yuedong Cui, Ruichen Zheng, Zhiqian Li, Zongyu Lin, Di Wu, Xueqing Wu, Chenchen Ye, Yu Zhou, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14969)  

**Abstract**: Digital agents require diverse, large-scale UI trajectories to generalize across real-world tasks, yet collecting such data is prohibitively expensive in both human annotation, infra and engineering perspectives. To this end, we introduce $\textbf{UI-Simulator}$, a scalable paradigm that generates structured UI states and transitions to synthesize training trajectories at scale. Our paradigm integrates a digital world simulator for diverse UI states, a guided rollout process for coherent exploration, and a trajectory wrapper that produces high-quality and diverse trajectories for agent training. We further propose $\textbf{UI-Simulator-Grow}$, a targeted scaling strategy that enables more rapid and data-efficient scaling by prioritizing high-impact tasks and synthesizes informative trajectory variants. Experiments on WebArena and AndroidWorld show that UI-Simulator rivals or surpasses open-source agents trained on real UIs with significantly better robustness, despite using weaker teacher models. Moreover, UI-Simulator-Grow matches the performance of Llama-3-70B-Instruct using only Llama-3-8B-Instruct as the base model, highlighting the potential of targeted synthesis scaling paradigm to continuously and efficiently enhance the digital agents. 

---
# Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents 

**Authors**: Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying  

**Link**: [PDF](https://arxiv.org/pdf/2510.14967)  

**Abstract**: Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency. 

---
# DialectGen: Benchmarking and Improving Dialect Robustness in Multimodal Generation 

**Authors**: Yu Zhou, Sohyun An, Haikang Deng, Da Yin, Clark Peng, Cho-Jui Hsieh, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14949)  

**Abstract**: Contact languages like English exhibit rich regional variations in the form of dialects, which are often used by dialect speakers interacting with generative models. However, can multimodal generative models effectively produce content given dialectal textual input? In this work, we study this question by constructing a new large-scale benchmark spanning six common English dialects. We work with dialect speakers to collect and verify over 4200 unique prompts and evaluate on 17 image and video generative models. Our automatic and human evaluation results show that current state-of-the-art multimodal generative models exhibit 32.26% to 48.17% performance degradation when a single dialect word is used in the prompt. Common mitigation methods such as fine-tuning and prompt rewriting can only improve dialect performance by small margins (< 7%), while potentially incurring significant performance degradation in Standard American English (SAE). To this end, we design a general encoder-based mitigation strategy for multimodal generative models. Our method teaches the model to recognize new dialect features while preserving SAE performance. Experiments on models such as Stable Diffusion 1.5 show that our method is able to simultaneously raise performance on five dialects to be on par with SAE (+34.4%), while incurring near zero cost to SAE performance. 

---
# MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics 

**Authors**: Yuxing Lu, Xukai Zhao, J. Ben Tamo, Micky C. Nnamdi, Rui Peng, Shuang Zeng, Xingyu Hu, Jinzhuo Wang, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14944)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities on general text; however, their proficiency in specialized scientific domains that require deep, interconnected knowledge remains largely uncharacterized. Metabolomics presents unique challenges with its complex biochemical pathways, heterogeneous identifier systems, and fragmented databases. To systematically evaluate LLM capabilities in this domain, we introduce MetaBench, the first benchmark for metabolomics assessment. Curated from authoritative public resources, MetaBench evaluates five capabilities essential for metabolomics research: knowledge, understanding, grounding, reasoning, and research. Our evaluation of 25 open- and closed-source LLMs reveals distinct performance patterns across metabolomics tasks: while models perform well on text generation tasks, cross-database identifier grounding remains challenging even with retrieval augmentation. Model performance also decreases on long-tail metabolites with sparse annotations. With MetaBench, we provide essential infrastructure for developing and evaluating metabolomics AI systems, enabling systematic progress toward reliable computational tools for metabolomics research. 

---
# LaSeR: Reinforcement Learning with Last-Token Self-Rewarding 

**Authors**: Wenkai Yang, Weijie Liu, Ruobing Xie, Yiju Guo, Lulu Wu, Saiyong Yang, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14943)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). To address the lack of verification signals at test time, prior studies incorporate the training of model's self-verification capability into the standard RLVR process, thereby unifying reasoning and verification capabilities within a single LLM. However, previous practice requires the LLM to sequentially generate solutions and self-verifications using two separate prompt templates, which significantly reduces efficiency. In this work, we theoretically reveal that the closed-form solution to the RL objective of self-verification can be reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model's next-token log-probability assigned to any pre-specified token at the solution's last token and a pre-calculated constant, scaled by the KL coefficient. Based on this insight, we propose LaSeR (Reinforcement Learning with Last-Token Self-Rewarding), an algorithm that simply augments the original RLVR loss with a MSE loss that aligns the last-token self-rewarding scores with verifier-based reasoning rewards, jointly optimizing the reasoning and self-rewarding capabilities of LLMs. The optimized self-rewarding scores can be utilized in both training and testing to enhance model performance. Notably, our algorithm derives these scores from the predicted next-token probability distribution of the last token immediately after generation, incurring only the minimal extra cost of one additional token inference. Experiments show that our method not only improves the model's reasoning performance but also equips it with remarkable self-rewarding capability, thereby boosting its inference-time scaling performance. 

---
# AI-Powered Early Diagnosis of Mental Health Disorders from Real-World Clinical Conversations 

**Authors**: Jianfeng Zhu, Julina Maharjan, Xinyu Li, Karin G. Coifman, Ruoming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14937)  

**Abstract**: Mental health disorders remain among the leading cause of disability worldwide, yet conditions such as depression, anxiety, and Post-Traumatic Stress Disorder (PTSD) are frequently underdiagnosed or misdiagnosed due to subjective assessments, limited clinical resources, and stigma and low awareness. In primary care settings, studies show that providers misidentify depression or anxiety in over 60% of cases, highlighting the urgent need for scalable, accessible, and context-aware diagnostic tools that can support early detection and intervention. In this study, we evaluate the effectiveness of machine learning models for mental health screening using a unique dataset of 553 real-world, semistructured interviews, each paried with ground-truth diagnoses for major depressive episodes (MDE), anxiety disorders, and PTSD. We benchmark multiple model classes, including zero-shot prompting with GPT-4.1 Mini and MetaLLaMA, as well as fine-tuned RoBERTa models using LowRank Adaptation (LoRA). Our models achieve over 80% accuracy across diagnostic categories, with especially strongperformance on PTSD (up to 89% accuracy and 98% recall). We also find that using shorter context, focused context segments improves recall, suggesting that focused narrative cues enhance detection sensitivity. LoRA fine-tuning proves both efficient and effective, with lower-rank configurations (e.g., rank 8 and 16) maintaining competitive performance across evaluation metrics. Our results demonstrate that LLM-based models can offer substantial improvements over traditional self-report screening tools, providing a path toward low-barrier, AI-powerd early diagnosis. This work lays the groundwork for integrating machine learning into real-world clinical workflows, particularly in low-resource or high-stigma environments where access to timely mental health care is most limited. 

---
# Predicting Task Performance with Context-aware Scaling Laws 

**Authors**: Kyle Montgomery, David Park, Jianhong Tu, Michael Bendersky, Beliz Gunel, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14919)  

**Abstract**: Scaling laws have transformed our understanding of large language models by linking upstream metrics like cross-entropy loss to design factors such as model size, training data, and compute. However, these conventional laws fail to capture downstream task performance, where context plays a critical role. In this work, we propose a straightforward, interpretable framework that jointly models downstream performance as a function of the training compute and the provided context. We empirically validate our framework by fitting it on the observed downstream performance of extended-context variants of Llama-2-7B and Llama-2-13B across 65,500 unique instances spanning three tasks: arithmetic reasoning, common sense reasoning, and machine translation. Our results demonstrate that our framework accurately models in-distribution downstream performance, generalizes across three orders of magnitude in training compute, and reliably extrapolates performance as the amount of context increases. These findings offer valuable insights into the interplay between training compute and context utilization, providing guidance for designing more efficient long-context LLMs for diverse downstream tasks. Our code is available at this https URL. 

---
# Harmonizing Diverse Models: A Layer-wise Merging Strategy for Consistent Generation 

**Authors**: Xujun Peng, Anoop Kumar, Jingyu Wu, Parker Glenn, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14915)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems leverage Large Language Models (LLMs) to generate accurate and reliable responses that are grounded in retrieved context. However, LLMs often generate inconsistent outputs for semantically equivalent inputs, a problem compounded by the scarcity of consistency-focused training data and the limitations of current fine-tuning techniques in enhancing output consistency. We propose a new approach combining systematic synthetic data generation, triplet loss for better embeddings, and a novel layer-wise model merging approach. Using consistency-aware weights derived from intermediate layer activations, our method effectively integrates knowledge from specialized models. Experimental results how that our merged model significantly enhances output consistency, achieving a ~47.5\% improvement in response similarity over the baseline, thus offering a practical solution for increasing the reliability of an industrial RAG system. 

---
# From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR 

**Authors**: Erwei Wang, Samuel Bayliss, Andra Bisca, Zachary Blair, Sangeeta Chowdhary, Kristof Denolf, Jeff Fifield, Brandon Freiberger, Erika Hunhoff, Phil James-Roxby, Jack Lo, Joseph Melber, Stephen Neuendorffer, Eddie Richter, Andre Rosti, Javier Setoain, Gagandeep Singh, Endri Taka, Pranathi Vasireddy, Zhewen Yu, Niansong Zhang, Jinming Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14871)  

**Abstract**: General-purpose compilers abstract away parallelism, locality, and synchronization, limiting their effectiveness on modern spatial architectures. As modern computing architectures increasingly rely on fine-grained control over data movement, execution order, and compute placement for performance, compiler infrastructure must provide explicit mechanisms for orchestrating compute and data to fully exploit such architectures. We introduce MLIR-AIR, a novel, open-source compiler stack built on MLIR that bridges the semantic gap between high-level workloads and fine-grained spatial architectures such as AMD's NPUs. MLIR-AIR defines the AIR dialect, which provides structured representations for asynchronous and hierarchical operations across compute and memory resources. AIR primitives allow the compiler to orchestrate spatial scheduling, distribute computation across hardware regions, and overlap communication with computation without relying on ad hoc runtime coordination or manual scheduling. We demonstrate MLIR-AIR's capabilities through two case studies: matrix multiplication and the multi-head attention block from the LLaMA 2 model. For matrix multiplication, MLIR-AIR achieves up to 78.7% compute efficiency and generates implementations with performance almost identical to state-of-the-art, hand-optimized matrix multiplication written using the lower-level, close-to-metal MLIR-AIE framework. For multi-head attention, we demonstrate that the AIR interface supports fused implementations using approximately 150 lines of code, enabling tractable expression of complex workloads with efficient mapping to spatial hardware. MLIR-AIR transforms high-level structured control flow into spatial programs that efficiently utilize the compute fabric and memory hierarchy of an NPU, leveraging asynchronous execution, tiling, and communication overlap through compiler-managed scheduling. 

---
# Midtraining Bridges Pretraining and Posttraining Distributions 

**Authors**: Emmy Liu, Graham Neubig, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.14865)  

**Abstract**: Recently, many language models have been pretrained with a "midtraining" phase, in which higher quality, often instruction-formatted data, is mixed in at the end of pretraining. Despite the popularity of this practice, there is little scientific understanding of this phase of model training or why it is effective. In this work, we conduct the first systematic investigation of midtraining through controlled experiments with language models pretrained from scratch and fine-tuned on supervised finetuning datasets in different domains. We find that when compared after supervised fine-tuning, the effectiveness of midtraining is highest in the math and code domains, where midtraining can best reduce the syntactic gap between pretraining and posttraining data. In these cases, midtraining consistently outperforms continued pretraining in both in-domain validation loss as well as pretraining data forgetting after posttraining. We conduct ablations on the starting time of the midtraining phase and mixture weights of the midtraining data, using code midtraining as a case study, and find that timing has a greater impact than mixture weights, with earlier introduction of specialized data, yielding greater benefits in-domain as well as preserving general language modeling better. These findings establish midtraining as a domain adaptation technique that compared to continued pretraining yields better performance through reduced forgetting. 

---
# Rewiring Experts on the Fly:Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models 

**Authors**: Guinan Su, Yanwu Yang, Li Shen, Lu Yin, Shiwei Liu, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2510.14853)  

**Abstract**: Mixture-of-Experts (MoE) models achieve efficient scaling through sparse expert activation, but often suffer from suboptimal routing decisions due to distribution shifts in deployment. While existing test-time adaptation methods could potentially address these issues, they primarily focus on dense models and require access to external data, limiting their practical applicability to MoE architectures. However, we find that, instead of relying on reference data, we can optimize MoE expert selection on-the-fly based only on input context. As such, we propose \textit{a data-free, online test-time framework} that continuously adapts MoE routing decisions during text generation without external supervision or data. Our method cycles between two phases: During the prefill stage, and later in regular intervals, we optimize the routing decisions of the model using self-supervision based on the already generated sequence. Then, we generate text as normal, maintaining the modified router until the next adaption. We implement this through lightweight additive vectors that only update router logits in selected layers, maintaining computational efficiency while preventing over-adaptation. The experimental results show consistent performance gains on challenging reasoning tasks while maintaining robustness to context shifts. For example, our method achieves a 5.5\% improvement on HumanEval with OLMoE. Furthermore, owing to its plug-and-play property, our method naturally complements existing test-time scaling techniques, e.g., achieving 6\% average gains when incorporated with self-consistency on DeepSeek-V2-Lite. 

---
# Supervised Fine-Tuning or Contrastive Learning? Towards Better Multimodal LLM Reranking 

**Authors**: Ziqi Dai, Xin Zhang, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14824)  

**Abstract**: In information retrieval, training reranking models mainly focuses on two types of objectives: metric learning (e.g. contrastive loss to increase the predicted scores on relevant query-document pairs) and classification (binary label prediction of relevance vs. irrelevance). For BERT-style encoders, various studies have shown that contrastive learning (CL) can be more effective than discriminative (classification) learning. However, for large language models (LLMs), classification via supervised fine-tuning (SFT), which predicts ''yes'' (resp. ''no'') token for relevant (resp. irrelevant) pairs, appears more promising as it aligns well with the generative nature of LLMs. This divergence raises a central question: which objective is intrinsically better suited to LLM-based reranking, and what mechanism underlies the difference? In this work, we conduct a comprehensive comparison and analysis between CL and SFT for reranking, taking the universal multimodal retrieval (UMR) as the experimental playground. We first decompose the objectives into two components: weight, which controls the magnitude of those updates, and direction, which guides the model updates, then present a unified framework for understanding their interactions. Through probing experiments, we find that SFT provides a substantially stronger weighting scheme than CL, whereas the preferred scoring direction shows no clear winner. Taken together, these results point to a consistent advantage of SFT over CL for LLM reranking. To further validate our findings, we conduct large-scale training with SFT and present new state-of-the-art rerankers on the MRB benchmark. We also provide ablations on SFT settings and expect our findings to benefit future research and applications in this area. 

---
# Finding Answers in Thought Matters: Revisiting Evaluation on Large Language Models with Reasoning 

**Authors**: Hwiyeol Jo, Joosung Lee, Jaehone Lee, Sang-Woo Lee, Joonsuk Park, Kang Min Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14773)  

**Abstract**: Evaluating generative models, such as large language models (LLMs), commonly involves question-answering tasks where the final answer is selected based on probability of answer choices. On the other hand, for models requiring reasoning, the method of answer extraction plays a critical role. Our research reveals that the performance of reasoning models and their final answer distributions are highly sensitive to the answer extraction algorithm employed. In order to mitigate this, we propose a basic framework: Answer Regeneration. The method uses an additional model inference, providing the prior input and output prefaced by the prompt "Answer:". The final answer is then selected or extracted from the regenerated output. We show that this extraction-rule-agnostic approach exhibits improved performance and enhanced robustness. Furthermore, we have applied this framework to general math problems and open-ended question answering tasks. Our analysis and this framework could offer a more reliable results for model evaluation. 

---
# COIG-Writer: A High-Quality Dataset for Chinese Creative Writing with Thought Processes 

**Authors**: Yunwen Li, Shuangshuang Ying, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Tianyu Zheng, Xeron Du, Qiguang Chen, Jiajun Shi, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Stephen Huang, Wanxiang Che, Chenghua Lin, Eli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14763)  

**Abstract**: Large language models exhibit systematic deficiencies in creative writing, particularly in non-English contexts where training data is scarce and lacks process-level supervision. We present COIG-Writer, a novel Chinese creative writing dataset that captures both diverse outputs and their underlying thought processes through systematic reverse-engineering of high-quality texts. Unlike existing datasets that provide only input-output pairs, COIG-Writer comprises 1,665 meticulously curated triplets spanning 51 genres, each containing: (1) a reverse-engineered prompt, (2) detailed creative reasoning documenting decision-making processes, and (3) the final text. Through comprehensive experiments, we identify a two-component model of creative writing: narrative logic (provided by process supervision) and linguistic expression (maintained by general-purpose data). Our findings reveal three critical insights: (1) Process supervision is highly effective but requires stabilization with general data. A ratio of at least one creative sample to twelve general samples is needed to achieve optimal performance; below this threshold, the win rate progressively degrades (from 62.75% down to 35.78%)., (2) creative capabilities are culturally-bound with no cross-lingual transfer (89.26pp gap between Chinese and English performance), and (3) lexical diversity inversely correlates with creative quality (TTR paradox), suggesting high diversity signals compensatory behavior for logical deficiencies. These findings establish that creative excellence emerges from the interaction between logical scaffolding and linguistic grounding, analogous to how mathematical reasoning enhances but cannot replace linguistic competence in foundation models. 

---
# Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code 

**Authors**: Manar Abdelatty, Maryam Nouh, Jacob K. Rosenstein, Sherief Reda  

**Link**: [PDF](https://arxiv.org/pdf/2510.14756)  

**Abstract**: Large Language Models (LLMs) are increasingly used to automate hardware design tasks, including the generation of Verilog code. While early benchmarks focus primarily on functional correctness, efficient hardware design demands additional optimization for synthesis metrics such as area, delay, and power. Existing benchmarks fall short in evaluating these aspects comprehensively: they often lack optimized baselines or testbenches for verification. To address these gaps, we present Pluto, a benchmark and evaluation framework designed to assess the efficiency of LLM-generated Verilog designs. Pluto presents a comprehensive evaluation set of 114 problems with self-checking testbenches and multiple Pareto-optimal reference implementations. Experimental results show that state-of-the-art LLMs can achieve high functional correctness, reaching 78.3\% at pass@1, but their synthesis efficiency still lags behind expert-crafted implementations, with area efficiency of 63.8\%, delay efficiency of 65.9\%, and power efficiency of 64.0\% at eff@1. This highlights the need for efficiency-aware evaluation frameworks such as Pluto to drive progress in hardware-focused LLM research. 

---
# AutoRubric-R1V: Rubric-Based Generative Rewards for Faithful Multimodal Reasoning 

**Authors**: Mengzhao Jia, Zhihan Zhang, Ignacio Cases, Zheyuan Liu, Meng Jiang, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14738)  

**Abstract**: Multimodal large language models (MLLMs) have rapidly advanced from perception tasks to complex multi-step reasoning, yet reinforcement learning with verifiable rewards (RLVR) often leads to spurious reasoning since only the final-answer correctness is rewarded. To address this limitation, we propose AutoRubric-R1V, a framework that integrates RLVR with process-level supervision through automatically collected rubric-based generative rewards. Our key innovation lies in a scalable self-aggregation method that distills consistent reasoning checkpoints from successful trajectories, enabling problem-specific rubric construction without human annotation or stronger teacher models. By jointly leveraging rubric-based and outcome rewards, AutoRubric-R1V achieves state-of-the-art performance on six multimodal reasoning benchmarks and substantially improves reasoning faithfulness in dedicated evaluations. 

---
# Speculative Model Risk in Healthcare AI: Using Storytelling to Surface Unintended Harms 

**Authors**: Xingmeng Zhao, Dan Schumacher, Veronica Rammouz, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2510.14718)  

**Abstract**: Artificial intelligence (AI) is rapidly transforming healthcare, enabling fast development of tools like stress monitors, wellness trackers, and mental health chatbots. However, rapid and low-barrier development can introduce risks of bias, privacy violations, and unequal access, especially when systems ignore real-world contexts and diverse user needs. Many recent methods use AI to detect risks automatically, but this can reduce human engagement in understanding how harms arise and who they affect. We present a human-centered framework that generates user stories and supports multi-agent discussions to help people think creatively about potential benefits and harms before deployment. In a user study, participants who read stories recognized a broader range of harms, distributing their responses more evenly across all 13 harm types. In contrast, those who did not read stories focused primarily on privacy and well-being (58.3%). Our findings show that storytelling helped participants speculate about a broader range of harms and benefits and think more creatively about AI's impact on users. 

---
# Semantic Prosody in Machine Translation: the English-Chinese Case of Passive Structures 

**Authors**: Xinyue Ma, Pol Pastells, Mireia Farrús, Mariona Taulé  

**Link**: [PDF](https://arxiv.org/pdf/2510.14662)  

**Abstract**: Semantic prosody is a collocational meaning formed through the co-occurrence of a linguistic unit and a consistent series of collocates, which should be treated separately from semantic meaning. Since words that are literal translations of each other may have different semantic prosody, more attention should be paid to this linguistic property to generate accurate translations. However, current machine translation models cannot handle this problem. To bridge the gap, we propose an approach to teach machine translation models about semantic prosody of a specific structure. We focus on Chinese BEI passives and create a dataset of English-Chinese sentence pairs with the purpose of demonstrating the negative semantic prosody of BEI passives. Then we fine-tune OPUS-MT, NLLB-600M and mBART50 models with our dataset for the English-Chinese translation task. Our results show that fine-tuned MT models perform better on using BEI passives for translating unfavourable content and avoid using it for neutral and favourable content. Also, in NLLB-600M, which is a multilingual model, this knowledge of semantic prosody can be transferred from English-Chinese translation to other language pairs, such as Spanish-Chinese. 

---
# An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs 

**Authors**: Linyue Ma, Yilong Xu, Xiang Long, Zhi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14660)  

**Abstract**: Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs. 

---
# Intent Clustering with Shared Pseudo-Labels 

**Authors**: I-Fan Lin, Faegheh Hasibi, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2510.14640)  

**Abstract**: In this paper, we propose an intuitive, training-free and label-free method for intent clustering that makes minimal assumptions using lightweight and open-source LLMs. Many current approaches rely on commercial LLMs, which are costly, and offer limited transparency. Additionally, their methods often explicitly depend on knowing the number of clusters in advance, which is often not the case in realistic settings. To address these challenges, instead of asking the LLM to match similar text directly, we first ask it to generate pseudo-labels for each text, and then perform multi-label classification in this pseudo-label set for each text. This approach is based on the hypothesis that texts belonging to the same cluster will share more labels, and will therefore be closer when encoded into embeddings. These pseudo-labels are more human-readable than direct similarity matches. Our evaluation on four benchmark sets shows that our approach achieves results comparable to and better than recent baselines, while remaining simple and computationally efficient. Our findings indicate that our method can be applied in low-resource scenarios and is stable across multiple models and datasets. 

---
# RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF 

**Authors**: Qing Yang, Zhenghao Liu, Junxin Wang, Yangfan Du, Pengcheng Huang, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14628)  

**Abstract**: Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation. 

---
# Code-driven Number Sequence Calculation: Enhancing the inductive Reasoning Abilities of Large Language Models 

**Authors**: Kedi Chen, Zhikai Lei, Xu Guo, Xuecheng Wu, Siyuan Zeng, Jianghao Yin, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Qipeng Guo, Kai Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14620)  

**Abstract**: Large language models (LLMs) make remarkable progress in reasoning tasks. Among different reasoning modes, inductive reasoning, due to its better alignment with human learning, attracts increasing interest. However, research on inductive reasoning faces certain challenges. First, existing inductive data mostly focuses on superficial regularities while lacking more complex internal patterns. Second, current works merely prompt LLMs or finetune on simple prompt-response pairs, but do not provide precise thinking processes nor implement difficulty control. Unlike previous work, we address these challenges by introducing \textit{CodeSeq}, a synthetic post-training dataset built from number sequences. We package number sequences into algorithmic problems to discover their general terms, defining a general term generation (GTG) task correspondingly. Our pipeline generates supervised finetuning data by reflecting on failed test cases and incorporating iterative corrections, thereby teaching LLMs to learn autonomous case generation and self-checking. Additionally, it leverages reinforcement learning with a novel Case-Synergy Solvability Scaling Reward based on both solvability, estimated from the problem pass rate, and the success rate of self-directed case generation, enabling models to learn more effectively from both successes and failures. Experimental results show that the models trained with \textit{CodeSeq} improve on various reasoning tasks and can preserve the models' OOD performance. 

---
# Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures 

**Authors**: Shuangshuang Ying, Yunwen Li, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Xeron Du, Tianyu Zheng, Yichi Zhang, Letian Ni, Yuyang Cheng, Qiguang Chen, Jingzhe Ding, Shengda Long, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Ge Zhang, Wenhao Huang, Wanxiang Che, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14616)  

**Abstract**: Current preference learning methods achieve high accuracy on standard benchmarks but exhibit significant performance degradation when objective quality signals are removed. We introduce WritingPreferenceBench, a dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where responses are matched for objective correctness, factual accuracy, and length. On this benchmark, sequence-based reward models--the standard architecture for RLHF--achieve only 52.7% mean accuracy, while zero-shot language model judges perform at 53.9%. In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%. This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification. 

---
# Assessing Socio-Cultural Alignment and Technical Safety of Sovereign LLMs 

**Authors**: Kyubyung Chae, Gihoon Kim, Gyuseong Lee, Taesup Kim, Jaejin Lee, Heejin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.14565)  

**Abstract**: Recent trends in LLMs development clearly show growing interest in the use and application of sovereign LLMs. The global debate over sovereign LLMs highlights the need for governments to develop their LLMs, tailored to their unique socio-cultural and historical contexts. However, there remains a shortage of frameworks and datasets to verify two critical questions: (1) how well these models align with users' socio-cultural backgrounds, and (2) whether they maintain safety and technical robustness without exposing users to potential harms and risks. To address this gap, we construct a new dataset and introduce an analytic framework for extracting and evaluating the socio-cultural elements of sovereign LLMs, alongside assessments of their technical robustness. Our experimental results demonstrate that while sovereign LLMs play a meaningful role in supporting low-resource languages, they do not always meet the popular claim that these models serve their target users well. We also show that pursuing this untested claim may lead to underestimating critical quality attributes such as safety. Our study suggests that advancing sovereign LLMs requires a more extensive evaluation that incorporates a broader range of well-grounded and practical criteria. 

---
# Efficient Seq2seq Coreference Resolution Using Entity Representations 

**Authors**: Matt Grenander, Shay B. Cohen, Mark Steedman  

**Link**: [PDF](https://arxiv.org/pdf/2510.14504)  

**Abstract**: Seq2seq coreference models have introduced a new paradigm for coreference resolution by learning to generate text corresponding to coreference labels, without requiring task-specific parameters. While these models achieve new state-of-the-art performance, they do so at the cost of flexibility and efficiency. In particular, they do not efficiently handle incremental settings such as dialogue, where text must processed sequentially. We propose a compressed representation in order to improve the efficiency of these methods in incremental settings. Our method works by extracting and re-organizing entity-level tokens, and discarding the majority of other input tokens. On OntoNotes, our best model achieves just 0.6 CoNLL F1 points below a full-prefix, incremental baseline while achieving a compression ratio of 1.8. On LitBank, where singleton mentions are annotated, it passes state-of-the-art performance. Our results indicate that discarding a wide portion of tokens in seq2seq resolvers is a feasible strategy for incremental coreference resolution. 

---
# LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models 

**Authors**: Haolin Li, Haipeng Zhang, Mang Li, Yaohua Wang, Lijie Wen, Yu Zhang, Biqing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14466)  

**Abstract**: As large language models (LLMs) rapidly advance, performance on high-resource languages (e.g., English, Chinese) is nearing saturation, yet remains substantially lower for low-resource languages (e.g., Urdu, Thai) due to limited training data, machine-translation noise, and unstable cross-lingual alignment. We introduce LiRA (Linguistic Robust Anchoring for Large Language Models), a training framework that robustly improves cross-lingual representations under low-resource conditions while jointly strengthening retrieval and reasoning. LiRA comprises two modules: (i) Arca (Anchored Representation Composition Architecture), which anchors low-resource languages to an English semantic space via anchor-based alignment and multi-agent collaborative encoding, preserving geometric stability in a shared embedding space; and (ii) LaSR (Language-coupled Semantic Reasoner), which adds a language-aware lightweight reasoning head with consistency regularization on top of Arca's multilingual representations, unifying the training objective to enhance cross-lingual understanding, retrieval, and reasoning robustness. We further construct and release a multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Experiments across low-resource benchmarks (cross-lingual retrieval, semantic similarity, and reasoning) show consistent gains and robustness under few-shot and noise-amplified settings; ablations validate the contribution of both Arca and LaSR. Code will be released on GitHub and the dataset on Hugging Face. 

---
# Natural Language Tools: A Natural Language Approach to Tool Calling In Large Language Agents 

**Authors**: Reid T. Johnson, Michelle D. Pain, Jordan D. West  

**Link**: [PDF](https://arxiv.org/pdf/2510.14453)  

**Abstract**: We present Natural Language Tools (NLT), a framework that replaces programmatic JSON tool calling in large language models (LLMs) with natural language outputs. By decoupling tool selection from response generation, NLT eliminates task interference and format constraints that degrade tool call performance. When evaluated across 10 models and 6,400 trials spanning customer service and mental health domains, NLT improves tool calling accuracy by 18.4 percentage points while reducing output variance by 70%. Open-weight models see the largest gains, surpassing flagship closed-weight alternatives, with implications for model training in both reinforcement learning and supervised fine-tuning stages. These improvements persist under prompt perturbations and extend tool-calling capabilities to models lacking native support. 

---
# Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents 

**Authors**: Rui Wang, Ce Zhang, Jun-Yu Ma, Jianshu Zhang, Hongru Wang, Yi Chen, Boyang Xue, Tianqing Fang, Zhisong Zhang, Hongming Zhang, Haitao Mi, Dong Yu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.14438)  

**Abstract**: Deep research web agents not only retrieve information from diverse sources such as web environments, files, and multimodal inputs, but more importantly, they need to rigorously analyze and aggregate knowledge for insightful research. However, existing open-source deep research agents predominantly focus on enhancing information-seeking capabilities of web agents to locate specific information, while overlooking the essential need for information aggregation, which would limit their ability to support in-depth research. We propose an Explore to Evolve paradigm to scalably construct verifiable training data for web agents. Begins with proactive online exploration, an agent sources grounded information by exploring the real web. Using the collected evidence, the agent then self-evolves an aggregation program by selecting, composing, and refining operations from 12 high-level logical types to synthesize a verifiable QA pair. This evolution from high-level guidance to concrete operations allowed us to scalably produce WebAggregatorQA, a dataset of 10K samples across 50K websites and 11 domains. Based on an open-source agent framework, SmolAgents, we collect supervised fine-tuning trajectories to develop a series of foundation models, WebAggregator. WebAggregator-8B matches the performance of GPT-4.1, while the 32B variant surpasses GPT-4.1 by more than 10% on GAIA-text and closely approaches Claude-3.7-sonnet. Moreover, given the limited availability of benchmarks that evaluate web agents' information aggregation abilities, we construct a human-annotated evaluation split of WebAggregatorQA as a challenging test set. On this benchmark, Claude-3.7-sonnet only achieves 28%, and GPT-4.1 scores 25.8%. Even when agents manage to retrieve all references, they still struggle on WebAggregatorQA, highlighting the need to strengthen the information aggregation capabilities of web agent foundations. 

---
# Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following 

**Authors**: Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14420)  

**Abstract**: Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL 

---
# MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering 

**Authors**: Yingpeng Ning, Yuanyuan Sun, Ling Luo, Yanhua Wang, Yuchen Pan, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14400)  

**Abstract**: Biomedical question answering (QA) requires accurate interpretation of complex medical knowledge. Large language models (LLMs) have shown promising capabilities in this domain, with retrieval-augmented generation (RAG) systems enhancing performance by incorporating external medical literature. However, RAG-based approaches in biomedical QA suffer from hallucinations due to post-retrieval noise and insufficient verification of retrieved evidence, undermining response reliability. We propose MedTrust-Guided Iterative RAG, a framework designed to enhance factual consistency and mitigate hallucinations in medical QA. Our method introduces three key innovations. First, it enforces citation-aware reasoning by requiring all generated content to be explicitly grounded in retrieved medical documents, with structured Negative Knowledge Assertions used when evidence is insufficient. Second, it employs an iterative retrieval-verification process, where a verification agent assesses evidence adequacy and refines queries through Medical Gap Analysis until reliable information is obtained. Third, it integrates the MedTrust-Align Module (MTAM) that combines verified positive examples with hallucination-aware negative samples, leveraging Direct Preference Optimization to reinforce citation-grounded reasoning while penalizing hallucination-prone response patterns. Experiments on MedMCQA, MedQA, and MMLU-Med demonstrate that our approach consistently outperforms competitive baselines across multiple model architectures, achieving the best average accuracy with gains of 2.7% for LLaMA3.1-8B-Instruct and 2.4% for Qwen3-8B. 

---
# Your Next Token Prediction: A Multilingual Benchmark for Personalized Response Generation 

**Authors**: Shiyao Ding, Takayuki Ito  

**Link**: [PDF](https://arxiv.org/pdf/2510.14398)  

**Abstract**: Large language models (LLMs) excel at general next-token prediction but still struggle to generate responses that reflect how individuals truly communicate, such as replying to emails or social messages in their own style. However, real SNS or email histories are difficult to collect due to privacy concerns. To address this, we propose the task of "Your Next Token Prediction (YNTP)", which models a user's precise word choices through controlled human-agent conversations. We build a multilingual benchmark of 100 dialogue sessions across English, Japanese, and Chinese, where users interact for five days with psychologically grounded NPCs based on MBTI dimensions. This setup captures natural, daily-life communication patterns and enables analysis of users' internal models. We evaluate prompt-based and fine-tuning-based personalization methods, establishing the first benchmark for YNTP and a foundation for user-aligned language modeling. The dataset is available at: this https URL 

---
# Suicidal Comment Tree Dataset: Enhancing Risk Assessment and Prediction Through Contextual Analysis 

**Authors**: Jun Li, Qun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14395)  

**Abstract**: Suicide remains a critical global public health issue. While previous studies have provided valuable insights into detecting suicidal expressions in individual social media posts, limited attention has been paid to the analysis of longitudinal, sequential comment trees for predicting a user's evolving suicidal risk. Users, however, often reveal their intentions through historical posts and interactive comments over time. This study addresses this gap by investigating how the information in comment trees affects both the discrimination and prediction of users' suicidal risk levels. We constructed a high-quality annotated dataset, sourced from Reddit, which incorporates users' posting history and comments, using a refined four-label annotation framework based on the Columbia Suicide Severity Rating Scale (C-SSRS). Statistical analysis of the dataset, along with experimental results from Large Language Models (LLMs) experiments, demonstrates that incorporating comment trees data significantly enhances the discrimination and prediction of user suicidal risk levels. This research offers a novel insight to enhancing the detection accuracy of at-risk individuals, thereby providing a valuable foundation for early suicide intervention strategies. 

---
# PluriHop: Exhaustive, Recall-Sensitive QA over Distractor-Rich Corpora 

**Authors**: Mykolas Sveistrys, Richard Kunert  

**Link**: [PDF](https://arxiv.org/pdf/2510.14377)  

**Abstract**: Recent advances in large language models (LLMs) and retrieval-augmented generation (RAG) have enabled progress on question answering (QA) when relevant evidence is in one (single-hop) or multiple (multi-hop) passages. Yet many realistic questions about recurring report data - medical records, compliance filings, maintenance logs - require aggregation across all documents, with no clear stopping point for retrieval and high sensitivity to even one missed passage. We term these pluri-hop questions and formalize them by three criteria: recall sensitivity, exhaustiveness, and exactness. To study this setting, we introduce PluriHopWIND, a diagnostic multilingual dataset of 48 pluri-hop questions built from 191 real-world wind industry reports in German and English. We show that PluriHopWIND is 8-40% more repetitive than other common datasets and thus has higher density of distractor documents, better reflecting practical challenges of recurring report corpora. We test a traditional RAG pipeline as well as graph-based and multimodal variants, and find that none of the tested approaches exceed 40% in statement-wise F1 score. Motivated by this, we propose PluriHopRAG, a RAG architecture that follows a "check all documents individually, filter cheaply" approach: it (i) decomposes queries into document-level subquestions and (ii) uses a cross-encoder filter to discard irrelevant documents before costly LLM reasoning. We find that PluriHopRAG achieves relative F1 score improvements of 18-52% depending on base LLM. Despite its modest size, PluriHopWIND exposes the limitations of current QA systems on repetitive, distractor-rich corpora. PluriHopRAG's performance highlights the value of exhaustive retrieval and early filtering as a powerful alternative to top-k methods. 

---
# From Binary to Bilingual: How the National Weather Service is Using Artificial Intelligence to Develop a Comprehensive Translation Program 

**Authors**: Joseph E. Trujillo-Falcon, Monica L. Bozeman, Liam E. Llewellyn, Samuel T. Halvorson, Meryl Mizell, Stuti Deshpande, Bob Manning, Todd Fagin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14369)  

**Abstract**: To advance a Weather-Ready Nation, the National Weather Service (NWS) is developing a systematic translation program to better serve the 68.8 million people in the U.S. who do not speak English at home. This article outlines the foundation of an automated translation tool for NWS products, powered by artificial intelligence. The NWS has partnered with LILT, whose patented training process enables large language models (LLMs) to adapt neural machine translation (NMT) tools for weather terminology and messaging. Designed for scalability across Weather Forecast Offices (WFOs) and National Centers, the system is currently being developed in Spanish, Simplified Chinese, Vietnamese, and other widely spoken non-English languages. Rooted in best practices for multilingual risk communication, the system provides accurate, timely, and culturally relevant translations, significantly reducing manual translation time and easing operational workloads across the NWS. To guide the distribution of these products, GIS mapping was used to identify language needs across different NWS regions, helping prioritize resources for the communities that need them most. We also integrated ethical AI practices throughout the program's design, ensuring that transparency, fairness, and human oversight guide how automated translations are created, evaluated, and shared with the public. This work has culminated into a website featuring experimental multilingual NWS products, including translated warnings, 7-day forecasts, and educational campaigns, bringing the country one step closer to a national warning system that reaches all Americans. 

---
# On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How? 

**Authors**: Anyun Zhuo, Xuefei Ning, Ningyuan Li, Yu Wang, Pinyan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14365)  

**Abstract**: This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce \nameshort{}, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and \textit{implicit} versus \textit{explicit} denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications. 

---
# CURE: Confidence-driven Unified Reasoning Ensemble Framework for Medical Question Answering 

**Authors**: Ziad Elshaer, Essam A. Rashed  

**Link**: [PDF](https://arxiv.org/pdf/2510.14353)  

**Abstract**: High-performing medical Large Language Models (LLMs) typically require extensive fine-tuning with substantial computational resources, limiting accessibility for resource-constrained healthcare institutions. This study introduces a confidence-driven multi-model framework that leverages model diversity to enhance medical question answering without fine-tuning. Our framework employs a two-stage architecture: a confidence detection module assesses the primary model's certainty, and an adaptive routing mechanism directs low-confidence queries to Helper models with complementary knowledge for collaborative reasoning. We evaluate our approach using Qwen3-30B-A3B-Instruct, Phi-4 14B, and Gemma 2 12B across three medical benchmarks; MedQA, MedMCQA, and PubMedQA. Result demonstrate that our framework achieves competitive performance, with particularly strong results in PubMedQA (95.0\%) and MedMCQA (78.0\%). Ablation studies confirm that confidence-aware routing combined with multi-model collaboration substantially outperforms single-model approaches and uniform reasoning strategies. This work establishes that strategic model collaboration offers a practical, computationally efficient pathway to improve medical AI systems, with significant implications for democratizing access to advanced medical AI in resource-limited settings. 

---
# Beyond One World: Benchmarking Super Heros in Role-Playing Across Multiversal Contexts 

**Authors**: Perapard Ngokpol, Kun Kerdthaisong, Pasin Buakhaw, Pitikorn Khlaisamniang, Supasate Vorathammathorn, Piyalitt Ittichaiwong, Nutchanon Yongsatianchot  

**Link**: [PDF](https://arxiv.org/pdf/2510.14351)  

**Abstract**: Large language models (LLMs) are increasingly used as role-playing agents, yet their capacity to faithfully and consistently portray version-specific characters -- for example, superheroes across comic and cinematic universes -- remains underexplored. Superhero canons such as Marvel and DC provide a rich testbed: decades of storytelling yield multiple incarnations of the same character with distinct histories, values, and moral codes. To study this problem, we introduce Beyond One World, a benchmark for character-grounded roleplay spanning 30 iconic heroes and 90 canon-specific versions. The benchmark comprises two tasks: (i) Canon Events, which probes factual recall of pivotal life stages, and (ii) Moral Dilemmas, which confronts models with ethically charged scenarios. We score responses for canonical accuracy and reasoning fidelity under a framework that separates internal deliberation ("thinking") from outward decisions ("acting"). We further propose Think-Act Matching, a metric that quantifies alignment between reasons and actions and serves as a proxy for model trustworthiness. Experiments across reasoning- and non-reasoning-oriented models yield three findings: (1) chain-of-thought prompting improves narrative coherence in weaker models but can reduce canonical accuracy in stronger ones; (2) cross-version generalization within a character remains a major obstacle; and (3) models often excel at either thinking or acting, but rarely both. Beyond One World exposes critical gaps in multiversal consistency and reasoning alignment, offering a challenging evaluation for role-playing LLMs. 

---
# A Robust Classification Method using Hybrid Word Embedding for Early Diagnosis of Alzheimer's Disease 

**Authors**: Yangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14332)  

**Abstract**: Early detection of Alzheimer's Disease (AD) is greatly beneficial to AD patients, leading to early treatments that lessen symptoms and alleviating financial burden of health care. As one of the leading signs of AD, language capability changes can be used for early diagnosis of AD. In this paper, I develop a robust classification method using hybrid word embedding and fine-tuned hyperparameters to achieve state-of-the-art accuracy in the early detection of AD. Specifically, we create a hybrid word embedding based on word vectors from Doc2Vec and ELMo to obtain perplexity scores of the sentences. The scores identify whether a sentence is fluent or not and capture semantic context of the sentences. I enrich the word embedding by adding linguistic features to analyze syntax and semantics. Further, we input an embedded feature vector into logistic regression and fine tune hyperparameters throughout the pipeline. By tuning hyperparameters of the machine learning pipeline (e.g., model regularization parameter, learning rate and vector size of Doc2Vec, and vector size of ELMo), I achieve 91% classification accuracy and an Area Under the Curve (AUC) of 97% in distinguishing early AD from healthy subjects. Based on my knowledge, my model with 91% accuracy and 97% AUC outperforms the best existing NLP model for AD diagnosis with an accuracy of 88% [32]. I study the model stability through repeated experiments and find that the model is stable even though the training data is split randomly (standard deviation of accuracy = 0.0403; standard deviation of AUC = 0.0174). This affirms our proposed method is accurate and stable. This model can be used as a large-scale screening method for AD, as well as a complementary examination for doctors to detect AD. 

---
# Evaluating & Reducing Deceptive Dialogue From Language Models with Multi-turn RL 

**Authors**: Marwa Abdulhai, Ryan Cheng, Aryansh Shrivastava, Natasha Jaques, Yarin Gal, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.14318)  

**Abstract**: Large Language Models (LLMs) interact with millions of people worldwide in applications such as customer support, education and healthcare. However, their ability to produce deceptive outputs, whether intentionally or inadvertently, poses significant safety concerns. The unpredictable nature of LLM behavior, combined with insufficient safeguards against hallucination, misinformation, and user manipulation, makes their misuse a serious, real-world risk. In this paper, we investigate the extent to which LLMs engage in deception within dialogue, and propose the belief misalignment metric to quantify deception. We evaluate deception across four distinct dialogue scenarios, using five established deception detection metrics and our proposed metric. Our findings reveal this novel deception measure correlates more closely with human judgments than any existing metrics we test. Additionally, our benchmarking of eight state-of-the-art models indicates that LLMs naturally exhibit deceptive behavior in approximately 26% of dialogue turns, even when prompted with seemingly benign objectives. When prompted to deceive, LLMs are capable of increasing deceptiveness by as much as 31% relative to baselines. Unexpectedly, models trained with RLHF, the predominant approach for ensuring the safety of widely-deployed LLMs, still exhibit deception at a rate of 43% on average. Given that deception in dialogue is a behavior that develops over an interaction history, its effective evaluation and mitigation necessitates moving beyond single-utterance analyses. We introduce a multi-turn reinforcement learning methodology to fine-tune LLMs to reduce deceptive behaviors, leading to a 77.6% reduction compared to other instruction-tuned models. 

---
# MERLIN: A Testbed for Multilingual Multimodal Entity Recognition and Linking 

**Authors**: Sathyanarayanan Ramamoorthy, Vishwa Shah, Simran Khanuja, Zaid Sheikh, Shan Jie, Ann Chia, Shearman Chua, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.14307)  

**Abstract**: This paper introduces MERLIN, a novel testbed system for the task of Multilingual Multimodal Entity Linking. The created dataset includes BBC news article titles, paired with corresponding images, in five languages: Hindi, Japanese, Indonesian, Vietnamese, and Tamil, featuring over 7,000 named entity mentions linked to 2,500 unique Wikidata entities. We also include several benchmarks using multilingual and multimodal entity linking methods exploring different language models like LLaMa-2 and Aya-23. Our findings indicate that incorporating visual data improves the accuracy of entity linking, especially for entities where the textual context is ambiguous or insufficient, and particularly for models that do not have strong multilingual abilities. For the work, the dataset, methods are available here at this https URL 

---
# MathMist: A Parallel Multilingual Benchmark Dataset for Mathematical Problem Solving and Reasoning 

**Authors**: Mahbub E Sobhani, Md. Faiyaz Abdullah Sayeedi, Tasnim Mohiuddin, Md Mofijul Islam, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2510.14305)  

**Abstract**: Mathematical reasoning remains one of the most challenging domains for large language models (LLMs), requiring not only linguistic understanding but also structured logical deduction and numerical precision. While recent LLMs demonstrate strong general-purpose reasoning abilities, their mathematical competence across diverse languages remains underexplored. Existing benchmarks primarily focus on English or a narrow subset of high-resource languages, leaving significant gaps in assessing multilingual and cross-lingual mathematical reasoning. To address this, we introduce MathMist, a parallel multilingual benchmark for mathematical problem solving and reasoning. MathMist encompasses over 21K aligned question-answer pairs across seven languages, representing a balanced coverage of high-, medium-, and low-resource linguistic settings. The dataset captures linguistic variety, multiple types of problem settings, and solution synthesizing capabilities. We systematically evaluate a diverse suite of models, including open-source small and medium LLMs, proprietary systems, and multilingual-reasoning-focused models, under zero-shot, chain-of-thought (CoT), and code-switched reasoning paradigms. Our results reveal persistent deficiencies in LLMs' ability to perform consistent and interpretable mathematical reasoning across languages, with pronounced degradation in low-resource settings. All the codes and data are available at GitHub: this https URL 

---
# Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers 

**Authors**: Ziye Xia, Sergei S. Ospichev  

**Link**: [PDF](https://arxiv.org/pdf/2510.14303)  

**Abstract**: In recent years, the rapid increase in academic publications across various fields has posed severe challenges for academic paper analysis: scientists struggle to timely and comprehensively track the latest research findings and methodologies. Key concept extraction has proven to be an effective analytical paradigm, and its automation has been achieved with the widespread application of language models in industrial and scientific domains. However, existing paper databases are mostly limited to similarity matching and basic classification of key concepts, failing to deeply explore the relational networks between concepts. This paper is based on the OpenAlex opensource knowledge graph. By analyzing nearly 8,000 open-source paper data from Novosibirsk State University, we discovered a strong correlation between the distribution patterns of paper key concept paths and both innovation points and rare paths. We propose a prompt engineering-based key concept path analysis method. This method leverages small language models to achieve precise key concept extraction and innovation point identification, and constructs an agent based on a knowledge graph constraint mechanism to enhance analysis accuracy. Through fine-tuning of the Qwen and DeepSeek models, we achieved significant improvements in accuracy, with the models publicly available on the Hugging Face platform. 

---
# Rethinking Schema Linking: A Context-Aware Bidirectional Retrieval Approach for Text-to-SQL 

**Authors**: Md Mahadi Hasan Nahid, Davood Rafiei, Weiwei Zhang, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14296)  

**Abstract**: Schema linking -- the process of aligning natural language questions with database schema elements -- is a critical yet underexplored component of Text-to-SQL systems. While recent methods have focused primarily on improving SQL generation, they often neglect the retrieval of relevant schema elements, which can lead to hallucinations and execution failures. In this work, we propose a context-aware bidirectional schema retrieval framework that treats schema linking as a standalone problem. Our approach combines two complementary strategies: table-first retrieval followed by column selection, and column-first retrieval followed by table selection. It is further augmented with techniques such as question decomposition, keyword extraction, and keyphrase extraction. Through comprehensive evaluations on challenging benchmarks such as BIRD and Spider, we demonstrate that our method significantly improves schema recall while reducing false positives. Moreover, SQL generation using our retrieved schema consistently outperforms full-schema baselines and closely approaches oracle performance, all without requiring query refinement. Notably, our method narrows the performance gap between full and perfect schema settings by 50\%. Our findings highlight schema linking as a powerful lever for enhancing Text-to-SQL accuracy and efficiency. 

---
# PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering 

**Authors**: Md Mahadi Hasan Nahid, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2510.14278)  

**Abstract**: Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines. 

---
# Qwen3Guard Technical Report 

**Authors**: Haiquan Zhao, Chenhan Yuan, Fei Huang, Xiaomeng Hu, Yichang Zhang, An Yang, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin, Baosong Yang, Chen Cheng, Jialong Tang, Jiandong Jiang, Jianwei Zhang, Jijie Xu, Ming Yan, Minmin Sun, Pei Zhang, Pengjun Xie, Qiaoyu Tang, Qin Zhu, Rong Zhang, Shibin Wu, Shuo Zhang, Tao He, Tianyi Tang, Tingyu Xia, Wei Liao, Weizhou Shen, Wenbiao Yin, Wenmeng Zhou, Wenyuan Yu, Xiaobin Wang, Xiaodong Deng, Xiaodong Xu, Xinyu Zhang, Yang Liu, Yeqiu Li, Yi Zhang, Yong Jiang, Yu Wan, Yuxin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14276)  

**Abstract**: As large language models (LLMs) become more capable and widely used, ensuring the safety of their outputs is increasingly critical. Existing guardrail models, though useful in static evaluation settings, face two major limitations in real-world applications: (1) they typically output only binary "safe/unsafe" labels, which can be interpreted inconsistently across diverse safety policies, rendering them incapable of accommodating varying safety tolerances across domains; and (2) they require complete model outputs before performing safety checks, making them fundamentally incompatible with streaming LLM inference, thereby preventing timely intervention during generation and increasing exposure to harmful partial outputs. To address these challenges, we present Qwen3Guard, a series of multilingual safety guardrail models with two specialized variants: Generative Qwen3Guard, which casts safety classification as an instruction-following task to enable fine-grained tri-class judgments (safe, controversial, unsafe); and Stream Qwen3Guard, which introduces a token-level classification head for real-time safety monitoring during incremental text generation. Both variants are available in three sizes (0.6B, 4B, and 8B parameters) and support up to 119 languages and dialects, providing comprehensive, scalable, and low-latency safety moderation for global LLM deployments. Evaluated across English, Chinese, and multilingual benchmarks, Qwen3Guard achieves state-of-the-art performance in both prompt and response safety classification. All models are released under the Apache 2.0 license for public use. 

---
# Retrofitting Small Multilingual Models for Retrieval: Matching 7B Performance with 300M Parameters 

**Authors**: Lifu Tu, Yingbo Zhou, Semih Yavuz  

**Link**: [PDF](https://arxiv.org/pdf/2510.14274)  

**Abstract**: Training effective multilingual embedding models presents unique challenges due to the diversity of languages and task objectives. Although small multilingual models (<1 B parameters) perform well on multilingual tasks generally, they consistently lag behind larger models (>1 B) in the most prevalent use case: retrieval. This raises a critical question: Can smaller models be retrofitted specifically for retrieval tasks to enhance their performance? In this work, we investigate key factors that influence the effectiveness of multilingual embeddings, focusing on training data scale, negative sampling strategies, and data diversity. We find that while increasing the scale of training data yields initial performance gains, these improvements quickly plateau - indicating diminishing returns. Incorporating hard negatives proves essential for consistently improving retrieval accuracy. Furthermore, our analysis reveals that task diversity in the training data contributes more significantly to performance than language diversity alone. As a result, we develop a compact (approximately 300M) multilingual model that achieves retrieval performance comparable to or even surpassing current strong 7B models. 

---
# Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation 

**Authors**: Yilun Zheng, Dan Yang, Jie Li, Lin Shang, Lihui Chen, Jiahao Xu, Sitao Luan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14271)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enable large language models (LLMs) instant access to relevant information for the generative process, demonstrating their superior performance in addressing common LLM challenges such as hallucination, factual inaccuracy, and the knowledge cutoff. Graph-based RAG further extends this paradigm by incorporating knowledge graphs (KGs) to leverage rich, structured connections for more precise and inferential responses. A critical challenge, however, is that most Graph-based RAG systems rely on LLMs for automated KG construction, often yielding noisy KGs with redundant entities and unreliable relationships. This noise degrades retrieval and generation performance while also increasing computational cost. Crucially, current research does not comprehensively address the denoising problem for LLM-generated KGs. In this paper, we introduce DEnoised knowledge Graphs for Retrieval Augmented Generation (DEG-RAG), a framework that addresses these challenges through: (1) entity resolution, which eliminates redundant entities, and (2) triple reflection, which removes erroneous relations. Together, these techniques yield more compact, higher-quality KGs that significantly outperform their unprocessed counterparts. Beyond the methods, we conduct a systematic evaluation of entity resolution for LLM-generated KGs, examining different blocking strategies, embedding choices, similarity metrics, and entity merging techniques. To the best of our knowledge, this is the first comprehensive exploration of entity resolution in LLM-generated KGs. Our experiments demonstrate that this straightforward approach not only drastically reduces graph size but also consistently improves question answering performance across diverse popular Graph-based RAG variants. 

---
# Rewriting History: A Recipe for Interventional Analyses to Study Data Effects on Model Behavior 

**Authors**: Rahul Nadkarni, Yanai Elazar, Hila Gonen, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2510.14261)  

**Abstract**: We present an experimental recipe for studying the relationship between training data and language model (LM) behavior. We outline steps for intervening on data batches -- i.e., ``rewriting history'' -- and then retraining model checkpoints over that data to test hypotheses relating data to behavior. Our recipe breaks down such an intervention into stages that include selecting evaluation items from a benchmark that measures model behavior, matching relevant documents to those items, and modifying those documents before retraining and measuring the effects. We demonstrate the utility of our recipe through case studies on factual knowledge acquisition in LMs, using both cooccurrence statistics and information retrieval methods to identify documents that might contribute to knowledge learning. Our results supplement past observational analyses that link cooccurrence to model behavior, while demonstrating that extant methods for identifying relevant training documents do not fully explain an LM's ability to correctly answer knowledge questions. Overall, we outline a recipe that researchers can follow to test further hypotheses about how training data affects model behavior. Our code is made publicly available to promote future work. 

---
# MoM: Mixtures of Scenario-Aware Document Memories for Retrieval-Augmented Generation Systems 

**Authors**: Jihao Zhao, Zhiyuan Ji, Simin Niu, Hanyu Wang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14252)  

**Abstract**: The traditional RAG paradigm, which typically engages in the comprehension of relevant text chunks in response to received queries, inherently restricts both the depth of knowledge internalization and reasoning capabilities. To address this limitation, our research transforms the text processing in RAG from passive chunking to proactive understanding, defining this process as document memory extraction with the objective of simulating human cognitive processes during reading. Building upon this, we propose the Mixtures of scenario-aware document Memories (MoM) framework, engineered to efficiently handle documents from multiple domains and train small language models (SLMs) to acquire the ability to proactively explore and construct document memories. The MoM initially instructs large language models (LLMs) to simulate domain experts in generating document logical outlines, thereby directing structured chunking and core content extraction. It employs a multi-path sampling and multi-perspective evaluation mechanism, specifically designing comprehensive metrics that represent chunk clarity and extraction completeness to select the optimal document memories. Additionally, to infuse deeper human-like reading abilities during the training of SLMs, we incorporate a reverse reasoning strategy, which deduces refined expert thinking paths from high-quality outcomes. Finally, leveraging diverse forms of content generated by MoM, we develop a three-layer document memory retrieval mechanism, which is grounded in our theoretical proof from the perspective of probabilistic modeling. Extensive experimental results across three distinct domains demonstrate that the MoM framework not only resolves text chunking challenges in existing RAG systems, providing LLMs with semantically complete document memories, but also paves the way for SLMs to achieve human-centric intelligent text processing. 

---
# Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs 

**Authors**: Parsa Hejabi, Elnaz Rahmati, Alireza S. Ziabari, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2510.14242)  

**Abstract**: Large Language Models (LLMs) often produce inconsistent answers when faced with different phrasings of the same prompt. In this paper, we propose Flip-Flop Consistency ($F^2C$), an unsupervised training method that improves robustness to such perturbations. $F^2C$ is composed of two key components. The first, Consensus Cross-Entropy (CCE), uses a majority vote across prompt variations to create a hard pseudo-label. The second is a representation alignment loss that pulls lower-confidence and non-majority predictors toward the consensus established by high-confidence, majority-voting variations. We evaluate our method on 11 datasets spanning four NLP tasks, with 4-15 prompt variations per dataset. On average, $F^2C$ raises observed agreement by 11.62%, improves mean $F_1$ by 8.94%, and reduces performance variance across formats by 3.29%. In out-of-domain evaluations, $F^2C$ generalizes effectively, increasing $\overline{F_1}$ and agreement while decreasing variance across most source-target pairs. Finally, when trained on only a subset of prompt perturbations and evaluated on held-out formats, $F^2C$ consistently improves both performance and agreement while reducing variance. These findings highlight $F^2C$ as an effective unsupervised method for enhancing LLM consistency, performance, and generalization under prompt perturbations. Code is available at this https URL. 

---
# LiteStage: Latency-aware Layer Skipping for Multi-stage Reasoning 

**Authors**: Beomseok Kang, Jiwon Song, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.14211)  

**Abstract**: Multi-stage reasoning has emerged as an effective strategy for enhancing the reasoning capability of small language models by decomposing complex problems into sequential sub-stages. However, this comes at the cost of increased latency. We observe that existing adaptive acceleration techniques, such as layer skipping, struggle to balance efficiency and accuracy in this setting due to two key challenges: (1) stage-wise variation in skip sensitivity, and (2) the generation of redundant output tokens. To address these, we propose LiteStage, a latency-aware layer skipping framework for multi-stage reasoning. LiteStage combines a stage-wise offline search that allocates optimal layer budgets with an online confidence-based generation early exit to suppress unnecessary decoding. Experiments on three benchmarks, e.g., OBQA, CSQA, and StrategyQA, show that LiteStage achieves up to 1.70x speedup with less than 4.0% accuracy loss, outperforming prior training-free layer skipping methods. 

---
# DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans 

**Authors**: Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14205)  

**Abstract**: The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these this http URL evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie this http URL can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and this http URL work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI. 

---
# RLSR: Reinforcement Learning with Supervised Reward Outperforms SFT in Instruction Following 

**Authors**: Zhichao Wang, Andy Wong, Ruslan Belkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14200)  

**Abstract**: After the pretraining stage of LLMs, techniques such as SFT, RLHF, RLVR, and RFT are applied to enhance instruction-following ability, mitigate undesired responses, improve reasoning capability and enable efficient domain adaptation with minimal data. SFT relies on the next-token prediction objective to strengthen instruction following in a base model using a large corpus of human-labeled responses. In contrast, RFT employs a RL-based approach to adapt fine-tuned reasoning models to specific domains with limited supervision. Inspired by RFT, we propose replacing SFT with RLSR to leverage the extensive SFT dataset in an RL framework, thereby improving the base model's instruction-following ability. In RLSR, the base model generates multiple responses for each prompt, and reward scores are computed as the cosine similarity in the semantic embedding space between the generated and human-labeled responses. RLSR can be utilized in multiple ways. It can directly replace SFT, achieving superior performance on instruction-following benchmarks-for example, RLSR (SB) on Qwen-7B (INFINITY) achieved an AlpacaEval win rate of 26.34%, surpassing SFT's 21.01%. Furthermore, combining SFT and RLSR further enhances downstream task performance; Qwen-7B (INFINITY) achieved a win rate of 30.73% when trained with SFT + RLSR. 

---
# Building a Macedonian Recipe Dataset: Collection, Parsing, and Comparative Analysis 

**Authors**: Darko Sasanski, Dimitar Peshevski, Riste Stojanov, Dimitar Trajanov  

**Link**: [PDF](https://arxiv.org/pdf/2510.14128)  

**Abstract**: Computational gastronomy increasingly relies on diverse, high-quality recipe datasets to capture regional culinary traditions. Although there are large-scale collections for major languages, Macedonian recipes remain under-represented in digital research. In this work, we present the first systematic effort to construct a Macedonian recipe dataset through web scraping and structured parsing. We address challenges in processing heterogeneous ingredient descriptions, including unit, quantity, and descriptor normalization. An exploratory analysis of ingredient frequency and co-occurrence patterns, using measures such as Pointwise Mutual Information and Lift score, highlights distinctive ingredient combinations that characterize Macedonian cuisine. The resulting dataset contributes a new resource for studying food culture in underrepresented languages and offers insights into the unique patterns of Macedonian culinary tradition. 

---
# Toward Cybersecurity-Expert Small Language Models 

**Authors**: Matan Levi, Daniel Ohayon, Ariel Blobstein, Ravid Sagi, Ian Molloy, Yair Allouche  

**Link**: [PDF](https://arxiv.org/pdf/2510.14113)  

**Abstract**: Large language models (LLMs) are transforming everyday applications, yet deployment in cybersecurity lags due to a lack of high-quality, domain-specific models and training datasets. To address this gap, we present CyberPal 2.0, a family of cybersecurity-expert small language models (SLMs) ranging from 4B-20B parameters. To train CyberPal 2.0, we generate an enriched chain-of-thought cybersecurity instruction dataset built with our data enrichment and formatting pipeline, SecKnowledge 2.0, which integrates expert-in-the-loop steering of reasoning formats alongside LLM-driven multi-step grounding, yielding higher-fidelity, task-grounded reasoning traces for security tasks. Across diverse cybersecurity benchmarks, CyberPal 2.0 consistently outperforms its baselines and matches or surpasses various open and closed-source frontier models, while remaining a fraction of their size. On core cyber threat intelligence knowledge tasks, our models outperform almost all tested frontier models, ranking second only to Sec-Gemini v1. On core threat-investigation tasks, such as correlating vulnerabilities and bug tickets with weaknesses, our best 20B-parameter model outperforms GPT-4o, o1, o3-mini, and Sec-Gemini v1, ranking first, while our smallest 4B-parameter model ranks second. 

---
# DROID: Dual Representation for Out-of-Scope Intent Detection 

**Authors**: Wael Rashwan, Hossam M. Zawbaa, Sourav Dutta, Haytham Assem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14110)  

**Abstract**: Detecting out-of-scope (OOS) user utterances remains a key challenge in task-oriented dialogue systems and, more broadly, in open-set intent recognition. Existing approaches often depend on strong distributional assumptions or auxiliary calibration modules. We present DROID (Dual Representation for Out-of-Scope Intent Detection), a compact end-to-end framework that combines two complementary encoders -- the Universal Sentence Encoder (USE) for broad semantic generalization and a domain-adapted Transformer-based Denoising Autoencoder (TSDAE) for domain-specific contextual distinctions. Their fused representations are processed by a lightweight branched classifier with a single calibrated threshold that separates in-domain and OOS intents without post-hoc scoring. To enhance boundary learning under limited supervision, DROID incorporates both synthetic and open-domain outlier augmentation. Despite using only 1.5M trainable parameters, DROID consistently outperforms recent state-of-the-art baselines across multiple intent benchmarks, achieving macro-F1 improvements of 6--15% for known and 8--20% for OOS intents, with the most significant gains in low-resource settings. These results demonstrate that dual-encoder representations with simple calibration can yield robust, scalable, and reliable OOS detection for neural dialogue systems. 

---
# ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models 

**Authors**: Haziq Mohammad Khalid, Athikash Jeyaganthan, Timothy Do, Yicheng Fu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14077)  

**Abstract**: Large Language Models (LLMs) suffer significant performance degradation in multi-turn conversations when information is presented incrementally. Given that multi-turn conversations characterize everyday interactions with LLMs, this degradation poses a severe challenge to real world usability. We hypothesize that abrupt increases in model uncertainty signal misalignment in multi-turn LLM interactions, and we exploit this insight to dynamically realign conversational context. We introduce ERGO (Entropy-guided Resetting for Generation Optimization), which continuously quantifies internal uncertainty via Shannon entropy over next token distributions and triggers adaptive prompt consolidation when a sharp spike in entropy is detected. By treating uncertainty as a first class signal rather than a nuisance to eliminate, ERGO embraces variability in language and modeling, representing and responding to uncertainty. In multi-turn tasks with incrementally revealed instructions, ERGO yields a 56.6% average performance gain over standard baselines, increases aptitude (peak performance capability) by 24.7%, and decreases unreliability (variability in performance) by 35.3%, demonstrating that uncertainty aware interventions can improve both accuracy and reliability in conversational AI. 

---
# Quantifying Phonosemantic Iconicity Distributionally in 6 Languages 

**Authors**: George Flint, Kaustubh Kislay  

**Link**: [PDF](https://arxiv.org/pdf/2510.14040)  

**Abstract**: Language is, as commonly theorized, largely arbitrary. Yet, systematic relationships between phonetics and semantics have been observed in many specific cases. To what degree could those systematic relationships manifest themselves in large scale, quantitative investigations--both in previously identified and unidentified phenomena? This work undertakes a distributional approach to quantifying phonosemantic iconicity at scale across 6 diverse languages (English, Spanish, Hindi, Finnish, Turkish, and Tamil). In each language, we analyze the alignment of morphemes' phonetic and semantic similarity spaces with a suite of statistical measures, and discover an array of interpretable phonosemantic alignments not previously identified in the literature, along with crosslinguistic patterns. We also analyze 5 previously hypothesized phonosemantic alignments, finding support for some such alignments and mixed results for others. 

---
# Think Globally, Group Locally: Evaluating LLMs Using Multi-Lingual Word Grouping Games 

**Authors**: César Guerra-Solano, Zhuochun Li, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14030)  

**Abstract**: Large language models (LLMs) can exhibit biases in reasoning capabilities due to linguistic modality, performing better on tasks in one language versus another, even with similar content. Most previous works evaluate this through reasoning tasks where reliance on strategies or knowledge can ensure success, such as in commonsense or math tasks. However, abstract reasoning is vital to reasoning for everyday life, where people apply "out-of-the-box thinking" to identify and use patterns for solutions, without a reliance on formulaic approaches. Comparatively, little work has evaluated linguistic biases in this task type. In this paper, we propose a task inspired by the New York Times Connections: GlobalGroup, that evaluates models in an abstract reasoning task across several languages. We constructed a game benchmark with five linguistic backgrounds -- English, Spanish, Chinese, Hindi, and Arabic -- in both the native language and an English translation for comparison. We also proposed game difficulty measurements to evaluate models on games with similar difficulty, enabling a more controlled comparison, which is particularly important in reasoning evaluations. Through experimentation, we find English modalities largely lead to better performance in this abstract reasoning task, and performance disparities between open- and closed-source models. 

---
# CRaFT: An Explanation-Based Framework for Evaluating Cultural Reasoning in Multilingual Language Models 

**Authors**: Shehenaz Hossain, Haithem Afli  

**Link**: [PDF](https://arxiv.org/pdf/2510.14014)  

**Abstract**: Correct answers do not necessarily reflect cultural understanding. We introduce CRaFT, an explanation-based multilingual evaluation framework designed to assess how large language models (LLMs) reason across cultural contexts. Rather than scoring outputs solely based on accuracy, CRaFT evaluates model explanations using four interpretable metrics: Cultural Fluency, Deviation, Consistency, and Linguistic Adaptation. We apply the framework to 50 culturally grounded questions from the World Values Survey, translated into Arabic, Bengali, and Spanish, and evaluate three models (GPT, DeepSeek, and FANAR) across over 2,100 answer-explanation pairs. Results reveal significant cross-lingual variation in reasoning: Arabic reduces fluency, Bengali enhances it, and Spanish remains largely stable. While GPT adapts more effectively across languages, it exhibits lower consistency; FANAR shows stable but rigid reasoning. These findings suggest that cultural awareness in LLMs is not intrinsic but emerges through linguistic framing. CRaFT offers a new lens for evaluating cross-cultural reasoning in multilingual settings, providing actionable insights for building culturally adaptive language models. 

---
# The German Commons - 154 Billion Tokens of Openly Licensed Text for German Language Models 

**Authors**: Lukas Gienapp, Christopher Schröder, Stefan Schweter, Christopher Akiki, Ferdinand Schlatt, Arden Zimmermann, Phillipe Genêt, Martin Potthast  

**Link**: [PDF](https://arxiv.org/pdf/2510.13996)  

**Abstract**: Large language model development relies on large-scale training corpora, yet most contain data of unclear licensing status, limiting the development of truly open models. This problem is exacerbated for non-English languages, where openly licensed text remains critically scarce. We introduce the German Commons, the largest collection of openly licensed German text to date. It compiles data from 41 sources across seven domains, encompassing legal, scientific, cultural, political, news, economic, and web text. Through systematic sourcing from established data providers with verifiable licensing, it yields 154.56 billion tokens of high-quality text for language model training. Our processing pipeline implements comprehensive quality filtering, deduplication, and text formatting fixes, ensuring consistent quality across heterogeneous text sources. All domain subsets feature licenses of at least CC-BY-SA 4.0 or equivalent, ensuring legal compliance for model training and redistribution. The German Commons therefore addresses the critical gap in openly licensed German pretraining data, and enables the development of truly open German language models. We also release code for corpus construction and data filtering tailored to German language text, rendering the German Commons fully reproducible and extensible. 

---
# Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems 

**Authors**: Kin Kwan Leung, Mouloud Belbahri, Yi Sui, Alex Labach, Xueying Zhang, Stephen Rose, Jesse C. Cresswell  

**Link**: [PDF](https://arxiv.org/pdf/2510.13975)  

**Abstract**: Retrieval-augmented generation (RAG) is a prevalent approach for building LLM-based question-answering systems that can take advantage of external knowledge databases. Due to the complexity of real-world RAG systems, there are many potential causes for erroneous outputs. Understanding the range of errors that can occur in practice is crucial for robust deployment. We present a new taxonomy of the error types that can occur in realistic RAG systems, examples of each, and practical advice for addressing them. Additionally, we curate a dataset of erroneous RAG responses annotated by error types. We then propose an auto-evaluation method aligned with our taxonomy that can be used in practice to track and address errors during development. Code and data are available at this https URL. 

---
# Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention 

**Authors**: Zhen Yang, Mingyang Zhang, Feng Chen, Ganggui Ding, Liang Hou, Xin Tao, Pengfei Wan, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13940)  

**Abstract**: Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +1.35% average improvement on eight benchmarks for Qwen3-8B-Base and +5% on AIME2024 using Qwen3-32B-Reasoning-while remaining highly efficient. 

---
# Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers 

**Authors**: Tuhin Chakrabarty, Jane C. Ginsburg, Paramveer Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2510.13939)  

**Abstract**: The use of copyrighted books for training AI models has led to numerous lawsuits from authors concerned about AI's ability to generate derivative this http URL it's unclear whether these models can generate high quality literary text while emulating authors' styles. To answer this we conducted a preregistered study comparing MFA-trained expert writers with three frontier AI models: ChatGPT, Claude & Gemini in writing up to 450 word excerpts emulating 50 award-winning authors' diverse styles. In blind pairwise evaluations by 159 representative expert & lay readers, AI-generated text from in-context prompting was strongly disfavored by experts for both stylistic fidelity (OR=0.16, p<10^8) & writing quality (OR=0.13, p<10^7) but showed mixed results with lay readers. However, fine-tuning ChatGPT on individual authors' complete works completely reversed these findings: experts now favored AI-generated text for stylistic fidelity (OR=8.16, p<10^13) & writing quality (OR=1.87, p=0.010), with lay readers showing similar shifts. These effects generalize across authors & styles. The fine-tuned outputs were rarely flagged as AI-generated (3% rate v. 97% for in-context prompting) by best AI detectors. Mediation analysis shows this reversal occurs because fine-tuning eliminates detectable AI stylistic quirks (e.g., cliche density) that penalize in-context outputs. While we do not account for additional costs of human effort required to transform raw AI output into cohesive, publishable prose, the median fine-tuning & inference cost of $81 per author represents a dramatic 99.7% reduction compared to typical professional writer compensation. Author-specific fine-tuning thus enables non-verbatim AI writing that readers prefer to expert human writing, providing empirical evidence directly relevant to copyright's fourth fair-use factor, the "effect upon the potential market or value" of the source works. 

---
# FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis 

**Authors**: Fengbin Zhu, Xiang Yao Ng, Ziyang Liu, Chang Liu, Xianwei Zeng, Chao Wang, Tianhui Tan, Xuan Yao, Pengyang Shao, Min Xu, Zixuan Wang, Jing Wang, Xin Lin, Junfeng Li, Jingxian Zhu, Yang Zhang, Wenjie Wang, Fuli Feng, Richang Hong, Huanbo Luan, Ke-Wei Huang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2510.13936)  

**Abstract**: Deep Research (DR) agents, powered by advanced Large Language Models (LLMs), have recently garnered increasing attention for their capability in conducting complex research tasks. However, existing literature lacks a rigorous and systematic evaluation of DR Agent's capabilities in critical research analysis. To address this gap, we first propose HisRubric, a novel evaluation framework with a hierarchical analytical structure and a fine-grained grading rubric for rigorously assessing DR agents' capabilities in corporate financial analysis. This framework mirrors the professional analyst's workflow, progressing from data recognition to metric calculation, and finally to strategic summarization and interpretation. Built on this framework, we construct a FinDeepResearch benchmark that comprises 64 listed companies from 8 financial markets across 4 languages, encompassing a total of 15,808 grading items. We further conduct extensive experiments on the FinDeepResearch using 16 representative methods, including 6 DR agents, 5 LLMs equipped with both deep reasoning and search capabilities, and 5 LLMs with deep reasoning capabilities only. The results reveal the strengths and limitations of these approaches across diverse capabilities, financial markets, and languages, offering valuable insights for future research and development. The benchmark and evaluation code will be made publicly available. 

---
# Big Reasoning with Small Models: Instruction Retrieval at Inference Time 

**Authors**: Kenan Alkiek, David Jurgens, Vinod Vydiswaran  

**Link**: [PDF](https://arxiv.org/pdf/2510.13935)  

**Abstract**: Can we bring large-scale reasoning to local-scale compute? Small language models (SLMs) are increasingly attractive because they run efficiently on local hardware, offering strong privacy, low cost, and reduced environmental impact. Yet they often struggle with tasks that require multi-step reasoning or domain-specific knowledge. We address this limitation through instruction intervention at inference time, where an SLM retrieves structured reasoning procedures rather than generating them from scratch. Our method builds an Instruction Corpus by grouping similar training questions and creating instructions via GPT-5. During inference, the SLM retrieves the most relevant instructions and follows their steps. Unlike retrieval-augmented generation, which retrieves text passages, instruction retrieval gives the model structured guidance for reasoning. We evaluate this framework on MedQA (medical board exams), MMLU Professional Law, and MathQA using models from 3B to 14B parameters without any additional fine-tuning. Instruction retrieval yields consistent gains: 9.4% on MedQA, 7.9% on MMLU Law, and 5.1% on MathQA. Concise instructions outperform longer ones, and the magnitude of improvement depends strongly on model family and intrinsic reasoning ability. 

---
# Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions 

**Authors**: Siying Liu, Shisheng Zhang, Indu Bala  

**Link**: [PDF](https://arxiv.org/pdf/2510.13931)  

**Abstract**: Large language models (LLMs) are increasingly applied in biomedical domains, yet their reliability in drug-safety prediction remains underexplored. In this work, we investigate whether LLMs incorporate socio-demographic information into adverse event (AE) predictions, despite such attributes being clinically irrelevant. Using structured data from the United States Food and Drug Administration Adverse Event Reporting System (FAERS) and a persona-based evaluation framework, we assess two state-of-the-art models, ChatGPT-4o and Bio-Medical-Llama-3.8B, across diverse personas defined by education, marital status, employment, insurance, language, housing stability, and religion. We further evaluate performance across three user roles (general practitioner, specialist, patient) to reflect real-world deployment scenarios where commercial systems often differentiate access by user type. Our results reveal systematic disparities in AE prediction accuracy. Disadvantaged groups (e.g., low education, unstable housing) were frequently assigned higher predicted AE likelihoods than more privileged groups (e.g., postgraduate-educated, privately insured). Beyond outcome disparities, we identify two distinct modes of bias: explicit bias, where incorrect predictions directly reference persona attributes in reasoning traces, and implicit bias, where predictions are inconsistent, yet personas are not explicitly mentioned. These findings expose critical risks in applying LLMs to pharmacovigilance and highlight the urgent need for fairness-aware evaluation protocols and mitigation strategies before clinical deployment. 

---
# LLMs Can Get "Brain Rot"! 

**Authors**: Shuo Xing, Junyuan Hong, Yifan Wang, Runjin Chen, Zhenyu Zhang, Ananth Grama, Zhengzhong Tu, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13928)  

**Abstract**: We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To causally isolate data quality, we run controlled experiments on real Twitter/X corpora, constructing junk and reversely controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Contrary to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' $g>0.3$) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain Of Thoughts drops $74.9 \rightarrow 57.2$ and RULER-CWE $84.4 \rightarrow 52.3$ as junk ratio rises from $0\%$ to $100\%$.
Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion: models increasingly truncate or skip reasoning chains, explaining most of the error growth. Second, partial but incomplete healing is observed: scaling instruction tuning and clean data pre-training improve the declined cognition yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that data quality is a causal driver of LLM capability decay, reframing curation for continual pretraining as a \textit{training-time safety} problem and motivating routine "cognitive health checks" for deployed LLMs. 

---
# BioMedSearch: A Multi-Source Biomedical Retrieval Framework Based on LLMs 

**Authors**: Congying Liu, Xingyuan Wei, Peipei Liu, Yiqing Shen, Yanxu Mao, Tiehan Cui  

**Link**: [PDF](https://arxiv.org/pdf/2510.13926)  

**Abstract**: Biomedical queries often rely on a deep understanding of specialized knowledge such as gene regulatory mechanisms and pathological processes of diseases. They require detailed analysis of complex physiological processes and effective integration of information from multiple data sources to support accurate retrieval and reasoning. Although large language models (LLMs) perform well in general reasoning tasks, their generated biomedical content often lacks scientific rigor due to the inability to access authoritative biomedical databases and frequently fabricates protein functions, interactions, and structural details that deviate from authentic information. Therefore, we present BioMedSearch, a multi-source biomedical information retrieval framework based on LLMs. The method integrates literature retrieval, protein database and web search access to support accurate and efficient handling of complex biomedical queries. Through sub-queries decomposition, keywords extraction, task graph construction, and multi-source information filtering, BioMedSearch generates high-quality question-answering results. To evaluate the accuracy of question answering, we constructed a multi-level dataset, BioMedMCQs, consisting of 3,000 questions. The dataset covers three levels of reasoning: mechanistic identification, non-adjacent semantic integration, and temporal causal reasoning, and is used to assess the performance of BioMedSearch and other methods on complex QA tasks. Experimental results demonstrate that BioMedSearch consistently improves accuracy over all baseline models across all levels. Specifically, at Level 1, the average accuracy increases from 59.1% to 91.9%; at Level 2, it rises from 47.0% to 81.0%; and at the most challenging Level 3, the average accuracy improves from 36.3% to 73.4%. The code and BioMedMCQs are available at: this https URL 

---
# An LLM-Powered AI Agent Framework for Holistic IoT Traffic Interpretation 

**Authors**: Daniel Adu Worae, Spyridon Mastorakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.13925)  

**Abstract**: Internet of Things (IoT) networks generate diverse and high-volume traffic that reflects both normal activity and potential threats. Deriving meaningful insight from such telemetry requires cross-layer interpretation of behaviors, protocols, and context rather than isolated detection. This work presents an LLM-powered AI agent framework that converts raw packet captures into structured and semantically enriched representations for interactive analysis. The framework integrates feature extraction, transformer-based anomaly detection, packet and flow summarization, threat intelligence enrichment, and retrieval-augmented question answering. An AI agent guided by a large language model performs reasoning over the indexed traffic artifacts, assembling evidence to produce accurate and human-readable interpretations. Experimental evaluation on multiple IoT captures and six open models shows that hybrid retrieval, which combines lexical and semantic search with reranking, substantially improves BLEU, ROUGE, METEOR, and BERTScore results compared with dense-only retrieval. System profiling further indicates low CPU, GPU, and memory overhead, demonstrating that the framework achieves holistic and efficient interpretation of IoT network traffic. 

---
# FACTS: Table Summarization via Offline Template Generation with Agentic Workflows 

**Authors**: Ye Yuan, Mohammad Amin Shabani, Siqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13920)  

**Abstract**: Query-focused table summarization requires generating natural language summaries of tabular data conditioned on a user query, enabling users to access insights beyond fact retrieval. Existing approaches face key limitations: table-to-text models require costly fine-tuning and struggle with complex reasoning, prompt-based LLM methods suffer from token-limit and efficiency issues while exposing sensitive data, and prior agentic pipelines often rely on decomposition, planning, or manual templates that lack robustness and scalability. To mitigate these issues, we introduce an agentic workflow, FACTS, a Fast, Accurate, and Privacy-Compliant Table Summarization approach via Offline Template Generation. FACTS produces offline templates, consisting of SQL queries and Jinja2 templates, which can be rendered into natural language summaries and are reusable across multiple tables sharing the same schema. It enables fast summarization through reusable offline templates, accurate outputs with executable SQL queries, and privacy compliance by sending only table schemas to LLMs. Evaluations on widely-used benchmarks show that FACTS consistently outperforms baseline methods, establishing it as a practical solution for real-world query-focused table summarization. 

---
# Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time Scaling 

**Authors**: Peng Kuang, Yanli Wang, Xiaoyu Han, Yaowenqi Liu, Kaidi Xu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13918)  

**Abstract**: Process reward models (PRMs) are a cornerstone of test-time scaling (TTS), designed to verify and select the best responses from large language models (LLMs). However, this promise is challenged by recent benchmarks where simple majority voting, which ignores PRM signals, occasionally outperforms standard PRM-based selection. This raises a critical question: How can we effectively utilize verification signals from PRMs for TTS? To address this, we start by developing a theoretical framework for optimally combining signals from both the LLM and the PRM. Our framework reveals that the optimal strategy is a weighted aggregation of responses, a strategy whose effectiveness hinges on estimating weights that capture the complex interplay between the models. Based on our theoretical results, we empirically show that these optimal weighting functions differ significantly across LLM-PRM pairs and, notably, often assign substantial negative weights. Motivated by these insights, we propose efficient pre-computation methods to calibrate these weighting functions. Extensive experiments across 5 LLMs and 7 PRMs demonstrate that our calibration method significantly boosts the TTS efficiency, surpassing the performance of vanilla weighted majority voting while using only $21.3\%$ of the computation. Ultimately, our work demonstrates that investing in a more intelligent aggregation strategy can be a more convincing path to performance gains than simply scaling test-time computation. 

---
# Element2Vec: Build Chemical Element Representation from Text for Property Prediction 

**Authors**: Yuanhao Li, Keyuan Lai, Tianqi Wang, Qihao Liu, Jiawei Ma, Yuan-Chao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13916)  

**Abstract**: Accurate property data for chemical elements is crucial for materials design and manufacturing, but many of them are difficult to measure directly due to equipment constraints. While traditional methods use the properties of other elements or related properties for prediction via numerical analyses, they often fail to model complex relationships. After all, not all characteristics can be represented as scalars. Recent efforts have been made to explore advanced AI tools such as language models for property estimation, but they still suffer from hallucinations and a lack of interpretability. In this paper, we investigate Element2Vecto effectively represent chemical elements from natural languages to support research in the natural sciences. Given the text parsed from Wikipedia pages, we use language models to generate both a single general-purpose embedding (Global) and a set of attribute-highlighted vectors (Local). Despite the complicated relationship across elements, the computational challenges also exist because of 1) the discrepancy in text distribution between common descriptions and specialized scientific texts, and 2) the extremely limited data, i.e., with only 118 known elements, data for specific properties is often highly sparse and incomplete. Thus, we also design a test-time training method based on self-attention to mitigate the prediction error caused by Vanilla regression clearly. We hope this work could pave the way for advancing AI-driven discovery in materials science. 

---
# Readability $\ne$ Learnability: Rethinking the Role of Simplicity in Training Small Language Models 

**Authors**: Ivan Lee, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2510.13915)  

**Abstract**: Recent studies suggest that very small language models (SLMs) can generate surprisingly coherent text when trained on simplified, child-directed corpora such as TinyStories. These findings have been interpreted as evidence that readability -- characterized by accessible vocabulary, familiar narrative structure, and simple syntax -- plays a key role in enabling such capabilities to emerge. In this paper, we challenge that interpretation. We construct synthetic datasets with matched structure but varied readability, and find that readability alone does not predict coherence or learning efficiency in SLMs. Models trained on complex, adult-level text perform comparably to those trained on simplified language, and even exhibit faster development of coherence during training. Instead, we show that statistical simplicity, as measured by n-gram diversity, is a stronger predictor of learnability. Our findings caution against the growing trend of anthropomorphizing language model training -- drawing parallels to human cognitive development without empirical basis -- and argue for more precise reasoning about what properties actually support capability emergence in small models. 

---
# Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms 

**Authors**: Shrey Pandit, Xuan-Phi Nguyen, Yifei Ming, Austin Xu, Jiayu Wang, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.13913)  

**Abstract**: Web-based 'deep research' agents aim to solve complex question - answering tasks through long-horizon interactions with online tools. These tasks remain challenging, as the underlying language models are often not optimized for long-horizon reasoning and exploration. Prior work has proposed workflows for constructing instruction-tuning datasets, often leveraging knowledge graphs. However, such methods typically lack fine-grained control over difficulty and quality, yielding synthetic data that falls short of capturing the complexity required for long-horizon reasoning. Furthermore, many studies conflate data and training effects by comparing models trained under different optimization recipes, making it difficult to isolate and evaluate the effectiveness of the data itself. We introduce a two-pronged data synthesis pipeline that generates question - answer pairs by progressively increasing task complexity until a frontier baseline web agent fails. The baseline agent plays multiple roles in this process: attempting the questions, validating factuality, checking for alternative answers, and enforcing filtering. To evaluate the effectiveness of our synthesis methods, we adopt a controlled training setup based on distillation from strong web agents. Experiments across multiple web-based benchmarks show that our dataset - despite being smaller - enables the training of more effective web agents than existing datasets. In particular, our data exhibits twice the diversity in tool-use actions, allowing models trained on it to achieve stronger performance while avoiding repetitive tool-calling behaviors. 

---
# AI Debaters are More Persuasive when Arguing in Alignment with Their Own Beliefs 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Facundo Nieto, Oscar Agustín Stanchi, Guido Ernesto Bergman, Mario Alejandro Leiva, Eitan Sprejer, Luca Nicolás Forziati Gangi, Francisca Gauna Selasco, Juan Gustavo Corvalán, Gerardo I. Simari, María Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13912)  

**Abstract**: The core premise of AI debate as a scalable oversight technique is that it is harder to lie convincingly than to refute a lie, enabling the judge to identify the correct position. Yet, existing debate experiments have relied on datasets with ground truth, where lying is reduced to defending an incorrect proposition. This overlooks a subjective dimension: lying also requires the belief that the claim defended is false. In this work, we apply debate to subjective questions and explicitly measure large language models' prior beliefs before experiments. Debaters were asked to select their preferred position, then presented with a judge persona deliberately designed to conflict with their identified priors. This setup tested whether models would adopt sycophantic strategies, aligning with the judge's presumed perspective to maximize persuasiveness, or remain faithful to their prior beliefs. We implemented and compared two debate protocols, sequential and simultaneous, to evaluate potential systematic biases. Finally, we assessed whether models were more persuasive and produced higher-quality arguments when defending positions consistent with their prior beliefs versus when arguing against them. Our main findings show that models tend to prefer defending stances aligned with the judge persona rather than their prior beliefs, sequential debate introduces significant bias favoring the second debater, models are more persuasive when defending positions aligned with their prior beliefs, and paradoxically, arguments misaligned with prior beliefs are rated as higher quality in pairwise comparison. These results can inform human judges to provide higher-quality training signals and contribute to more aligned AI systems, while revealing important aspects of human-AI interaction regarding persuasion dynamics in language models. 

---
# RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems 

**Authors**: Jingru Lin, Chen Zhang, Stephen Y. Liu, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13910)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates key limitations of Large Language Models (LLMs)-such as factual errors, outdated knowledge, and hallucinations-by dynamically retrieving external information. Recent work extends this paradigm through agentic RAG systems, where LLMs act as agents to iteratively plan, retrieve, and reason over complex queries. However, these systems still struggle with challenging multi-hop questions, and their intermediate reasoning capabilities remain underexplored. To address this, we propose RAGCap-Bench, a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs from state-of-the-art systems to identify common tasks and the core capabilities required for their execution, then construct a taxonomy of typical LLM errors to design targeted evaluation questions. Experiments show that "slow-thinking" models with stronger RAGCap performance achieve better end-to-end results, underscoring the benchmark's validity and the importance of enhancing these intermediate capabilities. 

---
# Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning 

**Authors**: Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Zhongyuan Wang, Jichen Zhang, Shirui Pan, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13909)  

**Abstract**: Inductive Knowledge Graph Reasoning (KGR) aims to discover facts in open-domain KGs containing unknown entities and relations, which poses a challenge for KGR models in comprehending uncertain KG components. Existing studies have proposed Knowledge Graph Foundation Models (KGFMs) that learn structural invariances across KGs to handle this uncertainty. Recently, Large Language Models (LLMs) have demonstrated strong capabilities for open-domain knowledge reasoning. As a result, the latest research has focused on LLM-based KGFMs that integrate LLM knowledge with KG context for inductive KGR. However, the intrinsic knowledge of LLMs may be overshadowed by sparse KG context, leading to LLM knowledge distortion, which can cause irreversible damage to model reasoning. Moreover, existing LLM-based KGR methods still struggle to fully constrain generative hallucinations in LLMs, severely limiting the credibility of reasoning results. To address these limitations, we propose a Knowledge Reasoning Language Model (KRLM) that achieves unified coordination between LLM knowledge and KG context throughout the KGR process. Specifically, we design a Knowledge Reasoning Language (KRL) instruction format and a KRL tokenizer to align LLM knowledge with KG representations. Then, we propose a KRL attention layer that coordinates intrinsic LLM knowledge with additional KG context through a dynamic knowledge memory mechanism. Finally, a structure-aware next-entity predictor is proposed, which strictly constrains the reasoning results within a trustworthy knowledge domain. Extensive experimental results on 25 real-world inductive KGR datasets demonstrate the significant superiority of the proposed KRLM\footnote{Our source codes are available at this https URL in both zero-shot reasoning and fine-tuning scenarios. 

---
# Interpreting the Latent Structure of Operator Precedence in Language Models 

**Authors**: Dharunish Yugeswardeenoo, Harshil Nukala, Cole Blondin, Sean O Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13908)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities but continue to struggle with arithmetic tasks. Prior works largely focus on outputs or prompting strategies, leaving the open question of the internal structure through which models do arithmetic computation. In this work, we investigate whether LLMs encode operator precedence in their internal representations via the open-source instruction-tuned LLaMA 3.2-3B model. We constructed a dataset of arithmetic expressions with three operands and two operators, varying the order and placement of parentheses. Using this dataset, we trace whether intermediate results appear in the residual stream of the instruction-tuned LLaMA 3.2-3B model. We apply interpretability techniques such as logit lens, linear classification probes, and UMAP geometric visualization. Our results show that intermediate computations are present in the residual stream, particularly after MLP blocks. We also find that the model linearly encodes precedence in each operator's embeddings post attention layer. We introduce partial embedding swap, a technique that modifies operator precedence by exchanging high-impact embedding dimensions between operators. 

---
# LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization 

**Authors**: Yuanchen Wu, Saurabh Verma, Justin Lee, Fangzhou Xiong, Poppy Zhang, Amel Awadelkarim, Xu Chen, Yubai Yuan, Shawndra Hill  

**Link**: [PDF](https://arxiv.org/pdf/2510.13907)  

**Abstract**: Large language models (LLMs) are highly sensitive to their input prompts, making prompt design a central challenge. While automatic prompt optimization (APO) reduces manual engineering, most approaches assume access to ground-truth references such as labeled validation data. In practice, however, collecting high-quality labels is costly and slow. We propose the Prompt Duel Optimizer (PDO), a sample-efficient framework for label-free prompt optimization. PDO formulates the problem as a dueling-bandit setting, where supervision signal comes from pairwise preference feedback provided by an LLM judge. The framework combines Double Thompson Sampling (D-TS), which prioritizes informative prompt comparisons, with Top-Performer Guided Mutation, which expands the candidate pool by mutating high-performing prompts. PDO naturally operates in label-free settings and can also incorporate partial labels to mitigate judge noise. Experiments on BIG-bench Hard (BBH) and MS MARCO show that PDO consistently outperforms baseline methods. Ablation studies further demonstrate the effectiveness of both D-TS and prompt mutation. 

---
# Schema for In-Context Learning 

**Authors**: Pan Chen, Shaohong Chen, Mark Wang, Shi Xuan Leong, Priscilla Fung, Varinia Bernales, Alan Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2510.13905)  

**Abstract**: In-Context Learning (ICL) enables transformer-based language models to adapt to new tasks by conditioning on demonstration examples. However, traditional example-driven in-context learning lacks explicit modules for knowledge retrieval and transfer at the abstraction level. Inspired by cognitive science, specifically schema theory, which holds that humans interpret new information by activating pre-existing mental frameworks (schemas) to structure understanding, we introduce SCHEMA ACTIVATED IN CONTEXT LEARNING (SA-ICL). This framework extracts the representation of the building blocks of cognition for the reasoning process instilled from prior examples, creating an abstracted schema, a lightweight, structured template of key inferential steps and their relationships, which is then used to augment a model's reasoning process when presented with a novel question. We demonstrate that a broad range of large language models (LLMs) lack the capacity to form and utilize internal schema-based learning representations implicitly, but instead benefit significantly from explicit schema-based scaffolding. Across chemistry and physics questions from the GPQA dataset, our experiments show that SA-ICL consistently boosts performance, up to 36.19 percent, when the single demonstration example is of high quality, which simultaneously reduces reliance on the number of demonstrations and enhances interpretability. SCHEMA ACTIVATED IN CONTEXT LEARNING not only bridges disparate ICL strategies ranging from pattern priming to Chain-of-Thought prompting, but also paves a new path for enhancing human-like reasoning in LLMs. 

---
# Investigating Political and Demographic Associations in Large Language Models Through Moral Foundations Theory 

**Authors**: Nicole Smith-Vaniz, Harper Lyon, Lorraine Steigner, Ben Armstrong, Nicholas Mattei  

**Link**: [PDF](https://arxiv.org/pdf/2510.13902)  

**Abstract**: Large Language Models (LLMs) have become increasingly incorporated into everyday life for many internet users, taking on significant roles as advice givers in the domains of medicine, personal relationships, and even legal matters. The importance of these roles raise questions about how and what responses LLMs make in difficult political and moral domains, especially questions about possible biases. To quantify the nature of potential biases in LLMs, various works have applied Moral Foundations Theory (MFT), a framework that categorizes human moral reasoning into five dimensions: Harm, Fairness, Ingroup Loyalty, Authority, and Purity. Previous research has used the MFT to measure differences in human participants along political, national, and cultural lines. While there has been some analysis of the responses of LLM with respect to political stance in role-playing scenarios, no work so far has directly assessed the moral leanings in the LLM responses, nor have they connected LLM outputs with robust human data. In this paper we analyze the distinctions between LLM MFT responses and existing human research directly, investigating whether commonly available LLM responses demonstrate ideological leanings: either through their inherent responses, straightforward representations of political ideologies, or when responding from the perspectives of constructed human personas. We assess whether LLMs inherently generate responses that align more closely with one political ideology over another, and additionally examine how accurately LLMs can represent ideological perspectives through both explicit prompting and demographic-based role-playing. By systematically analyzing LLM behavior across these conditions and experiments, our study provides insight into the extent of political and demographic dependency in AI-generated responses. 

---
# RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs 

**Authors**: Tuan T. Nguyen, John Le, Thai T. Vu, Willy Susilo, Heath Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2510.13901)  

**Abstract**: Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities. 

---
# Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences 

**Authors**: Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.13900)  

**Abstract**: Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research. 

---
# Attribution Quality in AI-Generated Content:Benchmarking Style Embeddings and LLM Judges 

**Authors**: Misam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2510.13898)  

**Abstract**: Attributing authorship in the era of large language models (LLMs) is increasingly challenging as machine-generated prose rivals human writing. We benchmark two complementary attribution mechanisms , fixed Style Embeddings and an instruction-tuned LLM judge (GPT-4o) on the Human AI Parallel Corpus, an open dataset of 600 balanced instances spanning six domains (academic, news, fiction, blogs, spoken transcripts, and TV/movie scripts). Each instance contains a human prompt with both a gold continuation and an LLM-generated continuation from either GPT-4o or LLaMA-70B-Instruct. The Style Embedding baseline achieves stronger aggregate accuracy on GPT continuations (82 pct vs. 68 pct). The LLM Judge is slightly better than the Style embeddings on LLaMA continuations (85 pct vs. 81 pct) but the results are not statistically significant. Crucially, the LLM judge significantly outperforms in fiction and academic prose, indicating semantic sensitivity, whereas embeddings dominate in spoken and scripted dialogue, reflecting structural strengths. These complementary patterns highlight attribution as a multidimensional problem requiring hybrid strategies. To support reproducibility we provide code on GitHub and derived data on Hugging Face under the MIT license. This open framework provides a reproducible benchmark for attribution quality assessment in AI-generated content, along with a review of related literature influencing this work. 

---
# Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection 

**Authors**: Olga E. Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13893)  

**Abstract**: Jailbreaking techniques pose a significant threat to the safety of Large Language Models (LLMs). Existing defenses typically focus on single-turn attacks, lack coverage across languages, and rely on limited taxonomies that either fail to capture the full diversity of attack strategies or emphasize risk categories rather than the jailbreaking techniques. To advance the understanding of the effectiveness of jailbreaking techniques, we conducted a structured red-teaming challenge. The outcome of our experiments are manifold. First, we developed a comprehensive hierarchical taxonomy of 50 jailbreak strategies, consolidating and extending prior classifications into seven broad families, including impersonation, persuasion, privilege escalation, cognitive overload, obfuscation, goal conflict, and data poisoning. Second, we analyzed the data collected from the challenge to examine the prevalence and success rates of different attack types, providing insights into how specific jailbreak strategies exploit model vulnerabilities and induce misalignment. Third, we benchmark a popular LLM for jailbreak detection, evaluating the benefits of taxonomy-guided prompting for improving automatic detection. Finally, we compiled a new Italian dataset of 1364 multi-turn adversarial dialogues, annotated with our taxonomy, enabling the study of interactions where adversarial intent emerges gradually and succeeds in bypassing traditional safeguards. 

---
# The Harder The Better: Maintaining Supervised Fine-tuning Generalization with Less but Harder Data 

**Authors**: Zhaoyang Shang, Sibo Wei, Jianbin Guo, Rui Zhou, Lifeng Dong, Yin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.13892)  

**Abstract**: Large Language Models (LLMs) excel in general tasks, but adapting them to specialized domains relies on high-quality supervised fine-tuning (SFT) data. Although existing methods can identify subsets of high-quality data and reduce training cost to some extent, their selection process still suffers from over-reliance on LLMs' internal knowledge, weak interpretability, and limited generalization. To address these limitations, we propose THTB (The Harder The Better), a cognitive science-inspired framework for instruction data selection and annotation guidance. THTB prioritizes higher-level cognitive instructions by combining quality filtering with intrinsic and extrinsic hardness scoring, offering interpretable and quantifiable criteria for efficient SFT, both in data selection and annotation guidance. Experiments show that THTB enables models trained on only 5% of the data to outperform full-dataset training, while achieving superior generalization compared with LLM-only selection. In addition, THTB provides effective annotation guidance in vertical domains, enabling a model trained on just 2% of the data to surpass models trained on much larger datasets, demonstrating strong potential for domain adaptation. Our code, datasets, and models are available on this https URL. 

---
# A Survey on Collaborating Small and Large Language Models for Performance, Cost-effectiveness, Cloud-edge Privacy, and Trustworthiness 

**Authors**: Fali Wang, Jihai Chen, Shuhua Yang, Ali Al-Lawati, Linli Tang, Hui Liu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13890)  

**Abstract**: Large language models (LLMs) have advanced many domains and applications but face high fine-tuning costs, inference latency, limited edge deployability, and reliability concerns. Small language models (SLMs), compact, efficient, and adaptable, offer complementary remedies. Recent work explores collaborative frameworks that fuse SLMs' specialization and efficiency with LLMs' generalization and reasoning to meet diverse objectives across tasks and deployment scenarios. Motivated by these developments, this paper presents a systematic survey of SLM-LLM collaboration organized by collaboration objectives. We propose a taxonomy with four goals: performance enhancement, cost-effectiveness, cloud-edge privacy, and trustworthiness. Within this framework, we review representative methods, summarize design paradigms, and outline open challenges and future directions toward efficient, secure, and scalable SLM-LLM collaboration. 

---
# Reliable Fine-Grained Evaluation of Natural Language Math Proofs 

**Authors**: Wenjie Ma, Andrei Cojocaru, Neel Kolhe, Bradley Louie, Robin Said Sharif, Haihan Zhang, Vincent Zhuang, Matei Zaharia, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2510.13888)  

**Abstract**: Recent advances in large language models (LLMs) for mathematical reasoning have largely focused on tasks with easily verifiable final answers; however, generating and verifying natural language math proofs remains an open challenge. We identify the absence of a reliable, fine-grained evaluator for LLM-generated math proofs as a critical gap. To address this, we propose a systematic methodology for developing and validating evaluators that assign fine-grained scores on a 0-7 scale to model-generated math proofs. To enable this study, we introduce ProofBench, the first expert-annotated dataset of fine-grained proof ratings, spanning 145 problems from six major math competitions (USAMO, IMO, Putnam, etc) and 435 LLM-generated solutions from Gemini-2.5-pro, o3, and DeepSeek-R1. %with expert gradings. Using ProofBench as a testbed, we systematically explore the evaluator design space across key axes: the backbone model, input context, instructions and evaluation workflow. Our analysis delivers ProofGrader, an evaluator that combines a strong reasoning backbone LM, rich context from reference solutions and marking schemes, and a simple ensembling method; it achieves a low Mean Absolute Error (MAE) of 0.926 against expert scores, significantly outperforming naive baselines. Finally, we demonstrate its practical utility in a best-of-$n$ selection task: at $n=16$, ProofGrader achieves an average score of 4.14 (out of 7), closing 78% of the gap between a naive binary evaluator (2.48) and the human oracle (4.62), highlighting its potential to advance downstream proof generation. 

---
# Order from Chaos: Comparative Study of Ten Leading LLMs on Unstructured Data Categorization 

**Authors**: Ariel Kamen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13885)  

**Abstract**: This study presents a comparative evaluation of ten state-of-the-art large language models (LLMs) applied to unstructured text categorization using the Interactive Advertising Bureau (IAB) 2.2 hierarchical taxonomy. The analysis employed a uniform dataset of 8,660 human-annotated samples and identical zero-shot prompts to ensure methodological consistency across all models. Evaluation metrics included four classic measures - accuracy, precision, recall, and F1-score - and three LLM-specific indicators: hallucination ratio, inflation ratio, and categorization cost.
Results show that, despite their rapid advancement, contemporary LLMs achieve only moderate classic performance, with average scores of 34% accuracy, 42% precision, 45% recall, and 41% F1-score. Hallucination and inflation ratios reveal that models frequently overproduce categories relative to human annotators. Among the evaluated systems, Gemini 1.5/2.0 Flash and GPT 20B/120B offered the most favorable cost-to-performance balance, while GPT 120B demonstrated the lowest hallucination ratio. The findings suggest that scaling and architectural improvements alone do not ensure better categorization accuracy, as the task requires compressing rich unstructured text into a limited taxonomy - a process that challenges current model architectures.
To address these limitations, a separate ensemble-based approach was developed and tested. The ensemble method, in which multiple LLMs act as independent experts, substantially improved accuracy, reduced inflation, and completely eliminated hallucinations. These results indicate that coordinated orchestration of models - rather than sheer scale - may represent the most effective path toward achieving or surpassing human-expert performance in large-scale text categorization. 

---
# Too Open for Opinion? Embracing Open-Endedness in Large Language Models for Social Simulation 

**Authors**: Bolei Ma, Yong Cao, Indira Sen, Anna-Carolina Haensch, Frauke Kreuter, Barbara Plank, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.13884)  

**Abstract**: Large Language Models (LLMs) are increasingly used to simulate public opinion and other social phenomena. Most current studies constrain these simulations to multiple-choice or short-answer formats for ease of scoring and comparison, but such closed designs overlook the inherently generative nature of LLMs. In this position paper, we argue that open-endedness, using free-form text that captures topics, viewpoints, and reasoning processes "in" LLMs, is essential for realistic social simulation. Drawing on decades of survey-methodology research and recent advances in NLP, we argue why this open-endedness is valuable in LLM social simulations, showing how it can improve measurement and design, support exploration of unanticipated views, and reduce researcher-imposed directive bias. It also captures expressiveness and individuality, aids in pretesting, and ultimately enhances methodological utility. We call for novel practices and evaluation frameworks that leverage rather than constrain the open-ended generative diversity of LLMs, creating synergies between NLP and social science. 

---
# PAGE: Prompt Augmentation for text Generation Enhancement 

**Authors**: Mauro Jose Pacchiotti, Luciana Ballejos, Mariel Ale  

**Link**: [PDF](https://arxiv.org/pdf/2510.13880)  

**Abstract**: In recent years, natural language generative models have shown outstanding performance in text generation tasks. However, when facing specific tasks or particular requirements, they may exhibit poor performance or require adjustments that demand large amounts of additional data. This work introduces PAGE (Prompt Augmentation for text Generation Enhancement), a framework designed to assist these models through the use of simple auxiliary modules. These modules, lightweight models such as classifiers or extractors, provide inferences from the input text. The output of these auxiliaries is then used to construct an enriched input that improves the quality and controllability of the generation. Unlike other generation-assistance approaches, PAGE does not require auxiliary generative models; instead, it proposes a simpler, modular architecture that is easy to adapt to different tasks. This paper presents the proposal, its components and architecture, and reports a proof of concept in the domain of requirements engineering, where an auxiliary module with a classifier is used to improve the quality of software requirements generation. 

---
# Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production 

**Authors**: Alexandre Galashov, Matt Jones, Rosemary Ke, Yuan Cao, Vaishnavh Nagarajan, Michael C. Mozer  

**Link**: [PDF](https://arxiv.org/pdf/2510.13879)  

**Abstract**: We explore a class of supervised training objectives that allow a language model to dynamically and autonomously scale the number of compute steps used for each input token. For any token, the model can request additional compute steps by emitting a <don't know> output. If the model is granted a delay, a specialized <pause> token is inserted at the next input step, providing the model with additional compute resources to generate an output. The model can request multiple pauses. To train the model to use <don't know> outputs judiciously and to calibrate its uncertainty, we frame the selection of each output token as a sequential-decision problem with a time cost. We refer to the class of methods as $\textit{Catch Your Breath}$ losses and we study three methods in this class: CYB-AP frames the model's task as anytime prediction, where an output may be required at any step and accuracy is discounted over time; CYB-VA is a variational approach that aims to maximize prediction accuracy subject to a specified distribution over stopping times; and CYB-DP imposes a penalty based on a computational budget. Through fine-tuning experiments, we identify the best performing loss variant. The CYB model needs only one third as much training data as the baseline (no pause) model needs to achieve the same performance, and half as much data as a model with pauses and a cross-entropy loss. We find that the CYB model requests additional steps when doing so improves accuracy, and the model adapts its processing time to token-level complexity and context. For example, it often pauses after plural nouns like $\textit{patients}$ and $\textit{challenges}$ but never pauses after the first token of contracted words like $\textit{wasn}$ and $\textit{didn}$, and it shows high variability for ambiguous tokens like $\textit{won}$, which could function as either a verb or part of a contraction. 

---
# TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks 

**Authors**: Jimin Lim, Arjun Damerla, Arthur Jiang, Nam Le  

**Link**: [PDF](https://arxiv.org/pdf/2510.13878)  

**Abstract**: Large language models (LLMs) have shown to be increasingly capable of performing reasoning tasks, but their ability to make sequential decisions under uncertainty only using natural language remains underexplored. We introduce a novel benchmark in which LLMs interact with multi-armed bandit environments using purely textual feedback, "you earned a token", without access to numerical cues or explicit probabilities, resulting in the model to infer latent reward structures purely off linguistic cues and to adapt accordingly. We evaluated the performance of four open-source LLMs and compare their performance to standard decision-making algorithms such as Thompson Sampling, Epsilon Greedy, Upper Confidence Bound (UCB), and random choice. While most of the LLMs underperformed compared to the baselines, Qwen3-4B, achieved the best-arm selection rate of 89.2% , which significantly outperformed both the larger LLMs and traditional methods. Our findings suggest that probabilistic reasoning is able to emerge from language alone, and we present this benchmark as a step towards evaluating decision-making capabilities in naturalistic, non-numeric contexts. 

---
# What Layers When: Learning to Skip Compute in LLMs with Residual Gates 

**Authors**: Filipe Laitenberger, Dawid Kopiczko, Cees G.M. Snoek, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2510.13876)  

**Abstract**: We introduce GateSkip, a simple residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch is equipped with a sigmoid-linear gate that condenses the branch's output before it re-enters the residual stream. During inference we rank tokens by the gate values and skip low-importance ones using a per-layer budget. While early-exit or router-based Mixture-of-Depths models are known to be unstable and need extensive retraining, our smooth, differentiable gates fine-tune stably on top of pretrained models. On long-form reasoning, we save up to 15\% compute while retaining over 90\% of baseline accuracy. On instruction-tuned models we see accuracy gains at full compute and match baseline quality near 50\% savings. The learned gates give insight into transformer information flow (e.g., BOS tokens act as anchors), and the method combines easily with quantization, pruning, and self-speculative decoding. 

---
# FRACCO: A gold-standard annotated corpus of oncological entities with ICD-O-3.1 normalisation 

**Authors**: Johann Pignat, Milena Vucetic, Christophe Gaudet-Blavignac, Jamil Zaghir, Amandine Stettler, Fanny Amrein, Jonatan Bonjour, Jean-Philippe Goldman, Olivier Michielin, Christian Lovis, Mina Bjelogrlic  

**Link**: [PDF](https://arxiv.org/pdf/2510.13873)  

**Abstract**: Developing natural language processing tools for clinical text requires annotated datasets, yet French oncology resources remain scarce. We present FRACCO (FRench Annotated Corpus for Clinical Oncology) an expert-annotated corpus of 1301 synthetic French clinical cases, initially translated from the Spanish CANTEMIST corpus as part of the FRASIMED initiative. Each document is annotated with terms related to morphology, topography, and histologic differentiation, using the International Classification of Diseases for Oncology (ICD-O) as reference. An additional annotation layer captures composite expression-level normalisations that combine multiple ICD-O elements into unified clinical concepts. Annotation quality was ensured through expert review: 1301 texts were manually annotated for entity spans by two domain experts. A total of 71127 ICD-O normalisations were produced through a combination of automated matching and manual validation by a team of five annotators. The final dataset representing 399 unique morphology codes (from 2549 different expressions), 272 topography codes (from 3143 different expressions), and 2043 unique composite expressions (from 11144 different expressions). This dataset provides a reference standard for named entity recognition and concept normalisation in French oncology texts. 

---
# Quechua Speech Datasets in Common Voice: The Case of Puno Quechua 

**Authors**: Elwin Huaman, Wendi Huaman, Jorge Luis Huaman, Ninfa Quispe  

**Link**: [PDF](https://arxiv.org/pdf/2510.13871)  

**Abstract**: Under-resourced languages, such as Quechuas, face data and resource scarcity, hindering their development in speech technology. To address this issue, Common Voice presents a crucial opportunity to foster an open and community-driven speech dataset creation. This paper examines the integration of Quechua languages into Common Voice. We detail the current 17 Quechua languages, presenting Puno Quechua (ISO 639-3: qxp) as a focused case study that includes language onboarding and corpus collection of both reading and spontaneous speech data. Our results demonstrate that Common Voice now hosts 191.1 hours of Quechua speech (86\% validated), with Puno Quechua contributing 12 hours (77\% validated), highlighting the Common Voice's potential. We further propose a research agenda addressing technical challenges, alongside ethical considerations for community engagement and indigenous data sovereignty. Our work contributes towards inclusive voice technology and digital empowerment of under-resourced language communities. 

---
# Unlocking the Potential of Diffusion Language Models through Template Infilling 

**Authors**: Junhoo Lee, Seungyeon Kim, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13870)  

**Abstract**: Diffusion Language Models (DLMs) have emerged as a promising alternative to Autoregressive Language Models, yet their inference strategies remain limited to prefix-based prompting inherited from the autoregressive paradigm. In this paper, we propose Template Infilling (TI), a tailored conditioning methodology for DLMs' generation process. Unlike conventional prefix prompting, TI first generates a structural template for the target response, then fills in the masked segments. To enhance the flexibility of this structural control, we introduce Dynamic Segment Allocation (DSA), which adaptively adjusts segment lengths based on generation confidence. We demonstrate the effectiveness of our approach on mathematical reasoning and code generation benchmarks, achieving consistent improvements of 17.01$\%$p over baseline. Furthermore, we show that TI provides additional advantages in multi-token generation settings, enabling effective speedup while maintaining generation quality. 

---
# Ensembling Large Language Models to Characterize Affective Dynamics in Student-AI Tutor Dialogues 

**Authors**: Chenyu Zhang, Sharifa Alghowinem, Cynthia Breazeal  

**Link**: [PDF](https://arxiv.org/pdf/2510.13862)  

**Abstract**: While recent studies have examined the leaning impact of large language model (LLM) in educational contexts, the affective dynamics of LLM-mediated tutoring remain insufficiently understood. This work introduces the first ensemble-LLM framework for large-scale affect sensing in tutoring dialogues, advancing the conversation on responsible pathways for integrating generative AI into education by attending to learners' evolving affective states. To achieve this, we analyzed two semesters' worth of 16,986 conversational turns exchanged between PyTutor, an LLM-powered AI tutor, and 261 undergraduate learners across three U.S. institutions. To investigate learners' emotional experiences, we generate zero-shot affect annotations from three frontier LLMs (Gemini, GPT-4o, Claude), including scalar ratings of valence, arousal, and learning-helpfulness, along with free-text emotion labels. These estimates are fused through rank-weighted intra-model pooling and plurality consensus across models to produce robust emotion profiles. Our analysis shows that during interaction with the AI tutor, students typically report mildly positive affect and moderate arousal. Yet learning is not uniformly smooth: confusion and curiosity are frequent companions to problem solving, and frustration, while less common, still surfaces in ways that can derail progress. Emotional states are short-lived--positive moments last slightly longer than neutral or negative ones, but they are fragile and easily disrupted. Encouragingly, negative emotions often resolve quickly, sometimes rebounding directly into positive states. Neutral moments frequently act as turning points, more often steering students upward than downward, suggesting opportunities for tutors to intervene at precisely these junctures. 

---
# ShishuLM: Lightweight Language Model with Hybrid Decoder-MLP Architecture and Paired Weight Sharing 

**Authors**: Shivanshu Kumar, Gopalakrishnan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13860)  

**Abstract**: While the transformer architecture has achieved state-of-the-art performance on natural language processing tasks, these models impose substantial memory and computational overhead. Recent research has identified significant architectural redundancies within these models, presenting opportunities for optimization without compromising performance. Taking insights from research in AI interpretability and inference-time layer pruning, we introduce an efficient language model architecture, referred to as ShishuLM, which reduces both the parameter count and Key-Value (KV) cache requirements. Given the increasing importance of Small Language Models (SLMs) in agentic AI systems, we evaluate our approach on two SLMs of different scales. Our analysis reveals that for moderate-context scenarios, normalization coupled with attention computation is roughly linear with the input, enabling entire transformer blocks to be approximated through Multi-Layer Perceptrons (MLPs). Our results show that ShishuLM provides up to 25% reduction in memory requirements and up to 40% improvement in latency during both training and inference, compared to parent models. Our experimental and analytical findings provide insights towards building more efficient SLM architectures from a pre-training standpoint. 

---
# Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2510.13856)  

**Abstract**: Medical Visual Question Answering (MedVQA) enables natural language queries over medical images to support clinical decision-making and patient care. The MEDIQA-WV 2025 shared task addressed wound-care VQA, requiring systems to generate free-text responses and structured wound attributes from images and patient queries. We present the MasonNLP system, which employs a general-domain, instruction-tuned large language model with a retrieval-augmented generation (RAG) framework that incorporates textual and visual examples from in-domain data. This approach grounds outputs in clinically relevant exemplars, improving reasoning, schema adherence, and response quality across dBLEU, ROUGE, BERTScore, and LLM-based metrics. Our best-performing system ranked 3rd among 19 teams and 51 submissions with an average score of 41.37%, demonstrating that lightweight RAG with general-purpose LLMs -- a minimal inference-time layer that adds a few relevant exemplars via simple indexing and fusion, with no extra training or complex re-ranking -- provides a simple and effective baseline for multimodal clinical NLP tasks. 

---
# Harnessing Consistency for Robust Test-Time LLM Ensemble 

**Authors**: Zhichen Zeng, Qi Yu, Xiao Lin, Ruizhong Qiu, Xuying Ning, Tianxin Wei, Yuchen Yan, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.13855)  

**Abstract**: Different large language models (LLMs) exhibit diverse strengths and weaknesses, and LLM ensemble serves as a promising approach to integrate their complementary capabilities. Despite substantial progress in improving ensemble quality, limited attention has been paid to the robustness of ensembles against potential erroneous signals, which often arise from heterogeneous tokenization schemes and varying model expertise. Our analysis shows that ensemble failures typically arise from both the token level and the model level: the former reflects severe disagreement in token predictions, while the latter involves low confidence and pronounced disparities among models. In light of this, we propose CoRE, a plug-and-play technique that harnesses model consistency for robust LLM ensemble, which can be seamlessly integrated with diverse ensemble methods. Token-level consistency captures fine-grained disagreements by applying a low-pass filter to downweight uncertain tokens with high inconsistency, often due to token misalignment, thereby improving robustness at a granular level. Model-level consistency models global agreement by promoting model outputs with high self-confidence and minimal divergence from others, enhancing robustness at a coarser level. Extensive experiments across diverse benchmarks, model combinations, and ensemble strategies demonstrate that CoRE consistently improves ensemble performance and robustness. 

---
# R2T: Rule-Encoded Loss Functions for Low-Resource Sequence Tagging 

**Authors**: Mamadou K. Keita, Christopher Homan, Sebastien Diarra  

**Link**: [PDF](https://arxiv.org/pdf/2510.13854)  

**Abstract**: We introduce the Rule-to-Tag (R2T) framework, a hybrid approach that integrates a multi-tiered system of linguistic rules directly into a neural network's training objective. R2T's novelty lies in its adaptive loss function, which includes a regularization term that teaches the model to handle out-of-vocabulary (OOV) words with principled uncertainty. We frame this work as a case study in a paradigm we call principled learning (PrL), where models are trained with explicit task constraints rather than on labeled examples alone. Our experiments on Zarma part-of-speech (POS) tagging show that the R2T-BiLSTM model, trained only on unlabeled text, achieves 98.2% accuracy, outperforming baselines like AfriBERTa fine-tuned on 300 labeled sentences. We further show that for more complex tasks like named entity recognition (NER), R2T serves as a powerful pre-training step; a model pre-trained with R2T and fine-tuned on just 50 labeled sentences outperformes a baseline trained on 300. 

---
# BenchPress: A Human-in-the-Loop Annotation System for Rapid Text-to-SQL Benchmark Curation 

**Authors**: Fabian Wenz, Omar Bouattour, Devin Yang, Justin Choi, Cecil Gregg, Nesime Tatbul, Çağatay Demiralp  

**Link**: [PDF](https://arxiv.org/pdf/2510.13853)  

**Abstract**: Large language models (LLMs) have been successfully applied to many tasks, including text-to-SQL generation. However, much of this work has focused on publicly available datasets, such as Fiben, Spider, and Bird. Our earlier work showed that LLMs are much less effective in querying large private enterprise data warehouses and released Beaver, the first private enterprise text-to-SQL benchmark. To create Beaver, we leveraged SQL logs, which are often readily available. However, manually annotating these logs to identify which natural language questions they answer is a daunting task. Asking database administrators, who are highly trained experts, to take on additional work to construct and validate corresponding natural language utterances is not only challenging but also quite costly. To address this challenge, we introduce BenchPress, a human-in-the-loop system designed to accelerate the creation of domain-specific text-to-SQL benchmarks. Given a SQL query, BenchPress uses retrieval-augmented generation (RAG) and LLMs to propose multiple natural language descriptions. Human experts then select, rank, or edit these drafts to ensure accuracy and domain alignment. We evaluated BenchPress on annotated enterprise SQL logs, demonstrating that LLM-assisted annotation drastically reduces the time and effort required to create high-quality benchmarks. Our results show that combining human verification with LLM-generated suggestions enhances annotation accuracy, benchmark reliability, and model evaluation robustness. By streamlining the creation of custom benchmarks, BenchPress offers researchers and practitioners a mechanism for assessing text-to-SQL models on a given domain-specific workload. BenchPress is freely available via our public GitHub repository at this https URL and is also accessible on our website at this http URL. 

---
# ConsistencyAI: A Benchmark to Assess LLMs' Factual Consistency When Responding to Different Demographic Groups 

**Authors**: Peter Banyas, Shristi Sharma, Alistair Simmons, Atharva Vispute  

**Link**: [PDF](https://arxiv.org/pdf/2510.13852)  

**Abstract**: Is an LLM telling you different facts than it's telling me? This paper introduces ConsistencyAI, an independent benchmark for measuring the factual consistency of large language models (LLMs) for different personas. ConsistencyAI tests whether, when users of different demographics ask identical questions, the model responds with factually inconsistent answers. Designed without involvement from LLM providers, this benchmark offers impartial evaluation and accountability. In our experiment, we queried 19 LLMs with prompts that requested 5 facts for each of 15 topics. We repeated this query 100 times for each LLM, each time adding prompt context from a different persona selected from a subset of personas modeling the general population. We processed the responses into sentence embeddings, computed cross-persona cosine similarity, and computed the weighted average of cross-persona cosine similarity to calculate factual consistency scores. In 100-persona experiments, scores ranged from 0.9065 to 0.7896, and the mean was 0.8656, which we adopt as a benchmark threshold. xAI's Grok-3 is most consistent, while several lightweight models rank lowest. Consistency varies by topic: the job market is least consistent, G7 world leaders most consistent, and issues like vaccines or the Israeli-Palestinian conflict diverge by provider. These results show that both the provider and the topic shape the factual consistency. We release our code and interactive demo to support reproducible evaluation and encourage persona-invariant prompting strategies. 

---
# EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing 

**Authors**: Sicheng Lyu, Yu Gu, Xinyu Wang, Jerry Huang, Sitao Luan, Yufei Cui, Xiao-Wen Chang, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13851)  

**Abstract**: Large language models (LLMs) require continual updates to rectify outdated or erroneous knowledge. Model editing has emerged as a compelling paradigm for introducing targeted modifications without the computational burden of full retraining. Existing approaches are mainly based on a locate-then-edit framework. However, in sequential editing contexts, where multiple updates are applied over time, they exhibit significant limitations and suffer from catastrophic interference, i.e., new edits compromise previously integrated updates and degrade preserved knowledge. To address these challenges, we introduce EvoEdit, a novel editing strategy that mitigates catastrophic interference through sequential null-space alignment, enabling stable and efficient model editing. By performing sequential null-space alignment for each incoming edit, EvoEdit preserves both original and previously modified knowledge representations and maintains output invariance on preserved knowledge even across long edit sequences, effectively mitigating interference. Evaluations on real-world sequential knowledge-editing benchmarks show that EvoEdit achieves better or comparable performance than prior state-of-the-art locate-then-edit techniques, with up to 3.53 times speedup. Overall, these results underscore the necessity of developing more principled approaches for designing LLMs in dynamically evolving information settings, while providing a simple yet effective solution with strong theoretical guarantees. 

---
# Revisiting the UID Hypothesis in LLM Reasoning Traces 

**Authors**: Minju Gwak, Guijin Son, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13850)  

**Abstract**: Large language models (LLMs) often solve problems using step-by-step Chain-of-Thought (CoT) reasoning, yet these intermediate steps are frequently unfaithful or hard to interpret. Inspired by the Uniform Information Density (UID) hypothesis in psycholinguistics -- which posits that humans communicate by maintaining a stable flow of information -- we introduce entropy-based metrics to analyze the information flow within reasoning traces. Surprisingly, across three challenging mathematical benchmarks, we find that successful reasoning in LLMs is globally non-uniform: correct solutions are characterized by uneven swings in information density, in stark contrast to human communication patterns. This result challenges assumptions about machine reasoning and suggests new directions for designing interpretable and adaptive reasoning models. 

---
# Language steering in latent space to mitigate unintended code-switching 

**Authors**: Andrey Goncharov, Nikolai Kondusov, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2510.13849)  

**Abstract**: Multilingual Large Language Models (LLMs) often exhibit unintended code-switching, reducing reliability in downstream tasks. We propose latent-space language steering, a lightweight inference-time method that identifies language directions via PCA on parallel translations and steers token embeddings along these axes to control language identity. Our approach mitigates code-switching while preserving semantics with negligible computational overhead and requires only minimal parallel data for calibration. Empirically, we achieve 95-99\% language classification accuracy using a single principal component and reduce next-token distributional divergence by up to 42% across multiple language pairs on Qwen2.5 and Llama-3.2 models. We further analyze the layer-wise evolution of language representations, revealing that language identity concentrates in final layers with near-perfect linear separability. 

---
# On-device System of Compositional Multi-tasking in Large Language Models 

**Authors**: Ondrej Bohdal, Konstantinos Theodosiadis, Asterios Mpatziakas, Dimitris Filippidis, Iro Spyrou, Christos Zonios, Anastasios Drosou, Dimosthenis Ioannidis, Kyeng-Hun Lee, Jijoong Moon, Hyeonmok Ko, Mete Ozay, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2510.13848)  

**Abstract**: Large language models (LLMs) are commonly adapted for diverse downstream tasks via parameter-efficient fine-tuning techniques such as Low-Rank Adapters (LoRA). While adapters can be combined to handle multiple tasks separately, standard approaches struggle when targeting the simultaneous execution of complex tasks, such as generating a translated summary from a long conversation. To address this challenge, we propose a novel approach tailored specifically for compositional multi-tasking scenarios involving summarization and translation. Our technique involves adding a learnable projection layer on top of the combined summarization and translation adapters. This design enables effective integration while maintaining efficiency through reduced computational overhead compared to alternative strategies requiring extensive retraining or sequential processing. We demonstrate the practical viability of our method within an on-device environment by developing an Android app capable of executing compositional tasks seamlessly. Experimental results indicate our solution performs well and is fast in both cloud-based and on-device implementations, highlighting the potential benefits of adopting our framework in real-world applications demanding high-speed operation alongside resource constraints. 

---
# DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models 

**Authors**: Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar  

**Link**: [PDF](https://arxiv.org/pdf/2510.13847)  

**Abstract**: Speculative decoding (a.k.a. speculative sampling) has become a standard way to accelerate LLM inference: a small drafter proposes multiple tokens and a large target model verifies them once per speculation length. Recently, scaling of the LLM vocabulary has pushed the number of tokens to grow substantially. While verification over the full vocabulary leaves the target model largely unaffected, the O(|V|d) parameters in the drafter's output head become a latency bottleneck, slowing the entire pipeline. Contemporary methods (e.g., FR-Spec, VocabTrim) restrict the drafter's vocabulary to a fixed subset of the target model's vocabulary, ranked in descending order of token frequency. Although this reduces draft-time compute, it is brittle, since: (i) frequency lists are corpus-dependent and require retuning to generalize, and (ii) static shortlists suppress rare or domain-specific tokens, lowering the expected number of tokens per verification step. We propose DynaSpec, a context-dependent dynamic shortlisting mechanism that is robust, speeds up drafting, and generalizes across diverse tasks. Concretely, we introduce lightweight, coarse-grained meta-classifiers that route contexts to a small number of token clusters; the union of the top-k selected clusters forms the drafter's shortlist, while verification retains the full vocabulary and exactness. The meta-classifier finishes its computation earlier than the drafter's hidden state generation by exploiting parallel execution of draft encoding and meta shortlisting on separate streams. On standard speculative-decoding benchmarks, we observe consistent gains in mean accepted length over fixed-shortlist baselines, while context-dependent selection enables smaller shortlists without degrading acceptance. 

---
# Serialized EHR make for good text representations 

**Authors**: Zhirong Chou, Quan Qin, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13843)  

**Abstract**: The emergence of foundation models in healthcare has opened new avenues for learning generalizable representations from large scale clinical data. Yet, existing approaches often struggle to reconcile the tabular and event based nature of Electronic Health Records (EHRs) with the sequential priors of natural language models. This structural mismatch limits their ability to capture longitudinal dependencies across patient encounters. We introduce SerialBEHRT, a domain aligned foundation model that extends SciBERT through additional pretraining on structured EHR sequences. SerialBEHRT is designed to encode temporal and contextual relationships among clinical events, thereby producing richer patient representations. We evaluate its effectiveness on the task of antibiotic susceptibility prediction, a clinically meaningful problem in antibiotic stewardship. Through extensive benchmarking against state of the art EHR representation strategies, we demonstrate that SerialBEHRT achieves superior and more consistent performance, highlighting the importance of temporal serialization in foundation model pretraining for healthcare. 

---
# ADMIT: Few-shot Knowledge Poisoning Attacks on RAG-based Fact Checking 

**Authors**: Yutao Wu, Xiao Liu, Yinghui Li, Yifeng Gao, Yifan Ding, Jiale Ding, Xiang Zheng, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13842)  

**Abstract**: Knowledge poisoning poses a critical threat to Retrieval-Augmented Generation (RAG) systems by injecting adversarial content into knowledge bases, tricking Large Language Models (LLMs) into producing attacker-controlled outputs grounded in manipulated context. Prior work highlights LLMs' susceptibility to misleading or malicious retrieved content. However, real-world fact-checking scenarios are more challenging, as credible evidence typically dominates the retrieval pool. To investigate this problem, we extend knowledge poisoning to the fact-checking setting, where retrieved context includes authentic supporting or refuting evidence. We propose \textbf{ADMIT} (\textbf{AD}versarial \textbf{M}ulti-\textbf{I}njection \textbf{T}echnique), a few-shot, semantically aligned poisoning attack that flips fact-checking decisions and induces deceptive justifications, all without access to the target LLMs, retrievers, or token-level control. Extensive experiments show that ADMIT transfers effectively across 4 retrievers, 11 LLMs, and 4 cross-domain benchmarks, achieving an average attack success rate (ASR) of 86\% at an extremely low poisoning rate of $0.93 \times 10^{-6}$, and remaining robust even in the presence of strong counter-evidence. Compared with prior state-of-the-art attacks, ADMIT improves ASR by 11.2\% across all settings, exposing significant vulnerabilities in real-world RAG-based fact-checking systems. 

---
# Meronymic Ontology Extraction via Large Language Models 

**Authors**: Dekai Zhang, Simone Conia, Antonio Rago  

**Link**: [PDF](https://arxiv.org/pdf/2510.13839)  

**Abstract**: Ontologies have become essential in today's digital age as a way of organising the vast amount of readily available unstructured text. In providing formal structure to this information, ontologies have immense value and application across various domains, e.g., e-commerce, where countless product listings necessitate proper product organisation. However, the manual construction of these ontologies is a time-consuming, expensive and laborious process. In this paper, we harness the recent advancements in large language models (LLMs) to develop a fully-automated method of extracting product ontologies, in the form of meronymies, from raw review texts. We demonstrate that the ontologies produced by our method surpass an existing, BERT-based baseline when evaluating using an LLM-as-a-judge. Our investigation provides the groundwork for LLMs to be used more generally in (product or otherwise) ontology extraction. 

---
# Seeing Hate Differently: Hate Subspace Modeling for Culture-Aware Hate Speech Detection 

**Authors**: Weibin Cai, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2510.13837)  

**Abstract**: Hate speech detection has been extensively studied, yet existing methods often overlook a real-world complexity: training labels are biased, and interpretations of what is considered hate vary across individuals with different cultural backgrounds. We first analyze these challenges, including data sparsity, cultural entanglement, and ambiguous labeling. To address them, we propose a culture-aware framework that constructs individuals' hate subspaces. To alleviate data sparsity, we model combinations of cultural attributes. For cultural entanglement and ambiguous labels, we use label propagation to capture distinctive features of each combination. Finally, individual hate subspaces, which in turn can further enhance classification performance. Experiments show our method outperforms state-of-the-art by 1.05\% on average across all metrics. 

---
# SIMBA UQ: Similarity-Based Aggregation for Uncertainty Quantification in Large Language Models 

**Authors**: Debarun Bhattacharjya, Balaji Ganesan, Junkyu Lee, Radu Marinescu, Katsiaryna Mirylenka, Michael Glass, Xiao Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.13836)  

**Abstract**: When does a large language model (LLM) know what it does not know? Uncertainty quantification (UQ) provides measures of uncertainty, such as an estimate of the confidence in an LLM's generated output, and is therefore increasingly recognized as a crucial component of trusted AI systems. Black-box UQ methods do not require access to internal model information from the generating LLM and therefore have numerous real-world advantages, such as robustness to system changes, adaptability to choice of LLM, reduced costs, and computational tractability. In this paper, we investigate the effectiveness of UQ techniques that are primarily but not necessarily entirely black-box, where the consistency between a generated output and other sampled generations is used as a proxy for confidence in its correctness. We propose a high-level non-verbalized similarity-based aggregation framework that subsumes a broad swath of UQ approaches suitable for complex generative tasks, as well as introduce specific novel techniques from the framework that train confidence estimation models using small training sets. Through an empirical study with datasets spanning the diverse tasks of question answering, summarization, and text-to-SQL, we demonstrate that our proposed similarity-based methods can yield better calibrated confidences than baselines. 

---
# ConDABench: Interactive Evaluation of Language Models for Data Analysis 

**Authors**: Avik Dutta, Priyanshu Gupta, Hosein Hasanbeig, Rahul Pratap Singh, Harshit Nigam, Sumit Gulwani, Arjun Radhakrishna, Gustavo Soares, Ashish Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2510.13835)  

**Abstract**: Real-world data analysis tasks often come with under-specified goals and unclean data. User interaction is necessary to understand and disambiguate a user's intent, and hence, essential to solving these complex tasks. Existing benchmarks for evaluating LLMs on data analysis tasks do not capture these complexities or provide first-class support for interactivity. We introduce ConDABench, a framework for generating conversational data analysis (ConDA) benchmarks and evaluating external tools on the generated benchmarks. \bench consists of (a) a multi-agent workflow for generating realistic benchmarks from articles describing insights gained from public datasets, (b) 1,420 ConDA problems generated using this workflow, and (c) an evaluation harness that, for the first time, makes it possible to systematically evaluate conversational data analysis tools on the generated ConDA problems. Evaluation of state-of-the-art LLMs on the benchmarks reveals that while the new generation of models are better at solving more instances, they are not necessarily better at solving tasks that require sustained, long-form engagement. ConDABench is an avenue for model builders to measure progress towards truly collaborative models that can complete complex interactive tasks. 

---
# Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning 

**Authors**: Minsik Choi, Hyegang Son, Changhoon Kim, Young Geun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13832)  

**Abstract**: Transformer-based models have achieved remarkable performance in NLP tasks. However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment. To address these challenges, various pruning methods have recently been proposed. Notably, gradient-based methods using Head Importance Scores (HIS) have gained traction for interpretability, efficiency, and ability to identify redundant heads. However, HIS alone has limitations as it captures only the gradient-driven contribution, overlooking the diversity of attention patterns. To overcome these limitations, we introduce a novel pruning criterion, HIES (Head Importance-Entropy Score), which integrates head importance scores with attention entropy, providing complementary evidence on per-head contribution. Empirically, HIES-based pruning yields up to 15.2% improvement in model quality and 2.04x improvement in stability over HIS-only methods, enabling substantial model compression without sacrificing either accuracy or stability. Code will be released upon publication. 

---
# Informed Routing in LLMs: Smarter Token-Level Computation for Faster Inference 

**Authors**: Chao Han, Yijuan Liang, Zihao Xuan, Daokuan Wu, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13831)  

**Abstract**: The deployment of large language models (LLMs) in real-world applications is increasingly limited by their high inference cost. While recent advances in dynamic token-level computation allocation attempt to improve efficiency by selectively activating model components per token, existing methods rely on greedy routing--a myopic execute-or-skip mechanism that often leads to irreversible information loss and suboptimal token selection. This paper introduces informed routing, a new paradigm that proactively addresses these issues. The key insight is to assess not only a token's immediate importance but also its recoverability, i.e., how well its transformation can be approximated. To this end, we propose the Lightweight Feature Forecaster (LFF), a small predictive module that estimates a unit's output before routing decisions are made. This enables a flexible execute-or-approximate policy that preserves model fidelity while drastically reducing computation. Extensive experiments on both language modeling and reasoning tasks show that informed routing achieves state-of-the-art efficiency-performance trade-offs across multiple sparsity levels. Notably, even without final LoRA fine-tuning, our method matches or surpasses strong baselines that require full fine-tuning, all while reducing training time by over 50%. The code is available at: this https URL 

---
# Users as Annotators: LLM Preference Learning from Comparison Mode 

**Authors**: Zhongze Cai, Xiaocheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13830)  

**Abstract**: Pairwise preference data have played an important role in the alignment of large language models (LLMs). Each sample of such data consists of a prompt, two different responses to the prompt, and a binary label indicating which of the two responses is better. The labels are usually annotated by professional human annotators. In this paper, we consider an alternative approach to collect pairwise preference data -- user annotation from comparison mode. With the increasingly wider adoption of LLMs among the population, users are contributing more and more of their preference labels through their daily interactions with the LLMs. The upside of such labels is that users are the best experts in judging the responses to their own queries/prompts, but the downside is the lack of quality control in these labels. In this paper, we consider a new idea of generating two responses from two different models or two different versions of the same model. The asymmetry allows us to make an inference of the user's data quality through our proposed user behavior model. We develop an expectation-maximization algorithm to estimate a latent quality factor of the user, and filter users' annotation data accordingly. The downstream task shows the effectiveness of our approach in both capturing the user behavior and data filtering for LLM alignment. 

---
# A Linguistics-Aware LLM Watermarking via Syntactic Predictability 

**Authors**: Shinwoo Park, Hyejin Park, Hyeseon Ahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.13829)  

**Abstract**: As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthen it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at this https URL. 

---
# From Explainability to Action: A Generative Operational Framework for Integrating XAI in Clinical Mental Health Screening 

**Authors**: Ratna Kandala, Akshata Kishore Moharir, Divya Arvinda Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13828)  

**Abstract**: Explainable Artificial Intelligence (XAI) has been presented as the critical component for unlocking the potential of machine learning in mental health screening (MHS). However, a persistent lab-to-clinic gap remains. Current XAI techniques, such as SHAP and LIME, excel at producing technically faithful outputs such as feature importance scores, but fail to deliver clinically relevant, actionable insights that can be used by clinicians or understood by patients. This disconnect between technical transparency and human utility is the primary barrier to real-world adoption. This paper argues that this gap is a translation problem and proposes the Generative Operational Framework, a novel system architecture that leverages Large Language Models (LLMs) as a central translation engine. This framework is designed to ingest the raw, technical outputs from diverse XAI tools and synthesize them with clinical guidelines (via RAG) to automatically generate human-readable, evidence-backed clinical narratives. To justify our solution, we provide a systematic analysis of the components it integrates, tracing the evolution from intrinsic models to generative XAI. We demonstrate how this framework directly addresses key operational barriers, including workflow integration, bias mitigation, and stakeholder-specific communication. This paper also provides a strategic roadmap for moving the field beyond the generation of isolated data points toward the delivery of integrated, actionable, and trustworthy AI in clinical practice. 

---
# Bridging the Semantic Gap: Contrastive Rewards for Multilingual Text-to-SQL 

**Authors**: Ashish Kattamuri, Ishita Prasad, Meetu Malhotra, Arpita Vats, Rahul Raja, Albert Lie  

**Link**: [PDF](https://arxiv.org/pdf/2510.13827)  

**Abstract**: Current Text-to-SQL methods are evaluated and only focused on executable queries, overlooking the semantic alignment challenge -- both in terms of the semantic meaning of the query and the correctness of the execution results. Even execution accuracy itself shows significant drops when moving from English to other languages, with an average decline of 6 percentage points across non-English languages. We address these challenges by presenting a new framework that combines Group Relative Policy Optimization (GRPO) within a multilingual contrastive reward signal to enhance both task efficiency and semantic accuracy in Text-to-SQL systems in cross-lingual scenarios. Our method teaches models to obtain better correspondence between SQL generation and user intent by combining a reward signal based on semantic similarity. On the seven-language MultiSpider dataset, fine-tuning the LLaMA-3-3B model with GRPO improved the execution accuracy up to 87.4 percent (+26 pp over zero-shot) and semantic accuracy up to 52.29 percent (+32.86 pp). Adding our contrastive reward signal in the GRPO framework further improved the average semantic accuracy to 59.14 percent (+6.85 pp, up to +10 pp for Vietnamese). Our experiments showcase that a smaller, parameter-efficient 3B LLaMA model fine-tuned with our contrastive reward signal outperforms a much larger zero-shot 8B LLaMA model, with an uplift of 7.43 pp in execution accuracy (from 81.43 percent on the 8B model to 88.86 percent on the 3B model), and nearly matches its semantic accuracy (59.14 percent vs. 68.57 percent) -- all using just 3,000 reinforcement learning training examples. These results demonstrate how we can improve the performance of Text-to-SQL systems with contrastive rewards for directed semantic alignment, without requiring large-scale training datasets. 

---
# Agentic Design of Compositional Machines 

**Authors**: Wenqian Zhang, Weiyang Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14980)  

**Abstract**: The design of complex machines stands as both a marker of human intelligence and a foundation of engineering practice. Given recent advances in large language models (LLMs), we ask whether they, too, can learn to create. We approach this question through the lens of compositional machine design: a task in which machines are assembled from standardized components to meet functional demands like locomotion or manipulation in a simulated physical environment. To support this investigation, we introduce BesiegeField, a testbed built on the machine-building game Besiege, which enables part-based construction, physical simulation and reward-driven evaluation. Using BesiegeField, we benchmark state-of-the-art LLMs with agentic workflows and identify key capabilities required for success, including spatial reasoning, strategic assembly, and instruction-following. As current open-source models fall short, we explore reinforcement learning (RL) as a path to improvement: we curate a cold-start dataset, conduct RL finetuning experiments, and highlight open challenges at the intersection of language, machine design, and physical reasoning. 

---
# Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models 

**Authors**: Jonas Geiping, Xinyu Yang, Guinan Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.14961)  

**Abstract**: Language models with recurrent depth, also referred to as universal or looped when considering transformers, are defined by the capacity to increase their computation through the repetition of layers. Recent efforts in pretraining have demonstrated that these architectures can scale to modern language modeling tasks while exhibiting advantages in reasoning tasks. In this work, we examine the relationship between recurrent-depth models and diffusion language models. Building on their similarities, we develop a new diffusion forcing sampler for these models to accelerate generation. The sampler advances by decoding new tokens at every forward pass of the model, while the latent states of these tokens can be further refined in parallel through recurrence. Theoretically, generation with our sampler is strictly more expressive than the baseline autoregressive generation using the same time budget on modern hardware. Moreover, this sampler, based on principles from diffusion literature, can be directly applied to existing 3.5B recurrent-depth transformers without any tuning, leading to up to a 5x speedup. Consequently, our findings not only provide an efficient mechanism for parallelizing the extra computation in recurrent-depth models at inference, but also suggest that such models can be naturally viewed as strong continuous, though causal, diffusion language models. 

---
# MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning 

**Authors**: Weikang Shi, Aldrich Yu, Rongyao Fang, Houxing Ren, Ke Wang, Aojun Zhou, Changyao Tian, Xinyu Fu, Yuxuan Hu, Zimu Lu, Linjiang Huang, Si Liu, Rui Liu, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14958)  

**Abstract**: While Large Language Models (LLMs) have excelled in textual reasoning, they struggle with mathematical domains like geometry that intrinsically rely on visual aids. Existing approaches to Visual Chain-of-Thought (VCoT) are often limited by rigid external tools or fail to generate the high-fidelity, strategically-timed diagrams necessary for complex problem-solving. To bridge this gap, we introduce MathCanvas, a comprehensive framework designed to endow unified Large Multimodal Models (LMMs) with intrinsic VCoT capabilities for mathematics. Our approach consists of two phases. First, a Visual Manipulation stage pre-trains the model on a novel 15.2M-pair corpus, comprising 10M caption-to-diagram pairs (MathCanvas-Imagen) and 5.2M step-by-step editing trajectories (MathCanvas-Edit), to master diagram generation and editing. Second, a Strategic Visual-Aided Reasoning stage fine-tunes the model on MathCanvas-Instruct, a new 219K-example dataset of interleaved visual-textual reasoning paths, teaching it when and how to leverage visual aids. To facilitate rigorous evaluation, we introduce MathCanvas-Bench, a challenging benchmark with 3K problems that require models to produce interleaved visual-textual solutions. Our model, BAGEL-Canvas, trained under this framework, achieves an 86% relative improvement over strong LMM baselines on MathCanvas-Bench, demonstrating excellent generalization to other public math benchmarks. Our work provides a complete toolkit-framework, datasets, and benchmark-to unlock complex, human-like visual-aided reasoning in LMMs. Project Page: this https URL 

---
# Circuit Insights: Towards Interpretability Beyond Activations 

**Authors**: Elena Golimblevskaia, Aakriti Jain, Bruno Puri, Ammar Ibrahim, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14936)  

**Abstract**: The fields of explainable AI and mechanistic interpretability aim to uncover the internal structure of neural networks, with circuit discovery as a central tool for understanding model computations. Existing approaches, however, rely on manual inspection and remain limited to toy tasks. Automated interpretability offers scalability by analyzing isolated features and their activations, but it often misses interactions between features and depends strongly on external LLMs and dataset quality. Transcoders have recently made it possible to separate feature attributions into input-dependent and input-invariant components, providing a foundation for more systematic circuit analysis. Building on this, we propose WeightLens and CircuitLens, two complementary methods that go beyond activation-based analysis. WeightLens interprets features directly from their learned weights, removing the need for explainer models or datasets while matching or exceeding the performance of existing methods on context-independent features. CircuitLens captures how feature activations arise from interactions between components, revealing circuit-level dynamics that activation-only approaches cannot identify. Together, these methods increase interpretability robustness and enhance scalable mechanistic analysis of circuits while maintaining efficiency and quality. 

---
# Stable but Miscalibrated: A Kantian View on Overconfidence from Filters to Large Language Models 

**Authors**: Akira Okutomi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14925)  

**Abstract**: We reinterpret Kant's Critique of Pure Reason as a theory of feedback stability, viewing reason as a regulator that keeps inference within the bounds of possible experience. We formalize this intuition via a composite instability index (H-Risk) combining spectral margin, conditioning, temporal sensitivity, and innovation amplification. In linear-Gaussian simulations, higher H-Risk predicts overconfident errors even under formal stability, revealing a gap between nominal and epistemic stability. Extending to large language models (LLMs), we find that fragile internal dynamics correlate with miscalibration and hallucination, while critique-style prompts show mixed effects on calibration and hallucination. These results suggest a structural bridge between Kantian self-limitation and feedback control, offering a principled lens for diagnosing -- and selectively reducing -- overconfidence in reasoning systems. This is a preliminary version; supplementary experiments and broader replication will be reported in a future revision. 

---
# TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG 

**Authors**: Annisaa Fitri Nurfidausi, Eleonora Mancini, Paolo Torroni  

**Link**: [PDF](https://arxiv.org/pdf/2510.14922)  

**Abstract**: Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection. 

---
# Budget-aware Test-time Scaling via Discriminative Verification 

**Authors**: Kyle Montgomery, Sijun Tan, Yuqi Chen, Siyuan Zhuang, Tianjun Zhang, Raluca Ada Popa, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14913)  

**Abstract**: Test-time scaling is a powerful strategy for boosting the performance of large language models on complex reasoning tasks. While state-of-the-art approaches often employ generative verifiers to select the best solution from a pool of candidates, this method incurs prohibitive computational costs, limiting its practicality. In this work, we shift the focus to a more budget-aware paradigm: discriminative verification. We conduct a thorough empirical analysis and demonstrate that while discriminative verifiers may underperform in isolation, combining them with self-consistency in a hybrid approach creates a powerful and efficient test-time scaling mechanism. Notably, under a fixed compute budget, this hybrid approach surpasses state-of-the-art generative verification by a significant margin: achieving up to 15.3\% higher accuracy on AIME2025. Our findings establish that for practical, real-world applications, budget-aware scaling with discriminative verifiers is not only a "free" upgrade over self-consistency, but also a more effective and efficient alternative to costly generative techniques. Code is available at this https URL. 

---
# Reasoning with Sampling: Your Base Model is Smarter Than You Think 

**Authors**: Aayush Karan, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.14901)  

**Abstract**: Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilites can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains. 

---
# Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media 

**Authors**: Soorya Ram Shimgekar, Ruining Zhao, Agam Goyal, Violeta J. Rodriguez, Paul A. Bloom, Hari Sundaram, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2510.14889)  

**Abstract**: On social media, many individuals experiencing suicidal ideation (SI) do not disclose their distress explicitly. Instead, signs may surface indirectly through everyday posts or peer interactions. Detecting such implicit signals early is critical but remains challenging. We frame early and implicit SI as a forward-looking prediction task and develop a computational framework that models a user's information environment, consisting of both their longitudinal posting histories as well as the discourse of their socially proximal peers. We adopted a composite network centrality measure to identify top neighbors of a user, and temporally aligned the user's and neighbors' interactions -- integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves early and implicit SI detection by 15% over individual-only baselines. These findings highlight that peer interactions offer valuable predictive signals and carry broader implications for designing early detection systems that capture indirect as well as masked expressions of risk in online environments. 

---
# You May Speak Freely: Improving the Fine-Grained Visual Recognition Capabilities of Multimodal Large Language Models with Answer Extraction 

**Authors**: Logan Lawrence, Oindrila Saha, Megan Wei, Chen Sun, Subhransu Maji, Grant Van Horn  

**Link**: [PDF](https://arxiv.org/pdf/2510.14885)  

**Abstract**: Despite the renewed interest in zero-shot visual classification due to the rise of Multimodal Large Language Models (MLLMs), the problem of evaluating free-form responses of auto-regressive models remains a persistent challenge. Most existing works focus on language-only tasks or don't consider Multiple Choice Questions (MCQs) beyond 5-way options, both of which are critical capabilities to solve tasks in Fine-Grained Visual Classification (FGVC) where choice counts are in the hundreds to thousands and the choices are highly related. Furthermore, in this highly multi-way MCQ setting it is not clear how to extend LLM choice extraction to retrieval-based problems, where computing probabilities over the choice set is computationally costly. In this work we investigate nlg2choice, a simple two-stage method which first asks the MLLM an open-ended question for the task with minimal constraints, then uses text-only constrained decoding to predict the most likely choice. In retrieval settings, we compute the probability of the constrained response taking that choice with an early stopping method to significantly improve throughput. Our results show improvement over a suite of seven fine-grained visual datasets when evaluating in terms of classification and retrieval, and show that this performance holds over the various ways that users of LLMs can implement tasks in natural language. 

---
# Benchmarking Multimodal Large Language Models for Face Recognition 

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2510.14866)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable performance across diverse vision-and-language tasks. However, their potential in face recognition remains underexplored. In particular, the performance of open-source MLLMs needs to be evaluated and compared with existing face recognition models on standard benchmarks with similar protocol. In this work, we present a systematic benchmark of state-of-the-art MLLMs for face recognition on several face recognition datasets, including LFW, CALFW, CPLFW, CFP, AgeDB and RFW. Experimental results reveal that while MLLMs capture rich semantic cues useful for face-related tasks, they lag behind specialized models in high-precision recognition scenarios in zero-shot applications. This benchmark provides a foundation for advancing MLLM-based face recognition, offering insights for the design of next-generation models with higher accuracy and generalization. The source code of our benchmark is publicly available in the project page. 

---
# Where to Search: Measure the Prior-Structured Search Space of LLM Agents 

**Authors**: Zhuo-Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.14846)  

**Abstract**: The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs. 

---
# TITAN: Graph-Executable Reasoning for Cyber Threat Intelligence 

**Authors**: Marco Simoni, Aleksandar Fontana, Andrea Saracino, Paolo Mori  

**Link**: [PDF](https://arxiv.org/pdf/2510.14670)  

**Abstract**: TITAN (Threat Intelligence Through Automated Navigation) is a framework that connects natural-language cyber threat queries with executable reasoning over a structured knowledge graph. It integrates a path planner model, which predicts logical relation chains from text, and a graph executor that traverses the TITAN Ontology to retrieve factual answers and supporting evidence. Unlike traditional retrieval systems, TITAN operates on a typed, bidirectional graph derived from MITRE, allowing reasoning to move clearly and reversibly between threats, behaviors, and defenses. To support training and evaluation, we introduce the TITAN Dataset, a corpus of 88209 examples (Train: 74258; Test: 13951) pairing natural language questions with executable reasoning paths and step by step Chain of Thought explanations. Empirical evaluations show that TITAN enables models to generate syntactically valid and semantically coherent reasoning paths that can be deterministically executed on the underlying graph. 

---
# ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks 

**Authors**: Yuanyi Song, Heyuan Huang, Qiqiang Lin, Yin Zhao, Xiangmou Qu, Jun Wang, Xingyu Lou, Weiwen Liu, Zhuosheng Zhang, Jun Wang, Yong Yu, Weinan Zhang, Zhaoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14621)  

**Abstract**: The rapid advancement of multimodal large language models has enabled agents to operate mobile devices by directly interacting with graphical user interfaces, opening new possibilities for mobile automation. However, real-world mobile tasks are often complex and allow for multiple valid solutions. This contradicts current mobile agent evaluation standards: offline static benchmarks can only validate a single predefined "golden path", while online dynamic testing is constrained by the complexity and non-reproducibility of real devices, making both approaches inadequate for comprehensively assessing agent capabilities. To bridge the gap between offline and online evaluation and enhance testing stability, this paper introduces a novel graph-structured benchmarking framework. By modeling the finite states observed during real-device interactions, it achieves static simulation of dynamic behaviors. Building on this, we develop ColorBench, a benchmark focused on complex long-horizon tasks. It supports evaluation of multiple valid solutions, subtask completion rate statistics, and atomic-level capability analysis. ColorBench contains 175 tasks (74 single-app, 101 cross-app) with an average length of over 13 steps. Each task includes at least two correct paths and several typical error paths, enabling quasi-dynamic interaction. By evaluating ColorBench across various baselines, we discover limitations of existing models and propose improvement directions and feasible technical pathways to enhance agents' performance on complex, long-horizon problems based on experimental results. Code and data are available at: this https URL. 

---
# Just-In-Time Objectives: A General Approach for Specialized AI Interactions 

**Authors**: Michelle S. Lam, Omar Shaikh, Hallie Xu, Alice Guo, Diyi Yang, Jeffrey Heer, James A. Landay, Michael S. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.14591)  

**Abstract**: Large language models promise a broad set of functions, but when not given a specific objective, they default to milquetoast results such as drafting emails littered with cliches. We demonstrate that inferring the user's in-the-moment objective, then rapidly optimizing for that singular objective, enables LLMs to produce tools, interfaces, and responses that are more responsive and desired. We contribute an architecture for automatically inducing just-in-time objectives by passively observing user behavior, then steering downstream AI systems through generation and evaluation against this objective. Inducing just-in-time objectives (e.g., "Clarify the abstract's research contribution") enables automatic generation of tools, e.g., those that critique a draft based on relevant HCI methodologies, anticipate related researchers' reactions, or surface ambiguous terminology. In a series of experiments (N=14, N=205) on participants' own tasks, JIT objectives enable LLM outputs that achieve 66-86% win rates over typical LLMs, and in-person use sessions (N=17) confirm that JIT objectives produce specialized tools unique to each participant. 

---
# Talking Points: Describing and Localizing Pixels 

**Authors**: Matan Rusanovsky, Shimon Malnick, Shai Avidan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14583)  

**Abstract**: Vision-language models have achieved remarkable success in cross-modal understanding. Yet, these models remain limited to object-level or region-level grounding, lacking the capability for pixel-precise keypoint comprehension through natural language. We introduce a novel framework for pixel level grounding. The framework consists of two complementary components: a Point Descriptor that generates rich, contextual descriptions of individual keypoints, and a Point Localizer that regresses precise pixel coordinates from these descriptions. Unlike prior work that relies on templated prompts or keypoint names, our approach produces free-form, coarse-to-fine descriptions that situate keypoints within their visual context. Since there is no available dataset to train such a system, we introduce LlamaPointInPart, a carefully curated dataset of 20K+ image-keypoint-description triplets synthesized from multiple vision-language models, capturing multi-scale information from scene-level context to visual features around the keypoint. For cross-category generalization, we optimize the Point Descriptor on AP-10K via GRPO, using the frozen Point Localizer as a reward model to produce descriptions that maximize localization accuracy. To evaluate our results we establish a new evaluation protocol. Instead of comparing the text description produced by our method to the ground truth, we use the localizer to determine how close is the predicted point generated to the ground truth point. Experiments demonstrate superior performance compared to baseline models on this http URL bidirectional nature of our framework should enable future applications in both keypoint-guided image understanding and language-guided precise localization. Our code and dataset are publicly available at this https URL. 

---
# Agentic Entropy-Balanced Policy Optimization 

**Authors**: Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14545)  

**Abstract**: Recently, Agentic Reinforcement Learning (Agentic RL) has made significant progress in incentivizing the multi-turn, long-horizon tool-use capabilities of web agents. While mainstream agentic RL algorithms autonomously explore high-uncertainty tool-call steps under the guidance of entropy, excessive reliance on entropy signals can impose further constraints, leading to the training collapse. In this paper, we delve into the challenges caused by entropy and propose the Agentic Entropy-Balanced Policy Optimization (AEPO), an agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. AEPO comprises two core components: (1) a dynamic entropy-balanced rollout mechanism that adaptively allocate global and branch sampling budget through entropy pre-monitoring, while imposing a branch penalty on consecutive high-entropy tool-call steps to prevent over-branching issues; and (2) Entropy-Balanced Policy Optimization that inserts a stop-gradient operation into the high-entropy clipping term to preserve and properly rescale gradients on high-entropy tokens, while incorporating entropy-aware advantage estimation to prioritize learning on high-uncertainty tokens. Results across 14 challenging datasets show that AEPO consistently outperforms 7 mainstream RL algorithms. With just 1K RL samples, Qwen3-14B with AEPO achieves impressive results: 47.6% on GAIA, 11.2% on Humanity's Last Exam, and 43.0% on WebWalker for Pass@1; 65.0% on GAIA, 26.0% on Humanity's Last Exam, and 70.0% on WebWalker for Pass@5. Further analysis reveals that AEPO improves rollout sampling diversity while maintaining stable policy entropy, facilitating scalable web agent training. 

---
# E2Edev: Benchmarking Large Language Models in End-to-End Software Development Task 

**Authors**: Jingyao Liu, Chen Huang, Zhizhao Guan, Wenqiang Lei, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14509)  

**Abstract**: E2EDev comprises (i) a fine-grained set of user requirements, (ii) {multiple BDD test scenarios with corresponding Python step implementations for each requirement}, and (iii) a fully automated testing pipeline built on the Behave framework. To ensure its quality while reducing the annotation effort, E2EDev leverages our proposed Human-in-the-Loop Multi-Agent Annotation Framework (HITL-MAA). {By evaluating various E2ESD frameworks and LLM backbones with E2EDev}, our analysis reveals a persistent struggle to effectively solve these tasks, underscoring the critical need for more effective and cost-efficient E2ESD solutions. Our codebase and benchmark are publicly available at this https URL. 

---
# IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning 

**Authors**: Xikai Zhang, Bo Wang, Likang Xiao, Yongzhi Li, Quan Chen, Wenju Wu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14406)  

**Abstract**: Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size. 

---
# Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers 

**Authors**: Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, Emily Lawton, Keegan Hines, Gao Huang, Jack W. Stokes  

**Link**: [PDF](https://arxiv.org/pdf/2510.14381)  

**Abstract**: Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks. 

---
# AI for Service: Proactive Assistance with AI Glasses 

**Authors**: Zichen Wen, Yiyu Wang, Chenfei Liao, Boxue Yang, Junxian Li, Weifeng Liu, Haocong He, Bolong Feng, Xuyang Liu, Yuanhuiyi Lyu, Xu Zheng, Xuming Hu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14359)  

**Abstract**: In an era where AI is evolving from a passive tool into an active and adaptive companion, we introduce AI for Service (AI4Service), a new paradigm that enables proactive and real-time assistance in daily life. Existing AI services remain largely reactive, responding only to explicit user commands. We argue that a truly intelligent and helpful assistant should be capable of anticipating user needs and taking actions proactively when appropriate. To realize this vision, we propose Alpha-Service, a unified framework that addresses two fundamental challenges: Know When to intervene by detecting service opportunities from egocentric video streams, and Know How to provide both generalized and personalized services. Inspired by the von Neumann computer architecture and based on AI glasses, Alpha-Service consists of five key components: an Input Unit for perception, a Central Processing Unit for task scheduling, an Arithmetic Logic Unit for tool utilization, a Memory Unit for long-term personalization, and an Output Unit for natural human interaction. As an initial exploration, we implement Alpha-Service through a multi-agent system deployed on AI glasses. Case studies, including a real-time Blackjack advisor, a museum tour guide, and a shopping fit assistant, demonstrate its ability to seamlessly perceive the environment, infer user intent, and provide timely and useful assistance without explicit prompts. 

---
# Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies 

**Authors**: Mason Nakamura, Abhinav Kumar, Saaduddin Mahmud, Sahar Abdelnabi, Shlomo Zilberstein, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2510.14312)  

**Abstract**: A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems. 

---
# CAST: Compositional Analysis via Spectral Tracking for Understanding Transformer Layer Functions 

**Authors**: Zihao Fu, Ming Liao, Chris Russell, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.14262)  

**Abstract**: Large language models have achieved remarkable success but remain largely black boxes with poorly understood internal mechanisms. To address this limitation, many researchers have proposed various interpretability methods including mechanistic analysis, probing classifiers, and activation visualization, each providing valuable insights from different perspectives. Building upon this rich landscape of complementary approaches, we introduce CAST (Compositional Analysis via Spectral Tracking), a probe-free framework that contributes a novel perspective by analyzing transformer layer functions through direct transformation matrix estimation and comprehensive spectral analysis. CAST offers complementary insights to existing methods by estimating the realized transformation matrices for each layer using Moore-Penrose pseudoinverse and applying spectral analysis with six interpretable metrics characterizing layer behavior. Our analysis reveals distinct behaviors between encoder-only and decoder-only models, with decoder models exhibiting compression-expansion cycles while encoder models maintain consistent high-rank processing. Kernel analysis further demonstrates functional relationship patterns between layers, with CKA similarity matrices clearly partitioning layers into three phases: feature extraction, compression, and specialization. 

---
# Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models 

**Authors**: Mehrzad Samadi, Aleksander Ficek, Sean Narenthiran, Siddhartha Jain, Wasi Uddin Ahmad, Somshubra Majumdar, Vahid Noroozi, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2510.14232)  

**Abstract**: Competitive programming has become a rigorous benchmark for evaluating the reasoning and problem-solving capabilities of large language models (LLMs). The International Olympiad in Informatics (IOI) stands out as one of the most prestigious annual competitions in competitive programming and has become a key benchmark for comparing human and AI-level programming ability. While several proprietary models have been claimed to achieve gold medal-level performance at the IOI, often with undisclosed methods, achieving comparable results with open-weight models remains a significant challenge. In this paper, we present \gencluster, a scalable and reproducible test-time compute framework that attains IOI gold-level performance using open-weight models. It combines large-scale generation, behavioral clustering, ranking, and a round-robin submission strategy to efficiently explore diverse solution spaces under limited validation budgets. Our experiments show that the performance of our proposed approach scales consistently with available compute, narrowing the gap between open and closed systems. Notably, we will show that GenCluster can achieve a gold medal at IOI 2025 for the first time with an open-weight model gpt-oss-120b, setting a new benchmark for transparent and reproducible evaluation of reasoning in LLMs. 

---
# Joint Modeling of Big Five and HEXACO for Multimodal Apparent Personality-trait Recognition 

**Authors**: Ryo Masumura, Shota Orihashi, Mana Ihori, Tomohiro Tanaka, Naoki Makishima, Taiga Yamane, Naotaka Kawata, Satoshi Suzuki, Taichi Katayama  

**Link**: [PDF](https://arxiv.org/pdf/2510.14203)  

**Abstract**: This paper proposes a joint modeling method of the Big Five, which has long been studied, and HEXACO, which has recently attracted attention in psychology, for automatically recognizing apparent personality traits from multimodal human behavior. Most previous studies have used the Big Five for multimodal apparent personality-trait recognition. However, no study has focused on apparent HEXACO which can evaluate an Honesty-Humility trait related to displaced aggression and vengefulness, social-dominance orientation, etc. In addition, the relationships between the Big Five and HEXACO when modeled by machine learning have not been clarified. We expect awareness of multimodal human behavior to improve by considering these relationships. The key advance of our proposed method is to optimize jointly recognizing the Big Five and HEXACO. Experiments using a self-introduction video dataset demonstrate that the proposed method can effectively recognize the Big Five and HEXACO. 

---
# MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation 

**Authors**: Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14184)  

**Abstract**: We present MAFA (Multi-Agent Framework for Annotation), a production-deployed system that transforms enterprise-scale annotation workflows through configurable multi-agent collaboration. Addressing the critical challenge of annotation backlogs in financial services, where millions of customer utterances require accurate categorization, MAFA combines specialized agents with structured reasoning and a judge-based consensus mechanism. Our framework uniquely supports dynamic task adaptation, allowing organizations to define custom annotation types (FAQs, intents, entities, or domain-specific categories) through configuration rather than code changes. Deployed at JP Morgan Chase, MAFA has eliminated a 1 million utterance backlog while achieving, on average, 86% agreement with human annotators, annually saving over 5,000 hours of manual annotation work. The system processes utterances with annotation confidence classifications, which are typically 85% high, 10% medium, and 5% low across all datasets we tested. This enables human annotators to focus exclusively on ambiguous and low-coverage cases. We demonstrate MAFA's effectiveness across multiple datasets and languages, showing consistent improvements over traditional and single-agent annotation baselines: 13.8% higher Top-1 accuracy, 15.1% improvement in Top-5 accuracy, and 16.9% better F1 in our internal intent classification dataset and similar gains on public benchmarks. This work bridges the gap between theoretical multi-agent systems and practical enterprise deployment, providing a blueprint for organizations facing similar annotation challenges. 

---
# Towards Reversible Model Merging For Low-rank Weights 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.14163)  

**Abstract**: Model merging aims to combine multiple fine-tuned models into a single set of weights that performs well across all source tasks. While prior work has shown that merging can approximate the performance of individual fine-tuned models for each task, it largely overlooks scenarios where models are compressed into low-rank representations, either through low-rank adaptation (LoRA) or post-training singular value decomposition (SVD). We first demonstrate that applying conventional merging methods to low-rank weights leads to severe performance degradation in the merged model. Motivated by this phenomenon, we propose a fundamentally different approach: instead of collapsing all adapters into one set of weights, we construct a compact basis (e.g., an equivalent of holding two or more models) from which original task-specific models can be recovered via linear combination. This reframes merging as generating a reconstruction-capable model space rather than producing a single merged model. Crucially, this allows us to ``revert'' to each individual model when needed, recognizing that no merged model can consistently outperform one specialized for its task. Building on this insight, we introduce our method, Reversible Model Merging (RMM), an efficient, data-free, and flexible method that provides a closed-form solution for selecting the optimal basis of model weights and task-specific coefficients for linear combination. Extensive experiments across diverse datasets and model scales demonstrate that RMM consistently outperforms existing merging approaches, preserving the performance of low-rank compressed models by a significant margin. 

---
# Generating Fair Consensus Statements with Social Choice on Token-Level MDPs 

**Authors**: Carter Blair, Kate Larson  

**Link**: [PDF](https://arxiv.org/pdf/2510.14106)  

**Abstract**: Current frameworks for consensus statement generation with large language models lack the inherent structure needed to provide provable fairness guarantees when aggregating diverse free-form opinions. We model the task as a multi-objective, token-level Markov Decision Process (MDP), where each objective corresponds to an agent's preference. Token-level rewards for each agent are derived from their policy (e.g., a personalized language model). This approach utilizes the finding that such policies implicitly define optimal Q-functions, providing a principled way to quantify rewards at each generation step without a value function (Rafailov et al., 2024). This MDP formulation creates a formal structure amenable to analysis using principles from social choice theory. We propose two approaches grounded in social choice theory. First, we propose a stochastic generation policy guaranteed to be in the ex-ante core, extending core stability concepts from voting theory to text generation. This policy is derived from an underlying distribution over complete statements that maximizes proportional fairness (Nash Welfare). Second, for generating a single statement, we target the maximization of egalitarian welfare using search algorithms within the MDP framework. Empirically, experiments using language models to instantiate agent policies show that search guided by the egalitarian objective generates consensus statements with improved worst-case agent alignment compared to baseline methods, including the Habermas Machine (Tessler et al., 2024). 

---
# BitNet Distillation 

**Authors**: Xun Wu, Shaohan Huang, Wenhui Wang, Ting Song, Li Dong, Yan Xia, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.13998)  

**Abstract**: In this paper, we present BitNet Distillation (BitDistill), a lightweight pipeline that fine-tunes off-the-shelf full-precision LLMs (e.g., Qwen) into 1.58-bit precision (i.e., ternary weights {-1, 0, 1}) for specific downstream tasks, achieving strong task-specific performance with minimal computational cost. Specifically, BitDistill incorporates three key techniques: the SubLN module, as introduced in BitNet; multi-head attention distillation, based on MiniLM; and continual pre-training, which serves as a crucial warm-up step to mitigate the scalability issue of the performance gap between finetuned full-precision and 1.58-bit LLMs on specific tasks. Experimental results show that BitDistill achieves performance comparable to the full-precision counterpart models across model size, while enabling up to 10x memory savings and 2.65x faster inference on CPUs. Code is available at this https URL. 

---
# Do Slides Help? Multi-modal Context for Automatic Transcription of Conference Talks 

**Authors**: Supriti Sinhamahapatra, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2510.13979)  

**Abstract**: State-of-the-art (SOTA) Automatic Speech Recognition (ASR) systems primarily rely on acoustic information while disregarding additional multi-modal context. However, visual information are essential in disambiguation and adaptation. While most work focus on speaker images to handle noise conditions, this work also focuses on integrating presentation slides for the use cases of scientific presentation.
In a first step, we create a benchmark for multi-modal presentation including an automatic analysis of transcribing domain-specific terminology. Next, we explore methods for augmenting speech models with multi-modal information. We mitigate the lack of datasets with accompanying slides by a suitable approach of data augmentation. Finally, we train a model using the augmented dataset, resulting in a relative reduction in word error rate of approximately 34%, across all words and 35%, for domain-specific terms compared to the baseline model. 

---
# LTR-ICD: A Learning-to-Rank Approach for Automatic ICD Coding 

**Authors**: Mohammad Mansoori, Amira Soliman, Farzaneh Etminani  

**Link**: [PDF](https://arxiv.org/pdf/2510.13922)  

**Abstract**: Clinical notes contain unstructured text provided by clinicians during patient encounters. These notes are usually accompanied by a sequence of diagnostic codes following the International Classification of Diseases (ICD). Correctly assigning and ordering ICD codes are essential for medical diagnosis and reimbursement. However, automating this task remains challenging. State-of-the-art methods treated this problem as a classification task, leading to ignoring the order of ICD codes that is essential for different purposes. In this work, as a first attempt, we approach this task from a retrieval system perspective to consider the order of codes, thus formulating this problem as a classification and ranking task. Our results and analysis show that the proposed framework has a superior ability to identify high-priority codes compared to other methods. For instance, our model accuracy in correctly ranking primary diagnosis codes is 47%, compared to 20% for the state-of-the-art classifier. Additionally, in terms of classification metrics, the proposed model achieves a micro- and macro-F1 scores of 0.6065 and 0.2904, respectively, surpassing the previous best model with scores of 0.597 and 0.2660. 

---
# Generative AI in Heritage Practice: Improving the Accessibility of Heritage Guidance 

**Authors**: Jessica Witte, Edmund Lee, Lisa Brausem, Verity Shillabeer, Chiara Bonacchi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13811)  

**Abstract**: This paper discusses the potential for integrating Generative Artificial Intelligence (GenAI) into professional heritage practice with the aim of enhancing the accessibility of public-facing guidance documents. We developed HAZEL, a GenAI chatbot fine-tuned to assist with revising written guidance relating to heritage conservation and interpretation. Using quantitative assessments, we compare HAZEL's performance to that of ChatGPT (GPT-4) in a series of tasks related to the guidance writing process. The results of this comparison indicate a slightly better performance of HAZEL over ChatGPT, suggesting that the GenAI chatbot is more effective once the underlying large language model (LLM) has been fine-tuned. However, we also note significant limitations, particularly in areas requiring cultural sensitivity and more advanced technical expertise. These findings suggest that, while GenAI cannot replace human heritage professionals in technical authoring tasks, its potential to automate and expedite certain aspects of guidance writing could offer valuable benefits to heritage organisations, especially in resource-constrained contexts. 

---
