# MENLO: From Preferences to Proficiency - Evaluating and Modeling Native-like Quality Across 47 Languages 

**Authors**: Chenxi Whitehouse, Sebastian Ruder, Tony Lin, Oksana Kurylo, Haruka Takagi, Janice Lam, Nicolò Busetto, Denise Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.26601)  

**Abstract**: Ensuring native-like quality of large language model (LLM) responses across many languages is challenging. To address this, we introduce MENLO, a framework that operationalizes the evaluation of native-like response quality based on audience design-inspired mechanisms. Using MENLO, we create a dataset of 6,423 human-annotated prompt-response preference pairs covering four quality dimensions with high inter-annotator agreement in 47 language varieties. Our evaluation reveals that zero-shot LLM judges benefit significantly from pairwise evaluation and our structured annotation rubrics, yet they still underperform human annotators on our dataset. We demonstrate substantial improvements through fine-tuning with reinforcement learning, reward shaping, and multi-task learning approaches. Additionally, we show that RL-trained judges can serve as generative reward models to enhance LLMs' multilingual proficiency, though discrepancies with human judgment remain. Our findings suggest promising directions for scalable multilingual evaluation and preference alignment. We release our dataset and evaluation framework to support further research in multilingual LLM evaluation. 

---
# Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning 

**Authors**: Jinyeop Song, Song Wang, Julian Shun, Yada Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.26383)  

**Abstract**: Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucinations and expose reasoning traces. However, many KG-RAG systems compose multiple LLM modules (e.g planning, reasoning, and responding), inflating inference cost and binding behavior to a specific target KG. To address this, we introduce KG-R1, an agentic KG retrieval-augmented generation (KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single agent that interacts with KGs as its environment, learning to retrieve at each step and incorporating the retrieved information into its reasoning and generation. The process is optimized through end-to-end RL. In controlled experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our method demonstrates both efficiency and transferability: Using Qwen-2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use larger foundation or fine-tuned models. Furthermore, KG-R1 enables plug and play: after training, it maintains strong accuracy on new KGs without modification. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at this https URL. 

---
# Latent Thinking Optimization: Your Latent Reasoning Language Model Secretly Encodes Reward Signals in its Latent Thoughts 

**Authors**: Hanwen Du, Yuxin Dong, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2509.26314)  

**Abstract**: Large Language Models (LLMs) excel at problem solving by generating chain of thoughts in natural language, but such verbal thinking is computationally costly and prone to overthinking. Recent work instead proposes a latent thinking architecture Huggin-3.5B, which represents intermediate reasoning steps as sequence of latent representations. However, latent thoughts lack interpretability and are difficult to supervise, raising concerns about the correctness and reliability of its latent thinking processes. In this paper, we provide a systematic study of how Huggin-3.5B thinks in the latent space and how external supervision signals can improve its latent thinking processes. We show that latent thoughts leading to correct versus incorrect answers exhibit highly distinguishable patterns, and that a latent classifier can reliably predict answer correctness directly from latent thoughts. Leveraging these insights, we propose Latent Thinking Optimization (LTO), a probabilistic algorithm that employs the latent classifier as a Latent Reward Model (LRM) to optimize the latent thinking processes. Extensive experiments across diverse reasoning tasks demonstrate that LRM is highly effective in detecting incorrect latent thinking patterns, and LTO can significantly improve the latent thinking processes. Furthermore, we show that LRM can generalize across diverse domains, and LTO can be seamlessly applied to general LLMs to improve their thinking processes. In contrast to verbal thinking, our method demonstrates that reward modeling and scaling test-time thinking with supervision can be performed directly in the latent space, highlighting its potential as a general, efficient, and domain-agnostic approach to improving the thinking processes of LLMs. 

---
# Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis 

**Authors**: Leitian Tao, Xuefeng Du, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.26074)  

**Abstract**: Reward modeling, crucial for aligning large language models (LLMs) with human preferences, is often bottlenecked by the high cost of preference data. Existing textual data synthesis methods are computationally expensive. We propose a novel framework LENS for synthesizing preference data directly in the LLM's latent embedding space. Our method employs a Variational Autoencoder (VAE) to learn a structured latent representation of response embeddings. By performing controlled perturbations in this latent space and decoding back to the embedding space, we efficiently generate diverse, semantically consistent synthetic preference pairs, bypassing costly text generation and annotation. We provide theoretical guarantees that our synthesized pairs approximately preserve original preference ordering and improve reward model generalization. Empirically, our latent-space synthesis significantly outperforms text-based augmentation on standard benchmarks, achieving superior results while being 18x faster in generation and using a 16,000x smaller model. Our work offers a scalable and effective alternative for enhancing reward modeling through efficient data augmentation. Code is publicly available at this https URL 

---
# RAGferee: Building Contextual Reward Models for Retrieval-Augmented Generation 

**Authors**: Andrei C. Coman, Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Bill Byrne, James Henderson, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2509.26011)  

**Abstract**: Existing Reward Models (RMs), typically trained on general preference data, struggle in Retrieval Augmented Generation (RAG) settings, which require judging responses for faithfulness to retrieved context, relevance to the user query, appropriate refusals when context is insufficient, completeness and conciseness of information. To address the lack of publicly available RAG-centric preference datasets and specialised RMs, we introduce RAGferee, a methodology that repurposes question-answering (QA) datasets into preference pairs that prioritise groundedness over stylistic features, enabling the training of contextual RMs better suited to judging RAG responses. Using RAGferee, we curate a small preference dataset of 4K samples and fine-tune RMs ranging from 7B to 24B parameters. Our RAG-centric RMs achieve state-of-the-art performance on ContextualJudgeBench, surpassing existing 70B+ RMs trained on much larger (up to 2.4M samples) general corpora, with an absolute improvement of +15.5%. 

---
# Mem-α: Learning Memory Construction via Reinforcement Learning 

**Authors**: Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25911)  

**Abstract**: Large language model (LLM) agents are constrained by limited context windows, necessitating external memory systems for long-term information understanding. Current memory-augmented agents typically depend on pre-defined instructions and tools for memory updates. However, language models may lack the ability to determine which information to store, how to structure it, and when to update it, especially as memory systems become more complex. This results in suboptimal memory construction and information loss. To this end, we propose Mem-alpha, a reinforcement learning framework that trains agents to effectively manage complex memory systems through interaction and feedback. We also construct a specialized training dataset spanning diverse multi-turn interaction patterns paired with comprehensive evaluation questions designed to teach effective memory management. During training, agents process sequential information chunks, learn to extract and store relevant content, then update the memory system. The reward signal derives from downstream question-answering accuracy over the full interaction history, directly optimizing for memory construction. To illustrate the effectiveness of our training framework, we design a memory architecture comprising core, episodic, and semantic components, equipped with multiple tools for memory operations. Empirical evaluation demonstrates that Mem-alpha achieves significant improvements over existing memory-augmented agent baselines. Despite being trained exclusively on instances with a maximum length of 30k tokens, our agents exhibit remarkable generalization to sequences exceeding 400k tokens, over 13x the training length, highlighting the robustness of Mem-alpha. 

---
# Unspoken Hints: Accuracy Without Acknowledgement in LLM Reasoning 

**Authors**: Arash Marioriyad, Shaygan Adim, Nima Alighardashi, Mahdieh Soleymani Banghshah, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2509.26041)  

**Abstract**: Large language models (LLMs) increasingly rely on chain-of-thought (CoT) prompting to solve mathematical and logical reasoning tasks. Yet, a central question remains: to what extent are these generated rationales \emph{faithful} to the underlying computations, rather than post-hoc narratives shaped by hints that function as answer shortcuts embedded in the prompt? Following prior work on hinted vs.\ unhinted prompting, we present a systematic study of CoT faithfulness under controlled hint manipulations. Our experimental design spans four datasets (AIME, GSM-Hard, MATH-500, UniADILR), two state-of-the-art models (GPT-4o and Gemini-2-Flash), and a structured set of hint conditions varying in correctness (correct and incorrect), presentation style (sycophancy and data leak), and complexity (raw answers, two-operator expressions, four-operator expressions). We evaluate both task accuracy and whether hints are explicitly acknowledged in the reasoning. Our results reveal three key findings. First, correct hints substantially improve accuracy, especially on harder benchmarks and logical reasoning, while incorrect hints sharply reduce accuracy in tasks with lower baseline competence. Second, acknowledgement of hints is highly uneven: equation-based hints are frequently referenced, whereas raw hints are often adopted silently, indicating that more complex hints push models toward verbalizing their reliance in the reasoning process. Third, presentation style matters: sycophancy prompts encourage overt acknowledgement, while leak-style prompts increase accuracy but promote hidden reliance. This may reflect RLHF-related effects, as sycophancy exploits the human-pleasing side and data leak triggers the self-censoring side. Together, these results demonstrate that LLM reasoning is systematically shaped by shortcuts in ways that obscure faithfulness. 

---
# TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning 

**Authors**: Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Sean Chen, Mohammad Kachuee, Teja Gollapudi, Tony Liao, Nicolas Scheffer, Rakesh Wanga, Anuj Kumar, Yu Meng, Wen-tau Yih, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.25760)  

**Abstract**: While large language models (LLMs) have demonstrated strong performance on factoid question answering, they are still prone to hallucination and untruthful responses, particularly when tasks demand information outside their parametric knowledge. Indeed, truthfulness requires more than accuracy -- models must also recognize uncertainty and abstain when unsure to avoid hallucinations. This presents a fundamental challenge for existing methods: approaches that optimize for accuracy often amplify hallucinations, while those that encourage abstention can become overly conservative, sacrificing correct answers. Both extremes ultimately compromise truthfulness. In this work, we present TruthRL, a general reinforcement learning (RL) framework that directly optimizes the truthfulness of LLMs. Specifically, we implement TruthRL using GRPO with a simple yet effective ternary reward that distinguishes correct answers, hallucinations, and abstentions. It incentivizes models to reduce hallucinations not only by providing correct responses, but also by enabling abstention when uncertain, thereby improving truthfulness. Extensive experiments across four knowledge-intensive benchmarks show that, compared to vanilla RL, TruthRL significantly reduces hallucinations by 28.9% and improves truthfulness by 21.1%, with consistent gains across various backbone models (e.g., Qwen, Llama) under both retrieval and non-retrieval setups. In-depth ablation study demonstrates that vanilla accuracy-driven methods, such as supervised fine-tuning or RL with a binary reward, struggle to balance factual correctness and uncertainty. In contrast, our proposed truthfulness-driven TruthRL achieves strong performance in both accuracy and truthfulness, underscoring the importance of learning objective design for developing truthful LLMs. 

---
# Reinforced Strategy Optimization for Conversational Recommender Systems via Network-of-Experts 

**Authors**: Xiaoyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.26093)  

**Abstract**: Conversational Recommender Systems (CRSs) provide personalized recommendations through multi-turn interactions. With the strong reasoning abilities of Large Language Models (LLMs), applying them to CRSs has become promising. Yet, existing methods often lack explicit optimization of interaction strategies, relying instead on unified prompts, which can yield suboptimal outcomes. We propose Reinforced Strategy Optimization (RSO), a hierarchical framework that decomposes response generation into macro-level strategy planning and micro-level adaptation within a network-of-experts. A Planner selects strategies (e.g., recommend, explain, encourage), while an Actor generates responses guided by auxiliary experts for preferences and factual grounding. This disentanglement enables more tractable learning. To address limited multi-turn data, we model strategy learning as reinforcement learning with an LLM-based reward for exploration. Experiments show RSO outperforms state-of-the-art baselines, validating the effectiveness of hierarchical strategy optimization. 

---
# Self-Rewarding Rubric-Based Reinforcement Learning for Open-Ended Reasoning 

**Authors**: Zhiling Ye, Yun Yue, Haowen Wang, Xudong Han, Jiadi Jiang, Cheng Wei, Lei Fan, Jiaxin Liang, Shuowen Zhang, Ji Li, Chunxiao Guo, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25534)  

**Abstract**: Open-ended evaluation is essential for deploying large language models in real-world settings. In studying HealthBench, we observe that using the model itself as a grader and generating rubric-based reward signals substantially improves reasoning performance. Remarkably, the trained model also becomes a stronger grader. Motivated by this, we introduce Self-Rewarding Rubric-Based Reinforcement Learning for Open-Ended Reasoning, a lightweight framework that enables faster and more resource-efficient training while surpassing baselines. Remarkably, on Qwen3-32B, training with just the 4000-sample HealthBench Easy subset is sufficient to obtain a model that exceeds GPT-5 on HealthBench Hard. Incorporating a small amount of teacher-graded data further enhances performance for less capable models. 

---
# RFG: Test-Time Scaling for Diffusion Large Language Model Reasoning with Reward-Free Guidance 

**Authors**: Tianlang Chen, Minkai Xu, Jure Leskovec, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2509.25604)  

**Abstract**: Diffusion large language models (dLLMs) have shown great potential in large-scale language modeling, and there is an increasing interest in further improving the capacity to solve complex problems by guiding the reasoning process step by step. Common practice for autoregressive language models typically learns a process reward model with dense annotation for each intermediate step. However, this is challenging for dLLMs where the generation is in an any-order fashion and intermediate states are partially masked sentences. To this end, in this paper, we propose reward-free guidance (RFG), a principled method for guiding the reasoning trajectory of dLLMs without explicit process reward. The key idea of RFG is to parameterize the process reward by log-likelihood ratios of the enhanced and reference dLLMs, where the enhanced model can be easily obtained by any off-the-shelf dLLM that has been post-trained with reinforcement learning (RL) or supervised fine-tuning (SFT). We provide theoretical justification that RFG induces the reward-guided sampling distribution with no additional reward. We conduct comprehensive experiments on four challenging mathematical reasoning and code generation benchmarks using a diverse suite of dLLMs enhanced with various post-training methods. RFG consistently yields significant improvements across all tasks and model types, achieving accuracy gains of up to 9.2%. These findings establish RFG as a general training-free framework that scales test-time reasoning without reliance on external reward models. 

---
# From Faithfulness to Correctness: Generative Reward Models that Think Critically 

**Authors**: Qiyao Ma, Yunsheng Shi, Hongtao Tian, Chao Wang, Weiming Chang, Ting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25409)  

**Abstract**: Through reinforcement learning with verifiable rewards (RLVR), large language models have achieved substantial progress in domains with easily verifiable outcomes, such as mathematics and coding. However, when applied to more complex tasks like open-domain question answering, RLVR faces significant challenges due to the difficulty of verifying correctness. The nuanced and ambiguous nature of real-world knowledge makes it difficult to reliably evaluate correctness in these settings, necessitating further abilities that extend beyond mere logical consistency to encompass an understanding and assessment of both external and internal knowledge. Recent work has primarily focused on improving faithfulness, defined as semantic alignment with supporting documents, which can cause models to rely excessively on external sources and diminish their capacity for critical assessment. To address this, we propose the Thinking-supervised Reward Model (TRM), which incorporates sentence-level thinking supervision to endow reward models with critical thinking abilities. Given a query, answer, and supporting documents, TRM first assesses the faithfulness of each answer sentence to the supporting documents, and then applies a reasoning step to evaluate sentence-level correctness. By structuring reward modeling as a sequence of faithfulness, reasoning, and correctness evaluations, TRM encourages models to critically assess and leverage both external and internal knowledge. Experiments on reward signals demonstrate that TRM substantially improves the identification of incorrect sentences, and incorporating TRM into policy optimization leads to significant gains in both answer correctness and usefulness. 

---
# Aligning Multilingual Reasoning with Verifiable Semantics from a High-Resource Expert Model 

**Authors**: Fahim Faisal, Kaiqiang Song, Song Wang, Simin Ma, Shujian Liu, Haoyun Deng, Sathish Reddy Indurthi  

**Link**: [PDF](https://arxiv.org/pdf/2509.25543)  

**Abstract**: While reinforcement learning has advanced the reasoning abilities of Large Language Models (LLMs), these gains are largely confined to English, creating a significant performance disparity across languages. To address this, we introduce Pivot-Based Reinforcement Learning with Semantically Verifiable Rewards (PB-RLSVR), a novel framework that enhances multilingual reasoning by circumventing the need for human-annotated data in target languages. Our approach employs a high-performing English LLM as a "pivot" model to generate reference responses for reasoning tasks. A multilingual model is then rewarded based on the semantic equivalence of its responses to the English reference, effectively transferring the pivot model's reasoning capabilities across languages. We investigate several cross-lingual semantic reward functions, including those based on embeddings and machine translation. Extensive experiments on a suite of multilingual reasoning benchmarks show that our method significantly narrows the performance gap between English and other languages, substantially outperforming traditional PPO baselines. Specifically, our PB-RLSVR framework improves the average multilingual performance of Llama-3.1-8B-Instruct and Qwen3-32B by 16.41% and 10.17%, respectively, demonstrating a powerful and data-efficient approach to building truly multilingual reasoning agents. 

---
# Attention as a Compass: Efficient Exploration for Process-Supervised RL in Reasoning Models 

**Authors**: Runze Liu, Jiakang Wang, Yuling Shi, Zhihui Xie, Chenxin An, Kaiyan Zhang, Jian Zhao, Xiaodong Gu, Lei Lin, Wenping Hu, Xiu Li, Fuzheng Zhang, Guorui Zhou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2509.26628)  

**Abstract**: Reinforcement Learning (RL) has shown remarkable success in enhancing the reasoning capabilities of Large Language Models (LLMs). Process-Supervised RL (PSRL) has emerged as a more effective paradigm compared to outcome-based RL. However, existing PSRL approaches suffer from limited exploration efficiency, both in terms of branching positions and sampling. In this paper, we introduce a novel PSRL framework (AttnRL), which enables efficient exploration for reasoning models. Motivated by preliminary observations that steps exhibiting high attention scores correlate with reasoning behaviors, we propose to branch from positions with high values. Furthermore, we develop an adaptive sampling strategy that accounts for problem difficulty and historical batch size, ensuring that the whole training batch maintains non-zero advantage values. To further improve sampling efficiency, we design a one-step off-policy training pipeline for PSRL. Extensive experiments on multiple challenging mathematical reasoning benchmarks demonstrate that our method consistently outperforms prior approaches in terms of performance and sampling and training efficiency. 

---
# RoRecomp: Enhancing Reasoning Efficiency via Rollout Response Recomposition in Reinforcement Learning 

**Authors**: Gang Li, Yulei Qin, Xiaoyu Tan, Dingkang Yang, Yuchen Shi, Zihan Xu, Xiang Li, Xing Sun, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.25958)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has proven effective in eliciting complex reasoning in large language models (LLMs). However, standard RLVR training often leads to excessively verbose processes (in reasoning tasks) and inefficient exploration trajectories (in agentic settings), as outcome-only rewards provide no incentive for efficiency and the high variance in response length within relatively small rollout groups results in noisy optimization signals. To address this, we propose Rollout Response Recomposition (RoRecomp), a plug-and-play method that guides models toward concise reasoning by strategically recomposing the training data. RoRecomp separates responses into two distinct batch types: 1) priority batches, which combine short-correct and long-incorrect responses selected from online batches to provide a clear gradient signal for brevity, and 2) compensation batches, which utilize remaining responses from a replay buffer to maintain stability and prevent model collapse. To comprehensively evaluate effectiveness, we test RoRecomp across three settings where results demonstrate substantial efficiency gains: reducing reasoning length by 27.7% in zero RL training, reducing unnecessary tool calls by 46.8% while improving accuracy in agentic RL, and achieving up to 52.5% length reduction in thinking compression, all with minimal performance impact. 

---
# Boosting Process-Correct CoT Reasoning by Modeling Solvability of Multiple-Choice QA 

**Authors**: Raphael Schumann, Stefan Riezler  

**Link**: [PDF](https://arxiv.org/pdf/2509.25941)  

**Abstract**: Reasoning quality in large language models depends not only on producing correct answers but also on generating valid intermediate steps. We study this through multiple-choice question answering (MCQA), which provides a controlled setting with fixed answer options. Our analysis shows that when questions are effectively unsolvable for a model, spurious chains of thought (CoTs) are more likely to appear, leading to false positives. By estimating the solvability of each question, we uncover an intermediate regime where learning is most effective. Building on this insight, we adapt outcome-supervised reward models and reinforcement learning with group-relative advantage to incorporate solvability into their objectives. Across experiments on math and multimodal datasets, these modifications consistently yield higher rates of process-correct reasoning and, in reinforcement learning, improved answer accuracy as well. Our results highlight solvability as a key factor for reducing hallucinations and increasing reliability in CoT reasoning. 

---
# Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation 

**Authors**: Ziniu Li, Congliang Chen, Tianyun Yang, Tian Ding, Ruoyu Sun, Ge Zhang, Wenhao Huang, Zhi-Quan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25849)  

**Abstract**: Large Language Models (LLMs) can self-improve through reinforcement learning, where they generate trajectories to explore and discover better solutions. However, this exploration process is computationally expensive, often forcing current methods to assign limited exploration budgets to each task. This uniform allocation creates problematic edge cases: easy tasks consistently succeed while difficult tasks consistently fail, both producing zero gradients during training updates for the widely used Group Relative Policy Optimization (GRPO). We address this problem from the lens of exploration budget allocation. Viewing each task's exploration as an "item" with a distinct "value" and "cost", we establish a connection to the classical knapsack problem. This formulation allows us to derive an optimal assignment rule that adaptively distributes resources based on the model's current learning status. When applied to GRPO, our method increases the effective ratio of non-zero policy gradients by 20-40% during training. Acting as a computational "free lunch", our approach could reallocate exploration budgets from tasks where learning is saturated to those where it is most impactful. This enables significantly larger budgets (e.g., 93 rollouts) for especially challenging problems, which would be computationally prohibitive under a uniform allocation. These improvements translate to meaningful gains on mathematical reasoning benchmarks, with average improvements of 2-4 points and peak gains of 9 points on specific tasks. Notably, achieving comparable performance with traditional homogeneous allocation would require about 2x the computational resources. 

---
# Learning to Reason as Action Abstractions with Scalable Mid-Training RL 

**Authors**: Shenao Zhang, Donghan Yu, Yihao Feng, Bowen Jin, Zhaoran Wang, John Peebles, Zirui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25810)  

**Abstract**: Large language models excel with reinforcement learning (RL), but fully unlocking this potential requires a mid-training stage. An effective mid-training phase should identify a compact set of useful actions and enable fast selection among them through online RL. We formalize this intuition by presenting the first theoretical result on how mid-training shapes post-training: it characterizes an action subspace that minimizes both the value approximation error from pruning and the RL error during subsequent planning. Our analysis reveals two key determinants of mid-training effectiveness: pruning efficiency, which shapes the prior of the initial RL policy, and its impact on RL convergence, which governs the extent to which that policy can be improved via online interactions. These results suggest that mid-training is most effective when the decision space is compact and the effective horizon is short, highlighting the importance of operating in the space of action abstractions rather than primitive actions. Building on these insights, we propose Reasoning as Action Abstractions (RA3), a scalable mid-training algorithm. Specifically, we derive a sequential variational lower bound and optimize it by iteratively discovering temporally-consistent latent structures via RL, followed by fine-tuning on the bootstrapped data. Experiments on code generation tasks demonstrate the effectiveness of our approach. Across multiple base models, RA3 improves the average performance on HumanEval and MBPP by 8 and 4 points over the base model and the next-token prediction baseline. Furthermore, RA3 achieves faster convergence and higher asymptotic performance in RLVR on HumanEval+, MBPP+, LiveCodeBench, and Codeforces. 

---
# EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing 

**Authors**: Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.26346)  

**Abstract**: Recently, we have witnessed great progress in image editing with natural language instructions. Several closed-source models like GPT-Image-1, Seedream, and Google-Nano-Banana have shown highly promising progress. However, the open-source models are still lagging. The main bottleneck is the lack of a reliable reward model to scale up high-quality synthetic training data. To address this critical bottleneck, we built \mname, trained with our new large-scale human preference dataset, meticulously annotated by trained experts following a rigorous protocol containing over 200K preference pairs. \mname demonstrates superior alignment with human preferences in instruction-guided image editing tasks. Experiments show that \mname achieves state-of-the-art human correlation on established benchmarks such as GenAI-Bench, AURORA-Bench, ImagenHub, and our new \benchname, outperforming a wide range of VLM-as-judge models. Furthermore, we use \mname to select a high-quality subset from the existing noisy ShareGPT-4o-Image dataset. We train Step1X-Edit on the selected subset, which shows significant improvement over training on the full set. This demonstrates \mname's ability to serve as a reward model to scale up high-quality training data for image editing. Furthermore, its strong alignment suggests potential for advanced applications like reinforcement learning-based post-training and test-time scaling of image editing models. \mname with its training dataset will be released to help the community build more high-quality image editing training datasets. 

---
# Importance Sampling for Multi-Negative Multimodal Direct Preference Optimization 

**Authors**: Xintong Li, Chuhan Wang, Junda Wu, Rohan Surana, Tong Yu, Julian McAuley, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25717)  

**Abstract**: Direct Preference Optimization (DPO) has recently been extended from text-only models to vision-language models. However, existing methods rely on oversimplified pairwise comparisons, generating a single negative image via basic perturbations or similarity-based retrieval, which fail to capture the complex nature of multimodal preferences, inducing optimization bias and hallucinations. To address this issue, we propose MISP-DPO, the first framework to incorporate multiple, semantically diverse negative images in multimodal DPO via the Plackett-Luce model. Our method embeds prompts and candidate images in CLIP (Contrastive Language-Image Pretraining) space and applies a sparse autoencoder to uncover semantic deviations into interpretable factors. Negative samples are selected based on reconstruction difficulty, semantic deviation from the positive, and mutual diversity, yielding broader and more informative supervision. To handle multi-negative comparisons, we adopt a Plackett-Luce objective and introduce an importance sampling strategy that improves training efficiency. Experiments across five diverse benchmarks demonstrate that MISP-DPO consistently improves multimodal alignment over prior methods, validating the effectiveness of semantic-aware, multi-negative sampling in preference-based learning. 

---
# Nudging the Boundaries of LLM Reasoning 

**Authors**: Justin Chih-Yao Chen, Becky Xiangyu Peng, Prafulla Kumar Choubey, Kung-Hsiang Huang, Jiaxin Zhang, Mohit Bansal, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25666)  

**Abstract**: Current online reinforcement learning (RL) algorithms like GRPO share a key limitation in LLM reasoning: they cannot learn from problems that are "unsolvable" to the model. In other words, they can only improve performance on problems where the model is capable of exploring the correct answer. Consequently, the model's "upper limit" remains unchanged after RL training, even though the likelihood of solving easier, solvable problems may increase. These hard samples cannot contribute to training, as no rollouts yield rewards and thus no gradients are produced. To unlock learning from these hard samples, we propose NuRL, a "nudging" method that aims to push the upper bound of LLM reasoning using self-generated hints, i.e., abstract cues that help reduce the problem difficulty for the model. Given a question and its gold answer, the model generates a CoT and then produces a hint containing the core knowledge needed to solve the problem. During training, we generate G rollouts from the base policy and use the pass rate to decide whether the hint should be injected. For hard samples with a 0% pass rate, we inject the hint and regenerate a new batch of trajectories. This yields two benefits: (1) the hint boosts pass rates (from 0% to non-zero), thereby introducing training signals for previously unsolvable samples, and (2) the hints are self-generated, avoiding distributional shift and do not rely on external models. NuRL achieves consistent improvements across 6 benchmarks and 3 models, while remaining complementary to test-time scaling. Notably, NuRL can raise the model's upper limit, whereas GRPO leaves pass@1024 unchanged from the base model. Furthermore, we present a systematic study of what makes an effective hint and when hints are most useful. Interestingly, the best hints are abstract and high-level, and are most beneficial when applied necessarily and after GRPO has converged. 

---
# Dynamic Policy Induction for Adaptive Prompt Optimization: Bridging the Efficiency-Accuracy Gap via Lightweight Reinforcement Learning 

**Authors**: Jiexi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25267)  

**Abstract**: The performance of Large Language Models (LLMs) depends heavily on the chosen prompting strategy, yet static approaches such as Zero-Shot, Few-Shot, or Chain-of-Thought (CoT) impose a rigid efficiency-accuracy trade-off. Highly accurate strategies like Self-Consistency (SC) incur substantial computational waste on simple tasks, while lightweight methods often fail on complex inputs. This paper introduces the Prompt Policy Network (PPN), a lightweight reinforcement learning framework that formalizes adaptive strategy selection as a single-step Markov Decision Process (MDP). The PPN, trained with Proximal Policy Optimization (PPO) and guided by a resource-explicit reward function, learns to allocate costly reasoning strategies only when necessary. Experiments on arithmetic reasoning benchmarks demonstrate that PPN achieves superior performance on the efficiency-accuracy Pareto front, delivering up to 61.5% token cost reduction compared to Self-Consistency while maintaining competitive accuracy. This work contributes a systematic, adaptive framework for cost-efficient LLM deployment, advancing the design of lightweight optimization techniques for scalable and sustainable language model applications. 

---
# Adaptive Test-Time Reasoning via Reward-Guided Dual-Phase Search 

**Authors**: Yingqian Cui, Zhenwei Dai, Pengfei He, Bing He, Hui Liu, Xianfeng Tang, Jingying Zeng, Suhang Wang, Yue Xing, Jiliang Tang, Benoit Dumoulin  

**Link**: [PDF](https://arxiv.org/pdf/2509.25420)  

**Abstract**: Large Language Models (LLMs) have achieved significant advances in reasoning tasks. A key approach is tree-based search with verifiers, which expand candidate reasoning paths and use reward models to guide pruning and selection. Although effective in improving accuracy, these methods are not optimal in terms of efficiency: they perform simple decomposition on the reasoning process, but ignore the planning-execution nature of tasks such as math reasoning or code generation. This results in inefficient exploration of reasoning process. To address this, we propose a dual-phase test-time scaling framework that explicitly separates reasoning into planning and execution, and performs search over the two phases individually. Specifically, we decompose reasoning trajectories and develop reward models for each phase, enabling the search to explore and prune plans and executions separately. We further introduce a dynamic budget allocation mechanism that adaptively redistributes sampling effort based on reward feedback, allowing early stopping on confident steps and reallocation of computation to more challenging parts of the reasoning process. Experiments on both mathematical reasoning and code generation benchmarks demonstrate that our approach consistently improves accuracy while reducing redundant computation. 

---
# LLM Agents for Knowledge Discovery in Atomic Layer Processing 

**Authors**: Andreas Werbrouck, Marshall B. Lindsay, Matthew Maschmann, Matthias J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.26201)  

**Abstract**: Large Language Models (LLMs) have garnered significant attention for several years now. Recently, their use as independently reasoning agents has been proposed. In this work, we test the potential of such agents for knowledge discovery in materials science. We repurpose LangGraph's tool functionality to supply agents with a black box function to interrogate. In contrast to process optimization or performing specific, user-defined tasks, knowledge discovery consists of freely exploring the system, posing and verifying statements about the behavior of this black box, with the sole objective of generating and verifying generalizable statements. We provide proof of concept for this approach through a children's parlor game, demonstrating the role of trial-and-error and persistence in knowledge discovery, and the strong path-dependence of results. We then apply the same strategy to show that LLM agents can explore, discover, and exploit diverse chemical interactions in an advanced Atomic Layer Processing reactor simulation using intentionally limited probe capabilities without explicit instructions. 

---
# Interactive Learning for LLM Reasoning 

**Authors**: Hehai Lin, Shilei Cao, Minzhi Li, Sudong Wang, Haotian Wu, Linyi Yang, Juepeng Zheng, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.26306)  

**Abstract**: Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3 (Idea Sharing, Idea Analysis, and Idea Fusion), an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We validate ILR on three LLMs across two model families of varying scales, evaluating performance on five mathematical benchmarks and one coding benchmark. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies. 

---
# Diversity-Incentivized Exploration for Versatile Reasoning 

**Authors**: Zican Hu, Shilin Zhang, Yafu Li, Jianhao Yan, Xuyang Hu, Leyang Cui, Xiaoye Qu, Chunlin Chen, Yu Cheng, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.26209)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a crucial paradigm for incentivizing reasoning capabilities in Large Language Models (LLMs). Due to vast state-action spaces and reward sparsity in reasoning tasks, existing methods often struggle with deficient exploration and poor sample efficiency. In the paper, we propose \textbf{DIVER} (\textbf{D}iversity-\textbf{I}ncentivized Exploration for \textbf{V}ersatil\textbf{E} \textbf{R}easoning), an innovative framework that highlights the pivotal role of global sequence-level diversity to incentivize deep exploration for versatile reasoning. We first conduct a primary empirical study to reveal a strong positive correlation between global diversity and reasoning capacity. Building on this insight, we introduce global diversity incentives as an intrinsic reward to promote deep exploration in a semantically structured space. Incorporating the intrinsic reward, we develop a potential-based reward shaping mechanism to preserve optimal policy invariance and design simple heuristics to mitigate possible reward hacking. Experimental results show that DIVER outperforms competitive RLVR baselines with various exploration strategies on both in-domain and out-of-domain tasks, excelling in both Pass@1 and Pass@k evaluations. Our code is available at this https URL. 

---
# Structural Reward Model: Enhancing Interpretability, Efficiency, and Scalability in Reward Modeling 

**Authors**: Xiaoyu Liu, Di Liang, Hongyu Shan, Peiyang Liu, Yonghao Liu, Muling Wu, Yuntao Li, Xianjie Wu, LI Miao, Jiangrong Shen, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25361)  

**Abstract**: Reward Models (RMs) are key components for evaluating and guiding language model outputs. However, traditional scalar RMs often struggle with incorporating contextual and background information during inference, leading to incomplete evaluations. Generative RMs (GRMs) attempt to address these limitations by generating intermediate reasoning steps. Yet, their uncontrolled black-box nature and inefficiency due to sequential decoding hinder their industrial deployment. Industrial scenarios, such as search and recommendation systems, often involve single-domain tasks requiring evaluation along specific dimensions. In such contexts, diagnosing "bad cases" necessitates structured feedback to identify and optimize dimension-specific issues. In this paper, we propose the Structural Reward Model (SRM), a modular and interpretable framework integrating side-branch models as auxiliary feature generators. By introducing fine-grained dimensions, SRMs enable interpretable and efficient evaluation, facilitating targeted diagnostics and optimization. This structured approach ensures adaptability and scalability for industrial applications. Through comprehensive experiments, we demonstrate that SRMs outperform scalar RMs and GRMs in robustness and alignment with human preferences. The modular design further supports efficient optimization for practical scenarios, allowing SRM to provide a practical reward modeling solution for industry. 

---
# Towards Continual Expansion of Data Coverage: Automatic Text-guided Edge-case Synthesis 

**Authors**: Kyeongryeol Go  

**Link**: [PDF](https://arxiv.org/pdf/2509.26158)  

**Abstract**: The performance of deep neural networks is strongly influenced by the quality of their training data. However, mitigating dataset bias by manually curating challenging edge cases remains a major bottleneck. To address this, we propose an automated pipeline for text-guided edge-case synthesis. Our approach employs a Large Language Model, fine-tuned via preference learning, to rephrase image captions into diverse textual prompts that steer a Text-to-Image model toward generating difficult visual scenarios. Evaluated on the FishEye8K object detection benchmark, our method achieves superior robustness, surpassing both naive augmentation and manually engineered prompts. This work establishes a scalable framework that shifts data curation from manual effort to automated, targeted synthesis, offering a promising direction for developing more reliable and continuously improving AI systems. Code is available at this https URL. 

---
# More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models 

**Authors**: Xinyu Tian, Shu Zou, Zhaoyuan Yang, Mengqi He, Fabian Waschkowski, Lukas Wesemann, Peter Tu, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25848)  

**Abstract**: Reasoning has emerged as a pivotal capability in Large Language Models (LLMs). Through Reinforcement Learning (RL), typically Group Relative Policy Optimization (GRPO), these models are able to solve complex tasks such as mathematics and code generation. Building on these advances, recent research has sought to extend reasoning to Vision-Language Models (VLMs), yielding promising results across diverse visual tasks. Despite this progress, our study uncovers the dual nature of multimodal reasoning: while it substantially enhances logical inference and facilitates performance on challenging problems, it may gradually impair perceptual grounding, leading to recognition failures on otherwise basic visual questions. Through further analysis, we attribute this phenomenon to visual forgetting, wherein prolonged reasoning causes the model to increasingly disregard visual input. To address this, we propose Vision-Anchored Policy Optimization (VAPO), a simple yet effective method that explicitly steers the reasoning process toward visually grounded trajectories. Our result model, VAPO-Thinker-7B, significantly strengthens the model's reliance on visual information and achieves new state-of-the-art results on a wide range of established benchmarks. Project page: this https URL 

---
# Free Lunch Alignment of Text-to-Image Diffusion Models without Preference Image Pairs 

**Authors**: Jia Jun Cheng Xian, Muchen Li, Haotian Yang, Xin Tao, Pengfei Wan, Leonid Sigal, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25771)  

**Abstract**: Recent advances in diffusion-based text-to-image (T2I) models have led to remarkable success in generating high-quality images from textual prompts. However, ensuring accurate alignment between the text and the generated image remains a significant challenge for state-of-the-art diffusion models. To address this, existing studies employ reinforcement learning with human feedback (RLHF) to align T2I outputs with human preferences. These methods, however, either rely directly on paired image preference data or require a learned reward function, both of which depend heavily on costly, high-quality human annotations and thus face scalability limitations. In this work, we introduce Text Preference Optimization (TPO), a framework that enables "free-lunch" alignment of T2I models, achieving alignment without the need for paired image preference data. TPO works by training the model to prefer matched prompts over mismatched prompts, which are constructed by perturbing original captions using a large language model. Our framework is general and compatible with existing preference-based algorithms. We extend both DPO and KTO to our setting, resulting in TDPO and TKTO. Quantitative and qualitative evaluations across multiple benchmarks show that our methods consistently outperform their original counterparts, delivering better human preference scores and improved text-to-image alignment. Our Open-source code is available at this https URL. 

---
# Dolphin v1.0 Technical Report 

**Authors**: Taohan Weng, Chi zhang, Chaoran Yan, Siya Liu, Xiaoyang Liu, Yalun Wu, Boyang Wang, Boyan Wang, Jiren Ren, Kaiwen Yan, Jinze Yu, Kaibing Hu, Henan Liu, Haoyun zheng, Anjie Le, Hongcheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25748)  

**Abstract**: Ultrasound is crucial in modern medicine but faces challenges like operator dependence, image noise, and real-time scanning, hindering AI integration. While large multimodal models excel in other medical imaging areas, they struggle with ultrasound's complexities. To address this, we introduce Dolphin v1.0 (V1) and its reasoning-augmented version, Dolphin R1-the first large-scale multimodal ultrasound foundation models unifying diverse clinical tasks in a single vision-language this http URL tackle ultrasound variability and noise, we curated a 2-million-scale multimodal dataset, combining textbook knowledge, public data, synthetic samples, and general corpora. This ensures robust perception, generalization, and clinical this http URL Dolphin series employs a three-stage training strategy: domain-specialized pretraining, instruction-driven alignment, and reinforcement-based refinement. Dolphin v1.0 delivers reliable performance in classification, detection, regression, and report generation. Dolphin R1 enhances diagnostic inference, reasoning transparency, and interpretability through reinforcement learning with ultrasound-specific this http URL on U2-Bench across eight ultrasound tasks, Dolphin R1 achieves a U2-score of 0.5835-over twice the second-best model (0.2968) setting a new state of the art. Dolphin v1.0 also performs competitively, validating the unified framework. Comparisons show reasoning-enhanced training significantly improves diagnostic accuracy, consistency, and interpretability, highlighting its importance for high-stakes medical AI. 

---
# R-Log: Incentivizing Log Analysis Capability in LLMs via Reasoning-based Reinforcement Learning 

**Authors**: Yilun Liu, Ziang Chen, Song Xu, Minggui He, Shimin Tao, Weibin Meng, Yuming Xie, Tao Han, Chunguang Zhao, Jingzhou Du, Daimeng Wei, Shenglin Zhang, Yongqian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.25987)  

**Abstract**: The growing complexity of log data in modern software systems has prompted the use of Large Language Models (LLMs) for automated log analysis. Current approaches typically rely on direct supervised fine-tuning (SFT) on log-label pairs. However, this exacerbates the domain discrepancy between general-purpose LLMs and specialized log data, causing overfitting. Furthermore, SFT's imbalanced loss computation often allows lengthy contexts to overwhelm critical, concise details in model answers, leading to hallucinations. To address these limitations, we propose R-Log, a novel reasoning-based paradigm that mirrors the structured, step-by-step analytical process of human engineers. This approach enhances generalizability by learning the underlying rules behind conclusions. We further employ Reinforcement Learning (RL) to optimize the model within a simulated O&M environment, thereby reducing hallucinations by directly rewarding correct outcomes. R-Log is first cold-started on a curated dataset of 2k+ reasoning trajectories, guided by 13 strategies from manual O&M practices, to establish an initial reasoning capability. This ability is then refined via RL using a joint reward function. Empirical evaluations on real-world logs show that R-Log outperforms existing methods across five log analysis tasks, particularly in unseen scenarios (by 228.05%). We also designed R-Log-fast with 5x speedup while keeping 93% of the efficacy. 

---
# Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play 

**Authors**: Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi, Yueqian Lin, Yiran Chen, Hai Helen Li, Kun Wan, Wentian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25541)  

**Abstract**: Although reinforcement learning (RL) can effectively enhance the reasoning capabilities of vision-language models (VLMs), current methods remain heavily dependent on labor-intensive datasets that require extensive manual construction and verification, leading to extremely high training costs and consequently constraining the practical deployment of VLMs. To address this challenge, we propose Vision-Zero, a domain-agnostic framework enabling VLM self-improvement through competitive visual games generated from arbitrary image pairs. Specifically, Vision-Zero encompasses three main attributes: (1) Strategic Self-Play Framework: Vision-Zero trains VLMs in "Who Is the Spy"-style games, where the models engage in strategic reasoning and actions across multiple roles. Through interactive gameplay, models autonomously generate their training data without human annotation. (2) Gameplay from Arbitrary Images: Unlike existing gamified frameworks, Vision-Zero can generate games from arbitrary images, thereby enhancing the model's reasoning ability across diverse domains and showing strong generalization to different tasks. We demonstrate this versatility using three distinct types of image datasets: CLEVR-based synthetic scenes, charts, and real-world images. (3) Sustainable Performance Gain: We introduce Iterative Self-Play Policy Optimization (Iterative-SPO), a novel training algorithm that alternates between Self-Play and reinforcement learning with verifiable rewards (RLVR), mitigating the performance plateau often seen in self-play-only training and achieving sustained long-term improvements. Despite using label-free data, Vision-Zero achieves state-of-the-art performance on reasoning, chart question answering, and vision-centric understanding tasks, surpassing other annotation-based methods. Models and code has been released at this https URL. 

---
# HAMMER: Hamiltonian Curiosity Augmented Large Language Model Reinforcement 

**Authors**: Ming Yang, Xiaofan Li, Zhiyuan Ma, Dengliang Shi, Jintao Du, Yu Cheng, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25240)  

**Abstract**: Recent curriculum reinforcement learning for large language models (LLMs) typically rely on difficulty-based annotations for data filtering and ordering. However, such methods suffer from local optimization, where continual training on simple samples in the early steps can cause the policy to lose its exploration. We propose a novel schema, namely Hamiltonian curiosity augmented large language model reinforcement (HAMMER), that transfers diversity metrics, commonly used in dataset evaluation, into the dynamic reinforcement learning procedure, where training samples are ordered via a minimum-semantic Hamiltonian path making the initial training retrain more exploration. From a theoretical perspective of generalization bounds, diversity-driven ordering facilitates stable convergence. Empirical evaluations indicate that HAMMER stimulates model "curiosity" and consistently achieves a 3% to 4% average accuracy gain across diverse inference benchmark. 

---
# Reinforcement Learning-Guided Chain-of-Draft for Token-Efficient Code Generation 

**Authors**: Xunzhu Tang, Iyiola Emmanuel Olatunji, Tiezhu Sun, Jacques Klein, Tegawende F. Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2509.25243)  

**Abstract**: LLMs demonstrate surface-level fluency in code generation but struggle with structured reasoning tasks requiring correctness and semantic alignment. While Chain-of-Thought (CoT) prompting enhances reasoning through intermediate steps, it suffers from verbosity and inefficiency. Chain-of-Draft (CoD) prompting offers more concise reasoning, but the stochastic nature of LLMs produces varying solution quality, making optimal selection challenging. We propose \multicod, a reinforcement learning framework that learns to select the most promising candidate from CoD-generated solutions. Our approach uses strategy-guided prompting to encourage diverse reasoning styles and models solution selection as a contextual bandit problem. The framework optimizes interpretable features including code complexity, reasoning structure, and strategic metadata through a reward function balancing correctness, efficiency, and clarity. Experiments on MBPP, BigCodeBench, SWE-bench Verified, and Defects4J show \multicod~outperforms and in some cases, on par with standard prompting, CoT, and CoD baselines while achieving cost and token efficiency from the user's perspective through a multi-candidate design that charges only for the selected output, reducing user billing by over 50\% and improving LLM response quality, making \multicod~more sustainable and scalable for real-world deployment. Our code is available: this https URL. 

---
# APRIL: API Synthesis with Automatic Prompt Optimization and Reinforcement Learning 

**Authors**: Hua Zhong, Shan Jiang, Sarfraz Khurshid  

**Link**: [PDF](https://arxiv.org/pdf/2509.25196)  

**Abstract**: APIs are central to modern software development, yet composing new APIs from large libraries is difficult due to the exponential search space; traditional component-based synthesis relies on costly exploration and hand-crafted specifications. While large language models (LLMs) can generate implementations from natural language, hallucinations and limited access to up-to-date contextual information often yield incorrect code. In this paper, we present APRIL, an approach that combines LLM-based synthesis with Automatic Prompt Optimization (APO) and Reinforcement Learning from Verifiable Rewards (RLVR): APO iteratively refines prompts for a frozen model, while RLVR fine-tunes the policy toward functional correctness, producing an efficient synthesis pipeline. Evaluated on 81 real-world APIs from widely used scientific Python libraries and benchmarked against instruction-tuned but unfine-tuned LLMs guided by expert prompts, APRIL achieves substantial improvements. These results indicate that integrating APO and RLVR provides a robust, scalable path for component-based API synthesis in large libraries. 

---
# UML-CoT: Structured Reasoning and Planning with Unified Modeling Language for Robotic Room Cleaning 

**Authors**: Hongyu Chen, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22628)  

**Abstract**: Chain-of-Thought (CoT) prompting improves reasoning in large language models (LLMs), but its reliance on unstructured text limits interpretability and executability in embodied tasks. Prior work has explored structured CoTs using scene or logic graphs, yet these remain fundamentally limited: they model only low-order relations, lack constructs like inheritance or behavioral abstraction, and provide no standardized semantics for sequential or conditional planning. We propose UML-CoT, a structured reasoning and planning framework that leverages Unified Modeling Language (UML) to generate symbolic CoTs and executable action plans. UML class diagrams capture compositional object semantics, while activity diagrams model procedural control flow. Our three-stage training pipeline combines supervised fine-tuning with Group Relative Policy Optimization (GRPO), including reward learning from answer-only data. We evaluate UML-CoT on MRoom-30k, a new benchmark of cluttered room-cleaning scenarios. UML-CoT outperforms unstructured CoTs in interpretability, planning coherence, and execution success, highlighting UML as a more expressive and actionable structured reasoning formalism. 

---
