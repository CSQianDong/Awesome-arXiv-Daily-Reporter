# The Realignment Problem: When Right becomes Wrong in LLMs 

**Authors**: Aakash Sen Sharma, Debdeep Sanyal, Vivek Srivastava, Shirish Karande, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2511.02623)  

**Abstract**: The alignment of Large Language Models (LLMs) with human values is central to their safe deployment, yet current practice produces static, brittle, and costly-to-maintain models that fail to keep pace with evolving norms and policies. This misalignment, which we term the Alignment-Reality Gap, poses a growing challenge for reliable long-term use. Existing remedies are inadequate: large-scale re-annotation is economically prohibitive, and standard unlearning methods act as blunt instruments that erode utility rather than enable precise policy updates. We introduce TRACE (Triage and Re-align by Alignment Conflict Evaluation), a framework for principled unlearning that reconceives re-alignment as a programmatic policy application problem. TRACE programmatically triages existing preference data against a new policy, identifies high-impact conflicts via a alignment impact score, and applies a hybrid optimization that cleanly inverts, discards, or preserves preferences while safeguarding model performance. Empirical results show that TRACE achieves robust re-alignment across diverse model families (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B). On both synthetic benchmarks and the PKU-SafeRLHF dataset under complex policy shift, TRACE enforces new principles without degrading general capabilities. Our work establishes a scalable, dynamic, and cost-effective paradigm for maintaining LLM alignment, providing a foundation for sustainable and responsible AI deployment. 

---
# MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning 

**Authors**: Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, Xianpei Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.02805)  

**Abstract**: Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead. The code and models will be publicly available at this https URL 

---
# Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning 

**Authors**: Bowen Jin, TJ Collins, Donghan Yu, Mert Cemri, Shenao Zhang, Mengyu Li, Jay Tang, Tian Qin, Zhiyang Xu, Jiarui Lu, Guoli Yin, Jiawei Han, Zirui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02755)  

**Abstract**: Large language models (LLMs) exhibit complementary strengths across domains and come with varying inference costs, motivating the design of multi-agent LLM systems where specialized models collaborate efficiently. Existing approaches predominantly rely on decentralized frameworks, which invoke multiple LLMs for every input and thus lead to substantial and uncontrolled inference costs. In this work, we introduce a centralized multi-LLM framework, where a controller LLM selectively coordinates a pool of expert models in a cost-efficient and cost-controllable manner. We formulate this coordination problem as reinforcement learning with dual objectives: maximizing task performance while minimizing the overall inference cost. In addition, we expect the multi-agent system to have adapted behavior with different budget conditions during inference. To this end, we propose CoRL, a reinforcement learning framework that optimizes the performance cost trade-off in a controllable multi-budget setting. Experiments on four diverse benchmarks demonstrate that CoRL enables a single system to surpass the best expert LLM under high-budget settings, while maintaining strong performance in more economical low-budget modes, highlighting the effectiveness of centralized coordination for scalable and cost-efficient multi-agent LLM systems. 

---
# PragExTra: A Multilingual Corpus of Pragmatic Explicitation in Translation 

**Authors**: Doreen Osmelak, Koel Dutta Chowdhury, Uliana Sentsova, Cristina España-Bonet, Josef van Genabith  

**Link**: [PDF](https://arxiv.org/pdf/2511.02721)  

**Abstract**: Translators often enrich texts with background details that make implicit cultural meanings explicit for new audiences. This phenomenon, known as pragmatic explicitation, has been widely discussed in translation theory but rarely modeled computationally. We introduce PragExTra, the first multilingual corpus and detection framework for pragmatic explicitation. The corpus covers eight language pairs from TED-Multi and Europarl and includes additions such as entity descriptions, measurement conversions, and translator remarks. We identify candidate explicitation cases through null alignments and refined using active learning with human annotation. Our results show that entity and system-level explicitations are most frequent, and that active learning improves classifier accuracy by 7-8 percentage points, achieving up to 0.88 accuracy and 0.82 F1 across languages. PragExTra establishes pragmatic explicitation as a measurable, cross-linguistic phenomenon and takes a step towards building culturally aware machine translation. Keywords: translation, multilingualism, explicitation 

---
# Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation 

**Authors**: Zhiwei Zhang, Xiaomin Li, Yudi Lin, Hui Liu, Ramraj Chandradevan, Linlin Wu, Minhua Lin, Fali Wang, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02303)  

**Abstract**: Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks. 

---
# SAIL-RL: Guiding MLLMs in When and How to Think via Dual-Reward RL Tuning 

**Authors**: Fangxun Shu, Yongjie Ye, Yue Liao, Zijian Kang, Weijie Yin, Jiacong Wang, Xiao Liang, Shuicheng Yan, Chao Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02280)  

**Abstract**: We introduce SAIL-RL, a reinforcement learning (RL) post-training framework that enhances the reasoning capabilities of multimodal large language models (MLLMs) by teaching them when and how to think. Existing approaches are limited by outcome-only supervision, which rewards correct answers without ensuring sound reasoning, and by uniform thinking strategies, which often lead to overthinking on simple tasks and underthinking on complex ones. SAIL-RL addresses these challenges with a dual reward system: the Thinking Reward, which evaluates reasoning quality through factual grounding, logical coherence, and answer consistency, and the Judging Reward, which adaptively determines whether deep reasoning or direct answering is appropriate. Experiments on the state-of-the-art SAIL-VL2 show that SAIL-RL improves reasoning and multimodal understanding benchmarks at both 4B and 8B scales, achieving competitive performance against commercial closed-source models such as GPT-4o, and substantially reduces hallucinations, establishing it as a principled framework for building more reliable and adaptive MLLMs. The code will be available at this https URL. 

---
# Training Proactive and Personalized LLM Agents 

**Authors**: Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02208)  

**Abstract**: While existing work focuses primarily on task success, we argue that effective real-world agents require optimizing three dimensions: productivity (task completion), proactivity (asking essential questions), and personalization (adapting to diverse user preferences). We introduce UserVille, an interactive environment with LLM-based user simulators enabling diverse, configurable user preferences. Leveraging UserVille, we introduce PPP, a multi-objective reinforcement learning approach that jointly optimizes all three dimensions: Productivity, Proactivity, and Personalization. Experiments on software engineering and deep research tasks show that agents trained with PPP achieve substantial improvements over strong baselines such as GPT-5 (+21.6 on average), demonstrating the ability to ask strategic clarifying questions, adapt to unseen user preferences, and improve task success through better interaction. This work demonstrates that explicitly optimizing for user-centered interaction is critical for building practical and effective AI agents. 

---
# Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning 

**Authors**: Renos Zabounidis, Aditya Golatkar, Michael Kleinman, Alessandro Achille, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2511.02130)  

**Abstract**: We propose Re-FORC, an adaptive reward prediction method that, given a context, enables prediction of the expected future rewards as a function of the number of future thinking tokens. Re-FORC trains a lightweight adapter on reasoning models, demonstrating improved prediction with longer reasoning and larger models. Re-FORC enables: 1) early stopping of unpromising reasoning chains, reducing compute by 26% while maintaining accuracy, 2) optimized model and thinking length selection that achieves 4% higher accuracy at equal compute and 55% less compute at equal accuracy compared to the largest model, 3) adaptive test-time scaling, which increases accuracy by 11% in high compute regime, and 7% in low compute regime. Re-FORC allows dynamic reasoning with length control via cost-per-token thresholds while estimating computation time upfront. 

---
# Shorter but not Worse: Frugal Reasoning via Easy Samples as Length Regularizers in Math RLVR 

**Authors**: Abdelaziz Bounhar, Hadi Abdine, Evan Dufraisse, Ahmad Chamma, Amr Mohamed, Dani Bouch, Michalis Vazirgiannis, Guokan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01937)  

**Abstract**: Large language models (LLMs) trained for step-by-step reasoning often become excessively verbose, raising inference cost. Standard Reinforcement Learning with Verifiable Rewards (RLVR) pipelines filter out ``easy'' problems for training efficiency, leaving the model to train primarily on harder problems that require longer reasoning chains. This skews the output length distribution upward, resulting in a \textbf{model that conflates ``thinking longer'' with ``thinking better''}. In this work, we show that retaining and modestly up-weighting moderately easy problems acts as an implicit length regularizer. Exposing the model to solvable short-chain tasks constrains its output distribution and prevents runaway verbosity. The result is \textbf{\emph{emergent brevity for free}}: the model learns to solve harder problems without inflating the output length, \textbf{ despite the absence of any explicit length penalization}. RLVR experiments using this approach on \textit{Qwen3-4B-Thinking-2507} (with a 16k token limit) achieve baseline pass@1 AIME25 accuracy while generating solutions that are, on average, nearly twice as short. The code is available at \href{this https URL}{GitHub}, with datasets and models on \href{this https URL}{Hugging Face}. 

---
# Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch 

**Authors**: Yirong Zeng, Xiao Ding, Yutai Hou, Yuxian Wang, Li Du, Juyi Dai, Qiuyang Ding, Duyu Tang, Dandan Tu, Weiwen Liu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01934)  

**Abstract**: Training tool-augmented LLMs has emerged as a promising approach to enhancing language models' capabilities for complex tasks. The current supervised fine-tuning paradigm relies on constructing extensive domain-specific datasets to train models. However, this approach often struggles to generalize effectively to unfamiliar or intricate tool-use scenarios. Recently, reinforcement learning (RL) paradigm can endow LLMs with superior reasoning and generalization abilities. In this work, we address a key question: Can the pure RL be used to effectively elicit a model's intrinsic reasoning capabilities and enhance the tool-agnostic generalization? We propose a dynamic generalization-guided reward design for rule-based RL, which progressively shifts rewards from exploratory to exploitative tool-use patterns. Based on this design, we introduce the Tool-Zero series models. These models are trained to enable LLMs to autonomously utilize general tools by directly scaling up RL from Zero models (i.e., base models without post-training). Experimental results demonstrate that our models achieve over 7% performance improvement compared to both SFT and RL-with-SFT models under the same experimental settings. These gains are consistently replicated across cross-dataset and intra-dataset evaluations, validating the effectiveness and robustness of our methods. 

---
