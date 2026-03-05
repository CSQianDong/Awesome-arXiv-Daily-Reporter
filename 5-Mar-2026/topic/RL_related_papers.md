# BeamPERL: Parameter-Efficient RL with Verifiable Rewards Specializes Compact LLMs for Structured Beam Mechanics Reasoning 

**Authors**: Tarjei Paule Hage, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2603.04124)  

**Abstract**: Can reinforcement learning with hard, verifiable rewards teach a compact language model to reason about physics, or does it primarily learn to pattern-match toward correct answers? We study this question by training a 1.5B-parameter reasoning model on beam statics, a classic engineering problem, using parameter-efficient RLVR with binary correctness rewards from symbolic solvers, without teacher-generated reasoning traces. The best BeamPERL checkpoint achieves a 66.7% improvement in Pass@1 over the base model. However, the learned competence is anisotropic: the model generalizes compositionally (more loads) but fails under topological shifts (moved supports) that require the same equilibrium equations. Intermediate checkpoints yield the strongest reasoning, while continued optimization degrades robustness while maintaining reward. These findings reveal a key limitation of outcome-level alignment: reinforcement learning with exact physics rewards induces procedural solution templates rather than internalization of governing equations. The precision of the reward signal - even when analytically exact - does not by itself guarantee transferable physical reasoning. Our results suggest that verifiable rewards may need to be paired with structured reasoning scaffolding to move beyond template matching toward robust scientific reasoning. 

---
# CodeTaste: Can LLMs Generate Human-Level Code Refactorings? 

**Authors**: Alex Thillen, Niels Mündler, Veselin Raychev, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2603.04177)  

**Abstract**: Large language model (LLM) coding agents can generate working code, but their solutions often accumulate complexity, duplication, and architectural debt. Human developers address such issues through refactoring: behavior-preserving program transformations that improve structure and maintainability. In this paper, we investigate if LLM agents (i) can execute refactorings reliably and (ii) identify the refactorings that human developers actually chose in real codebases. We present CodeTaste, a benchmark of refactoring tasks mined from large-scale multi-file changes in open-source repositories. To score solutions, we combine repository test suites with custom static checks that verify removal of undesired patterns and introduction of desired patterns using dataflow reasoning.
Our experimental results indicate a clear gap across frontier models: agents perform well when refactorings are specified in detail, but often fail to discover the human refactoring choices when only presented with a focus area for improvement. A propose-then-implement decomposition improves alignment, and selecting the best-aligned proposal before implementation can yield further gains. CodeTaste provides an evaluation target and a potential preference signal for aligning coding agents with human refactoring decisions in realistic codebases. 

---
# Confidence-Calibrated Small-Large Language Model Collaboration for Cost-Efficient Reasoning 

**Authors**: Chuang Zhang, Zizhen Zhu, Yihao Wei, Bing Tian, Junyi Liu, Henan Wang, Xavier Wang, Yaxiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03752)  

**Abstract**: Large language models (LLMs) demonstrate superior reasoning capabilities compared to small language models (SLMs), but incur substantially higher costs. We propose COllaborative REAsoner (COREA), a system that cascades an SLM with an LLM to achieve a balance between accuracy and cost in complex reasoning tasks. COREA first attempts to answer questions using the SLM, which outputs both an answer and a verbalized confidence score. Questions with confidence below a predefined threshold are deferred to the LLM for more accurate resolution. We introduce a reinforcement learning-based training algorithm that aligns the SLM's confidence through an additional confidence calibration reward. Extensive experiments demonstrate that our method jointly improves the SLM's reasoning ability and confidence calibration across diverse datasets and model backbones. Compared to using the LLM alone, COREA reduces cost by 21.5% and 16.8% on out-of-domain math and non-math datasets, respectively, with only an absolute pass@1 drop within 2%. 

---
# MIND: Unified Inquiry and Diagnosis RL with Criteria Grounded Clinical Supports for Psychiatric Consultation 

**Authors**: Guoyi Li, Shihao Xu, Jiatong Ma, Yunyun Han, Jianhua Chen, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.03677)  

**Abstract**: Large language models (LLMs) have advanced medical dialogue systems, yet psychiatric consultation poses substantially higher demands due to subjective ambiguity and comorbidity complexity: an agent must continuously extract psychopathological cues from incomplete and inconsistent patient reports in multi-turn interactions and perform rigorous differential diagnostic reasoning. However, existing methods face two fundamental challenges. First, without criteria-grounded clinical supports, they are prone to unsupported clinical assertions when symptoms are atypical or underspecified. Second, in multi-turn interactions, they struggle to mitigate inquiry drift (off-topic or low-yield questioning) and optimize questioning strategies. To address these challenges, we propose MIND, a unified inquiry--diagnosis reinforcement learning framework for psychiatric consultation. Specifically, we build a Criteria-Grounded Psychiatric Reasoning Bank (PRB) that summarizes dialogue context into clinical retrieval states, retrieves semantically similar reference consultations, and distills reusable criteria-grounded clinical supports to guide criteria-aligned inquiry and reasoning. Building on this foundation, MIND enforces explicit clinical reasoning with rubric-based process rewards to provide fine-grained supervision over intermediate decision steps, and incorporates a value-aware trajectory rectification mechanism to jointly improve information acquisition and diagnostic decision-making across turns. Extensive experiments demonstrate that MIND consistently outperforms strong baselines in diagnostic accuracy, empathetic interaction quality, interpretability, and generalization. 

---
# SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems 

**Authors**: Haochang Hao, Yifan Xu, Xinzhuo Li, Yingqiang Ge, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.03536)  

**Abstract**: Current LLM-based conversational recommender systems (CRS) primarily optimize recommendation accuracy and user satisfaction. We identify an underexplored vulnerability in which recommendation outputs may negatively impact users by violating personalized safety constraints, when individualized safety sensitivities -- such as trauma triggers, self-harm history, or phobias -- are implicitly inferred from the conversation but not respected during recommendation. We formalize this challenge as personalized CRS safety and introduce SafeRec, a new benchmark dataset designed to systematically evaluate safety risks in LLM-based CRS under user-specific constraints. To further address this problem, we propose SafeCRS, a safety-aware training framework that integrates Safe Supervised Fine-Tuning (Safe-SFT) with Safe Group reward-Decoupled Normalization Policy Optimization (Safe-GDPO) to jointly optimize recommendation quality and personalized safety alignment. Extensive experiments on SafeRec demonstrate that SafeCRS reduces safety violation rates by up to 96.5% relative to the strongest recommendation-quality baseline while maintaining competitive recommendation quality. Warning: This paper contains potentially harmful and offensive content. 

---
# PhyPrompt: RL-based Prompt Refinement for Physically Plausible Text-to-Video Generation 

**Authors**: Shang Wu, Chenwei Xu, Zhuofan Xia, Weijian Li, Lie Lu, Pranav Maneriker, Fan Du, Manling Li, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03505)  

**Abstract**: State-of-the-art text-to-video (T2V) generators frequently violate physical laws despite high visual quality. We show this stems from insufficient physical constraints in prompts rather than model limitations: manually adding physics details reliably produces physically plausible videos, but requires expertise and does not scale. We present PhyPrompt, a two-stage reinforcement learning framework that automatically refines prompts for physically realistic generation. First, we fine-tune a large language model on a physics-focused Chain-of-Thought dataset to integrate principles like object motion and force interactions while preserving user intent. Second, we apply Group Relative Policy Optimization with a dynamic reward curriculum that initially prioritizes semantic fidelity, then progressively shifts toward physical commonsense. This curriculum achieves synergistic optimization: PhyPrompt-7B reaches 40.8\% joint success on VideoPhy2 (8.6pp gain), improving physical commonsense by 11pp (55.8\% to 66.8\%) while simultaneously increasing semantic adherence by 4.4pp (43.4\% to 47.8\%). Remarkably, our curriculum exceeds single-objective training on both metrics, demonstrating compositional prompt discovery beyond conventional multi-objective trade-offs. PhyPrompt outperforms GPT-4o (+3.8\% joint) and DeepSeek-V3 (+2.2\%, 100$\times$ larger) using only 7B parameters. The approach transfers zero-shot across diverse T2V architectures (Lavie, VideoCrafter2, CogVideoX-5B) with up to 16.8\% improvement, establishing that domain-specialized reinforcement learning with compositional curricula surpasses general-purpose scaling for physics-aware generation. 

---
# MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning 

**Authors**: Jiejun Tan, Zhicheng Dou, Liancheng Zhang, Yuyang Hu, Yiruo Cheng, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.03379)  

**Abstract**: As Large Language Models (LLMs) are increasingly used for long-duration tasks, maintaining effective long-term memory has become a critical challenge. Current methods often face a trade-off between cost and accuracy. Simple storage methods often fail to retrieve relevant information, while complex indexing methods (such as memory graphs) require heavy computation and can cause information loss. Furthermore, relying on the working LLM to process all memories is computationally expensive and slow. To address these limitations, we propose MemSifter, a novel framework that offloads the memory retrieval process to a small-scale proxy model. Instead of increasing the burden on the primary working LLM, MemSifter uses a smaller model to reason about the task before retrieving the necessary information. This approach requires no heavy computation during the indexing phase and adds minimal overhead during inference. To optimize the proxy model, we introduce a memory-specific Reinforcement Learning (RL) training paradigm. We design a task-outcome-oriented reward based on the working LLM's actual performance in completing the task. The reward measures the actual contribution of retrieved memories by mutiple interactions with the working LLM, and discriminates retrieved rankings by stepped decreasing contributions. Additionally, we employ training techniques such as Curriculum Learning and Model Merging to improve performance. We evaluated MemSifter on eight LLM memory benchmarks, including Deep Research tasks. The results demonstrate that our method meets or exceeds the performance of existing state-of-the-art approaches in both retrieval accuracy and final task completion. MemSifter offers an efficient and scalable solution for long-term LLM memory. We have open-sourced the model weights, code, and training data to support further research. 

---
# IntPro: A Proxy Agent for Context-Aware Intent Understanding via Retrieval-conditioned Inference 

**Authors**: Guanming Liu, Meng Wu, Peng Zhang, Yu Zhang, Yubo Shu, Xianliang Huang, Kainan Tu, Ning Gu, Liuxin Zhang, Qianying Wang, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03325)  

**Abstract**: Large language models (LLMs) have become integral to modern Human-AI collaboration workflows, where accurately understanding user intent serves as a crucial step for generating satisfactory responses. Context-aware intent understanding, which involves inferring user intentions from situational environments, is inherently challenging because it requires reasoning over both the immediate context and the user's underlying motivations that drive their behavior. Moreover, existing approaches often treat intent understanding as a static recognition task, overlooking users' accumulated intent patterns that could provide valuable references for more accurate and generalizable understanding. To address this gap, we propose IntPro, a proxy agent that learns to adapt to individual users via retrieval-conditioned intent inference. We design intent explanations that abstract how contextual signals connect to expressed intents, and store them in an individual intent history library for retrieval. We train IntPro through supervised fine-tuning on retrieval-conditioned trajectories and multi-turn Group Relative Policy Optimization (GRPO) with tool-aware reward functions, enabling the agent to learn when to leverage historical intent patterns and when to infer directly. Experiments across three diverse scenarios (Highlight-Intent, MIntRec2.0, and Weibo Post-Sync) demonstrate that IntPro achieves strong intent understanding performance with effective context-aware reasoning capabilities across different scenarios and model types. 

---
# HumanLM: Simulating Users with State Alignment Beats Response Imitation 

**Authors**: Shirley Wu, Evelyn Choi, Arpandeep Khatua, Zhanghan Wang, Joy He-Yueya, Tharindu Cyril Weerasooriya, Wei Wei, Diyi Yang, Jure Leskovec, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.03303)  

**Abstract**: Large Language Models (LLMs) are increasingly used to simulate how specific users respond to a given context, enabling more user-centric applications that rely on user feedback. However, existing user simulators mostly imitate surface-level patterns and language styles, which fail to reflect the underlying states of real users (e.g., beliefs and emotions). To address these limitations, we propose a novel training framework, HumanLM, which builds user simulators that accurately reflect real users. Our key insight is that, in addition to generating responses, the model should generate natural-language latent states that align with ground-truth responses through reinforcement learning. These latent states correspond to a set of psychologically grounded state dimensions that drive how real users respond. HumanLM further synthesizes these aligned latent states into responses that accurately represent real users. For extensive evaluation, we develop Humanual, a comprehensive benchmark for simulating real users based on public data. Humanual consists of six large-scale datasets with 26k users and 216k responses in total, spanning diverse tasks such as generating user responses to daily life issues, political blogs, and chat sessions with LLM assistants. Across datasets, HumanLM significantly outperforms alternative approaches, achieving an average relative improvement of 16.3% in alignment scores from an LLM judge. In a real-time simulation study with 111 participants, HumanLM achieves the highest similarity to real user responses and competitive human-likeness scores. 

---
# TTSR: Test-Time Self-Reflection for Continual Reasoning Improvement 

**Authors**: Haoyang He, Zihua Rong, Liangjie Zhao, Yunjia Zhao, Lan Yang, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03297)  

**Abstract**: Test-time Training enables model adaptation using only test questions and offers a promising paradigm for improving the reasoning ability of large language models (LLMs). However, it faces two major challenges: test questions are often highly difficult, making self-generated pseudo-labels unreliable, and existing methods lack effective mechanisms to adapt to a model's specific reasoning weaknesses, leading to inefficient learning. To address these issues, we propose \textbf{TTSR}, a self-reflective test-time self-evolving training framework. TTSR employs a single pretrained language model that alternates between the roles of a \textit{Student} and a \textit{Teacher} at test time. The Student focuses on solving problems and learning from synthesized variant questions, while the Teacher analyzes the Student's failed reasoning trajectories, summarizes recurring reasoning weaknesses, and synthesizes targeted variant questions accordingly. This process guides the model to improve within a learnable regime through a continual self-evolving loop. Experimental results on multiple challenging mathematical reasoning benchmarks show that TTSR consistently improves reasoning performance and generalizes well across different model backbones and general-domain reasoning tasks. These findings suggest that teacher-mediated self-reflection provides an effective pathway for stable and continual reasoning improvement at test time. 

---
# One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models 

**Authors**: Daniel Fein, Max Lamparth, Violet Xiang, Mykel J. Kochenderfer, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2603.03291)  

**Abstract**: Reward Models (RMs) are crucial for online alignment of language models (LMs) with human preferences. However, RM-based preference-tuning is vulnerable to reward hacking, whereby LM policies learn undesirable behaviors from flawed RMs. By systematically measuring biases in five high-quality RMs, including the state-of-the-art, we find that issues persist despite prior work with respect to length, sycophancy, and overconfidence. We also discover new issues related to bias toward model-specific styles and answer-order. We categorize RM failures by complexity and propose a simple post-hoc intervention to mitigate low-complexity biases that arise from spurious correlations. Our proposed mechanistic reward shaping reduces targeted biases without degrading reward quality and while using minimal labeled data. The method is extensible to new biases, model-internal, and generalizes out-of-distribution. 

---
# $V_1$: Unifying Generation and Self-Verification for Parallel Reasoners 

**Authors**: Harman Singh, Xiuyu Li, Kusha Sareen, Monishwaran Maheswaran, Sijun Tan, Xiaoxia Wu, Junxiong Wang, Alpay Ariyak, Qingyang Wu, Samir Khaki, Rishabh Tiwari, Long Lian, Yucheng Lu, Boyi Li, Alane Suhr, Ben Athiwaratkun, Kurt Keutzer  

**Link**: [PDF](https://arxiv.org/pdf/2603.04304)  

**Abstract**: Test-time scaling for complex reasoning tasks shows that leveraging inference-time compute, by methods such as independently sampling and aggregating multiple solutions, results in significantly better task outcomes. However, a critical bottleneck is verification: sampling is only effective if correct solutions can be reliably identified among candidates. While existing approaches typically evaluate candidates independently via scalar scoring, we demonstrate that models are substantially stronger at pairwise self-verification. Leveraging this insight, we introduce $V_1$, a framework that unifies generation and verification through efficient pairwise ranking. $V_1$ comprises two components: $V_1$-Infer, an uncertainty-guided algorithm using a tournament-based ranking that dynamically allocates self-verification compute to candidate pairs whose relative correctness is most uncertain; and $V_1$-PairRL, an RL framework that jointly trains a single model as both generator and pairwise self-verifier, ensuring the verifier adapts to the generator's evolving distribution. On code generation (LiveCodeBench, CodeContests, SWE-Bench) and math reasoning (AIME, HMMT) benchmarks, $V_1$-Infer improves Pass@1 by up to $10%$ over pointwise verification and outperforms recent test-time scaling methods while being significantly more efficient. Furthermore, $V_1$-PairRL achieves $7$--$9%$ test-time scaling gains over standard RL and pointwise joint training, and improves base Pass@1 by up to 8.7% over standard RL in a code-generation setting. 

---
# Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory 

**Authors**: Zhenting Wang, Huancheng Chen, Jiayun Wang, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.04257)  

**Abstract**: Large language model (LLM) agents are fundamentally bottlenecked by finite context windows on long-horizon tasks. As trajectories grow, retaining tool outputs and intermediate reasoning in-context quickly becomes infeasible: the working context becomes prohibitively long, eventually exceeds the context budget, and makes distant evidence harder to use even when it is still present. Existing solutions typically shorten context through truncation or running summaries, but these methods are fundamentally lossy because they compress or discard past evidence itself. We introduce Memex, an indexed experience memory mechanism that instead compresses context without discarding evidence. Memex maintains a compact working context consisting of concise structured summaries and stable indices, while storing full-fidelity underlying interactions in an external experience database under those indices. The agent can then decide when to dereference an index and recover the exact past evidence needed for the current subgoal. We optimize both write and read behaviors with our reinforcement learning framework MemexRL, using reward shaping tailored to indexed memory usage under a context budget, so the agent learns what to summarize, what to archive, how to index it, and when to retrieve it. This yields a substantially less lossy form of long-horizon memory than summary-only approaches. We further provide a theoretical analysis showing the potential of the Memex loop to preserve decision quality with bounded dereferencing while keeping effective in-context computation bounded as history grows. Empirically, on challenging long-horizon tasks, Memex agent trained with MemexRL improves task success while using a significantly smaller working context. 

---
