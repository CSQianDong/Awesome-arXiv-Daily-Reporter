# LLM-in-Sandbox Elicits General Agentic Intelligence 

**Authors**: Daixuan Cheng, Shaohan Huang, Yuxian Gu, Huatong Song, Guoxin Chen, Li Dong, Wayne Xin Zhao, Ji-Rong Wen, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.16206)  

**Abstract**: We introduce LLM-in-Sandbox, enabling LLMs to explore within a code sandbox (i.e., a virtual computer), to elicit general intelligence in non-code domains. We first demonstrate that strong LLMs, without additional training, exhibit generalization capabilities to leverage the code sandbox for non-code tasks. For example, LLMs spontaneously access external resources to acquire new knowledge, leverage the file system to handle long contexts, and execute scripts to satisfy formatting requirements. We further show that these agentic capabilities can be enhanced through LLM-in-Sandbox Reinforcement Learning (LLM-in-Sandbox-RL), which uses only non-agentic data to train models for sandbox exploration. Experiments demonstrate that LLM-in-Sandbox, in both training-free and post-trained settings, achieves robust generalization spanning mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following. Finally, we analyze LLM-in-Sandbox's efficiency from computational and system perspectives, and open-source it as a Python package to facilitate real-world deployment. 

---
# Hallucination Mitigating for Medical Report Generation 

**Authors**: Ruoqing Zhao, Runze Xia, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.15745)  

**Abstract**: In the realm of medical report generation (MRG), the integration of natural language processing has emerged as a vital tool to alleviate the workload of radiologists. Despite the impressive capabilities demonstrated by large vision language models (LVLMs) in understanding natural language, their susceptibility to generating plausible yet inaccurate claims, known as ``hallucinations'', raises concerns-especially in the nuanced and critical field of medical. In this work, we introduce a framework, \textbf{K}nowledge-\textbf{E}nhanced with Fine-Grained \textbf{R}einforced Rewards \textbf{M}edical Report Generation (KERM), to tackle the issue. Our approach refines the input to the LVLM by first utilizing MedCLIP for knowledge retrieval, incorporating relevant lesion fact sentences from a curated knowledge corpus. We then introduce a novel purification module to ensure the retrieved knowledge is contextually relevant to the patient's clinical context. Subsequently, we employ fine-grained rewards to guide these models in generating highly supportive and clinically relevant descriptions, ensuring the alignment of model's outputs with desired behaviors. Experimental results on IU-Xray and MIMIC-CXR datasets validate the effectiveness of our approach in mitigating hallucinations and enhancing report quality. 

---
# Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind 

**Authors**: Tamunotonye Harry, Ivoline Ngong, Chima Nweke, Yuanyuan Feng, Joseph Near  

**Link**: [PDF](https://arxiv.org/pdf/2601.15395)  

**Abstract**: User interactions with language models vary due to static properties of the user (trait) and the specific context of the interaction (state). However, existing persona datasets (like PersonaChat, PANDORA etc.) capture only trait, and ignore the impact of state. We introduce Chameleon, a dataset of 5,001 contextual psychological profiles from 1,667 Reddit users, each measured across multiple contexts. Using the Chameleon dataset, we present three key findings. First, inspired by Latent State-Trait theory, we decompose variance and find that 74\% is within-person(state) while only 26\% is between-person (trait). Second, we find that LLMs are state-blind: they focus on trait only, and produce similar responses regardless of state. Third, we find that reward models react to user state, but inconsistently: different models favor or penalize the same users in opposite directions. We release Chameleon to support research on affective computing, personalized dialogue, and RLHF alignment. 

---
# ICPO: Illocution-Calibrated Policy Optimization for Multi-Turn Conversation 

**Authors**: Zhebo Wang, Xiaohu Mu, Zijie Zhou, Mohan Li, Wenpeng Xing, Dezhang Kong, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.15330)  

**Abstract**: Large Language Models (LLMs) in multi-turn conversations often suffer from a ``lost-in-conversation'' phenomenon, where they struggle to recover from early incorrect assumptions, particularly when users provide ambiguous initial instructions. We find that standard post-training techniques like Reinforcement Learning with Verifiable Rewards (RLVR) exacerbate this issue by rewarding confident, direct answers, thereby inducing overconfidence and discouraging the model from seeking clarification. To address this, we propose Illocution-Calibrated Policy Optimization (ICPO), a novel training framework that sensitizes the model to instruction ambiguity. ICPO augments the training corpus with underspecified prompts and conditions the reward signal on the user's illocutionary intent, rewarding the model for expressing uncertainty or asking for clarification when faced with ambiguity. Experiments demonstrate that ICPO fosters appropriate humility, yielding a substantial average improvement of 75\% in multi-turn conversation, while preserving robust performance on single-turn benchmarks. Our work presents a practical path toward more robust and collaborative conversational AI that can better navigate the nuances of human interaction. 

---
# PhysProver: Advancing Automatic Theorem Proving for Physics 

**Authors**: Hanning Zhang, Ruida Wang, Rui Pan, Wenyuan Wang, Bingxu Meng, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15737)  

**Abstract**: The combination of verifiable languages and LLMs has significantly influenced both the mathematical and computer science communities because it provides a rigorous foundation for theorem proving. Recent advancements in the field provide foundation models and sophisticated agentic systems pushing the boundaries of formal mathematical reasoning to approach the natural language capability of LLMs. However, little attention has been given to the formal physics reasoning, which also heavily relies on similar problem-solving and theorem-proving frameworks. To solve this problem, this paper presents, to the best of our knowledge, the first approach to enhance formal theorem proving in the physics domain. We compose a dedicated dataset PhysLeanData for the task. It is composed of theorems sampled from PhysLean and data generated by a conjecture-based formal data generation pipeline. In the training pipeline, we leverage DeepSeek-Prover-V2-7B, a strong open-source mathematical theorem prover, and apply Reinforcement Learning with Verifiable Rewards (RLVR) to train our model PhysProver. Comprehensive experiments demonstrate that, using only $\sim$5K training samples, PhysProver achieves an overall 2.4\% improvement in multiple sub-domains. Furthermore, after formal physics training, we observe 1.3\% gains on the MiniF2F-Test benchmark, which indicates non-trivial generalization beyond physics domains and enhancement for formal math capability as well. The results highlight the effectiveness and efficiency of our approach, which provides a paradigm for extending formal provers outside mathematical domains. To foster further research, we will release both our dataset and model to the community. 

---
# Knowledge Graphs are Implicit Reward Models: Path-Derived Signals Enable Compositional Reasoning 

**Authors**: Yuval Kansal, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2601.15160)  

**Abstract**: Large language models have achieved near-expert performance in structured reasoning domains like mathematics and programming, yet their ability to perform compositional multi-hop reasoning in specialized scientific fields remains limited. We propose a bottom-up learning paradigm in which models are grounded in axiomatic domain facts and compose them to solve complex, unseen tasks. To this end, we present a post-training pipeline, based on a combination of supervised fine-tuning and reinforcement learning (RL), in which knowledge graphs act as implicit reward models. By deriving novel reward signals from knowledge graph paths, we provide verifiable, scalable, and grounded supervision that encourages models to compose intermediate axioms rather than optimize only final answers during RL. We validate this approach in the medical domain, training a 14B model on short-hop reasoning paths (1-3 hops) and evaluating its zero-shot generalization to complex multi-hop queries (4-5 hops). Our experiments show that path-derived rewards act as a "compositional bridge", enabling our model to significantly outperform much larger models and frontier systems like GPT-5.2 and Gemini 3 Pro, on the most difficult reasoning tasks. Furthermore, we demonstrate the robustness of our approach to adversarial perturbations against option-shuffling stress tests. This work suggests that grounding the reasoning process in structured knowledge is a scalable and efficient path toward intelligent reasoning. 

---
# Structured Hints for Sample-Efficient Lean Theorem Proving 

**Authors**: Zachary Burton  

**Link**: [PDF](https://arxiv.org/pdf/2601.16172)  

**Abstract**: State-of-the-art neural theorem provers like DeepSeek-Prover-V1.5 combine large language models with reinforcement learning, achieving impressive results through sophisticated training. We ask: do these highly-trained models still benefit from simple structural guidance at inference time? We evaluate a lightweight intervention -- a fixed prompt schedule over 15 common tactic skeletons -- on the miniF2F benchmark. This simple approach yields 21.7% pass@16 compared to 15.2% for standard sampling from the same model, a 43% relative improvement using the same number of samples (k=16) and same maximum generation length (1024 tokens). Our results suggest that even capable RL-trained provers underutilize structural priors available in the tactic language, and that simple inference-time guidance remains a cheap, complementary boost. 

---
# Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification 

**Authors**: Yuxuan Wan, Tianqing Fang, Zaitang Li, Yintong Huo, Wenxuan Wang, Haitao Mi, Dong Yu, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2601.15808)  

**Abstract**: Recent advances in Deep Research Agents (DRAs) are transforming automated knowledge discovery and problem-solving. While the majority of existing efforts focus on enhancing policy capabilities via post-training, we propose an alternative paradigm: self-evolving the agent's ability by iteratively verifying the policy model's outputs, guided by meticulously crafted rubrics. This approach gives rise to the inference-time scaling of verification, wherein an agent self-improves by evaluating its generated answers to produce iterative feedback and refinements. We derive the rubrics based on an automatically constructed DRA Failure Taxonomy, which systematically classifies agent failures into five major categories and thirteen sub-categories. We present DeepVerifier, a rubrics-based outcome reward verifier that leverages the asymmetry of verification and outperforms vanilla agent-as-judge and LLM judge baselines by 12%-48% in meta-evaluation F1 score. To enable practical self-evolution, DeepVerifier integrates as a plug-and-play module during test-time inference. The verifier produces detailed rubric-based feedback, which is fed back to the agent for iterative bootstrapping, refining responses without additional training. This test-time scaling delivers 8%-11% accuracy gains on challenging subsets of GAIA and XBench-DeepResearch when powered by capable closed-source LLMs. Finally, to support open-source advancement, we release DeepVerifier-4K, a curated supervised fine-tuning dataset of 4,646 high-quality agent steps focused on DRA verification. These examples emphasize reflection and self-critique, enabling open models to develop robust verification capabilities. 

---
# Q-Probe: Scaling Image Quality Assessment to High Resolution via Context-Aware Agentic Probing 

**Authors**: Xiang Li, XueHeng Li, Yu Wang, XuanHua He, ZhangChi Hu, WeiWei Yu, ChengJun Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.15356)  

**Abstract**: Reinforcement Learning (RL) has empowered Multimodal Large Language Models (MLLMs) to achieve superior human preference alignment in Image Quality Assessment (IQA). However, existing RL-based IQA models typically rely on coarse-grained global views, failing to capture subtle local degradations in high-resolution scenarios. While emerging "Thinking with Images" paradigms enable multi-scale visual perception via zoom-in mechanisms, their direct adaptation to IQA induces spurious "cropping-implies-degradation" biases and misinterprets natural depth-of-field as artifacts. To address these challenges, we propose Q-Probe, the first agentic IQA framework designed to scale IQA to high resolution via context-aware probing. First, we construct Vista-Bench, a pioneering benchmark tailored for fine-grained local degradation analysis in high-resolution IQA settings. Furthermore, we propose a three-stage training paradigm that progressively aligns the model with human preferences, while simultaneously eliminating causal bias through a novel context-aware cropping strategy. Extensive experiments demonstrate that Q-Probe achieves state-of-the-art performance in high-resolution settings while maintaining superior efficacy across resolution scales. 

---
